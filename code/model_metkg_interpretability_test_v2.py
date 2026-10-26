# -*- coding: utf-8 -*-
"""
explain_mtl_graphormer.py

使用已训练好的 MultiTask Graphormer + PLE 模型，
对内部 Test 集（train 文件的 test 划分）和外部 Q300 评价集做可解释分析：

1) 正常前向预测并评估性能；
2) 提取 TransformerEncoder 各层 self-attention（任务无关）；
3) 计算任务无关的节点–节点关系权重矩阵 & 节点重要性；
4) 基于梯度归因，分别针对 Age / DM / Shared 三个目标：
   - 计算节点重要性（per-sample, per-task）；
   - 结合 attention 得到 task-specific 边重要性；
   - 得到全局关键节点和关键边。

注意：
- 不修改模型结构和权重，只在前向阶段显式展开 encoder，并额外做 backward。
- 此版本中 Shared 任务来自 PLE 框架中最终 shared tower 表征（其维度平均），
  而不是简单 Age + DM 的和。
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ============ PyTorch 相关 ============
import torch
import torch.nn as nn

# ============ PyTorch Geometric ============
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.utils import to_dense_batch

# ============ Sklearn（仅用于评估） ============
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score
)

# ----------------- 路径配置（与原脚本保持一致） -----------------
MODEL_PATH = "../saved_models/mtl_graphormer_ple_20251214_122448.pt"

TRAIN_DATA_DIR = "../data/model_training_dataset"
TRAIN_DATA_PATH = os.path.join(TRAIN_DATA_DIR, "MetKG_training_set_sample_1.txt")
SPLIT_DIR = os.path.join(TRAIN_DATA_DIR, "saved_splits")

EVAL_DATA_DIR = "../data/model_evaluation_dataset"
Q300_DATA_PATH = os.path.join(EVAL_DATA_DIR, "MetKG_Q300_Evaluation_Set.txt")

# 解释性结果输出目录
OUT_EXPLAIN_DIR = "../result/model_evaluation_explain/v2"
os.makedirs(OUT_EXPLAIN_DIR, exist_ok=True)

# 列名设置（与训练保持一致）
target_age_col = "Age"
target_dm_col = "DM"
extra_cols = ["Sex", "BMI"]

CONC_MASK_DIM = 3  # [z-score, 样本内归一化, 缺失 mask]

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# =========================================================
# 1. 模型结构（与训练脚本保持一致）
# =========================================================


class ExpertMLP(nn.Module):
    """单个 expert"""

    def __init__(self, in_dim, out_dim, hidden_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(out_dim * hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class PLELayer(nn.Module):
    """单层标准 PLE"""

    def __init__(
        self,
        in_dim,
        expert_dim,
        num_tasks=2,
        num_task_experts=2,
        num_shared_experts=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_task_experts = num_task_experts
        self.num_shared_experts = num_shared_experts
        self.expert_dim = expert_dim

        # Task-specific experts
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                ExpertMLP(in_dim, expert_dim, dropout=dropout)
                for _ in range(num_task_experts)
            ])
            for _ in range(num_tasks)
        ])

        # Shared experts
        self.shared_experts = nn.ModuleList([
            ExpertMLP(in_dim, expert_dim, dropout=dropout)
            for _ in range(num_shared_experts)
        ])

        # Task gates
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, num_task_experts + num_shared_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_tasks)
        ])

        # Shared gate
        total_task_experts = num_tasks * num_task_experts
        total_experts_for_shared = total_task_experts + num_shared_experts
        self.shared_gate = nn.Sequential(
            nn.Linear(in_dim * (num_tasks + 1), total_experts_for_shared),
            nn.Softmax(dim=-1),
        )

    def forward(self, task_inputs, shared_input):
        num_tasks = self.num_tasks
        B = shared_input.size(0)
        D = self.expert_dim

        # shared experts: [E_s, B, D]
        shared_outs = torch.stack(
            [exp(shared_input) for exp in self.shared_experts],
            dim=0
        )

        # task-specific experts: list[t][E_t, B, D]
        task_outs = []
        for t in range(num_tasks):
            outs_t = torch.stack(
                [exp(task_inputs[t]) for exp in self.task_experts[t]],
                dim=0
            )
            task_outs.append(outs_t)

        # task towers
        new_task_outputs = []
        for t in range(num_tasks):
            all_experts = torch.cat([task_outs[t], shared_outs], dim=0)  # [E_t+E_s, B, D]
            gate = self.task_gates[t](task_inputs[t])  # [B, E_t+E_s]
            gate = gate.view(B, -1, 1)  # [B, E_total, 1]
            all_experts_b = all_experts.permute(1, 0, 2)  # [B, E_total, D]
            fused = torch.sum(gate * all_experts_b, dim=1)  # [B, D]
            new_task_outputs.append(fused)

        # shared tower
        shared_gate_in = torch.cat(task_inputs + [shared_input], dim=-1)  # [B, (T+1)*in_dim]
        gate_s = self.shared_gate(shared_gate_in)  # [B, T*E_t+E_s]
        gate_s = gate_s.view(B, -1, 1)  # [B, E_all, 1]
        all_task_experts = torch.cat(task_outs, dim=0)  # [T*E_t, B, D]
        all_experts_for_shared = torch.cat([all_task_experts, shared_outs], dim=0)
        all_experts_for_shared_b = all_experts_for_shared.permute(1, 0, 2)  # [B, E_all, D]
        new_shared_output = torch.sum(gate_s * all_experts_for_shared_b, dim=1)  # [B, D]

        return new_task_outputs, new_shared_output


class PLEMultiTaskTower(nn.Module):
    """多层 PLE 堆叠"""

    def __init__(
        self,
        input_dim,
        expert_dim,
        num_tasks=2,
        num_layers=2,
        num_task_experts=2,
        num_shared_experts=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_tasks = num_tasks

        layers = []
        in_dims = [input_dim] + [expert_dim] * (num_layers - 1)
        for l in range(num_layers):
            layer = PLELayer(
                in_dim=in_dims[l],
                expert_dim=expert_dim,
                num_tasks=num_tasks,
                num_task_experts=num_task_experts,
                num_shared_experts=num_shared_experts,
                dropout=dropout,
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, z_init):
        """
        返回：
        - task_reps: list[Tensor]，每个 [B, expert_dim]，对应各任务塔最终输出
        - shared_rep: Tensor [B, expert_dim]，对应最终 shared tower 输出
        """
        task_reps = [z_init for _ in range(self.num_tasks)]
        shared_rep = z_init
        for layer in self.layers:
            task_reps, shared_rep = layer(task_reps, shared_rep)
        return task_reps, shared_rep


class MultiTaskGraphormer(nn.Module):
    def __init__(
        self,
        conc_dim,
        extra_dim,
        num_nodes,
        hidden_dim=128,
        num_layers=2,
        nhead=8,
        dropout=0.1,
        num_dm_classes=2,
        adj_with_self=None,
        use_node_embedding=False,
        node_embed_mat=None,
        embed_dim=0,
        embed_proj_dim=32,
        embed_alpha=0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_node_embedding = use_node_embedding
        self.conc_dim = conc_dim
        self.embed_dim = embed_dim
        self.embed_proj_dim = embed_proj_dim
        self.embed_alpha = embed_alpha

        # 节点结构嵌入
        if self.use_node_embedding:
            if node_embed_mat is None or embed_dim <= 0:
                raise ValueError("use_node_embedding=True 时，必须提供有效的 node_embed_mat 和 embed_dim")
            node_embed_mat = torch.from_numpy(node_embed_mat.astype(np.float32))
            self.register_buffer("node_embed_mat", node_embed_mat)
            self.node_embed_proj = nn.Linear(embed_dim, embed_proj_dim)
            fused_dim = conc_dim + embed_proj_dim
        else:
            fused_dim = conc_dim

        self.input_proj = nn.Linear(fused_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_nodes, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 注意力 mask
        if adj_with_self is None:
            raise ValueError("必须传入 adj_with_self")
        if isinstance(adj_with_self, torch.Tensor):
            adj = adj_with_self.float()
        else:
            adj = torch.from_numpy(adj_with_self.astype(np.float32))
        # True 表示不能注意
        attn_mask = (adj == 0)
        self.register_buffer("attn_mask", attn_mask)

        ple_input_dim = hidden_dim + extra_dim
        ple_expert_dim = hidden_dim

        self.ple = PLEMultiTaskTower(
            input_dim=ple_input_dim,
            expert_dim=ple_expert_dim,
            num_tasks=2,
            num_layers=2,
            num_task_experts=2,
            num_shared_experts=2,
            dropout=dropout,
        )

        self.age_out = nn.Linear(ple_expert_dim, 1)
        self.dm_out = nn.Linear(ple_expert_dim, num_dm_classes)

    def forward(self, data):
        x, batch = data.x, data.batch
        x_dense, mask = to_dense_batch(x, batch)  # [B, N, conc_dim]
        B, N, _ = x_dense.shape
        assert N == self.num_nodes, f"batch 中节点数 {N} 与 num_nodes {self.num_nodes} 不匹配"

        # 节点特征 + 节点结构嵌入
        if self.use_node_embedding:
            node_embed = self.node_embed_mat.unsqueeze(0).expand(B, -1, -1)
            node_embed_low = self.node_embed_proj(node_embed)
            node_embed_low = self.embed_alpha * node_embed_low
            fused = torch.cat([x_dense, node_embed_low], dim=-1)
        else:
            fused = x_dense

        h = self.input_proj(fused) + self.pos_embedding
        h = self.encoder(h, mask=self.attn_mask)

        graph_emb = h.mean(dim=1)  # [B, hidden_dim]

        extra = data.extra
        if extra.dim() == 1:
            extra = extra.view(data.num_graphs, -1)

        z = torch.cat([graph_emb, extra], dim=-1)

        # PLE 返回任务特定表征 + shared 表征
        task_reps, shared_rep = self.ple(z)
        z_age, z_dm = task_reps

        age_pred = self.age_out(z_age).squeeze(-1)
        dm_logits = self.dm_out(z_dm)
        return age_pred, dm_logits


# =========================================================
# 2. Dataset & 评估函数
# =========================================================


class SampleGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        idxs,
        common_hmdb,
        node_means,
        node_stds,
        extra_cols,
        extra_means,
        extra_stds,
        edge_index,
    ):
        self.df = df
        self.idxs = np.array(idxs, dtype=np.int64)
        self.common_hmdb = common_hmdb
        self.node_means = node_means
        self.node_stds = node_stds
        self.extra_cols = extra_cols
        self.extra_means = extra_means
        self.extra_stds = extra_stds
        self.edge_index = edge_index

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        idx = self.idxs[i]
        row = self.df.loc[idx]

        # 代谢物节点特征
        raw_vals = row[self.common_hmdb].to_numpy(dtype=np.float32)
        mask_vals = (~np.isnan(raw_vals)).astype(np.float32)

        # 通道1：z-score
        scaled_vals = np.zeros_like(raw_vals, dtype=np.float32)
        for j in range(len(self.common_hmdb)):
            v = raw_vals[j]
            if not np.isnan(v):
                scaled_vals[j] = (v - self.node_means[j]) / self.node_stds[j]
            else:
                scaled_vals[j] = 0.0

        # 通道2：样本内归一化
        raw_nonan = raw_vals.copy()
        raw_nonan[np.isnan(raw_nonan)] = 0.0
        sample_sum = raw_nonan.sum()
        if sample_sum > 0:
            within_vals = raw_nonan / sample_sum
        else:
            within_vals = np.zeros_like(raw_nonan, dtype=np.float32)

        x = np.stack([scaled_vals, within_vals, mask_vals], axis=1).astype(np.float32)

        # 额外特征
        extra_raw = row[self.extra_cols].to_numpy(dtype=np.float32)
        extra_mask = (~np.isnan(extra_raw)).astype(np.float32)
        extra_scaled = np.zeros_like(extra_raw, dtype=np.float32)
        for j in range(len(self.extra_cols)):
            v = extra_raw[j]
            if not np.isnan(v):
                extra_scaled[j] = (v - self.extra_means[j]) / self.extra_stds[j]
            else:
                extra_scaled[j] = 0.0

        extra_list = []
        for j in range(len(self.extra_cols)):
            extra_list.append(extra_scaled[j])
            extra_list.append(extra_mask[j])
        extra = np.array(extra_list, dtype=np.float32)

        y_age = float(row[target_age_col])
        y_dm = int(row[target_dm_col])

        data = Data(
            x=torch.from_numpy(x),
            edge_index=self.edge_index
        )
        data.extra = torch.from_numpy(extra)
        data.y_age = torch.tensor(y_age, dtype=torch.float32)
        data.y_dm = torch.tensor(y_dm, dtype=torch.long)
        data.sample_idx = int(idx)  # 记录原始 DataFrame 行索引，便于输出

        return data


def evaluate_model(model, loader, num_dm_classes):
    model.eval()
    all_age_true, all_age_pred = [], []
    all_dm_true, all_dm_prob = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            age_pred, dm_logits = model(batch)
            all_age_pred.append(age_pred.cpu().numpy())
            all_age_true.append(batch.y_age.cpu().numpy())
            prob = torch.softmax(dm_logits, dim=1).cpu().numpy()
            all_dm_prob.append(prob)
            all_dm_true.append(batch.y_dm.cpu().numpy())

    if not all_age_true:
        raise RuntimeError("评估 loader 为空。")

    y_age_true = np.concatenate(all_age_true, axis=0)
    y_age_pred = np.concatenate(all_age_pred, axis=0)
    y_dm_true = np.concatenate(all_dm_true, axis=0)
    y_dm_prob = np.concatenate(all_dm_prob, axis=0)
    y_dm_pred = np.argmax(y_dm_prob, axis=1)

    age_mae = mean_absolute_error(y_age_true, y_age_pred)
    age_rmse = np.sqrt(mean_squared_error(y_age_true, y_age_pred))
    age_r2 = r2_score(y_age_true, y_age_pred)
    dm_acc = accuracy_score(y_dm_true, y_dm_pred)

    dm_auc = np.nan
    try:
        if num_dm_classes == 2:
            dm_auc = roc_auc_score(y_dm_true, y_dm_prob[:, 1])
        else:
            dm_auc = roc_auc_score(y_dm_true, y_dm_prob, multi_class="ovr")
    except Exception:
        pass

    return {
        "age_mae": age_mae,
        "age_rmse": age_rmse,
        "age_r2": age_r2,
        "dm_acc": dm_acc,
        "dm_auc": dm_auc,
        "y_age_true": y_age_true,
        "y_age_pred": y_age_pred,
        "y_dm_true": y_dm_true,
        "y_dm_prob": y_dm_prob,
        "y_dm_pred": y_dm_pred,
    }


# =========================================================
# 3. 显式展开 encoder，输出 attention + fused 表征 + shared_rep
# =========================================================


def forward_with_attention_and_fused(model, data, requires_grad=False):
    """
    显式展开 encoder 各层，返回：
    - age_pred, dm_logits
    - attn_weights_all_layers: list[num_layers]，每个 [B, num_heads, N, N]
    - h: encoder 最后一层输出 [B, N, hidden_dim]
    - fused: encoder 之前的节点输入特征 [B, N, fused_dim]（用于梯度归因）
    - shared_rep: PLE 最终 shared tower 表征 [B, expert_dim]
    """
    model.eval()
    x, batch = data.x, data.batch
    x_dense, mask = to_dense_batch(x, batch)  # [B, N, conc_dim]
    B, N, _ = x_dense.shape
    assert N == model.num_nodes, f"batch 中节点数 {N} 与 num_nodes {model.num_nodes} 不匹配"

    # 节点特征 + 节点结构嵌入
    if model.use_node_embedding:
        node_embed = model.node_embed_mat.unsqueeze(0).expand(B, -1, -1)  # [B, N, embed_dim]
        node_embed_low = model.node_embed_proj(node_embed)  # [B, N, embed_proj_dim]
        node_embed_low = model.embed_alpha * node_embed_low
        fused = torch.cat([x_dense, node_embed_low], dim=-1)  # [B, N, conc_dim+embed_proj_dim]
    else:
        fused = x_dense  # [B, N, conc_dim]

    if requires_grad:
        fused = fused.detach()
        fused.requires_grad_(True)

    h = model.input_proj(fused) + model.pos_embedding  # [B, N, hidden_dim]
    attn_mask = model.attn_mask  # [N, N]

    attn_weights_all_layers = []
    for layer in model.encoder.layers:
        src = h
        attn_output, attn_weights = layer.self_attn(
            src, src, src,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        # 残差 + Norm
        src2 = layer.dropout1(attn_output)
        src = layer.norm1(src + src2)
        # FFN
        src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
        src2 = layer.dropout2(src2)
        src = layer.norm2(src + src2)
        h = src
        attn_weights_all_layers.append(attn_weights)

    graph_emb = h.mean(dim=1)  # [B, hidden_dim]

    extra = data.extra
    if extra.dim() == 1:
        extra = extra.view(data.num_graphs, -1)
    z = torch.cat([graph_emb, extra], dim=-1)

    # PLE 返回任务特定表征 + shared 表征
    task_reps, shared_rep = model.ple(z)
    z_age, z_dm = task_reps

    age_pred = model.age_out(z_age).squeeze(-1)
    dm_logits = model.dm_out(z_dm)

    return age_pred, dm_logits, attn_weights_all_layers, h, fused, shared_rep


# =========================================================
# 4. 通用：对一个数据集做解释
# =========================================================


def explain_for_dataset(
    dataset_name,
    df,
    indices,
    common_hmdb,
    node_means,
    node_stds,
    extra_cols,
    extra_means,
    extra_stds,
    edge_index,
    model,
    num_dm_classes,
    label_encoder=None,
    apply_log1p=False,
):
    print(f"\n===== 开始解释数据集: {dataset_name} =====")
    hmdb_ids = list(common_hmdb)
    num_nodes = len(hmdb_ids)
    N = num_nodes

    # 构建 Dataset & Loader
    dataset = SampleGraphDataset(
        df=df,
        idxs=indices,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
    )

    eval_batch_size = 200 if dataset_name.lower().startswith("internal") else 128
    eval_loader = GeoDataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0
    )

    explain_loader = GeoDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # A. 性能评估
    print(f"\n[{dataset_name}] 预测性能（用于对照）")
    metrics = evaluate_model(model, eval_loader, num_dm_classes)
    print(f"Age MAE : {metrics['age_mae']:.4f}")
    print(f"Age RMSE: {metrics['age_rmse']:.4f}")
    print(f"Age R2 : {metrics['age_r2']:.4f}")
    print(f"DM Acc : {metrics['dm_acc']:.4f}")
    print(f"DM AUC : {metrics['dm_auc'] if not np.isnan(metrics['dm_auc']) else 'nan'}")

    metrics_path = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_Pred_Metrics.tsv")
    pd.DataFrame([{
        "age_mae": metrics["age_mae"],
        "age_rmse": metrics["age_rmse"],
        "age_r2": metrics["age_r2"],
        "dm_acc": metrics["dm_acc"],
        "dm_auc": metrics["dm_auc"],
        "num_samples": int(len(indices)),
    }]).to_csv(metrics_path, sep="\t", index=False)
    print(f"[{dataset_name}] 预测指标已保存到: {metrics_path}")

    # B. attention + 梯度解释
    print(f"\n[{dataset_name}] attention + 梯度解释：计算节点–节点关系权重和节点重要性")

    global_edge_sum_attn = torch.zeros((N, N), dtype=torch.float64)
    global_sample_count = 0

    per_sample_node_rows_attn = []
    per_sample_edge_rows_attn = []

    tasks = ["Age", "DM", "Shared"]
    global_edge_sum_task = {
        t: torch.zeros((N, N), dtype=torch.float64) for t in tasks
    }
    per_sample_node_rows_task = {t: [] for t in tasks}
    per_sample_edge_rows_task = {t: [] for t in tasks}

    TOP_K = 50

    model.eval()
    sample_counter = 0
    num_samples = len(indices)
    dm_positive_class = 1 if num_dm_classes >= 2 else 0

    for batch in explain_loader:
        batch = batch.to(device)

        with torch.enable_grad():
            age_pred, dm_logits, attn_layers, h, fused, shared_rep = forward_with_attention_and_fused(
                model, batch, requires_grad=True
            )

        L = len(attn_layers)
        B_cur, H, N1, N2 = attn_layers[0].shape
        assert B_cur == 1
        assert N1 == N and N2 == N, "attention 矩阵大小与节点数不一致"

        attn_tensor = torch.stack(attn_layers, dim=0).to(torch.float64)  # [L,B,H,N,N]
        attn_mean_head = attn_tensor.mean(dim=2)  # [L,B,N,N]
        edge_weight_batch = attn_mean_head.mean(dim=0)  # [B,N,N]
        W_attn = edge_weight_batch[0]  # [N,N]

        # 全局（任务无关）累积
        global_edge_sum_attn += W_attn.detach().cpu()
        global_sample_count += 1

        sample_idx = int(batch.sample_idx.cpu().numpy()[0])

        # 任务无关：节点重要性
        node_importance_in_attn = W_attn.mean(dim=0)   # [N]
        node_importance_out_attn = W_attn.mean(dim=1)  # [N]
        for i in range(N):
            per_sample_node_rows_attn.append({
                "sample_index": sample_idx,
                "HMDB": hmdb_ids[i],
                "importance_in": float(node_importance_in_attn[i].item()),
                "importance_out": float(node_importance_out_attn[i].item()),
            })

        # 任务无关：Top-K 边（按 attention 排序）
        W_flat = W_attn.view(-1)
        if TOP_K < W_flat.numel():
            topk_vals, topk_idx = torch.topk(W_flat, TOP_K)
        else:
            topk_vals = W_flat
            topk_idx = torch.arange(W_flat.numel(), device=W_attn.device)

        topk_vals = topk_vals.detach().cpu().numpy()
        topk_idx = topk_idx.detach().cpu().numpy() if isinstance(topk_idx, torch.Tensor) else topk_idx

        for val, idx_flat in zip(topk_vals, topk_idx):
            i = idx_flat // N
            j = idx_flat % N
            per_sample_edge_rows_attn.append({
                "sample_index": sample_idx,
                "source_idx": int(i),
                "target_idx": int(j),
                "source_hmdb": hmdb_ids[int(i)],
                "target_hmdb": hmdb_ids[int(j)],
                "attention_weight": float(val),
            })

        # 三个目标的梯度节点重要性
        node_imp_grad = {}

        # 1) Age 任务
        model.zero_grad(set_to_none=True)
        if fused.grad is not None:
            fused.grad.zero_()
        target_age = age_pred[0]  # 标量
        target_age.backward(retain_graph=True)
        grad_age = fused.grad[0]  # [N, fused_dim]
        node_imp_age = grad_age.abs().mean(dim=-1)  # [N]
        node_imp_grad["Age"] = node_imp_age.detach()

        # 2) DM 任务（正类 logit）
        model.zero_grad(set_to_none=True)
        if fused.grad is not None:
            fused.grad.zero_()
        logit_pos = dm_logits[0, dm_positive_class]
        logit_pos.backward(retain_graph=True)
        grad_dm = fused.grad[0]
        node_imp_dm = grad_dm.abs().mean(dim=-1)
        node_imp_grad["DM"] = node_imp_dm.detach()

        # 3) Shared 任务（来自 PLE shared tower：shared_rep 的均值）
        model.zero_grad(set_to_none=True)
        if fused.grad is not None:
            fused.grad.zero_()
        # shared_rep: [B, D_ple]，这里 B=1，取所有维度的平均作为 Shared 标量
        shared_scalar = shared_rep[0].mean()
        shared_scalar.backward(retain_graph=False)
        grad_shared = fused.grad[0]  # [N, fused_dim]
        node_imp_shared = grad_shared.abs().mean(dim=-1)  # [N]
        node_imp_grad["Shared"] = node_imp_shared.detach()

        # 根据梯度节点重要性 + attention 计算任务相关边重要性
        for task_name in tasks:
            node_imp = node_imp_grad[task_name]  # [N]

            # 记录 per-sample 节点重要性
            for i in range(N):
                per_sample_node_rows_task[task_name].append({
                    "sample_index": sample_idx,
                    "HMDB": hmdb_ids[i],
                    "importance_grad": float(node_imp[i].item()),
                    "importance_in_attn": float(node_importance_in_attn[i].item()),
                    "importance_out_attn": float(node_importance_out_attn[i].item()),
                })

            node_imp_col = node_imp.view(N, 1)  # [N,1]
            node_imp_row = node_imp.view(1, N)  # [1,N]
            node_pair_imp = 0.5 * (node_imp_col + node_imp_row)  # [N,N]
            edge_imp = W_attn * node_pair_imp  # [N,N]

            # 全局累积（任务相关边）
            global_edge_sum_task[task_name] += edge_imp.detach().cpu().to(torch.float64)

            # Top-K 边（按 edge_imp 排序）
            edge_flat = edge_imp.view(-1)
            if TOP_K < edge_flat.numel():
                topk_vals_t, topk_idx_t = torch.topk(edge_flat, TOP_K)
            else:
                topk_vals_t = edge_flat
                topk_idx_t = torch.arange(edge_flat.numel(), device=edge_imp.device)

            topk_vals_t = topk_vals_t.detach().cpu().numpy()
            topk_idx_t = topk_idx_t.detach().cpu().numpy() if isinstance(topk_idx_t, torch.Tensor) else topk_idx_t

            for val, idx_flat in zip(topk_vals_t, topk_idx_t):
                i = idx_flat // N
                j = idx_flat % N
                per_sample_edge_rows_task[task_name].append({
                    "sample_index": sample_idx,
                    "source_idx": int(i),
                    "target_idx": int(j),
                    "source_hmdb": hmdb_ids[int(i)],
                    "target_hmdb": hmdb_ids[int(j)],
                    "attention_weight": float(W_attn[i, j].item()),
                    "edge_importance_task": float(val),
                })

        sample_counter += 1
        if sample_counter % 50 == 0 or sample_counter == num_samples:
            print(f"[{dataset_name}] 已处理样本数: {sample_counter}/{num_samples}", end="\r")

    print(f"\n[{dataset_name}] 全部样本 attention + 梯度处理完成。")

    # C. 汇总与输出

    # C.1 任务无关全局边
    global_edge_mean_attn = global_edge_sum_attn / float(global_sample_count)  # [N,N]
    rows_global_edge = []
    global_edge_mean_np = global_edge_mean_attn.numpy()
    for i in range(N):
        for j in range(N):
            rows_global_edge.append({
                "source_idx": i,
                "target_idx": j,
                "source_hmdb": hmdb_ids[i],
                "target_hmdb": hmdb_ids[j],
                "mean_attention_weight": global_edge_mean_np[i, j],
            })
    df_global_edge = pd.DataFrame(rows_global_edge)
    out_global_edge = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_Global_Edge_Attention.tsv")
    df_global_edge.to_csv(out_global_edge, sep="\t", index=False)
    print(f"[{dataset_name}] 全局节点–节点关系权重已保存到: {out_global_edge}")

    # C.2 任务无关全局节点重要性
    global_node_importance_in_attn = global_edge_mean_attn.mean(axis=0)
    global_node_importance_out_attn = global_edge_mean_attn.mean(axis=1)
    df_global_node_attn = pd.DataFrame({
        "HMDB": hmdb_ids,
        "global_importance_in": global_node_importance_in_attn.numpy(),
        "global_importance_out": global_node_importance_out_attn.numpy(),
    })
    out_global_node = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_Global_Node_Importance.tsv")
    df_global_node_attn.to_csv(out_global_node, sep="\t", index=False)
    print(f"[{dataset_name}] 全局节点重要性（任务无关）已保存到: {out_global_node}")

    # C.3 任务无关每样本节点
    df_per_sample_node_attn = pd.DataFrame(per_sample_node_rows_attn)
    out_per_sample_node_attn = os.path.join(
        OUT_EXPLAIN_DIR, f"{dataset_name}_PerSample_Node_Importance.tsv"
    )
    df_per_sample_node_attn.to_csv(out_per_sample_node_attn, sep="\t", index=False)
    print(f"[{dataset_name}] 每个样本的节点重要性（任务无关）已保存到: {out_per_sample_node_attn}")

    # C.4 任务无关每样本 Top-K 边
    df_per_sample_edge_attn = pd.DataFrame(per_sample_edge_rows_attn)
    out_per_sample_edge_attn = os.path.join(
        OUT_EXPLAIN_DIR, f"{dataset_name}_PerSample_Edge_Attention_Top{TOP_K}.tsv"
    )
    df_per_sample_edge_attn.to_csv(out_per_sample_edge_attn, sep="\t", index=False)
    print(f"[{dataset_name}] 每个样本 Top-{TOP_K} 边（任务无关 attention）已保存到: {out_per_sample_edge_attn}")

    # C.5 三个任务的全局 & per-sample
    for task_name in tasks:
        global_edge_mean_task = global_edge_sum_task[task_name] / float(global_sample_count)
        rows_global_edge_task = []
        ge_np = global_edge_mean_task.numpy()
        for i in range(N):
            for j in range(N):
                rows_global_edge_task.append({
                    "source_idx": i,
                    "target_idx": j,
                    "source_hmdb": hmdb_ids[i],
                    "target_hmdb": hmdb_ids[j],
                    "mean_edge_importance_task": ge_np[i, j],
                })
        df_global_edge_task = pd.DataFrame(rows_global_edge_task)
        out_global_edge_task = os.path.join(
            OUT_EXPLAIN_DIR, f"{dataset_name}_{task_name}_Global_Edge_Importance.tsv"
        )
        df_global_edge_task.to_csv(out_global_edge_task, sep="\t", index=False)
        print(f"[{dataset_name}] {task_name} 全局边重要性已保存到: {out_global_edge_task}")

        global_node_imp_in_task = global_edge_mean_task.mean(axis=0)
        global_node_imp_out_task = global_edge_mean_task.mean(axis=1)
        df_global_node_task = pd.DataFrame({
            "HMDB": hmdb_ids,
            "global_importance_in": global_node_imp_in_task.numpy(),
            "global_importance_out": global_node_imp_out_task.numpy(),
        })
        out_global_node_task = os.path.join(
            OUT_EXPLAIN_DIR, f"{dataset_name}_{task_name}_Global_Node_Importance.tsv"
        )
        df_global_node_task.to_csv(out_global_node_task, sep="\t", index=False)
        print(f"[{dataset_name}] {task_name} 全局节点重要性已保存到: {out_global_node_task}")

        df_per_sample_node_task = pd.DataFrame(per_sample_node_rows_task[task_name])
        out_per_sample_node_task = os.path.join(
            OUT_EXPLAIN_DIR, f"{dataset_name}_{task_name}_PerSample_Node_Importance.tsv"
        )
        df_per_sample_node_task.to_csv(out_per_sample_node_task, sep="\t", index=False)
        print(f"[{dataset_name}] {task_name} 每个样本节点重要性已保存到: {out_per_sample_node_task}")

        df_per_sample_edge_task = pd.DataFrame(per_sample_edge_rows_task[task_name])
        out_per_sample_edge_task = os.path.join(
            OUT_EXPLAIN_DIR, f"{dataset_name}_{task_name}_PerSample_Edge_Top{TOP_K}.tsv"
        )
        df_per_sample_edge_task.to_csv(out_per_sample_edge_task, sep="\t", index=False)
        print(f"[{dataset_name}] {task_name} 每个样本 Top-{TOP_K} 边已保存到: {out_per_sample_edge_task}")


# =========================================================
# 5. 主流程：加载模型与数据，分别解释内部 Test 与 Q300
# =========================================================


def main():
    # 5.1 加载 checkpoint
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    print(f"已加载模型: {MODEL_PATH}")

    gnn_type = ckpt.get("gnn_type", "graphormer")
    hidden_dim = ckpt["hidden_dim"]
    num_layers = ckpt["num_layers"]
    node_in_dim = ckpt["node_in_dim"]
    extra_dim = ckpt["extra_dim"]
    num_dm_classes = ckpt["num_dm_classes"]
    node_means = ckpt["node_means"]
    node_stds = ckpt["node_stds"]
    extra_means = ckpt["extra_means"]
    extra_stds = ckpt["extra_stds"]
    label_encoder = ckpt["label_encoder"]
    common_hmdb = ckpt["common_hmdb"]
    adj_with_self = ckpt["adj_with_self"]
    use_node_embedding = ckpt.get("use_node_embedding", False)
    embed_proj_dim = ckpt.get("embed_proj_dim", 32)
    apply_log1p = ckpt.get("apply_log1p", False)

    num_nodes = len(common_hmdb)
    print(f"建模节点数: {num_nodes}, DM 类别数: {num_dm_classes}")
    print(f"use_node_embedding={use_node_embedding}, apply_log1p={apply_log1p}")

    # 5.2 从邻接矩阵构造 edge_index
    if isinstance(adj_with_self, torch.Tensor):
        adj_np = adj_with_self.cpu().numpy().astype(np.int8)
    else:
        adj_np = np.asarray(adj_with_self, dtype=np.int8)

    adj_no_self = adj_np.copy()
    np.fill_diagonal(adj_no_self, 0)
    src, dst = np.nonzero(adj_no_self)
    edge_index = torch.tensor(
        np.stack([src, dst], axis=0),
        dtype=torch.long
    )

    # 5.3 节点嵌入
    node_embed_mat = None
    embed_dim = 0
    if use_node_embedding:
        nodes_embedding_path = os.path.join(TRAIN_DATA_DIR, "nodes_embedding.pkl")
        with open(nodes_embedding_path, "rb") as f:
            nodes_embedding_dict = pickle.load(f)
        example_key = next(iter(nodes_embedding_dict.keys()))
        embed_dim = len(nodes_embedding_dict[example_key])
        print(f"检测到节点嵌入维度: {embed_dim}")

        node_embed_mat = np.zeros((num_nodes, embed_dim), dtype=np.float32)
        missing_embed_nodes = []
        for i, hmdb in enumerate(common_hmdb):
            if hmdb in nodes_embedding_dict:
                node_embed_mat[i] = np.asarray(nodes_embedding_dict[hmdb], dtype=np.float32)
            else:
                missing_embed_nodes.append(hmdb)
        if len(missing_embed_nodes) > 0:
            print(f"警告: {len(missing_embed_nodes)} 个节点在 nodes_embedding.pkl 中缺失，使用全 0 向量。")

        embed_means = node_embed_mat.mean(axis=0, keepdims=True)
        embed_stds = node_embed_mat.std(axis=0, keepdims=True)
        embed_stds[embed_stds == 0] = 1.0
        node_embed_mat = (node_embed_mat - embed_means) / embed_stds

    # 5.4 构建模型并加载参数
    if gnn_type.lower() != "graphormer":
        raise ValueError("当前 checkpoint 是 Graphormer + PLE，解释脚本只实现该分支")

    model = MultiTaskGraphormer(
        conc_dim=node_in_dim,
        extra_dim=extra_dim,
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        nhead=8,
        dropout=0.1,
        num_dm_classes=num_dm_classes,
        adj_with_self=adj_with_self,
        use_node_embedding=use_node_embedding,
        node_embed_mat=node_embed_mat,
        embed_dim=embed_dim,
        embed_proj_dim=embed_proj_dim,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print("模型参数已加载（best_epoch = {})".format(ckpt.get("best_epoch", "unknown")))

    # A. 内部 Test 集
    print("\n===== 加载内部训练文件并准备 Test 划分 =====")
    df_train = pd.read_csv(TRAIN_DATA_PATH, sep="\t")
    print(f"训练文件样本数: {len(df_train)}")

    # 补齐缺失的 HMDB 列
    existing_cols = set(df_train.columns)
    missing_hmdb = [h for h in common_hmdb if h not in existing_cols]
    if missing_hmdb:
        add_df = pd.DataFrame(np.nan, index=df_train.index, columns=missing_hmdb)
        df_train = pd.concat([df_train, add_df], axis=1)

    # 补齐缺失的 extra 特征列
    missing_extra = [c for c in extra_cols if c not in df_train.columns]
    if missing_extra:
        add_df = pd.DataFrame(np.nan, index=df_train.index, columns=missing_extra)
        df_train = pd.concat([df_train, add_df], axis=1)

    # 是否做 log1p，与训练保持一致
    if apply_log1p:
        vals = df_train[common_hmdb].to_numpy(dtype=np.float32)
        non_nan_mask = ~np.isnan(vals)
        non_neg_mask = (vals >= 0) & non_nan_mask
        vals[non_neg_mask] = np.log1p(vals[non_neg_mask])
        df_train[common_hmdb] = vals
        print("内部集已做 log1p")
    else:
        print("内部集不做 log1p")

    # DM 编码方式与训练保持一致
    if label_encoder is not None:
        df_train[target_dm_col] = label_encoder.transform(
            df_train[target_dm_col].astype(str)
        )
    else:
        df_train[target_dm_col] = df_train[target_dm_col].astype(int)

    # 载入训练时保存的 Test 索引
    test_idx_path = os.path.join(SPLIT_DIR, "test_idx.npy")
    test_idx = np.load(test_idx_path)
    print(f"内部 Test 样本数: {len(test_idx)}")

    explain_for_dataset(
        dataset_name="InternalTest",
        df=df_train,
        indices=test_idx,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
        model=model,
        num_dm_classes=num_dm_classes,
        label_encoder=label_encoder,
        apply_log1p=apply_log1p,
    )

    # B. 外部 Q300
    print("\n===== 加载 Q300 评价集 =====")
    df_q300 = pd.read_csv(Q300_DATA_PATH, sep="\t")
    print(f"Q300 样本数: {len(df_q300)}")

    # 补齐缺失的 HMDB 列
    existing_cols = set(df_q300.columns)
    missing_hmdb = [h for h in common_hmdb if h not in existing_cols]
    if missing_hmdb:
        add_df = pd.DataFrame(np.nan, index=df_q300.index, columns=missing_hmdb)
        df_q300 = pd.concat([df_q300, add_df], axis=1)

    # 补齐缺失的 extra 特征列
    missing_extra = [c for c in extra_cols if c not in df_q300.columns]
    if missing_extra:
        add_df = pd.DataFrame(np.nan, index=df_q300.index, columns=missing_extra)
        df_q300 = pd.concat([df_q300, add_df], axis=1)

    # 是否做 log1p，与训练保持一致
    if apply_log1p:
        vals = df_q300[common_hmdb].to_numpy(dtype=np.float32)
        non_nan_mask = ~np.isnan(vals)
        non_neg_mask = (vals >= 0) & non_nan_mask
        vals[non_neg_mask] = np.log1p(vals[non_neg_mask])
        df_q300[common_hmdb] = vals
        print("Q300 已做 log1p")
    else:
        print("Q300 不做 log1p")

    # DM 编码方式与训练保持一致
    if label_encoder is not None:
        df_q300[target_dm_col] = label_encoder.transform(
            df_q300[target_dm_col].astype(str)
        )
        print("Q300 DM 使用训练时 LabelEncoder 编码")
    else:
        df_q300[target_dm_col] = df_q300[target_dm_col].astype(int)

    q300_indices = np.arange(len(df_q300))

    explain_for_dataset(
        dataset_name="Q300",
        df=df_q300,
        indices=q300_indices,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
        model=model,
        num_dm_classes=num_dm_classes,
        label_encoder=label_encoder,
        apply_log1p=apply_log1p,
    )


if __name__ == "__main__":
    main()
