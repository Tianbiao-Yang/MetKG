# -*- coding: utf-8 -*-
"""
使用已训练好的 MultiTask Graphormer + PLE 模型，对新的样本做预测
- 加载 ../saved_models/mtl_graphormer_ple_20251214_122448.pt
- 读取 ../data/model_evaluation_dataset/MetKG_evaluation_set_sample_2.txt
- 按训练时完全一致的方式做预处理（含 log1p、z-score、样本内归一化等）
- 输出：
    1) 逐样本预测结果文件：MetKG_evaluation_set_sample_2_with_pred.txt
    2) 整体性能指标文件：   MetKG_evaluation_set_sample_2_metrics.txt
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.utils import to_dense_batch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score
)

# ----------------- 路径配置 -----------------
MODEL_PATH = "../saved_models/mtl_graphormer_ple_20251214_122448.pt"
TRAIN_DATA_DIR = "../data/model_training_dataset"   # 用于加载节点嵌入（nodes_embedding.pkl）
EVAL_DATA_PATH = "../data/model_evaluation_dataset/MetKG_Q300_Evaluation_Set.txt"

# 预测结果（逐样本）和 性能指标（整体）
OUT_PRED_PATH = "../result/model_evaluation_result/MetKG_Q300_Evaluation_Set_MScore_261019.txt"
OUT_METRICS_PATH = "../result/model_evaluation_result/MetKG_Q300_Evaluation_Set_Metrics_261019.txt"


# 与训练脚本保持一致的列名
target_age_col = "Age"
target_dm_col = "DM"
extra_cols = ["Sex", "BMI"]   # 训练时就是这两个额外特征

CONC_MASK_DIM = 3  # [跨样本 z-score, 样本内总量归一化, 缺失 mask]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# =========================================================
# 1. 定义模型结构（需与训练脚本保持一致）
# =========================================================

class ExpertMLP(nn.Module):
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

        # task-specific experts
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                ExpertMLP(in_dim, expert_dim, dropout=dropout)
                for _ in range(num_task_experts)
            ])
            for _ in range(num_tasks)
        ])
        # shared experts
        self.shared_experts = nn.ModuleList([
            ExpertMLP(in_dim, expert_dim, dropout=dropout)
            for _ in range(num_shared_experts)
        ])

        # task gates
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, num_task_experts + num_shared_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_tasks)
        ])

        # shared gate
        total_task_experts = num_tasks * num_task_experts
        total_experts_for_shared = total_task_experts + num_shared_experts
        self.shared_gate = nn.Sequential(
            nn.Linear(in_dim * (num_tasks + 1), total_experts_for_shared),
            nn.Softmax(dim=-1),
        )

    def forward(self, task_inputs, shared_input):
        """
        task_inputs: list[num_tasks]，每个 [B, in_dim]
        shared_input: [B, in_dim]
        """
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
            gate = self.task_gates[t](task_inputs[t])                    # [B, E_t+E_s]
            gate = gate.view(B, -1, 1)                                   # [B, E_total, 1]

            all_experts_b = all_experts.permute(1, 0, 2)                 # [B, E_total, D]
            fused = torch.sum(gate * all_experts_b, dim=1)               # [B, D]
            new_task_outputs.append(fused)

        # shared tower
        shared_gate_in = torch.cat(task_inputs + [shared_input], dim=-1)  # [B, (T+1)*in_dim]
        gate_s = self.shared_gate(shared_gate_in)                         # [B, T*E_t + E_s]
        gate_s = gate_s.view(B, -1, 1)                                    # [B, E_all, 1]

        all_task_experts = torch.cat(task_outs, dim=0)                    # [T*E_t, B, D]
        all_experts_for_shared = torch.cat(
            [all_task_experts, shared_outs], dim=0
        )                                                                 # [T*E_t + E_s, B, D]
        all_experts_for_shared_b = all_experts_for_shared.permute(1, 0, 2)  # [B, E_all, D]
        new_shared_output = torch.sum(gate_s * all_experts_for_shared_b, dim=1)  # [B, D]

        return new_task_outputs, new_shared_output


class PLEMultiTaskTower(nn.Module):
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
        self.num_layers = num_layers

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
        z_init: [B, input_dim]
        """
        task_reps = [z_init for _ in range(self.num_tasks)]
        shared_rep = z_init
        for layer in self.layers:
            task_reps, shared_rep = layer(task_reps, shared_rep)
        return task_reps


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
        self.dropout = dropout
        self.use_node_embedding = use_node_embedding
        self.conc_dim = conc_dim
        self.embed_dim = embed_dim
        self.embed_proj_dim = embed_proj_dim
        self.embed_alpha = embed_alpha

        # 节点结构嵌入
        if self.use_node_embedding:
            if node_embed_mat is None or embed_dim <= 0:
                raise ValueError("use_node_embedding=True 时，必须提供 node_embed_mat 和 embed_dim")
            node_embed_mat = torch.from_numpy(node_embed_mat.astype(np.float32))
            self.register_buffer("node_embed_mat", node_embed_mat)
            self.node_embed_proj = nn.Linear(embed_dim, embed_proj_dim)
            fused_dim = conc_dim + embed_proj_dim
        else:
            fused_dim = conc_dim

        # 节点输入投影
        self.input_proj = nn.Linear(fused_dim, hidden_dim)
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_nodes, hidden_dim))

        # Transformer Encoder
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
            adj = torch.from_numpy(np.asarray(adj_with_self, dtype=np.float32))
        attn_mask = (adj == 0)
        self.register_buffer("attn_mask", attn_mask)

        # PLE 多任务头
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
        task_reps = self.ple(z)
        z_age, z_dm = task_reps

        age_pred = self.age_out(z_age).squeeze(-1)
        dm_logits = self.dm_out(z_dm)
        return age_pred, dm_logits


# =========================================================
# 2. Dataset 定义（与训练版基本一致）
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
        return data


# =========================================================
# 3. 评估函数
# =========================================================

def evaluate_model(model, loader, num_dm_classes):
    model.eval()
    all_age_true = []
    all_age_pred = []
    all_dm_true = []
    all_dm_prob = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            age_pred, dm_logits = model(batch)

            all_age_pred.append(age_pred.cpu().numpy())
            all_age_true.append(batch.y_age.cpu().numpy())

            prob = torch.softmax(dm_logits, dim=1).cpu().numpy()
            all_dm_prob.append(prob)
            all_dm_true.append(batch.y_dm.cpu().numpy())

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
# 4. 主流程：加载模型 + 预处理 + 预测
# =========================================================

def main():
    # ---------- 4.1 加载模型 checkpoint ----------
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

    # ---------- 4.2 构造 edge_index ----------
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

    # ---------- 4.3 载入新的评价集 ----------
    df_eval = pd.read_csv(EVAL_DATA_PATH, sep="\t")
    print(f"评价集样本数: {len(df_eval)}")

    # 确保包含所有 common_hmdb 列
    existing_cols = set(df_eval.columns)
    missing_hmdb = [h for h in common_hmdb if h not in existing_cols]
    if len(missing_hmdb) > 0:
        add_df = pd.DataFrame(np.nan, index=df_eval.index, columns=missing_hmdb)
        df_eval = pd.concat([df_eval, add_df], axis=1)

    # 确保额外特征列存在
    missing_extra = [c for c in extra_cols if c not in df_eval.columns]
    if len(missing_extra) > 0:
        add_df = pd.DataFrame(np.nan, index=df_eval.index, columns=missing_extra)
        df_eval = pd.concat([df_eval, add_df], axis=1)

    # ---------- 4.3.1 在任何变换前备份原始代谢物浓度 ----------
    raw_conc_df = df_eval[common_hmdb].copy()  # 新增：用于后面恢复

    # ---------- 4.4 按训练方式进行 log1p ----------
    if apply_log1p:
        vals = df_eval[common_hmdb].to_numpy(dtype=np.float32)
        nan_mask = np.isnan(vals)
        non_nan_mask = ~nan_mask
        non_neg_mask = (vals >= 0) & non_nan_mask
        vals[non_neg_mask] = np.log1p(vals[non_neg_mask])
        df_eval[common_hmdb] = vals
        print("已对评价集代谢物浓度做 log1p 预处理")
    else:
        print("评价集代谢物浓度不做 log1p 预处理")

    # ---------- 4.5 DM 标签编码（用于评价指标） ----------
    if label_encoder is not None:
        df_eval["DM_raw"] = df_eval[target_dm_col]
        df_eval[target_dm_col] = label_encoder.transform(df_eval[target_dm_col].astype(str))
        print("评价集 DM 使用训练时的 LabelEncoder 做编码")
    else:
        df_eval[target_dm_col] = df_eval[target_dm_col].astype(int)

    # ---------- 4.6 构造节点嵌入矩阵 ----------
    node_embed_mat = None
    embed_dim = 0
    if use_node_embedding:
        import pickle
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

    # ---------- 4.7 构建 Dataset & DataLoader ----------
    eval_indices = np.arange(len(df_eval))
    eval_dataset = SampleGraphDataset(
        df_eval,
        eval_indices,
        common_hmdb,
        node_means,
        node_stds,
        extra_cols,
        extra_means,
        extra_stds,
        edge_index,
    )
    eval_loader = GeoDataLoader(
        eval_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    # ---------- 4.8 构建模型并加载参数 ----------
    if gnn_type.lower() != "graphormer":
        raise ValueError("当前 checkpoint 是 Graphormer + PLE，预测脚本只实现了该分支")

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
    print("模型参数已加载")

    # ---------- 4.9 评估并获得预测 ----------
    metrics = evaluate_model(model, eval_loader, num_dm_classes)
    print("\n===== 评价集指标 =====")
    print(f"Age MAE : {metrics['age_mae']:.4f}")
    print(f"Age RMSE: {metrics['age_rmse']:.4f}")
    print(f"Age R2  : {metrics['age_r2']:.4f}")
    print(f"DM Acc  : {metrics['dm_acc']:.4f}")
    print(f"DM AUC  : {metrics['dm_auc'] if not np.isnan(metrics['dm_auc']) else 'nan'}")

    # 综合 score：Age MAE + (1 - AUC)
    auc_for_score = metrics['dm_auc'] if not np.isnan(metrics['dm_auc']) else 0.0
    combined_score = metrics['age_mae'] + (1.0 - auc_for_score)

    # ---------- 4.10 写回预测结果（逐样本，使用 concat 避免 fragment warning） ----------
    y_age_pred = metrics["y_age_pred"]
    y_dm_prob = metrics["y_dm_prob"]
    y_dm_pred = metrics["y_dm_pred"]

    pred_cols = {
        "Age_pred": y_age_pred,
        "DM_pred": y_dm_pred,
    }
    for c in range(num_dm_classes):
        pred_cols[f"DM_prob_class{c}"] = y_dm_prob[:, c]

    if label_encoder is not None:
        pred_cols["DM_pred_label"] = label_encoder.inverse_transform(y_dm_pred)

    pred_df = pd.DataFrame(pred_cols, index=df_eval.index)

    # 关键一步：在保存前恢复原始代谢物浓度
    df_eval[common_hmdb] = raw_conc_df

    df_eval = pd.concat([df_eval, pred_df], axis=1)

    df_eval.to_csv(OUT_PRED_PATH, sep="\t", index=False)
    print(f"\n预测结果已保存到: {OUT_PRED_PATH}")

    # ---------- 4.11 单独保存性能指标 ----------
    metrics_to_save = {
        "age_mae": metrics["age_mae"],
        "age_rmse": metrics["age_rmse"],
        "age_r2": metrics["age_r2"],
        "dm_acc": metrics["dm_acc"],
        "dm_auc": metrics["dm_auc"],
        "combined_score_age_mae_plus_1_minus_auc": combined_score,
        "num_samples": int(len(df_eval)),
    }
    pd.DataFrame([metrics_to_save]).to_csv(OUT_METRICS_PATH, sep="\t", index=False)
    print(f"评价指标已保存到: {OUT_METRICS_PATH}")


if __name__ == "__main__":
    main()
