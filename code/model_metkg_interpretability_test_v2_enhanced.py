# -*- coding: utf-8 -*-
"""
model_metkg_interpretability_test_v2_enhanced.py

在原 model_metkg_interpretability_test_v2.py 基础上新增：
A) Graph perturbation / node permutation / masking benchmark（无需重训练）
B) Adjusted metabolite associations（年龄、性别、BMI校正 + 分层 + 可选敏感性分析）

注意：
- 不改变模型结构和训练权重，只在推理阶段替换 attention mask、打乱节点输入或掩蔽输入节点。
- Shared 任务来自 PLE 最终 shared tower 表征均值（与原脚本一致）。
"""

import os
import pickle
import copy
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

# 统计分析（协变量校正）
import statsmodels.api as sm

# 可选：用于度保持重连
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

# ----------------- 路径配置 -----------------
MODEL_PATH = "../saved_models/mtl_graphormer_ple_20251214_122448.pt"
TRAIN_DATA_DIR = "../data/model_training_dataset"
TRAIN_DATA_PATH = os.path.join(TRAIN_DATA_DIR, "MetKG_training_set_sample_1.txt")
SPLIT_DIR = os.path.join(TRAIN_DATA_DIR, "saved_splits")
EVAL_DATA_DIR = "../data/model_evaluation_dataset"
Q300_DATA_PATH = os.path.join(EVAL_DATA_DIR, "MetKG_Q300_Evaluation_Set.txt")

OUT_EXPLAIN_DIR = "../result/model_evaluation_explain/v2_enhanced"
os.makedirs(OUT_EXPLAIN_DIR, exist_ok=True)

target_age_col = "Age"
target_dm_col = "DM"
extra_cols = ["Sex", "BMI"]
CONC_MASK_DIM = 3  # [z-score, sample-wise norm, missing mask]

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# -------- 新增：你要关注的优先代谢物（列名或HMDB） --------
# 请根据你的数据实际列名/HMDB改
PRIORITIZED_METABOLITES = ["HMDB0000684", "HMDB0000118", "HMDB0000128","HMDB0000641","HMDB0000243","HMDB0002183","HMDB0000159","HMDB0000094","HMDB0000673","HMDB0000883","HMDB0000177","HMDB0000158","HMDB0000123","HMDB0000687","HMDB0000122"]

# 可选：名称映射文件（若你有 HMDB->Name 对照，可提供）
# 例如两列：HMDB, Name
METAB_ANNOT_PATH = None  # "../data/metabolite_annotation.tsv"


# =========================================================
# 1) 模型结构（与训练保持一致）
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

        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                ExpertMLP(in_dim, expert_dim, dropout=dropout)
                for _ in range(num_task_experts)
            ])
            for _ in range(num_tasks)
        ])

        self.shared_experts = nn.ModuleList([
            ExpertMLP(in_dim, expert_dim, dropout=dropout)
            for _ in range(num_shared_experts)
        ])

        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, num_task_experts + num_shared_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_tasks)
        ])

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

        shared_outs = torch.stack([exp(shared_input) for exp in self.shared_experts], dim=0)
        task_outs = []
        for t in range(num_tasks):
            outs_t = torch.stack([exp(task_inputs[t]) for exp in self.task_experts[t]], dim=0)
            task_outs.append(outs_t)

        new_task_outputs = []
        for t in range(num_tasks):
            all_experts = torch.cat([task_outs[t], shared_outs], dim=0)
            gate = self.task_gates[t](task_inputs[t]).view(B, -1, 1)
            all_experts_b = all_experts.permute(1, 0, 2)
            fused = torch.sum(gate * all_experts_b, dim=1)
            new_task_outputs.append(fused)

        shared_gate_in = torch.cat(task_inputs + [shared_input], dim=-1)
        gate_s = self.shared_gate(shared_gate_in).view(B, -1, 1)
        all_task_experts = torch.cat(task_outs, dim=0)
        all_experts_for_shared = torch.cat([all_task_experts, shared_outs], dim=0)
        all_experts_for_shared_b = all_experts_for_shared.permute(1, 0, 2)
        new_shared_output = torch.sum(gate_s * all_experts_for_shared_b, dim=1)

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

        if self.use_node_embedding:
            if node_embed_mat is None or embed_dim <= 0:
                raise ValueError("use_node_embedding=True 时必须提供 node_embed_mat/embed_dim")
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

        if adj_with_self is None:
            raise ValueError("必须传入 adj_with_self")
        if isinstance(adj_with_self, torch.Tensor):
            adj = adj_with_self.float()
        else:
            adj = torch.from_numpy(adj_with_self.astype(np.float32))
        attn_mask = (adj == 0)  # True=不可注意
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
        x_dense, _ = to_dense_batch(x, batch)
        B, N, _ = x_dense.shape
        assert N == self.num_nodes

        if self.use_node_embedding:
            node_embed = self.node_embed_mat.unsqueeze(0).expand(B, -1, -1)
            node_embed_low = self.embed_alpha * self.node_embed_proj(node_embed)
            fused = torch.cat([x_dense, node_embed_low], dim=-1)
        else:
            fused = x_dense

        h = self.input_proj(fused) + self.pos_embedding
        h = self.encoder(h, mask=self.attn_mask)
        graph_emb = h.mean(dim=1)

        extra = data.extra
        if extra.dim() == 1:
            extra = extra.view(data.num_graphs, -1)
        z = torch.cat([graph_emb, extra], dim=-1)

        task_reps, shared_rep = self.ple(z)
        z_age, z_dm = task_reps
        age_pred = self.age_out(z_age).squeeze(-1)
        dm_logits = self.dm_out(z_dm)
        return age_pred, dm_logits


# =========================================================
# 2) Dataset
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

        raw_vals = row[self.common_hmdb].to_numpy(dtype=np.float32)
        mask_vals = (~np.isnan(raw_vals)).astype(np.float32)

        scaled_vals = np.zeros_like(raw_vals, dtype=np.float32)
        for j in range(len(self.common_hmdb)):
            v = raw_vals[j]
            if not np.isnan(v):
                scaled_vals[j] = (v - self.node_means[j]) / self.node_stds[j]
            else:
                scaled_vals[j] = 0.0

        raw_nonan = raw_vals.copy()
        raw_nonan[np.isnan(raw_nonan)] = 0.0
        sample_sum = raw_nonan.sum()
        within_vals = raw_nonan / sample_sum if sample_sum > 0 else np.zeros_like(raw_nonan, dtype=np.float32)

        x = np.stack([scaled_vals, within_vals, mask_vals], axis=1).astype(np.float32)

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
            extra_list += [extra_scaled[j], extra_mask[j]]
        extra = np.array(extra_list, dtype=np.float32)

        data = Data(
            x=torch.from_numpy(x),
            edge_index=self.edge_index
        )
        data.extra = torch.from_numpy(extra)
        data.y_age = torch.tensor(float(row[target_age_col]), dtype=torch.float32)
        data.y_dm = torch.tensor(int(row[target_dm_col]), dtype=torch.long)
        data.sample_idx = int(idx)
        return data


# =========================================================
# 3) Forward（显式展开 + 可选扰动）
# =========================================================
def forward_with_attention_and_fused(
    model,
    data,
    requires_grad=False,
    attn_mask_override=None,
    node_permutation=None,
    input_mask_node_indices=None,
):
    model.eval()
    x, batch = data.x, data.batch
    x_dense, _ = to_dense_batch(x, batch)  # [B,N,C]
    B, N, _ = x_dense.shape
    assert N == model.num_nodes

    # 1) 节点置零掩蔽（用于 masking benchmark）
    if input_mask_node_indices is not None and len(input_mask_node_indices) > 0:
        x_dense = x_dense.clone()
        x_dense[:, input_mask_node_indices, :] = 0.0

    # 2) 节点顺序打乱（node-label permutation）
    if node_permutation is not None:
        x_dense = x_dense[:, node_permutation, :]

    if model.use_node_embedding:
        node_embed = model.node_embed_mat.unsqueeze(0).expand(B, -1, -1)
        if node_permutation is not None:
            node_embed = node_embed[:, node_permutation, :]
        node_embed_low = model.embed_alpha * model.node_embed_proj(node_embed)
        fused = torch.cat([x_dense, node_embed_low], dim=-1)
    else:
        fused = x_dense

    if requires_grad:
        fused = fused.detach().requires_grad_(True)

    h = model.input_proj(fused) + model.pos_embedding
    attn_mask = model.attn_mask if attn_mask_override is None else attn_mask_override

    attn_weights_all_layers = []
    for layer in model.encoder.layers:
        src = h
        attn_output, attn_weights = layer.self_attn(
            src, src, src,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        src2 = layer.dropout1(attn_output)
        src = layer.norm1(src + src2)
        src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
        src2 = layer.dropout2(src2)
        src = layer.norm2(src + src2)
        h = src
        attn_weights_all_layers.append(attn_weights)

    graph_emb = h.mean(dim=1)
    extra = data.extra
    if extra.dim() == 1:
        extra = extra.view(data.num_graphs, -1)
    z = torch.cat([graph_emb, extra], dim=-1)

    task_reps, shared_rep = model.ple(z)
    z_age, z_dm = task_reps
    age_pred = model.age_out(z_age).squeeze(-1)
    dm_logits = model.dm_out(z_dm)

    return age_pred, dm_logits, attn_weights_all_layers, h, fused, shared_rep


def evaluate_model(
    model,
    loader,
    num_dm_classes,
    attn_mask_override=None,
    node_permutation=None,
    input_mask_node_indices=None,
):
    model.eval()
    all_age_true, all_age_pred = [], []
    all_dm_true, all_dm_prob = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            age_pred, dm_logits, *_ = forward_with_attention_and_fused(
                model,
                batch,
                requires_grad=False,
                attn_mask_override=attn_mask_override,
                node_permutation=node_permutation,
                input_mask_node_indices=input_mask_node_indices,
            )
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
        "age_mae": age_mae, "age_rmse": age_rmse, "age_r2": age_r2,
        "dm_acc": dm_acc, "dm_auc": dm_auc
    }


# =========================================================
# 4) 新增：图扰动工具
# =========================================================
def build_rewired_adj_degree_preserving(adj_with_self, seed=42, nswap_factor=10):
    """度保持重连；返回含自环邻接矩阵（0/1）"""
    if isinstance(adj_with_self, torch.Tensor):
        adj = adj_with_self.detach().cpu().numpy().astype(np.int8)
    else:
        adj = np.asarray(adj_with_self, dtype=np.int8)

    n = adj.shape[0]
    adj_noself = adj.copy()
    np.fill_diagonal(adj_noself, 0)

    if not HAS_NX:
        # fallback：随机对称图（近似保持边数，不严格度保持）
        rng = np.random.default_rng(seed)
        triu_idx = np.triu_indices(n, k=1)
        m = int(adj_noself[triu_idx].sum())
        sel = rng.choice(len(triu_idx[0]), size=m, replace=False)
        new_adj = np.zeros((n, n), dtype=np.int8)
        new_adj[triu_idx[0][sel], triu_idx[1][sel]] = 1
        new_adj = new_adj + new_adj.T
    else:
        G = nx.from_numpy_array(adj_noself)
        H = G.copy()
        m_edges = H.number_of_edges()
        if m_edges > 0:
            nswap = min(nswap_factor * m_edges, 200000)
            max_tries = nswap * 10
            try:
                nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=seed)
            except Exception:
                pass
        new_adj = nx.to_numpy_array(H, dtype=np.int8)

    np.fill_diagonal(new_adj, 1)
    return new_adj


def build_self_only_adj(num_nodes):
    """非图基线：仅允许自注意力（对角=1）"""
    adj = np.eye(num_nodes, dtype=np.int8)
    return adj


def adj_to_attn_mask(adj_with_self):
    if isinstance(adj_with_self, torch.Tensor):
        adj = adj_with_self.float()
    else:
        adj = torch.from_numpy(np.asarray(adj_with_self, dtype=np.float32))
    return (adj == 0).to(device)


# =========================================================
# 5) 新增：全局节点归因（DM/Age/Shared）
# =========================================================
def compute_global_node_importance(
    model,
    loader,
    task_name="DM",
    num_dm_classes=2,
    attn_mask_override=None,
    node_permutation=None,
    input_mask_node_indices=None,
):
    """返回 N 维全局节点重要性（梯度绝对值均值）"""
    model.eval()
    node_sum = None
    count = 0
    dm_positive_class = 1 if num_dm_classes >= 2 else 0

    for batch in loader:
        batch = batch.to(device)
        with torch.enable_grad():
            age_pred, dm_logits, _, _, fused, shared_rep = forward_with_attention_and_fused(
                model, batch,
                requires_grad=True,
                attn_mask_override=attn_mask_override,
                node_permutation=node_permutation,
                input_mask_node_indices=input_mask_node_indices,
            )

        model.zero_grad(set_to_none=True)
        if fused.grad is not None:
            fused.grad.zero_()

        if task_name == "Age":
            target = age_pred.mean()
        elif task_name == "DM":
            target = dm_logits[:, dm_positive_class].mean()
        elif task_name == "Shared":
            target = shared_rep.mean()
        else:
            raise ValueError("task_name 必须为 Age / DM / Shared")

        target.backward()
        grad = fused.grad.abs().mean(dim=-1)  # [B,N]
        g = grad.detach().cpu().sum(dim=0).numpy()

        if node_sum is None:
            node_sum = g
        else:
            node_sum += g
        count += grad.shape[0]

    return node_sum / max(count, 1)


def rank_nodes(node_scores, hmdb_ids):
    order = np.argsort(-node_scores)
    ranks = {hmdb_ids[idx]: int(r + 1) for r, idx in enumerate(order)}
    return ranks, order


# =========================================================
# 6) 新增：Graph perturbation 分析（无需重训练）
# =========================================================
def run_graph_perturbation_analysis(
    dataset_name,
    loader_eval,
    loader_attr,
    model,
    hmdb_ids,
    adj_with_self,
    num_dm_classes,
    prioritized_metabolites=PRIORITIZED_METABOLITES,
):
    print(f"\n===== [{dataset_name}] Graph perturbation analysis（无重训练）=====")
    N = len(hmdb_ids)

    # 原图
    attn_mask_orig = adj_to_attn_mask(adj_with_self)

    # 度保持重连图
    rewired_adj = build_rewired_adj_degree_preserving(adj_with_self, seed=42, nswap_factor=10)
    attn_mask_rewired = adj_to_attn_mask(rewired_adj)

    # 节点标签打乱（输入顺序打乱）
    node_perm = torch.randperm(N, device=device)

    # 性能对比
    rows_perf = []
    for setting, mask_override, perm in [
        ("OriginalGraph", attn_mask_orig, None),
        ("RewiredGraph", attn_mask_rewired, None),
        ("NodePermutation", attn_mask_orig, node_perm),
    ]:
        m = evaluate_model(
            model, loader_eval, num_dm_classes,
            attn_mask_override=mask_override,
            node_permutation=perm,
        )
        m["setting"] = setting
        rows_perf.append(m)
        print(f"[{dataset_name}] {setting}: DM_AUC={m['dm_auc']}, DM_ACC={m['dm_acc']:.4f}, Age_R2={m['age_r2']:.4f}")

    df_perf = pd.DataFrame(rows_perf)
    out_perf = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_GraphPerturb_Performance.tsv")
    df_perf.to_csv(out_perf, sep="\t", index=False)

    # 归因排名对比（以 DM 为例，也可改 Shared）
    rows_rank = []
    for setting, mask_override, perm in [
        ("OriginalGraph", attn_mask_orig, None),
        ("RewiredGraph", attn_mask_rewired, None),
        ("NodePermutation", attn_mask_orig, node_perm),
    ]:
        score_dm = compute_global_node_importance(
            model, loader_attr, task_name="DM", num_dm_classes=num_dm_classes,
            attn_mask_override=mask_override, node_permutation=perm
        )
        ranks, order = rank_nodes(score_dm, hmdb_ids)

        for met in prioritized_metabolites:
            if met in ranks:
                rows_rank.append({
                    "setting": setting,
                    "metabolite": met,
                    "rank_dm_importance": ranks[met],
                    "score_dm_importance": float(score_dm[hmdb_ids.index(met)]),
                })

    df_rank = pd.DataFrame(rows_rank)
    out_rank = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_GraphPerturb_PrioritizedRanks.tsv")
    df_rank.to_csv(out_rank, sep="\t", index=False)

    print(f"[{dataset_name}] 图扰动性能结果: {out_perf}")
    print(f"[{dataset_name}] 图扰动归因排名结果: {out_rank}")


# =========================================================
# 7) 新增：Masking benchmark（无需重训练）
# =========================================================
def select_top_associated_measured_metabolites(df, measured_nodes, top_m=30):
    """用简单效应量 |mean1-mean0|/std 选 DM 相关测量代谢物"""
    rows = []
    y = df[target_dm_col].values
    for m in measured_nodes:
        if m not in df.columns:
            continue
        v = pd.to_numeric(df[m], errors="coerce").values.astype(float)
        mask = ~np.isnan(v)
        if mask.sum() < 20:
            continue
        v0 = v[(y == 0) & mask]
        v1 = v[(y == 1) & mask]
        if len(v0) < 5 or len(v1) < 5:
            continue
        std = np.nanstd(v[mask]) + 1e-8
        effect = np.abs(np.nanmean(v1) - np.nanmean(v0)) / std
        rows.append((m, effect))
    rows = sorted(rows, key=lambda x: -x[1])
    return [x[0] for x in rows[:top_m]]


def run_masking_benchmark(
    dataset_name,
    df_dataset,
    indices,
    common_hmdb,
    node_means,
    node_stds,
    extra_means,
    extra_stds,
    edge_index,
    model,
    num_dm_classes,
    adj_with_self,
    top_m=30,
    mask_ratio=0.3,
    random_seed=42
):
    """
    掩蔽 benchmark：
    1) 选一批“测量且与DM相关”的代谢物作为候选
    2) 每轮随机掩蔽其中一部分节点输入（置零）
    3) 比较原图 / 随机重连图 / 非图(self-only)在“找回被掩蔽节点”的能力
       用 top-k recall 评估（按DM梯度归因排名）
    """
    print(f"\n===== [{dataset_name}] Masking benchmark（无重训练）=====")
    rng = np.random.default_rng(random_seed)

    dataset = SampleGraphDataset(
        df=df_dataset,
        idxs=indices,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
    )

    loader_attr = GeoDataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    hmdb_ids = list(common_hmdb)
    N = len(hmdb_ids)

    # 候选“可测”代谢物（这里简化为在df中存在列的节点）
    measured_nodes = [m for m in hmdb_ids if m in df_dataset.columns]
    top_assoc = select_top_associated_measured_metabolites(df_dataset.loc[indices], measured_nodes, top_m=top_m)
    if len(top_assoc) == 0:
        print(f"[{dataset_name}] 未找到可用于masking的候选代谢物，跳过。")
        return

    candidate_idx = [hmdb_ids.index(m) for m in top_assoc if m in hmdb_ids]
    if len(candidate_idx) < 5:
        print(f"[{dataset_name}] 候选节点过少（{len(candidate_idx)}），跳过。")
        return

    rewired_adj = build_rewired_adj_degree_preserving(adj_with_self, seed=42, nswap_factor=10)
    self_only_adj = build_self_only_adj(N)

    settings = {
        "OriginalGraph": adj_to_attn_mask(adj_with_self),
        "RewiredGraph": adj_to_attn_mask(rewired_adj),
        "SelfOnly": adj_to_attn_mask(self_only_adj),
    }

    # 多轮随机掩蔽
    rounds = 10
    topk_list = [10, 20, 50]
    rows = []

    for rd in range(rounds):
        m_cnt = max(1, int(len(candidate_idx) * mask_ratio))
        masked = rng.choice(candidate_idx, size=m_cnt, replace=False).tolist()
        masked_set = set(masked)

        for setting_name, attn_mask in settings.items():
            score_dm = compute_global_node_importance(
                model=model,
                loader=loader_attr,
                task_name="DM",
                num_dm_classes=num_dm_classes,
                attn_mask_override=attn_mask,
                node_permutation=None,
                input_mask_node_indices=masked,
            )
            order = np.argsort(-score_dm)  # descending
            for k in topk_list:
                topk_idx = set(order[:k].tolist())
                hit = len(masked_set & topk_idx)
                rec = hit / max(len(masked_set), 1)
                rows.append({
                    "round": rd,
                    "setting": setting_name,
                    "masked_count": len(masked_set),
                    "topk": k,
                    "recall_masked_nodes": rec
                })

    df_rows = pd.DataFrame(rows)
    out_raw = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_MaskingBenchmark_Raw.tsv")
    df_rows.to_csv(out_raw, sep="\t", index=False)

    df_sum = (
        df_rows.groupby(["setting", "topk"], as_index=False)["recall_masked_nodes"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out_sum = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_MaskingBenchmark_Summary.tsv")
    df_sum.to_csv(out_sum, sep="\t", index=False)

    print(f"[{dataset_name}] Masking raw: {out_raw}")
    print(f"[{dataset_name}] Masking summary: {out_sum}")


# =========================================================
# 8) 新增：Adjusted metabolite associations
# =========================================================
def _safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _encode_sex_binary(df, sex_col="Sex"):
    if sex_col not in df.columns:
        return df
    x = df[sex_col]
    if np.issubdtype(x.dtype, np.number):
        return df
    mp = {"M": 1, "Male": 1, "F": 0, "Female": 0}
    df = df.copy()
    df[sex_col] = x.map(mp)
    return df


def run_adjusted_metabolite_association(
    dataset_name,
    df,
    prioritized_metabolites=PRIORITIZED_METABOLITES,
    dm_col=target_dm_col,
    age_col="Age",
    sex_col="Sex",
    bmi_col="BMI",
    extra_sensitivity_covariates=None
):
    """
    对每个代谢物做：
      metabolite ~ DM + Age + Sex + BMI
    并输出 DM 系数/SE/P
    再做年龄分层（tertiles）。
    """
    print(f"\n===== [{dataset_name}] Adjusted metabolite associations =====")
    if extra_sensitivity_covariates is None:
        extra_sensitivity_covariates = []

    d = df.copy()
    d = _encode_sex_binary(d, sex_col=sex_col)

    # 基础字段数值化
    for c in [dm_col, age_col, sex_col, bmi_col] + prioritized_metabolites + extra_sensitivity_covariates:
        if c in d.columns:
            d[c] = _safe_numeric(d[c])

    # 主模型
    rows_main = []
    for met in prioritized_metabolites:
        if met not in d.columns:
            continue

        sub = d[[met, dm_col, age_col, sex_col, bmi_col]].dropna()
        if len(sub) < 30:
            continue

        y = sub[met].astype(float).values
        X = sub[[dm_col, age_col, sex_col, bmi_col]].astype(float)
        X = sm.add_constant(X, has_constant="add")
        try:
            fit = sm.OLS(y, X).fit()
            rows_main.append({
                "metabolite": met,
                "n": len(sub),
                "beta_DM": fit.params.get(dm_col, np.nan),
                "se_DM": fit.bse.get(dm_col, np.nan),
                "p_DM": fit.pvalues.get(dm_col, np.nan),
                "beta_Age": fit.params.get(age_col, np.nan),
                "p_Age": fit.pvalues.get(age_col, np.nan),
                "r2": fit.rsquared
            })
        except Exception:
            pass

    df_main = pd.DataFrame(rows_main)
    out_main = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_AdjustedAssoc_DM_Age_Sex_BMI.tsv")
    df_main.to_csv(out_main, sep="\t", index=False)
    print(f"[{dataset_name}] 主模型结果: {out_main}")

    # 年龄分层（3分位）
    rows_strata = []
    if age_col in d.columns:
        age_valid = d[age_col].dropna()
        if len(age_valid) >= 30:
            q1, q2 = np.quantile(age_valid, [1/3, 2/3])
            d["_age_group"] = pd.cut(
                d[age_col],
                bins=[-np.inf, q1, q2, np.inf],
                labels=["young", "middle", "older"]
            )

            for g in ["young", "middle", "older"]:
                dg = d[d["_age_group"] == g]
                for met in prioritized_metabolites:
                    if met not in dg.columns:
                        continue
                    need_cols = [met, dm_col, sex_col, bmi_col]
                    sub = dg[need_cols].dropna()
                    if len(sub) < 20:
                        continue
                    y = sub[met].astype(float).values
                    X = sub[[dm_col, sex_col, bmi_col]].astype(float)
                    X = sm.add_constant(X, has_constant="add")
                    try:
                        fit = sm.OLS(y, X).fit()
                        rows_strata.append({
                            "age_group": g,
                            "metabolite": met,
                            "n": len(sub),
                            "beta_DM": fit.params.get(dm_col, np.nan),
                            "se_DM": fit.bse.get(dm_col, np.nan),
                            "p_DM": fit.pvalues.get(dm_col, np.nan),
                            "r2": fit.rsquared
                        })
                    except Exception:
                        pass

    df_strata = pd.DataFrame(rows_strata)
    out_strata = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_AdjustedAssoc_AgeStratified.tsv")
    df_strata.to_csv(out_strata, sep="\t", index=False)
    print(f"[{dataset_name}] 年龄分层结果: {out_strata}")

    # 敏感性分析（可用才加）
    rows_sens = []
    sens_cov = [c for c in extra_sensitivity_covariates if c in d.columns]
    if len(sens_cov) > 0:
        for met in prioritized_metabolites:
            cols = [met, dm_col, age_col, sex_col, bmi_col] + sens_cov
            if met not in d.columns:
                continue
            sub = d[cols].dropna()
            if len(sub) < 30:
                continue
            y = sub[met].astype(float).values
            X = sub[[dm_col, age_col, sex_col, bmi_col] + sens_cov].astype(float)
            X = sm.add_constant(X, has_constant="add")
            try:
                fit = sm.OLS(y, X).fit()
                rows_sens.append({
                    "metabolite": met,
                    "n": len(sub),
                    "covariates_extra": ",".join(sens_cov),
                    "beta_DM": fit.params.get(dm_col, np.nan),
                    "se_DM": fit.bse.get(dm_col, np.nan),
                    "p_DM": fit.pvalues.get(dm_col, np.nan),
                    "r2": fit.rsquared
                })
            except Exception:
                pass

    df_sens = pd.DataFrame(rows_sens)
    out_sens = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_AdjustedAssoc_Sensitivity.tsv")
    df_sens.to_csv(out_sens, sep="\t", index=False)
    print(f"[{dataset_name}] 敏感性分析结果: {out_sens}")


# =========================================================
# 9) 你原有 explain_for_dataset（保留）
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
):
    """
    这里使用简化版：先评估，再导出DM全局节点归因排名
    （如果你希望保留你原脚本里完整 attention+edge 导出，可直接替换成原函数）
    """
    print(f"\n===== 开始解释数据集: {dataset_name} =====")
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
    eval_loader = GeoDataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    attr_loader = GeoDataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    metrics = evaluate_model(model, eval_loader, num_dm_classes)
    out_metrics = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_Pred_Metrics.tsv")
    pd.DataFrame([metrics]).to_csv(out_metrics, sep="\t", index=False)
    print(f"[{dataset_name}] 预测指标已保存: {out_metrics}")

    score_dm = compute_global_node_importance(
        model, attr_loader, task_name="DM", num_dm_classes=num_dm_classes
    )
    hmdb_ids = list(common_hmdb)
    order = np.argsort(-score_dm)
    df_rank = pd.DataFrame({
        "rank": np.arange(1, len(hmdb_ids) + 1),
        "HMDB": [hmdb_ids[i] for i in order],
        "dm_importance": [float(score_dm[i]) for i in order]
    })
    out_rank = os.path.join(OUT_EXPLAIN_DIR, f"{dataset_name}_DM_GlobalNodeRank.tsv")
    df_rank.to_csv(out_rank, sep="\t", index=False)
    print(f"[{dataset_name}] 全局DM节点归因已保存: {out_rank}")

    return dataset, eval_loader, attr_loader


# =========================================================
# 10) 数据预处理辅助
# =========================================================
def align_and_preprocess_df(df, common_hmdb, label_encoder=None, apply_log1p=False):
    df = df.copy()

    # 补列
    missing_hmdb = [h for h in common_hmdb if h not in df.columns]
    if missing_hmdb:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing_hmdb)], axis=1)

    missing_extra = [c for c in extra_cols if c not in df.columns]
    if missing_extra:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing_extra)], axis=1)

    # log1p
    if apply_log1p:
        vals = df[common_hmdb].to_numpy(dtype=np.float32)
        non_nan = ~np.isnan(vals)
        non_neg = (vals >= 0) & non_nan
        vals[non_neg] = np.log1p(vals[non_neg])
        df[common_hmdb] = vals

    # DM编码
    if label_encoder is not None:
        df[target_dm_col] = label_encoder.transform(df[target_dm_col].astype(str))
    else:
        df[target_dm_col] = df[target_dm_col].astype(int)

    return df


# =========================================================
# 11) main
# =========================================================
def main():
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    print(f"已加载模型: {MODEL_PATH}")

    hidden_dim = ckpt["hidden_dim"]
    num_layers = ckpt["num_layers"]
    node_in_dim = ckpt["node_in_dim"]
    extra_dim = ckpt["extra_dim"]
    num_dm_classes = ckpt["num_dm_classes"]

    node_means = ckpt["node_means"]
    node_stds = ckpt["node_stds"]
    extra_means = ckpt["extra_means"]
    extra_stds = ckpt["extra_stds"]
    label_encoder = ckpt.get("label_encoder", None)

    common_hmdb = ckpt["common_hmdb"]
    adj_with_self = ckpt["adj_with_self"]

    use_node_embedding = ckpt.get("use_node_embedding", False)
    embed_proj_dim = ckpt.get("embed_proj_dim", 32)
    apply_log1p = ckpt.get("apply_log1p", False)

    num_nodes = len(common_hmdb)
    print(f"建模节点数: {num_nodes}, DM类别数: {num_dm_classes}")

    # edge_index（dataset需要）
    if isinstance(adj_with_self, torch.Tensor):
        adj_np = adj_with_self.cpu().numpy().astype(np.int8)
    else:
        adj_np = np.asarray(adj_with_self, dtype=np.int8)
    adj_no_self = adj_np.copy()
    np.fill_diagonal(adj_no_self, 0)
    src, dst = np.nonzero(adj_no_self)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # node embedding
    node_embed_mat = None
    embed_dim = 0
    if use_node_embedding:
        pkl_path = os.path.join(TRAIN_DATA_DIR, "nodes_embedding.pkl")
        with open(pkl_path, "rb") as f:
            mp = pickle.load(f)
        example_key = next(iter(mp.keys()))
        embed_dim = len(mp[example_key])
        node_embed_mat = np.zeros((num_nodes, embed_dim), dtype=np.float32)
        for i, h in enumerate(common_hmdb):
            if h in mp:
                node_embed_mat[i] = np.asarray(mp[h], dtype=np.float32)
        em = node_embed_mat.mean(0, keepdims=True)
        es = node_embed_mat.std(0, keepdims=True)
        es[es == 0] = 1.0
        node_embed_mat = (node_embed_mat - em) / es

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
    print("模型参数已加载。")

    # ========== Internal Test ==========
    df_train = pd.read_csv(TRAIN_DATA_PATH, sep="\t")
    test_idx = np.load(os.path.join(SPLIT_DIR, "test_idx.npy"))

    df_train = align_and_preprocess_df(
        df_train, common_hmdb, label_encoder=label_encoder, apply_log1p=apply_log1p
    )
    internal_dataset, internal_eval_loader, internal_attr_loader = explain_for_dataset(
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
    )

    # 新增A：Graph perturbation（Internal）
    run_graph_perturbation_analysis(
        dataset_name="InternalTest",
        loader_eval=internal_eval_loader,
        loader_attr=internal_attr_loader,
        model=model,
        hmdb_ids=list(common_hmdb),
        adj_with_self=adj_with_self,
        num_dm_classes=num_dm_classes,
        prioritized_metabolites=PRIORITIZED_METABOLITES,
    )

    # 新增A：Masking（Internal）
    run_masking_benchmark(
        dataset_name="InternalTest",
        df_dataset=df_train,
        indices=test_idx,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
        model=model,
        num_dm_classes=num_dm_classes,
        adj_with_self=adj_with_self,
        top_m=30,
        mask_ratio=0.3,
    )

    # 新增B：Adjusted association（Internal）
    run_adjusted_metabolite_association(
        dataset_name="InternalTest",
        df=df_train.loc[test_idx].copy(),
        prioritized_metabolites=PRIORITIZED_METABOLITES,
        dm_col=target_dm_col,
        age_col="Age",
        sex_col="Sex",
        bmi_col="BMI",
        extra_sensitivity_covariates=["HbA1c", "LipidLoweringMed", "eGFR"]  # 有则使用，无则自动忽略
    )

    # ========== Q300 ==========
    df_q300 = pd.read_csv(Q300_DATA_PATH, sep="\t")
    q300_idx = np.arange(len(df_q300))
    df_q300 = align_and_preprocess_df(
        df_q300, common_hmdb, label_encoder=label_encoder, apply_log1p=apply_log1p
    )

    q300_dataset, q300_eval_loader, q300_attr_loader = explain_for_dataset(
        dataset_name="Q300",
        df=df_q300,
        indices=q300_idx,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
        model=model,
        num_dm_classes=num_dm_classes,
    )

    # 新增A：Graph perturbation（Q300）
    run_graph_perturbation_analysis(
        dataset_name="Q300",
        loader_eval=q300_eval_loader,
        loader_attr=q300_attr_loader,
        model=model,
        hmdb_ids=list(common_hmdb),
        adj_with_self=adj_with_self,
        num_dm_classes=num_dm_classes,
        prioritized_metabolites=PRIORITIZED_METABOLITES,
    )

    # 新增A：Masking（Q300）
    run_masking_benchmark(
        dataset_name="Q300",
        df_dataset=df_q300,
        indices=q300_idx,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
        model=model,
        num_dm_classes=num_dm_classes,
        adj_with_self=adj_with_self,
        top_m=30,
        mask_ratio=0.3,
    )

    # 新增B：Adjusted association（Q300）
    run_adjusted_metabolite_association(
        dataset_name="Q300",
        df=df_q300.copy(),
        prioritized_metabolites=PRIORITIZED_METABOLITES,
        dm_col=target_dm_col,
        age_col="Age",
        sex_col="Sex",
        bmi_col="BMI",
        extra_sensitivity_covariates=["HbA1c", "LipidLoweringMed", "eGFR"]
    )

    print("\n全部分析完成。结果目录：", OUT_EXPLAIN_DIR)


if __name__ == "__main__":
    main()