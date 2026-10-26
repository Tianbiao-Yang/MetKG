# -*- coding: utf-8 -*-
"""
多任务 Graphormer（Age 回归 + DM 分类）训练脚本 + 标准 PLE 多任务头

1. 在 80% training/development set 内部再划分 validation set：
   - development set 的 87.5% 用于真正训练
   - development set 的 12.5% 用于 validation
   - 等价于总体 70% train / 10% validation / 20% test
2. Early stopping 和模型选择只使用 validation set。
3. Test set 只在每个 seed 的最佳模型确定后评估一次。
4. 重复 5 个 seeds，输出 mean ± SD。
5. 保存每个 seed 的 test predictions，方便后续 DeLong test 和 Dunnett-type MAE comparison。

"""

import os
import sys
import copy
import pickle
import datetime
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

# ============ PyTorch ============
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ PyTorch Geometric ============
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch

# ============ Sklearn ============
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 全局配置
# ============================================================

BASE_SEED = 42

# 用于 5 次重复实验的 seeds
REPEAT_SEEDS = [42, 43, 44, 45, 46]

DATA_DIR = "../data/model_training_dataset"
SPLIT_DIR = os.path.join(DATA_DIR, "saved_splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

SAVED_MODELS_DIR = "../saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

LOGS_DIR = "../logs"
os.makedirs(LOGS_DIR, exist_ok=True)

RESULTS_DIR = "../results/internal_5seeds"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 是否使用节点嵌入
USE_NODE_EMBEDDING = True
EMBED_PROJ_DIM = 32

# 是否对代谢物浓度做 log1p 预处理
APPLY_LOG1P = True

# 训练超参数
HIDDEN_DIM = 128
NUM_LAYERS = 2
MAX_EPOCHS = 200
BATCH_SIZE = 200
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 25
AGE_LOSS_WEIGHT = 0.025
DM_LOSS_WEIGHT = 1.0

# validation 占原始 80% development set 的比例
# 0.125 * 0.8 = 0.1，即总体 10%
VAL_RATIO_WITHIN_DEV = 0.125

# 图邻接矩阵阈值
ADJ_THRESHOLD = 0.5


# ============================================================
# 2. 可复现性函数
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(BASE_SEED)


# ============================================================
# 3. 日志
# ============================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOGS_DIR, f"mtl_graphormer_ple_5seeds_{timestamp}.log")
log_f = open(log_file_path, "w", buffering=1)


class Tee(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


orig_stdout = sys.stdout
orig_stderr = sys.stderr
sys.stdout = Tee(orig_stdout, log_f)
sys.stderr = Tee(orig_stderr, log_f)


print("========== Multi-task Graphormer + PLE: 5-seed internal validation experiment ==========")
print(f"Log file: {log_file_path}")
print(f"Base seed: {BASE_SEED}")
print(f"Repeat seeds: {REPEAT_SEEDS}")
print("=====================================================================================\n")


# ============================================================
# 4. 设备设置
# ============================================================

if torch.cuda.is_available():
    # 如需指定 GPU，可修改这里，例如 cuda:3
    # 如果机器没有 cuda:3，会自动回退到 cuda:0
    preferred_gpu = 3
    if torch.cuda.device_count() > preferred_gpu:
        device = torch.device(f"cuda:{preferred_gpu}")
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}\n")


# ============================================================
# 5. 读取数据
# ============================================================

print("1. Loading training data table")

train_path = os.path.join(DATA_DIR, "MetKG_training_set_sample_1.txt")
df = pd.read_csv(train_path, sep="\t")
df = df.reset_index(drop=True)

hmdb_cols = [c for c in df.columns if c.startswith("HMDB")]
extra_cols = ["Sex", "BMI"]
target_age_col = "Age"
target_dm_col = "DM"

print(f"Total samples: {len(df)}")
print(f"HMDB metabolite columns: {len(hmdb_cols)}")
print(f"Extra columns: {extra_cols}")
print(f"Age column: {target_age_col}, DM column: {target_dm_col}")


# ============================================================
# 6. 额外特征处理
# ============================================================

print("\n2. Processing extra covariates")

for col in extra_cols:
    if col not in df.columns:
        raise ValueError(f"Missing extra column: {col}")

    if df[col].dtype == object or not np.issubdtype(df[col].dtype, np.number):
        print(f"{col} is non-numeric. Applying LabelEncoder.")
        le_extra = LabelEncoder()
        non_na_mask = df[col].notna()
        df.loc[non_na_mask, col] = le_extra.fit_transform(df.loc[non_na_mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 7. 载入图结构
# ============================================================

print("\n3. Loading graph structure: nodes_list.pkl + mat_matrix.pkl")

with open(os.path.join(DATA_DIR, "nodes_list.pkl"), "rb") as f:
    nodes_list = pickle.load(f)

with open(os.path.join(DATA_DIR, "mat_matrix.pkl"), "rb") as f:
    mat_matrix = pickle.load(f)

mat_matrix = np.asarray(mat_matrix, dtype=np.float32)

N = len(nodes_list)
assert mat_matrix.shape == (N, N), f"mat_matrix shape {mat_matrix.shape} != ({N}, {N})"

print(f"Original graph node number N: {N}")

nodes_set = set(nodes_list)
measured_hmdb = [c for c in hmdb_cols if c in nodes_set]

print(f"Measured HMDB nodes found in graph: {len(measured_hmdb)}")

if len(measured_hmdb) == 0:
    raise ValueError("No HMDB columns from df were found in nodes_list.")

hmdb_to_full_idx = {hmdb: i for i, hmdb in enumerate(nodes_list)}
measured_idx = np.array([hmdb_to_full_idx[h] for h in measured_hmdb], dtype=np.int64)

print("\n4. Constructing expanded subgraph: measured nodes + one-hop neighbors")

full_adj_bin = (mat_matrix >= ADJ_THRESHOLD).astype(np.int8)

neighbor_mask = full_adj_bin[measured_idx].any(axis=0)
all_idx = np.where(neighbor_mask)[0]
sub_idx = np.union1d(all_idx, measured_idx)

sub_adj_bin = full_adj_bin[np.ix_(sub_idx, sub_idx)]
model_hmdb = [nodes_list[i] for i in sub_idx]

print(f"Expanded modeling HMDB nodes: {len(model_hmdb)}")

# 给没有实际测量值的一跳邻居节点补 NaN 列
existing_cols = set(df.columns)
missing_cols = [h for h in model_hmdb if h not in existing_cols]

if len(missing_cols) > 0:
    add_df = pd.DataFrame(np.nan, index=df.index, columns=missing_cols)
    df = pd.concat([df, add_df], axis=1)

common_hmdb = model_hmdb

adj_bin = sub_adj_bin.astype(np.int8)
adj_with_self = adj_bin.copy()
np.fill_diagonal(adj_with_self, 1)

src, dst = np.nonzero(adj_bin)
edge_index = torch.tensor(
    np.stack([src, dst], axis=0),
    dtype=torch.long,
).contiguous()

num_nodes = len(common_hmdb)
num_edges = edge_index.shape[1]

print(f"Subgraph nodes: {num_nodes}, edges: {num_edges}")


# ============================================================
# 8. log1p 预处理
# ============================================================

if APPLY_LOG1P:
    print("\n5. Applying log1p preprocessing to metabolite intensities")
    vals = df[common_hmdb].to_numpy(dtype=np.float32)
    nan_mask = np.isnan(vals)
    non_nan_mask = ~nan_mask
    non_neg_mask = (vals >= 0) & non_nan_mask
    vals[non_neg_mask] = np.log1p(vals[non_neg_mask])
    df[common_hmdb] = vals
else:
    print("\n5. Skipping log1p preprocessing")


# ============================================================
# 9. 节点嵌入
# ============================================================

print("\n6. Loading optional node embeddings")

CONC_MASK_DIM = 3

if USE_NODE_EMBEDDING:
    nodes_embedding_path = os.path.join(DATA_DIR, "nodes_embedding.pkl")

    if not os.path.exists(nodes_embedding_path):
        raise FileNotFoundError(f"Cannot find node embedding file: {nodes_embedding_path}")

    with open(nodes_embedding_path, "rb") as f:
        nodes_embedding_dict = pickle.load(f)

    example_key = next(iter(nodes_embedding_dict.keys()))
    embed_dim = len(nodes_embedding_dict[example_key])

    print(f"Detected raw node embedding dimension: {embed_dim}")

    node_embed_mat = np.zeros((num_nodes, embed_dim), dtype=np.float32)
    missing_embed_nodes = []

    for i, hmdb in enumerate(common_hmdb):
        if hmdb in nodes_embedding_dict:
            node_embed_mat[i] = np.asarray(nodes_embedding_dict[hmdb], dtype=np.float32)
        else:
            missing_embed_nodes.append(hmdb)

    if len(missing_embed_nodes) > 0:
        print(f"Warning: {len(missing_embed_nodes)} nodes missing embeddings. Using zero vectors.")

    embed_means = node_embed_mat.mean(axis=0, keepdims=True)
    embed_stds = node_embed_mat.std(axis=0, keepdims=True)
    embed_stds[embed_stds == 0] = 1.0
    node_embed_mat = (node_embed_mat - embed_means) / embed_stds

    print(
        f"Concentration feature dim: {CONC_MASK_DIM}, "
        f"raw node embedding dim: {embed_dim}, "
        f"projected dim: {EMBED_PROJ_DIM}"
    )

else:
    embed_dim = 0
    node_embed_mat = None
    print(f"Concentration feature dim: {CONC_MASK_DIM}; not using node embeddings.")


# ============================================================
# 10. DM 标签编码
# ============================================================

print("\n7. Processing DM labels")

label_encoder = None

if df[target_dm_col].dtype == object or not np.issubdtype(df[target_dm_col].dtype, np.number):
    print("DM is non-numeric. Applying LabelEncoder.")
    label_encoder = LabelEncoder()
    df[target_dm_col] = label_encoder.fit_transform(df[target_dm_col].astype(str))

df[target_dm_col] = df[target_dm_col].astype(int)
num_dm_classes = int(df[target_dm_col].nunique())

print(f"Number of DM classes: {num_dm_classes}")

if label_encoder is not None:
    with open(os.path.join(SPLIT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)


# ============================================================
# 11. 固定原始 80:20 train/test split
# ============================================================

print("\n8. Loading or creating fixed 80:20 train/test split")

train_idx_path = os.path.join(SPLIT_DIR, "train_idx.npy")
test_idx_path = os.path.join(SPLIT_DIR, "test_idx.npy")

if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
    print("Found existing split files. Loading them.")
    dev_idx = np.load(train_idx_path)
    test_idx = np.load(test_idx_path)
else:
    print("No existing split files found. Creating and saving fixed 80:20 split.")
    all_indices = np.arange(len(df))

    # 为了尽量保持原代码行为，这里不强制 stratify。
    # 如果你希望 DM 分层划分，可把 stratify=df[target_dm_col] 加上。
    dev_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=BASE_SEED,
        shuffle=True,
    )

    np.save(train_idx_path, dev_idx)
    np.save(test_idx_path, test_idx)

dev_idx = np.asarray(dev_idx, dtype=np.int64)
test_idx = np.asarray(test_idx, dtype=np.int64)

print(f"Development set size, original 80%: {len(dev_idx)}")
print(f"Fixed held-out test set size, original 20%: {len(test_idx)}")
print("Important: The fixed test set will not be used for early stopping or hyperparameter tuning.")


# ============================================================
# 12. 计算特征统计量
# ============================================================

def fit_feature_statistics(df, train_indices, common_hmdb, extra_cols):
    """
    只在当前 seed 的 subtrain set 上拟合 mean/std。
    validation 和 test 不能参与均值方差估计。
    """
    X_nodes_train_raw = df.loc[train_indices, common_hmdb].to_numpy(dtype=np.float32)

    num_features = X_nodes_train_raw.shape[1]
    node_means = np.zeros(num_features, dtype=np.float32)
    node_stds = np.ones(num_features, dtype=np.float32)

    valid_mask = ~np.isnan(X_nodes_train_raw).all(axis=0)

    if valid_mask.any():
        node_means[valid_mask] = np.nanmean(X_nodes_train_raw[:, valid_mask], axis=0)
        node_stds[valid_mask] = np.nanstd(X_nodes_train_raw[:, valid_mask], axis=0)

    node_stds[node_stds == 0] = 1.0
    node_stds[np.isnan(node_stds)] = 1.0
    node_means[np.isnan(node_means)] = 0.0

    X_extra_train_raw = df.loc[train_indices, extra_cols].to_numpy(dtype=np.float32)

    extra_means = np.nanmean(X_extra_train_raw, axis=0)
    extra_stds = np.nanstd(X_extra_train_raw, axis=0)

    extra_stds[extra_stds == 0] = 1.0
    extra_stds[np.isnan(extra_stds)] = 1.0
    extra_means[np.isnan(extra_means)] = 0.0

    return node_means, node_stds, extra_means, extra_stds


# ============================================================
# 13. Graph Dataset
# ============================================================

class SampleGraphDataset(torch.utils.data.Dataset):
    """
    每个样本：
    - 图结构相同
    - 节点特征 x: [num_nodes, 3]
        通道1：跨样本标准化 z-score 浓度
        通道2：样本内总量归一化浓度
        通道3：缺失 mask
    - extra: [2 * len(extra_cols)]
        每个 extra 特征为 [标准化数值, 缺失 mask]
    - y_age
    - y_dm
    - sample_idx：原始样本 index，方便保存 prediction
    """

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
        idx = int(self.idxs[i])
        row = self.df.loc[idx]

        raw_vals = row[self.common_hmdb].to_numpy(dtype=np.float32)
        mask_vals = (~np.isnan(raw_vals)).astype(np.float32)

        # 通道1：跨样本 z-score
        scaled_vals = np.zeros_like(raw_vals, dtype=np.float32)
        observed = ~np.isnan(raw_vals)
        scaled_vals[observed] = (
            raw_vals[observed] - self.node_means[observed]
        ) / self.node_stds[observed]
        scaled_vals[~observed] = 0.0

        # 通道2：样本内总量归一化
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
        observed_extra = ~np.isnan(extra_raw)
        extra_scaled[observed_extra] = (
            extra_raw[observed_extra] - self.extra_means[observed_extra]
        ) / self.extra_stds[observed_extra]
        extra_scaled[~observed_extra] = 0.0

        extra_list = []
        for j in range(len(self.extra_cols)):
            extra_list.append(extra_scaled[j])
            extra_list.append(extra_mask[j])

        extra = np.array(extra_list, dtype=np.float32)

        y_age = float(row[target_age_col])
        y_dm = int(row[target_dm_col])

        data = Data(
            x=torch.from_numpy(x),
            edge_index=self.edge_index,
        )

        data.extra = torch.from_numpy(extra)
        data.y_age = torch.tensor(y_age, dtype=torch.float32)
        data.y_dm = torch.tensor(y_dm, dtype=torch.long)
        data.sample_idx = torch.tensor(idx, dtype=torch.long)

        return data


# ============================================================
# 14. GNN baseline 模型，可选保留
# ============================================================

def build_conv_layer(gnn_type, in_channels, out_channels, heads=4):
    gnn_type = gnn_type.lower()

    if gnn_type == "gcn":
        return GCNConv(in_channels, out_channels)
    elif gnn_type == "sage":
        return SAGEConv(in_channels, out_channels)
    elif gnn_type == "gat":
        return GATConv(in_channels, out_channels, heads=heads, concat=False)
    else:
        raise ValueError(f"Unsupported gnn_type: {gnn_type}")


class MultiTaskGNN(nn.Module):
    def __init__(
        self,
        node_in_dim,
        extra_dim,
        hidden_dim=128,
        num_layers=2,
        gnn_type="gcn",
        dropout=0.3,
        num_dm_classes=2,
    ):
        super().__init__()

        self.gnn_type = gnn_type
        self.dropout = dropout

        convs = []
        in_dim = node_in_dim

        for _ in range(num_layers):
            conv = build_conv_layer(gnn_type, in_dim, hidden_dim)
            convs.append(conv)
            in_dim = hidden_dim

        self.convs = nn.ModuleList(convs)

        self.age_head = nn.Sequential(
            nn.Linear(hidden_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.dm_head = nn.Sequential(
            nn.Linear(hidden_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_dm_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_emb = global_mean_pool(h, batch)

        extra = data.extra
        if extra.dim() == 1:
            extra = extra.view(data.num_graphs, -1)

        z = torch.cat([graph_emb, extra], dim=-1)

        age_pred = self.age_head(z).squeeze(-1)
        dm_logits = self.dm_head(z)

        return age_pred, dm_logits


# ============================================================
# 15. PLE 模块
# ============================================================

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

        shared_outs = torch.stack(
            [exp(shared_input) for exp in self.shared_experts],
            dim=0,
        )

        task_outs = []

        for t in range(num_tasks):
            outs_t = torch.stack(
                [exp(task_inputs[t]) for exp in self.task_experts[t]],
                dim=0,
            )
            task_outs.append(outs_t)

        new_task_outputs = []

        for t in range(num_tasks):
            all_experts = torch.cat([task_outs[t], shared_outs], dim=0)
            gate = self.task_gates[t](task_inputs[t])
            gate = gate.view(B, -1, 1)

            all_experts_b = all_experts.permute(1, 0, 2)
            fused = torch.sum(gate * all_experts_b, dim=1)

            new_task_outputs.append(fused)

        shared_gate_in = torch.cat(task_inputs + [shared_input], dim=-1)
        gate_s = self.shared_gate(shared_gate_in)
        gate_s = gate_s.view(B, -1, 1)

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
        task_reps = [z_init for _ in range(self.num_tasks)]
        shared_rep = z_init

        for layer in self.layers:
            task_reps, shared_rep = layer(task_reps, shared_rep)

        return task_reps


# ============================================================
# 16. Graphormer + PLE
# ============================================================

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
        embed_proj_dim=EMBED_PROJ_DIM,
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

        if self.use_node_embedding:
            if node_embed_mat is None or embed_dim <= 0:
                raise ValueError(
                    "When use_node_embedding=True, node_embed_mat and embed_dim must be provided."
                )

            node_embed_mat_tensor = torch.from_numpy(node_embed_mat.astype(np.float32))
            self.register_buffer("node_embed_mat", node_embed_mat_tensor)
            self.node_embed_proj = nn.Linear(embed_dim, embed_proj_dim)
            fused_dim = conc_dim + embed_proj_dim

        else:
            fused_dim = conc_dim

        self.input_proj = nn.Linear(fused_dim, hidden_dim)

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_nodes, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        if adj_with_self is None:
            raise ValueError("adj_with_self must be provided for Graphormer attention mask.")

        adj = torch.from_numpy(adj_with_self.astype(np.float32))
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

        x_dense, mask = to_dense_batch(x, batch)
        B, N, _ = x_dense.shape

        assert N == self.num_nodes, (
            f"Batch node number {N} does not match model num_nodes {self.num_nodes}"
        )

        if self.use_node_embedding:
            node_embed = self.node_embed_mat.unsqueeze(0).expand(B, -1, -1)
            node_embed_low = self.node_embed_proj(node_embed)
            node_embed_low = self.embed_alpha * node_embed_low
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

        task_reps = self.ple(z)
        z_age, z_dm = task_reps

        age_pred = self.age_out(z_age).squeeze(-1)
        dm_logits = self.dm_out(z_dm)

        return age_pred, dm_logits


# ============================================================
# 17. 评估函数
# ============================================================

def evaluate_model(model, loader, num_dm_classes, return_predictions=False):
    model.eval()

    all_sample_idx = []
    all_age_true = []
    all_age_pred = []
    all_dm_true = []
    all_dm_prob = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            age_pred, dm_logits = model(batch)

            prob = torch.softmax(dm_logits, dim=1).cpu().numpy()

            all_sample_idx.append(batch.sample_idx.cpu().numpy())
            all_age_pred.append(age_pred.cpu().numpy())
            all_age_true.append(batch.y_age.cpu().numpy())
            all_dm_prob.append(prob)
            all_dm_true.append(batch.y_dm.cpu().numpy())

    if len(all_age_true) == 0:
        raise RuntimeError("Evaluation loader is empty.")

    sample_idx = np.concatenate(all_sample_idx, axis=0)
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

    metrics = {
        "age_mae": age_mae,
        "age_rmse": age_rmse,
        "age_r2": age_r2,
        "dm_acc": dm_acc,
        "dm_auc": dm_auc,
    }

    if return_predictions:
        pred_dict = {
            "sample_idx": sample_idx,
            "age_true": y_age_true,
            "age_pred": y_age_pred,
            "age_abs_error": np.abs(y_age_true - y_age_pred),
            "dm_true": y_dm_true,
            "dm_pred": y_dm_pred,
        }

        for c in range(y_dm_prob.shape[1]):
            pred_dict[f"dm_prob_class_{c}"] = y_dm_prob[:, c]

        return metrics, pred_dict

    return metrics


# ============================================================
# 18. 单个 seed 的训练函数
# ============================================================

def train_one_seed(
    seed,
    subtrain_idx,
    val_idx,
    test_idx,
    model_save_name=None,
):
    set_seed(seed)

    print("\n" + "=" * 90)
    print(f"Running seed = {seed}")
    print(f"Subtrain size: {len(subtrain_idx)}")
    print(f"Validation size: {len(val_idx)}")
    print(f"Fixed test size: {len(test_idx)}")
    print("=" * 90)

    # 关键：只在 subtrain 上计算 mean/std
    node_means, node_stds, extra_means, extra_stds = fit_feature_statistics(
        df=df,
        train_indices=subtrain_idx,
        common_hmdb=common_hmdb,
        extra_cols=extra_cols,
    )

    train_dataset = SampleGraphDataset(
        df=df,
        idxs=subtrain_idx,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
    )

    val_dataset = SampleGraphDataset(
        df=df,
        idxs=val_idx,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
    )

    test_dataset = SampleGraphDataset(
        df=df,
        idxs=test_idx,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = GeoDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    val_loader = GeoDataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = GeoDataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    node_in_dim = CONC_MASK_DIM
    extra_dim = 2 * len(extra_cols)

    model = MultiTaskGraphormer(
        conc_dim=node_in_dim,
        extra_dim=extra_dim,
        num_nodes=num_nodes,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        nhead=8,
        dropout=0.1,
        num_dm_classes=num_dm_classes,
        adj_with_self=adj_with_self,
        use_node_embedding=USE_NODE_EMBEDDING,
        node_embed_mat=node_embed_mat,
        embed_dim=embed_dim,
        embed_proj_dim=EMBED_PROJ_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion_age = nn.MSELoss()
    criterion_dm = nn.CrossEntropyLoss()

    best_val_score = float("inf")
    best_state = None
    best_epoch = None
    best_val_metrics = None
    patience_cnt = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            age_pred, dm_logits = model(batch)

            loss_age = criterion_age(age_pred, batch.y_age)
            loss_dm = criterion_dm(dm_logits, batch.y_dm)

            loss = AGE_LOSS_WEIGHT * loss_age + DM_LOSS_WEIGHT * loss_dm

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        train_metrics = evaluate_model(model, train_loader, num_dm_classes)
        val_metrics = evaluate_model(model, val_loader, num_dm_classes)

        val_auc_for_score = val_metrics["dm_auc"] if not np.isnan(val_metrics["dm_auc"]) else 0.0

        # 综合 validation early stopping 指标
        # 越小越好：Age MAE + (1 - AUC)
        val_score = val_metrics["age_mae"] + (1.0 - val_auc_for_score)

        print(
            f"[Seed {seed}] Epoch {epoch:03d} | "
            f"TrainLoss={avg_loss:.4f} | "
            f"Train MAE={train_metrics['age_mae']:.4f}, AUC={train_metrics['dm_auc']:.4f} | "
            f"Val MAE={val_metrics['age_mae']:.4f}, "
            f"Val RMSE={val_metrics['age_rmse']:.4f}, "
            f"Val R2={val_metrics['age_r2']:.4f}, "
            f"Val Acc={val_metrics['dm_acc']:.4f}, "
            f"Val AUC={val_metrics['dm_auc'] if not np.isnan(val_metrics['dm_auc']) else 'nan'}, "
            f"ValScore={val_score:.4f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_metrics = val_metrics.copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

            if patience_cnt >= PATIENCE:
                print(f"[Seed {seed}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 注意：test set 只在这里评估一次，用于最终报告
    test_metrics, test_pred_dict = evaluate_model(
        model,
        test_loader,
        num_dm_classes,
        return_predictions=True,
    )

    print(f"\n[Seed {seed}] Best epoch based on validation: {best_epoch}")
    print(f"[Seed {seed}] Best validation metrics: {best_val_metrics}")
    print(f"[Seed {seed}] Final fixed-test metrics: {test_metrics}")

    if model_save_name is not None:
        save_path = os.path.join(SAVED_MODELS_DIR, model_save_name)

        save_dict = {
            "model": model.state_dict(),
            "seed": seed,
            "gnn_type": "graphormer",
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "node_in_dim": node_in_dim,
            "extra_dim": extra_dim,
            "num_dm_classes": num_dm_classes,
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "node_means": node_means,
            "node_stds": node_stds,
            "extra_means": extra_means,
            "extra_stds": extra_stds,
            "label_encoder": label_encoder,
            "common_hmdb": common_hmdb,
            "adj_with_self": adj_with_self,
            "use_node_embedding": USE_NODE_EMBEDDING,
            "embed_proj_dim": EMBED_PROJ_DIM,
            "apply_log1p": APPLY_LOG1P,
            "subtrain_idx": subtrain_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }

        torch.save(save_dict, save_path)
        print(f"[Seed {seed}] Model saved to: {save_path}")

    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "val_age_mae": best_val_metrics["age_mae"],
        "val_age_rmse": best_val_metrics["age_rmse"],
        "val_age_r2": best_val_metrics["age_r2"],
        "val_dm_acc": best_val_metrics["dm_acc"],
        "val_dm_auc": best_val_metrics["dm_auc"],
        "test_age_mae": test_metrics["age_mae"],
        "test_age_rmse": test_metrics["age_rmse"],
        "test_age_r2": test_metrics["age_r2"],
        "test_dm_acc": test_metrics["dm_acc"],
        "test_dm_auc": test_metrics["dm_auc"],
    }, test_pred_dict


# ============================================================
# 19. 主程序：5 seeds 重复实验
# ============================================================

if __name__ == "__main__":

    print("\nStarting 5-seed internal validation experiment")
    print("The fixed 20% test set will remain unchanged across all seeds.")
    print("Within the original 80% development set, validation is carved out separately for each seed.\n")

    all_metrics = []

    for seed in REPEAT_SEEDS:
        set_seed(seed)

        # 从原始 80% development set 中划分 subtrain/validation
        y_dev = df.loc[dev_idx, target_dm_col].values

        # 如果某个类别样本太少导致 stratify 报错，则自动退回非分层划分
        try:
            subtrain_idx, val_idx = train_test_split(
                dev_idx,
                test_size=VAL_RATIO_WITHIN_DEV,
                random_state=seed,
                shuffle=True,
                stratify=y_dev,
            )
        except ValueError as e:
            print(f"[Seed {seed}] Stratified validation split failed: {e}")
            print(f"[Seed {seed}] Falling back to non-stratified validation split.")

            subtrain_idx, val_idx = train_test_split(
                dev_idx,
                test_size=VAL_RATIO_WITHIN_DEV,
                random_state=seed,
                shuffle=True,
            )

        subtrain_idx = np.asarray(subtrain_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)

        # 保存每个 seed 的三划分 index
        np.save(os.path.join(SPLIT_DIR, f"subtrain_idx_seed{seed}.npy"), subtrain_idx)
        np.save(os.path.join(SPLIT_DIR, f"val_idx_seed{seed}.npy"), val_idx)
        np.save(os.path.join(SPLIT_DIR, f"test_idx_seed{seed}.npy"), test_idx)

        model_save_name = f"mtl_graphormer_ple_internal_seed{seed}_{timestamp}.pt"

        model, metrics, pred_dict = train_one_seed(
            seed=seed,
            subtrain_idx=subtrain_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            model_save_name=model_save_name,
        )

        all_metrics.append(metrics)

        # 保存该 seed 的 test predictions
        pred_df = pd.DataFrame(pred_dict)
        pred_df["seed"] = seed

        pred_path = os.path.join(
            RESULTS_DIR,
            f"test_predictions_seed{seed}_{timestamp}.csv",
        )
        pred_df.to_csv(pred_path, index=False)

        print(f"[Seed {seed}] Test predictions saved to: {pred_path}")

    # 汇总 metrics
    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = os.path.join(
        RESULTS_DIR,
        f"test_metrics_5seeds_{timestamp}.csv",
    )
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 90)
    print("Per-seed test metrics")
    print(metrics_df)
    print(f"Saved per-seed metrics to: {metrics_path}")

    # 计算 mean ± SD
    metric_cols = [
        "test_age_mae",
        "test_age_rmse",
        "test_age_r2",
        "test_dm_acc",
        "test_dm_auc",
    ]

    summary_rows = []

    for col in metric_cols:
        vals = metrics_df[col].astype(float).values
        summary_rows.append({
            "metric": col,
            "mean": np.nanmean(vals),
            "sd": np.nanstd(vals, ddof=1),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        RESULTS_DIR,
        f"test_metrics_5seeds_mean_sd_{timestamp}.csv",
    )
    summary_df.to_csv(summary_path, index=False)

    print("\nMean ± SD over 5 seeds")
    for _, row in summary_df.iterrows():
        print(f"{row['metric']}: {row['mean']:.6f} ± {row['sd']:.6f}")

    print(f"\nSaved mean±SD summary to: {summary_path}")

    print("\nExperiment finished.")
    print("Reminder:")
    print("1. These 5-seed results are for internal robustness/statistical analysis.")
    print("2. Your original locked model can still be used for external validation.")
    print("3. For DeLong test, use saved dm_prob_class_1 and dm_true from each model/baseline.")
    print("4. For Dunnett-type MAE comparison, use saved age_abs_error at the subject level.")

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_f.close()