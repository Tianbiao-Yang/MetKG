# -*- coding: utf-8 -*-
"""
多任务 Graphormer（Age 回归 + DM 分类）训练脚本 + 标准 PLE 多任务头
- 基于 mat_matrix 邻接矩阵，扩展一跳邻居
- 每个样本是一张固定节点数的图

节点特征设计：
- 通道1：跨样本标准化（列 z-score）后的浓度（在训练集上按列 mean/std）
- 通道2：样本内总量归一化后的浓度（反映谱图形状 / 样本内代谢物之间的相对差异）
- 通道3：缺失 mask
- 节点编码：加入节点结构嵌入（nodes_embedding.pkl，原始 1280 维），在模型中线性降维到 EMBED_PROJ_DIM 维后，再与上述浓度特征融合
- 增加 log1p 预处理开关，NMR / LC-MS/MS 数据都用相同预处理
"""

import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
import datetime
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ============ PyTorch 相关 ============
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
    roc_auc_score
)
from sklearn.model_selection import train_test_split


# ============ 1. 可复现性 ============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============ 2. 路径设置 ============
DATA_DIR = "../data/model_training_dataset"
SPLIT_DIR = os.path.join(DATA_DIR, "saved_splits")
os.makedirs(SPLIT_DIR, exist_ok=True)
SAVED_MODELS_DIR = "../saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
LOGS_DIR = "../logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# ============ 2+. 是否使用节点嵌入 ============
USE_NODE_EMBEDDING = True   # 开关：True/False 控制是否使用节点结构嵌入
EMBED_PROJ_DIM = 32         # 节点嵌入降维到 EMBED_PROJ_DIM 维

# ============ 2++. 是否对代谢物浓度做 log1p 预处理 ============
APPLY_LOG1P = True


# ============ 3. Tee：同时输出到终端和日志文件 ============
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOGS_DIR, f"mtl_graphormer_from_mat_run_{timestamp}.log")
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


print("========== 多任务 Graphormer（Age 回归 + DM 分类）训练脚本（基于 mat_matrix 邻接矩阵，扩展一跳邻居 + 可选节点嵌入 + PLE） ==========")
print(f"日志文件：{log_file_path}")
print(f"随机种子：{SEED}")
print("===========================================================\n")


# ============ 4. 载入训练数据（表格） ============
print("1. 载入训练数据表（代谢物 + Age + DM + Sex/BMI）")
train_path = os.path.join(DATA_DIR, "MetKG_training_set_sample_1.txt")
df = pd.read_csv(train_path, sep='\t')

hmdb_cols = [c for c in df.columns if c.startswith("HMDB")]
extra_cols = ["Sex", "BMI"]
target_age_col = "Age"
target_dm_col = "DM"

print(f"数据总样本数: {len(df)}")
print(f"HMDB 代谢物列数: {len(hmdb_cols)}")
print(f"额外特征列: {extra_cols}")
print(f"Age 列: {target_age_col}, DM 列: {target_dm_col}")


# ============ 5. 载入图结构：nodes_list + mat_matrix，并扩展一阶邻居 ============
print("\n2. 载入图结构（nodes_list.pkl + mat_matrix.pkl）")
with open(os.path.join(DATA_DIR, "nodes_list.pkl"), "rb") as f:
    nodes_list = pickle.load(f)
with open(os.path.join(DATA_DIR, "mat_matrix.pkl"), "rb") as f:
    mat_matrix = pickle.load(f)

mat_matrix = np.asarray(mat_matrix, dtype=np.float32)
N = len(nodes_list)
assert mat_matrix.shape == (N, N), f"mat_matrix 形状 {mat_matrix.shape} != (N, N)=({N},{N})"
print(f"原始图节点数 N: {N}")

nodes_set = set(nodes_list)

# 1) 样本里真正有测量的 HMDB
measured_hmdb = [c for c in hmdb_cols if c in nodes_set]
print(f"样本中有测量的 HMDB 节点数: {len(measured_hmdb)}")
if len(measured_hmdb) == 0:
    raise ValueError("在 df 中找不到任何同时出现在 nodes_list 的 HMDB 列")

hmdb_to_full_idx = {hmdb: i for i, hmdb in enumerate(nodes_list)}
measured_idx = np.array([hmdb_to_full_idx[h] for h in measured_hmdb], dtype=np.int64)

# 2) 在全图上基于阈值构造二值邻接矩阵
print("3. 从全图邻接矩阵 + 测量节点，构建扩展子图（包括一阶邻居）")
threshold = 0.5
full_adj_bin = (mat_matrix >= threshold).astype(np.int8)  # [N, N]

# 3) 找到所有与任一测量节点相连的节点（包括自身）
neighbor_mask = full_adj_bin[measured_idx].any(axis=0)  # [N]
all_idx = np.where(neighbor_mask)[0]
sub_idx = np.union1d(all_idx, measured_idx)

sub_adj_bin = full_adj_bin[np.ix_(sub_idx, sub_idx)]

# 4) 扩展后的建模节点列表：测量节点 + 其一跳邻居
model_hmdb = [nodes_list[i] for i in sub_idx]
print(f"扩展后参与建模的 HMDB 节点数（测量节点 + 一阶邻居）: {len(model_hmdb)}")

# 5) 确保 df 里有这些列；对没有测量的节点列，批量补成 NaN
existing_cols = set(df.columns)
missing_cols = [h for h in model_hmdb if h not in existing_cols]
if len(missing_cols) > 0:
    add_df = pd.DataFrame(np.nan, index=df.index, columns=missing_cols)
    df = pd.concat([df, add_df], axis=1)

common_hmdb = model_hmdb

# 为 Graphormer / GNN 准备带自环的邻接矩阵（numpy）
adj_bin = sub_adj_bin.astype(np.int8)
adj_with_self = adj_bin.copy()
np.fill_diagonal(adj_with_self, 1)

src, dst = np.nonzero(adj_bin)
edge_index = torch.tensor(
    np.stack([src, dst], axis=0),
    dtype=torch.long
).contiguous()

num_nodes = len(common_hmdb)
num_edges = edge_index.shape[1]
print(f"扩展子图节点数: {num_nodes}, 边数: {num_edges}")


# ============ 5+. 可选：对代谢物浓度做 log1p 预处理 ============
if APPLY_LOG1P:
    print("\n2+. 对代谢物浓度做 log1p 预处理（仅对非 NaN 且 >=0 的值生效）")
    vals = df[common_hmdb].to_numpy(dtype=np.float32)
    nan_mask = np.isnan(vals)
    non_nan_mask = ~nan_mask

    non_neg_mask = (vals >= 0) & non_nan_mask
    vals[non_neg_mask] = np.log1p(vals[non_neg_mask])

    df[common_hmdb] = vals
else:
    print("\n2+. 不进行 log1p 预处理")


# ============ 5++. 载入节点结构嵌入（nodes_embedding.pkl） ============
print("\n3. 载入节点结构嵌入（nodes_embedding.pkl）")

CONC_MASK_DIM = 3  # [跨样本 z-score, 样本内总量归一化, 缺失 mask]

if USE_NODE_EMBEDDING:
    nodes_embedding_path = os.path.join(DATA_DIR, "nodes_embedding.pkl")
    if not os.path.exists(nodes_embedding_path):
        raise FileNotFoundError(f"未找到节点嵌入文件: {nodes_embedding_path}")

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

    # 按维标准化
    embed_means = node_embed_mat.mean(axis=0, keepdims=True)
    embed_stds = node_embed_mat.std(axis=0, keepdims=True)
    embed_stds[embed_stds == 0] = 1.0
    node_embed_mat = (node_embed_mat - embed_means) / embed_stds

    print(
        f"节点浓度特征维度: {CONC_MASK_DIM}, 节点嵌入原始维度: {embed_dim} "
        f"（将在模型中降到 {EMBED_PROJ_DIM} 维）"
    )
else:
    embed_dim = 0
    node_embed_mat = None
    print(f"节点浓度特征维度: {CONC_MASK_DIM}, 不使用节点嵌入。")


# ============ 6. DM 标签编码 ============
print("\n4. 处理 DM 标签")
label_encoder = None
if df[target_dm_col].dtype == object or not np.issubdtype(df[target_dm_col].dtype, np.number):
    print("DM 为非数值类型，执行 LabelEncoder ...")
    label_encoder = LabelEncoder()
    df[target_dm_col] = label_encoder.fit_transform(df[target_dm_col].astype(str))
    with open(os.path.join(SPLIT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

num_dm_classes = int(df[target_dm_col].nunique())
print(f"DM 不同类别数 = {num_dm_classes}")


# ============ 7. Train/Test 固定划分 ============
print("\n5. Train/Test 划分（8:2），优先使用已保存的 splits")
train_idx_path = os.path.join(SPLIT_DIR, "train_idx.npy")
test_idx_path = os.path.join(SPLIT_DIR, "test_idx.npy")

if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
    print("发现现有划分文件，直接加载。")
    train_idx = np.load(train_idx_path)
    test_idx = np.load(test_idx_path)
else:
    print("未发现划分文件，重新划分并保存。")
    all_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.2, random_state=SEED, shuffle=True
    )
    np.save(train_idx_path, train_idx)
    np.save(test_idx_path, test_idx)

print(f"Train 样本数: {len(train_idx)}, Test 样本数: {len(test_idx)}")


# ============ 8. 在训练集上计算均值/方差（缺失值忽略） ============
print("\n6. 在训练集上拟合特征统计量（mean/std），缺失值忽略")
print("（注意：此处使用的是『可能已 log1p』后的代谢物浓度）")

X_nodes_train_raw = df.loc[train_idx, common_hmdb].to_numpy(dtype=np.float32)
num_features = X_nodes_train_raw.shape[1]

node_means = np.zeros(num_features, dtype=np.float32)
node_stds = np.ones(num_features, dtype=np.float32)

valid_mask = ~np.isnan(X_nodes_train_raw).all(axis=0)
if valid_mask.any():
    node_means[valid_mask] = np.nanmean(X_nodes_train_raw[:, valid_mask], axis=0)
    node_stds[valid_mask] = np.nanstd(X_nodes_train_raw[:, valid_mask], axis=0)
node_stds[node_stds == 0] = 1.0

X_extra_train_raw = df.loc[train_idx, extra_cols].to_numpy(dtype=np.float32)
extra_means = np.nanmean(X_extra_train_raw, axis=0)
extra_stds = np.nanstd(X_extra_train_raw, axis=0)
extra_stds[extra_stds == 0] = 1.0

print("特征统计量计算完成。")


# ============ 9. Graph Dataset ============
class SampleGraphDataset(torch.utils.data.Dataset):
    """
    每个样本：
    - 图结构：相同的 edge_index，num_nodes = len(common_hmdb)
    - 节点特征 x: [num_nodes, 3]
        通道1：跨样本标准化的浓度（列 z-score）
        通道2：样本内总量归一化的浓度（反映谱图形状）
        通道3：缺失 mask
    - 额外特征 extra: [2 * len(extra_cols)] => 每个 extra 特征也做 [标准化数值, 缺失mask]
    - 标签：y_age (float), y_dm (long)
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
        idx = self.idxs[i]
        row = self.df.loc[idx]

        # 代谢物节点特征
        raw_vals = row[self.common_hmdb].to_numpy(dtype=np.float32)
        mask_vals = (~np.isnan(raw_vals)).astype(np.float32)

        # 通道1：跨样本 z-score
        scaled_vals = np.zeros_like(raw_vals, dtype=np.float32)
        for j in range(len(self.common_hmdb)):
            v = raw_vals[j]
            if not np.isnan(v):
                scaled_vals[j] = (v - self.node_means[j]) / self.node_stds[j]
            else:
                scaled_vals[j] = 0.0

        # 通道2：样本内总量归一化
        raw_nonan = raw_vals.copy()
        raw_nonan[np.isnan(raw_nonan)] = 0.0
        sample_sum = raw_nonan.sum()
        if sample_sum > 0:
            within_vals = raw_nonan / sample_sum
        else:
            within_vals = np.zeros_like(raw_nonan, dtype=np.float32)

        x = np.stack([scaled_vals, within_vals, mask_vals], axis=1).astype(np.float32)

        # 额外特征 Sex / BMI
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


print("\n7. 构建 Train/Test Graph Dataset")
train_dataset = SampleGraphDataset(
    df, train_idx,
    common_hmdb, node_means, node_stds,
    extra_cols, extra_means, extra_stds,
    edge_index,
)
test_dataset = SampleGraphDataset(
    df, test_idx,
    common_hmdb, node_means, node_stds,
    extra_cols, extra_means, extra_stds,
    edge_index,
)

print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
print(f"单个样本图的节点数: {num_nodes}, 节点特征维度: {CONC_MASK_DIM}, 额外特征维度: {2 * len(extra_cols)}\n")


# ============ 10. 原 GNN 模型（保留，可选） ============
def build_conv_layer(gnn_type, in_channels, out_channels, heads=4):
    gnn_type = gnn_type.lower()
    if gnn_type == "gcn":
        return GCNConv(in_channels, out_channels)
    elif gnn_type == "sage":
        return SAGEConv(in_channels, out_channels)
    elif gnn_type == "gat":
        return GATConv(in_channels, out_channels, heads=heads, concat=False)
    else:
        raise ValueError(f"不支持的 gnn_type: {gnn_type}")


class MultiTaskGNN(nn.Module):
    def __init__(
        self,
        node_in_dim,
        extra_dim,
        hidden_dim=128,
        num_layers=2,
        gnn_type="gcn",
        dropout=0.3,
        num_dm_classes=2
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


# ============ 10+. 标准 PLE 模块 ============
class ExpertMLP(nn.Module):
    """单个 expert，用一个小 MLP 表示"""
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
    """
    单层【标准 PLE】：
    - 有 T 个任务塔 + 1 个 shared 塔
    - 每个 task 塔有 num_task_experts 个 experts
    - shared 塔有 num_shared_experts 个 experts
    - 每个 task 有自己的 gate，混合「本 task 的 experts + shared 的 experts」
    - shared 塔也有 gate，混合「所有 task experts + shared experts」
    """
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

        # 1) Experts
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

        # 2) Gates
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
        """
        task_inputs: list[num_tasks]，每个 [B, in_dim]
        shared_input: [B, in_dim]

        返回:
        - new_task_outputs: list[num_tasks]，每个 [B, expert_dim]
        - new_shared_output: [B, expert_dim]
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

        # task 塔 gating & 混合
        new_task_outputs = []
        for t in range(num_tasks):
            all_experts = torch.cat([task_outs[t], shared_outs], dim=0)  # [E_t + E_s, B, D]
            gate = self.task_gates[t](task_inputs[t])                    # [B, E_t + E_s]
            gate = gate.view(B, -1, 1)                                   # [B, E_total, 1]

            all_experts_b = all_experts.permute(1, 0, 2)                 # [B, E_total, D]
            fused = torch.sum(gate * all_experts_b, dim=1)               # [B, D]
            new_task_outputs.append(fused)

        # shared 塔 gating & 混合
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
    """
    多层标准 PLE 堆叠：
    - 输入一个共享表示 z_init（例如 graph_emb + extra）
    - 底层：所有 task 塔输入、shared 塔输入都用 z_init
    - 每层：输入是上一层各个塔的输出
    - 最终：返回每个任务塔的表示
    """
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
        z_init: [B, input_dim]，底层共享输入（graph_emb + extra）

        返回:
        - task_reps: list[num_tasks]，每个 [B, expert_dim]
        """
        task_reps = [z_init for _ in range(self.num_tasks)]
        shared_rep = z_init

        for layer in self.layers:
            task_reps, shared_rep = layer(task_reps, shared_rep)

        return task_reps  # list[num_tasks][B, expert_dim]


# ============ 10++. Graphormer + PLE ============
class MultiTaskGraphormer(nn.Module):
    def __init__(
        self,
        conc_dim,             # 节点浓度特征维度，这里是 3
        extra_dim,            # 额外 Sample 特征维度 = 2 * len(extra_cols)
        num_nodes,            # 图中节点数 = len(common_hmdb)
        hidden_dim=128,
        num_layers=2,
        nhead=8,
        dropout=0.1,
        num_dm_classes=2,
        adj_with_self=None,   # numpy 形式的带自环邻接矩阵
        use_node_embedding=False,
        node_embed_mat=None,  # [num_nodes, embed_dim]，numpy
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

        # 1) 节点输入特征 -> hidden_dim
        self.input_proj = nn.Linear(fused_dim, hidden_dim)

        # 2) 节点位置编码
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_nodes, hidden_dim)
        )

        # 3) Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) 注意力 mask
        if adj_with_self is None:
            raise ValueError("必须传入 adj_with_self 用于构造 Graphormer 的注意力 mask")
        adj = torch.from_numpy(adj_with_self.astype(np.float32))
        attn_mask = (adj == 0)
        self.register_buffer("attn_mask", attn_mask)

        # 5) 多任务头：标准 PLE
        ple_input_dim = hidden_dim + extra_dim
        ple_expert_dim = hidden_dim

        self.ple = PLEMultiTaskTower(
            input_dim=ple_input_dim,
            expert_dim=ple_expert_dim,
            num_tasks=2,               # Age & DM 两个任务
            num_layers=2,
            num_task_experts=2,
            num_shared_experts=2,
            dropout=dropout,
        )

        self.age_out = nn.Linear(ple_expert_dim, 1)
        self.dm_out = nn.Linear(ple_expert_dim, num_dm_classes)

    def forward(self, data):
        """
        data.x: [total_nodes_in_batch, conc_dim]
        data.batch: [total_nodes_in_batch]
        data.extra: [batch_size * extra_dim] 一维拼接
        """
        x, batch = data.x, data.batch

        x_dense, mask = to_dense_batch(x, batch)  # [B, N, conc_dim]
        B, N, _ = x_dense.shape
        assert N == self.num_nodes, f"batch 中节点数 {N} 与 num_nodes {self.num_nodes} 不匹配"

        if self.use_node_embedding:
            node_embed = self.node_embed_mat.unsqueeze(0).expand(B, -1, -1)  # [B, N, embed_dim]
            node_embed_low = self.node_embed_proj(node_embed)                # [B, N, proj_dim]
            node_embed_low = self.embed_alpha * node_embed_low
            fused = torch.cat([x_dense, node_embed_low], dim=-1)             # [B, N, conc+proj]
        else:
            fused = x_dense

        h = self.input_proj(fused) + self.pos_embedding  # [B, N, hidden_dim]
        h = self.encoder(h, mask=self.attn_mask)         # [B, N, hidden_dim]

        graph_emb = h.mean(dim=1)  # [B, hidden_dim]

        extra = data.extra
        if extra.dim() == 1:
            extra = extra.view(data.num_graphs, -1)  # [B, extra_dim]

        z = torch.cat([graph_emb, extra], dim=-1)  # [B, hidden_dim + extra_dim]

        task_reps = self.ple(z)   # list[2]，[B, hidden_dim]
        z_age, z_dm = task_reps

        age_pred = self.age_out(z_age).squeeze(-1)   # [B]
        dm_logits = self.dm_out(z_dm)                # [B, num_dm_classes]
        return age_pred, dm_logits


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")


# ============ 11. 评估函数 ============
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

    if len(all_age_true) == 0:
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
            dm_auc = roc_auc_score(y_dm_true, y_dm_prob, multi_class='ovr')
    except Exception:
        pass

    return {
        "age_mae": age_mae,
        "age_rmse": age_rmse,
        "age_r2": age_r2,
        "dm_acc": dm_acc,
        "dm_auc": dm_auc,
    }


# ============ 12. 训练函数 ============
def train_multitask_gnn(
    train_dataset,
    test_dataset,
    num_dm_classes=2,
    gnn_type="graphormer",   # 默认使用 Graphormer + PLE
    hidden_dim=128,
    num_layers=2,
    max_epochs=200,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    age_loss_weight=0.025,
    dm_loss_weight=1.0,
    model_save_name=None
):
    train_loader = GeoDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0
    )
    test_loader = GeoDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0
    )

    node_in_dim = CONC_MASK_DIM
    extra_dim = 2 * len(extra_cols)

    if gnn_type.lower() == "graphormer":
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
            use_node_embedding=USE_NODE_EMBEDDING,
            node_embed_mat=node_embed_mat,
            embed_dim=embed_dim,
            embed_proj_dim=EMBED_PROJ_DIM,
        ).to(device)
    else:
        model = MultiTaskGNN(
            node_in_dim=node_in_dim,
            extra_dim=extra_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=0.1,
            num_dm_classes=num_dm_classes,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_age = nn.MSELoss()
    criterion_dm = nn.CrossEntropyLoss()

    best_val_score = float("inf")
    best_state = None
    best_epoch = None
    best_metrics = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            age_pred, dm_logits = model(batch)

            loss_age = criterion_age(age_pred, batch.y_age)
            loss_dm = criterion_dm(dm_logits, batch.y_dm)
            loss = age_loss_weight * loss_age + dm_loss_weight * loss_dm

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        train_metrics = evaluate_model(model, train_loader, num_dm_classes)
        test_metrics = evaluate_model(model, test_loader, num_dm_classes)

        print(
            f"[MTL-{gnn_type}] Epoch {epoch:03d} | "
            f"TrainLoss={avg_loss:.4f} | "
            f"Test Age MAE={test_metrics['age_mae']:.4f} "
            f"RMSE={test_metrics['age_rmse']:.4f} R2={test_metrics['age_r2']:.4f} | "
            f"Test DM Acc={test_metrics['dm_acc']:.4f} "
            f"AUC={test_metrics['dm_auc'] if not np.isnan(test_metrics['dm_auc']) else 'nan'}"
        )

        #
        auc_for_score = test_metrics['dm_auc'] if not np.isnan(test_metrics['dm_auc']) else 0.0
        score = test_metrics["age_mae"] + (1.0 - auc_for_score)

        if score < best_val_score:
            best_val_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

            best_metrics = {
                "best_epoch": best_epoch,
                "best_val_age_mae": best_val_score,  # 名字沿用原字段
                "train_age_mae": train_metrics["age_mae"],
                "train_age_rmse": train_metrics["age_rmse"],
                "train_age_r2": train_metrics["age_r2"],
                "test_age_mae": test_metrics["age_mae"],
                "test_age_rmse": test_metrics["age_rmse"],
                "test_age_r2": test_metrics["age_r2"],
                "train_dm_acc": train_metrics["dm_acc"],
                "train_dm_auc": train_metrics["dm_auc"],
                "test_dm_acc": test_metrics["dm_acc"],
                "test_dm_auc": test_metrics["dm_auc"],
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    if model_save_name is not None:
        save_path = os.path.join(SAVED_MODELS_DIR, model_save_name)
        save_dict = {
            "model": model.state_dict(),
            "gnn_type": gnn_type,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "node_in_dim": node_in_dim,
            "extra_dim": extra_dim,
            "num_dm_classes": num_dm_classes,
            "best_epoch": best_epoch,
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
        }
        torch.save(save_dict, save_path)
        print(f"[MTL-{gnn_type}] 最优模型已保存到: {save_path}")

    print("\n===== 最优结果 =====")
    print(best_metrics)
    return model, best_metrics


if __name__ == "__main__":
    print("开始训练 MultiTask Graphormer + 标准 PLE 头 ...\n")
    model, best_metrics = train_multitask_gnn(
        train_dataset,
        test_dataset,
        num_dm_classes=num_dm_classes,
        gnn_type="graphormer",          # 使用 Graphormer + PLE
        hidden_dim=128,
        num_layers=2,
        max_epochs=81,
        batch_size=200,
        lr=0.0001,
        weight_decay=1e-4,
        age_loss_weight=0.025,
        dm_loss_weight=1.0,
        model_save_name=f"mtl_graphormer_ple_{timestamp}.pt"
    )

    print("\n训练结束，最优指标如下：")
    for k, v in best_metrics.items():
        print(f"{k}: {v}")

    # 恢复标准输出 / 关闭日志文件
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_f.close()
