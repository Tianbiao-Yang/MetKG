# -*- coding: utf-8 -*-
"""
Baseline training script with one-hop neighbor expansion.

Models:
- RF
- MLP
- GCN
- GAT
- GraphSAGE

Protocol:
1. Internal dataset:
   - Use fixed 80:20 train/test split.
   - 80% train is used for training.
   - 20% test is used as internal held-out test.
   - No validation split.
   - No early stopping.
   - Torch models are trained for fixed MAX_EPOCHS.

2. External dataset:
   - Evaluated once using the trained model.
   - No retraining, tuning, early stopping, or model selection.

3. Graph / feature setting:
   - Use measured HMDB metabolites + one-hop neighbors.
   - Unmeasured one-hop neighbor columns are added as NaN.
   - Node features:
       channel 1: z-score concentration
       channel 2: within-sample normalized concentration
       channel 3: missing mask
   - Extra features: Sex/BMI as [z-score value, missing mask]
"""

import os
import sys
import copy
import pickle
import datetime
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    global_mean_pool,
)

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ============================================================
# 1. Global configuration
# ============================================================

BASE_SEED = 42
REPEAT_SEEDS = [42]

DATA_DIR = "../data/model_training_dataset"
INTERNAL_DATA_PATH = os.path.join(DATA_DIR, "MetKG_training_set_sample_1.txt")

EXTERNAL_DATA_PATH = "../data/model_evaluation_dataset/MetKG_Q300_Evaluation_Set.txt"

SPLIT_DIR = os.path.join(DATA_DIR, "saved_splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

SAVED_MODELS_DIR = "../saved_models/final_gcn_gat_sage_80dev"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

LOGS_DIR = "../logs"
os.makedirs(LOGS_DIR, exist_ok=True)

RESULTS_DIR = "../results/final_gcn_gat_sage_80dev_test_external"
os.makedirs(RESULTS_DIR, exist_ok=True)

APPLY_LOG1P = True
ADJ_THRESHOLD = 0.5

# 这里按你的要求：baseline 也拓展一阶邻居
USE_ONE_HOP_NEIGHBORS = True

# Torch model hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 2
MAX_EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
AGE_LOSS_WEIGHT = 0.025
DM_LOSS_WEIGHT = 1.0

# 你当前只跑 GCN；如果要全跑，改成 ["RF", "MLP", "GCN", "GAT", "GraphSAGE"]
BASELINE_MODELS = ["GCN","GraphSAGE","GAT"]

extra_cols = ["Sex", "BMI"]
target_age_col = "Age"
target_dm_col = "DM"

CONC_MASK_DIM = 3


# ============================================================
# 2. Reproducibility
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
# 3. Logging
# ============================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(
    LOGS_DIR,
    f"baseline_onehop_80train_external_noearlystop_{timestamp}.log"
)
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

print("========== Baseline one-hop 80:20 internal + external evaluation ==========")
print(f"Log file: {log_file_path}")
print(f"Internal data: {INTERNAL_DATA_PATH}")
print(f"External data: {EXTERNAL_DATA_PATH}")
print(f"Base seed: {BASE_SEED}")
print(f"Repeat seeds: {REPEAT_SEEDS}")
print(f"Baselines: {BASELINE_MODELS}")
print(f"USE_ONE_HOP_NEIGHBORS = {USE_ONE_HOP_NEIGHBORS}")
print("==========================================================================\n")


# ============================================================
# 4. Device
# ============================================================

if torch.cuda.is_available():
    preferred_gpu = 3
    if torch.cuda.device_count() > preferred_gpu:
        device = torch.device(f"cuda:{preferred_gpu}")
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}\n")


# ============================================================
# 5. Load internal and external data
# ============================================================

print("1. Loading internal and external data")

df_internal = pd.read_csv(INTERNAL_DATA_PATH, sep="\t").reset_index(drop=True)

if not os.path.exists(EXTERNAL_DATA_PATH):
    raise FileNotFoundError(f"External data file not found: {EXTERNAL_DATA_PATH}")

df_external = pd.read_csv(EXTERNAL_DATA_PATH, sep="\t").reset_index(drop=True)

hmdb_cols = [c for c in df_internal.columns if c.startswith("HMDB")]

print(f"Internal samples: {len(df_internal)}")
print(f"External samples: {len(df_external)}")
print(f"Internal HMDB columns: {len(hmdb_cols)}")
print(f"Extra columns: {extra_cols}")
print(f"Targets: {target_age_col}, {target_dm_col}")

for col in [target_age_col, target_dm_col]:
    if col not in df_internal.columns:
        raise ValueError(f"Internal data missing target column: {col}")
    if col not in df_external.columns:
        raise ValueError(f"External data missing target column: {col}")


# ============================================================
# 6. Process Sex/BMI
# ============================================================

print("\n2. Processing extra covariates")

def process_extra_covariates(train_df, external_df, extra_cols):
    train_df = train_df.copy()
    external_df = external_df.copy()
    extra_encoders = {}

    for col in extra_cols:
        if col not in train_df.columns:
            raise ValueError(f"Internal data missing extra column: {col}")

        if col not in external_df.columns:
            print(f"External data missing extra column {col}; adding NaN.")
            external_df[col] = np.nan

        train_numeric = np.issubdtype(train_df[col].dtype, np.number)
        external_numeric = np.issubdtype(external_df[col].dtype, np.number)

        if train_numeric and external_numeric:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            external_df[col] = pd.to_numeric(external_df[col], errors="coerce")
            extra_encoders[col] = None
        else:
            print(f"{col} treated as categorical. Fitting mapping on internal data.")
            vals = train_df[col].dropna().astype(str).unique().tolist()
            mapping = {v: i for i, v in enumerate(vals)}

            train_df[col] = train_df[col].astype(str).map(mapping)
            external_df[col] = external_df[col].astype(str).map(mapping)

            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            external_df[col] = pd.to_numeric(external_df[col], errors="coerce")

            extra_encoders[col] = mapping

    return train_df, external_df, extra_encoders


df_internal, df_external, extra_encoders = process_extra_covariates(
    df_internal,
    df_external,
    extra_cols,
)


# ============================================================
# 7. Load graph structure and expand one-hop neighbors
# ============================================================

print("\n3. Loading graph structure and expanding one-hop neighbors")

with open(os.path.join(DATA_DIR, "nodes_list.pkl"), "rb") as f:
    nodes_list = pickle.load(f)

with open(os.path.join(DATA_DIR, "mat_matrix.pkl"), "rb") as f:
    mat_matrix = pickle.load(f)

mat_matrix = np.asarray(mat_matrix, dtype=np.float32)

N = len(nodes_list)
assert mat_matrix.shape == (N, N), f"mat_matrix shape {mat_matrix.shape} != ({N}, {N})"

nodes_set = set(nodes_list)

measured_hmdb = [c for c in hmdb_cols if c in nodes_set]

if len(measured_hmdb) == 0:
    raise ValueError("No measured HMDB columns found in nodes_list.")

hmdb_to_full_idx = {hmdb: i for i, hmdb in enumerate(nodes_list)}
measured_idx = np.array([hmdb_to_full_idx[h] for h in measured_hmdb], dtype=np.int64)

print(f"Measured HMDB nodes found in KG: {len(measured_hmdb)}")

full_adj_bin = (mat_matrix >= ADJ_THRESHOLD).astype(np.int8)

if USE_ONE_HOP_NEIGHBORS:
    neighbor_mask = full_adj_bin[measured_idx].any(axis=0)
    all_idx = np.where(neighbor_mask)[0]
    sub_idx = np.union1d(all_idx, measured_idx)
    print("Using measured HMDB nodes + one-hop neighbors.")
else:
    sub_idx = measured_idx
    print("Using measured HMDB nodes only.")

sub_adj_bin = full_adj_bin[np.ix_(sub_idx, sub_idx)]
model_hmdb = [nodes_list[i] for i in sub_idx]

# 内部数据中没有的 one-hop 节点补 NaN
missing_internal_cols = [h for h in model_hmdb if h not in df_internal.columns]

if len(missing_internal_cols) > 0:
    print(f"Internal data missing {len(missing_internal_cols)} one-hop HMDB columns; adding NaN.")
    add_df = pd.DataFrame(np.nan, index=df_internal.index, columns=missing_internal_cols)
    df_internal = pd.concat([df_internal, add_df], axis=1)

# 外部数据中没有的节点也补 NaN
missing_external_cols = [h for h in model_hmdb if h not in df_external.columns]

if len(missing_external_cols) > 0:
    print(f"External data missing {len(missing_external_cols)} HMDB columns; adding NaN.")
    add_df = pd.DataFrame(np.nan, index=df_external.index, columns=missing_external_cols)
    df_external = pd.concat([df_external, add_df], axis=1)

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

print(f"Expanded graph nodes: {num_nodes}")
print(f"Expanded graph edges: {num_edges}")


# ============================================================
# 8. log1p preprocessing
# ============================================================

if APPLY_LOG1P:
    print("\n4. Applying log1p preprocessing to internal and external data")

    for name, dfx in [("internal", df_internal), ("external", df_external)]:
        vals = dfx[common_hmdb].to_numpy(dtype=np.float32)
        nan_mask = np.isnan(vals)
        non_nan_mask = ~nan_mask
        non_neg_mask = (vals >= 0) & non_nan_mask
        vals[non_neg_mask] = np.log1p(vals[non_neg_mask])
        dfx[common_hmdb] = vals
        print(f"Applied log1p to {name} data.")
else:
    print("\n4. Skipping log1p preprocessing")


# ============================================================
# 9. DM label encoding
# ============================================================

print("\n5. Processing DM labels")

label_encoder = None

if df_internal[target_dm_col].dtype == object or not np.issubdtype(df_internal[target_dm_col].dtype, np.number):
    print("DM is non-numeric. Fitting LabelEncoder on internal data.")
    label_encoder = LabelEncoder()
    df_internal[target_dm_col] = label_encoder.fit_transform(df_internal[target_dm_col].astype(str))

    external_labels = df_external[target_dm_col].astype(str).values
    unknown_labels = sorted(set(external_labels) - set(label_encoder.classes_))

    if len(unknown_labels) > 0:
        raise ValueError(f"External DM contains unseen labels: {unknown_labels}")

    df_external[target_dm_col] = label_encoder.transform(df_external[target_dm_col].astype(str))
else:
    df_internal[target_dm_col] = pd.to_numeric(df_internal[target_dm_col], errors="raise").astype(int)
    df_external[target_dm_col] = pd.to_numeric(df_external[target_dm_col], errors="raise").astype(int)

num_dm_classes = int(df_internal[target_dm_col].nunique())

print(f"Number of DM classes: {num_dm_classes}")
print("Internal DM distribution:")
print(df_internal[target_dm_col].value_counts().sort_index())
print("External DM distribution:")
print(df_external[target_dm_col].value_counts().sort_index())


# ============================================================
# 10. Load fixed 80:20 internal split
# ============================================================

print("\n6. Loading fixed 80:20 internal train/test split")

train_idx_path = os.path.join(SPLIT_DIR, "train_idx.npy")
test_idx_path = os.path.join(SPLIT_DIR, "test_idx.npy")

if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
    train_idx = np.load(train_idx_path)
    internal_test_idx = np.load(test_idx_path)
    print("Loaded existing train_idx.npy and test_idx.npy.")
else:
    print("No existing split files found. Creating fixed 80:20 split.")
    all_indices = np.arange(len(df_internal))

    train_idx, internal_test_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=BASE_SEED,
        shuffle=True,
    )

    np.save(train_idx_path, train_idx)
    np.save(test_idx_path, internal_test_idx)

train_idx = np.asarray(train_idx, dtype=np.int64)
internal_test_idx = np.asarray(internal_test_idx, dtype=np.int64)
external_idx = np.arange(len(df_external), dtype=np.int64)

print(f"Internal train size: {len(train_idx)}")
print(f"Internal test size: {len(internal_test_idx)}")
print(f"External test size: {len(external_idx)}")
print("No validation split. No early stopping.")


# ============================================================
# 11. Feature statistics and feature builders
# ============================================================

def fit_feature_statistics(df, train_indices, common_hmdb, extra_cols):
    """
    Fit normalization statistics using internal 80% training set only.
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


def build_node_features_for_indices(
    df,
    indices,
    common_hmdb,
    node_means,
    node_stds,
):
    raw = df.loc[indices, common_hmdb].to_numpy(dtype=np.float32)

    mask = (~np.isnan(raw)).astype(np.float32)

    scaled = np.zeros_like(raw, dtype=np.float32)
    observed = ~np.isnan(raw)

    row_idx, col_idx = np.where(observed)

    scaled[row_idx, col_idx] = (
        raw[row_idx, col_idx] - node_means[col_idx]
    ) / node_stds[col_idx]

    raw_nonan = raw.copy()
    raw_nonan[np.isnan(raw_nonan)] = 0.0

    sample_sum = raw_nonan.sum(axis=1, keepdims=True)

    within = np.zeros_like(raw_nonan, dtype=np.float32)
    valid_sum = sample_sum[:, 0] > 0

    within[valid_sum] = raw_nonan[valid_sum] / sample_sum[valid_sum]

    X_node = np.stack([scaled, within, mask], axis=-1).astype(np.float32)

    return X_node


def build_extra_features_for_indices(
    df,
    indices,
    extra_cols,
    extra_means,
    extra_stds,
):
    raw = df.loc[indices, extra_cols].to_numpy(dtype=np.float32)

    mask = (~np.isnan(raw)).astype(np.float32)

    scaled = np.zeros_like(raw, dtype=np.float32)
    observed = ~np.isnan(raw)

    row_idx, col_idx = np.where(observed)

    scaled[row_idx, col_idx] = (
        raw[row_idx, col_idx] - extra_means[col_idx]
    ) / extra_stds[col_idx]

    extra_list = []

    for j in range(len(extra_cols)):
        extra_list.append(scaled[:, j])
        extra_list.append(mask[:, j])

    X_extra = np.stack(extra_list, axis=1).astype(np.float32)

    return X_extra


def build_tabular_features(
    df,
    indices,
    common_hmdb,
    node_means,
    node_stds,
    extra_cols,
    extra_means,
    extra_stds,
):
    X_node = build_node_features_for_indices(
        df=df,
        indices=indices,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
    )

    X_extra = build_extra_features_for_indices(
        df=df,
        indices=indices,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
    )

    X_flat = X_node.reshape(len(indices), -1)
    X = np.concatenate([X_flat, X_extra], axis=1).astype(np.float32)

    y_age = df.loc[indices, target_age_col].to_numpy(dtype=np.float32)
    y_dm = df.loc[indices, target_dm_col].to_numpy(dtype=np.int64)

    return X, y_age, y_dm


# ============================================================
# 12. Dataset classes
# ============================================================

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
        idx = int(self.idxs[i])
        row = self.df.loc[idx]

        raw_vals = row[self.common_hmdb].to_numpy(dtype=np.float32)
        mask_vals = (~np.isnan(raw_vals)).astype(np.float32)

        scaled_vals = np.zeros_like(raw_vals, dtype=np.float32)
        observed = ~np.isnan(raw_vals)

        scaled_vals[observed] = (
            raw_vals[observed] - self.node_means[observed]
        ) / self.node_stds[observed]

        raw_nonan = raw_vals.copy()
        raw_nonan[np.isnan(raw_nonan)] = 0.0

        sample_sum = raw_nonan.sum()

        if sample_sum > 0:
            within_vals = raw_nonan / sample_sum
        else:
            within_vals = np.zeros_like(raw_nonan, dtype=np.float32)

        x = np.stack(
            [scaled_vals, within_vals, mask_vals],
            axis=1
        ).astype(np.float32)

        extra_raw = row[self.extra_cols].to_numpy(dtype=np.float32)
        extra_mask = (~np.isnan(extra_raw)).astype(np.float32)

        extra_scaled = np.zeros_like(extra_raw, dtype=np.float32)
        observed_extra = ~np.isnan(extra_raw)

        extra_scaled[observed_extra] = (
            extra_raw[observed_extra] - self.extra_means[observed_extra]
        ) / self.extra_stds[observed_extra]

        extra_list = []

        for j in range(len(self.extra_cols)):
            extra_list.append(extra_scaled[j])
            extra_list.append(extra_mask[j])

        extra = np.array(extra_list, dtype=np.float32)

        data = Data(
            x=torch.from_numpy(x),
            edge_index=self.edge_index,
        )

        data.extra = torch.from_numpy(extra)
        data.y_age = torch.tensor(float(row[target_age_col]), dtype=torch.float32)
        data.y_dm = torch.tensor(int(row[target_dm_col]), dtype=torch.long)
        data.sample_idx = torch.tensor(idx, dtype=torch.long)

        return data


class TabularTorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y_age, y_dm, sample_idx):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_age = torch.tensor(y_age, dtype=torch.float32)
        self.y_dm = torch.tensor(y_dm, dtype=torch.long)
        self.sample_idx = torch.tensor(sample_idx, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return {
            "x": self.X[i],
            "y_age": self.y_age[i],
            "y_dm": self.y_dm[i],
            "sample_idx": self.sample_idx[i],
        }


# ============================================================
# 13. Models
# ============================================================

class MultiTaskMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        dropout=0.2,
        num_dm_classes=2,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.age_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_dm_classes),
        )

    def forward(self, x):
        h = self.shared(x)
        age_pred = self.age_head(h).squeeze(-1)
        dm_logits = self.dm_head(h)
        return age_pred, dm_logits


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
        dropout=0.1,
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
# 14. Metrics and prediction helpers
# ============================================================

def compute_metrics_from_arrays(
    y_age_true,
    y_age_pred,
    y_dm_true,
    y_dm_prob,
    num_dm_classes,
):
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
    }


def make_prediction_dict(
    sample_idx,
    y_age_true,
    y_age_pred,
    y_dm_true,
    y_dm_prob,
):
    y_dm_pred = np.argmax(y_dm_prob, axis=1)

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

    return pred_dict


def proba_to_full_classes(proba, classes, num_classes):
    n = proba.shape[0]
    full = np.zeros((n, num_classes), dtype=np.float32)

    for j, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, j]

    return full


def evaluate_torch_gnn(model, loader, num_dm_classes, return_predictions=False):
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
            all_age_true.append(batch.y_age.cpu().numpy())
            all_age_pred.append(age_pred.cpu().numpy())
            all_dm_true.append(batch.y_dm.cpu().numpy())
            all_dm_prob.append(prob)

    sample_idx = np.concatenate(all_sample_idx)
    y_age_true = np.concatenate(all_age_true)
    y_age_pred = np.concatenate(all_age_pred)
    y_dm_true = np.concatenate(all_dm_true)
    y_dm_prob = np.concatenate(all_dm_prob)

    metrics = compute_metrics_from_arrays(
        y_age_true=y_age_true,
        y_age_pred=y_age_pred,
        y_dm_true=y_dm_true,
        y_dm_prob=y_dm_prob,
        num_dm_classes=num_dm_classes,
    )

    if return_predictions:
        pred_dict = make_prediction_dict(
            sample_idx=sample_idx,
            y_age_true=y_age_true,
            y_age_pred=y_age_pred,
            y_dm_true=y_dm_true,
            y_dm_prob=y_dm_prob,
        )
        return metrics, pred_dict

    return metrics


def evaluate_torch_mlp(model, loader, num_dm_classes, return_predictions=False):
    model.eval()

    all_sample_idx = []
    all_age_true = []
    all_age_pred = []
    all_dm_true = []
    all_dm_prob = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_age = batch["y_age"].to(device)
            y_dm = batch["y_dm"].to(device)

            age_pred, dm_logits = model(x)
            prob = torch.softmax(dm_logits, dim=1).cpu().numpy()

            all_sample_idx.append(batch["sample_idx"].cpu().numpy())
            all_age_true.append(y_age.cpu().numpy())
            all_age_pred.append(age_pred.cpu().numpy())
            all_dm_true.append(y_dm.cpu().numpy())
            all_dm_prob.append(prob)

    sample_idx = np.concatenate(all_sample_idx)
    y_age_true = np.concatenate(all_age_true)
    y_age_pred = np.concatenate(all_age_pred)
    y_dm_true = np.concatenate(all_dm_true)
    y_dm_prob = np.concatenate(all_dm_prob)

    metrics = compute_metrics_from_arrays(
        y_age_true=y_age_true,
        y_age_pred=y_age_pred,
        y_dm_true=y_dm_true,
        y_dm_prob=y_dm_prob,
        num_dm_classes=num_dm_classes,
    )

    if return_predictions:
        pred_dict = make_prediction_dict(
            sample_idx=sample_idx,
            y_age_true=y_age_true,
            y_age_pred=y_age_pred,
            y_dm_true=y_dm_true,
            y_dm_prob=y_dm_prob,
        )
        return metrics, pred_dict

    return metrics


# ============================================================
# 15. Training functions
# ============================================================

def train_rf_baseline(seed):
    print("\n" + "-" * 80)
    print(f"Training RF baseline | seed={seed}")
    print("-" * 80)

    set_seed(seed)

    node_means, node_stds, extra_means, extra_stds = fit_feature_statistics(
        df=df_internal,
        train_indices=train_idx,
        common_hmdb=common_hmdb,
        extra_cols=extra_cols,
    )

    X_train, y_age_train, y_dm_train = build_tabular_features(
        df_internal, train_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    X_internal_test, y_age_internal_test, y_dm_internal_test = build_tabular_features(
        df_internal, internal_test_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    X_external, y_age_external, y_dm_external = build_tabular_features(
        df_external, external_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    rf_reg = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )

    rf_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )

    rf_reg.fit(X_train, y_age_train)
    rf_clf.fit(X_train, y_dm_train)

    # Internal test
    internal_age_pred = rf_reg.predict(X_internal_test)
    internal_prob_raw = rf_clf.predict_proba(X_internal_test)
    internal_dm_prob = proba_to_full_classes(internal_prob_raw, rf_clf.classes_, num_dm_classes)

    internal_metrics = compute_metrics_from_arrays(
        y_age_true=y_age_internal_test,
        y_age_pred=internal_age_pred,
        y_dm_true=y_dm_internal_test,
        y_dm_prob=internal_dm_prob,
        num_dm_classes=num_dm_classes,
    )

    internal_pred_dict = make_prediction_dict(
        sample_idx=internal_test_idx,
        y_age_true=y_age_internal_test,
        y_age_pred=internal_age_pred,
        y_dm_true=y_dm_internal_test,
        y_dm_prob=internal_dm_prob,
    )

    # External test
    external_age_pred = rf_reg.predict(X_external)
    external_prob_raw = rf_clf.predict_proba(X_external)
    external_dm_prob = proba_to_full_classes(external_prob_raw, rf_clf.classes_, num_dm_classes)

    external_metrics = compute_metrics_from_arrays(
        y_age_true=y_age_external,
        y_age_pred=external_age_pred,
        y_dm_true=y_dm_external,
        y_dm_prob=external_dm_prob,
        num_dm_classes=num_dm_classes,
    )

    external_pred_dict = make_prediction_dict(
        sample_idx=external_idx,
        y_age_true=y_age_external,
        y_age_pred=external_age_pred,
        y_dm_true=y_dm_external,
        y_dm_prob=external_dm_prob,
    )

    save_path = os.path.join(SAVED_MODELS_DIR, f"RF_seed{seed}_{timestamp}.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "rf_reg": rf_reg,
                "rf_clf": rf_clf,
                "seed": seed,
                "node_means": node_means,
                "node_stds": node_stds,
                "extra_means": extra_means,
                "extra_stds": extra_stds,
                "common_hmdb": common_hmdb,
                "extra_cols": extra_cols,
                "apply_log1p": APPLY_LOG1P,
                "use_one_hop_neighbors": USE_ONE_HOP_NEIGHBORS,
                "train_idx": train_idx,
                "internal_test_idx": internal_test_idx,
                "internal_metrics": internal_metrics,
                "external_metrics": external_metrics,
            },
            f,
        )

    print(f"[RF seed={seed}] Internal test metrics: {internal_metrics}")
    print(f"[RF seed={seed}] External test metrics: {external_metrics}")

    return internal_metrics, external_metrics, internal_pred_dict, external_pred_dict


def train_mlp_baseline(seed):
    print("\n" + "-" * 80)
    print(f"Training MLP baseline | seed={seed}")
    print("-" * 80)

    set_seed(seed)

    node_means, node_stds, extra_means, extra_stds = fit_feature_statistics(
        df=df_internal,
        train_indices=train_idx,
        common_hmdb=common_hmdb,
        extra_cols=extra_cols,
    )

    X_train, y_age_train, y_dm_train = build_tabular_features(
        df_internal, train_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    X_internal_test, y_age_internal_test, y_dm_internal_test = build_tabular_features(
        df_internal, internal_test_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    X_external, y_age_external, y_dm_external = build_tabular_features(
        df_external, external_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    train_dataset = TabularTorchDataset(X_train, y_age_train, y_dm_train, train_idx)
    internal_test_dataset = TabularTorchDataset(
        X_internal_test,
        y_age_internal_test,
        y_dm_internal_test,
        internal_test_idx,
    )
    external_dataset = TabularTorchDataset(
        X_external,
        y_age_external,
        y_dm_external,
        external_idx,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    internal_test_loader = torch.utils.data.DataLoader(
        internal_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    external_loader = torch.utils.data.DataLoader(
        external_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    input_dim = X_train.shape[1]

    model = MultiTaskMLP(
        input_dim=input_dim,
        hidden_dim=256,
        dropout=0.2,
        num_dm_classes=num_dm_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion_age = nn.MSELoss()
    criterion_dm = nn.CrossEntropyLoss()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []

        for batch in train_loader:
            x = batch["x"].to(device)
            y_age = batch["y_age"].to(device)
            y_dm = batch["y_dm"].to(device)

            optimizer.zero_grad()

            age_pred, dm_logits = model(x)

            loss_age = criterion_age(age_pred, y_age)
            loss_dm = criterion_dm(dm_logits, y_dm)
            loss = AGE_LOSS_WEIGHT * loss_age + DM_LOSS_WEIGHT * loss_dm

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else float("nan")

        if epoch == 1 or epoch % 10 == 0 or epoch == MAX_EPOCHS:
            train_metrics = evaluate_torch_mlp(model, train_loader, num_dm_classes)
            print(
                f"[MLP seed={seed}] Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Loss={avg_loss:.4f} | "
                f"Train MAE={train_metrics['age_mae']:.4f}, "
                f"Train AUC={train_metrics['dm_auc'] if not np.isnan(train_metrics['dm_auc']) else 'nan'}"
            )

    internal_metrics, internal_pred_dict = evaluate_torch_mlp(
        model,
        internal_test_loader,
        num_dm_classes,
        return_predictions=True,
    )

    external_metrics, external_pred_dict = evaluate_torch_mlp(
        model,
        external_loader,
        num_dm_classes,
        return_predictions=True,
    )

    save_path = os.path.join(SAVED_MODELS_DIR, f"MLP_seed{seed}_{timestamp}.pt")

    torch.save(
        {
            "model": model.state_dict(),
            "seed": seed,
            "input_dim": input_dim,
            "num_dm_classes": num_dm_classes,
            "max_epochs": MAX_EPOCHS,
            "node_means": node_means,
            "node_stds": node_stds,
            "extra_means": extra_means,
            "extra_stds": extra_stds,
            "common_hmdb": common_hmdb,
            "extra_cols": extra_cols,
            "apply_log1p": APPLY_LOG1P,
            "use_one_hop_neighbors": USE_ONE_HOP_NEIGHBORS,
            "train_idx": train_idx,
            "internal_test_idx": internal_test_idx,
            "internal_metrics": internal_metrics,
            "external_metrics": external_metrics,
        },
        save_path,
    )

    print(f"[MLP seed={seed}] Internal test metrics: {internal_metrics}")
    print(f"[MLP seed={seed}] External test metrics: {external_metrics}")

    return internal_metrics, external_metrics, internal_pred_dict, external_pred_dict


def train_gnn_baseline(seed, baseline_name):
    print("\n" + "-" * 80)
    print(f"Training {baseline_name} baseline | seed={seed}")
    print("-" * 80)

    set_seed(seed)

    if baseline_name == "GCN":
        gnn_type = "gcn"
    elif baseline_name == "GAT":
        gnn_type = "gat"
    elif baseline_name == "GraphSAGE":
        gnn_type = "sage"
    else:
        raise ValueError(f"Unsupported GNN baseline: {baseline_name}")

    node_means, node_stds, extra_means, extra_stds = fit_feature_statistics(
        df=df_internal,
        train_indices=train_idx,
        common_hmdb=common_hmdb,
        extra_cols=extra_cols,
    )

    train_dataset = SampleGraphDataset(
        df=df_internal,
        idxs=train_idx,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
    )

    internal_test_dataset = SampleGraphDataset(
        df=df_internal,
        idxs=internal_test_idx,
        common_hmdb=common_hmdb,
        node_means=node_means,
        node_stds=node_stds,
        extra_cols=extra_cols,
        extra_means=extra_means,
        extra_stds=extra_stds,
        edge_index=edge_index,
    )

    external_dataset = SampleGraphDataset(
        df=df_external,
        idxs=external_idx,
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

    internal_test_loader = GeoDataLoader(
        internal_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    external_loader = GeoDataLoader(
        external_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    node_in_dim = CONC_MASK_DIM
    extra_dim = 2 * len(extra_cols)

    model = MultiTaskGNN(
        node_in_dim=node_in_dim,
        extra_dim=extra_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        gnn_type=gnn_type,
        dropout=0.1,
        num_dm_classes=num_dm_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion_age = nn.MSELoss()
    criterion_dm = nn.CrossEntropyLoss()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []

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

            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else float("nan")

        if epoch == 1 or epoch % 10 == 0 or epoch == MAX_EPOCHS:
            train_metrics = evaluate_torch_gnn(model, train_loader, num_dm_classes)
            print(
                f"[{baseline_name} seed={seed}] Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Loss={avg_loss:.4f} | "
                f"Train MAE={train_metrics['age_mae']:.4f}, "
                f"Train AUC={train_metrics['dm_auc'] if not np.isnan(train_metrics['dm_auc']) else 'nan'}"
            )

    internal_metrics, internal_pred_dict = evaluate_torch_gnn(
        model,
        internal_test_loader,
        num_dm_classes,
        return_predictions=True,
    )

    external_metrics, external_pred_dict = evaluate_torch_gnn(
        model,
        external_loader,
        num_dm_classes,
        return_predictions=True,
    )

    save_path = os.path.join(
        SAVED_MODELS_DIR,
        f"{baseline_name}_seed{seed}_{timestamp}.pt"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "seed": seed,
            "baseline_name": baseline_name,
            "gnn_type": gnn_type,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "node_in_dim": node_in_dim,
            "extra_dim": extra_dim,
            "num_dm_classes": num_dm_classes,
            "max_epochs": MAX_EPOCHS,
            "node_means": node_means,
            "node_stds": node_stds,
            "extra_means": extra_means,
            "extra_stds": extra_stds,
            "common_hmdb": common_hmdb,
            "adj_with_self": adj_with_self,
            "edge_index": edge_index,
            "apply_log1p": APPLY_LOG1P,
            "use_one_hop_neighbors": USE_ONE_HOP_NEIGHBORS,
            "train_idx": train_idx,
            "internal_test_idx": internal_test_idx,
            "internal_metrics": internal_metrics,
            "external_metrics": external_metrics,
        },
        save_path,
    )

    print(f"[{baseline_name} seed={seed}] Internal test metrics: {internal_metrics}")
    print(f"[{baseline_name} seed={seed}] External test metrics: {external_metrics}")
    print(f"[{baseline_name} seed={seed}] Model saved to: {save_path}")

    return internal_metrics, external_metrics, internal_pred_dict, external_pred_dict


# ============================================================
# 16. Main
# ============================================================

if __name__ == "__main__":

    all_metrics = []
    all_prediction_files = []

    for seed in REPEAT_SEEDS:
        print("\n" + "=" * 100)
        print(f"Running seed = {seed}")
        print("=" * 100)

        set_seed(seed)

        for baseline_name in BASELINE_MODELS:

            if baseline_name == "RF":
                internal_metrics, external_metrics, internal_pred_dict, external_pred_dict = train_rf_baseline(seed)

            elif baseline_name == "MLP":
                internal_metrics, external_metrics, internal_pred_dict, external_pred_dict = train_mlp_baseline(seed)

            elif baseline_name in ["GCN", "GAT", "GraphSAGE"]:
                internal_metrics, external_metrics, internal_pred_dict, external_pred_dict = train_gnn_baseline(
                    seed=seed,
                    baseline_name=baseline_name,
                )

            else:
                raise ValueError(f"Unsupported baseline: {baseline_name}")

            # 保存 metrics
            row_internal = {
                "model": baseline_name,
                "seed": seed,
                "cohort": "internal_test",
                "age_mae": internal_metrics["age_mae"],
                "age_rmse": internal_metrics["age_rmse"],
                "age_r2": internal_metrics["age_r2"],
                "dm_acc": internal_metrics["dm_acc"],
                "dm_auc": internal_metrics["dm_auc"],
            }

            row_external = {
                "model": baseline_name,
                "seed": seed,
                "cohort": "external_test",
                "age_mae": external_metrics["age_mae"],
                "age_rmse": external_metrics["age_rmse"],
                "age_r2": external_metrics["age_r2"],
                "dm_acc": external_metrics["dm_acc"],
                "dm_auc": external_metrics["dm_auc"],
            }

            all_metrics.append(row_internal)
            all_metrics.append(row_external)

            # 保存 prediction
            internal_pred_df = pd.DataFrame(internal_pred_dict)
            internal_pred_df["model"] = baseline_name
            internal_pred_df["seed"] = seed
            internal_pred_df["cohort"] = "internal_test"

            external_pred_df = pd.DataFrame(external_pred_dict)
            external_pred_df["model"] = baseline_name
            external_pred_df["seed"] = seed
            external_pred_df["cohort"] = "external_test"

            internal_pred_path = os.path.join(
                RESULTS_DIR,
                f"predictions_{baseline_name}_seed{seed}_internal_test_{timestamp}.csv",
            )

            external_pred_path = os.path.join(
                RESULTS_DIR,
                f"predictions_{baseline_name}_seed{seed}_external_test_{timestamp}.csv",
            )

            internal_pred_df.to_csv(internal_pred_path, index=False)
            external_pred_df.to_csv(external_pred_path, index=False)

            all_prediction_files.extend([internal_pred_path, external_pred_path])

            print(f"[{baseline_name} seed={seed}] Internal predictions saved to: {internal_pred_path}")
            print(f"[{baseline_name} seed={seed}] External predictions saved to: {external_pred_path}")

    # 保存 per-seed metrics
    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = os.path.join(
        RESULTS_DIR,
        f"metrics_internal_external_5seeds_{timestamp}.csv",
    )

    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 100)
    print("Per-seed metrics")
    print(metrics_df)
    print(f"Saved metrics to: {metrics_path}")

    # mean ± SD summary
    summary_rows = []

    metric_cols = [
        "age_mae",
        "age_rmse",
        "age_r2",
        "dm_acc",
        "dm_auc",
    ]

    for model_name in BASELINE_MODELS:
        for cohort in ["internal_test", "external_test"]:
            sub_df = metrics_df[
                (metrics_df["model"] == model_name) &
                (metrics_df["cohort"] == cohort)
            ]

            for metric in metric_cols:
                vals = sub_df[metric].astype(float).values

                summary_rows.append({
                    "model": model_name,
                    "cohort": cohort,
                    "metric": metric,
                    "mean": np.nanmean(vals),
                    "sd": np.nanstd(vals, ddof=1),
                    "n": np.sum(~np.isnan(vals)),
                })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        RESULTS_DIR,
        f"summary_internal_external_5seeds_mean_sd_{timestamp}.csv",
    )

    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("Mean ± SD summary")
    print(summary_df)
    print(f"Saved summary to: {summary_path}")

    # 合并所有 predictions
    if len(all_prediction_files) > 0:
        pred_dfs = [pd.read_csv(p) for p in all_prediction_files]
        combined_pred_df = pd.concat(pred_dfs, axis=0, ignore_index=True)

        combined_pred_path = os.path.join(
            RESULTS_DIR,
            f"predictions_internal_external_all_5seeds_{timestamp}.csv",
        )

        combined_pred_df.to_csv(combined_pred_path, index=False)

        print(f"\nCombined predictions saved to: {combined_pred_path}")

    print("\nFinished baseline training and evaluation.")
    print("Outputs can be used for:")
    print("1. Internal test performance")
    print("2. External test performance")
    print("3. AUROC DeLong test: dm_true + dm_prob_class_1")
    print("4. MAE comparison: sample_idx + age_abs_error")
    print("5. Bar plots: mean ± SD from summary CSV")

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_f.close()