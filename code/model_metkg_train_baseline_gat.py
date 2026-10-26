# -*- coding: utf-8 -*-
"""
Baseline training script for:
- RF
- MLP
- GCN
- GAT
- GraphSAGE

Designed to match the updated MetKG internal evaluation protocol:
1. Keep the original 80:20 train/test split unchanged.
2. Treat the original 80% train split as development set.
3. For each seed, split development set into subtrain/validation:
   - 87.5% of development for subtrain
   - 12.5% of development for validation
   Equivalent to approximately 70%/10%/20% train/validation/test.
4. Test set is never used for early stopping, hyperparameter tuning, or model selection.
5. Repeat across 5 seeds.
6. Save subject-level predictions for AUROC DeLong test and MAE Dunnett-type comparison.

Output:
- Per-seed metrics for each baseline
- Mean ± SD summary
- Per-sample test predictions for each baseline and seed
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier


# ============================================================
# 1. Global configuration
# ============================================================

BASE_SEED = 42
REPEAT_SEEDS = [42, 43, 44, 45, 46]

DATA_DIR = "../data/model_training_dataset"
SPLIT_DIR = os.path.join(DATA_DIR, "saved_splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

SAVED_MODELS_DIR = "../saved_models/baselines"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

LOGS_DIR = "../logs"
os.makedirs(LOGS_DIR, exist_ok=True)

RESULTS_DIR = "../results/baselines_5seeds"
os.makedirs(RESULTS_DIR, exist_ok=True)

APPLY_LOG1P = True
ADJ_THRESHOLD = 0.5
VAL_RATIO_WITHIN_DEV = 0.125

# Torch model hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 2
MAX_EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 25
AGE_LOSS_WEIGHT = 0.025
DM_LOSS_WEIGHT = 1.0

# Baselines to run
BASELINE_MODELS = ["GAT"]


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
log_file_path = os.path.join(LOGS_DIR, f"baseline_5seeds_{timestamp}.log")
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


print("========== Baseline 5-seed training script ==========")
print(f"Log file: {log_file_path}")
print(f"Base seed: {BASE_SEED}")
print(f"Repeat seeds: {REPEAT_SEEDS}")
print(f"Baselines: {BASELINE_MODELS}")
print("=====================================================\n")


# ============================================================
# 4. Device
# ============================================================

if torch.cuda.is_available():
    preferred_gpu = 1
    if torch.cuda.device_count() > preferred_gpu:
        device = torch.device(f"cuda:{preferred_gpu}")
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}\n")


# ============================================================
# 5. Load data
# ============================================================

print("1. Loading data")

train_path = os.path.join(DATA_DIR, "MetKG_training_set_sample_1.txt")
df = pd.read_csv(train_path, sep="\t")
df = df.reset_index(drop=True)

hmdb_cols = [c for c in df.columns if c.startswith("HMDB")]
extra_cols = ["Sex", "BMI"]
target_age_col = "Age"
target_dm_col = "DM"

print(f"Total samples: {len(df)}")
print(f"HMDB columns: {len(hmdb_cols)}")
print(f"Extra columns: {extra_cols}")
print(f"Targets: {target_age_col}, {target_dm_col}")


# ============================================================
# 6. Process extra covariates
# ============================================================

print("\n2. Processing extra covariates")

for col in extra_cols:
    if col not in df.columns:
        raise ValueError(f"Missing extra column: {col}")

    if df[col].dtype == object or not np.issubdtype(df[col].dtype, np.number):
        print(f"{col} is non-numeric. Applying LabelEncoder.")
        le = LabelEncoder()
        mask = df[col].notna()
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 7. Load graph structure
# ============================================================

print("\n3. Loading graph structure")

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

full_adj_bin = (mat_matrix >= ADJ_THRESHOLD).astype(np.int8)

neighbor_mask = full_adj_bin[measured_idx].any(axis=0)
all_idx = np.where(neighbor_mask)[0]
sub_idx = np.union1d(all_idx, measured_idx)

sub_adj_bin = full_adj_bin[np.ix_(sub_idx, sub_idx)]
model_hmdb = [nodes_list[i] for i in sub_idx]

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

print(f"Expanded graph nodes: {num_nodes}")
print(f"Expanded graph edges: {num_edges}")


# ============================================================
# 8. log1p preprocessing
# ============================================================

if APPLY_LOG1P:
    print("\n4. Applying log1p preprocessing")
    vals = df[common_hmdb].to_numpy(dtype=np.float32)
    nan_mask = np.isnan(vals)
    non_nan_mask = ~nan_mask
    non_neg_mask = (vals >= 0) & non_nan_mask
    vals[non_neg_mask] = np.log1p(vals[non_neg_mask])
    df[common_hmdb] = vals
else:
    print("\n4. Skipping log1p preprocessing")


# ============================================================
# 9. DM label encoding
# ============================================================

print("\n5. Processing DM labels")

label_encoder = None

if df[target_dm_col].dtype == object or not np.issubdtype(df[target_dm_col].dtype, np.number):
    print("DM is non-numeric. Applying LabelEncoder.")
    label_encoder = LabelEncoder()
    df[target_dm_col] = label_encoder.fit_transform(df[target_dm_col].astype(str))

df[target_dm_col] = df[target_dm_col].astype(int)
num_dm_classes = int(df[target_dm_col].nunique())

print(f"Number of DM classes: {num_dm_classes}")


# ============================================================
# 10. Load fixed 80:20 train/test split
# ============================================================

print("\n6. Loading fixed 80:20 train/test split")

train_idx_path = os.path.join(SPLIT_DIR, "train_idx.npy")
test_idx_path = os.path.join(SPLIT_DIR, "test_idx.npy")

if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
    dev_idx = np.load(train_idx_path)
    test_idx = np.load(test_idx_path)
    print("Loaded existing train_idx.npy and test_idx.npy.")
else:
    print("No existing split files found. Creating fixed 80:20 split.")
    all_indices = np.arange(len(df))
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

print(f"Development set size: {len(dev_idx)}")
print(f"Fixed test set size: {len(test_idx)}")
print("The fixed test set will not be used for early stopping or tuning.")


# ============================================================
# 11. Feature statistics
# ============================================================

CONC_MASK_DIM = 3


def fit_feature_statistics(df, train_indices, common_hmdb, extra_cols):
    """
    Fit normalization statistics using subtrain set only.
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
    """
    Return node-level features:
    X_node: [n_samples, num_nodes, 3]
    """
    raw = df.loc[indices, common_hmdb].to_numpy(dtype=np.float32)

    mask = (~np.isnan(raw)).astype(np.float32)

    scaled = np.zeros_like(raw, dtype=np.float32)
    observed = ~np.isnan(raw)
    scaled[observed] = (raw[observed] - np.take(node_means, np.where(observed)[1])) / np.take(node_stds, np.where(observed)[1])

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
    """
    Return extra features:
    X_extra: [n_samples, 2 * len(extra_cols)]
    """
    raw = df.loc[indices, extra_cols].to_numpy(dtype=np.float32)

    mask = (~np.isnan(raw)).astype(np.float32)

    scaled = np.zeros_like(raw, dtype=np.float32)
    observed = ~np.isnan(raw)
    scaled[observed] = (raw[observed] - np.take(extra_means, np.where(observed)[1])) / np.take(extra_stds, np.where(observed)[1])

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
    """
    For RF and MLP baselines:
    Flatten node features [num_nodes, 3] to [num_nodes * 3],
    then concatenate extra features.
    """
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
# 12. Graph Dataset
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

        x = np.stack([scaled_vals, within_vals, mask_vals], axis=1).astype(np.float32)

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


# ============================================================
# 13. Torch baseline models
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
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
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
# 14. Metric helpers
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
    """
    Ensure predict_proba output has shape [n_samples, num_classes].
    Useful if some class is absent in a training split.
    """
    n = proba.shape[0]
    full = np.zeros((n, num_classes), dtype=np.float32)

    for j, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, j]

    return full


# ============================================================
# 15. Evaluate torch models
# ============================================================

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
# 16. Train RF baseline
# ============================================================

def train_rf_baseline(seed, subtrain_idx, val_idx, test_idx):
    print("\n" + "-" * 80)
    print(f"Training RF baseline | seed={seed}")
    print("-" * 80)

    set_seed(seed)

    node_means, node_stds, extra_means, extra_stds = fit_feature_statistics(
        df=df,
        train_indices=subtrain_idx,
        common_hmdb=common_hmdb,
        extra_cols=extra_cols,
    )

    X_train, y_age_train, y_dm_train = build_tabular_features(
        df, subtrain_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    X_val, y_age_val, y_dm_val = build_tabular_features(
        df, val_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    X_test, y_age_test, y_dm_test = build_tabular_features(
        df, test_idx, common_hmdb, node_means, node_stds,
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

    val_age_pred = rf_reg.predict(X_val)
    val_prob_raw = rf_clf.predict_proba(X_val)
    val_dm_prob = proba_to_full_classes(val_prob_raw, rf_clf.classes_, num_dm_classes)

    val_metrics = compute_metrics_from_arrays(
        y_age_true=y_age_val,
        y_age_pred=val_age_pred,
        y_dm_true=y_dm_val,
        y_dm_prob=val_dm_prob,
        num_dm_classes=num_dm_classes,
    )

    test_age_pred = rf_reg.predict(X_test)
    test_prob_raw = rf_clf.predict_proba(X_test)
    test_dm_prob = proba_to_full_classes(test_prob_raw, rf_clf.classes_, num_dm_classes)

    test_metrics = compute_metrics_from_arrays(
        y_age_true=y_age_test,
        y_age_pred=test_age_pred,
        y_dm_true=y_dm_test,
        y_dm_prob=test_dm_prob,
        num_dm_classes=num_dm_classes,
    )

    pred_dict = make_prediction_dict(
        sample_idx=test_idx,
        y_age_true=y_age_test,
        y_age_pred=test_age_pred,
        y_dm_true=y_dm_test,
        y_dm_prob=test_dm_prob,
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
                "subtrain_idx": subtrain_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
            f,
        )

    print(f"[RF seed={seed}] Validation metrics: {val_metrics}")
    print(f"[RF seed={seed}] Test metrics: {test_metrics}")
    print(f"[RF seed={seed}] Model saved to: {save_path}")

    return {
        "model": "RF",
        "seed": seed,
        "best_epoch": np.nan,
        "val_age_mae": val_metrics["age_mae"],
        "val_age_rmse": val_metrics["age_rmse"],
        "val_age_r2": val_metrics["age_r2"],
        "val_dm_acc": val_metrics["dm_acc"],
        "val_dm_auc": val_metrics["dm_auc"],
        "test_age_mae": test_metrics["age_mae"],
        "test_age_rmse": test_metrics["age_rmse"],
        "test_age_r2": test_metrics["age_r2"],
        "test_dm_acc": test_metrics["dm_acc"],
        "test_dm_auc": test_metrics["dm_auc"],
    }, pred_dict


# ============================================================
# 17. Train MLP baseline
# ============================================================

def train_mlp_baseline(seed, subtrain_idx, val_idx, test_idx):
    print("\n" + "-" * 80)
    print(f"Training MLP baseline | seed={seed}")
    print("-" * 80)

    set_seed(seed)

    node_means, node_stds, extra_means, extra_stds = fit_feature_statistics(
        df=df,
        train_indices=subtrain_idx,
        common_hmdb=common_hmdb,
        extra_cols=extra_cols,
    )

    X_train, y_age_train, y_dm_train = build_tabular_features(
        df, subtrain_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    X_val, y_age_val, y_dm_val = build_tabular_features(
        df, val_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    X_test, y_age_test, y_dm_test = build_tabular_features(
        df, test_idx, common_hmdb, node_means, node_stds,
        extra_cols, extra_means, extra_stds,
    )

    train_dataset = TabularTorchDataset(X_train, y_age_train, y_dm_train, subtrain_idx)
    val_dataset = TabularTorchDataset(X_val, y_age_val, y_dm_val, val_idx)
    test_dataset = TabularTorchDataset(X_test, y_age_test, y_dm_test, test_idx)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
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

    best_val_score = float("inf")
    best_state = None
    best_epoch = None
    best_val_metrics = None
    patience_cnt = 0

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

        train_metrics = evaluate_torch_mlp(model, train_loader, num_dm_classes)
        val_metrics = evaluate_torch_mlp(model, val_loader, num_dm_classes)

        val_auc_for_score = val_metrics["dm_auc"] if not np.isnan(val_metrics["dm_auc"]) else 0.0
        val_score = val_metrics["age_mae"] + (1.0 - val_auc_for_score)

        print(
            f"[MLP seed={seed}] Epoch {epoch:03d} | "
            f"Loss={avg_loss:.4f} | "
            f"Train MAE={train_metrics['age_mae']:.4f}, AUC={train_metrics['dm_auc']:.4f} | "
            f"Val MAE={val_metrics['age_mae']:.4f}, AUC={val_metrics['dm_auc']:.4f}, "
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
                print(f"[MLP seed={seed}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, pred_dict = evaluate_torch_mlp(
        model,
        test_loader,
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
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "node_means": node_means,
            "node_stds": node_stds,
            "extra_means": extra_means,
            "extra_stds": extra_stds,
            "common_hmdb": common_hmdb,
            "extra_cols": extra_cols,
            "apply_log1p": APPLY_LOG1P,
            "subtrain_idx": subtrain_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        },
        save_path,
    )

    print(f"[MLP seed={seed}] Best epoch: {best_epoch}")
    print(f"[MLP seed={seed}] Test metrics: {test_metrics}")
    print(f"[MLP seed={seed}] Model saved to: {save_path}")

    return {
        "model": "MLP",
        "seed": seed,
        "best_epoch": best_epoch,
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
    }, pred_dict


# ============================================================
# 18. Train GNN baseline: GCN / GAT / GraphSAGE
# ============================================================

def train_gnn_baseline(seed, subtrain_idx, val_idx, test_idx, baseline_name):
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

    best_val_score = float("inf")
    best_state = None
    best_epoch = None
    best_val_metrics = None
    patience_cnt = 0

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

        train_metrics = evaluate_torch_gnn(model, train_loader, num_dm_classes)
        val_metrics = evaluate_torch_gnn(model, val_loader, num_dm_classes)

        val_auc_for_score = val_metrics["dm_auc"] if not np.isnan(val_metrics["dm_auc"]) else 0.0
        val_score = val_metrics["age_mae"] + (1.0 - val_auc_for_score)

        print(
            f"[{baseline_name} seed={seed}] Epoch {epoch:03d} | "
            f"Loss={avg_loss:.4f} | "
            f"Train MAE={train_metrics['age_mae']:.4f}, AUC={train_metrics['dm_auc']:.4f} | "
            f"Val MAE={val_metrics['age_mae']:.4f}, AUC={val_metrics['dm_auc']:.4f}, "
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
                print(f"[{baseline_name} seed={seed}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, pred_dict = evaluate_torch_gnn(
        model,
        test_loader,
        num_dm_classes,
        return_predictions=True,
    )

    save_path = os.path.join(SAVED_MODELS_DIR, f"{baseline_name}_seed{seed}_{timestamp}.pt")

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
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "node_means": node_means,
            "node_stds": node_stds,
            "extra_means": extra_means,
            "extra_stds": extra_stds,
            "common_hmdb": common_hmdb,
            "adj_with_self": adj_with_self,
            "edge_index": edge_index,
            "apply_log1p": APPLY_LOG1P,
            "subtrain_idx": subtrain_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        },
        save_path,
    )

    print(f"[{baseline_name} seed={seed}] Best epoch: {best_epoch}")
    print(f"[{baseline_name} seed={seed}] Test metrics: {test_metrics}")
    print(f"[{baseline_name} seed={seed}] Model saved to: {save_path}")

    return {
        "model": baseline_name,
        "seed": seed,
        "best_epoch": best_epoch,
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
    }, pred_dict


# ============================================================
# 19. Main
# ============================================================

if __name__ == "__main__":

    all_metrics = []

    for seed in REPEAT_SEEDS:
        print("\n" + "=" * 100)
        print(f"Preparing split for seed={seed}")
        print("=" * 100)

        set_seed(seed)

        subtrain_path = os.path.join(SPLIT_DIR, f"subtrain_idx_seed{seed}.npy")
        val_path = os.path.join(SPLIT_DIR, f"val_idx_seed{seed}.npy")

        if os.path.exists(subtrain_path) and os.path.exists(val_path):
            print(f"Loading existing subtrain/validation split for seed={seed}")
            subtrain_idx = np.load(subtrain_path)
            val_idx = np.load(val_path)
        else:
            print(f"Creating subtrain/validation split for seed={seed}")

            y_dev = df.loc[dev_idx, target_dm_col].values

            try:
                subtrain_idx, val_idx = train_test_split(
                    dev_idx,
                    test_size=VAL_RATIO_WITHIN_DEV,
                    random_state=seed,
                    shuffle=True,
                    stratify=y_dev,
                )
            except ValueError as e:
                print(f"[Seed {seed}] Stratified split failed: {e}")
                print(f"[Seed {seed}] Falling back to non-stratified split.")
                subtrain_idx, val_idx = train_test_split(
                    dev_idx,
                    test_size=VAL_RATIO_WITHIN_DEV,
                    random_state=seed,
                    shuffle=True,
                )

            subtrain_idx = np.asarray(subtrain_idx, dtype=np.int64)
            val_idx = np.asarray(val_idx, dtype=np.int64)

            np.save(subtrain_path, subtrain_idx)
            np.save(val_path, val_idx)

        print(f"Subtrain size: {len(subtrain_idx)}")
        print(f"Validation size: {len(val_idx)}")
        print(f"Fixed test size: {len(test_idx)}")

        for baseline_name in BASELINE_MODELS:

            if baseline_name == "RF":
                metrics, pred_dict = train_rf_baseline(
                    seed=seed,
                    subtrain_idx=subtrain_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                )

            elif baseline_name == "MLP":
                metrics, pred_dict = train_mlp_baseline(
                    seed=seed,
                    subtrain_idx=subtrain_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                )

            elif baseline_name in ["GAT", "GraphSAGE", "GCN"]:
                metrics, pred_dict = train_gnn_baseline(
                    seed=seed,
                    subtrain_idx=subtrain_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    baseline_name=baseline_name,
                )

            else:
                raise ValueError(f"Unsupported baseline: {baseline_name}")

            all_metrics.append(metrics)

            pred_df = pd.DataFrame(pred_dict)
            pred_df["model"] = baseline_name
            pred_df["seed"] = seed

            pred_path = os.path.join(
                RESULTS_DIR,
                f"test_predictions_{baseline_name}_seed{seed}_{timestamp}.csv",
            )
            pred_df.to_csv(pred_path, index=False)

            print(f"[{baseline_name} seed={seed}] Test predictions saved to: {pred_path}")

    # Save all per-seed metrics
    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = os.path.join(
        RESULTS_DIR,
        f"baseline_test_metrics_5seeds_{timestamp}.csv",
    )
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 100)
    print("All baseline per-seed test metrics")
    print(metrics_df)
    print(f"Saved to: {metrics_path}")

    # Summary mean ± SD
    summary_rows = []

    metric_cols = [
        "test_age_mae",
        "test_age_rmse",
        "test_age_r2",
        "test_dm_acc",
        "test_dm_auc",
    ]

    for model_name in BASELINE_MODELS:
        sub_df = metrics_df[metrics_df["model"] == model_name]

        for metric in metric_cols:
            vals = sub_df[metric].astype(float).values

            summary_rows.append({
                "model": model_name,
                "metric": metric,
                "mean": np.nanmean(vals),
                "sd": np.nanstd(vals, ddof=1),
                "n": np.sum(~np.isnan(vals)),
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        RESULTS_DIR,
        f"baseline_test_metrics_5seeds_mean_sd_{timestamp}.csv",
    )
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("Baseline mean ± SD over 5 seeds")
    print(summary_df)
    print(f"Saved to: {summary_path}")

    # Also combine all prediction files generated in this run
    combined_pred_files = []

    for model_name in BASELINE_MODELS:
        for seed in REPEAT_SEEDS:
            pred_path = os.path.join(
                RESULTS_DIR,
                f"test_predictions_{model_name}_seed{seed}_{timestamp}.csv",
            )
            if os.path.exists(pred_path):
                combined_pred_files.append(pred_path)

    if len(combined_pred_files) > 0:
        pred_dfs = [pd.read_csv(p) for p in combined_pred_files]
        combined_pred_df = pd.concat(pred_dfs, axis=0, ignore_index=True)

        combined_pred_path = os.path.join(
            RESULTS_DIR,
            f"baseline_test_predictions_all_5seeds_{timestamp}.csv",
        )
        combined_pred_df.to_csv(combined_pred_path, index=False)

        print(f"\nCombined baseline predictions saved to: {combined_pred_path}")

    print("\nFinished baseline 5-seed training.")
    print("Use the saved prediction CSV files together with MetKG predictions for:")
    print("1. AUROC DeLong test: dm_true + dm_prob_class_1")
    print("2. MAE Dunnett-type comparison: sample_idx + age_abs_error")
    print("3. Bar plots: mean ± SD from summary CSV")

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_f.close()