# -*- coding: utf-8 -*-
"""
Glucose baseline training script.

Baselines:
1. Glucose
   - Features: HMDB0000122 only

2. Glucose+Covariates
   - Features: HMDB0000122 + Sex + BMI

Protocol:
1. Keep the original 80:20 train/test split unchanged.
2. Treat the original 80% train split as development set.
3. For each seed, split development set into subtrain/validation:
   - 87.5% of development for subtrain
   - 12.5% of development for validation
   Equivalent to approximately 70%/10%/20% train/validation/test.
4. Test set is never used for model selection or tuning.
5. Repeat across 5 seeds.
6. Save subject-level predictions for AUROC DeLong test and MAE Dunnett-type comparison.

Models:
- Age regression: Ridge regression
- DM classification: Logistic regression

Outputs:
- Per-seed metrics
- Mean ± SD summary
- Per-sample test predictions
"""

import os
import sys
import pickle
import datetime
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.dummy import DummyClassifier


# ============================================================
# 1. Global configuration
# ============================================================

BASE_SEED = 42
REPEAT_SEEDS = [42, 43, 44, 45, 46]

DATA_DIR = "../data/model_training_dataset"
SPLIT_DIR = os.path.join(DATA_DIR, "saved_splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

SAVED_MODELS_DIR = "../saved_models/glucose_baselines"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

LOGS_DIR = "../logs"
os.makedirs(LOGS_DIR, exist_ok=True)

RESULTS_DIR = "../results/glucose_baselines_5seeds"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_FILE = "MetKG_training_set_sample_1.txt"

GLUCOSE_COL = "HMDB0000122"
EXTRA_COLS = ["Sex", "BMI"]

TARGET_AGE_COL = "Age"
TARGET_DM_COL = "DM"

APPLY_LOG1P_TO_GLUCOSE = True

VAL_RATIO_WITHIN_DEV = 0.125

BASELINE_MODELS = [
    "Glucose",
    "Glucose+Covariates",
]


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


set_seed(BASE_SEED)


# ============================================================
# 3. Logging
# ============================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOGS_DIR, f"glucose_baselines_5seeds_{timestamp}.log")
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


print("========== Glucose baseline 5-seed training script ==========")
print(f"Log file: {log_file_path}")
print(f"Base seed: {BASE_SEED}")
print(f"Repeat seeds: {REPEAT_SEEDS}")
print(f"Baselines: {BASELINE_MODELS}")
print(f"Glucose column: {GLUCOSE_COL}")
print(f"Extra covariates: {EXTRA_COLS}")
print("=============================================================\n")


# ============================================================
# 4. Load data
# ============================================================

print("1. Loading data")

train_path = os.path.join(DATA_DIR, TRAIN_FILE)
df = pd.read_csv(train_path, sep="\t")
df = df.reset_index(drop=True)

print(f"Total samples: {len(df)}")

if GLUCOSE_COL not in df.columns:
    raise ValueError(
        f"Glucose column {GLUCOSE_COL} not found in the dataset. "
        f"Please check the table header."
    )

for col in EXTRA_COLS:
    if col not in df.columns:
        raise ValueError(f"Covariate column {col} not found in the dataset.")

if TARGET_AGE_COL not in df.columns:
    raise ValueError(f"Age column {TARGET_AGE_COL} not found.")

if TARGET_DM_COL not in df.columns:
    raise ValueError(f"DM column {TARGET_DM_COL} not found.")

print(f"Found glucose column: {GLUCOSE_COL}")
print(f"Found covariates: {EXTRA_COLS}")
print(f"Targets: {TARGET_AGE_COL}, {TARGET_DM_COL}")


# ============================================================
# 5. Process covariates
# ============================================================

print("\n2. Processing covariates")

for col in EXTRA_COLS:
    if df[col].dtype == object or not np.issubdtype(df[col].dtype, np.number):
        print(f"{col} is non-numeric. Applying LabelEncoder.")
        le = LabelEncoder()
        mask = df[col].notna()
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 6. log1p preprocessing for glucose
# ============================================================

if APPLY_LOG1P_TO_GLUCOSE:
    print(f"\n3. Applying log1p preprocessing to {GLUCOSE_COL}")
    vals = df[GLUCOSE_COL].to_numpy(dtype=np.float32)
    non_nan_mask = ~np.isnan(vals)
    non_neg_mask = (vals >= 0) & non_nan_mask
    vals[non_neg_mask] = np.log1p(vals[non_neg_mask])
    df[GLUCOSE_COL] = vals
else:
    print(f"\n3. Skipping log1p preprocessing for {GLUCOSE_COL}")


# ============================================================
# 7. DM label encoding
# ============================================================

print("\n4. Processing DM labels")

label_encoder = None

if df[TARGET_DM_COL].dtype == object or not np.issubdtype(df[TARGET_DM_COL].dtype, np.number):
    print("DM is non-numeric. Applying LabelEncoder.")
    label_encoder = LabelEncoder()
    df[TARGET_DM_COL] = label_encoder.fit_transform(df[TARGET_DM_COL].astype(str))

df[TARGET_DM_COL] = df[TARGET_DM_COL].astype(int)
num_dm_classes = int(df[TARGET_DM_COL].nunique())

print(f"Number of DM classes: {num_dm_classes}")

if num_dm_classes < 2:
    raise ValueError("DM classification requires at least two classes.")


# ============================================================
# 8. Load fixed 80:20 train/test split
# ============================================================

print("\n5. Loading fixed 80:20 train/test split")

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
print("The fixed test set will not be used for model selection or tuning.")


# ============================================================
# 9. Feature construction
# ============================================================

def get_feature_columns(model_name):
    if model_name == "Glucose":
        return [GLUCOSE_COL]
    elif model_name == "Glucose+Covariates":
        return [GLUCOSE_COL] + EXTRA_COLS
    else:
        raise ValueError(f"Unsupported glucose baseline: {model_name}")


def fit_feature_statistics(df, train_indices, feature_cols):
    """
    Fit mean/std using subtrain set only.
    Missing values are ignored.
    """
    X_train_raw = df.loc[train_indices, feature_cols].to_numpy(dtype=np.float32)

    means = np.nanmean(X_train_raw, axis=0)
    stds = np.nanstd(X_train_raw, axis=0)

    means[np.isnan(means)] = 0.0
    stds[np.isnan(stds)] = 1.0
    stds[stds == 0] = 1.0

    return means.astype(np.float32), stds.astype(np.float32)


def transform_features(df, indices, feature_cols, means, stds):
    """
    Feature representation:
    For each variable:
      - z-score value if observed, otherwise 0
      - missing mask, 1 if observed, 0 if missing

    Example:
      Glucose baseline:
        [glucose_z, glucose_mask]

      Glucose+Covariates:
        [glucose_z, glucose_mask, sex_z, sex_mask, bmi_z, bmi_mask]
    """
    raw = df.loc[indices, feature_cols].to_numpy(dtype=np.float32)

    mask = (~np.isnan(raw)).astype(np.float32)

    scaled = np.zeros_like(raw, dtype=np.float32)
    observed = ~np.isnan(raw)

    row_idx, col_idx = np.where(observed)
    scaled[row_idx, col_idx] = (
        raw[row_idx, col_idx] - means[col_idx]
    ) / stds[col_idx]

    feature_list = []

    for j in range(len(feature_cols)):
        feature_list.append(scaled[:, j])
        feature_list.append(mask[:, j])

    X = np.stack(feature_list, axis=1).astype(np.float32)

    return X


def build_features_and_targets(df, indices, feature_cols, means, stds):
    X = transform_features(
        df=df,
        indices=indices,
        feature_cols=feature_cols,
        means=means,
        stds=stds,
    )

    y_age = df.loc[indices, TARGET_AGE_COL].to_numpy(dtype=np.float32)
    y_dm = df.loc[indices, TARGET_DM_COL].to_numpy(dtype=np.int64)

    return X, y_age, y_dm


# ============================================================
# 10. Metrics and prediction utilities
# ============================================================

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


# ============================================================
# 11. Train one glucose baseline
# ============================================================

def train_one_glucose_baseline(
    model_name,
    seed,
    subtrain_idx,
    val_idx,
    test_idx,
):
    print("\n" + "-" * 80)
    print(f"Training {model_name} baseline | seed={seed}")
    print("-" * 80)

    set_seed(seed)

    feature_cols = get_feature_columns(model_name)

    print(f"Feature columns: {feature_cols}")

    means, stds = fit_feature_statistics(
        df=df,
        train_indices=subtrain_idx,
        feature_cols=feature_cols,
    )

    X_train, y_age_train, y_dm_train = build_features_and_targets(
        df=df,
        indices=subtrain_idx,
        feature_cols=feature_cols,
        means=means,
        stds=stds,
    )

    X_val, y_age_val, y_dm_val = build_features_and_targets(
        df=df,
        indices=val_idx,
        feature_cols=feature_cols,
        means=means,
        stds=stds,
    )

    X_test, y_age_test, y_dm_test = build_features_and_targets(
        df=df,
        indices=test_idx,
        feature_cols=feature_cols,
        means=means,
        stds=stds,
    )

    # Age regression
    age_model = Ridge(
        alpha=1.0,
        random_state=seed,
    )

    age_model.fit(X_train, y_age_train)

    # DM classification
    unique_train_classes = np.unique(y_dm_train)

    if len(unique_train_classes) < 2:
        print(
            f"Warning: only one DM class in subtrain for seed={seed}. "
            f"Using DummyClassifier."
        )
        dm_model = DummyClassifier(strategy="most_frequent")
        dm_model.fit(X_train, y_dm_train)
    else:
        dm_model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
            multi_class="auto",
        )
        dm_model.fit(X_train, y_dm_train)

    # Validation evaluation
    val_age_pred = age_model.predict(X_val)

    val_prob_raw = dm_model.predict_proba(X_val)
    val_dm_prob = proba_to_full_classes(
        val_prob_raw,
        dm_model.classes_,
        num_dm_classes,
    )

    val_metrics = compute_metrics_from_arrays(
        y_age_true=y_age_val,
        y_age_pred=val_age_pred,
        y_dm_true=y_dm_val,
        y_dm_prob=val_dm_prob,
        num_dm_classes=num_dm_classes,
    )

    # Test evaluation
    test_age_pred = age_model.predict(X_test)

    test_prob_raw = dm_model.predict_proba(X_test)
    test_dm_prob = proba_to_full_classes(
        test_prob_raw,
        dm_model.classes_,
        num_dm_classes,
    )

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

    safe_model_name = model_name.replace("+", "_plus_").replace(" ", "_")

    save_path = os.path.join(
        SAVED_MODELS_DIR,
        f"{safe_model_name}_seed{seed}_{timestamp}.pkl",
    )

    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "model_name": model_name,
                "seed": seed,
                "age_model": age_model,
                "dm_model": dm_model,
                "feature_cols": feature_cols,
                "means": means,
                "stds": stds,
                "apply_log1p_to_glucose": APPLY_LOG1P_TO_GLUCOSE,
                "glucose_col": GLUCOSE_COL,
                "extra_cols": EXTRA_COLS,
                "label_encoder": label_encoder,
                "num_dm_classes": num_dm_classes,
                "subtrain_idx": subtrain_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
            f,
        )

    print(f"[{model_name} seed={seed}] Validation metrics: {val_metrics}")
    print(f"[{model_name} seed={seed}] Test metrics: {test_metrics}")
    print(f"[{model_name} seed={seed}] Model saved to: {save_path}")

    metrics = {
        "model": model_name,
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
    }

    return metrics, pred_dict


# ============================================================
# 12. Main
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

            y_dev = df.loc[dev_idx, TARGET_DM_COL].values

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

        subtrain_idx = np.asarray(subtrain_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)

        print(f"Subtrain size: {len(subtrain_idx)}")
        print(f"Validation size: {len(val_idx)}")
        print(f"Fixed test size: {len(test_idx)}")

        for model_name in BASELINE_MODELS:

            metrics, pred_dict = train_one_glucose_baseline(
                model_name=model_name,
                seed=seed,
                subtrain_idx=subtrain_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )

            all_metrics.append(metrics)

            pred_df = pd.DataFrame(pred_dict)
            pred_df["model"] = model_name
            pred_df["seed"] = seed
            pred_df["glucose_col"] = GLUCOSE_COL
            pred_df["apply_log1p_to_glucose"] = APPLY_LOG1P_TO_GLUCOSE

            safe_model_name = model_name.replace("+", "_plus_").replace(" ", "_")

            pred_path = os.path.join(
                RESULTS_DIR,
                f"test_predictions_{safe_model_name}_seed{seed}_{timestamp}.csv",
            )

            pred_df.to_csv(pred_path, index=False)

            print(f"[{model_name} seed={seed}] Test predictions saved to: {pred_path}")

    # Save all per-seed metrics
    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = os.path.join(
        RESULTS_DIR,
        f"glucose_baseline_test_metrics_5seeds_{timestamp}.csv",
    )

    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 100)
    print("All glucose baseline per-seed test metrics")
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
        f"glucose_baseline_test_metrics_5seeds_mean_sd_{timestamp}.csv",
    )

    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("Glucose baseline mean ± SD over 5 seeds")
    print(summary_df)
    print(f"Saved to: {summary_path}")

    # Combine all prediction files generated in this run
    combined_pred_files = []

    for model_name in BASELINE_MODELS:
        safe_model_name = model_name.replace("+", "_plus_").replace(" ", "_")

        for seed in REPEAT_SEEDS:
            pred_path = os.path.join(
                RESULTS_DIR,
                f"test_predictions_{safe_model_name}_seed{seed}_{timestamp}.csv",
            )

            if os.path.exists(pred_path):
                combined_pred_files.append(pred_path)

    if len(combined_pred_files) > 0:
        pred_dfs = [pd.read_csv(p) for p in combined_pred_files]
        combined_pred_df = pd.concat(pred_dfs, axis=0, ignore_index=True)

        combined_pred_path = os.path.join(
            RESULTS_DIR,
            f"glucose_baseline_test_predictions_all_5seeds_{timestamp}.csv",
        )

        combined_pred_df.to_csv(combined_pred_path, index=False)

        print(f"\nCombined glucose baseline predictions saved to: {combined_pred_path}")

    print("\nFinished glucose baseline 5-seed training.")
    print("Use the saved prediction CSV files together with MetKG and other baseline predictions for:")
    print("1. AUROC DeLong test: dm_true + dm_prob_class_1")
    print("2. MAE Dunnett-type comparison: sample_idx + age_abs_error")
    print("3. Bar plots: mean ± SD from summary CSV")

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_f.close()