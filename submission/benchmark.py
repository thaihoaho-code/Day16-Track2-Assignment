import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

seed = 16

# Threshold candidates for fraud detection.
# Lower threshold -> generally higher recall, lower precision.
thresholds = np.arange(0.05, 0.51, 0.01)

# Benchmark repeats
latency_repeats = 50
batch_repeats = 30


# ============================================================
# LOAD DATA
# ============================================================

started = time.perf_counter()

df = pd.read_csv("creditcard.csv")

data_load_seconds = time.perf_counter() - started

X = df.drop(columns="Class")
y = df["Class"]


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# 60% train / 20% validation / 20% test
# Stratified to preserve Class distribution.
# ============================================================

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=seed,
    stratify=y,
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=seed,
    stratify=y_trainval,
)


# ============================================================
# MODEL
# ============================================================

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=seed,
    n_jobs=2,
    verbosity=-1,
)


# ============================================================
# TRAIN
# ============================================================

started = time.perf_counter()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(
            20,
            verbose=False,
        )
    ],
)

training_seconds = time.perf_counter() - started


# ============================================================
# VALIDATION
# Select threshold based ONLY on validation set.
#
# Here we maximize F1 while checking recall.
# The test set remains untouched until final evaluation.
# ============================================================

valid_probabilities = model.predict_proba(X_valid)[:, 1]

threshold_results = []

for threshold in thresholds:
    valid_predictions = (
        valid_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        valid_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_valid,
        valid_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_valid,
        valid_predictions,
        zero_division=0,
    )

    threshold_results.append(
        {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    )


# Maximize F1 on validation set.
best_threshold_result = max(
    threshold_results,
    key=lambda x: x["f1"],
)

decision_threshold = best_threshold_result["threshold"]


# ============================================================
# FINAL TEST PREDICTION
# ============================================================

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= decision_threshold
).astype(int)


# ============================================================
# TEST METRICS
# ============================================================

auc_roc = roc_auc_score(
    y_test,
    probabilities,
)

accuracy = accuracy_score(
    y_test,
    predictions,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)


# ============================================================
# INFERENCE BENCHMARK
# ============================================================

one_row = X_test.iloc[:1]
batch = X_test.iloc[:1000]


# Warm-up outside measurement
model.predict_proba(one_row)
model.predict_proba(batch)


def measured_seconds(data, repeats):
    elapsed = []

    for _ in range(repeats):
        started = time.perf_counter()

        model.predict_proba(data)

        elapsed.append(
            time.perf_counter() - started
        )

    return float(np.median(elapsed))


single_seconds = measured_seconds(
    one_row,
    latency_repeats,
)

batch_seconds = measured_seconds(
    batch,
    batch_repeats,
)


# ============================================================
# RESULT
# ============================================================

result = {
    "recorded_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "architecture": platform.machine(),

    "versions": {
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },

    # Dataset
    "dataset_rows": len(df),
    "fraud_rows": int(y.sum()),

    # Reproducibility
    "seed": seed,

    # Split
    "split": {
        "train": len(X_train),
        "validation": len(X_valid),
        "test": len(X_test),
    },

    # Model
    "n_jobs": 2,
    "best_iteration": int(
        model.best_iteration_
    ),

    # Decision
    "decision_threshold": decision_threshold,

    # Validation threshold selection
    "threshold_selection": {
        "method": "maximize_f1_on_validation",
        "validation_precision": best_threshold_result[
            "precision"
        ],
        "validation_recall": best_threshold_result[
            "recall"
        ],
        "validation_f1": best_threshold_result[
            "f1"
        ],
    },

    # Test metrics
    "auc_roc": float(auc_roc),
    "accuracy": float(accuracy),
    "f1": float(f1),
    "precision": float(precision),
    "recall": float(recall),

    # Runtime
    "data_load_seconds": data_load_seconds,
    "training_seconds": training_seconds,

    # Single-row latency
    "latency_1_row_ms": (
        single_seconds * 1000
    ),
    "latency_repeats": latency_repeats,

    # Batch latency
    "batch_rows": len(batch),
    "batch_repeats": batch_repeats,
    "batch_1000_rows_seconds": batch_seconds,

    "throughput_1000_rows_per_second": (
        len(batch) / batch_seconds
    ),

    "timing_summary": (
        "median; warm-up excluded; "
        "predict_proba on pandas input"
    ),
}


# ============================================================
# SAVE RESULT
# ============================================================

Path(
    "benchmark_result.json"
).write_text(
    json.dumps(
        result,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print(
    json.dumps(
        result,
        indent=2,
        allow_nan=False,
    )
)
