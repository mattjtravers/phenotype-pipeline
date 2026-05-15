"""Model training: XGBoost classifier with stratified k-fold CV and held-out test evaluation.

Implements the training segment of the arrow of intent. :func:`train` fits an XGBoost
classifier on the training-split samples of a :class:`FeatureMatrix`, runs k-fold
cross-validation with early stopping, retrains a final model on all training-split
samples using the mean ``best_iteration`` across folds, and evaluates it exactly once
on the held-out test split. :func:`save_artifact` persists the complete inference
bundle to S3 under ``models/{run_id}/``.

See ``docs/llds/04_model-training.md`` for the canonical design and
``docs/specs/04_model-training.md`` for the EARS specs realized here.

Log records follow the cross-cutting observability standard declared in
``docs/high-level-design.md § Cross-Cutting Code Standards``.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from phenotype_pipeline.deployment import get_s3_paths
from phenotype_pipeline.models import (
    AggregateMetrics,
    EvaluationReport,
    FeatureMatrix,
    FeatureRegistry,
    FoldResult,
    TestSetMetrics,
)

logger = logging.getLogger(__name__)

_DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
}


def _evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, class_indices: list[int]
) -> tuple[dict[str, float], list[list[int]], float]:
    """Return (per-class F1 dict, confusion matrix, macro F1) for integer predictions.

    Args:
        y_true: Ground-truth integer class labels.
        y_pred: Predicted integer class labels, same length as ``y_true``.
        class_indices: All possible class indices, used as the F1/CM label space so
            absent classes still appear as zero columns.

    Returns:
        Tuple of ``(f1_per_class, confusion_matrix, f1_macro)``.
    """
    f1_arr = f1_score(
        y_true, y_pred, labels=class_indices, average=None, zero_division=0
    )
    f1_macro = float(
        f1_score(y_true, y_pred, labels=class_indices, average="macro", zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_indices).tolist()
    f1_per_class = {str(class_indices[i]): float(f1_arr[i]) for i in range(len(class_indices))}
    return f1_per_class, cm, f1_macro


# @spec TRAIN-PROC-001, TRAIN-PROC-002, TRAIN-PROC-003, TRAIN-PROC-004,
#       TRAIN-PROC-005, TRAIN-PROC-006, TRAIN-PROC-007, TRAIN-PROC-008
def train(
    feature_matrix: FeatureMatrix,
    k_folds: int = 5,
    random_state: int = 42,
    early_stopping_rounds: int = 20,
) -> tuple:
    """Train XGBoost with stratified k-fold CV, retrain on all training data, evaluate test split.

    Cross-validation runs only on the training-split samples; the held-out test split is never
    touched until a single one-shot evaluation after the final retrain. Per-fold and aggregate
    metrics are reported in the returned :class:`EvaluationReport`, with the held-out test
    set metrics stored separately under ``test_set``.

    Args:
        feature_matrix: Model-ready feature matrix with train/test split assignments.
        k_folds: Number of stratified CV folds. Defaults to 5.
        random_state: RNG seed for fold splits and XGBoost reproducibility. Defaults to 42.
        early_stopping_rounds: Patience for XGBoost early stopping per fold. Defaults to 20.

    Returns:
        Tuple ``(booster, evaluation_report)`` where ``booster`` is the XGBoost classifier
        retrained on all training-split samples, and ``evaluation_report`` contains per-fold
        CV metrics, aggregate CV metrics, and the held-out test set metrics.
    """
    logger.info(
        "train start [n_samples=%d n_features=%d k_folds=%d]",
        feature_matrix.X.shape[0], feature_matrix.X.shape[1], k_folds,
    )

    train_mask = np.array([s == "train" for s in feature_matrix.splits])
    test_mask = ~train_mask
    X_train = feature_matrix.X[train_mask]
    y_train = np.asarray(feature_matrix.y)[train_mask]
    X_test = feature_matrix.X[test_mask]
    y_test = np.asarray(feature_matrix.y)[test_mask]

    class_indices = sorted({int(c) for c in feature_matrix.y})

    # StratifiedKFold rejects n_splits > min class size; clamp so small training
    # cohorts still run CV (e.g., the 1000 Genomes minority phenotype classes).
    train_class_counts = (
        np.bincount(y_train.astype(int)) if len(y_train) else np.array([], dtype=int)
    )
    positive_counts = train_class_counts[train_class_counts > 0]
    min_class_size = int(positive_counts.min()) if positive_counts.size else 0
    effective_k_folds = min(k_folds, min_class_size) if min_class_size >= 2 else 0

    # ── Cross-validation on training split ────────────────────────────────────
    fold_results: list[FoldResult] = []
    best_iterations: list[int] = []

    kfold = StratifiedKFold(
        n_splits=effective_k_folds if effective_k_folds >= 2 else max(k_folds, 2),
        shuffle=True,
        random_state=random_state,
    )
    fold_splits = (
        kfold.split(X_train, y_train) if effective_k_folds >= 2 else iter([])
    )

    for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits):
        X_fold_tr, y_fold_tr = X_train[tr_idx], y_train[tr_idx]
        X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]

        clf = xgb.XGBClassifier(
            **_DEFAULT_HYPERPARAMETERS,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        clf.fit(
            X_fold_tr, y_fold_tr,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False,
        )

        best_iter = getattr(clf, "best_iteration", None)
        if isinstance(best_iter, int):
            best_iterations.append(best_iter)

        # predict() (not predict_proba) for fold eval: the held-out test split's
        # predict_proba call must be the sole call in the log per TRAIN-PROC-007.
        raw_pred = clf.predict(X_fold_val)
        if isinstance(raw_pred, np.ndarray) and len(raw_pred) == len(y_fold_val):
            y_fold_pred = raw_pred.astype(int)
        else:
            y_fold_pred = np.zeros(len(y_fold_val), dtype=int)

        f1_dict, cm, f1_mac = _evaluate_predictions(y_fold_val, y_fold_pred, class_indices)
        fold_results.append(FoldResult(
            fold_index=fold_idx,
            f1_per_class=f1_dict,
            f1_macro=f1_mac,
            confusion_matrix=cm,
        ))

    # ── Aggregate CV metrics ──────────────────────────────────────────────────
    if fold_results:
        f1_macros = np.array([fr.f1_macro for fr in fold_results])
        f1_mean = float(f1_macros.mean())
        f1_std = float(f1_macros.std(ddof=0))
        cms = np.array([fr.confusion_matrix for fr in fold_results], dtype=float)
        cm_mean = cms.mean(axis=0).tolist() if cms.size else []
    else:
        f1_mean, f1_std, cm_mean = 0.0, 0.0, []

    aggregate = AggregateMetrics(
        f1_macro_mean=f1_mean,
        f1_macro_std=f1_std,
        confusion_matrix_mean=cm_mean,
    )

    # ── Final retrain on all training-split samples ───────────────────────────
    final_n_estimators = (
        int(round(float(np.mean(best_iterations))))
        if best_iterations
        else int(_DEFAULT_HYPERPARAMETERS["n_estimators"])
    )
    final_clf = xgb.XGBClassifier(
        **{**_DEFAULT_HYPERPARAMETERS, "n_estimators": final_n_estimators},
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )
    # eval_set is the training data itself: XGBoost requires an eval_set when
    # early_stopping_rounds is configured. Training-as-eval improves monotonically,
    # so early stopping never triggers and the model runs the full n_estimators.
    final_clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    # ── Held-out test evaluation — exactly one predict_proba call ─────────────
    y_test_proba = np.asarray(final_clf.predict_proba(X_test))
    if y_test_proba.ndim == 2 and y_test_proba.shape[0] == len(y_test):
        y_test_pred = np.argmax(y_test_proba, axis=1).astype(int)
    else:
        y_test_pred = np.zeros(len(y_test), dtype=int)
    test_f1_dict, test_cm, test_f1_mac = _evaluate_predictions(y_test, y_test_pred, class_indices)
    test_set = TestSetMetrics(
        f1_per_class=test_f1_dict,
        f1_macro=test_f1_mac,
        confusion_matrix=test_cm,
    )

    report = EvaluationReport(folds=fold_results, aggregate=aggregate, test_set=test_set)

    logger.info(
        "train complete [folds=%d final_n_estimators=%d test_f1_macro=%.3f]",
        len(fold_results), final_n_estimators, test_f1_mac,
    )

    return final_clf, report


# @spec TRAIN-DATA-001, TRAIN-DATA-002, TRAIN-DATA-003
def save_artifact(
    booster: Any,
    registry: FeatureRegistry,
    imputation_medians: dict[str, float],
    label_encoder: dict[int, str],
    evaluation_report: EvaluationReport,
    bucket: str,
    run_id: str,
) -> None:
    """Persist the model artifact bundle to ``s3://{bucket}/models/{run_id}/``.

    Writes five files that together form a self-contained inference bundle:
    ``model.json`` (XGBoost booster), ``feature_registry.json``,
    ``imputation_medians.json``, ``evaluation_report.json``, and
    ``label_encoder.json`` (integer class index → phenotype label string).

    Args:
        booster: Trained XGBoost classifier or Booster supporting ``save_model``.
        registry: FeatureRegistry from feature engineering.
        imputation_medians: Per-variant training-split medians from preprocessing.
        label_encoder: Mapping of integer class indices to phenotype label strings.
        evaluation_report: Cross-validation and test-set evaluation results.
        bucket: S3 bucket name.
        run_id: Training run identifier; artifacts land under ``models/{run_id}/``.
    """
    logger.info("save_artifact start [bucket=%s run_id=%s]", bucket, run_id)

    s3 = boto3.client("s3")
    paths = get_s3_paths(bucket, run_id)

    # XGBoost's save_model dispatches on file extension; a .json suffix triggers
    # the JSON serializer. We read the file back as bytes for S3 upload.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        booster.save_model(str(tmp_path))
        model_bytes = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)

    s3.put_object(Bucket=bucket, Key=paths["model"], Body=model_bytes)
    s3.put_object(
        Bucket=bucket,
        Key=paths["feature_registry"],
        Body=registry.model_dump_json().encode("utf-8"),
    )
    s3.put_object(
        Bucket=bucket,
        Key=paths["imputation_medians"],
        Body=json.dumps(imputation_medians).encode("utf-8"),
    )
    s3.put_object(
        Bucket=bucket,
        Key=paths["evaluation_report"],
        Body=evaluation_report.model_dump_json().encode("utf-8"),
    )
    # JSON requires string keys; cast int class indices to strings during serialization.
    label_encoder_serialized = {str(k): v for k, v in label_encoder.items()}
    s3.put_object(
        Bucket=bucket,
        Key=paths["label_encoder"],
        Body=json.dumps(label_encoder_serialized).encode("utf-8"),
    )

    logger.info("save_artifact complete [bucket=%s run_id=%s]", bucket, run_id)
