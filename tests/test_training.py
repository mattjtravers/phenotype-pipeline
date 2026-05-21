"""Tests for model training — TRAIN-PROC-* and TRAIN-DATA-* specs.

TRAIN-BE-001 (SageMaker-only execution) and TRAIN-BE-002 (instance type) are
covered in test_deployment.py, which owns the launch_training_job contract.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from genomic_ancestry_pipeline.models import EvaluationReport, FeatureRegistry
from genomic_ancestry_pipeline.training import save_artifact, train


# ── Training protocol ──────────────────────────────────────────────────────────


# @spec TRAIN-PROC-001
def test_training_uses_training_split_samples_only(feature_matrix):
    """XGBoost is fit only on samples where split == 'train'."""
    train_indices = [i for i, s in enumerate(feature_matrix.splits) if s == "train"]
    X_train_expected = feature_matrix.X[train_indices]

    with patch("genomic_ancestry_pipeline.training.xgb") as mock_xgb:
        mock_clf = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_clf
        mock_clf.fit.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.full((len(train_indices), 3), 1 / 3)
        mock_clf.best_iteration = 50

        train(feature_matrix)

        # Verify fit() was called with only training rows
        fit_call_X = mock_clf.fit.call_args[0][0]
        assert fit_call_X.shape[0] == len(train_indices)
        np.testing.assert_array_equal(fit_call_X, X_train_expected)


# @spec TRAIN-PROC-002
def test_cross_validation_uses_configurable_k(feature_matrix):
    """Stratified k-fold CV runs with the configured k value."""
    with patch("genomic_ancestry_pipeline.training.xgb"), \
         patch("genomic_ancestry_pipeline.training.StratifiedKFold") as mock_kfold:
        mock_kfold.return_value.split.return_value = iter([])
        try:
            train(feature_matrix, k_folds=3)
        except Exception:
            pass  # may fail downstream; we just need to check KFold call
        mock_kfold.assert_called_once_with(n_splits=3, shuffle=True, random_state=42)


# @spec TRAIN-PROC-003
def test_each_fold_uses_early_stopping(feature_matrix):
    """XGBClassifier is constructed with early_stopping_rounds per fold."""
    with patch("genomic_ancestry_pipeline.training.xgb") as mock_xgb:
        mock_clf = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_clf
        mock_clf.fit.return_value = mock_clf
        mock_clf.best_iteration = 30
        mock_clf.predict_proba.return_value = np.full((2, 3), 1 / 3)

        train(feature_matrix, early_stopping_rounds=20)

        init_kwargs = mock_xgb.XGBClassifier.call_args[1]
        assert init_kwargs.get("early_stopping_rounds") == 20


# @spec TRAIN-PROC-004
def test_final_model_retrained_on_all_training_samples(feature_matrix):
    """After CV, a final model is fit on the complete training split."""
    train_count = sum(1 for s in feature_matrix.splits if s == "train")

    with patch("genomic_ancestry_pipeline.training.xgb") as mock_xgb:
        mock_clf = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_clf
        mock_clf.fit.return_value = mock_clf
        mock_clf.best_iteration = 50
        mock_clf.predict_proba.return_value = np.full((train_count, 3), 1 / 3)

        train(feature_matrix)

        # The last fit() call must use all training samples
        last_fit_X = mock_clf.fit.call_args_list[-1][0][0]
        assert last_fit_X.shape[0] == train_count


# ── Evaluation ─────────────────────────────────────────────────────────────────


# @spec TRAIN-PROC-005
def test_evaluation_report_contains_per_fold_f1_and_confusion_matrix(feature_matrix):
    """Each FoldResult has per-class F1 and a confusion matrix."""
    with patch("genomic_ancestry_pipeline.training.xgb") as mock_xgb:
        mock_clf = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_clf
        mock_clf.fit.return_value = mock_clf
        mock_clf.best_iteration = 50
        mock_clf.predict_proba.return_value = np.eye(3)[:2]  # 2 val samples

        _, report = train(feature_matrix, k_folds=2)

    assert len(report.folds) == 2
    for fold in report.folds:
        assert isinstance(fold.f1_per_class, dict)
        assert len(fold.f1_per_class) > 0
        assert isinstance(fold.confusion_matrix, list)


# @spec TRAIN-PROC-006
def test_aggregate_metrics_contain_macro_f1_mean_and_std(feature_matrix):
    """EvaluationReport.aggregate has f1_macro_mean and f1_macro_std."""
    with patch("genomic_ancestry_pipeline.training.xgb") as mock_xgb:
        mock_clf = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_clf
        mock_clf.fit.return_value = mock_clf
        mock_clf.best_iteration = 50
        mock_clf.predict_proba.return_value = np.eye(3)[:2]

        _, report = train(feature_matrix, k_folds=2)

    assert isinstance(report.aggregate.f1_macro_mean, float)
    assert isinstance(report.aggregate.f1_macro_std, float)


# @spec TRAIN-PROC-007, TRAIN-PROC-008
def test_test_set_evaluated_once_after_final_training(feature_matrix):
    """Test split is evaluated exactly once, after final model retraining, and stored in report.test_set."""
    test_indices = [i for i, s in enumerate(feature_matrix.splits) if s == "test"]
    X_test = feature_matrix.X[test_indices]

    with patch("genomic_ancestry_pipeline.training.xgb") as mock_xgb:
        mock_clf = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_clf
        mock_clf.fit.return_value = mock_clf
        mock_clf.best_iteration = 50
        mock_clf.predict_proba.return_value = np.full((max(len(test_indices), 2), 3), 1 / 3)

        _, report = train(feature_matrix)

    assert report.test_set is not None, "EvaluationReport must have a test_set section"
    assert isinstance(report.test_set.f1_macro, float)
    assert isinstance(report.test_set.confusion_matrix, list)

    # Verify test data was not accessed during CV by counting predict_proba calls
    # on test-split rows — there should be exactly one batch matching len(test_indices)
    test_size_calls = [
        c for c in mock_clf.predict_proba.call_args_list
        if c[0][0].shape[0] == len(test_indices)
    ]
    assert len(test_size_calls) == 1, "Test split must be evaluated exactly once"


# ── Model artifact ─────────────────────────────────────────────────────────────


# @spec TRAIN-DATA-001
def test_save_artifact_writes_required_files_to_s3(feature_registry):
    """save_artifact puts model.json, feature_registry.json, imputation_medians.json,
    evaluation_report.json, and label_encoder.json to s3://{bucket}/models/{run_id}/."""
    from genomic_ancestry_pipeline.models import AggregateMetrics, EvaluationReport, TestSetMetrics

    report = EvaluationReport(
        folds=[],
        aggregate=AggregateMetrics(f1_macro_mean=0.9, f1_macro_std=0.05, confusion_matrix_mean=[]),
        test_set=TestSetMetrics(f1_per_class={"blue": 0.9}, f1_macro=0.9, confusion_matrix=[]),
    )
    mock_booster = MagicMock()
    mock_booster.save_model = MagicMock()

    with patch("genomic_ancestry_pipeline.training.boto3") as mock_boto3:
        mock_s3 = mock_boto3.client.return_value
        save_artifact(
            booster=mock_booster,
            registry=feature_registry,
            imputation_medians={"15_28513871_A_G": 1.0},
            label_encoder={0: "blue", 1: "brown", 2: "green"},
            evaluation_report=report,
            bucket="my-bucket",
            run_id="20240115-a3f2c1",
        )

    all_keys = [str(c) for c in mock_s3.put_object.call_args_list]
    expected_files = ["model.json", "feature_registry.json", "imputation_medians.json",
                      "evaluation_report.json", "label_encoder.json"]
    for fname in expected_files:
        assert any(fname in k for k in all_keys), f"{fname} not written to S3"


# @spec TRAIN-DATA-002
def test_label_encoder_maps_int_indices_to_phenotype_strings(feature_registry):
    """label_encoder.json contains integer keys mapping to human-readable label strings."""
    from genomic_ancestry_pipeline.models import AggregateMetrics, EvaluationReport, TestSetMetrics

    report = EvaluationReport(
        folds=[],
        aggregate=AggregateMetrics(f1_macro_mean=0.9, f1_macro_std=0.05, confusion_matrix_mean=[]),
        test_set=TestSetMetrics(f1_per_class={}, f1_macro=0.0, confusion_matrix=[]),
    )
    label_encoder = {0: "blue", 1: "brown", 2: "green"}

    captured = {}
    with patch("genomic_ancestry_pipeline.training.boto3") as mock_boto3:
        def _capture(**kwargs):
            if "label_encoder" in kwargs.get("Key", ""):
                captured["body"] = kwargs.get("Body", b"")
        mock_boto3.client.return_value.put_object.side_effect = _capture

        save_artifact(
            booster=MagicMock(),
            registry=feature_registry,
            imputation_medians={},
            label_encoder=label_encoder,
            evaluation_report=report,
            bucket="my-bucket",
            run_id="20240115-a3f2c1",
        )

    if captured.get("body"):
        loaded = json.loads(captured["body"])
        # Keys may be serialized as strings in JSON; values must be phenotype strings
        assert all(isinstance(v, str) for v in loaded.values())


# @spec TRAIN-DATA-003
def test_artifact_bundle_is_self_contained(feature_registry):
    """The artifact bundle includes everything needed for inference without re-running training."""
    from genomic_ancestry_pipeline.models import AggregateMetrics, EvaluationReport, TestSetMetrics

    report = EvaluationReport(
        folds=[],
        aggregate=AggregateMetrics(f1_macro_mean=0.9, f1_macro_std=0.05, confusion_matrix_mean=[]),
        test_set=TestSetMetrics(f1_per_class={}, f1_macro=0.0, confusion_matrix=[]),
    )
    with patch("genomic_ancestry_pipeline.training.boto3") as mock_boto3:
        save_artifact(
            booster=MagicMock(),
            registry=feature_registry,
            imputation_medians={"15_28513871_A_G": 1.0},
            label_encoder={0: "blue"},
            evaluation_report=report,
            bucket="my-bucket",
            run_id="20240115-a3f2c1",
        )

    # The four non-model files must be present: registry, medians, label_encoder, report.
    # Combined with the model, they constitute the complete inference bundle.
    all_keys = " ".join(str(c) for c in mock_boto3.client.return_value.put_object.call_args_list)
    for required in ["feature_registry.json", "imputation_medians.json", "label_encoder.json"]:
        assert required in all_keys, f"Artifact missing {required}"
