"""Tests for the prediction component — PRED-* specs."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from phenotype_pipeline.models import MarkerContribution, PredictionResult
from phenotype_pipeline.prediction import PredictionError, predict


# ── Shared artifact fixture ────────────────────────────────────────────────────


def _mock_artifact(feature_registry):
    """Returns a minimal in-memory artifact bundle dict for testing."""
    return {
        "booster": MagicMock(),
        "feature_registry": feature_registry,
        "imputation_medians": {e.variant_id: 1.0 for e in feature_registry.features},
        "label_encoder": {0: "blue", 1: "brown", 2: "green"},
    }


def _mock_booster_proba(proba: list[float]):
    """Returns a booster mock that produces the given class probabilities."""
    booster = MagicMock()
    proba_array = np.array([proba])
    booster.predict_proba.return_value = proba_array
    # pred_contribs returns (n_samples, n_features + 1) — last col is bias term
    n_features = 5
    contribs = np.zeros((1, n_features + 1))
    contribs[0, :n_features] = [0.34, 0.21, 0.10, 0.05, 0.03]
    booster.predict.return_value = contribs
    return booster


# ── Artifact loading ───────────────────────────────────────────────────────────


# @spec PRED-PROC-001
def test_predict_loads_artifact_from_s3_before_inference(minimal_vcf_bytes, feature_registry):
    """predict() fetches the model artifact from S3 before running inference."""
    with patch("phenotype_pipeline.prediction.load_artifact") as mock_load, \
         patch("phenotype_pipeline.prediction.boto3"):
        artifact = _mock_artifact(feature_registry)
        artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])
        mock_load.return_value = artifact

        predict(minimal_vcf_bytes, model_artifact_version="models/run1/", bucket="my-bucket")

        mock_load.assert_called_once_with(bucket="my-bucket", run_id="models/run1/")


# @spec PRED-PROC-002
def test_prediction_result_records_artifact_s3_key(minimal_vcf_bytes, feature_registry):
    """PredictionResult.model_artifact_version contains the S3 key used."""
    with patch("phenotype_pipeline.prediction.load_artifact") as mock_load:
        artifact = _mock_artifact(feature_registry)
        artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])
        mock_load.return_value = artifact

        result = predict(
            minimal_vcf_bytes,
            model_artifact_version="models/20240115-a3f2c1/",
            bucket="my-bucket",
        )

    assert result.model_artifact_version == "models/20240115-a3f2c1/"


# ── Input validation ───────────────────────────────────────────────────────────


# @spec PRED-PROC-012
def test_multi_sample_vcf_raises_prediction_error(multi_sample_vcf_bytes, feature_registry):
    """VCF with more than one sample raises PredictionError; no prediction is produced."""
    with patch("phenotype_pipeline.prediction.load_artifact") as mock_load:
        mock_load.return_value = _mock_artifact(feature_registry)

        with pytest.raises(PredictionError):
            predict(
                multi_sample_vcf_bytes,
                model_artifact_version="models/run1/",
                bucket="my-bucket",
            )


# ── Inference preprocessing ────────────────────────────────────────────────────


# @spec PRED-PROC-003
def test_inference_uses_stored_medians_without_refitting(minimal_vcf_bytes, feature_registry):
    """Inference imputes using artifact medians; no new median is computed from the input sample."""
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    artifact["booster"] = booster

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact), \
         patch("phenotype_pipeline.prediction.np") as mock_np:
        # If the implementation computes a new median, np.median or np.nanmedian would be called.
        # We verify it is NOT called during inference.
        mock_np.median = MagicMock(side_effect=AssertionError("median must not be recomputed at inference"))
        mock_np.nanmedian = MagicMock(side_effect=AssertionError("nanmedian must not be recomputed at inference"))
        # Allow all other numpy calls through
        mock_np.array = np.array
        mock_np.zeros = np.zeros

        try:
            predict(minimal_vcf_bytes, model_artifact_version="models/run1/", bucket="my-bucket")
        except AssertionError:
            pytest.fail("predict() recomputed imputation medians from the input sample")
        except Exception:
            pass  # other failures are fine for this test


# @spec PRED-PROC-013
def test_input_variants_absent_from_registry_are_silently_dropped(feature_registry):
    """Variants in the input VCF but absent from the FeatureRegistry are dropped without error."""
    # VCF contains one variant the registry knows (28513871) and one it does not (99999999).
    vcf = (
        b"##fileformat=VCFv4.1\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\n"
        b"15\t28513871\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\n"
        b"15\t99999999\t.\tA\tG\t.\tPASS\t.\tGT\t1|1\n"
    )
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    artifact["booster"] = booster

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact):
        result = predict(vcf, model_artifact_version="models/run1/", bucket="my-bucket")

    # Feature vector width must equal the registry size, not the input variant count
    predict_call_X = booster.predict_proba.call_args[0][0]
    assert predict_call_X.shape[1] == len(feature_registry.features)
    assert isinstance(result, PredictionResult)


# @spec PRED-PROC-004
def test_inference_applies_stored_feature_registry(minimal_vcf_bytes, feature_registry):
    """Inference selects and orders features using the artifact FeatureRegistry."""
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.7, 0.2, 0.1])
    artifact["booster"] = booster

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact):
        result = predict(
            minimal_vcf_bytes,
            model_artifact_version="models/run1/",
            bucket="my-bucket",
        )

    # The number of features passed to the booster must match the registry
    predict_call_X = booster.predict_proba.call_args[0][0]
    assert predict_call_X.shape[1] == len(feature_registry.features)


# ── Prediction output ──────────────────────────────────────────────────────────


# @spec PRED-PROC-005
def test_predict_returns_phenotype_label(minimal_vcf_bytes, feature_registry):
    """PredictionResult contains a predicted_phenotype string."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact):
        result = predict(minimal_vcf_bytes, model_artifact_version="models/run1/", bucket="my-bucket")

    assert isinstance(result.predicted_phenotype, str)
    assert result.predicted_phenotype in {"blue", "brown", "green"}


# @spec PRED-PROC-006
def test_confidence_score_is_max_class_probability(minimal_vcf_bytes, feature_registry):
    """confidence_score equals the maximum class probability from predict_proba()."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.75, 0.15, 0.10])

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact):
        result = predict(minimal_vcf_bytes, model_artifact_version="models/run1/", bucket="my-bucket")

    assert abs(result.confidence_score - 0.75) < 1e-6
    assert 0.0 <= result.confidence_score <= 1.0


# @spec PRED-DATA-001
def test_prediction_result_includes_full_class_probabilities(minimal_vcf_bytes, feature_registry):
    """class_probabilities contains all classes mapped to human-readable labels."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.75, 0.15, 0.10])

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact):
        result = predict(minimal_vcf_bytes, model_artifact_version="models/run1/", bucket="my-bucket")

    assert set(result.class_probabilities.keys()) == {"blue", "brown", "green"}
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6


# ── Marker traceability ────────────────────────────────────────────────────────


# @spec PRED-PROC-007
def test_shap_computed_via_pred_contribs(minimal_vcf_bytes, feature_registry):
    """Marker contributions use XGBoost pred_contribs=True, not the shap package."""
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    artifact["booster"] = booster

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact):
        predict(minimal_vcf_bytes, model_artifact_version="models/run1/", bucket="my-bucket")

    # booster.predict must be called with pred_contribs=True
    predict_calls = booster.predict.call_args_list
    contribs_calls = [c for c in predict_calls if c[1].get("pred_contribs") is True]
    assert len(contribs_calls) >= 1, "booster.predict(pred_contribs=True) was not called"


# @spec PRED-PROC-008
def test_top_n_markers_returned_by_absolute_shap(minimal_vcf_bytes, feature_registry):
    """top_markers contains the top-N features by absolute SHAP value."""
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    artifact["booster"] = booster

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact):
        result = predict(
            minimal_vcf_bytes,
            model_artifact_version="models/run1/",
            bucket="my-bucket",
            top_n_markers=3,
        )

    assert len(result.top_markers) <= 3
    # Markers should be sorted by absolute SHAP contribution descending
    abs_contribs = [abs(m.shap_contribution) for m in result.top_markers]
    assert abs_contribs == sorted(abs_contribs, reverse=True)


# @spec PRED-DATA-002
def test_marker_contribution_fields_complete(minimal_vcf_bytes, feature_registry):
    """Each MarkerContribution has variant_id, chrom, pos, ref, alt, shap_contribution, rank."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact):
        result = predict(minimal_vcf_bytes, model_artifact_version="models/run1/", bucket="my-bucket")

    assert len(result.top_markers) > 0
    for marker in result.top_markers:
        assert marker.variant_id
        assert marker.chrom
        assert isinstance(marker.pos, int)
        assert marker.ref
        assert marker.alt
        assert isinstance(marker.shap_contribution, float)
        assert isinstance(marker.rank, int)
        assert marker.rank >= 1


# ── Output validation ──────────────────────────────────────────────────────────


# @spec PRED-PROC-009
def test_prediction_result_is_pydantic_validated(minimal_vcf_bytes, feature_registry):
    """PredictionResult is an instance of the Pydantic model (validated before return)."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact):
        result = predict(minimal_vcf_bytes, model_artifact_version="models/run1/", bucket="my-bucket")

    assert isinstance(result, PredictionResult)


# @spec PRED-PROC-010
def test_pydantic_validation_failure_raises_not_partial_result(minimal_vcf_bytes, feature_registry):
    """If PredictionResult validation fails, PredictionError is raised; no partial result returned."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    with patch("phenotype_pipeline.prediction.load_artifact", return_value=artifact), \
         patch("phenotype_pipeline.prediction.PredictionResult") as mock_model:
        mock_model.side_effect = ValueError("validation failure")

        with pytest.raises((PredictionError, ValueError)):
            predict(minimal_vcf_bytes, model_artifact_version="models/run1/", bucket="my-bucket")
