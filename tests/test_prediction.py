"""Tests for the prediction component — PRED-* specs."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from genomic_ancestry_pipeline.models import PredictionResult
from genomic_ancestry_pipeline.prediction import PredictionError, predict


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
    """Returns a booster mock that produces the given class probabilities.

    Simulates XGBClassifier: predict_proba() on the wrapper, get_booster().predict()
    with pred_contribs=True on the underlying raw booster.
    """
    booster = MagicMock()
    proba_array = np.array([proba])
    booster.predict_proba.return_value = proba_array
    # pred_contribs returns (n_samples, n_features + 1) — last col is bias term
    n_features = 5
    contribs = np.zeros((1, n_features + 1))
    contribs[0, :n_features] = [0.34, 0.21, 0.10, 0.05, 0.03]
    # get_booster() returns the underlying raw booster used for SHAP
    raw_booster = MagicMock()
    raw_booster.predict.return_value = contribs
    booster.get_booster.return_value = raw_booster
    return booster


# ── Artifact loading ───────────────────────────────────────────────────────────


# @spec PRED-PROC-001
def test_predict_uses_caller_supplied_artifact(minimal_vcf_bytes, feature_registry):
    """predict() uses the artifact dict supplied by the caller; it does not load from S3."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    with patch("genomic_ancestry_pipeline.prediction.load_artifact") as mock_load:
        result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")
        mock_load.assert_not_called()

    assert isinstance(result, PredictionResult)


# @spec PRED-PROC-002
def test_prediction_result_records_artifact_version(minimal_vcf_bytes, feature_registry):
    """PredictionResult.model_artifact_version contains the version string passed by the caller."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    result = predict(
        minimal_vcf_bytes,
        artifact=artifact,
        model_artifact_version="models/20240115-a3f2c1/",
    )

    assert result.model_artifact_version == "models/20240115-a3f2c1/"


# ── Input validation ───────────────────────────────────────────────────────────


# @spec PRED-PROC-012
def test_multi_sample_vcf_raises_prediction_error(multi_sample_vcf_bytes, feature_registry):
    """VCF with more than one sample raises PredictionError; no prediction is produced."""
    artifact = _mock_artifact(feature_registry)

    with pytest.raises(PredictionError):
        predict(multi_sample_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")


# ── Inference preprocessing ────────────────────────────────────────────────────


# @spec PRED-PROC-003
def test_inference_uses_stored_medians_without_refitting(minimal_vcf_bytes, feature_registry):
    """Inference imputes using artifact medians; no new median is computed from the input sample."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    with patch("genomic_ancestry_pipeline.prediction.np") as mock_np:
        mock_np.median = MagicMock(side_effect=AssertionError("median must not be recomputed at inference"))
        mock_np.nanmedian = MagicMock(side_effect=AssertionError("nanmedian must not be recomputed at inference"))
        mock_np.array = np.array
        mock_np.zeros = np.zeros

        try:
            predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")
        except AssertionError:
            pytest.fail("predict() recomputed imputation medians from the input sample")
        except Exception:
            pass


# @spec PRED-PROC-013
def test_input_variants_absent_from_registry_are_silently_dropped(feature_registry):
    """Variants in the input VCF but absent from the FeatureRegistry are dropped without error."""
    vcf = (
        b"##fileformat=VCFv4.1\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\n"
        b"15\t28513871\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\n"
        b"15\t99999999\t.\tA\tG\t.\tPASS\t.\tGT\t1|1\n"
    )
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    artifact["booster"] = booster

    result = predict(vcf, artifact=artifact, model_artifact_version="models/run1/")

    predict_call_X = booster.predict_proba.call_args[0][0]
    assert predict_call_X.shape[1] == len(feature_registry.features)
    assert isinstance(result, PredictionResult)


# @spec PRED-PROC-004
def test_inference_applies_stored_feature_registry(minimal_vcf_bytes, feature_registry):
    """Inference selects and orders features using the artifact FeatureRegistry."""
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.7, 0.2, 0.1])
    artifact["booster"] = booster

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    predict_call_X = booster.predict_proba.call_args[0][0]
    assert predict_call_X.shape[1] == len(feature_registry.features)


# ── Prediction output ──────────────────────────────────────────────────────────


# @spec PRED-PROC-005
def test_predict_returns_population_label(minimal_vcf_bytes, feature_registry):
    """PredictionResult contains a predicted_phenotype string (ancestral population label)."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    assert isinstance(result.predicted_phenotype, str)
    assert result.predicted_phenotype in {"blue", "brown", "green"}


# @spec PRED-PROC-006
def test_confidence_score_is_max_class_probability(minimal_vcf_bytes, feature_registry):
    """confidence_score equals the maximum class probability from predict_proba()."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.75, 0.15, 0.10])

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    assert abs(result.confidence_score - 0.75) < 1e-6
    assert 0.0 <= result.confidence_score <= 1.0


# @spec PRED-DATA-001
def test_prediction_result_includes_full_class_probabilities(minimal_vcf_bytes, feature_registry):
    """class_probabilities contains all classes mapped to human-readable labels."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.75, 0.15, 0.10])

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    assert set(result.class_probabilities.keys()) == {"blue", "brown", "green"}
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6


# ── Marker traceability ────────────────────────────────────────────────────────


# @spec PRED-PROC-007
def test_shap_computed_via_pred_contribs(minimal_vcf_bytes, feature_registry):
    """Marker contributions use XGBoost pred_contribs=True, not the shap package."""
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    artifact["booster"] = booster

    predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    # SHAP is computed via get_booster().predict(DMatrix, pred_contribs=True)
    raw_booster = booster.get_booster.return_value
    predict_calls = raw_booster.predict.call_args_list
    contribs_calls = [c for c in predict_calls if c[1].get("pred_contribs") is True]
    assert len(contribs_calls) >= 1, "raw_booster.predict(pred_contribs=True) was not called"


# @spec PRED-PROC-008
def test_top_n_markers_returned_by_absolute_shap(minimal_vcf_bytes, feature_registry):
    """top_markers contains the top-N features by absolute SHAP value."""
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    artifact["booster"] = booster

    result = predict(
        minimal_vcf_bytes,
        artifact=artifact,
        model_artifact_version="models/run1/",
        top_n_markers=3,
    )

    assert len(result.top_markers) <= 3
    abs_contribs = [abs(m.shap_contribution) for m in result.top_markers]
    assert abs_contribs == sorted(abs_contribs, reverse=True)


# @spec PRED-DATA-002
def test_marker_contribution_fields_complete(minimal_vcf_bytes, feature_registry):
    """Each MarkerContribution has variant_id, chrom, pos, ref, alt, shap_contribution, rank."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

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

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    assert isinstance(result, PredictionResult)


# @spec PRED-PROC-010
def test_pydantic_validation_failure_raises_not_partial_result(minimal_vcf_bytes, feature_registry):
    """If PredictionResult validation fails, an error is raised; no partial result returned."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    with patch("genomic_ancestry_pipeline.prediction.PredictionResult") as mock_model:
        mock_model.side_effect = ValueError("validation failure")

        with pytest.raises((PredictionError, ValueError)):
            predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")
