from __future__ import annotations

from phenotype_pipeline.models import PredictionResult


class PredictionError(Exception):
    pass


def predict(
    vcf_bytes: bytes,
    model_artifact_version: str,
    bucket: str,
    top_n_markers: int = 20,
) -> PredictionResult:
    """Loads the artifact bundle and runs inference on a single-sample VCF.

    Raises PredictionError if the VCF contains more than one sample.
    Uses stored imputation medians and FeatureRegistry — no refitting.
    Computes per-sample SHAP via XGBoost pred_contribs=True.
    """
    raise NotImplementedError


def load_artifact(bucket: str, run_id: str) -> dict:
    """Loads and returns the model artifact bundle from S3."""
    raise NotImplementedError
