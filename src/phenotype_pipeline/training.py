from __future__ import annotations

from phenotype_pipeline.models import EvaluationReport, FeatureMatrix, FeatureRegistry


def train(
    feature_matrix: FeatureMatrix,
    k_folds: int = 5,
    random_state: int = 42,
    early_stopping_rounds: int = 20,
) -> tuple:
    """Trains XGBoost with stratified k-fold CV on training-split samples.

    Returns (booster, EvaluationReport). The report includes per-fold metrics,
    aggregate CV metrics, and final test-set evaluation run exactly once after
    the final model is retrained on all training-split samples.
    """
    raise NotImplementedError


def save_artifact(
    booster,
    registry: FeatureRegistry,
    imputation_medians: dict[str, float],
    label_encoder: dict[int, str],
    evaluation_report: EvaluationReport,
    bucket: str,
    run_id: str,
) -> None:
    """Persists the model artifact bundle to s3://{bucket}/models/{run_id}/."""
    raise NotImplementedError
