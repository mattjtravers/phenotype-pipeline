"""SageMaker training entry point.

Reads pipeline parameters from environment variables injected by
deployment.launch_training_job, runs the full training pipeline
(ingest → preprocess → feature engineering → train → save artifact), and exits.
Artifact is written directly to S3 via boto3; SageMaker supplies credentials
from the execution role automatically.

Environment variables:
    PHENO_S3_BUCKET  — project S3 bucket (required)
    MODEL_RUN_ID     — unique run identifier; artifacts land under models/{run_id}/ (required)
    K_FOLDS          — k for stratified CV (default 5)
    MAF_THRESHOLD    — minimum MAF to retain a variant (default 0.01)
    TOP_N            — maximum features after association filter (default 10000)
    RANDOM_STATE     — RNG seed (default 42)
"""
from __future__ import annotations

import logging
import os

from phenotype_pipeline.features import build_feature_matrix
from phenotype_pipeline.ingest import load_raw_dataset
from phenotype_pipeline.logging_config import configure_logging
from phenotype_pipeline.preprocessing import preprocess
from phenotype_pipeline.training import save_artifact, train

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


# @spec TRAIN-BE-001, DEPLOY-BE-024
def main() -> None:
    bucket = os.environ["PHENO_S3_BUCKET"]
    run_id = os.environ["MODEL_RUN_ID"]
    k_folds = int(os.environ.get("K_FOLDS", "5"))
    top_n = int(os.environ.get("TOP_N", "10000"))
    maf_threshold = float(os.environ.get("MAF_THRESHOLD", "0.01"))
    random_state = int(os.environ.get("RANDOM_STATE", "42"))
    max_variants_env = os.environ.get("MAX_VARIANTS", "")
    max_variants = int(max_variants_env) if max_variants_env else None

    logger.info(
        "sagemaker_train start [bucket=%s run_id=%s k_folds=%d top_n=%d max_variants=%s]",
        bucket,
        run_id,
        k_folds,
        top_n,
        max_variants,
    )

    raw = load_raw_dataset(bucket=bucket, max_variants=max_variants)
    clean = preprocess(raw, random_state=random_state)
    feature_matrix = build_feature_matrix(
        clean,
        maf_threshold=maf_threshold,
        association_filter=True,
        top_n=top_n,
    )

    # Build the int→label decoder for save_artifact.
    # Uses the same sort-and-enumerate logic as features._build_label_encoder so
    # integer indices here match the encoded y values in feature_matrix.
    unique_labels = sorted(
        {lbl for lbl in clean.metadata.phenotype_labels.values() if lbl}
    )
    label_encoder: dict[int, str] = {idx: lbl for idx, lbl in enumerate(unique_labels)}

    booster, evaluation_report = train(
        feature_matrix, k_folds=k_folds, random_state=random_state
    )

    save_artifact(
        booster=booster,
        registry=feature_matrix.registry,
        imputation_medians=clean.imputation_medians,
        label_encoder=label_encoder,
        evaluation_report=evaluation_report,
        bucket=bucket,
        run_id=run_id,
    )

    logger.info("sagemaker_train complete [bucket=%s run_id=%s]", bucket, run_id)


if __name__ == "__main__":
    main()
