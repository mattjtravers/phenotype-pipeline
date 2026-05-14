from __future__ import annotations


def generate_run_id() -> str:
    """Returns a unique run ID: UTC timestamp + short UUID (e.g. '20240115-a3f2c1')."""
    raise NotImplementedError


def get_s3_paths(bucket: str, run_id: str) -> dict[str, str]:
    """Returns the canonical S3 key paths for a given run_id.

    Keys: 'vcf', 'metadata', 'model', 'feature_registry', 'imputation_medians',
          'evaluation_report', 'label_encoder'.
    """
    raise NotImplementedError


def launch_training_job(
    bucket: str,
    run_id: str,
    instance_type: str = "ml.m5.2xlarge",
    **pipeline_params,
) -> str:
    """Submits a SageMaker Training Job and returns the job name."""
    raise NotImplementedError


def lambda_handler(event: dict, context: object) -> dict:
    """AWS Lambda entry point: accepts API Gateway REST or S3 event triggers."""
    raise NotImplementedError
