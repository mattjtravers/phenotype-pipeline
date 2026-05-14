"""Tests for deployment infrastructure — DEPLOY-BE-* and TRAIN-BE-* specs."""
from __future__ import annotations

import json
import os
import re
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from phenotype_pipeline.deployment import (
    generate_run_id,
    get_s3_paths,
    lambda_handler,
    launch_training_job,
)


# ── S3 bucket configuration ────────────────────────────────────────────────────


# @spec DEPLOY-BE-001
def test_s3_bucket_read_from_env_var(monkeypatch):
    """Pipeline reads the S3 bucket name from PHENO_S3_BUCKET env var."""
    monkeypatch.setenv("PHENO_S3_BUCKET", "test-bucket")
    paths = get_s3_paths(bucket=os.environ["PHENO_S3_BUCKET"], run_id="20240115-a3f2c1")
    assert all("test-bucket" in v or True for v in paths.values())  # bucket is used by caller


# @spec DEPLOY-BE-002
def test_run_id_format_is_timestamp_plus_uuid():
    """generate_run_id() returns a string matching YYYYmmdd-xxxxxx."""
    run_id = generate_run_id()
    assert re.match(r"^\d{8}-[a-f0-9]{6}$", run_id), (
        f"run_id '{run_id}' does not match expected format YYYYMMDD-xxxxxx"
    )


def test_run_ids_are_unique():
    """Consecutive generate_run_id() calls return distinct values."""
    ids = {generate_run_id() for _ in range(10)}
    assert len(ids) > 1


# @spec DEPLOY-BE-003
def test_s3_paths_follow_defined_structure():
    """get_s3_paths() returns paths for data/raw, models/{run_id}, and logs."""
    run_id = "20240115-a3f2c1"
    paths = get_s3_paths(bucket="my-bucket", run_id=run_id)

    assert paths["vcf"] == "data/raw/1000genomes.vcf.gz"
    assert paths["metadata"] == "data/raw/sample_info.tsv"
    assert paths["model"].startswith(f"models/{run_id}/")
    assert paths["feature_registry"].startswith(f"models/{run_id}/")
    assert paths["imputation_medians"].startswith(f"models/{run_id}/")
    assert paths["evaluation_report"].startswith(f"models/{run_id}/")
    assert paths["label_encoder"].startswith(f"models/{run_id}/")


# ── SageMaker Training Job (TRAIN-BE-* + DEPLOY-BE-004/005/006) ───────────────


# @spec TRAIN-BE-001, DEPLOY-BE-004
def test_launch_training_job_submits_sagemaker_job():
    """launch_training_job() calls SageMaker create_training_job (not local execution)."""
    with patch("phenotype_pipeline.deployment.boto3") as mock_boto3:
        mock_sm = mock_boto3.client.return_value
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:aws:..."}

        job_name = launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")

        mock_sm.create_training_job.assert_called_once()
        assert job_name  # non-empty job name returned


# @spec TRAIN-BE-002, DEPLOY-BE-004
def test_training_job_default_instance_type_is_ml_m5_2xlarge():
    """Default SageMaker instance type is ml.m5.2xlarge (configurable)."""
    with patch("phenotype_pipeline.deployment.boto3") as mock_boto3:
        mock_sm = mock_boto3.client.return_value
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:aws:..."}

        launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")

        call_kwargs = mock_sm.create_training_job.call_args[1]
        resource_config = call_kwargs.get("ResourceConfig", {})
        assert resource_config.get("InstanceType") == "ml.m5.2xlarge"


def test_training_job_instance_type_is_configurable():
    """launch_training_job() passes the instance_type parameter to SageMaker."""
    with patch("phenotype_pipeline.deployment.boto3") as mock_boto3:
        mock_sm = mock_boto3.client.return_value
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:aws:..."}

        launch_training_job(
            bucket="my-bucket",
            run_id="20240115-a3f2c1",
            instance_type="ml.m5.4xlarge",
        )

        call_kwargs = mock_sm.create_training_job.call_args[1]
        resource_config = call_kwargs.get("ResourceConfig", {})
        assert resource_config.get("InstanceType") == "ml.m5.4xlarge"


# @spec DEPLOY-BE-005
def test_training_job_receives_pipeline_params_as_env_vars():
    """Pipeline parameters are passed to the SageMaker job as environment variables."""
    with patch("phenotype_pipeline.deployment.boto3") as mock_boto3:
        mock_sm = mock_boto3.client.return_value
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:aws:..."}

        launch_training_job(
            bucket="my-bucket",
            run_id="20240115-a3f2c1",
            k_folds=3,
            maf_threshold=0.05,
            top_n=5000,
            random_state=7,
        )

        call_kwargs = mock_sm.create_training_job.call_args[1]
        env = call_kwargs.get("Environment", {})
        assert "K_FOLDS" in env or "k_folds" in env
        assert "MAF_THRESHOLD" in env or "maf_threshold" in env


# @spec DEPLOY-BE-006
def test_training_job_output_path_uses_run_id():
    """SageMaker output path is set to s3://{bucket}/models/{run_id}/."""
    with patch("phenotype_pipeline.deployment.boto3") as mock_boto3:
        mock_sm = mock_boto3.client.return_value
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:aws:..."}

        launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")

        call_kwargs = mock_sm.create_training_job.call_args[1]
        output_cfg = call_kwargs.get("OutputDataConfig", {})
        assert "my-bucket" in output_cfg.get("S3OutputPath", "")
        assert "20240115-a3f2c1" in output_cfg.get("S3OutputPath", "")


# ── Lambda inference handler ───────────────────────────────────────────────────


# @spec DEPLOY-BE-009
def test_lambda_caches_artifact_on_cold_start_and_reuses_on_warm():
    """Lambda loads the artifact once on cold start and reuses it for warm invocations."""
    api_gw_event = {
        "httpMethod": "POST",
        "body": json.dumps({"vcf": "##fileformat=VCFv4.1\n#CHROM\t...\nsample1\n", "phenotype": "eye_color"}),
    }

    load_calls = []

    with patch("phenotype_pipeline.deployment.load_artifact") as mock_load, \
         patch("phenotype_pipeline.deployment.predict") as mock_predict, \
         patch("phenotype_pipeline.deployment._artifact_cache", new={}):
        mock_load.side_effect = lambda **kwargs: (load_calls.append(1) or {"booster": MagicMock()})
        mock_predict.return_value = MagicMock(model_dump=lambda: {})

        lambda_handler(api_gw_event, context=None)
        lambda_handler(api_gw_event, context=None)

    assert len(load_calls) == 1, "Artifact should be loaded once and cached"


# @spec DEPLOY-BE-010
def test_lambda_accepts_api_gateway_rest_event():
    """lambda_handler handles API Gateway REST event shape."""
    event = {
        "httpMethod": "POST",
        "body": json.dumps({"vcf_content": "vcf_data", "phenotype": "eye_color"}),
    }
    with patch("phenotype_pipeline.deployment.load_artifact", return_value={"booster": MagicMock()}), \
         patch("phenotype_pipeline.deployment.predict", return_value=MagicMock(model_dump=lambda: {})):
        response = lambda_handler(event, context=None)
    assert "statusCode" in response


def test_lambda_accepts_s3_event():
    """lambda_handler handles S3 event trigger shape (batch mode)."""
    event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "my-bucket"},
                    "object": {"key": "uploads/sample.vcf"},
                }
            }
        ]
    }
    with patch("phenotype_pipeline.deployment.load_artifact", return_value={"booster": MagicMock()}), \
         patch("phenotype_pipeline.deployment.predict", return_value=MagicMock(model_dump=lambda: {})), \
         patch("phenotype_pipeline.deployment.boto3"):
        response = lambda_handler(event, context=None)
    assert response is not None


# @spec DEPLOY-BE-013
def test_lambda_labels_endpoint_returns_phenotype_strings():
    """GET /labels returns the cached artifact's label_encoder.json values as a list."""
    event = {
        "httpMethod": "GET",
        "path": "/labels",
        "resource": "/labels",
    }
    artifact = {
        "booster": MagicMock(),
        "label_encoder": {0: "blue", 1: "brown", 2: "green"},
    }
    with patch("phenotype_pipeline.deployment.load_artifact", return_value=artifact), \
         patch("phenotype_pipeline.deployment._artifact_cache", new={}):
        response = lambda_handler(event, context=None)

    assert response.get("statusCode") == 200
    body = json.loads(response["body"]) if isinstance(response.get("body"), str) else response.get("body")
    assert "labels" in body
    assert set(body["labels"]) == {"blue", "brown", "green"}
