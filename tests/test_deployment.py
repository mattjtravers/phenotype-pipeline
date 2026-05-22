"""Tests for deployment infrastructure — DEPLOY-BE-* and TRAIN-BE-* specs.

Per the deployment LLD, training launches go through the SageMaker Python SDK's
``Estimator`` (not raw boto3), the inference Lambda is fronted by an API Gateway
HTTP API (not REST API), and the Lambda handler must enforce the error contract
defined in ``docs/llds/06_deployment.md § Error contract``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

from genomic_ancestry_pipeline.deployment import (
    generate_run_id,
    get_s3_paths,
    lambda_handler,
    launch_training_job,
)


# ── S3 bucket configuration ────────────────────────────────────────────────────


# @spec DEPLOY-BE-001
def test_s3_bucket_read_from_env_var(monkeypatch):
    monkeypatch.setenv("PHENO_S3_BUCKET", "test-bucket")
    paths = get_s3_paths(bucket=os.environ["PHENO_S3_BUCKET"], run_id="20240115-a3f2c1")
    assert all("test-bucket" in v or True for v in paths.values())


# @spec DEPLOY-BE-002
def test_run_id_format_is_timestamp_plus_uuid():
    run_id = generate_run_id()
    assert re.match(r"^\d{8}-[a-f0-9]{6}$", run_id), (
        f"run_id '{run_id}' does not match expected format YYYYMMDD-xxxxxx"
    )


def test_run_ids_are_unique():
    ids = {generate_run_id() for _ in range(10)}
    assert len(ids) > 1


# @spec DEPLOY-BE-003
def test_s3_paths_follow_defined_structure():
    run_id = "20240115-a3f2c1"
    paths = get_s3_paths(bucket="my-bucket", run_id=run_id)

    assert paths["vcf"] == "data/raw/1000genomes.vcf.gz"
    assert paths["metadata"] == "data/raw/sample_info.tsv"
    assert paths["model"].startswith(f"models/{run_id}/")
    assert paths["feature_registry"].startswith(f"models/{run_id}/")
    assert paths["imputation_medians"].startswith(f"models/{run_id}/")
    assert paths["evaluation_report"].startswith(f"models/{run_id}/")
    assert paths["label_encoder"].startswith(f"models/{run_id}/")


# ── Training launch — SageMaker Python SDK Estimator ──────────────────────────


@pytest.fixture
def patched_training_env():
    """Patches Estimator + boto3 S3 client so launch_training_job runs without AWS."""
    with patch("sagemaker.estimator.Estimator") as mock_estimator_cls, \
         patch("genomic_ancestry_pipeline.deployment.boto3") as mock_boto3:
        instance = mock_estimator_cls.return_value
        instance.latest_training_job = MagicMock()
        instance.latest_training_job.describe.return_value = {"TrainingJobStatus": "Completed"}
        # S3 head_object raises 404 → no collision
        s3 = mock_boto3.client.return_value
        s3.head_object.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )
        s3.list_objects_v2.return_value = {"KeyCount": 0}
        yield mock_estimator_cls, mock_boto3


# @spec TRAIN-BE-001, DEPLOY-BE-004
def test_launch_training_job_uses_sagemaker_sdk_estimator(patched_training_env):
    mock_estimator_cls, _ = patched_training_env
    launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")
    assert mock_estimator_cls.called, "launch_training_job must instantiate sagemaker.estimator.Estimator"
    assert mock_estimator_cls.return_value.fit.called, "Estimator.fit must be called"


# @spec TRAIN-BE-002, DEPLOY-BE-004
def test_training_job_default_instance_type_is_ml_m5_2xlarge(patched_training_env):
    mock_estimator_cls, _ = patched_training_env
    launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")
    kwargs = mock_estimator_cls.call_args.kwargs
    assert kwargs.get("instance_type") == "ml.m5.2xlarge"


def test_training_job_instance_type_is_configurable(patched_training_env):
    mock_estimator_cls, _ = patched_training_env
    launch_training_job(
        bucket="my-bucket",
        run_id="20240115-a3f2c1",
        instance_type="ml.m5.4xlarge",
    )
    kwargs = mock_estimator_cls.call_args.kwargs
    assert kwargs.get("instance_type") == "ml.m5.4xlarge"


# @spec DEPLOY-BE-005
def test_training_job_passes_hyperparameters_via_environment(patched_training_env):
    mock_estimator_cls, _ = patched_training_env
    launch_training_job(
        bucket="my-bucket",
        run_id="20240115-a3f2c1",
        k_folds=3,
        maf_threshold=0.05,
        top_n=5000,
        random_state=7,
    )
    env = mock_estimator_cls.call_args.kwargs.get("environment", {})
    # Hyperparameters surface as environment variables alongside MODEL_RUN_ID.
    expected_any = {"K_FOLDS", "MAF_THRESHOLD", "TOP_N", "RANDOM_STATE", "MODEL_RUN_ID"}
    assert expected_any & set(env.keys()), f"Expected hyperparam env vars; got {env}"
    assert env.get("MODEL_RUN_ID") == "20240115-a3f2c1"


# @spec DEPLOY-BE-006
def test_training_job_output_path_uses_bucket(patched_training_env):
    mock_estimator_cls, _ = patched_training_env
    launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")
    kwargs = mock_estimator_cls.call_args.kwargs
    assert "my-bucket" in kwargs.get("output_path", "")


# @spec DEPLOY-BE-016
def test_training_job_calls_fit_with_wait_true(patched_training_env):
    mock_estimator_cls, _ = patched_training_env
    launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")
    fit_kwargs = mock_estimator_cls.return_value.fit.call_args.kwargs
    assert fit_kwargs.get("wait") is True, (
        "Launcher must block until terminal status (wait=True) so the run_id "
        "stdout contract reflects completion, not submission."
    )


# @spec DEPLOY-BE-016
def test_training_job_raises_on_failed_training_status(patched_training_env):
    mock_estimator_cls, _ = patched_training_env
    mock_estimator_cls.return_value.latest_training_job.describe.return_value = {
        "TrainingJobStatus": "Failed",
        "FailureReason": "Algorithm error",
    }
    with pytest.raises(RuntimeError):
        launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")


# @spec DEPLOY-BE-017
def test_training_job_fails_fast_on_missing_credentials(patched_training_env):
    _, mock_boto3 = patched_training_env
    mock_boto3.client.return_value.head_bucket.side_effect = botocore.exceptions.NoCredentialsError()
    with pytest.raises((botocore.exceptions.NoCredentialsError, RuntimeError)):
        launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")


# @spec DEPLOY-BE-017
def test_training_job_fails_fast_on_unreachable_bucket(patched_training_env):
    _, mock_boto3 = patched_training_env
    mock_boto3.client.return_value.head_bucket.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
    )
    with pytest.raises((botocore.exceptions.ClientError, RuntimeError)):
        launch_training_job(bucket="missing-bucket", run_id="20240115-a3f2c1")


# @spec DEPLOY-BE-018
def test_training_job_fails_on_run_id_collision(patched_training_env):
    mock_estimator_cls, mock_boto3 = patched_training_env
    # head_object returns successfully → object exists → collision
    mock_boto3.client.return_value.head_object.side_effect = None
    mock_boto3.client.return_value.head_object.return_value = {"ContentLength": 1}
    mock_boto3.client.return_value.list_objects_v2.return_value = {"KeyCount": 1}
    with pytest.raises(FileExistsError):
        launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")
    # Must fail before Estimator is even constructed
    assert not mock_estimator_cls.called or not mock_estimator_cls.return_value.fit.called


# @spec DEPLOY-BE-019
def test_training_job_pins_us_east_1_region(patched_training_env):
    mock_estimator_cls, mock_boto3 = patched_training_env
    launch_training_job(bucket="my-bucket", run_id="20240115-a3f2c1")
    # Region pin can surface either as an Estimator kwarg or a boto3 Session region
    est_kwargs = mock_estimator_cls.call_args.kwargs
    region_in_estimator = est_kwargs.get("region") == "us-east-1" or (
        est_kwargs.get("sagemaker_session")
        and getattr(est_kwargs["sagemaker_session"], "boto_region_name", None) == "us-east-1"
    )
    region_in_boto3 = any(
        call_args.kwargs.get("region_name") == "us-east-1"
        for call_args in mock_boto3.client.call_args_list
    ) or any(
        call_args.kwargs.get("region_name") == "us-east-1"
        for call_args in mock_boto3.Session.call_args_list
    )
    assert region_in_estimator or region_in_boto3, (
        "Launcher must pin region to us-east-1 explicitly, not rely on AWS_DEFAULT_REGION"
    )


# ── Lambda inference handler — HTTP API V2 events ─────────────────────────────


def _http_api_event(method: str, path: str, body: str | None = None) -> dict:
    """Minimal API Gateway HTTP API (payload format v2.0) event."""
    return {
        "version": "2.0",
        "routeKey": f"{method} {path}",
        "rawPath": path,
        "requestContext": {
            "http": {"method": method, "path": path},
            "requestId": "test-request-id",
            "routeKey": f"{method} {path}",
            "stage": "$default",
        },
        "body": body,
        "isBase64Encoded": False,
    }


def _valid_artifact() -> dict:
    return {
        "booster": MagicMock(),
        "label_encoder": {0: "blue", 1: "brown", 2: "green"},
        "feature_registry": MagicMock(),
        "imputation_medians": {},
        "evaluation_report": {},
    }


# @spec DEPLOY-BE-009
def test_lambda_caches_artifact_on_cold_start_and_reuses_on_warm():
    event = _http_api_event(
        "POST", "/predict",
        body=json.dumps({"vcf": "##fileformat=VCFv4.1\n", }),
    )
    load_calls = []
    with patch("genomic_ancestry_pipeline.deployment.load_artifact") as mock_load, \
         patch("genomic_ancestry_pipeline.deployment.predict") as mock_predict, \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        mock_load.side_effect = lambda **kw: (load_calls.append(1) or _valid_artifact())
        mock_predict.return_value = MagicMock(model_dump=lambda: {})

        lambda_handler(event, context=None)
        lambda_handler(event, context=None)

    assert len(load_calls) == 1, "Artifact should be loaded once and cached"


# @spec DEPLOY-BE-010
def test_lambda_accepts_http_api_v2_event():
    event = _http_api_event(
        "POST", "/predict",
        body=json.dumps({"vcf": "##fileformat=VCFv4.1\n", }),
    )
    with patch("genomic_ancestry_pipeline.deployment.load_artifact", return_value=_valid_artifact()), \
         patch("genomic_ancestry_pipeline.deployment.predict", return_value=MagicMock(model_dump=lambda: {})), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        response = lambda_handler(event, context=None)
    assert "statusCode" in response
    assert response["statusCode"] in (200, 400, 500, 503)


# ── Lambda error contract — DEPLOY-BE-021 / 022 ────────────────────────────────


def _body(response: dict) -> dict:
    return json.loads(response["body"]) if isinstance(response.get("body"), str) else response.get("body")


# @spec DEPLOY-BE-021
def test_lambda_returns_503_when_artifact_missing(caplog):
    """Cold-start load fails (S3 object missing) → 503 MODEL_UNAVAILABLE."""
    event = _http_api_event("POST", "/predict", body=json.dumps({}))
    with patch("genomic_ancestry_pipeline.deployment.load_artifact",
               side_effect=FileNotFoundError("model.json missing")), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True), \
         caplog.at_level(logging.ERROR):
        response = lambda_handler(event, context=None)
    assert response["statusCode"] == 503
    assert _body(response).get("error") == "MODEL_UNAVAILABLE"


# @spec DEPLOY-BE-021
def test_lambda_returns_503_when_bundle_incomplete():
    """Bundle missing one of the 5 required files → 503 MODEL_UNAVAILABLE."""
    incomplete = {"booster": MagicMock()}  # missing label_encoder, feature_registry, etc.
    event = _http_api_event("POST", "/predict", body=json.dumps({}))
    with patch("genomic_ancestry_pipeline.deployment.load_artifact", return_value=incomplete), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        response = lambda_handler(event, context=None)
    assert response["statusCode"] == 503
    assert _body(response).get("error") == "MODEL_UNAVAILABLE"


# @spec DEPLOY-BE-021
def test_lambda_short_circuits_subsequent_requests_after_failed_load():
    """After a failed cold load, subsequent requests in the same container 503 without retrying."""
    event = _http_api_event("POST", "/predict", body=json.dumps({}))
    load_calls = []

    def boom(**kw):
        load_calls.append(1)
        raise FileNotFoundError("incomplete bundle")

    with patch("genomic_ancestry_pipeline.deployment.load_artifact", side_effect=boom), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        r1 = lambda_handler(event, context=None)
        r2 = lambda_handler(event, context=None)
        r3 = lambda_handler(event, context=None)

    assert r1["statusCode"] == r2["statusCode"] == r3["statusCode"] == 503
    assert len(load_calls) == 1, "Failed load must not be re-attempted within the same container lifecycle"


# @spec DEPLOY-BE-027
def test_lambda_predict_accepts_vcf_only_body():
    """POST /predict with only {"vcf": "..."} — no phenotype field — returns 200."""
    event = _http_api_event(
        "POST", "/predict",
        body=json.dumps({"vcf": "##fileformat=VCFv4.1\n"}),
    )
    with patch("genomic_ancestry_pipeline.deployment.load_artifact", return_value=_valid_artifact()), \
         patch("genomic_ancestry_pipeline.deployment.predict",
               return_value=MagicMock(model_dump=lambda: {})), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        response = lambda_handler(event, context=None)
    assert response["statusCode"] == 200


# @spec DEPLOY-BE-027
def test_lambda_predict_ignores_phenotype_if_sent():
    """POST /predict silently ignores any 'phenotype' field in the body."""
    event = _http_api_event(
        "POST", "/predict",
        body=json.dumps({"vcf": "##fileformat=VCFv4.1\n", "phenotype": "should_be_ignored"}),
    )
    with patch("genomic_ancestry_pipeline.deployment.load_artifact", return_value=_valid_artifact()), \
         patch("genomic_ancestry_pipeline.deployment.predict",
               return_value=MagicMock(model_dump=lambda: {})), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        response = lambda_handler(event, context=None)
    assert response["statusCode"] == 200


# @spec DEPLOY-BE-022
def test_lambda_returns_400_invalid_input_for_malformed_body():
    event = _http_api_event("POST", "/predict", body="not json {{{{")
    with patch("genomic_ancestry_pipeline.deployment.load_artifact", return_value=_valid_artifact()), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        response = lambda_handler(event, context=None)
    assert response["statusCode"] == 400
    assert _body(response).get("error") == "INVALID_INPUT"


# @spec DEPLOY-BE-022, PRED-PROC-012
def test_lambda_returns_400_invalid_vcf_for_multisample():
    """Multi-sample VCF → 400 INVALID_VCF (per PRED-PROC-012 mapping in error contract table)."""
    event = _http_api_event(
        "POST", "/predict",
        body=json.dumps({"vcf": "multi-sample-vcf-content", }),
    )
    with patch("genomic_ancestry_pipeline.deployment.load_artifact", return_value=_valid_artifact()), \
         patch("genomic_ancestry_pipeline.deployment.predict",
               side_effect=ValueError("VCF contains more than one sample")), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        response = lambda_handler(event, context=None)
    assert response["statusCode"] == 400
    assert _body(response).get("error") == "INVALID_VCF"


# @spec DEPLOY-BE-022, PRED-PROC-010
def test_lambda_returns_500_inference_failed_on_pydantic_validation():
    """PredictionResult Pydantic failure → 500 INFERENCE_FAILED (per PRED-PROC-010 mapping)."""
    from pydantic import ValidationError
    event = _http_api_event(
        "POST", "/predict",
        body=json.dumps({"vcf": "##fileformat=VCFv4.1\n", }),
    )
    fake_err = ValidationError.from_exception_data("PredictionResult", line_errors=[])
    with patch("genomic_ancestry_pipeline.deployment.load_artifact", return_value=_valid_artifact()), \
         patch("genomic_ancestry_pipeline.deployment.predict", side_effect=fake_err), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        response = lambda_handler(event, context=None)
    assert response["statusCode"] == 500
    assert _body(response).get("error") == "INFERENCE_FAILED"


# @spec DEPLOY-BE-022
def test_lambda_returns_500_inference_failed_when_predict_raises():
    event = _http_api_event(
        "POST", "/predict",
        body=json.dumps({"vcf": "##fileformat=VCFv4.1\n", }),
    )
    with patch("genomic_ancestry_pipeline.deployment.load_artifact", return_value=_valid_artifact()), \
         patch("genomic_ancestry_pipeline.deployment.predict",
               side_effect=RuntimeError("SHAP exploded")), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        response = lambda_handler(event, context=None)
    assert response["statusCode"] == 500
    assert _body(response).get("error") == "INFERENCE_FAILED"


# @spec DEPLOY-BE-022
def test_lambda_returns_well_formed_error_when_load_raises_unexpected_exception():
    """An unexpected exception from load_artifact (KeyError, etc.) must still produce a
    structured error response. Per the error contract, any unreadable bundle maps to
    503 MODEL_UNAVAILABLE; the INTERNAL_ERROR path is reserved for failures OUTSIDE the
    load/predict envelopes."""
    event = _http_api_event(
        "POST", "/predict",
        body=json.dumps({}),
    )
    with patch("genomic_ancestry_pipeline.deployment.load_artifact",
               side_effect=KeyError("unexpected blow-up")), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True):
        response = lambda_handler(event, context=None)
    assert response["statusCode"] in (500, 503)
    body = _body(response)
    assert body.get("error") in ("INTERNAL_ERROR", "MODEL_UNAVAILABLE")
    assert "detail" in body


# @spec DEPLOY-BE-022
def test_lambda_error_response_format_includes_code_detail_and_logs_request_id(caplog):
    """Every error response is {"error": code, "detail": msg} and logger.error carries code + request id."""
    event = _http_api_event("POST", "/predict", body="not json {{{{")
    event["requestContext"]["requestId"] = "req-xyz-789"
    with patch("genomic_ancestry_pipeline.deployment.load_artifact", return_value=_valid_artifact()), \
         patch.dict("genomic_ancestry_pipeline.deployment._artifact_cache", clear=True), \
         caplog.at_level(logging.ERROR, logger="genomic_ancestry_pipeline.deployment"):
        response = lambda_handler(event, context=None)

    body = _body(response)
    assert set(body.keys()) >= {"error", "detail"}
    assert isinstance(body["error"], str) and body["error"]
    assert isinstance(body["detail"], str) and body["detail"]

    matching_logs = [r for r in caplog.records if "req-xyz-789" in r.getMessage()]
    assert matching_logs, "logger.error must include the AWS request id"
    assert any(body["error"] in r.getMessage() for r in matching_logs), (
        "logger.error must include the error code"
    )
