"""AWS deployment surface: S3 conventions, SageMaker training launch, Lambda inference.

See ``docs/llds/06_deployment.md`` for the canonical design and
``docs/specs/06_deployment.md`` for the EARS specs that this module realizes.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import boto3
from pydantic import ValidationError
from sagemaker.estimator import Estimator
from sagemaker.session import Session as SagemakerSession

from phenotype_pipeline.prediction import load_artifact, predict

# @spec DEPLOY-BE-019
REGION = "us-east-1"

# Dict keys required in the loaded artifact bundle (DEPLOY-BE-021); maps to the
# files listed in TRAIN-DATA-001 with model.json → "booster".
_BUNDLE_KEYS: tuple[str, ...] = (
    "booster",
    "feature_registry",
    "imputation_medians",
    "evaluation_report",
    "label_encoder",
)

logger = logging.getLogger(__name__)

# Module-level Lambda state. Persists across warm invocations; reset on cold start.
_artifact_cache: dict[str, Any] = {}
_LOAD_FAILED_SENTINEL = "__load_failed__"


# ── S3 conventions ─────────────────────────────────────────────────────────────


# @spec DEPLOY-BE-002
def generate_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}-{short_uuid}"


# @spec DEPLOY-BE-001, DEPLOY-BE-003
def get_s3_paths(bucket: str, run_id: str) -> dict[str, str]:
    model_prefix = f"models/{run_id}/"
    return {
        "vcf": "data/raw/1000genomes.vcf.gz",
        "metadata": "data/raw/sample_info.tsv",
        "model": f"{model_prefix}model.json",
        "feature_registry": f"{model_prefix}feature_registry.json",
        "imputation_medians": f"{model_prefix}imputation_medians.json",
        "evaluation_report": f"{model_prefix}evaluation_report.json",
        "label_encoder": f"{model_prefix}label_encoder.json",
    }


# ── Training launch (SageMaker Python SDK) ─────────────────────────────────────


# @spec TRAIN-BE-001, TRAIN-BE-002, DEPLOY-BE-004, DEPLOY-BE-005, DEPLOY-BE-006,
#       DEPLOY-BE-016, DEPLOY-BE-017, DEPLOY-BE-018, DEPLOY-BE-019
def launch_training_job(
    bucket: str,
    run_id: str,
    instance_type: str = "ml.m5.2xlarge",
    image_uri: str | None = None,
    role_arn: str | None = None,
    **pipeline_params: Any,
) -> str:
    s3 = boto3.client("s3", region_name=REGION)

    s3.head_bucket(Bucket=bucket)

    listing = s3.list_objects_v2(Bucket=bucket, Prefix=f"models/{run_id}/", MaxKeys=1)
    if listing.get("KeyCount", 0) > 0:
        raise FileExistsError(
            f"S3 prefix s3://{bucket}/models/{run_id}/ is already populated; "
            f"refusing to overwrite. Generate a fresh run_id and retry."
        )

    environment: dict[str, str] = {
        "MODEL_RUN_ID": run_id,
        "PHENO_S3_BUCKET": bucket,
    }
    for key, env_name in (
        ("k_folds", "K_FOLDS"),
        ("maf_threshold", "MAF_THRESHOLD"),
        ("top_n", "TOP_N"),
        ("random_state", "RANDOM_STATE"),
    ):
        if key in pipeline_params:
            environment[env_name] = str(pipeline_params[key])

    boto_session = boto3.Session(region_name=REGION)
    sagemaker_session = SagemakerSession(boto_session=boto_session)

    estimator = Estimator(
        image_uri=image_uri or os.environ.get("PHENO_TRAINING_IMAGE_URI", ""),
        role=role_arn or os.environ.get("PHENO_TRAINING_ROLE_ARN", ""),
        instance_type=instance_type,
        instance_count=1,
        output_path=f"s3://{bucket}/models/{run_id}/",
        environment=environment,
        sagemaker_session=sagemaker_session,
    )

    estimator.fit(
        inputs={"raw": f"s3://{bucket}/data/raw/"},
        wait=True,
    )

    job = getattr(estimator, "latest_training_job", None)
    status: str | None = None
    if job is not None:
        try:
            status = job.describe().get("TrainingJobStatus")
        except Exception:  # noqa: BLE001 — describe failures are reported as terminal failure
            status = None

    if status != "Completed":
        raise RuntimeError(
            f"SageMaker training job for run_id={run_id} ended with status={status!r}; "
            f"inspect CloudWatch logs and relaunch with a fresh run_id."
        )

    return run_id


# ── Lambda inference handler (API Gateway HTTP API v2) ─────────────────────────


def _error_response(
    status: int,
    code: str,
    detail: str,
    request_id: str,
) -> dict[str, Any]:
    logger.error(
        "Lambda error [request_id=%s] code=%s detail=%s",
        request_id, code, detail,
    )
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": code, "detail": detail}),
    }


def _ok_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def _ensure_bundle(request_id: str) -> dict[str, Any] | None:
    """Returns an error response if the bundle cannot be loaded, else None.

    Caches the loaded bundle in ``_artifact_cache``. After a failed load,
    plants a sentinel so subsequent requests short-circuit without retrying.
    """
    if _artifact_cache.get(_LOAD_FAILED_SENTINEL):
        return _error_response(
            503, "MODEL_UNAVAILABLE",
            "Model bundle previously failed to load in this container lifecycle.",
            request_id,
        )

    if "booster" in _artifact_cache:
        return None

    try:
        artifact = load_artifact(
            bucket=os.environ.get("PHENO_S3_BUCKET", ""),
            run_id=os.environ.get("MODEL_RUN_ID", ""),
        )
    except Exception as e:  # noqa: BLE001 — any load failure → MODEL_UNAVAILABLE per error contract
        _artifact_cache[_LOAD_FAILED_SENTINEL] = True
        return _error_response(
            503, "MODEL_UNAVAILABLE",
            f"Failed to load model bundle: {e}",
            request_id,
        )

    missing = [k for k in _BUNDLE_KEYS if k not in artifact]
    if missing:
        _artifact_cache[_LOAD_FAILED_SENTINEL] = True
        return _error_response(
            503, "MODEL_UNAVAILABLE",
            f"Model bundle incomplete; missing keys: {missing}",
            request_id,
        )

    _artifact_cache.update(artifact)
    return None


def _route(event: dict[str, Any]) -> tuple[str, str]:
    route_key = event.get("routeKey") or ""
    if " " in route_key:
        method, path = route_key.split(" ", 1)
        return method.upper(), path

    rc_http = event.get("requestContext", {}).get("http", {})
    method = (rc_http.get("method") or "").upper()
    path = rc_http.get("path") or event.get("rawPath") or ""
    return method, path


# @spec DEPLOY-BE-007, DEPLOY-BE-008, DEPLOY-BE-009, DEPLOY-BE-010, DEPLOY-BE-013,
#       DEPLOY-BE-021, DEPLOY-BE-022
def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    request_id = event.get("requestContext", {}).get("requestId", "unknown")

    try:
        bundle_error = _ensure_bundle(request_id)
        if bundle_error is not None:
            return bundle_error

        method, path = _route(event)

        if method == "GET" and path == "/labels":
            label_encoder = _artifact_cache.get("label_encoder") or {}
            if isinstance(label_encoder, dict):
                labels = sorted(label_encoder.values())
            else:
                labels = sorted(label_encoder)
            return _ok_response({"labels": list(labels)})

        if method == "POST" and path == "/predict":
            body_raw = event.get("body")
            try:
                body = json.loads(body_raw) if isinstance(body_raw, str) else (body_raw or {})
            except (json.JSONDecodeError, TypeError) as e:
                return _error_response(
                    400, "INVALID_INPUT",
                    f"Request body is not valid JSON: {e}",
                    request_id,
                )

            if not isinstance(body, dict) or "vcf" not in body:
                return _error_response(
                    400, "INVALID_INPUT",
                    "Request body must be a JSON object with a 'vcf' field.",
                    request_id,
                )

            try:
                result = predict(
                    vcf=body.get("vcf"),
                    phenotype=body.get("phenotype"),
                    artifact=_artifact_cache,
                )
            except ValidationError as e:
                # PredictionResult Pydantic validation (PRED-PROC-010) → 500 INFERENCE_FAILED.
                # ValidationError is a ValueError subclass, so this clause must come first.
                return _error_response(
                    500, "INFERENCE_FAILED",
                    f"PredictionResult validation failed: {e}",
                    request_id,
                )
            except ValueError as e:
                return _error_response(400, "INVALID_VCF", str(e), request_id)
            except Exception as e:  # noqa: BLE001 — predict-path failures map to INFERENCE_FAILED
                return _error_response(500, "INFERENCE_FAILED", str(e), request_id)

            payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            return _ok_response(payload)

        return _error_response(
            400, "INVALID_INPUT",
            f"Unknown route: {method} {path}",
            request_id,
        )

    except Exception as e:  # noqa: BLE001 — catch-all for handler bugs
        logger.exception(
            "Unhandled exception in lambda_handler [request_id=%s]: %s",
            request_id, e,
        )
        return _error_response(500, "INTERNAL_ERROR", str(e), request_id)
