"""AWS deployment surface: S3 conventions, SageMaker training launch, Lambda inference.

This module realizes the deployment segment of the arrow of intent:

- :func:`generate_run_id` and :func:`get_s3_paths` enforce the canonical S3
  layout for raw data, model artifacts, and logs.
- :func:`launch_training_job` submits a SageMaker training job via the
  SageMaker Python SDK, performs pre-flight checks (credentials, bucket
  reachability, ``run_id`` collision), blocks until terminal status, and
  returns the ``run_id`` on success.
- :func:`lambda_handler` is the API Gateway HTTP API entry point for
  ``POST /predict``. It enforces the error contract
  defined in ``docs/llds/06_deployment.md § Error contract``.

See ``docs/llds/06_deployment.md`` for the canonical design and
``docs/specs/06_deployment.md`` for the EARS specs this module realizes.

Log records emitted from this module follow the cross-cutting observability
standard declared in ``docs/high-level-design.md § Cross-Cutting Code Standards``:
every error path emits a ``logger.error`` line containing operation, error
code, detail, and request/run identifier so an operator can triage from log
output alone.
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

from genomic_ancestry_pipeline.prediction import load_artifact, predict

# @spec DEPLOY-BE-019 — region pin; do not rely on AWS_DEFAULT_REGION.
REGION = "us-east-1"

# Dict keys required in the loaded artifact bundle (DEPLOY-BE-021); maps to the
# files listed in TRAIN-DATA-001 with model.json deserializing to "booster".
_BUNDLE_KEYS: tuple[str, ...] = (
    "booster",
    "feature_registry",
    "imputation_medians",
    "evaluation_report",
    "label_encoder",
)

logger = logging.getLogger(__name__)

# Module-level Lambda state. Persists across warm invocations within a single
# container lifecycle; reset on cold start. The sentinel key encodes "we tried
# and failed" so subsequent requests short-circuit without retrying the load.
_artifact_cache: dict[str, Any] = {}
_LOAD_FAILED_SENTINEL = "__load_failed__"


# ── S3 conventions ─────────────────────────────────────────────────────────────


# @spec DEPLOY-BE-002
def generate_run_id() -> str:
    """Generate a unique training run identifier.

    The format is ``YYYYMMDD-xxxxxx`` where the date portion is UTC (so runs
    sort lexicographically by submission time) and the suffix is the first
    six hex characters of a UUID4. Collisions are vanishingly unlikely but
    are also detected explicitly by :func:`launch_training_job`.

    Returns:
        The new ``run_id`` string (e.g. ``"20240115-a3f2c1"``).
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}-{short_uuid}"


# @spec DEPLOY-BE-001, DEPLOY-BE-003
def get_s3_paths(bucket: str, run_id: str) -> dict[str, str]:
    """Return the canonical S3 key paths for a given training run.

    Paths are returned as bucket-relative keys (no ``s3://bucket/`` prefix),
    matching the convention used by :mod:`boto3` low-level S3 calls. Callers
    that need a full URI should prepend ``f"s3://{bucket}/"`` themselves.

    Args:
        bucket: S3 bucket name (typically the value of ``PHENO_S3_BUCKET``).
        run_id: Training run identifier produced by :func:`generate_run_id`.

    Returns:
        A dict with keys ``"vcf"``, ``"metadata"``, ``"model"``,
        ``"feature_registry"``, ``"imputation_medians"``, ``"evaluation_report"``,
        ``"label_encoder"`` mapping to the canonical S3 key path for each artifact.
    """
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
    """Submit a SageMaker training job, block until completion, return the run_id.

    The function performs three pre-flight checks before constructing an
    :class:`~sagemaker.estimator.Estimator`:

    1. ``s3.head_bucket(Bucket=bucket)`` — surfaces ``NoCredentialsError`` or
       ``ClientError`` immediately rather than after the training job has
       been quoted (DEPLOY-BE-017).
    2. ``s3.list_objects_v2`` on ``models/{run_id}/`` — refuses to launch if
       the prefix already contains objects, preventing an in-flight job from
       silently overwriting a prior run's artifacts (DEPLOY-BE-018).
    3. A region-pinned :class:`SagemakerSession` is constructed from a
       :class:`boto3.Session` with ``region_name="us-east-1"``, so the job
       lands in the configured region regardless of ``AWS_DEFAULT_REGION``
       (DEPLOY-BE-019).

    The job is launched synchronously (``wait=True``). On terminal status,
    the function inspects ``TrainingJobStatus`` and raises ``RuntimeError``
    for anything other than ``Completed``. This makes the function safe to
    use in the operator pattern ``sam deploy --parameter-overrides MODEL_RUN_ID=$(launch_training.py ...)``:
    the ``run_id`` only reaches stdout on a confirmed-good model.

    Args:
        bucket: S3 bucket containing inputs and receiving artifacts.
        run_id: Identifier for this run; must be unique (no collision under
            ``s3://bucket/models/{run_id}/``).
        instance_type: SageMaker training instance type. Defaults to
            ``"ml.m5.2xlarge"`` per TRAIN-BE-002.
        image_uri: ECR image URI for the training container. If ``None``,
            falls back to the ``PHENO_TRAINING_IMAGE_URI`` environment
            variable.
        role_arn: IAM role ARN that SageMaker assumes during training. If
            ``None``, falls back to the ``PHENO_TRAINING_ROLE_ARN`` env var.
        **pipeline_params: Hyperparameters surfaced to the container as
            environment variables. Recognized keys: ``k_folds``,
            ``maf_threshold``, ``top_n``, ``random_state``.

    Returns:
        The same ``run_id`` that was passed in, returned only after the
        training job reports ``TrainingJobStatus = "Completed"``.

    Raises:
        botocore.exceptions.NoCredentialsError: AWS credentials are missing.
        botocore.exceptions.ClientError: The bucket is unreachable, or the
            S3 prefix check failed for a reason other than 404.
        FileExistsError: ``models/{run_id}/`` already contains objects;
            generate a fresh ``run_id`` and retry.
        RuntimeError: The training job reached terminal status without
            success (``Failed`` or ``Stopped``). Inspect CloudWatch logs.
    """
    s3 = boto3.client("s3", region_name=REGION)

    logger.info(
        "launch_training_job start [run_id=%s bucket=%s instance_type=%s]",
        run_id, bucket, instance_type,
    )

    s3.head_bucket(Bucket=bucket)

    listing = s3.list_objects_v2(Bucket=bucket, Prefix=f"models/{run_id}/", MaxKeys=1)
    if listing.get("KeyCount", 0) > 0:
        logger.error(
            "launch_training_job collision [run_id=%s bucket=%s]: S3 prefix already populated",
            run_id, bucket,
        )
        raise FileExistsError(
            f"S3 prefix s3://{bucket}/models/{run_id}/ is already populated; "
            f"refusing to overwrite. Generate a fresh run_id and retry."
        )

    environment: dict[str, str] = {
        "MODEL_RUN_ID": run_id,
        "PHENO_S3_BUCKET": bucket,
    }
    # Pipeline hyperparameters travel as environment variables (DEPLOY-BE-005)
    # rather than SageMaker HyperParameters so the training container can read
    # them with a single os.environ.get() lookup, identical to how the Lambda
    # reads MODEL_RUN_ID. This keeps the train-vs-infer interface symmetric.
    for kwarg_name, env_name in (
        ("k_folds", "K_FOLDS"),
        ("maf_threshold", "MAF_THRESHOLD"),
        ("top_n", "TOP_N"),
        ("random_state", "RANDOM_STATE"),
    ):
        if kwarg_name in pipeline_params:
            environment[env_name] = str(pipeline_params[kwarg_name])

    # Lazy imports: sagemaker is not installed in the training container, only
    # on the developer machine / launcher. Importing at call time rather than
    # module level lets training.py import deployment.get_s3_paths without
    # pulling in the sagemaker SDK.
    from sagemaker.estimator import Estimator  # noqa: PLC0415
    from sagemaker.session import Session as SagemakerSession  # noqa: PLC0415

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

    # wait=True is load-bearing: synchronous launch is what makes
    # `sam deploy --parameter-overrides MODEL_RUN_ID=$(launch_training.py ...)`
    # safe — the run_id only reaches stdout (DEPLOY-BE-016) after a successful
    # terminal status check below.
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
            logger.exception(
                "launch_training_job describe failed [run_id=%s]", run_id,
            )
            status = None

    if status != "Completed":
        logger.error(
            "launch_training_job terminal-non-success [run_id=%s status=%s]",
            run_id, status,
        )
        raise RuntimeError(
            f"SageMaker training job for run_id={run_id} ended with status={status!r}; "
            f"inspect CloudWatch logs and relaunch with a fresh run_id."
        )

    logger.info("launch_training_job completed [run_id=%s]", run_id)
    return run_id


# ── Lambda inference handler (API Gateway HTTP API v2) ─────────────────────────


def _error_response(
    status: int,
    code: str,
    detail: str,
    request_id: str,
) -> dict[str, Any]:
    """Build an error response and emit a matching structured log line.

    Every Lambda error path funnels through this helper to guarantee the
    response body shape and log line shape both satisfy DEPLOY-BE-022.

    Args:
        status: HTTP status code (e.g. 400, 500, 503).
        code: Short error code from the contract table (``MODEL_UNAVAILABLE``,
            ``INVALID_INPUT``, ``INVALID_VCF``, ``INFERENCE_FAILED``,
            ``INTERNAL_ERROR``).
        detail: Human-readable description; safe to surface to clients.
        request_id: The AWS request identifier from the API Gateway event,
            embedded in the log line so an operator can grep CloudWatch by
            request and find the same record they're triaging.

    Returns:
        A Lambda response dict suitable for return from :func:`lambda_handler`
        with ``statusCode``, ``headers``, and a JSON ``body`` of shape
        ``{"error": code, "detail": detail}``.
    """
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
    """Build a 200 OK response with a JSON body.

    Args:
        payload: Dict to serialize as the response body.

    Returns:
        Lambda response dict with ``statusCode=200``.
    """
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def _ensure_bundle(request_id: str) -> dict[str, Any] | None:
    """Lazily load and validate the model artifact bundle.

    Reads :data:`_artifact_cache`. If empty, calls :func:`load_artifact` and
    verifies every key in :data:`_BUNDLE_KEYS` is present. After a failed
    load (either an exception or an incomplete bundle), plants a sentinel
    in the cache so subsequent requests in the same container lifecycle
    short-circuit to ``503 MODEL_UNAVAILABLE`` without retrying. This is the
    behavior mandated by DEPLOY-BE-021.

    Args:
        request_id: The AWS request id for log correlation.

    Returns:
        ``None`` if the bundle is loaded and complete; otherwise an error
        response dict that the caller should return directly.
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
    logger.info("Model bundle loaded successfully")
    return None


def _route(event: dict[str, Any]) -> tuple[str, str]:
    """Extract the HTTP method and path from an API Gateway HTTP API v2 event.

    The HTTP API v2 event carries the route under ``routeKey`` (a string of
    the form ``"POST /predict"``) and redundantly under
    ``requestContext.http``. This helper prefers the former and falls back
    to the latter for defensive parsing.

    Args:
        event: The Lambda event dict as delivered by API Gateway.

    Returns:
        A ``(method, path)`` tuple with method in uppercase.
    """
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
    """AWS Lambda entry point for the inference API.

    Dispatches API Gateway HTTP API v2 events to the single endpoint:

    - ``POST /predict`` — runs inference on a single-sample VCF and returns a
      :class:`PredictionResult` JSON body.

    Before dispatch, :func:`_ensure_bundle` confirms the model artifact has
    been loaded and is complete; any failure short-circuits to
    ``503 MODEL_UNAVAILABLE`` for the rest of the container's lifecycle.

    Every error path returns a structured ``{"error", "detail"}`` body and
    logs a matching ``logger.error`` line including the AWS request id, per
    DEPLOY-BE-022.

    Args:
        event: The API Gateway HTTP API v2 event payload. Expected keys
            include ``routeKey``, ``body``, and ``requestContext.requestId``.
        context: AWS Lambda context object. Unused; accepted for the
            standard handler signature.

    Returns:
        A Lambda response dict with ``statusCode``, ``headers``, and ``body``.
    """
    request_id = event.get("requestContext", {}).get("requestId", "unknown")

    try:
        bundle_error = _ensure_bundle(request_id)
        if bundle_error is not None:
            return bundle_error

        method, path = _route(event)

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

            vcf_raw = body.get("vcf", "")
            vcf_bytes = vcf_raw.encode("utf-8") if isinstance(vcf_raw, str) else vcf_raw
            try:
                result = predict(
                    vcf_bytes=vcf_bytes,
                    artifact=_artifact_cache,
                    model_artifact_version=os.environ.get("MODEL_RUN_ID", ""),
                )
            except ValidationError as e:
                # PredictionResult Pydantic validation (PRED-PROC-010) → 500 INFERENCE_FAILED.
                # ValidationError is a ValueError subclass, so this clause must come first;
                # otherwise the broader ValueError handler below would catch it as INVALID_VCF.
                return _error_response(
                    500, "INFERENCE_FAILED",
                    f"PredictionResult validation failed: {e}",
                    request_id,
                )
            except ValueError as e:
                # Plain ValueError from predict() → bad input (multi-sample VCF,
                # malformed VCF, no usable variants) per the error-contract table.
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
