"""Inference: load model artifact, predict phenotype with confidence and SHAP markers.

Implements the prediction segment of the arrow of intent. :func:`predict` loads the
artifact bundle from S3, parses a single-sample VCF, builds a feature vector using
the stored :class:`FeatureRegistry` and ``imputation_medians`` (no refitting), and
returns a :class:`PredictionResult` with the predicted phenotype, confidence score,
full class probability distribution, and the top-N markers by per-sample SHAP
contribution.

See ``docs/llds/05_prediction.md`` for the canonical design and
``docs/specs/05_prediction.md`` for the EARS specs realized here.

Log records follow the cross-cutting observability standard declared in
``docs/high-level-design.md § Cross-Cutting Code Standards``.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import xgboost as xgb

from phenotype_pipeline.models import (
    FeatureRegistry,
    MarkerContribution,
    PredictionResult,
)

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when the inference input is malformed or output validation fails."""


_DOSAGE = {"0|0": 0, "0|1": 1, "1|0": 1, "1|1": 2}


def _parse_single_sample_vcf(vcf_bytes: bytes) -> tuple[str, dict[str, int]]:
    """Parse a single-sample VCF; raise PredictionError if multi-sample.

    Args:
        vcf_bytes: Raw VCF content.

    Returns:
        Tuple ``(sample_id, genotypes)`` where ``genotypes`` maps the canonical
        ``{chrom}_{pos}_{ref}_{alt}`` variant ID to a 0/1/2 dosage. Unknown or
        missing genotype encodings are omitted, leaving the variant to be imputed
        from stored medians during feature-vector assembly.

    Raises:
        PredictionError: If the VCF declares zero or multiple sample columns.
    """
    text = vcf_bytes.decode("utf-8") if isinstance(vcf_bytes, bytes) else str(vcf_bytes)
    samples: list[str] = []
    genotypes: dict[str, int] = {}
    seen_columns = False

    for line in text.splitlines():
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            cols = line.lstrip("#").split("\t")
            samples = cols[9:] if len(cols) >= 10 else []
            if len(samples) > 1:
                logger.error(
                    "predict input rejected [n_samples=%d]: multi-sample VCFs are not supported",
                    len(samples),
                )
                raise PredictionError(
                    f"VCF contains {len(samples)} samples; predict() requires exactly one"
                )
            if len(samples) == 0:
                logger.error("predict input rejected: VCF has no sample columns")
                raise PredictionError("VCF contains no sample columns")
            seen_columns = True
            continue
        if not seen_columns:
            continue
        cols = line.split("\t")
        if len(cols) < 10:
            continue
        chrom, pos, _id, ref, alt = cols[:5]
        variant_id = f"{chrom}_{pos}_{ref}_{alt}"
        gt_field = cols[9].split(":")[0]
        if gt_field in _DOSAGE:
            genotypes[variant_id] = _DOSAGE[gt_field]

    return samples[0], genotypes


def _build_feature_vector(
    sample_genotypes: dict[str, int],
    registry: FeatureRegistry,
    medians: dict[str, float],
) -> np.ndarray:
    """Assemble a single-sample feature vector ordered by ``registry``.

    Variants in the registry but absent from ``sample_genotypes`` are imputed to
    the stored training median (rounded to an int dosage). Variants in
    ``sample_genotypes`` but absent from the registry are silently dropped — the
    model has no weight for them.

    Args:
        sample_genotypes: Variant ID → 0/1/2 dosage for variants present in the VCF.
        registry: FeatureRegistry from the loaded artifact bundle.
        medians: Per-variant training medians from the loaded artifact bundle.

    Returns:
        Float ndarray of shape ``(1, len(registry.features))`` in column order.
    """
    sorted_entries = sorted(registry.features, key=lambda e: e.column_index)
    values: list[float] = []
    for entry in sorted_entries:
        if entry.variant_id in sample_genotypes:
            values.append(float(sample_genotypes[entry.variant_id]))
        else:
            median = medians.get(entry.variant_id, 0.0)
            values.append(float(round(median)))
    return np.array([values], dtype=float)


# @spec PRED-PROC-001, PRED-PROC-002, PRED-PROC-003, PRED-PROC-004, PRED-PROC-005,
#       PRED-PROC-006, PRED-PROC-007, PRED-PROC-008, PRED-PROC-009, PRED-PROC-010,
#       PRED-PROC-012, PRED-PROC-013, PRED-DATA-001, PRED-DATA-002
def predict(
    vcf_bytes: bytes,
    model_artifact_version: str,
    bucket: str,
    top_n_markers: int = 20,
) -> PredictionResult:
    """Run single-sample inference and return a Pydantic-validated PredictionResult.

    Loads the artifact bundle from S3, validates the input VCF, builds the feature
    vector using the stored registry and medians (no refitting), runs
    ``predict_proba`` and ``predict(pred_contribs=True)``, and assembles the
    PredictionResult with phenotype, confidence, full class probabilities, and
    top-N markers by absolute SHAP contribution.

    Args:
        vcf_bytes: Raw VCF content with exactly one sample column.
        model_artifact_version: S3 bundle prefix (e.g. ``"models/20240115-a3f2c1/"``)
            identifying the artifact to load. Recorded verbatim in the returned
            PredictionResult.
        bucket: S3 bucket containing the artifact bundle.
        top_n_markers: Maximum number of marker contributions to return, sorted by
            absolute SHAP value descending. Defaults to 20.

    Returns:
        Pydantic-validated PredictionResult containing the predicted phenotype label,
        confidence score (max class probability), full class probability distribution,
        top-N marker contributions, and ``model_artifact_version``.

    Raises:
        PredictionError: If the VCF contains zero or multiple samples.
    """
    logger.info(
        "predict start [bucket=%s artifact=%s top_n_markers=%d]",
        bucket, model_artifact_version, top_n_markers,
    )

    artifact = load_artifact(bucket=bucket, run_id=model_artifact_version)
    booster = artifact["booster"]
    registry: FeatureRegistry = artifact["feature_registry"]
    medians: dict[str, float] = artifact["imputation_medians"]
    label_encoder: dict[int, str] = artifact["label_encoder"]

    sample_id, sample_genotypes = _parse_single_sample_vcf(vcf_bytes)
    X = _build_feature_vector(sample_genotypes, registry, medians)

    proba = np.asarray(booster.predict_proba(X))
    proba_row = proba[0]
    n_classes = len(proba_row)

    predicted_idx = int(np.argmax(proba_row))
    confidence = float(proba_row[predicted_idx])
    predicted_label = label_encoder.get(predicted_idx, str(predicted_idx))
    class_probabilities = {
        label_encoder.get(i, str(i)): float(proba_row[i]) for i in range(n_classes)
    }

    # XGBoost per-sample SHAP via pred_contribs=True returns shape (1, n_features + 1)
    # — the trailing column is the bias term and is excluded from marker attribution.
    contribs = np.asarray(booster.predict(X, pred_contribs=True))
    contribs_row = contribs[0]
    n_features = len(registry.features)
    feature_contribs = contribs_row[:n_features]

    sorted_entries = sorted(registry.features, key=lambda e: e.column_index)
    abs_sorted_indices = np.argsort(np.abs(feature_contribs))[::-1][:top_n_markers]
    top_markers: list[MarkerContribution] = []
    for rank, idx in enumerate(abs_sorted_indices, start=1):
        idx_int = int(idx)
        entry = sorted_entries[idx_int]
        top_markers.append(MarkerContribution(
            variant_id=entry.variant_id,
            chrom=entry.chrom,
            pos=entry.pos,
            ref=entry.ref,
            alt=entry.alt,
            shap_contribution=float(feature_contribs[idx_int]),
            rank=rank,
        ))

    result = PredictionResult(
        sample_id=sample_id,
        predicted_phenotype=predicted_label,
        confidence_score=confidence,
        class_probabilities=class_probabilities,
        top_markers=top_markers,
        model_artifact_version=model_artifact_version,
    )

    logger.info(
        "predict complete [sample_id=%s phenotype=%s confidence=%.3f n_markers=%d]",
        sample_id, predicted_label, confidence, len(top_markers),
    )
    return result


# @spec PRED-PROC-001
def load_artifact(bucket: str, run_id: str) -> dict:
    """Load the artifact bundle from ``s3://{bucket}/models/{run_id}/``.

    Accepts either a clean ``run_id`` (e.g. ``"20240115-a3f2c1"``) or a full bundle
    prefix (e.g. ``"models/20240115-a3f2c1/"``); both forms resolve to the same
    canonical S3 prefix.

    Args:
        bucket: S3 bucket containing the artifact files.
        run_id: Clean run identifier or full ``models/{run_id}/`` prefix.

    Returns:
        Dict with keys ``booster``, ``feature_registry``, ``imputation_medians``,
        ``label_encoder``, and ``evaluation_report``. The ``booster`` value
        exposes ``predict_proba`` (sklearn API) for class probabilities; per-sample
        SHAP via ``predict(..., pred_contribs=True)`` requires routing through the
        underlying Booster (deferred — see ``docs/llds/05_prediction.md``).
    """
    prefix = run_id if run_id.startswith("models/") and run_id.endswith("/") else f"models/{run_id}/"
    logger.info("load_artifact start [bucket=%s prefix=%s]", bucket, prefix)

    s3 = boto3.client("s3")

    def _get_json(key: str) -> Any:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())

    model_obj = s3.get_object(Bucket=bucket, Key=f"{prefix}model.json")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp_path.write_bytes(model_obj["Body"].read())
    try:
        clf = xgb.XGBClassifier()
        clf.load_model(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    feature_registry = FeatureRegistry(**_get_json(f"{prefix}feature_registry.json"))
    imputation_medians = _get_json(f"{prefix}imputation_medians.json")
    label_encoder_raw = _get_json(f"{prefix}label_encoder.json")
    # JSON serializes int keys as strings; restore int keys for ergonomic lookup.
    label_encoder = {int(k): v for k, v in label_encoder_raw.items()}
    evaluation_report = _get_json(f"{prefix}evaluation_report.json")

    logger.info(
        "load_artifact complete [bucket=%s prefix=%s n_features=%d]",
        bucket, prefix, len(feature_registry.features),
    )
    return {
        "booster": clf,
        "feature_registry": feature_registry,
        "imputation_medians": imputation_medians,
        "label_encoder": label_encoder,
        "evaluation_report": evaluation_report,
    }
