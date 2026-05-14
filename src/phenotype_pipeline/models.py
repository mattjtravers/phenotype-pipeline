from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel


class SampleMetadata(BaseModel):
    population: dict[str, str]  # sample_id → population code
    phenotype_labels: dict[str, str]  # sample_id → phenotype label string


class RawVariant(BaseModel):
    chrom: str
    pos: int
    ref: str
    alt: str
    genotypes: dict[str, str | None]  # sample_id → "0|0" | "0|1" | "1|1" | None (missing)


class RawSnpDataset(BaseModel):
    samples: list[str]
    variants: list[RawVariant]
    metadata: SampleMetadata


class CleanVariant(BaseModel):
    variant_id: str  # "{chrom}_{pos}_{ref}_{alt}"
    genotypes: dict[str, int]  # sample_id → 0 | 1 | 2 (dosage encoding)


class CleanSnpDataset(BaseModel):
    samples: list[str]
    variants: list[CleanVariant]
    metadata: SampleMetadata
    imputation_medians: dict[str, float]  # variant_id → training-split median
    sample_splits: dict[str, Literal["train", "test"]]  # sample_id → split assignment


class FeatureEntry(BaseModel):
    column_index: int
    variant_id: str  # "{chrom}_{pos}_{ref}_{alt}"
    chrom: str
    pos: int
    ref: str
    alt: str
    maf: float


class FeatureRegistry(BaseModel):
    features: list[FeatureEntry]


@dataclass
class FeatureMatrix:
    X: np.ndarray  # shape (n_samples, n_features), dtype float
    y: np.ndarray  # shape (n_samples,), phenotype class indices
    sample_ids: list[str]
    splits: list[Literal["train", "test"]]  # parallel to sample_ids
    registry: FeatureRegistry


class MarkerContribution(BaseModel):
    variant_id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    shap_contribution: float  # signed; magnitude = importance
    rank: int  # 1 = largest absolute contribution


class PredictionResult(BaseModel):
    sample_id: str
    predicted_phenotype: str
    confidence_score: float  # max class probability, in [0.0, 1.0]
    class_probabilities: dict[str, float]  # all classes, human-readable labels
    top_markers: list[MarkerContribution]
    model_artifact_version: str  # S3 key of the artifact bundle used


class FoldResult(BaseModel):
    fold_index: int
    f1_per_class: dict[str, float]  # class label → F1
    f1_macro: float
    confusion_matrix: list[list[int]]


class AggregateMetrics(BaseModel):
    f1_macro_mean: float
    f1_macro_std: float
    confusion_matrix_mean: list[list[float]]


class TestSetMetrics(BaseModel):
    f1_per_class: dict[str, float]
    f1_macro: float
    confusion_matrix: list[list[int]]


class EvaluationReport(BaseModel):
    folds: list[FoldResult]
    aggregate: AggregateMetrics
    test_set: TestSetMetrics | None = None
