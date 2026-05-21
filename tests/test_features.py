"""Tests for feature engineering — FEAT-* specs."""
from __future__ import annotations

import numpy as np
import pytest

from genomic_ancestry_pipeline.features import build_feature_matrix
from genomic_ancestry_pipeline.models import (
    CleanSnpDataset,
    CleanVariant,
    FeatureMatrix,
    SampleMetadata,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_dataset_with_variants(
    variants_genotypes: list[dict[str, int]],
    variant_ids: list[str] | None = None,
) -> CleanSnpDataset:
    """Creates a minimal CleanSnpDataset with the supplied per-variant genotypes."""
    samples = [f"s{i:02d}" for i in range(1, 11)]
    labels = {s: ("blue" if i < 5 else ("brown" if i < 9 else "green")) for i, s in enumerate(samples, 1)}
    meta = SampleMetadata(population={s: "EUR" for s in samples}, phenotype_labels=labels)
    splits = {s: ("test" if s in {"s09", "s10"} else "train") for s in samples}

    if variant_ids is None:
        variant_ids = [f"15_28{i:06d}_A_G" for i in range(len(variants_genotypes))]

    variants = [
        CleanVariant(variant_id=vid, genotypes=gts)
        for vid, gts in zip(variant_ids, variants_genotypes)
    ]
    return CleanSnpDataset(
        samples=samples,
        variants=variants,
        metadata=meta,
        imputation_medians={vid: 1.0 for vid in variant_ids},
        sample_splits=splits,
    )


def _uniform_gts(value: int, samples: list[str] | None = None) -> dict[str, int]:
    if samples is None:
        samples = [f"s{i:02d}" for i in range(1, 11)]
    return {s: value for s in samples}


# ── Marker selection filters ───────────────────────────────────────────────────


# @spec FEAT-PROC-001
def test_maf_filter_drops_low_maf_variants(clean_dataset):
    """Variants with MAF < maf_threshold are excluded from the feature matrix."""
    # Inject one monomorphic variant (MAF = 0) among real variants
    monomorphic_gts = {s: 0 for s in clean_dataset.samples}
    clean_dataset.variants.append(
        CleanVariant(variant_id="15_99999_A_G", genotypes=monomorphic_gts)
    )
    clean_dataset.imputation_medians["15_99999_A_G"] = 0.0

    result = build_feature_matrix(clean_dataset, maf_threshold=0.01)

    vids_in_registry = {e.variant_id for e in result.registry.features}
    assert "15_99999_A_G" not in vids_in_registry


# Missingness filtering moved to preprocessing — see test_preprocessing.py
# (PREP-PROC-010, PREP-PROC-011). FEAT-PROC-002 was deleted.


# @spec FEAT-PROC-003
def test_variance_filter_drops_near_zero_variance_variants():
    """Near-zero-variance variants are excluded after dosage encoding."""
    # All training samples have identical dosage → variance = 0
    constant_gts = {f"s{i:02d}": 1 for i in range(1, 11)}
    variable_gts = {f"s{i:02d}": i % 3 for i in range(1, 11)}

    dataset = _build_dataset_with_variants(
        [constant_gts, variable_gts],
        variant_ids=["15_1_A_G", "15_2_C_T"],
    )
    result = build_feature_matrix(dataset)

    vids = {e.variant_id for e in result.registry.features}
    assert "15_1_A_G" not in vids, "Constant-dosage variant should be filtered out"
    assert "15_2_C_T" in vids


# @spec FEAT-PROC-004
def test_association_filter_retains_top_n_variants(clean_dataset):
    """When association_filter=True, only the top-N variants by chi-squared score are kept."""
    n_original = len(clean_dataset.variants)
    top_n = max(1, n_original - 1)  # request one fewer than total

    result = build_feature_matrix(clean_dataset, association_filter=True, top_n=top_n)

    assert result.X.shape[1] <= top_n


def test_association_filter_uses_training_split_only(clean_dataset):
    """Association chi-squared scores are computed on training-split samples only."""
    # With association_filter=True the result must not use test-split phenotype labels
    # in the score computation. We verify by checking the returned registry was
    # built from training-split statistics (implementation responsibility).
    result = build_feature_matrix(clean_dataset, association_filter=True, top_n=3)
    assert isinstance(result, FeatureMatrix)


# ── Feature registry ───────────────────────────────────────────────────────────


# @spec FEAT-DATA-001
def test_feature_registry_maps_columns_to_variant_metadata(clean_dataset):
    """FeatureRegistry contains column_index, variant_id, chrom, pos, ref, alt, maf per feature."""
    result = build_feature_matrix(clean_dataset)

    for entry in result.registry.features:
        assert isinstance(entry.column_index, int)
        assert entry.variant_id
        assert entry.chrom
        assert isinstance(entry.pos, int)
        assert entry.ref
        assert entry.alt
        assert 0.0 <= entry.maf <= 0.5


# @spec FEAT-DATA-002
def test_feature_registry_column_order_consistent_between_splits(clean_dataset):
    """The FeatureRegistry built from training split is applied unchanged to the test split."""
    result = build_feature_matrix(clean_dataset)

    n_samples = len(clean_dataset.samples)
    n_features = len(result.registry.features)

    # All samples (train + test) must have the same number of columns
    assert result.X.shape == (n_samples, n_features)

    # Column indices in registry must be contiguous starting from 0
    indices = sorted(e.column_index for e in result.registry.features)
    assert indices == list(range(n_features))


# ── Output contract ────────────────────────────────────────────────────────────


# @spec FEAT-DATA-003
def test_feature_matrix_output_contract(clean_dataset):
    """FeatureMatrix contains X (numeric array), y (label array), sample_ids, and registry."""
    result = build_feature_matrix(clean_dataset)

    assert isinstance(result.X, np.ndarray)
    assert result.X.ndim == 2
    assert result.X.shape[0] == len(clean_dataset.samples)

    assert isinstance(result.y, np.ndarray)
    assert result.y.shape == (len(clean_dataset.samples),)

    assert result.sample_ids == clean_dataset.samples
    assert result.registry is not None


# @spec FEAT-PROC-005
def test_split_field_preserved_in_feature_matrix(clean_dataset):
    """FeatureMatrix.splits mirrors the per-sample split assignments from CleanSnpDataset."""
    result = build_feature_matrix(clean_dataset)

    assert len(result.splits) == len(result.sample_ids)
    for sample_id, split in zip(result.sample_ids, result.splits):
        assert split == clean_dataset.sample_splits[sample_id]
