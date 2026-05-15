"""Feature engineering: marker selection and feature matrix assembly.

Applies MAF, variance, and optional chi-squared association filters.
"""
from __future__ import annotations

import logging

import numpy as np

from phenotype_pipeline.models import (
    CleanSnpDataset,
    FeatureEntry,
    FeatureMatrix,
    FeatureRegistry,
)

logger = logging.getLogger(__name__)


# @spec FEAT-PROC-001, FEAT-PROC-003, FEAT-PROC-004, FEAT-PROC-005,
# @spec FEAT-DATA-001, FEAT-DATA-002, FEAT-DATA-003
def build_feature_matrix(
    dataset: CleanSnpDataset,
    maf_threshold: float = 0.01,
    association_filter: bool = False,
    top_n: int = 10_000,
) -> FeatureMatrix:
    """Apply marker selection filters and assemble the model-ready FeatureMatrix.

    MAF, variance, and optional chi-squared association filters are computed on
    training-split samples only; the resulting FeatureRegistry is applied unchanged
    to the test split.

    Args:
        dataset: Preprocessed SNP dataset with dosage-encoded genotypes and
            train/test split assignments.
        maf_threshold: Minimum minor allele frequency on the training split for
            a variant to pass the MAF filter. Defaults to 0.01.
        association_filter: If True, apply a chi-squared association filter after
            MAF/variance filtering and retain at most ``top_n`` features.
            Defaults to False.
        top_n: Maximum number of features to retain when ``association_filter``
            is True. Defaults to 10,000.

    Returns:
        FeatureMatrix with model-ready arrays X and y, sample identifiers,
        split assignments, and a FeatureRegistry mapping column indices to
        genomic coordinates.
    """
    logger.info(
        "build_feature_matrix start [n_samples=%d n_variants=%d maf_threshold=%.4f association_filter=%s]",
        len(dataset.samples), len(dataset.variants), maf_threshold, association_filter,
    )
    samples = list(dataset.samples)
    splits = [dataset.sample_splits[s] for s in samples]
    train_indices = [i for i, sp in enumerate(splits) if sp == "train"]

    n_samples = len(samples)
    n_variants = len(dataset.variants)

    X_full = np.zeros((n_samples, n_variants), dtype=float)
    for j, variant in enumerate(dataset.variants):
        for i, s in enumerate(samples):
            X_full[i, j] = variant.genotypes[s]

    mafs = np.zeros(n_variants, dtype=float)
    kept_mask = np.ones(n_variants, dtype=bool)
    if train_indices:
        X_train = X_full[train_indices, :]
        n_train = len(train_indices)
        allele_sums = X_train.sum(axis=0)
        p_alt = allele_sums / (2.0 * n_train)
        mafs = np.minimum(p_alt, 1.0 - p_alt)
        kept_mask &= mafs >= maf_threshold
        variances = X_train.var(axis=0)
        kept_mask &= variances > 0.0
    else:
        kept_mask[:] = False

    kept_indices = [int(i) for i in np.where(kept_mask)[0]]

    label_encoder = _build_label_encoder(
        labels=dataset.metadata.phenotype_labels,
        samples=samples,
    )
    y_full = np.array(
        [label_encoder.get(dataset.metadata.phenotype_labels.get(s, ""), -1) for s in samples],
        dtype=int,
    )

    if association_filter and kept_indices and train_indices:
        kept_indices = _select_top_n_by_chi2(
            X_train=X_full[np.ix_(train_indices, kept_indices)],
            y_train=y_full[train_indices],
            kept_indices=kept_indices,
            top_n=top_n,
        )

    X_selected = (
        X_full[:, kept_indices]
        if kept_indices
        else np.zeros((n_samples, 0), dtype=float)
    )

    feature_entries: list[FeatureEntry] = []
    for col_idx, var_j in enumerate(kept_indices):
        variant = dataset.variants[var_j]
        chrom, pos, ref, alt = _parse_variant_id(variant.variant_id)
        feature_entries.append(
            FeatureEntry(
                column_index=col_idx,
                variant_id=variant.variant_id,
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                maf=float(mafs[var_j]),
            )
        )

    result = FeatureMatrix(
        X=X_selected,
        y=y_full,
        sample_ids=samples,
        splits=splits,
        registry=FeatureRegistry(features=feature_entries),
    )
    logger.info(
        "build_feature_matrix complete [n_features=%d n_samples=%d]",
        len(feature_entries), len(samples),
    )
    return result


def _build_label_encoder(labels: dict[str, str], samples: list[str]) -> dict[str, int]:
    """Map sorted unique phenotype label strings to integer class indices."""
    unique = sorted({labels[s] for s in samples if s in labels})
    return {label: idx for idx, label in enumerate(unique)}


def _parse_variant_id(vid: str) -> tuple[str, int, str, str]:
    """Split a variant_id string into (chrom, pos, ref, alt) components."""
    parts = vid.rsplit("_", 3)
    if len(parts) != 4:
        logger.error("_parse_variant_id malformed [variant_id=%r]", vid)
        raise ValueError(f"Cannot parse variant_id {vid!r}")
    chrom, pos_str, ref, alt = parts
    return chrom, int(pos_str), ref, alt


def _select_top_n_by_chi2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kept_indices: list[int],
    top_n: int,
) -> list[int]:
    """Select the top-n feature indices by chi-squared association score."""
    from sklearn.feature_selection import chi2

    n_features = X_train.shape[1]
    k = min(top_n, n_features)
    if k == 0:
        return []
    scores, _ = chi2(X_train, y_train)
    scores = np.nan_to_num(scores, nan=0.0)
    order = np.argsort(scores, kind="stable")[::-1][:k]
    selected_local = sorted(int(o) for o in order)
    return [kept_indices[i] for i in selected_local]
