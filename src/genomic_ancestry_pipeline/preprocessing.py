"""Preprocessing: stratified split, missingness filter, imputation, dosage encoding."""
from __future__ import annotations

import logging

import numpy as np

from genomic_ancestry_pipeline.models import CleanSnpDataset, CleanVariant, RawSnpDataset


class PreprocessingError(Exception):
    """Raised when preprocessing input or output validation fails."""


_DOSAGE = {"0|0": 0, "0|1": 1, "1|0": 1, "1|1": 2}

logger = logging.getLogger(__name__)


# @spec PREP-PROC-001, PREP-PROC-002, PREP-PROC-003, PREP-PROC-004, PREP-PROC-005,
# @spec PREP-PROC-006, PREP-PROC-007, PREP-PROC-008, PREP-PROC-009,
# @spec PREP-PROC-010, PREP-PROC-011, PREP-DATA-001, PREP-DATA-002
def preprocess(
    dataset: RawSnpDataset,
    random_state: int = 42,
    test_size: float = 0.2,
    missingness_threshold: float = 0.10,
) -> CleanSnpDataset:
    """Splits, filters, imputes, encodes, and validates the raw dataset.

    Args:
        dataset: Raw SNP dataset produced by the ingestion step.
        random_state: RNG seed for reproducible stratified splits. Defaults to 42.
        test_size: Fraction of samples to assign to the held-out test split.
            Defaults to 0.2.
        missingness_threshold: Maximum allowable fraction of training-split samples
            with a missing genotype for a variant to be retained. Defaults to 0.10.

    Returns:
        CleanSnpDataset containing dosage-encoded variants (training medians
        imputed), per-variant imputation medians, and train/test split assignments.

    Raises:
        PreprocessingError: If ``dataset`` is not a :class:`RawSnpDataset`.
    """
    if not isinstance(dataset, RawSnpDataset):
        logger.error(
            "preprocess type_error [expected=RawSnpDataset got=%s]",
            type(dataset).__name__,
        )
        raise PreprocessingError(
            f"preprocess() expected RawSnpDataset, got {type(dataset).__name__}"
        )
    logger.info(
        "preprocess start [n_samples=%d n_variants=%d test_size=%.2f missingness_threshold=%.2f]",
        len(dataset.samples), len(dataset.variants), test_size, missingness_threshold,
    )

    sample_splits = _stratified_split(
        samples=dataset.samples,
        labels=dataset.metadata.phenotype_labels,
        test_size=test_size,
        random_state=random_state,
    )
    train_samples = [s for s, sp in sample_splits.items() if sp == "train"]

    kept_variants = _apply_missingness_filter(
        variants=dataset.variants,
        train_samples=train_samples,
        threshold=missingness_threshold,
    )

    clean_variants, imputation_medians = _impute_and_encode(
        variants=kept_variants,
        all_samples=dataset.samples,
        train_samples=train_samples,
    )

    result = CleanSnpDataset(
        samples=dataset.samples,
        variants=clean_variants,
        metadata=dataset.metadata,
        imputation_medians=imputation_medians,
        sample_splits=sample_splits,
    )
    logger.info(
        "preprocess complete [n_samples=%d n_kept_variants=%d]",
        len(result.samples), len(result.variants),
    )
    return result


def _stratified_split(
    samples: list[str],
    labels: dict[str, str],
    test_size: float,
    random_state: int,
) -> dict[str, str]:
    """Custom stratified split that handles small classes sklearn refuses (n_test < n_classes).

    Sklearn's StratifiedShuffleSplit errors when the test fold cannot represent every class.
    Phenotype label classes here can be small (e.g., 2 samples for a minority class), so
    the split prioritizes small classes for test representation and distributes leftover
    slots to the largest classes.
    """
    n_total = len(samples)
    if n_total == 0:
        return {}
    n_test = max(1, int(round(n_total * test_size))) if n_total >= 2 else 0
    n_test = min(n_test, max(0, n_total - 1))

    rng = np.random.default_rng(random_state)
    by_label: dict[str, list[str]] = {}
    for s in samples:
        by_label.setdefault(labels.get(s, ""), []).append(s)
    sorted_labels = sorted(by_label.keys())

    shuffled: dict[str, list[str]] = {}
    for label in sorted_labels:
        arr = sorted(by_label[label])
        rng.shuffle(arr)
        shuffled[label] = arr

    test_alloc = {label: 0 for label in sorted_labels}
    remaining = n_test

    for label in sorted(sorted_labels, key=lambda label_: (len(shuffled[label_]), label_)):
        if remaining == 0:
            break
        if len(shuffled[label]) >= 2:
            test_alloc[label] = 1
            remaining -= 1

    while remaining > 0:
        candidates = [
            label
            for label in sorted_labels
            if test_alloc[label] < len(shuffled[label])
        ]
        if not candidates:
            break
        target = max(
            candidates,
            key=lambda label_: (
                len(shuffled[label_]) - test_alloc[label_],
                -test_alloc[label_],
                label_,
            ),
        )
        test_alloc[target] += 1
        remaining -= 1

    splits: dict[str, str] = {}
    for label in sorted_labels:
        k = test_alloc[label]
        arr = shuffled[label]
        train_slice = arr[:-k] if k > 0 else arr
        test_slice = arr[-k:] if k > 0 else []
        for s in train_slice:
            splits[s] = "train"
        for s in test_slice:
            splits[s] = "test"
    return splits


def _apply_missingness_filter(
    variants,
    train_samples: list[str],
    threshold: float,
):
    """Return variants whose training-split missingness rate is at or below threshold."""
    if not train_samples:
        return list(variants)
    n_train = len(train_samples)
    kept = []
    for variant in variants:
        n_missing = sum(1 for s in train_samples if variant.genotypes.get(s) is None)
        if (n_missing / n_train) > threshold:
            continue
        kept.append(variant)
    return kept


def _impute_and_encode(
    variants,
    all_samples: list[str],
    train_samples: list[str],
) -> tuple[list[CleanVariant], dict[str, float]]:
    """Compute training-split medians, impute missing values, and encode all samples to dosage ints."""
    clean_variants: list[CleanVariant] = []
    imputation_medians: dict[str, float] = {}

    for variant in variants:
        vid = f"{variant.chrom}_{variant.pos}_{variant.ref}_{variant.alt}"
        train_dosages = [
            _DOSAGE[gt]
            for s in train_samples
            if (gt := variant.genotypes.get(s)) in _DOSAGE
        ]
        median = float(np.median(train_dosages)) if train_dosages else 0.0
        imputation_medians[vid] = median

        sample_dosages: dict[str, int] = {}
        impute_value = int(round(median))
        for s in all_samples:
            gt = variant.genotypes.get(s)
            sample_dosages[s] = _DOSAGE[gt] if gt in _DOSAGE else impute_value

        clean_variants.append(CleanVariant(variant_id=vid, genotypes=sample_dosages))

    return clean_variants, imputation_medians
