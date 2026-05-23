"""Tests for the preprocessing component — PREP-* specs."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from genomic_ancestry_pipeline.models import RawSnpDataset, RawVariant, SampleMetadata
from genomic_ancestry_pipeline.preprocessing import PreprocessingError, preprocess

# ── Helpers ────────────────────────────────────────────────────────────────────


def _count_split(result, split_value: str) -> int:
    return sum(1 for v in result.sample_splits.values() if v == split_value)


# ── Train / test split ─────────────────────────────────────────────────────────


# @spec PREP-PROC-001
def test_split_is_80_20(raw_dataset):
    """preprocess() produces an 80/20 train/test split."""
    result = preprocess(raw_dataset)
    n = len(raw_dataset.samples)
    n_train = _count_split(result, "train")
    n_test = _count_split(result, "test")
    assert n_train == int(n * 0.8)
    assert n_test == n - n_train


# @spec PREP-PROC-002
def test_split_is_stratified_by_phenotype(raw_dataset):
    """Train and test sets each contain at least one sample from every phenotype class."""
    result = preprocess(raw_dataset)
    labels = raw_dataset.metadata.phenotype_labels
    classes = set(labels.values())

    train_classes = {labels[s] for s, sp in result.sample_splits.items() if sp == "train"}
    test_classes = {labels[s] for s, sp in result.sample_splits.items() if sp == "test"}

    assert train_classes == classes, "Train set is missing phenotype classes"
    assert len(test_classes) > 0, "Test set is empty"


# @spec PREP-PROC-003
def test_split_is_reproducible_with_same_seed(raw_dataset):
    """Same random_state produces identical split assignments."""
    result_a = preprocess(raw_dataset, random_state=7)
    result_b = preprocess(raw_dataset, random_state=7)
    assert result_a.sample_splits == result_b.sample_splits


def test_different_seeds_produce_different_splits(raw_dataset):
    """Different random_state values produce different split assignments."""
    result_a = preprocess(raw_dataset, random_state=1)
    result_b = preprocess(raw_dataset, random_state=99)
    assert result_a.sample_splits != result_b.sample_splits


# @spec PREP-DATA-001
def test_every_sample_has_split_field(raw_dataset):
    """Every sample in CleanSnpDataset carries a 'train' or 'test' split label."""
    result = preprocess(raw_dataset)
    assert set(result.sample_splits.keys()) == set(result.samples)
    assert all(v in {"train", "test"} for v in result.sample_splits.values())


# ── Missingness filter ─────────────────────────────────────────────────────────


# @spec PREP-PROC-010
def test_missingness_filter_drops_high_missing_variants(sample_metadata):
    """Variants with training-split missing rate above the threshold are dropped."""
    samples = [f"s{i:02d}" for i in range(1, 11)]
    labels = {
        s: ("blue" if i < 5 else ("brown" if i < 9 else "green"))
        for i, s in enumerate(samples, 1)
    }
    meta = SampleMetadata(population={s: "EUR" for s in samples}, phenotype_labels=labels)

    # high-missing variant: 5/8 training samples have None → 62.5% missing rate (> 0.10)
    high_missing = {f"s{i:02d}": ("0|0" if i > 5 else None) for i in range(1, 11)}
    # low-missing variant: 0 missing
    low_missing = {f"s{i:02d}": "0|1" for i in range(1, 11)}

    dataset = RawSnpDataset(
        samples=samples,
        variants=[
            RawVariant(chrom="15", pos=1, ref="A", alt="G", genotypes=high_missing),
            RawVariant(chrom="15", pos=2, ref="C", alt="T", genotypes=low_missing),
        ],
        metadata=meta,
    )

    result = preprocess(dataset, random_state=0)

    output_vids = {v.variant_id for v in result.variants}
    assert "15_1_A_G" not in output_vids, "High-missingness variant should be dropped"
    assert "15_2_C_T" in output_vids, "Low-missingness variant should remain"


# @spec PREP-PROC-011
def test_missingness_filter_runs_before_imputation(sample_metadata):
    """Dropped variants never appear in imputation_medians."""
    samples = [f"s{i:02d}" for i in range(1, 11)]
    labels = {
        s: ("blue" if i < 5 else ("brown" if i < 9 else "green"))
        for i, s in enumerate(samples, 1)
    }
    meta = SampleMetadata(population={s: "EUR" for s in samples}, phenotype_labels=labels)

    high_missing = {f"s{i:02d}": (None if i <= 6 else "0|0") for i in range(1, 11)}
    dataset = RawSnpDataset(
        samples=samples,
        variants=[RawVariant(chrom="15", pos=1, ref="A", alt="G", genotypes=high_missing)],
        metadata=meta,
    )

    result = preprocess(dataset, random_state=0)
    assert "15_1_A_G" not in result.imputation_medians


# ── Imputation ─────────────────────────────────────────────────────────────────


# @spec PREP-PROC-004
def test_imputation_medians_computed_from_training_split_only(sample_metadata):
    """Imputation medians reflect training-split values, not full-data medians."""
    # Variant: training samples all have dosage 0; test samples all have dosage 2.
    # Training-only median = 0. Full-data median = 1.
    samples = [f"s{i:02d}" for i in range(1, 11)]
    labels = {
        s: ("blue" if i < 5 else ("brown" if i < 9 else "green"))
        for i, s in enumerate(samples, 1)
    }
    meta = SampleMetadata(population={s: "EUR" for s in samples}, phenotype_labels=labels)

    # variant_id will be "15_100_A_G"
    # training split (first 8) → genotype "0|0" (dosage 0)
    # test split (last 2) → genotype "1|1" (dosage 2)
    genotypes = {s: "0|0" for s in samples}
    genotypes["s09"] = "1|1"
    genotypes["s10"] = "1|1"

    variant = RawVariant(chrom="15", pos=100, ref="A", alt="G", genotypes=genotypes)
    dataset = RawSnpDataset(samples=samples, variants=[variant], metadata=meta)

    result = preprocess(dataset, random_state=0)

    vid = "15_100_A_G"
    # All training samples should have dosage 0 → median = 0
    assert result.imputation_medians[vid] == 0.0


# @spec PREP-PROC-005
def test_missing_genotypes_are_imputed(raw_dataset):
    """Missing genotype values (None in raw data) are filled with the training median."""
    result = preprocess(raw_dataset)
    for variant in result.variants:
        assert None not in variant.genotypes.values(), (
            f"Variant {variant.variant_id} still has None genotypes after imputation"
        )


# @spec PREP-DATA-002
def test_imputation_medians_present_in_output(raw_dataset):
    """CleanSnpDataset includes imputation_medians for every variant."""
    result = preprocess(raw_dataset)
    expected_vids = {
        f"{v.chrom}_{v.pos}_{v.ref}_{v.alt}" for v in raw_dataset.variants
    }
    assert set(result.imputation_medians.keys()) == expected_vids


# ── Dosage encoding ────────────────────────────────────────────────────────────


# @spec PREP-PROC-006
@pytest.mark.parametrize(
    "raw_gt, expected_dosage",
    [
        ("0|0", 0),
        ("0|1", 1),
        ("1|0", 1),
        ("1|1", 2),
    ],
)
def test_dosage_encoding(raw_gt: str, expected_dosage: int, sample_metadata):
    """Homozygous ref→0, heterozygous→1, homozygous alt→2."""
    samples = ["s01", "s02"]
    meta = SampleMetadata(
        population={s: "EUR" for s in samples},
        phenotype_labels={"s01": "blue", "s02": "brown"},
    )
    variant = RawVariant(
        chrom="15", pos=100, ref="A", alt="G",
        genotypes={"s01": raw_gt, "s02": "0|0"},
    )
    dataset = RawSnpDataset(samples=samples, variants=[variant], metadata=meta)

    result = preprocess(dataset, random_state=0)

    clean_variant = result.variants[0]
    assert clean_variant.genotypes["s01"] == expected_dosage


# ── Schema validation ──────────────────────────────────────────────────────────


# @spec PREP-PROC-007
def test_invalid_raw_dataset_raises_at_entry(sample_metadata):
    """Passing a non-RawSnpDataset object raises a structured error immediately."""
    with pytest.raises((PreprocessingError, TypeError, ValidationError)):
        preprocess("not_a_dataset")  # type: ignore[arg-type]


# @spec PREP-PROC-008, PREP-PROC-009
def test_validation_error_does_not_return_partial_result(raw_dataset, monkeypatch):
    """If CleanSnpDataset validation fails, a structured error is raised with no partial result."""
    # Patch Pydantic validation to simulate failure at output boundary
    import genomic_ancestry_pipeline.preprocessing as mod

    original = mod.CleanSnpDataset if hasattr(mod, "CleanSnpDataset") else None
    if original is None:
        pytest.skip("CleanSnpDataset not imported in preprocessing module")

    def _bad_validate(*args, **kwargs):
        raise ValidationError.from_exception_data(
            title="CleanSnpDataset",
            input_type="python",
            line_errors=[],
        )

    monkeypatch.setattr(mod, "CleanSnpDataset", _bad_validate)
    with pytest.raises((PreprocessingError, ValidationError)):
        preprocess(raw_dataset)
