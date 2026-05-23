"""Shared fixtures for all test modules."""
from __future__ import annotations

import numpy as np
import pytest

from genomic_ancestry_pipeline.models import (
    CleanSnpDataset,
    CleanVariant,
    FeatureEntry,
    FeatureMatrix,
    FeatureRegistry,
    RawSnpDataset,
    RawVariant,
    SampleMetadata,
)


@pytest.fixture(autouse=True)
def _set_pheno_api_endpoint(monkeypatch):
    """Ensure PHENO_API_ENDPOINT is set for all tests so the UI doesn't st.stop()."""
    monkeypatch.setenv("PHENO_API_ENDPOINT", "https://mock.example.com")

# 10 samples across 3 phenotype classes to support stratified splitting
# and k-fold cross-validation with k=5 (minimum 2 samples per class).
SAMPLE_IDS = [f"s{i:02d}" for i in range(1, 11)]
PHENOTYPE_LABELS = {
    "s01": "blue",
    "s02": "blue",
    "s03": "blue",
    "s04": "blue",
    "s05": "brown",
    "s06": "brown",
    "s07": "brown",
    "s08": "brown",
    "s09": "green",
    "s10": "green",
}
POPULATIONS = {s: "EUR" for s in SAMPLE_IDS}

VARIANTS_DEF = [
    {"chrom": "15", "pos": 28513871, "ref": "A", "alt": "G"},
    {"chrom": "15", "pos": 28230318, "ref": "C", "alt": "T"},
    {"chrom": "15", "pos": 28514000, "ref": "G", "alt": "A"},
    {"chrom": "15", "pos": 28514100, "ref": "T", "alt": "C"},
    {"chrom": "15", "pos": 28514200, "ref": "A", "alt": "C"},
]


def _variant_id(v: dict) -> str:
    return f"{v['chrom']}_{v['pos']}_{v['ref']}_{v['alt']}"


@pytest.fixture
def sample_metadata() -> SampleMetadata:
    return SampleMetadata(
        population=POPULATIONS,
        phenotype_labels=PHENOTYPE_LABELS,
    )


@pytest.fixture
def raw_dataset(sample_metadata: SampleMetadata) -> RawSnpDataset:
    # s10 is missing on all variants to exercise imputation paths
    genotypes = {
        s: "0|0" if i % 3 == 0 else ("0|1" if i % 3 == 1 else "1|1")
        for i, s in enumerate(SAMPLE_IDS)
    }
    genotypes["s10"] = None  # missing for every variant

    variants = [
        RawVariant(
            chrom=v["chrom"],
            pos=v["pos"],
            ref=v["ref"],
            alt=v["alt"],
            genotypes=dict(genotypes),
        )
        for v in VARIANTS_DEF
    ]
    return RawSnpDataset(samples=SAMPLE_IDS, variants=variants, metadata=sample_metadata)


@pytest.fixture
def clean_dataset(sample_metadata: SampleMetadata) -> CleanSnpDataset:
    genotypes = {
        s: i % 3 for i, s in enumerate(SAMPLE_IDS)
    }
    variants = [
        CleanVariant(
            variant_id=_variant_id(v),
            genotypes=dict(genotypes),
        )
        for v in VARIANTS_DEF
    ]
    # 8 train / 2 test, matching 80/20 split of 10 samples
    splits = {s: ("test" if s in {"s09", "s10"} else "train") for s in SAMPLE_IDS}
    return CleanSnpDataset(
        samples=SAMPLE_IDS,
        variants=variants,
        metadata=sample_metadata,
        imputation_medians={_variant_id(v): 1.0 for v in VARIANTS_DEF},
        sample_splits=splits,
    )


@pytest.fixture
def feature_registry() -> FeatureRegistry:
    return FeatureRegistry(
        features=[
            FeatureEntry(
                column_index=i,
                variant_id=_variant_id(v),
                chrom=v["chrom"],
                pos=v["pos"],
                ref=v["ref"],
                alt=v["alt"],
                maf=0.25,
            )
            for i, v in enumerate(VARIANTS_DEF)
        ]
    )


@pytest.fixture
def feature_matrix(feature_registry: FeatureRegistry) -> FeatureMatrix:
    rng = np.random.default_rng(42)
    n_samples, n_features = len(SAMPLE_IDS), len(VARIANTS_DEF)
    X = rng.integers(0, 3, size=(n_samples, n_features)).astype(float)
    # phenotype indices: 0=blue (s01-s04), 1=brown (s05-s08), 2=green (s09-s10)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    splits = ["train"] * 8 + ["test"] * 2
    return FeatureMatrix(
        X=X,
        y=y,
        sample_ids=SAMPLE_IDS,
        splits=splits,
        registry=feature_registry,
    )


@pytest.fixture
def minimal_vcf_bytes() -> bytes:
    header = (
        b"##fileformat=VCFv4.1\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\n"
    )
    row = b"15\t28513871\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\n"
    return header + row


@pytest.fixture
def multi_sample_vcf_bytes() -> bytes:
    header = (
        b"##fileformat=VCFv4.1\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\tsample2\n"
    )
    row = b"15\t28513871\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|1\n"
    return header + row
