from __future__ import annotations

from phenotype_pipeline.models import RawSnpDataset


class IngestError(Exception):
    pass


def run_etl(
    source_bucket: str,
    dest_bucket: str,
    chromosomes: list[int],
) -> None:
    """One-time ETL: streams and filters VCF from source S3 to project S3."""
    raise NotImplementedError


def load_raw_dataset(
    bucket: str,
    local_data_dir: str | None = None,
) -> RawSnpDataset:
    """Streams the project VCF from S3 (or local dir) and emits a RawSnpDataset."""
    raise NotImplementedError


def is_biallelic_snp(vcf_line: str) -> bool:
    """Returns True iff the VCF data line represents a bi-allelic SNP."""
    raise NotImplementedError
