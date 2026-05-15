"""Data ingestion: one-time ETL from public 1000 Genomes S3 + runtime VCF loader."""
from __future__ import annotations

import csv
import gzip
import io
import logging
from typing import IO

import boto3

from phenotype_pipeline.models import RawSnpDataset, RawVariant, SampleMetadata


class IngestError(Exception):
    """Raised when ingestion encounters malformed or missing data."""


_VCF_DEST_KEY = "data/raw/1000genomes.vcf.gz"
_METADATA_DEST_KEY = "data/raw/sample_info.tsv"
_SOURCE_METADATA_KEY = "technical/working/20130606_sample_info/20130606_sample_info.txt"
_VCF_REQUIRED_COLS = (
    "CHROM",
    "POS",
    "ID",
    "REF",
    "ALT",
    "QUAL",
    "FILTER",
    "INFO",
    "FORMAT",
)
_MISSING_GENOTYPES = frozenset({".", "./.", ".|."})

logger = logging.getLogger(__name__)


# @spec INGEST-PROC-002
def is_biallelic_snp(vcf_line: str) -> bool:
    """Returns True iff the VCF data line represents a bi-allelic SNP."""
    if not vcf_line or vcf_line.startswith("#"):
        return False
    cols = vcf_line.split("\t")
    if len(cols) < 5:
        return False
    ref, alt = cols[3], cols[4]
    if len(ref) != 1 or len(alt) != 1:
        return False
    return ref.isalpha() and alt.isalpha()


# @spec INGEST-PROC-001, INGEST-PROC-003
def run_etl(
    source_bucket: str,
    dest_bucket: str,
    chromosomes: list[int],
) -> None:
    """One-time ETL: streams and filters VCF from source S3 to project S3.

    Args:
        source_bucket: Public 1000 Genomes S3 bucket to read from.
        dest_bucket: Project S3 bucket to write filtered VCF and metadata.
        chromosomes: Chromosome numbers to include in the filtered VCF.
    """
    s3 = boto3.client("s3")
    logger.info(
        "run_etl start [source_bucket=%s dest_bucket=%s chromosomes=%s]",
        source_bucket, dest_bucket, chromosomes,
    )

    out_lines: list[str] = []
    header_written = False
    for chrom in chromosomes:
        source_key = (
            f"release/20130502/ALL.chr{chrom}."
            "phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
        )
        obj = s3.get_object(Bucket=source_bucket, Key=source_key)
        text = _read_vcf_body(obj["Body"])
        for line in text.splitlines(keepends=True):
            if line.startswith("##"):
                if not header_written:
                    out_lines.append(line)
                continue
            if line.startswith("#CHROM"):
                if not header_written:
                    out_lines.append(line)
                    header_written = True
                continue
            if is_biallelic_snp(line.rstrip("\n")):
                out_lines.append(line)

    filtered_bytes = gzip.compress("".join(out_lines).encode("utf-8"))
    s3.put_object(Bucket=dest_bucket, Key=_VCF_DEST_KEY, Body=filtered_bytes)

    meta = s3.get_object(Bucket=source_bucket, Key=_SOURCE_METADATA_KEY)
    s3.put_object(
        Bucket=dest_bucket,
        Key=_METADATA_DEST_KEY,
        Body=meta["Body"].read(),
    )
    logger.info("run_etl complete [dest_bucket=%s]", dest_bucket)


# @spec INGEST-PROC-004, INGEST-PROC-005, INGEST-PROC-006, INGEST-PROC-007, INGEST-DATA-001
def load_raw_dataset(
    bucket: str,
    local_data_dir: str | None = None,
) -> RawSnpDataset:
    """Stream the project VCF and metadata from S3 and return a RawSnpDataset.

    Args:
        bucket: Project S3 bucket containing the VCF and metadata files.
        local_data_dir: Reserved for future local-path support; currently unused.

    Returns:
        RawSnpDataset containing the parsed sample list, variant records, and
        sample metadata.

    Raises:
        IngestError: If either the VCF or metadata file cannot be fetched from
            S3, or if the VCF fails to parse.
    """
    s3 = boto3.client("s3")

    try:
        vcf_obj = s3.get_object(Bucket=bucket, Key=_VCF_DEST_KEY)
        vcf_text = _read_vcf_body(vcf_obj["Body"])
    except IngestError:
        raise
    except Exception as e:
        logger.error(
            "load_raw_dataset vcf_fetch failed [bucket=%s key=%s]: %s",
            bucket, _VCF_DEST_KEY, e,
        )
        raise IngestError(
            f"Failed to fetch VCF from s3://{bucket}/{_VCF_DEST_KEY}: {e}"
        ) from e

    samples, variants = _parse_vcf(vcf_text)

    try:
        meta_obj = s3.get_object(Bucket=bucket, Key=_METADATA_DEST_KEY)
        meta_bytes = meta_obj["Body"].read()
    except IngestError:
        raise
    except Exception as e:
        logger.error(
            "load_raw_dataset metadata_fetch failed [bucket=%s key=%s]: %s",
            bucket, _METADATA_DEST_KEY, e,
        )
        raise IngestError(
            f"Failed to fetch metadata from s3://{bucket}/{_METADATA_DEST_KEY}: {e}"
        ) from e

    metadata = _parse_metadata(meta_bytes)
    return RawSnpDataset(samples=samples, variants=variants, metadata=metadata)


def _read_vcf_body(body: IO[bytes]) -> str:
    """Read, decompress if gzip, and decode a VCF S3 body to a UTF-8 string."""
    raw = body.read()
    if isinstance(raw, str):
        return raw
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def _parse_vcf(text: str) -> tuple[list[str], list[RawVariant]]:
    """Parse VCF text into an ordered sample list and a list of RawVariant objects."""
    samples: list[str] = []
    variants: list[RawVariant] = []
    seen_columns = False

    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            cols = line.lstrip("#").split("\t")
            if (
                len(cols) < len(_VCF_REQUIRED_COLS)
                or tuple(cols[: len(_VCF_REQUIRED_COLS)]) != _VCF_REQUIRED_COLS
            ):
                raise IngestError(
                    f"Malformed VCF header at line {lineno}: "
                    f"expected columns {_VCF_REQUIRED_COLS}"
                )
            samples = cols[len(_VCF_REQUIRED_COLS):]
            seen_columns = True
            continue
        if not seen_columns:
            raise IngestError(
                f"Malformed VCF: data line at {lineno} before #CHROM header"
            )
        cols = line.split("\t")
        if len(cols) < 9:
            raise IngestError(
                f"Malformed VCF data at line {lineno}: "
                f"expected at least 9 columns, got {len(cols)}"
            )
        chrom, pos_str, _id, ref, alt = cols[:5]
        try:
            pos = int(pos_str)
        except ValueError as e:
            raise IngestError(f"Malformed POS at line {lineno}: {pos_str!r}") from e
        sample_genotypes: dict[str, str | None] = {}
        for sid, gt_field in zip(samples, cols[9:]):
            gt = gt_field.split(":")[0]
            sample_genotypes[sid] = None if gt in _MISSING_GENOTYPES else gt
        variants.append(
            RawVariant(
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                genotypes=sample_genotypes,
            )
        )

    if not seen_columns:
        raise IngestError("Malformed VCF: missing #CHROM column header")

    return samples, variants


def _parse_metadata(raw: bytes) -> SampleMetadata:
    """Parse TSV metadata bytes into a SampleMetadata object."""
    population: dict[str, str] = {}
    phenotype_labels: dict[str, str] = {}

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return SampleMetadata(population=population, phenotype_labels=phenotype_labels)

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    fields = reader.fieldnames or []
    sample_col = _pick_col(fields, {"sample", "sample_id", "sampleid"})
    pop_col = _pick_col(fields, {"population", "pop"})
    pheno_col = _pick_col(
        fields, {"phenotype", "phenotype_label", "eye_color", "phenotype_eye_color"}
    )
    if sample_col:
        for row in reader:
            sid = (row.get(sample_col) or "").strip()
            if not sid:
                continue
            if pop_col and (pop_val := row.get(pop_col)):
                population[sid] = pop_val.strip()
            if pheno_col and (pheno_val := row.get(pheno_col)):
                phenotype_labels[sid] = pheno_val.strip()

    return SampleMetadata(population=population, phenotype_labels=phenotype_labels)


def _pick_col(fields: list[str], candidates: set[str]) -> str | None:
    """Return the first field name whose lowercase form is in candidates, or None."""
    for field in fields:
        if field.lower() in candidates:
            return field
    return None
