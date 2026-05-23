"""Tests for data ingestion — INGEST-* specs."""
from __future__ import annotations

import gzip
import io
from unittest.mock import patch

import pytest

from genomic_ancestry_pipeline.ingest import (
    IngestError,
    is_biallelic_snp,
    load_raw_dataset,
    run_etl,
)
from genomic_ancestry_pipeline.models import RawSnpDataset

# ── minimal VCF fragments used across tests ────────────────────────────────────

_VCF_HEADER = (
    "##fileformat=VCFv4.1\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts01\ts02\n"
)
_BIALLELIC_ROW = "15\t28513871\t.\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|1\n"
_MULTIALLELIC_ROW = "15\t28513872\t.\tA\tG,T\t.\tPASS\t.\tGT\t0|0\t0|1\n"
_INDEL_ROW = "15\t28513873\t.\tAT\tG\t.\tPASS\t.\tGT\t0|0\t0|1\n"
_SV_ROW = "15\t28513874\t.\tA\t<DEL>\t.\tPASS\t.\tGT\t0|0\t0|1\n"


def _vcf_bytes(*extra_rows: str) -> bytes:
    return (_VCF_HEADER + "".join(extra_rows)).encode()


def _vcf_gz_bytes(*extra_rows: str) -> bytes:
    """Gzip-compressed VCF — matches the format run_etl reads from the source bucket."""
    return gzip.compress((_VCF_HEADER + "".join(extra_rows)).encode())


# ── ETL tests ─────────────────────────────────────────────────────────────────


# @spec INGEST-PROC-001
def test_etl_copies_vcf_to_project_s3():
    """ETL writes the filtered VCF to s3://{dest_bucket}/data/raw/1000genomes.vcf.gz."""
    with patch("genomic_ancestry_pipeline.ingest.boto3") as mock_boto3:
        mock_s3 = mock_boto3.client.return_value
        mock_s3.get_object.return_value = {"Body": io.BytesIO(_vcf_gz_bytes(_BIALLELIC_ROW))}
        mock_s3.get_object.side_effect = None

        run_etl(source_bucket="1000genomes", dest_bucket="my-bucket", chromosomes=[15])

        put_calls = mock_s3.put_object.call_args_list + mock_s3.upload_fileobj.call_args_list
        dest_keys = [str(c) for c in put_calls]
        assert any("data/raw/1000genomes.vcf.gz" in k for k in dest_keys)


# @spec INGEST-PROC-002
@pytest.mark.parametrize(
    "vcf_line, expected",
    [
        (_BIALLELIC_ROW.strip(), True),
        (_MULTIALLELIC_ROW.strip(), False),
        (_INDEL_ROW.strip(), False),
        (_SV_ROW.strip(), False),
    ],
)
def test_biallelic_snp_filter(vcf_line: str, expected: bool):
    """is_biallelic_snp correctly classifies SNPs, multiallelic sites, indels, and SVs."""
    assert is_biallelic_snp(vcf_line) is expected


# @spec INGEST-PROC-003
def test_etl_copies_sample_metadata_tsv():
    """ETL copies the sample metadata TSV to s3://{dest_bucket}/data/raw/sample_info.tsv."""
    with patch("genomic_ancestry_pipeline.ingest.boto3") as mock_boto3:
        mock_s3 = mock_boto3.client.return_value
        mock_s3.get_object.return_value = {"Body": io.BytesIO(_vcf_gz_bytes(_BIALLELIC_ROW))}

        run_etl(source_bucket="1000genomes", dest_bucket="my-bucket", chromosomes=[15])

        all_calls = str(mock_s3.method_calls)
        assert "sample_info.tsv" in all_calls


# ── Runtime loader tests ───────────────────────────────────────────────────────


# @spec INGEST-PROC-004
def test_runtime_loader_streams_vcf_not_full_download():
    """load_raw_dataset uses get_object (streaming), not download_file."""
    with patch("genomic_ancestry_pipeline.ingest.boto3") as mock_boto3:
        mock_s3 = mock_boto3.client.return_value
        mock_s3.get_object.return_value = {"Body": io.BytesIO(_vcf_bytes(_BIALLELIC_ROW))}

        load_raw_dataset(bucket="my-bucket")

        mock_s3.get_object.assert_called()
        mock_s3.download_file.assert_not_called()


# @spec INGEST-PROC-005, INGEST-DATA-001
def test_load_raw_dataset_returns_valid_raw_snp_dataset():
    """load_raw_dataset returns a RawSnpDataset with samples, variants, and metadata."""
    with patch("genomic_ancestry_pipeline.ingest.boto3") as mock_boto3:
        mock_s3 = mock_boto3.client.return_value
        mock_s3.get_object.return_value = {"Body": io.BytesIO(_vcf_bytes(_BIALLELIC_ROW))}

        result = load_raw_dataset(bucket="my-bucket")

    assert isinstance(result, RawSnpDataset)
    assert len(result.samples) >= 1
    assert len(result.variants) >= 1
    assert result.metadata is not None


# @spec INGEST-PROC-006
def test_malformed_vcf_raises_ingest_error_before_any_records():
    """Structurally malformed VCF raises IngestError; no partial records emitted."""
    malformed = io.BytesIO(b"##fileformat=VCFv4.1\nnot_a_vcf_line\n")
    with patch("genomic_ancestry_pipeline.ingest.boto3") as mock_boto3:
        mock_boto3.client.return_value.get_object.return_value = {"Body": malformed}

        with pytest.raises(IngestError) as exc_info:
            load_raw_dataset(bucket="my-bucket")

    assert exc_info.value is not None


# @spec INGEST-PROC-007
def test_missing_metadata_raises_ingest_error():
    """Unreadable metadata file raises IngestError before a RawSnpDataset is returned."""
    with patch("genomic_ancestry_pipeline.ingest.boto3") as mock_boto3:
        mock_s3 = mock_boto3.client.return_value

        def _get_object(**kwargs):
            if "sample_info" in kwargs.get("Key", ""):
                raise Exception("NoSuchKey")
            return {"Body": io.BytesIO(_vcf_bytes(_BIALLELIC_ROW))}

        mock_s3.get_object.side_effect = _get_object

        with pytest.raises(IngestError):
            load_raw_dataset(bucket="my-bucket")
