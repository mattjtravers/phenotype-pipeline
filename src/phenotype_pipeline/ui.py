from __future__ import annotations

from phenotype_pipeline.models import PredictionResult


def validate_vcf_upload(
    filename: str,
    file_bytes: bytes,
    size_bytes: int,
) -> list[str]:
    """Returns a list of validation error messages; empty list means valid."""
    raise NotImplementedError


def count_vcf_samples(vcf_bytes: bytes) -> int:
    """Returns the number of samples declared in the VCF header."""
    raise NotImplementedError


def dispatch_prediction(
    vcf_bytes: bytes,
    phenotype: str,
    api_endpoint: str,
) -> PredictionResult:
    """POSTs to the Lambda API Gateway endpoint and returns the PredictionResult."""
    raise NotImplementedError


def fetch_phenotype_labels(api_endpoint: str) -> list[str]:
    """GETs {api_endpoint}/labels and returns the supported phenotype labels."""
    raise NotImplementedError
