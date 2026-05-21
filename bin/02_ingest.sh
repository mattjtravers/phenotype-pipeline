#!/usr/bin/env bash
# One-time ETL: streams chromosome 15 from the public 1000 Genomes S3 bucket
# into the project bucket. Run once; rerun only to refresh the source data.
# Writes: s3://$PHENO_S3_BUCKET/data/raw/1000genomes.vcf.gz
#         s3://$PHENO_S3_BUCKET/data/raw/sample_info.tsv
set -euo pipefail

: "${PHENO_S3_BUCKET:?Set PHENO_S3_BUCKET before running this script}"

uv run python -c "
from genomic_ancestry_pipeline.ingest import run_etl
run_etl(
    source_bucket='1000genomes',
    dest_bucket='${PHENO_S3_BUCKET}',
    chromosomes=[15],
)
"
