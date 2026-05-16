#!/usr/bin/env bash
# Launches a SageMaker training job (~30 min) and captures the run_id.
# On success, prints an export statement for RUN_ID — run it before 04_deploy.sh.
set -euo pipefail

: "${PHENO_S3_BUCKET:?Set PHENO_S3_BUCKET before running this script}"
: "${PHENO_TRAINING_ROLE_ARN:?Set PHENO_TRAINING_ROLE_ARN before running this script (see 01_setup_aws.sh output)}"

RUN_ID=$(LOG_LEVEL=WARNING uv run python -m phenotype_pipeline.launch_training \
    --bucket "$PHENO_S3_BUCKET" \
    --instance-type ml.m5.2xlarge \
    --k-folds 5 \
    --top-n 10000 \
    | tail -n 1)

echo ""
echo "Training complete. Artifact at: s3://${PHENO_S3_BUCKET}/models/${RUN_ID}/"
echo "Run the following before 04_deploy.sh:"
echo "  export RUN_ID=${RUN_ID}"
