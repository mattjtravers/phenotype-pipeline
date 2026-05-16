#!/usr/bin/env bash
# Builds the Lambda container image and deploys the SAM stack.
# sam deploy will prompt for confirmation before creating resources.
# On success, prints an export statement for PHENO_API_ENDPOINT — run it before 05_ui.sh.
set -euo pipefail

: "${RUN_ID:?Set RUN_ID before running this script (see 03_train.sh output)}"
: "${PHENO_S3_BUCKET:?Set PHENO_S3_BUCKET before running this script}"

sam build

sam deploy \
    --parameter-overrides \
        ModelRunId="$RUN_ID" \
        PhenoS3Bucket="$PHENO_S3_BUCKET"

ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name phenotype-inference \
    --region us-east-1 \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' \
    --output text \
    --no-cli-pager)

echo ""
echo "Deployment complete. Run the following before 05_ui.sh:"
echo "  export PHENO_API_ENDPOINT=${ENDPOINT}"
