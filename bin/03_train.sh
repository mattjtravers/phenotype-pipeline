#!/usr/bin/env bash
# Builds and pushes the training container to ECR, then launches a SageMaker
# training job (~30 min on ml.m5.2xlarge).
# On success, prints an export statement for RUN_ID — run it before 04_deploy.sh.
set -euo pipefail

: "${PHENO_S3_BUCKET:?Set PHENO_S3_BUCKET before running this script}"
: "${PHENO_TRAINING_ROLE_ARN:?Set PHENO_TRAINING_ROLE_ARN before running this script (see 01_setup_aws.sh output)}"

REGION="us-east-1"
REPO_NAME="phenotype-pipeline-training"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --no-cli-pager)
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"

# ── Build and push training container ─────────────────────────────────────────
echo "Ensuring ECR repository: ${REPO_NAME}"
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" \
    --no-cli-pager > /dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION" \
        --no-cli-pager > /dev/null

echo "Authenticating Docker to ECR"
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin \
        "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "Building training image"
docker build --provenance=false -f Dockerfile.train -t "$IMAGE_URI" .

echo "Pushing training image: ${IMAGE_URI}"
docker push "$IMAGE_URI"

export PHENO_TRAINING_IMAGE_URI="$IMAGE_URI"

# ── Launch SageMaker training job ─────────────────────────────────────────────
echo "Launching SageMaker training job (~30 min on ml.m5.2xlarge)"
RUN_ID=$(LOG_LEVEL=WARNING uv run python -m genomic_ancestry_pipeline.launch_training \
    --bucket "$PHENO_S3_BUCKET" \
    --instance-type ml.m5.2xlarge \
    --k-folds 5 \
    --top-n 10000 \
    | tail -n 1)

echo ""
echo "Training complete. Artifact at: s3://${PHENO_S3_BUCKET}/models/${RUN_ID}/"
echo "Run the following before 04_deploy.sh:"
echo "  export RUN_ID=${RUN_ID}"
