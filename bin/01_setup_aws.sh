#!/usr/bin/env bash
# Creates the S3 bucket and SageMaker IAM role required by the pipeline.
# Usage: export PHENO_S3_BUCKET=<bucket-name> && bash bin/01_setup_aws.sh
# On success, prints an export statement for PHENO_TRAINING_ROLE_ARN — run it before 03_train.sh.
set -euo pipefail

: "${PHENO_S3_BUCKET:?Set PHENO_S3_BUCKET to your desired S3 bucket name before running this script}"

ROLE_NAME="phenotype-sagemaker-role"
REGION="us-east-1"

echo "Creating S3 bucket: s3://${PHENO_S3_BUCKET}"
aws s3 mb "s3://${PHENO_S3_BUCKET}" --region "$REGION" --no-cli-pager

echo "Creating IAM role: ${ROLE_NAME}"
aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' \
    --no-cli-pager > /dev/null

aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess \
    --no-cli-pager

aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name phenotype-s3-access \
    --policy-document "$(printf '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": "arn:aws:s3:::1000genomes/*"
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": "arn:aws:s3:::%s/*"
            }
        ]
    }' "$PHENO_S3_BUCKET")" \
    --no-cli-pager

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --no-cli-pager)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

echo ""
echo "ACTION REQUIRED — add this statement to your developer IAM policy so bin/02_ingest.sh can write to the bucket:"
printf '{
    "Sid": "ProjectBucketDataAccess",
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
    "Resource": [
        "arn:aws:s3:::%s",
        "arn:aws:s3:::%s/*"
    ]
}\n' "$PHENO_S3_BUCKET" "$PHENO_S3_BUCKET"

echo ""
echo "Setup complete. Run the following before 03_train.sh:"
echo "  export PHENO_TRAINING_ROLE_ARN=${ROLE_ARN}"
