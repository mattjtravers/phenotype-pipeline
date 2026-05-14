# Deployment

## Context and Design Philosophy

This component owns all AWS infrastructure concerns: S3 bucket structure, SageMaker training job configuration, Lambda inference wrapper, and IAM policies.

The two AWS execution environments have distinct roles:
- **SageMaker Training Job** — canonical environment for training; runs `pipeline/train.py` in a custom container
- **Lambda** — inference only; loads the trained model artifact from S3 and serves predictions

The deployment layer is a thin wrapper. It does not contain ML logic. It adapts the pipeline's interfaces to these two AWS execution environments.

## S3 Bucket Structure

All pipeline data lives under a single S3 bucket (name configurable via environment variable `PHENO_S3_BUCKET`):

```
s3://{bucket}/
  data/
    raw/          ← downloaded 1000 Genomes VCF files and sample metadata
  models/
    {run_id}/
      model.json
      feature_registry.json
      imputation_medians.json
      evaluation_report.json
      label_encoder.json
  logs/
    training/
    inference/
```

`run_id` is a timestamp + short UUID (e.g., `20240115-a3f2c1`). This enables multiple model versions to coexist without overwriting.

## SageMaker Training Job

Training runs as a SageMaker Training Job using a custom Docker container (built from the project's `Dockerfile`). The container entry point calls `pipeline/train.py`.

Key configuration:
- **Instance type**: `ml.m5.2xlarge` (8 vCPU, 32 GB RAM) — sufficient for 1000 Genomes scale
- **Input channels**: S3 URIs for raw data and (optionally) cached processed data
- **Output path**: S3 model artifact path
- **Environment variables**: pipeline parameters (k-fold k, MAF threshold, N markers, etc.)

SageMaker is used for training only. Inference runs in Lambda (see below).

## Lambda Execution

The prediction component runs as a Lambda function:
- **Runtime**: Python 3.12 (container image Lambda to include XGBoost)
- **Trigger**: API Gateway (REST endpoint) or S3 event (batch mode)
- **Memory**: 3008 MB (XGBoost model loading requires headroom)
- **Timeout**: 60 seconds
- **Environment variables**: `PHENO_S3_BUCKET`, `MODEL_RUN_ID`

The Lambda handler downloads the model artifact from S3 on cold start and caches it in `/tmp` for warm invocations.

## IAM Policies

Two roles:

**SageMaker Training Role**
- `s3:GetObject` on `s3://1000genomes/*` (public bucket)
- `s3:PutObject`, `s3:GetObject` on `s3://{bucket}/*`
- `logs:CreateLogGroup`, `logs:PutLogEvents`

**Lambda Execution Role**
- `s3:GetObject` on `s3://{bucket}/models/*`
- `logs:CreateLogGroup`, `logs:PutLogEvents`

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|---|---|---|---|
| Training execution | SageMaker Training Job | SageMaker Pipelines, EC2, local only | Training Jobs are the simplest managed compute option; Pipelines adds orchestration complexity unnecessary for a demo |
| Inference execution | Lambda (container image) | SageMaker endpoint, ECS | Lambda is serverless and cost-effective for low-frequency demo predictions; SageMaker endpoint is always-on and expensive |
| Local dev S3 | Local filesystem via `PHENO_LOCAL_DATA_DIR` env var | LocalStack, moto mocking | Filesystem adapter is simple and transparent; LocalStack/moto add unnecessary dependencies for a demo |
| Run ID format | Timestamp + UUID | Incrementing integer, git SHA | Timestamps sort naturally; UUIDs prevent collisions; git SHA would require a git context in SageMaker |

## Open Questions & Future Decisions

### Resolved
1. ✅ SageMaker Training Job for training, Lambda for inference — one process per component, no dual-mode switching
2. ✅ Lambda chosen over SageMaker endpoint for inference (serverless, cost-effective for low-frequency demo use)

### Deferred
1. API Gateway authentication (API key vs Cognito) — out of scope for SDD demo
2. CI/CD pipeline for container image builds and SageMaker job triggers

## References

- `docs/llds/01_data-ingestion.md` — S3 read patterns
- `docs/llds/04_model-training.md` — artifact bundle written to S3
- `docs/llds/05_prediction.md` — artifact bundle read from S3, results written to S3
