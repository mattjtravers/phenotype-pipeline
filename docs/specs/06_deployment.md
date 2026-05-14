# Deployment Specs

## S3 Bucket Structure

- [ ] **DEPLOY-BE-001**: All pipeline data shall be stored under a single S3 bucket configured via the `PHENO_S3_BUCKET` environment variable.
- [ ] **DEPLOY-BE-002**: The system shall assign each training run a unique `run_id` composed of a UTC timestamp and a short UUID (e.g., `20240115-a3f2c1`), used as the S3 key prefix for model artifacts and prediction results.
- [ ] **DEPLOY-BE-003**: The S3 bucket shall follow the structure: `data/raw/` for source VCF and metadata, `models/{run_id}/` for model artifact bundles, and `logs/` for training and inference logs.

## SageMaker Training Job

- [ ] **DEPLOY-BE-004**: The SageMaker Training Job shall use the project's custom Docker container (built from `Dockerfile`) as its execution environment.
- [ ] **DEPLOY-BE-005**: The SageMaker Training Job shall receive pipeline parameters (k-fold k, MAF threshold, marker count N, random seed) as environment variables.
- [ ] **DEPLOY-BE-006**: The SageMaker Training Job shall write the model artifact bundle to the S3 output path corresponding to the run's `run_id`.

## Lambda Inference

- [ ] **DEPLOY-BE-007**: The inference Lambda function shall use a container image runtime with Python 3.12, including XGBoost and all pipeline dependencies.
- [ ] **DEPLOY-BE-008**: The Lambda function shall be allocated 3008 MB memory and a 60-second timeout.
- [ ] **DEPLOY-BE-009**: The Lambda function shall cache the model artifact bundle in `/tmp` after the first load, and reuse the cached bundle on subsequent warm invocations within the same container lifecycle.
- [ ] **DEPLOY-BE-010**: The Lambda function shall accept invocation via API Gateway (REST) and S3 event trigger (batch mode).
- [ ] **DEPLOY-BE-013**: The Lambda function shall expose a `GET /labels` API Gateway route that returns the deployed model's phenotype label strings (sourced from the cached artifact's `label_encoder.json`) as a JSON body of shape `{"labels": [...]}`.

## IAM

- [ ] **DEPLOY-BE-011**: The SageMaker Training Role shall have `s3:GetObject` on `s3://1000genomes/*` and `s3:GetObject` / `s3:PutObject` on `s3://{PHENO_S3_BUCKET}/*`.
- [ ] **DEPLOY-BE-012**: The Lambda Execution Role shall have `s3:GetObject` on `s3://{PHENO_S3_BUCKET}/models/*`.
