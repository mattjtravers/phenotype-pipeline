# Deployment Specs

## S3 Bucket Structure

- [ ] **DEPLOY-BE-001**: All pipeline data shall be stored under a single S3 bucket configured via the `PHENO_S3_BUCKET` environment variable.
- [ ] **DEPLOY-BE-002**: The system shall assign each training run a unique `run_id` composed of a UTC timestamp and a short UUID (e.g., `20240115-a3f2c1`), used as the S3 key prefix for model artifacts and prediction results.
- [ ] **DEPLOY-BE-003**: The S3 bucket shall follow the structure: `data/raw/` for source VCF and metadata, `models/{run_id}/` for model artifact bundles, and `logs/` for training and inference logs.

## Container Images

- [ ] **DEPLOY-BE-023**: The Lambda inference container image shall be built from `Dockerfile.predict` using `public.ecr.aws/lambda/python:3.12` as the base; all runtime Python dependencies (XGBoost, scikit-learn, pandas, NumPy, Pydantic, boto3, requests) shall be installed; the `genomic_ancestry_pipeline` package shall be copied to `${LAMBDA_TASK_ROOT}`; the Lambda handler entry point shall be `genomic_ancestry_pipeline.deployment.lambda_handler`.
- [ ] **DEPLOY-BE-024**: The SageMaker training container shall provide an executable `train` script on `PATH` (at `/opt/ml/code/train`) that invokes `python -m genomic_ancestry_pipeline.sagemaker_train`; SageMaker BYOC calls `docker run <image> train`, overriding `CMD`, so the named executable is required. The entry point module shall read `PHENO_S3_BUCKET`, `MODEL_RUN_ID`, `K_FOLDS`, `MAF_THRESHOLD`, `TOP_N`, and `RANDOM_STATE` from environment variables and orchestrate the full training pipeline in sequence: `load_raw_dataset` → `preprocess` → `build_feature_matrix` → `train` → `save_artifact`.
- [ ] **DEPLOY-BE-025**: Before each SageMaker training job submission, `bin/03_train.sh` shall build `Dockerfile.train`, push the resulting image to the ECR repository `phenotype-pipeline-training` in `us-east-1`, and export the image URI as `PHENO_TRAINING_IMAGE_URI`; the training launcher shall read `PHENO_TRAINING_IMAGE_URI` as the container image URI passed to `sagemaker.estimator.Estimator`.
- [ ] **DEPLOY-BE-026**: The SageMaker Training Role shall have ECR pull permissions (`ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`, `ecr:BatchCheckLayerAvailability`) on the `phenotype-pipeline-training` repository; these shall be satisfied by the `AmazonSageMakerFullAccess` managed policy attached in `bin/01_setup_aws.sh`.

## SageMaker Training Job

- [ ] **DEPLOY-BE-004**: The SageMaker Training Job shall use the project's custom training Docker container (built from `Dockerfile.train`) as its execution environment.
- [ ] **DEPLOY-BE-005**: The SageMaker Training Job shall receive pipeline parameters (k-fold k, MAF threshold, marker count N, random seed) as environment variables.
- [ ] **DEPLOY-BE-006**: The SageMaker Training Job shall write the model artifact bundle to the S3 output path corresponding to the run's `run_id`.

## Lambda Inference

- [ ] **DEPLOY-BE-007**: The inference Lambda function shall use a container image runtime with Python 3.12, including XGBoost and all pipeline dependencies.
- [ ] **DEPLOY-BE-008**: The Lambda function shall be allocated 3008 MB memory and a 60-second timeout.
- [ ] **DEPLOY-BE-009**: The Lambda function shall cache the model artifact bundle in `/tmp` after the first load, and reuse the cached bundle on subsequent warm invocations within the same container lifecycle.
- [ ] **DEPLOY-BE-010**: The Lambda function shall accept invocation via an API Gateway HTTP API endpoint.
- [ ] **DEPLOY-BE-027**: The `POST /predict` Lambda route shall accept a JSON request body of shape `{"vcf": "<vcf_string>"}` where `vcf` is the raw VCF content as a UTF-8 string; no `phenotype` field is accepted or required — the model determines the ancestral population from the VCF data alone.

## IAM

- [ ] **DEPLOY-BE-011**: The SageMaker Training Role shall have `s3:GetObject` on `s3://1000genomes/*` and `s3:GetObject` / `s3:PutObject` on `s3://{PHENO_S3_BUCKET}/*`.
- [ ] **DEPLOY-BE-012**: The Lambda Execution Role shall have `s3:GetObject` on `s3://{PHENO_S3_BUCKET}/models/*`.

## Infrastructure-as-Code (SAM)

- [ ] **DEPLOY-BE-014**: The inference Lambda, its HTTP API routes, its execution role, and its CloudWatch log group shall be declared in a SAM template at the repo root and deployable end-to-end via `sam build && sam deploy`.
- [ ] **DEPLOY-BE-015**: The SAM template shall expose `MODEL_RUN_ID` and `PHENO_S3_BUCKET` as CloudFormation stack parameters, surfaced to the Lambda as environment variables of the same name.
- [ ] **DEPLOY-BE-020**: The SAM template shall declare an `AWS::Logs::LogGroup` for the inference Lambda with `RetentionInDays: 14`.

## Training Launcher

- [ ] **DEPLOY-BE-016**: SageMaker training jobs shall be launched via `pipeline/launch_training.py` using `sagemaker.estimator.Estimator`; the script shall accept k-fold k, MAF threshold, marker count N, random seed, and SageMaker instance type as CLI flags (instance type defaulting to `ml.m5.2xlarge`); the script shall block until the training job reports terminal status and shall emit the resulting `run_id` to stdout only on successful completion.
- [ ] **DEPLOY-BE-017**: The training launcher shall exit non-zero before calling `Estimator.fit` if AWS credentials are missing or the configured S3 bucket is unreachable.
- [ ] **DEPLOY-BE-018**: The training launcher shall check `s3://{PHENO_S3_BUCKET}/models/{run_id}/` for any existing object before submission and shall exit non-zero with a clear error if the prefix is occupied.

## Region

- [ ] **DEPLOY-BE-019**: All AWS resources (SAM stack, SageMaker training jobs, S3 bucket reads/writes, ECR repository) shall be configured for the `us-east-1` region; the launcher and `samconfig.toml` shall pin the region explicitly rather than rely on `AWS_DEFAULT_REGION`.

## Inference Error Contract

- [ ] **DEPLOY-BE-021**: On its first model load (cold start or first request), the Lambda shall verify that every file in the artifact bundle defined by `TRAIN-DATA-001` is present and readable at `s3://{PHENO_S3_BUCKET}/models/{MODEL_RUN_ID}/`; if any file is missing or unreadable, every request in that container lifecycle shall short-circuit to a `503 MODEL_UNAVAILABLE` response without re-attempting the load.
- [ ] **DEPLOY-BE-022**: Every Lambda error response shall carry the HTTP status defined in the LLD's `§ Error contract` table (`503 MODEL_UNAVAILABLE`, `400 INVALID_INPUT`, `400 INVALID_VCF`, `500 INFERENCE_FAILED`, `500 INTERNAL_ERROR`), a JSON body of shape `{"error": <code>, "detail": <message>}`, and a `logger.error` line carrying the same code, the detail, and the AWS request id.
