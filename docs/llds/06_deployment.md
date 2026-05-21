# Deployment

## Context and Design Philosophy

This component owns all AWS infrastructure concerns: S3 bucket structure, SageMaker training job configuration, Lambda inference wrapper, and IAM policies.

The two AWS execution environments have distinct roles:
- **SageMaker Training Job** — canonical environment for training; runs `genomic_ancestry_pipeline.sagemaker_train` in a custom container built from `Dockerfile.train`
- **Lambda** — inference only; loads the trained model artifact from S3 and serves predictions; packaged as a container image built from `Dockerfile`

The deployment layer is a thin wrapper. It does not contain ML logic. It adapts the pipeline's interfaces to these two AWS execution environments.

**Region.** All AWS resources (S3 bucket, ECR repository, SageMaker training jobs, Lambda + API Gateway stack) live in **`us-east-1`**. The SAM stack region, the SageMaker SDK's session region, and the S3 bucket region must all match — the launcher and `samconfig.toml` both pin `us-east-1` explicitly; cross-region deployment is not supported.

**Failure philosophy.** This component fails loudly. There is no silent retry, no degraded mode, no fallback artifact. When training fails, deployment halts. When inference cannot load a model, it returns a 5xx error with a clear log line, not a stale prediction. The cost of a noisy failure (one bad request) is much smaller than the cost of a silently wrong prediction.

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

## Container Images

The pipeline uses two distinct container images — one for SageMaker training, one for Lambda inference. They share Python 3.12 and the same `src/genomic_ancestry_pipeline/` package source but have incompatible base images and entry-point contracts.

| Image | Dockerfile | Base image | Entry point | Published to ECR by |
|---|---|---|---|---|
| Training | `Dockerfile.train` | `python:3.12-slim` | `python -m genomic_ancestry_pipeline.sagemaker_train` | `bin/03_train.sh` (before each training run) |
| Inference | `Dockerfile` | `public.ecr.aws/lambda/python:3.12` | `genomic_ancestry_pipeline.deployment.lambda_handler` | `sam build && sam deploy` (step 4) |

**Training image** (`Dockerfile.train`): installs runtime ML dependencies (XGBoost, scikit-learn, pandas, NumPy, Pydantic, boto3), copies `src/genomic_ancestry_pipeline/` to `/opt/ml/code/genomic_ancestry_pipeline/`, and sets `PYTHONPATH=/opt/ml/code`. Built and pushed to the ECR repository `phenotype-pipeline-training` by `bin/03_train.sh`; the resulting URI is exported as `PHENO_TRAINING_IMAGE_URI` for the training launcher.

**Inference image** (`Dockerfile`): built from the AWS Lambda Python 3.12 base, which bundles the Lambda Runtime Interface Client as `ENTRYPOINT`. Copies `src/genomic_ancestry_pipeline/` to `${LAMBDA_TASK_ROOT}`. Does not include `sagemaker` or `streamlit` (training and UI dependencies not needed at inference time). Published automatically by `sam build && sam deploy`.

The two images are kept separate because Lambda requires the Lambda RIC as `ENTRYPOINT` and SageMaker requires a training script as `CMD`; combining them into one image would require entry-point switching logic and obscure both contracts.

## SageMaker Training Job

Training runs as a SageMaker Training Job using the custom training container built from `Dockerfile.train`. The container entry point is `genomic_ancestry_pipeline.sagemaker_train` (invoked as `python -m genomic_ancestry_pipeline.sagemaker_train`), which reads pipeline parameters from the environment variables injected by the launcher and orchestrates the full training pipeline in sequence: `load_raw_dataset` → `preprocess` → `build_feature_matrix` → `train` → `save_artifact`.

Key configuration:
- **Instance type**: `ml.m5.2xlarge` (8 vCPU, 32 GB RAM) — sufficient for 1000 Genomes scale
- **Input channels**: S3 URIs for raw data (the container reads directly from S3 via boto3; the SageMaker channel is declared but not used for local file access)
- **Output path**: S3 model artifact path
- **Environment variables**: `PHENO_S3_BUCKET`, `MODEL_RUN_ID`, `K_FOLDS`, `MAF_THRESHOLD`, `TOP_N`, `RANDOM_STATE`

SageMaker is used for training only. Inference runs in Lambda (see below).

### Training Launch

Training jobs are launched programmatically via the **SageMaker Python SDK** (`sagemaker` package), not raw `boto3` and not CloudFormation. A dedicated launcher script `src/genomic_ancestry_pipeline/launch_training.py` is the single entry point for kicking off a training run.

Before invoking the launcher, `bin/03_train.sh` builds `Dockerfile.train`, pushes the resulting image to the ECR repository `phenotype-pipeline-training` in `us-east-1`, and exports the image URI as `PHENO_TRAINING_IMAGE_URI`. The launcher reads this variable as the container image URI passed to `Estimator`.

The launcher:

1. Parses CLI flags for hyperparameters (k-fold k, MAF threshold, N markers, random seed) and for SageMaker instance type (default `ml.m5.2xlarge`).
2. Generates `run_id` locally before submitting the job (timestamp + short UUID, per `DEPLOY-BE-002`) and passes it to the container as an environment variable. The launcher owns `run_id` so that the S3 output path is known *before* the job starts — needed for downstream automation and for the launcher's stdout contract.
3. **Checks for `run_id` collision**: issues `HeadObject` (or `ListObjectsV2` with prefix) against `s3://{PHENO_S3_BUCKET}/models/{run_id}/`. If anything exists at that prefix, the launcher exits non-zero with a clear error rather than risk overwriting a prior run.
4. Instantiates `sagemaker.estimator.Estimator` configured with:
   - the training container image URI from `PHENO_TRAINING_IMAGE_URI` (ECR)
   - `instance_type=<--instance-type flag>` (default `ml.m5.2xlarge`), `instance_count=1`
   - `output_path='s3://{PHENO_S3_BUCKET}/models/'` (SageMaker appends a job-name prefix; the container itself writes the final artifact bundle under `models/{run_id}/`)
   - `environment={...}` carrying both hyperparameters and `MODEL_RUN_ID`
   - region pinned to `us-east-1`
5. Calls `estimator.fit(inputs={'raw': '<s3 uri>'}, wait=True)` and **blocks until the job reaches terminal status** (`Completed`, `Failed`, or `Stopped`). Synchronous launch trades a long-blocking process (~30 min typical) for a simple operator UX: `sam deploy --parameter-overrides MODEL_RUN_ID=$(launch_training.py ...)` is safe because the launcher only returns its `run_id` on completion.
6. On `Completed`, emits the `run_id` to stdout (and the SageMaker job name to stderr) — this is what CI/CD or the deploy step consumes to bind a Lambda to the trained model. On `Failed` or `Stopped`, exits non-zero without emitting a `run_id`.
7. Fails fast (non-zero exit) if AWS credentials are missing or the bucket is unreachable, before attempting to call `Estimator.fit`.

**Training job failure semantics.** If a submitted training job fails (SageMaker `TrainingJobStatus = Failed` or `Stopped`), the pipeline halts: no automatic retry, no partial-artifact cleanup, no downstream `sam deploy` (the launcher's non-zero exit prevents shell-piping into `sam deploy`). Operator inspects CloudWatch logs, decides whether to relaunch with a fresh `run_id`. The partial S3 prefix at `models/{run_id}/` is left in place as a forensic record; the next launcher run uses a new `run_id` and will not collide.

The launcher runs from a developer machine or CI; it is **not packaged inside the SAM stack** (SAM deploys the inference Lambda; training launch is a separate operational step).

## Lambda Execution

The prediction component runs as a Lambda function:
- **Runtime**: Python 3.12 (container image Lambda to include XGBoost)
- **Trigger**: API Gateway **HTTP API** endpoint (REST API is rejected — see § API routes)
- **Memory**: 3008 MB (XGBoost model loading requires headroom)
- **Timeout**: 60 seconds
- **Environment variables**: `PHENO_S3_BUCKET`, `MODEL_RUN_ID`

The pipeline trains exactly one model. The Lambda handler loads the model artifact from the S3 path identified by the `MODEL_RUN_ID` environment variable, set once at deploy time. The artifact is cached in `/tmp` on cold start and reused for warm invocations. No model selection logic is needed.

### API routes

The Lambda is fronted by an **API Gateway HTTP API** (not REST API) — HTTP API is materially cheaper and lower-latency; we use none of REST API's extra features.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Runs inference on a single-sample VCF; returns a `PredictionResult` JSON body |
| `GET`  | `/labels`  | Returns the deployed model's phenotype label list as JSON `{"labels": [...]}` |

The `/labels` endpoint reads `label_encoder.json` from the cached artifact (or loads the artifact if cold). It exists so the Streamlit UI can populate its phenotype dropdown from the deployed model without hardcoding labels in the UI image. The response is the sorted list of label strings (the integer keys are an internal detail of training and are not exposed).

### Error contract

The handler fails immediately and loudly on any error. There is no retry, no degraded mode, no fallback artifact. Every failure path returns a JSON body `{"error": "<short_code>", "detail": "<human_readable>"}` and emits a `logger.error` line carrying the same code and detail plus the request id.

| Condition | HTTP status | `error` code |
|---|---|---|
| Model artifact at `s3://{bucket}/models/{MODEL_RUN_ID}/` is missing, incomplete (any of the bundled files absent), or unreadable on cold-start load | `503` | `MODEL_UNAVAILABLE` |
| `/predict` request body is not valid JSON, missing required fields, or fails Pydantic validation | `400` | `INVALID_INPUT` |
| `/predict` VCF parsing fails (malformed VCF, no usable variants for the model's feature set) | `400` | `INVALID_VCF` |
| `/predict` VCF contains more than one sample (per `PRED-PROC-012`) | `400` | `INVALID_VCF` |
| `PredictionResult` Pydantic validation fails (per `PRED-PROC-010`) | `500` | `INFERENCE_FAILED` |
| Model inference or SHAP computation raises | `500` | `INFERENCE_FAILED` |
| Any other unhandled exception | `500` | `INTERNAL_ERROR` |

Model-bundle completeness is checked on the first load (cold start or first `/labels`/`/predict` call): all of `model.json`, `feature_registry.json`, `imputation_medians.json`, `evaluation_report.json`, `label_encoder.json` must be present. If any are missing, the handler raises and every subsequent request in that container lifecycle short-circuits to `503 MODEL_UNAVAILABLE` without re-attempting the load.

### Infrastructure-as-Code (AWS SAM)

The inference Lambda, API Gateway routes, and execution role are declared in an **AWS SAM** template at the repo root. SAM is used for the inference side only — SageMaker training launches are imperative (see § Training Launch above) and intentionally not part of the SAM stack.

Files:

| Path | Purpose |
|---|---|
| `template.yaml` | SAM template: Lambda function (container image), API Gateway routes, execution role |
| `samconfig.toml` | CLI defaults: stack name, region, deployment parameters |

Template structure (summary):

- `AWS::Serverless::Function` — `PackageType: Image`, `MemorySize: 3008`, `Timeout: 60`, env vars `PHENO_S3_BUCKET` and `MODEL_RUN_ID` sourced from CloudFormation parameters
- `AWS::Serverless::HttpApi` (NOT `AWS::Serverless::Api`) — HTTP API is materially cheaper and lower-latency, and we use none of REST API's extra features
- Two `Events` of type `HttpApi`: `POST /predict` and `GET /labels`
- An `AWS::IAM::Role` granting `s3:GetObject` on `s3://{PHENO_S3_BUCKET}/models/*` and CloudWatch Logs write access (matching the existing `DEPLOY-BE-012` policy shape)
- An `AWS::Logs::LogGroup` for the Lambda with `RetentionInDays: 14` — default ("never expire") is a silent cost trap; 14 days is plenty for a demo's forensic window

Workflow:

| Command | Purpose |
|---|---|
| `sam validate` | Lints the template (CI-runnable, no AWS calls) |
| `sam build` | Builds the Lambda container image |
| `sam deploy` | Pushes image to ECR + deploys/updates the CloudFormation stack |
| `sam local invoke` / `sam local start-api` | Runs the Lambda locally inside Docker — useful for smoke-testing the handler against a real model artifact in S3 (requires AWS credentials and a valid `MODEL_RUN_ID`) |

**Binding a model to the deployment.** `MODEL_RUN_ID` is a CloudFormation *parameter* of the SAM stack, not an output of the training job. After a SageMaker training run completes successfully, the operator (or a CI step) runs `sam deploy --parameter-overrides MODEL_RUN_ID=<run_id>` to bind the inference Lambda to the new artifact. This keeps SAM as the single source of truth for *what's deployed* without coupling the deploy lifecycle to the training lifecycle.

**Concurrent or rapid re-deploys.** When two `sam deploy` calls land in close succession with different `MODEL_RUN_ID` values, the second wins at the CloudFormation level, but warm Lambda containers from the prior version may continue to serve requests against their cached `/tmp` model until they are recycled (typically minutes). This is accepted behavior — the brief overlap is bounded, and forcing a hard cutover would require explicit container kill or alias shifting that is out of scope.

## IAM Policies

Two roles:

**SageMaker Training Role**
- `s3:GetObject` on `s3://1000genomes/*` (public bucket)
- `s3:PutObject`, `s3:GetObject` on `s3://{bucket}/*`
- `logs:CreateLogGroup`, `logs:PutLogEvents`
- ECR pull on `phenotype-pipeline-training` (`ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`, `ecr:BatchCheckLayerAvailability`) — provided by the `AmazonSageMakerFullAccess` managed policy attached in `bin/01_setup_aws.sh`

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
| Inference IaC | AWS SAM | AWS CDK, Terraform, plain boto3 scripts | SAM is the industry-standard declarative tool for the Lambda + API Gateway shape; ships with `sam local invoke` for in-Docker local execution; CDK is heavier for a single function; Terraform would not gain us anything for a one-stack project |
| Training launch interface | SageMaker Python SDK (`Estimator`) | Raw boto3 `create_training_job`, SAM/CFN `AWS::SageMaker::TrainingJob`, Step Functions | SDK is the canonical AWS-blessed way to launch training; CFN resource is rarely used in practice because training is imperative/one-shot; Step Functions adds orchestration we don't need yet |
| Container image strategy | Two separate images (`Dockerfile.train` for SageMaker, `Dockerfile` for Lambda) | Single multi-purpose image with entry-point switching | Lambda requires the Lambda RIC as `ENTRYPOINT`; SageMaker requires a training `CMD`. A shared image would need runtime switching logic and would obscure both contracts; separate images keep each boundary explicit and independently deployable |
| `run_id` ownership | Generated by the launcher before `Estimator.fit` | Generated inside the training container | Launcher needs to know the output S3 path before the job starts (for stdout contract + downstream `sam deploy` binding); container would not be able to publish `run_id` synchronously |
| Model→Lambda binding | `sam deploy --parameter-overrides MODEL_RUN_ID=<run_id>` after a successful training run | SSM Parameter Store lookup at Lambda cold start, "latest" S3 pointer, dynamic discovery | Explicit binding keeps the CFN stack as the source of truth for what's deployed; avoids hidden runtime coupling between training and inference |

## Open Questions & Future Decisions

### Resolved
1. ✅ SageMaker Training Job for training, Lambda for inference — one process per component, no dual-mode switching
2. ✅ Lambda chosen over SageMaker endpoint for inference (serverless, cost-effective for low-frequency demo use)
3. ✅ Single training run; model version is fixed at deploy time via `MODEL_RUN_ID` — no model selection UI or versioning logic needed
4. ✅ SAM owns inference IaC; SageMaker Python SDK owns training launch; the two halves are intentionally separate tools

### Deferred
1. API Gateway authentication (API key vs Cognito) — out of scope for SDD demo
2. End-to-end CI/CD pipeline (build container → push to ECR → launch training → run `sam deploy --parameter-overrides`) — the pieces exist as standalone commands; wiring them into a CI workflow is a future step

## References

- `docs/llds/01_data-ingestion.md` — S3 read patterns
- `docs/llds/04_model-training.md` — artifact bundle written to S3
- `docs/llds/05_prediction.md` — artifact bundle read from S3, results written to S3
