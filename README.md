# Spec-Driven Analysis to Predict Human Phenotypes

---

## Vision Statement
Leverage Spec-Driven Development to collapse the traditional silos between **Business, Data Science, and Engineering** by using the SDD **Linked-Intent Development (LID)** framework to drive the autonomous generation of a genomic machine learning pipeline. This project demonstrates how a single "Spec-to-Production" lifecycle can replace the compartmentalized development of legacy ML workflows.

---

## Overview
The success of the pipeline depends on integrating the specific requirements and constraints of three key professional personas within a single specification which is subsequently implemented by the **AI-Orchestrator**.

### Persona-Specific Constraints

| Persona | Domain | Core Constraints & Requirements |
| --- | --- | --- |
| **Business Analyst** | **Intent** | • Predict human phenotypes (e.g., eye color) from raw SNP data.<br>• Use the public **1000 Genomes Project** dataset.<br>• Every result must include a **Confidence Score**.<br>• Trace results back to specific genomic markers. |
| **Data Scientist** | **Rigor** | • Implement **XGBoost** (Gradient Boosted Decision Trees).<br>• Use **k-fold cross-validation** and median imputation for missing data.<br>• Generate a confusion matrix and F1-score for model evaluation. |
| **Software Engineer** | **Integrity** | • Architect for **AWS Services** (S3 storage, Lambda/SageMaker execution).<br>• Enforce all data schemas and validation using **Pydantic**.<br>• Follow strict **TDD** (Test-Driven Development) using EARS-defined unit tests. |

---

## Quick Start

All AWS resources must be in **`us-east-1`**. Scripts live in [`bin/`](bin/).

### Prerequisites

- AWS account with credentials configured (`aws configure` or environment variables)
- Docker (required for `sam build` container image packaging)
- [uv](https://docs.astral.sh/uv/) package manager

```bash
uv sync
```

---

### 1. Create AWS resources

Creates the S3 bucket (raw data, model artifacts, logs) and the SageMaker IAM training role.

```bash
export PHENO_S3_BUCKET=<YOUR_BUCKET>
bash bin/01_setup_aws.sh
```

The script prints an `export PHENO_TRAINING_ROLE_ARN=...` line — run that export before the next step.

---

### 2. Ingest data (one-time)

Streams chromosome 15 (OCA2/HERC2 eye-color region) from the public 1000 Genomes S3 bucket into your project bucket. Run once; rerun only to refresh the source data.

```bash
bash bin/02_ingest.sh
```

---

### 3. Train the model

Launches a SageMaker training job (~30 min on `ml.m5.2xlarge`). The model artifact bundle is written to `s3://<YOUR_BUCKET>/models/<run_id>/`.

```bash
bash bin/03_train.sh
```

The script prints an `export RUN_ID=...` line — run that export before the next step.

---

### 4. Deploy the inference Lambda

Builds the Lambda container image and deploys the SAM stack (API Gateway + Lambda). `sam deploy` will prompt for confirmation before creating resources.

```bash
bash bin/04_deploy.sh
```

The script prints an `export PHENO_API_ENDPOINT=...` line — run that export before the next step.

---

### 5. Run the UI

```bash
bash bin/05_ui.sh
```

Open [http://localhost:8501](http://localhost:8501), upload a single-sample `.vcf` file, select a phenotype, and click **Predict**.

