# Genomic Ancestry Pipeline — Spec-Driven Machine Learning

## Overview

The Genomic Ancestry Pipeline is an automated machine learning workflow that predicts ancestral population from raw single-nucleotide polymorphism (SNP) genomic data. It establishes an end-to-end pipeline that translates structured biological requirements into a reproducible model training and serverless inference architecture on AWS.

---

## Live Application Dashboard

<a href="https://genomic-ancestry-pipeline.streamlit.app"><img src="assets/open-demo-button.svg" alt="Open Demo Dashboard"/></a>

> **Note on Hosting Site:** If the application dashboard is asleep, click the presented button to wake it up. Please allow 30–60 seconds for the container to spin up.

Upload a VCF file (or pick one of the bundled samples) and click **Run Prediction** to see the predicted ancestral population with a confidence score and top contributing genomic markers.

---

## Architectural Blueprint

```text
[Raw SNP Data / VCF Input]
            │
            ▼
┌───────────────────────┐
│   Data Ingestion      │ ──► Validates file headers and chromosomal coordinates
└───────────┬───────────┘
            │ (Clean Data Stream)
            ▼
┌───────────────────────┐
│ Feature Engineering   │ ──► Executes median imputation & quality-score filtering
└───────────┬───────────┘
            │ (Feature Matrices → Amazon S3)
            ▼
┌───────────────────────┐
│  SageMaker Training   │ ──► Executes XGBoost training with cross-validation
└───────────┬───────────┘
            │ (Model Artifact → Amazon S3)
            ▼
┌───────────────────────┐
│  AWS SAM Deployment   │ ──► Configures API Gateway and AWS Lambda environment
└───────────┬───────────┘
            │ (Public REST Endpoint)
            ▼
 [Streamlit Dashboard]   ──► Renders ancestral population prediction, confidence, and top SNP markers

```

The system coordinates data from ingestion to deployment. Each state transition is managed by strict interface definitions, ensuring data validation occurs prior to handoffs between local compute and AWS infrastructure components.

---

## Features

* **Spec-Driven Ingestion Pipelines:** Translates structural biological specifications into deterministic Python data pipelines, using Pydantic to enforce runtime schema validation and eliminate silent data corruption.
* **Production AWS Infrastructure via IaC:** Provisions and manages the entire cloud footprint—including API Gateway, AWS Lambda, and IAM roles—using AWS SAM templates to guarantee zero configuration drift.
* **Automated Cloud Training Workflows:** Coordinates data handoffs from local environments to AWS SageMaker for structured XGBoost training, utilizing Amazon S3 as a secure feature and model artifact store.
* **Enterprise Observability & Logging:** Integrates structured logging directly into the Python application runtime, enabling native AWS CloudWatch monitoring and alerting without custom parsing overhead.


