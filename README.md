# Genomic Ancestry Pipeline — Spec-Driven Machine Learning

## Overview

The Genomic Ancestry Pipeline is an automated machine learning workflow that predicts ancestral population from raw single-nucleotide polymorphism (SNP) genomic data. It establishes an end-to-end pipeline that translates structured biological requirements into a reproducible model training and serverless inference architecture on AWS.

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

## The Spec-Driven Pattern

This project utilizes [Linked-Intent Development (LID)](https://github.com/jszmajda/lid) to reduce configuration drift, enforce strict boundary runtime validation, and maintain alignment between requirements and implementation.

* **Requirements Traceability via EARS:** System logic is structured around the Easy Approach to Requirements Syntax (EARS) framework. Every engineering decision maps back to a documented product behavior in the specifications directory, allowing verification of system intent through targeted integration testing.
* **Deterministic Boundary Validation:** Untyped arrays and loose JSON structures are rejected at component entry points. The pipeline relies on Pydantic models to validate genomic features, sample indices, and inference payloads at runtime. Malformed genomic inputs trigger controlled, fast-failing validations before reaching downstream training algorithms or hosting resources.
* **Infrastructure as Code (IaC) via AWS SAM:** Cloud infrastructure definitions are maintained inside version-controlled template files. This setup bypasses manual AWS Console configurations, ensuring consistency between local testing environments and deployed staging resources.
* **Production Traceability & Observability:** Predictions are output alongside their calculated confidence metrics and the localized identification tags of the specific SNPs that influenced the classification model. The software architecture routes standardized log signatures directly to standard output and error streams, allowing performance tracking in AWS CloudWatch without custom parsing scripts.

---

## Quick Start

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://genomic-ancestry-pipeline.streamlit.app)

Click the badge above or open **[genomic-ancestry-pipeline.streamlit.app](https://genomic-ancestry-pipeline.streamlit.app)** directly. Upload a VCF file (or pick one of the bundled samples) and click **Run Prediction** to see the predicted ancestral population with a confidence score and top contributing genomic markers.

> **Run locally** — export `PHENO_API_ENDPOINT=https://zynpjy3gyk.execute-api.us-east-1.amazonaws.com` then run `bash bin/05_ui.sh`.

