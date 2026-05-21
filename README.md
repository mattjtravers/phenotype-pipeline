# Phenotype Pipeline — Spec-Driven Genomic Machine Learning

## Overview

The Phenotype Pipeline is an automated machine learning workflow designed to predict observable human traits from raw single-nucleotide polymorphism (SNP) genomic data. It establishes an engineering pipeline that translates structured biological requirements into a reproducible model training and serverless inference architecture on AWS.

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
            │ (Validated Numpy Matrices / Pydantic Contracts)
            ▼
┌───────────────────────┐
│  SageMaker Training   │ ──► Executes XGBoost training with cross-validation
└───────────┬───────────┘
            │ (Model Artifact to Amazon S3)
            ▼
┌───────────────────────┐
│  AWS SAM Deployment   │ ──► Configures API Gateway and AWS Lambda environment
└───────────┬───────────┘
            │ (Secure REST Endpoint)
            ▼
 [Streamlit Dashboard]   ──► Renders traits, predictive confidence, and SNP tags

```

The system coordinates data from ingestion to deployment. Each state transition is managed by strict interface definitions, ensuring data validation occurs prior to handoffs between local compute and AWS infrastructure components.

---

## The Spec-Driven Pattern

This project utilizes **Linked-Intent Development (LID)** to reduce configuration drift, enforce strict boundary runtime validation, and maintain alignment between requirements and implementation.

* **Requirements Traceability via EARS:** System logic is structured around the Easy Approach to Requirements Syntax (EARS) framework. Every engineering decision maps back to a documented product behavior in the specifications directory, allowing verification of system intent through targeted integration testing.
* **Deterministic Boundary Validation:** Untyped arrays and loose JSON structures are rejected at component entry points. The pipeline relies on Pydantic models to validate genomic features, sample indices, and inference payloads at runtime. Malformed genomic inputs trigger controlled, fast-failing validations before reaching downstream training algorithms or hosting resources.
* **Infrastructure as Code (IaC) via AWS SAM:** Cloud infrastructure definitions are maintained inside version-controlled template files. This setup bypasses manual AWS Console configurations, ensuring consistency between local testing environments and deployed staging resources.
* **Production Traceability & Observability:** Predictions are output alongside their calculated confidence metrics and the localized identification tags of the specific SNPs that influenced the classification model. The software architecture routes standardized log signatures directly to standard output and error streams, allowing performance tracking in AWS CloudWatch without custom parsing scripts.

---

## Quick Start

