# Phenotype Pipeline — Spec-Driven Genomic Machine Learning

## The Pitch

The Phenotype Pipeline is an automated machine learning workflow that predicts observable human traits (e.g., eye color) from raw SNP genomic data while guaranteeing precise confidence scores and marker traceability. It leverages Linked-Intent Development (LID) to collapse the traditional silos between Business, Data Science, and Engineering into a single, traceable "Spec-to-Production" lifecycle.

---

## Architectural Blueprint

```text
[Raw SNP Data (1000 Genomes Project)]
          │
          ▼
┌───────────────────┐
│   Preprocessing   │ (Median Imputation & Marker Selection)
└─────────┬─────────┘ (Strict Pydantic Schema Validation)
          │
          ▼
┌───────────────────┐
│ SageMaker Training│ (XGBoost & K-Fold Cross-Validation)
└─────────┬─────────┘ (S3 Model Artifact Storage)
          │
          ▼
┌───────────────────┐
│ Serverless Deploy │ (AWS Lambda + API Gateway via AWS SAM)
└─────────┬─────────┘ (Emits Phenotype, Confidence Score, & Marker Trace)
          │
          ▼
 [Streamlit Web UI]   (Interactive Business Analyst Dashboard)

```

Data flows strictly from local acquisition and preparation to distributed cloud execution. Each boundary enforces data integrity before passing artifacts to AWS serverless infrastructure for highly available inference.

---

## The Spec-Driven Pattern

This project implements **Linked-Intent Development (LID)** to guarantee zero-maintenance stability, complete execution predictability, and exact alignment across diverse professional personas.

* **Living Specifications via LID & EARS:** Rather than relying on disconnected Jira tickets, requirements are defined using the EARS (Easy Approach to Requirements Syntax) framework. This ensures the Business Analyst's intent, the Data Scientist's mathematical rigor, and the Engineer's infrastructure constraints are permanently linked and validated via Test-Driven Development (TDD).
* **Instant Hallucination & Data Trapping:** Pydantic `BaseModel` schemas enforce strict data contracts at every pipeline boundary. Malformed genomic arrays or missing features are intercepted at the type level instantly, preventing silent corruption during distributed SageMaker training or Lambda inference.
* **Enterprise Cloud Integrity (IaC):** Serverless architecture is deployed predictably via AWS SAM (Serverless Application Model). By treating infrastructure as code, the pipeline avoids manual AWS Console configurations and guarantees reproducible environments.
* **Production Observability & Traceability:** Black-box ML models are unacceptable in production. Every XGBoost inference explicitly outputs a confidence score and traces the prediction back to the contributing SNP markers. Furthermore, the pipeline emits structured, aggregator-ready logs to stdout/stderr, ensuring any operational failure explains exactly *what* failed and *why* without interactive debugging.

---

## Quick Start

