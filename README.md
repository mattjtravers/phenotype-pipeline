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

