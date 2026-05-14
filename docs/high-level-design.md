# High-Level Design: Phenotype Pipeline

## Problem

 Building machine learning pipelines can be a siloed process — business analysts, data scientists, and software engineers each maintain separate artifacts with no enforced traceability between intent and implementation. This project, predicting observable human phenotypes (e.g., eye color) from raw SNP (single nucleotide polymorphism) data, addresses both the ML prediction task and the process problem by using [Linked-Intent Development (LID)](https://github.com/jszmajda/lid) to unify intent from requirements through production code in a single "Spec-to-Production" lifecycle.

## Approach

- **Input**: Raw SNP data from the public [1000 Genomes Project](https://www.internationalgenome.org/) dataset
- **Model**: XGBoost (gradient boosted decision trees)
- **Preprocessing**: Median imputation for missing genotype values
- **Validation**: K-fold cross-validation
- **Output**: Phenotype prediction + confidence score, traceable to specific genomic markers
- **Evaluation**: Confusion matrix and F1-score

## Target Users

Three personas collaborate on this pipeline:

| Persona | Domain | Role |
|---|---|---|
| **Business Analyst** | Intent | Defines phenotype targets, interprets confidence scores, requires marker traceability |
| **Data Scientist** | Rigor | Owns model selection, validation methodology, and evaluation metrics |
| **Software Engineer** | Integrity | Owns infrastructure (AWS S3/Lambda/SageMaker), data contracts (Pydantic), and test coverage (EARS-driven TDD) |

## Goals

- Predict human phenotypes from SNP data with per-prediction confidence scores
- Trace every prediction to the contributing genomic markers
- Enforce data schemas and validation at every pipeline stage via Pydantic
- Demonstrate a single "Spec-to-Production" LID lifecycle that replaces compartmentalized ML development

## Non-Goals

- Real-time / low-latency inference (batch pipeline only)
- Support for omics data types beyond SNPs (e.g., RNA-seq, methylation) — deferred
- Clinical diagnostic use (research and demonstration scope only)
- SageMaker-native managed services as a dev dependency — AWS is the deployment target, not a local dev requirement
- Production-hardened UI (Streamlit is a demo interface, not a production web app)

## System Design

```
1. Data Ingestion      — fetch SNP data from S3 (1000 Genomes)
2. Preprocessing       — median imputation, Pydantic schema validation
3. Feature Engineering — SNP encoding, marker selection
4. Model Training      — XGBoost with k-fold cross-validation
5. Evaluation          — confusion matrix, F1-score reporting
6. Prediction          — phenotype label + confidence score + marker contribution trace
7. Deployment          — S3 storage, SageMaker training job, Lambda execution, IAM
8. UI                  — Streamlit frontend; Business Analyst submits SNP data and views predictions
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| ML algorithm | Open-source XGBoost | Handles sparse SNP features well; produces per-sample SHAP values for marker traceability; custom SageMaker container gives full control over training logic and artifact format |
| Missing data | Median imputation | Simple, robust baseline for genotype missingness |
| Validation strategy | K-fold cross-validation | Prevents overfitting on the relatively small 1000 Genomes cohort |
| Schema enforcement | Pydantic | Catches malformed data at every pipeline boundary |
| Infrastructure | AWS S3 + Lambda/SageMaker | Scalable serverless execution; standard for genomics workloads |
| UI framework | Streamlit | Minimal code; locally runnable; suitable for demo use by Business Analyst persona without a full web app |
| Development process | LID (EARS → Tests → Code) | Collapses BA/DS/Eng silos into one traceable artifact chain |

## Success Metrics

- F1-score on held-out test set reported per phenotype class
- 100% of predictions include a confidence score
- 100% of predictions traceable to contributing SNP markers
- All EARS specs have corresponding passing tests

## References

- [1000 Genomes Project](https://www.internationalgenome.org/)
- [Linked-Intent Development](https://github.com/jszmajda/lid)
- [README.md](../README.md) — persona constraints and vision statement
