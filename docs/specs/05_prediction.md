# Prediction Specs

## Artifact Loading

- [x] **PRED-PROC-001**: The system shall load the model artifact bundle (as defined in TRAIN-DATA-001) from S3 before running inference.
- [x] **PRED-PROC-002**: The system shall record the S3 bundle prefix of the artifact used in every `PredictionResult` under `model_artifact_version`, in the form `models/{run_id}/` (e.g. `models/20240115-a3f2c1/`).

## Input Validation

- [x] **PRED-PROC-012**: If the input VCF contains more than one sample, then the system shall raise a structured error and shall not produce a prediction.

## Inference Preprocessing

- [x] **PRED-PROC-003**: At inference time, the system shall impute missing genotype values using the `imputation_medians` from the loaded artifact bundle, without refitting on the input sample.
- [x] **PRED-PROC-004**: At inference time, the system shall select and order features using the `FeatureRegistry` from the loaded artifact bundle, without refitting marker selection. Variants present in the registry but absent from the input VCF shall be imputed to their stored training median.
- [x] **PRED-PROC-013**: At inference time, variants present in the input VCF but absent from the loaded `FeatureRegistry` shall be silently dropped before the feature vector is assembled — the model has no weight for them and they cannot affect the prediction.

## Prediction Output

- [x] **PRED-PROC-005**: The system shall produce a predicted phenotype label for each input sample by applying the loaded XGBoost model to the preprocessed feature vector.
- [x] **PRED-PROC-006**: The system shall produce a confidence score for each prediction, defined as the probability assigned by `predict_proba()` to the predicted class (i.e. the maximum class probability), in the range [0.0, 1.0].
- [x] **PRED-DATA-001**: The system shall include the full class probability distribution (all classes, mapped to human-readable labels via `label_encoder.json`) in the `PredictionResult`.

## Marker Traceability

- [x] **PRED-PROC-007**: The system shall compute per-sample SHAP contribution values using XGBoost's built-in `predict(X, pred_contribs=True)`, producing a contribution score per feature per sample.
- [x] **PRED-PROC-008**: The system shall return the top-20 features by absolute SHAP contribution (configurable N) for each prediction, mapped to genomic coordinates via the `FeatureRegistry`.
- [x] **PRED-DATA-002**: Each marker contribution in the `PredictionResult` shall include: variant ID, chromosome, position, reference allele, alternate allele, signed SHAP contribution value, and rank.

## Output Validation

- [x] **PRED-PROC-009**: The system shall validate the `PredictionResult` using its Pydantic schema before returning it to the caller.
- [x] **PRED-PROC-010**: If Pydantic validation of the `PredictionResult` fails, then the system shall raise a structured error without returning a partial result.
