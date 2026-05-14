# Prediction Specs

## Artifact Loading

- [ ] **PRED-PROC-001**: The system shall load the model artifact bundle (as defined in TRAIN-DATA-001) from S3 before running inference.
- [ ] **PRED-PROC-002**: The system shall record the S3 key of the artifact bundle used in every `PredictionResult`, for traceability.

## Inference Preprocessing

- [ ] **PRED-PROC-003**: At inference time, the system shall impute missing genotype values using the `imputation_medians` from the loaded artifact bundle, without refitting on the input sample.
- [ ] **PRED-PROC-004**: At inference time, the system shall select and order features using the `FeatureRegistry` from the loaded artifact bundle, without refitting marker selection.

## Prediction Output

- [ ] **PRED-PROC-005**: The system shall produce a predicted phenotype label for each input sample by applying the loaded XGBoost model to the preprocessed feature vector.
- [ ] **PRED-PROC-006**: The system shall produce a confidence score for each prediction, defined as the maximum class probability from `predict_proba()`, in the range [0.0, 1.0].
- [ ] **PRED-DATA-001**: The system shall include the full class probability distribution (all classes, mapped to human-readable labels via `label_encoder.json`) in the `PredictionResult`.

## Marker Traceability

- [ ] **PRED-PROC-007**: The system shall compute per-sample SHAP contribution values using XGBoost's built-in `predict(X, pred_contribs=True)`, producing a contribution score per feature per sample.
- [ ] **PRED-PROC-008**: The system shall return the top-20 features by absolute SHAP contribution (configurable N) for each prediction, mapped to genomic coordinates via the `FeatureRegistry`.
- [ ] **PRED-DATA-002**: Each marker contribution in the `PredictionResult` shall include: variant ID, chromosome, position, reference allele, alternate allele, signed SHAP contribution value, and rank.

## Output Validation

- [ ] **PRED-PROC-009**: The system shall validate the `PredictionResult` using its Pydantic schema before returning it to the caller.
- [ ] **PRED-PROC-010**: If Pydantic validation of the `PredictionResult` fails, then the system shall raise a structured error without returning a partial result.
