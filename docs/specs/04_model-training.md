# Model Training Specs

## Execution Environment

- [x] **TRAIN-BE-001**: Model training shall run exclusively as a SageMaker Training Job using a custom Docker container built from `Dockerfile.train`, containing the full pipeline stack; the container entry point shall be `genomic_ancestry_pipeline.sagemaker_train` (see `DEPLOY-BE-004`, `DEPLOY-BE-024`, `DEPLOY-BE-025`).
- [x] **TRAIN-BE-002**: The SageMaker Training Job shall use `ml.m5.2xlarge` as its default instance type, overridable via the launcher's `--instance-type` CLI flag (see `DEPLOY-BE-016`).

## Training Protocol

- [x] **TRAIN-PROC-001**: The system shall train an XGBoost classifier on the training-split samples of the `FeatureMatrix` (identified by `split == "train"` as defined in PREP-DATA-001).
- [x] **TRAIN-PROC-002**: The system shall evaluate the model using stratified k-fold cross-validation with k=5 (configurable), applied to the training split only.
- [x] **TRAIN-PROC-003**: Each cross-validation fold shall use early stopping with a patience of 20 rounds evaluated on the fold's validation subset.
- [x] **TRAIN-PROC-004**: After cross-validation, the system shall retrain the final model on all training-split samples using the mean `n_estimators` across folds derived from early stopping.

## Evaluation

- [x] **TRAIN-PROC-005**: The system shall compute per-class F1-score and a confusion matrix for each cross-validation fold.
- [x] **TRAIN-PROC-006**: The system shall compute mean and standard deviation of macro F1-score across all folds and include them in the evaluation report.

## Final Test-Set Evaluation

- [x] **TRAIN-PROC-007**: After cross-validation is complete and the final model has been retrained on all training-split samples (TRAIN-PROC-004), the system shall evaluate it exactly once on the held-out test split (as defined in PREP-PROC-001), producing per-class F1-score, macro F1-score, and a confusion matrix. The test split shall not be used at any earlier point in the training sequence.
- [x] **TRAIN-PROC-008**: The test-set evaluation results shall be included in the `EvaluationReport` under a `test_set` section, separate from the cross-validation fold results.

## Model Artifact

- [x] **TRAIN-DATA-001**: The system shall persist the model artifact bundle to `s3://{PHENO_S3_BUCKET}/models/{run_id}/` containing: `model.json` (XGBoost booster), `feature_registry.json`, `imputation_medians.json`, `evaluation_report.json`, and `label_encoder.json`.
- [x] **TRAIN-DATA-002**: The `label_encoder.json` shall map integer class indices used internally by XGBoost to human-readable phenotype label strings.
- [x] **TRAIN-DATA-003**: The model artifact bundle shall be self-contained — no other pipeline stage shall be required to run inference against it.
