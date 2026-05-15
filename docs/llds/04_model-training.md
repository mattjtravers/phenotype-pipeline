# Model Training

## Context and Design Philosophy

This component trains an XGBoost classifier on the feature matrix produced by feature engineering, evaluates it via k-fold cross-validation, and persists the trained model artifact. The artifact includes the model weights, the feature registry, and the imputation medians — everything needed for inference without re-running the training pipeline.

Training runs exclusively as a **SageMaker Training Job**. The training script (`pipeline/train.py`) is packaged in a Docker container and executed on a managed SageMaker instance — by default `ml.m5.2xlarge`, overridable per-run via the launcher's `--instance-type` flag (see `docs/llds/06_deployment.md § Training Launch`). Training is a batch operation; it does not need to be fast at inference time — it needs to be reproducible and evaluable.

## Training Protocol

### Algorithm
XGBoost (`xgboost.XGBClassifier`) with the following default hyperparameters (all configurable):

| Parameter | Default | Rationale |
|---|---|---|
| `n_estimators` | 200 | Sufficient for SNP data without overfitting at this scale |
| `max_depth` | 6 | Standard depth for tabular data |
| `learning_rate` | 0.1 | Conservative; pairs with early stopping |
| `subsample` | 0.8 | Reduces overfitting on high-dimensional SNP features |
| `colsample_bytree` | 0.8 | Randomly samples features per tree — important for sparse SNP matrices |
| `eval_metric` | `mlogloss` | Multi-class log loss; drives early stopping |
| `early_stopping_rounds` | 20 | Prevents overfitting; requires a validation set during training |

### Cross-Validation

K-fold cross-validation with **k=5** (configurable). Each fold:
1. Splits samples into train/validation sets (stratified by phenotype class)
2. Trains XGBoost on train split
3. Evaluates on validation split: computes per-class F1-score and generates a confusion matrix
4. Records fold metrics

Final reported metrics are the mean and standard deviation across all folds.

The final production model is then retrained on **all training-split samples** using the mean `n_estimators` from early stopping across folds.

## Evaluation

Per-fold and aggregate metrics are written to a JSON evaluation report:

```
EvaluationReport
  folds: list[FoldResult]
  aggregate: AggregateMetrics

FoldResult
  fold_index: int
  f1_per_class: dict[str, float]   # class label → F1
  f1_macro: float
  confusion_matrix: list[list[int]]

AggregateMetrics
  f1_macro_mean: float
  f1_macro_std: float
  confusion_matrix_mean: list[list[float]]
```

The confusion matrix and F1-scores are the primary evaluation artefacts for the Business Analyst and Data Scientist personas.

## Final Test-Set Evaluation

After the final model is retrained on all training-split samples, it is evaluated once on the held-out test split (20% of data, defined in PREP-PROC-001). This is the definitive evaluation reported to the Business Analyst and Data Scientist personas.

Test-set metrics added to the `EvaluationReport`:

```
EvaluationReport
  ...
  test_set: TestSetMetrics

TestSetMetrics
  f1_per_class: dict[str, float]
  f1_macro: float
  confusion_matrix: list[list[int]]
```

The test split is used exactly once — after all training and CV decisions are finalised — to prevent optimistic bias from repeated test-set evaluation.

## Model Artifact

The model artifact bundle persisted to S3 contains:

- `model.json` — XGBoost booster in JSON format
- `feature_registry.json` — `FeatureRegistry` from feature engineering
- `imputation_medians.json` — per-variant medians from preprocessing
- `evaluation_report.json` — cross-validation results
- `label_encoder.json` — maps integer class indices to phenotype label strings

This bundle is the sole input to the prediction component. No other pipeline stage is needed at inference time.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|---|---|---|---|
| Algorithm | Open-source XGBoost (in custom SageMaker container) | SageMaker built-in XGBoost | Custom container allows the full preprocessing stack and Pydantic validation to run inside the training job; open-source XGBoost gives full control over artifact format and feature importance extraction |
| Cross-validation | Stratified k-fold (k=5) | Hold-out split, leave-one-out | k=5 balances variance reduction with compute cost; stratification preserves class distribution in small cohorts |
| Final model training | Retrain on all data | Use best fold's model | All-data retraining maximises information use; early stopping round mean prevents overfitting |
| Artifact format | JSON bundle | Pickle, ONNX | JSON is human-readable and portable; pickle has security and version concerns; ONNX is unnecessary for a single-framework pipeline |

## Open Questions & Future Decisions

### Resolved
1. ✅ SageMaker Training Job is the sole training execution environment — no local training path
2. ✅ Open-source XGBoost in custom container chosen over SageMaker built-in for full control over preprocessing and artifact format
3. ✅ k=5 chosen as default fold count

### Deferred
1. Hyperparameter optimisation — manual defaults are sufficient for the SDD demo; HPO via SageMaker Experiments can be added later
2. Multi-label phenotype prediction (currently single phenotype per run)

## References

- `docs/llds/03_feature-engineering.md` — upstream producer of `FeatureMatrix`
- `docs/llds/05_prediction.md` — downstream consumer of model artifact
- `docs/llds/06_deployment.md` — S3 paths for artifact storage
