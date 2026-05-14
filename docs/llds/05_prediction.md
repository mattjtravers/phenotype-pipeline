# Prediction

## Context and Design Philosophy

This component loads a trained model artifact and runs inference on new SNP data. It produces three outputs per sample: a predicted phenotype label, a confidence score, and a list of contributing genomic markers. Marker traceability is the key differentiator from a bare XGBoost `predict` call — it is a first-class output, not an afterthought.

The prediction component is the component most directly serving the Business Analyst persona. Every output must be interpretable and auditable.

## Inference Pipeline

1. Load model artifact bundle from S3
2. Accept raw SNP input for new samples (VCF format)
3. Apply preprocessing using stored `imputation_medians` (no re-fitting)
4. Apply feature selection using stored `feature_registry` (select same variants, same column order)
5. Run `model.predict_proba()` to get class probabilities
6. Derive predicted label and confidence score from probabilities
7. Compute per-sample SHAP marker contributions via `predict(X, pred_contribs=True)`
8. Assemble and validate `PredictionResult` via Pydantic
9. Return result to caller

## Confidence Score

The confidence score is the **maximum class probability** from `predict_proba()`. This is a simple, interpretable proxy for model certainty:

- Score near 1.0 → model strongly favours one class
- Score near `1/n_classes` → model is uncertain

No probability calibration (e.g., Platt scaling) is applied in the initial version — calibration requires a held-out calibration set and adds complexity that is out of scope for the SDD demo. This is noted as a deferred decision.

## Marker Traceability

Marker contributions are derived using XGBoost's built-in per-sample SHAP values via `booster.predict(X, pred_contribs=True)`. This returns a contribution score for each feature for each individual sample — the markers shown to the Business Analyst reflect the specific input, not a global model average.

No `shap` package is required; XGBoost's tree SHAP implementation is used directly.

For each prediction, the top-N features by absolute SHAP contribution are returned, mapped back to genomic coordinates via the `FeatureRegistry`. Default N = 20 (configurable).

```
MarkerContribution
  variant_id: str
  chrom: str
  pos: int
  ref: str
  alt: str
  shap_contribution: float  # per-sample SHAP value (signed; magnitude = importance)
  rank: int                 # 1 = largest absolute contribution
```

## Output Contract

```
PredictionResult
  sample_id: str
  predicted_phenotype: str
  confidence_score: float                  # in [0.0, 1.0]
  class_probabilities: dict[str, float]    # all classes
  top_markers: list[MarkerContribution]
  model_artifact_version: str              # S3 key of artifact bundle used
```

Inference results are returned to the caller only — they are not persisted to S3.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|---|---|---|---|
| Confidence score | Max class probability | Calibrated probability, entropy | Max probability is simple and interpretable; calibration is deferred; entropy is less intuitive for non-technical users |
| Marker attribution | XGBoost built-in per-sample SHAP (`pred_contribs=True`) | Global gain importance, `shap` package, permutation importance | Per-sample attributions are more meaningful for the Business Analyst persona; XGBoost's tree SHAP requires no extra dependency and adds negligible compute |
| Inference-time preprocessing | Stored medians + registry from artifact | Re-run preprocessing pipeline | Avoids reprocessing training data at inference time; ensures identical transformations |

## Open Questions & Future Decisions

### Resolved
1. ✅ Per-sample SHAP via XGBoost `pred_contribs=True` — no extra dependency, per-sample not global

### Deferred
1. Probability calibration (Platt scaling / isotonic regression) for more accurate confidence scores
2. Batch prediction API vs single-sample prediction — current design processes one sample at a time; batching deferred

## References

- `docs/llds/04_model-training.md` — produces the model artifact bundle
- `docs/llds/03_feature-engineering.md` — defines `FeatureRegistry`
- `docs/llds/06_deployment.md` — S3 paths for artifact and result storage
- `docs/llds/07_ui.md` — primary consumer of `PredictionResult`
