# Feature Engineering

## Context and Design Philosophy

This component transforms the `CleanSnpDataset` into a model-ready feature matrix. Its two responsibilities are SNP encoding (already partially handled by dosage encoding in preprocessing) and marker selection — reducing the variant space to a tractable, informative subset before model training.

The output is a numeric feature matrix (samples × selected markers) with a corresponding feature registry that maps column indices back to variant IDs. This registry is essential for marker traceability in the prediction layer.

## SNP Encoding

Dosage encoding (0/1/2) is applied during preprocessing. This component's encoding role is limited to:

- Normalizing dosage values to [0, 1] if needed (i.e., dividing by 2) — **deferred** until model experiments show it improves convergence
- No one-hot encoding: XGBoost handles ordinal integer inputs natively; one-hot would triple the feature count with no benefit

## Marker Selection

The full 1000 Genomes SNP space is large (~80M variants). A marker selection step reduces this to a computationally tractable subset before training.

Selection strategy (in order of application):

1. **MAF filter** — drop variants with minor allele frequency < 0.01 (rare variants with insufficient signal)
2. **Missingness filter** — drop variants with > 10% missing rate before imputation (high-missingness variants are unreliable even after imputation)
3. **Variance filter** — drop near-zero-variance variants after dosage encoding
4. **Phenotype-specific filter** — optionally, top-N variants by univariate association score (chi-squared for categorical phenotypes). N is a configurable parameter; default 10,000.

All filter thresholds are configurable via pipeline parameters, not hard-coded.

## Feature Registry

The feature registry is a Pydantic model that maps each column index in the feature matrix to its source variant:

```
FeatureRegistry
  features: list[FeatureEntry]

FeatureEntry
  column_index: int
  variant_id: str     # "{chrom}_{pos}_{ref}_{alt}"
  chrom: str
  pos: int
  ref: str
  alt: str
  maf: float
```

The registry is persisted alongside the trained model artifact so prediction can trace output back to genomic coordinates.

## Output Contract

```
FeatureMatrix
  X: np.ndarray           # shape (n_samples, n_features)
  y: np.ndarray           # shape (n_samples,) — phenotype labels
  sample_ids: list[str]
  registry: FeatureRegistry
```

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|---|---|---|---|
| Encoding | Dosage (0/1/2) | One-hot, principal components | XGBoost handles integer ordinal inputs natively; one-hot triples feature count; PCA loses variant identity needed for traceability |
| Marker selection | MAF + missingness + variance + optional association filter | LD pruning, GWAS-informed selection | Simple filters are transparent and reproducible; LD pruning adds complexity with marginal benefit for XGBoost |
| Threshold configuration | Pipeline parameters | Hard-coded constants | Allows experimentation without code changes |
| Feature registry format | Pydantic + JSON | CSV lookup table | Consistent with project-wide schema approach; JSON serializes cleanly alongside model artifact |

## Open Questions & Future Decisions

### Resolved
1. ✅ One-hot encoding rejected in favour of dosage encoding for XGBoost compatibility

### Deferred
1. Whether to add dosage normalization to [0,1] — defer to model-training experiments
2. Optimal default N for association filter — defer to first training run on 1000 Genomes data

## References

- `docs/llds/preprocessing.md` — upstream producer of `CleanSnpDataset`
- `docs/llds/model-training.md` — downstream consumer of `FeatureMatrix`
- `docs/llds/prediction.md` — consumes `FeatureRegistry` for marker traceability
