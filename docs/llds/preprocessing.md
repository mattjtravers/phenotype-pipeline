# Preprocessing

## Context and Design Philosophy

This component accepts the raw `RawSnpDataset` from data ingestion and produces a clean, schema-validated dataset ready for feature engineering. Two responsibilities: impute missing genotype values and enforce Pydantic schemas at the pipeline boundary.

Pydantic validation runs at the entry point of this component, not in data ingestion, because ingestion only guarantees structural VCF validity. Genotype-level validity (valid allele codes, expected value ranges) is a domain constraint that belongs to the preprocessing layer.

## Train/Test Split

The train/test split is owned by this component and applied at the entry point, before any statistics are fit. This prevents leakage from the test set into imputation medians (and into the feature engineering association filter downstream).

Split ratio: **80% train / 20% test**, stratified by phenotype class. The split is seeded for reproducibility (`random_state` configurable via pipeline parameter).

The `CleanSnpDataset` output carries a `split` field on each sample (`"train"` or `"test"`). Downstream components (feature engineering, model training) are responsible for respecting this split — they must fit only on samples marked `"train"`.

## Imputation Strategy

Missing genotype values (encoded as `./.` in VCF) are imputed using the **per-variant median** across all samples. This is the simplest robust strategy for SNP data: median is insensitive to outliers, and per-variant imputation preserves population-level allele frequency signals better than a global constant.

Imputation is fit on the **training split only** and applied to validation/test splits using the stored training medians. This prevents data leakage.

## Schema Validation

All data structures are defined as Pydantic models. Validation runs at two points:

1. **Input** — `RawSnpDataset` validated on arrival from data ingestion
2. **Output** — `CleanSnpDataset` validated before passing to feature engineering

If validation fails, the component raises a structured `ValidationError` with the offending field and value. The pipeline does not silently drop invalid records.

### Key Pydantic Models

```
RawSnpDataset
  samples: list[str]
  variants: list[RawVariant]
  metadata: SampleMetadata

RawVariant
  chrom: str
  pos: int
  ref: str
  alt: str
  genotypes: dict[str, str | None]   # sample_id → "0|0" | "0|1" | "1|1" | None

CleanSnpDataset
  samples: list[str]
  variants: list[CleanVariant]
  metadata: SampleMetadata
  imputation_medians: dict[str, float]  # variant_id → median used

CleanVariant
  variant_id: str   # "{chrom}_{pos}_{ref}_{alt}"
  genotypes: dict[str, int]  # sample_id → 0 | 1 | 2 (dosage encoding)
```

Dosage encoding (0/1/2 = homozygous ref / het / homozygous alt) is applied during cleaning, not feature engineering, because it is a canonical representation for bi-allelic SNPs.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|---|---|---|---|
| Imputation method | Per-variant median | Mean, mode, KNN imputation | Median is robust to rare variants with skewed allele frequencies; simpler than KNN; mode would bias toward reference allele |
| Split ownership | Preprocessing (first component to fit statistics) | Feature engineering, model training | Splitting here is the earliest safe point; ensures no downstream component accidentally fits on test data |
| Imputation scope | Training split only | All data together | Prevents leakage of test-split information into imputation statistics |
| Dosage encoding location | Preprocessing | Feature engineering | Dosage encoding is a data normalization step (0/1/2), not a feature design choice; belongs with cleaning |
| Validation framework | Pydantic | Pandera, manual assertions | Pydantic is already the project-wide schema tool; consistency over introducing a second framework |

## Open Questions & Future Decisions

### Resolved
1. ✅ Median imputation chosen for robustness and simplicity
2. ✅ Pydantic selected as sole schema enforcement tool

### Deferred
1. Whether to persist `imputation_medians` alongside the model artifact for inference-time reuse — defer to prediction LLD

## References

- `docs/llds/data-ingestion.md` — upstream producer of `RawSnpDataset`
- `docs/llds/feature-engineering.md` — downstream consumer of `CleanSnpDataset`
