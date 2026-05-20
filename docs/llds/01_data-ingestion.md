# Data Ingestion

## Context and Design Philosophy

This component has two distinct phases:

1. **One-time ETL** — copies the relevant VCF file(s) from the public 1000 Genomes S3 bucket into the project S3 bucket. Runs once; output is the canonical source of truth for all training runs.
2. **Runtime loader** — reads the cached VCF from the project S3 bucket, parses it, and emits a `RawSnpDataset` to preprocessing.

Separating these phases keeps the hot path (training and inference) fast and independent of the public 1000 Genomes bucket. The VCF is the storage format throughout — no intermediate format conversion.

All downstream stages treat this component as a black box: they receive a `RawSnpDataset` and do not know whether it came from S3 or a local file path.

## Data Source

The [1000 Genomes Project](https://www.internationalgenome.org/) provides phased SNP data in VCF format, hosted on a public S3 bucket (`s3://1000genomes`). The pipeline targets bi-allelic SNPs only; structural variants and indels are out of scope.

Relevant S3 paths:
- Variant calls: `s3://1000genomes/release/20130502/ALL.chr{N}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz` — one file per chromosome; the target chromosome(s) are configurable, defaulting to chromosome 15 (OCA2/HERC2 region for eye color prediction)
- Sample metadata (population, sex, phenotype labels): `s3://1000genomes/technical/working/20130606_sample_info/20130606_sample_info.txt`

Expected data volume after MAF and missingness filtering: **10K–50K variants × 2,504 samples (~50–250 MB VCF)**.

## One-Time ETL

The ETL step is a standalone script (`pipeline/ingest_etl.py`). It:

1. Streams the source VCF(s) from `s3://1000genomes/...` for the configured target chromosome(s) (default: chromosome 15)
2. Filters to bi-allelic SNPs only
3. Writes the filtered VCF to `s3://{bucket}/data/raw/1000genomes.vcf.gz`
4. Copies the sample metadata TSV to `s3://{bucket}/data/raw/sample_info.tsv`

This script is run manually once during project setup, not as part of the recurring training pipeline.

## Runtime Loading

The runtime loader reads `s3://{bucket}/data/raw/1000genomes.vcf.gz` and parses it into a `RawSnpDataset`. It streams the VCF rather than downloading it fully to disk, to stay within Lambda's 512 MB `/tmp` constraint.

Locally, the loader reads from a local file path set via the `PHENO_LOCAL_DATA_DIR` env var — no S3 access required for development.

## Output Contract

The component emits a `RawSnpDataset` — a Pydantic model defined in the preprocessing LLD's schema registry:

- `samples`: list of sample IDs
- `variants`: list of variant records (CHROM, POS, REF, ALT, genotype matrix)
- `metadata`: sample-level metadata (population, phenotype labels)

The ingestion component does **not** validate genotype values or impute missing data — that is preprocessing's responsibility. It only validates that the VCF is structurally well-formed.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|---|---|---|---|
| Source format | VCF throughout (no conversion) | VCF → CSV conversion | CSV conversion is an unnecessary step; VCF is the canonical format and parseable directly; removing the conversion reduces pipeline surface area |
| ETL frequency | One-time manual script | ETL on every training run | 1000 Genomes is a fixed dataset; re-fetching it each run wastes time and bandwidth |
| Runtime access | Cached VCF in project S3 | Stream directly from public 1000 Genomes S3 | Decouples training runs from public bucket availability; faster access from same AWS region |
| Streaming vs full download | Streaming | Full download to `/tmp` | Lambda has a 512 MB `/tmp` limit; streaming avoids this constraint |
| Local dev substitution | Local file path via env var | Mocked S3 client | Env var swap keeps local dev fast without a mock framework |

## Open Questions & Future Decisions

### Resolved
1. ✅ VCF kept as storage format throughout — CSV conversion removed as unnecessary
2. ✅ One-time ETL pattern chosen; runtime loader reads from cached project S3

### Deferred
1. Whether to store the filtered VCF as a compressed BCF (binary VCF) for faster parsing — defer until parsing speed is measured on the filtered dataset

## References

- [1000 Genomes S3 bucket](https://registry.opendata.aws/1000-genomes/)
- `docs/llds/02_preprocessing.md` — downstream consumer, defines `RawSnpDataset`
- `docs/llds/06_deployment.md` — IAM policy for S3 access; `PHENO_S3_BUCKET` env var
