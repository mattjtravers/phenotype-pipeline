# Data Ingestion Specs

## One-Time ETL

- [ ] **INGEST-PROC-001**: During the one-time setup ETL step, the system shall stream and copy the VCF file(s) for the configured target chromosome(s) (default: chromosome 15, for eye color prediction via the OCA2/HERC2 region) from `s3://1000genomes/release/20130502/` to `s3://{PHENO_S3_BUCKET}/data/raw/1000genomes.vcf.gz`.
- [ ] **INGEST-PROC-002**: During the one-time ETL step, the system shall filter the source VCF to bi-allelic SNPs only, excluding structural variants and indels.
- [ ] **INGEST-PROC-003**: During the one-time ETL step, the system shall copy the sample metadata TSV to `s3://{PHENO_S3_BUCKET}/data/raw/sample_info.tsv`.

## Runtime Loading

- [ ] **INGEST-PROC-004**: The system shall stream the VCF from `s3://{PHENO_S3_BUCKET}/data/raw/1000genomes.vcf.gz` rather than downloading it fully to disk, to remain within execution environment memory constraints.
- [ ] **INGEST-PROC-005**: The system shall validate that the VCF is structurally well-formed (valid header, required columns present) before emitting a `RawSnpDataset`.
- [ ] **INGEST-DATA-001**: The system shall emit a `RawSnpDataset` containing: a list of sample IDs, a list of variant records (CHROM, POS, REF, ALT, genotype matrix), and sample-level metadata (population, phenotype labels).

## Error Handling

- [ ] **INGEST-PROC-006**: If the VCF is structurally malformed, then the system shall raise a structured error identifying the malformed field before any records are emitted.
- [ ] **INGEST-PROC-007**: If the sample metadata file is missing or unreadable, then the system shall raise a structured error before emitting a `RawSnpDataset`.
