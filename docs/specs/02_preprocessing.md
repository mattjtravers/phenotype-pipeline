# Preprocessing Specs

## Train/Test Split

- [ ] **PREP-PROC-001**: The system shall split the `RawSnpDataset` into train (80%) and test (20%) subsets before fitting any statistics, to prevent leakage of test-split information into imputation or feature selection.
- [ ] **PREP-PROC-002**: The train/test split shall be stratified by phenotype class label.
- [ ] **PREP-PROC-003**: The train/test split shall use a configurable random seed (`random_state` pipeline parameter) for reproducibility.
- [ ] **PREP-DATA-001**: Each sample in the `CleanSnpDataset` shall carry a `split` field with value `"train"` or `"test"` indicating its assigned subset.

## Imputation

- [ ] **PREP-PROC-004**: The system shall compute per-variant median genotype values using training-split samples only.
- [ ] **PREP-PROC-005**: The system shall impute missing genotype values (VCF `./.` encoding) using the per-variant median computed from the training split.
- [ ] **PREP-DATA-002**: The `CleanSnpDataset` shall include an `imputation_medians` mapping of variant ID to the median value used, for use at inference time.

## Encoding

- [ ] **PREP-PROC-006**: The system shall apply dosage encoding to all genotype values, mapping homozygous reference to 0, heterozygous to 1, and homozygous alternate to 2.

## Schema Validation

- [ ] **PREP-PROC-007**: The system shall validate the `RawSnpDataset` using its Pydantic schema at the entry point of preprocessing, before any processing begins.
- [ ] **PREP-PROC-008**: The system shall validate the `CleanSnpDataset` using its Pydantic schema before passing it to feature engineering.
- [ ] **PREP-PROC-009**: If Pydantic validation fails on either input or output, then the system shall raise a `ValidationError` identifying the offending field and value without silently dropping invalid records.
