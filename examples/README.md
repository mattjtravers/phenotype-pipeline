# examples/

Sample single-sample VCF files for testing the prediction UI. Upload any of these via the Streamlit interface at http://localhost:8501.

| File | Sample ID | Genotype profile |
|---|---|---|
| [sample_blue_eyes.vcf](sample_blue_eyes.vcf) | HG_BLUE001 | Homozygous reference at rs12913832 (HERC2); heterozygous at rs1800407 (OCA2) — blue-eye-associated profile |
| [sample_brown_eyes.vcf](sample_brown_eyes.vcf) | HG_BROWN001 | Homozygous alt at rs12913832 and surrounding OCA2/HERC2 markers — brown-eye-associated profile |
| [sample_green_eyes.vcf](sample_green_eyes.vcf) | HG_GREEN001 | Heterozygous at rs12913832; homozygous alt at rs1800407 — green-eye-associated profile |
| [sample_hazel_eyes.vcf](sample_hazel_eyes.vcf) | HG_HAZEL001 | Mixed heterozygous profile across OCA2/HERC2 region |
| [sample_dark_eyes.vcf](sample_dark_eyes.vcf) | HG_DARK001 | High alt-allele dosage across all markers — dark/brown profile with high melanin signal |

## Format notes

- VCF 4.2, GRCh37 reference, chromosome 15 (OCA2/HERC2 eye-color region)
- 20 bi-allelic SNPs per file, all phased (`0|0`, `0|1`, `1|0`, `1|1`)
- Any variant not present in the trained model's feature registry is silently ignored; missing registry features are imputed from training-set medians
- The predicted phenotype depends on the trained model — these profiles are designed to produce distinct inputs, not guaranteed outputs
