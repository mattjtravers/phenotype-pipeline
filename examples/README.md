# examples/

Sample single-sample VCF files for testing the prediction UI. Upload any of these via the Streamlit interface or select one from the "Try a sample" expander.

| File | SNP profile |
|---|---|
| [test_snp_1.vcf](test_snp_1.vcf) | Homozygous reference at rs12913832 (HERC2); heterozygous at rs1800407 (OCA2) |
| [test_snp_2.vcf](test_snp_2.vcf) | Homozygous alt at rs12913832 and surrounding OCA2/HERC2 markers |
| [test_snp_3.vcf](test_snp_3.vcf) | High alt-allele dosage across all markers |
| [test_snp_4.vcf](test_snp_4.vcf) | Heterozygous at rs12913832; homozygous alt at rs1800407 |
| [test_snp_5.vcf](test_snp_5.vcf) | Mixed heterozygous profile across OCA2/HERC2 region |

## Format notes

- VCF 4.2, GRCh37 reference, chromosome 15 (OCA2/HERC2 region)
- 20 bi-allelic SNPs per file, all phased (`0|0`, `0|1`, `1|0`, `1|1`)
- Any variant not present in the trained model's feature registry is silently ignored; missing registry features are imputed from training-set medians
- The predicted ancestral population depends on the trained model — these profiles are designed to produce distinct inputs, not guaranteed outputs
