# samples/

Bundled single-sample VCF files for the prediction UI. Select one from the "Don't have a file? Try a sample" expander in the Streamlit app, or upload `sample_snp_upload.vcf` manually to test the file-upload path.

| File | Target population | Expected confidence | Shown in UI |
|---|---|---|---|
| [sample_snp_1.vcf](sample_snp_1.vcf) | YRI — Yoruba in Ibadan, Nigeria | ~93% | Yes |
| [sample_snp_2.vcf](sample_snp_2.vcf) | CHB — Han Chinese in Beijing | ~91% | Yes |
| [sample_snp_3.vcf](sample_snp_3.vcf) | CEU — Northern Europeans from Utah | ~91% | Yes |
| [sample_snp_upload.vcf](sample_snp_upload.vcf) | JPT — Japanese in Tokyo | ~91% | No (upload demo only) |

## Format notes

- VCF 4.2, GRCh37 reference, chromosome 15
- ~1100–1150 bi-allelic SNPs per file, all phased (`0|0`, `0|1`, `1|1`)
- Variants are a subset of the model's feature registry; remaining registry features are imputed from training-set medians
- Confidence values are model predictions — minor variation is expected across deployments
