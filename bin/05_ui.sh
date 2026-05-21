#!/usr/bin/env bash
# Starts the Streamlit prediction UI at http://localhost:8501.
set -euo pipefail

: "${PHENO_API_ENDPOINT:?Set PHENO_API_ENDPOINT before running this script (see 04_deploy.sh output)}"

uv run streamlit run src/genomic_ancestry_pipeline/ui.py
