# bin/

Runnable scripts for the Quick Start pipeline. Run them in order; each script prints the `export VAR=...` line that the next script depends on.

| Script | What it does |
|---|---|
| [01_setup_aws.sh](01_setup_aws.sh) | Creates S3 bucket + SageMaker IAM role via AWS CLI |
| [02_ingest.sh](02_ingest.sh) | One-time ETL from public 1000 Genomes bucket |
| [03_train.sh](03_train.sh) | Launches SageMaker training job, captures `run_id` |
| [04_deploy.sh](04_deploy.sh) | `sam build` + `sam deploy`, extracts API endpoint from CloudFormation |
| [05_ui.sh](05_ui.sh) | Starts Streamlit on localhost:8501 |

See the [Quick Start](../README.md#quick-start) section in the root README for full usage instructions.
