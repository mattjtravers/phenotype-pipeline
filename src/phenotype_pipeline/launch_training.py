"""CLI entry point for launching a SageMaker training job.

Thin wrapper around ``phenotype_pipeline.deployment.launch_training_job``:
parses argv, generates a fresh ``run_id``, calls the library, emits the
resulting ``run_id`` to stdout on completion, and translates library
exceptions into non-zero exit codes.

Designed to be piped: ``sam deploy --parameter-overrides MODEL_RUN_ID=$(uv run python -m phenotype_pipeline.launch_training --bucket ... )``
is safe because stdout is empty on any failure path.
"""
from __future__ import annotations

import argparse
import sys

import botocore.exceptions

from phenotype_pipeline.deployment import generate_run_id, launch_training_job


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launch_training",
        description="Submit a SageMaker training job for the phenotype pipeline.",
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name (PHENO_S3_BUCKET)")
    parser.add_argument(
        "--instance-type",
        default="ml.m5.2xlarge",
        help="SageMaker training instance type (default: ml.m5.2xlarge)",
    )
    parser.add_argument("--k-folds", type=int, default=5, help="Cross-validation fold count")
    parser.add_argument(
        "--maf-threshold", type=float, default=0.01,
        help="Minor allele frequency threshold for feature filtering",
    )
    parser.add_argument("--top-n", type=int, default=10000, help="Top-N markers to keep")
    parser.add_argument("--random-state", type=int, default=42, help="RNG seed")
    return parser


# @spec DEPLOY-BE-016, DEPLOY-BE-017, DEPLOY-BE-018, DEPLOY-BE-019
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_id = generate_run_id()

    try:
        completed_run_id = launch_training_job(
            bucket=args.bucket,
            run_id=run_id,
            instance_type=args.instance_type,
            k_folds=args.k_folds,
            maf_threshold=args.maf_threshold,
            top_n=args.top_n,
            random_state=args.random_state,
        )
    except botocore.exceptions.NoCredentialsError as e:
        print(f"AWS credentials not found; cannot launch training: {e}", file=sys.stderr)
        return 2
    except botocore.exceptions.ClientError as e:
        print(f"AWS API error while launching training: {e}", file=sys.stderr)
        return 3
    except FileExistsError as e:
        print(f"run_id collision (S3 prefix already exists): {e}", file=sys.stderr)
        return 4
    except RuntimeError as e:
        print(f"Training job did not complete successfully: {e}", file=sys.stderr)
        return 5

    print(completed_run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
