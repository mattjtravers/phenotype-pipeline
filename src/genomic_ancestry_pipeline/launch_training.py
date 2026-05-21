"""CLI entry point for launching a SageMaker training job.

This module is a thin command-line wrapper around
:func:`genomic_ancestry_pipeline.deployment.launch_training_job`. Its single
responsibility is operational ergonomics:

- Parse and validate CLI arguments.
- Generate a fresh ``run_id`` before submission.
- Translate library exceptions into distinct non-zero exit codes so an
  operator (or a wrapping CI script) can disambiguate failure causes from
  the shell.
- Emit the resulting ``run_id`` to **stdout only** on success — empty stdout
  on any failure path — so that
  ``sam deploy --parameter-overrides MODEL_RUN_ID=$(launch_training ...)``
  is safe to use literally without additional shell guards.

Log lines are routed by :mod:`genomic_ancestry_pipeline.logging_config`:
``INFO``/``DEBUG`` → stdout, ``WARNING``/``ERROR`` → stderr. Because the
``run_id`` is the only intentional stdout payload on success, INFO logs
during a successful run are also emitted to stdout but distinguishable
from the ``run_id`` line by their leading timestamp + level prefix.
Operators piping stdout into ``sam deploy`` should suppress logs with
``LOG_LEVEL=WARNING`` or pipe through ``tail -n 1``.

See ``docs/llds/06_deployment.md § Training Launch`` for the design rationale.
"""
from __future__ import annotations

import argparse
import logging
import sys

import botocore.exceptions

from genomic_ancestry_pipeline.deployment import generate_run_id, launch_training_job
from genomic_ancestry_pipeline.logging_config import configure_logging

logger = logging.getLogger(__name__)

# Distinct exit codes so callers can disambiguate failure cause from the shell.
_EXIT_OK = 0
_EXIT_MISSING_CREDENTIALS = 2
_EXIT_AWS_CLIENT_ERROR = 3
_EXIT_RUN_ID_COLLISION = 4
_EXIT_TRAINING_FAILED = 5


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the launcher CLI.

    Flag names follow argparse convention (hyphenated); they translate to
    snake_case keyword arguments when passed to
    :func:`launch_training_job`.

    Returns:
        An :class:`argparse.ArgumentParser` configured with every flag
        documented in DEPLOY-BE-016.
    """
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
    """Run the launcher CLI.

    Configures logging, parses argv, generates a ``run_id``, and invokes
    :func:`launch_training_job`. On success, prints the ``run_id`` to
    stdout and returns 0. On failure, logs an ``ERROR`` line to stderr
    (carrying enough context to triage from logs alone per the
    cross-cutting observability standard) and returns a non-zero exit code.

    Args:
        argv: Argument list excluding the program name. ``None`` (the
            default) means parse :data:`sys.argv` as argparse normally does.

    Returns:
        ``0`` on success; a distinct non-zero code per failure mode:

        - ``2`` — missing AWS credentials
        - ``3`` — AWS API ``ClientError`` (unreachable bucket, etc.)
        - ``4`` — ``run_id`` collision under ``s3://{bucket}/models/``
        - ``5`` — training job reached terminal status without success
    """
    configure_logging()

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
        logger.error(
            "launch_training abort [run_id=%s bucket=%s cause=missing_aws_credentials]: %s",
            run_id, args.bucket, e,
        )
        return _EXIT_MISSING_CREDENTIALS
    except botocore.exceptions.ClientError as e:
        logger.error(
            "launch_training abort [run_id=%s bucket=%s cause=aws_client_error]: %s",
            run_id, args.bucket, e,
        )
        return _EXIT_AWS_CLIENT_ERROR
    except FileExistsError as e:
        logger.error(
            "launch_training abort [run_id=%s bucket=%s cause=run_id_collision]: %s",
            run_id, args.bucket, e,
        )
        return _EXIT_RUN_ID_COLLISION
    except RuntimeError as e:
        logger.error(
            "launch_training abort [run_id=%s bucket=%s cause=training_job_failed]: %s",
            run_id, args.bucket, e,
        )
        return _EXIT_TRAINING_FAILED

    # stdout receives ONLY the run_id on success — this is the operator
    # contract that makes shell-piping into `sam deploy` safe.
    print(completed_run_id)
    return _EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
