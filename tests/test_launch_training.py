"""Tests for the SageMaker training launcher CLI — DEPLOY-BE-016/017/018/019.

The CLI module is ``src/genomic_ancestry_pipeline/launch_training.py``; it is a thin
wrapper around ``genomic_ancestry_pipeline.deployment.launch_training_job`` that
parses argv, calls the library function, emits the resulting ``run_id`` to
stdout on completion, and translates library exceptions into non-zero exit codes.
"""
from __future__ import annotations

from unittest.mock import patch

import botocore.exceptions
import pytest

from genomic_ancestry_pipeline import launch_training


# @spec DEPLOY-BE-016
def test_cli_help_lists_expected_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        launch_training.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--bucket", "--k-folds", "--maf-threshold", "--top-n", "--random-state",
                 "--instance-type"):
        assert flag in out, f"CLI --help missing flag {flag}"


# @spec DEPLOY-BE-016
def test_cli_emits_run_id_to_stdout_on_completion(capsys, monkeypatch):
    monkeypatch.setenv("PHENO_S3_BUCKET", "my-bucket")
    with patch("genomic_ancestry_pipeline.launch_training.launch_training_job") as mock_lib:
        mock_lib.return_value = "20240115-a3f2c1"
        exit_code = launch_training.main(["--bucket", "my-bucket"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    assert out == "20240115-a3f2c1", f"stdout must contain only the run_id, got: {out!r}"


# @spec DEPLOY-BE-016
def test_cli_default_instance_type_is_ml_m5_2xlarge():
    with patch("genomic_ancestry_pipeline.launch_training.launch_training_job") as mock_lib:
        mock_lib.return_value = "20240115-a3f2c1"
        launch_training.main(["--bucket", "my-bucket"])
    kwargs = mock_lib.call_args.kwargs
    assert kwargs.get("instance_type") == "ml.m5.2xlarge"


# @spec DEPLOY-BE-016
def test_cli_instance_type_override():
    with patch("genomic_ancestry_pipeline.launch_training.launch_training_job") as mock_lib:
        mock_lib.return_value = "20240115-a3f2c1"
        launch_training.main(["--bucket", "my-bucket", "--instance-type", "ml.m5.4xlarge"])
    kwargs = mock_lib.call_args.kwargs
    assert kwargs.get("instance_type") == "ml.m5.4xlarge"


# @spec DEPLOY-BE-016
def test_cli_passes_hyperparameters_through(capsys):
    with patch("genomic_ancestry_pipeline.launch_training.launch_training_job") as mock_lib:
        mock_lib.return_value = "20240115-a3f2c1"
        launch_training.main([
            "--bucket", "my-bucket",
            "--k-folds", "3",
            "--maf-threshold", "0.05",
            "--top-n", "5000",
            "--random-state", "7",
        ])
    kwargs = mock_lib.call_args.kwargs
    assert kwargs.get("k_folds") == 3
    assert kwargs.get("maf_threshold") == 0.05
    assert kwargs.get("top_n") == 5000
    assert kwargs.get("random_state") == 7


# @spec DEPLOY-BE-016
def test_cli_exits_nonzero_on_training_failure(capsys):
    with patch("genomic_ancestry_pipeline.launch_training.launch_training_job",
               side_effect=RuntimeError("Training job Failed: algorithm error")):
        exit_code = launch_training.main(["--bucket", "my-bucket"])
    assert exit_code != 0
    out = capsys.readouterr().out.strip()
    assert out == "", (
        "On failure, stdout must NOT contain a run_id (would mislead sam deploy piping)"
    )


# @spec DEPLOY-BE-017
def test_cli_exits_nonzero_on_missing_credentials(capsys):
    with patch("genomic_ancestry_pipeline.launch_training.launch_training_job",
               side_effect=botocore.exceptions.NoCredentialsError()):
        exit_code = launch_training.main(["--bucket", "my-bucket"])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "credential" in err.lower() or "AWS" in err


# @spec DEPLOY-BE-017
def test_cli_exits_nonzero_on_unreachable_bucket(capsys):
    err = botocore.exceptions.ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
    )
    with patch("genomic_ancestry_pipeline.launch_training.launch_training_job", side_effect=err):
        exit_code = launch_training.main(["--bucket", "missing-bucket"])
    assert exit_code != 0


# @spec DEPLOY-BE-018
def test_cli_exits_nonzero_on_run_id_collision(capsys):
    with patch("genomic_ancestry_pipeline.launch_training.launch_training_job",
               side_effect=FileExistsError("models/20240115-a3f2c1/ already populated")):
        exit_code = launch_training.main(["--bucket", "my-bucket"])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "collision" in err.lower() or "exist" in err.lower() or "already" in err.lower()


# @spec DEPLOY-BE-019
def test_cli_default_region_pins_us_east_1():
    """When the CLI doesn't take a --region flag, the underlying library call must still
    receive us-east-1 (either by explicit kwarg, env var, or session config)."""
    with patch("genomic_ancestry_pipeline.launch_training.launch_training_job") as mock_lib:
        mock_lib.return_value = "20240115-a3f2c1"
        launch_training.main(["--bucket", "my-bucket"])
    kwargs = mock_lib.call_args.kwargs
    # The CLI may pass region explicitly OR delegate to the library's hardcoded us-east-1.
    # Either is acceptable; what is NOT acceptable is a different region.
    region = kwargs.get("region", "us-east-1")
    assert region == "us-east-1"
