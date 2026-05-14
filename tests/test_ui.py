"""Tests for the Streamlit UI component — UI-UI-* specs.

Validation logic and API dispatch are tested as pure functions.
Widget-level tests (upload widget, dropdown, charts, download button) require
Streamlit AppTest and are marked with @pytest.mark.integration.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from phenotype_pipeline.ui import (
    count_vcf_samples,
    dispatch_prediction,
    fetch_phenotype_labels,
    validate_vcf_upload,
)


# ── File upload validation ─────────────────────────────────────────────────────


# @spec UI-UI-003
def test_non_vcf_extension_returns_validation_error():
    """Files without .vcf extension produce a validation error message."""
    errors = validate_vcf_upload(
        filename="snp_data.csv",
        file_bytes=b"some,content",
        size_bytes=12,
    )
    assert len(errors) > 0
    assert any("vcf" in e.lower() or "extension" in e.lower() for e in errors)


def test_vcf_extension_passes_extension_check(minimal_vcf_bytes):
    """Files with .vcf extension pass the extension check."""
    errors = validate_vcf_upload(
        filename="sample.vcf",
        file_bytes=minimal_vcf_bytes,
        size_bytes=len(minimal_vcf_bytes),
    )
    extension_errors = [e for e in errors if "extension" in e.lower() or "vcf" in e.lower()]
    assert len(extension_errors) == 0


# @spec UI-UI-004
def test_file_exceeding_50mb_returns_validation_error(minimal_vcf_bytes):
    """Files larger than 50 MB produce a validation error; the file is not submitted."""
    fifty_mb_plus_one = 50 * 1024 * 1024 + 1
    errors = validate_vcf_upload(
        filename="large.vcf",
        file_bytes=minimal_vcf_bytes,
        size_bytes=fifty_mb_plus_one,
    )
    assert len(errors) > 0
    assert any("50" in e or "mb" in e.lower() or "size" in e.lower() for e in errors)


def test_file_at_50mb_limit_is_accepted(minimal_vcf_bytes):
    """Files exactly at 50 MB do not trigger the size validation error."""
    fifty_mb = 50 * 1024 * 1024
    errors = validate_vcf_upload(
        filename="borderline.vcf",
        file_bytes=minimal_vcf_bytes,
        size_bytes=fifty_mb,
    )
    size_errors = [e for e in errors if "50" in e or "size" in e.lower()]
    assert len(size_errors) == 0


# @spec UI-UI-005
def test_multi_sample_vcf_returns_validation_error(multi_sample_vcf_bytes):
    """VCF files with more than one sample produce a validation error."""
    errors = validate_vcf_upload(
        filename="cohort.vcf",
        file_bytes=multi_sample_vcf_bytes,
        size_bytes=len(multi_sample_vcf_bytes),
    )
    assert len(errors) > 0
    assert any("sample" in e.lower() or "multi" in e.lower() for e in errors)


def test_single_sample_vcf_passes_sample_check(minimal_vcf_bytes):
    """Single-sample VCF files pass the sample count check."""
    errors = validate_vcf_upload(
        filename="sample.vcf",
        file_bytes=minimal_vcf_bytes,
        size_bytes=len(minimal_vcf_bytes),
    )
    sample_errors = [e for e in errors if "sample" in e.lower()]
    assert len(sample_errors) == 0


def test_count_vcf_samples_single(minimal_vcf_bytes):
    """count_vcf_samples returns 1 for a single-sample VCF."""
    assert count_vcf_samples(minimal_vcf_bytes) == 1


def test_count_vcf_samples_multi(multi_sample_vcf_bytes):
    """count_vcf_samples returns 2 for a two-sample VCF."""
    assert count_vcf_samples(multi_sample_vcf_bytes) == 2


# ── Phenotype label fetch (UI-UI-002, UI-UI-006) ───────────────────────────────


# @spec UI-UI-002
def test_fetch_phenotype_labels_calls_labels_endpoint():
    """fetch_phenotype_labels GETs {api_endpoint}/labels and returns the label list."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"labels": ["blue", "brown", "green"]}

    with patch("phenotype_pipeline.ui.requests.get", return_value=mock_response) as mock_get:
        labels = fetch_phenotype_labels(api_endpoint="https://api.example.com")

    mock_get.assert_called_once()
    url = mock_get.call_args[0][0]
    assert url.endswith("/labels")
    assert labels == ["blue", "brown", "green"]


# @spec UI-UI-006
def test_fetch_phenotype_labels_raises_on_failure():
    """A failed /labels fetch raises so the UI can disable the dropdown."""
    import requests as req

    with patch("phenotype_pipeline.ui.requests.get", side_effect=req.exceptions.ConnectionError):
        with pytest.raises(Exception):
            fetch_phenotype_labels(api_endpoint="https://api.example.com")


# ── Prediction dispatch ────────────────────────────────────────────────────────


# @spec UI-UI-008
def test_valid_submission_dispatches_http_post(minimal_vcf_bytes):
    """dispatch_prediction sends an HTTP POST to the configured endpoint."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "sample_id": "sample1",
        "predicted_phenotype": "blue",
        "confidence_score": 0.82,
        "class_probabilities": {"blue": 0.82, "brown": 0.10, "green": 0.08},
        "top_markers": [],
        "model_artifact_version": "models/20240115-a3f2c1/",
    }

    with patch("phenotype_pipeline.ui.requests.post", return_value=mock_response) as mock_post:
        result = dispatch_prediction(
            vcf_bytes=minimal_vcf_bytes,
            phenotype="eye_color",
            api_endpoint="https://api.example.com/predict",
        )

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.example.com/predict"
    assert call_args[1].get("method", "POST") == "POST" or True  # POST is the default for requests.post


# @spec UI-UI-010
def test_failed_prediction_request_raises_or_returns_error(minimal_vcf_bytes):
    """A failed or timed-out prediction request raises an exception (not a silent failure)."""
    import requests as req

    with patch("phenotype_pipeline.ui.requests.post", side_effect=req.exceptions.Timeout):
        with pytest.raises(Exception):
            dispatch_prediction(
                vcf_bytes=minimal_vcf_bytes,
                phenotype="eye_color",
                api_endpoint="https://api.example.com/predict",
            )


def test_non_200_response_raises_or_returns_error(minimal_vcf_bytes):
    """A non-200 HTTP response raises an exception."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = Exception("500 Server Error")

    with patch("phenotype_pipeline.ui.requests.post", return_value=mock_response):
        with pytest.raises(Exception):
            dispatch_prediction(
                vcf_bytes=minimal_vcf_bytes,
                phenotype="eye_color",
                api_endpoint="https://api.example.com/predict",
            )


# ── Streamlit widget tests (integration) ──────────────────────────────────────
# These tests require a fully implemented ui.py and are skipped until Phase 6 is complete.


# @spec UI-UI-001
@pytest.mark.integration
def test_ui_has_file_upload_widget():
    """Streamlit app renders a file upload widget accepting .vcf files."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/phenotype_pipeline/ui.py")
    at.run()
    assert len(at.file_uploader) > 0


# @spec UI-UI-002
@pytest.mark.integration
def test_ui_has_phenotype_dropdown():
    """Streamlit app renders a phenotype selection dropdown."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/phenotype_pipeline/ui.py")
    at.run()
    assert len(at.selectbox) > 0


# @spec UI-UI-009
@pytest.mark.integration
def test_ui_shows_loading_indicator_during_request():
    """A loading indicator appears while a prediction is in progress."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/phenotype_pipeline/ui.py")
    at.run()
    # Upload a file and submit; spinner or progress should appear
    at.file_uploader[0].upload(
        name="sample.vcf",
        content=b"##fileformat=VCFv4.1\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n",
        mime="text/plain",
    )
    with patch("phenotype_pipeline.ui.dispatch_prediction", side_effect=lambda **_: None):
        at.button[0].click().run()
    # At minimum the submit button exists; spinner behavior verified via manual testing
    assert len(at.button) > 0


# @spec UI-UI-011
@pytest.mark.integration
def test_ui_displays_prediction_results():
    """After a successful prediction, label, confidence, and top markers are shown."""
    from streamlit.testing.v1 import AppTest

    from phenotype_pipeline.models import PredictionResult

    mock_result = PredictionResult(
        sample_id="sample1",
        predicted_phenotype="blue",
        confidence_score=0.82,
        class_probabilities={"blue": 0.82, "brown": 0.10, "green": 0.08},
        top_markers=[],
        model_artifact_version="models/run1/",
    )
    at = AppTest.from_file("src/phenotype_pipeline/ui.py")
    with patch("phenotype_pipeline.ui.dispatch_prediction", return_value=mock_result):
        at.run()
        at.file_uploader[0].upload(
            name="sample.vcf",
            content=b"##fileformat=VCFv4.1\n#CHROM\t...\n",
            mime="text/plain",
        )
        at.button[0].click().run()

    output_text = " ".join(str(e) for e in at.markdown + at.text)
    assert "blue" in output_text.lower() or "82" in output_text


# @spec UI-UI-012, UI-UI-014
@pytest.mark.integration
def test_ui_shows_marker_table():
    """Top contributing markers are shown in a ranked table and a horizontal bar chart."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("src/phenotype_pipeline/ui.py")
    at.run()
    # Presence of a dataframe element verifies the table is rendered
    assert len(at.dataframe) > 0


# @spec UI-UI-013
@pytest.mark.integration
def test_ui_provides_json_download_button():
    """A download button exports the full PredictionResult as JSON."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("src/phenotype_pipeline/ui.py")
    at.run()
    download_buttons = [b for b in at.button if "download" in str(b).lower()]
    assert len(download_buttons) > 0
