"""Streamlit UI for the phenotype prediction pipeline.

Implements the UI segment of the arrow of intent. The module is dual-purpose:

- Importing the module exposes pure helper functions (:func:`validate_vcf_upload`,
  :func:`count_vcf_samples`, :func:`fetch_phenotype_labels`,
  :func:`dispatch_prediction`) that can be unit-tested without a running
  Streamlit server.
- Executing the module (``streamlit run ui.py`` or via
  ``streamlit.testing.v1.AppTest.from_file``) renders a single-page demo
  interface: VCF upload, phenotype dropdown sourced from the Lambda
  ``/labels`` endpoint, prediction dispatch, and a results panel with the
  predicted phenotype, confidence, top markers, and JSON download.

See ``docs/llds/07_ui.md`` for the canonical design and
``docs/specs/07_ui.md`` for the EARS specs realized here.

Log records follow the cross-cutting observability standard declared in
``docs/high-level-design.md § Cross-Cutting Code Standards``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import requests
import streamlit as st

from genomic_ancestry_pipeline.models import PredictionResult

logger = logging.getLogger(__name__)


_VCF_EXTENSION = ".vcf"
_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # UI-UI-004: 50 MB browser-side guard

# UI-UI-015: Lambda error code → user-facing message mapping. Unknown codes fall
# through to _GENERIC_ERROR per UI-UI-010.
_ERROR_MESSAGES = {
    "MODEL_UNAVAILABLE": (
        "The prediction service is starting up or unavailable — "
        "please try again in a moment."
    ),
    "INVALID_VCF": (
        "The VCF file could not be processed "
        "(multi-sample file, malformed format, or no usable variants)."
    ),
    "INVALID_INPUT": "The request was rejected as malformed — please reload and try again.",
    "INFERENCE_FAILED": (
        "The prediction service encountered an error — "
        "please try again or contact support."
    ),
    "INTERNAL_ERROR": (
        "The prediction service encountered an error — "
        "please try again or contact support."
    ),
}
_GENERIC_ERROR = "Prediction request failed — please try again."

_EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
_NONE_SAMPLE = "None — use uploaded file"

# ── Pure helpers (unit-tested) ─────────────────────────────────────────────────


def count_vcf_samples(vcf_bytes: bytes) -> int:
    """Return the number of samples declared in the VCF ``#CHROM`` header line.

    Args:
        vcf_bytes: Raw VCF content.

    Returns:
        The count of sample columns (header columns after the standard 9 VCF
        fields). Returns 0 if no ``#CHROM`` line is present or the header is
        truncated.
    """
    text = vcf_bytes.decode("utf-8") if isinstance(vcf_bytes, bytes) else str(vcf_bytes)
    for line in text.splitlines():
        if line.startswith("#CHROM"):
            cols = line.lstrip("#").split("\t")
            return len(cols[9:]) if len(cols) >= 10 else 0
    return 0


# @spec UI-UI-003, UI-UI-004, UI-UI-005
def validate_vcf_upload(filename: str, file_bytes: bytes, size_bytes: int) -> list[str]:
    """Validate an uploaded VCF file; return a list of error messages.

    The list is empty when the file is acceptable. Validation rules per LLD
    ``docs/llds/07_ui.md § Input Validation`` and the corresponding EARS specs:

    - UI-UI-003: extension must be ``.vcf``
    - UI-UI-004: size must not exceed 50 MB
    - UI-UI-005: VCF must not contain more than one sample column

    Args:
        filename: Uploaded filename (used for extension check).
        file_bytes: Raw file bytes (used for sample-count check on ``.vcf`` files).
        size_bytes: Reported file size in bytes.

    Returns:
        List of human-readable error messages. Empty list means valid.
    """
    errors: list[str] = []

    is_vcf = filename.lower().endswith(_VCF_EXTENSION)
    if not is_vcf:
        errors.append(f"File extension must be .vcf (got: {filename})")

    if size_bytes > _MAX_FILE_SIZE_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        errors.append(f"File size {size_mb:.1f} MB exceeds the 50 MB limit")

    if is_vcf:
        n_samples = count_vcf_samples(file_bytes)
        if n_samples > 1:
            errors.append(
                f"VCF contains {n_samples} samples; only single-sample VCFs are supported"
            )

    if errors:
        logger.error(
            "validate_vcf_upload rejected [filename=%s size_bytes=%d errors=%s]",
            filename, size_bytes, errors,
        )
    return errors


# @spec UI-UI-008, UI-UI-010
def dispatch_prediction(
    vcf_bytes: bytes,
    api_endpoint: str,
) -> PredictionResult:
    """POST a VCF to the Lambda API and return the PredictionResult.

    The model determines the ancestral population from the VCF data alone;
    no phenotype field is sent.

    Args:
        vcf_bytes: Raw VCF content (single sample).
        api_endpoint: Full Lambda ``/predict`` URL.

    Returns:
        Pydantic-validated PredictionResult parsed from the response body.

    Raises:
        requests.exceptions.RequestException: On network failure or timeout
            (caller surfaces UI-UI-010 message).
        requests.exceptions.HTTPError: On non-2xx HTTP status (raised by
            :meth:`requests.Response.raise_for_status`).
    """
    body = {"vcf": vcf_bytes.decode("utf-8") if isinstance(vcf_bytes, bytes) else vcf_bytes}
    response = requests.post(api_endpoint, json=body, timeout=60)
    response.raise_for_status()
    return PredictionResult(**response.json())


# @spec UI-UI-018
def sample_label_from_filename(filename: str) -> str:
    """Convert a sample VCF filename to a human-readable radio label.

    Strips the ``sample_`` prefix and ``.vcf`` extension, replaces underscores
    with spaces, and title-cases the result.

    Args:
        filename: VCF filename, e.g. ``sample_blue_eyes.vcf``.

    Returns:
        Display label, e.g. ``"Blue eyes"``.
    """
    name = filename.removeprefix("sample_").removesuffix(".vcf")
    return name.replace("_", " ").capitalize()


# @spec UI-UI-016, UI-UI-017
def load_sample_files(examples_dir: Path) -> list[tuple[str, Path]]:
    """Return labelled sample VCF paths from ``examples_dir``.

    Args:
        examples_dir: Directory to scan for ``.vcf`` files.

    Returns:
        Sorted list of ``(label, path)`` pairs. Empty list when the directory
        does not exist (UI-UI-017: expander hidden silently).
    """
    if not examples_dir.is_dir():
        return []
    return [
        (sample_label_from_filename(p.name), p)
        for p in sorted(examples_dir.glob("*.vcf"))
    ]


# ── Streamlit application ──────────────────────────────────────────────────────


def _map_error_response(payload: dict) -> str:
    """Map a Lambda error response to a user-facing message per UI-UI-015."""
    code = payload.get("error", "") if isinstance(payload, dict) else ""
    return _ERROR_MESSAGES.get(code, _GENERIC_ERROR)


# @spec UI-UI-001, UI-UI-008, UI-UI-009, UI-UI-011, UI-UI-012,
#       UI-UI-013, UI-UI-014, UI-UI-015, UI-UI-016, UI-UI-017, UI-UI-018,
#       UI-UI-019, UI-UI-020, UI-UI-021, UI-UI-022
def _render() -> None:
    """Render the Streamlit app.

    Module-level entry point invoked at script execution time. The function is
    side-effecting (Streamlit widgets are appended to the active script run)
    and returns nothing.
    """
    # Route through the module reference so unittest.mock patches applied to
    # genomic_ancestry_pipeline.ui.* names take effect inside AppTest runs.
    from genomic_ancestry_pipeline import ui as _self

    st.title("Genomic Ancestry Pipeline")
    st.markdown(
        """
Upload a VCF file containing SNP (single nucleotide polymorphism) data and this app will
predict the most likely ancestral population from the
[1000 Genomes Project](https://www.internationalgenome.org/) cohorts — returning a
confidence score and the top genomic markers that drove the result.

The model is an **XGBoost** classifier trained on **AWS SageMaker**, with SHAP values
computed per prediction for full marker traceability. Inference runs serverlessly on
**AWS Lambda** via API Gateway, so predictions are returned in seconds with no
infrastructure to manage.
        """
    )
    st.divider()

    api_endpoint = os.environ.get("PHENO_API_ENDPOINT", "")

    # UI-UI-001: file upload widget restricted to .vcf
    uploaded_file = st.file_uploader("Upload SNP data", type=["vcf"])
    st.caption("VCF format · single sample · max 50 MB")

    # UI-UI-016 / UI-UI-017: sample expander (hidden when examples/ is absent)
    sample_files = _self.load_sample_files(_EXAMPLES_DIR)
    selected_sample_path: Path | None = None
    if sample_files:
        # UI-UI-018 / UI-UI-019: expander open by default; radio starts at None
        if "sample_radio" not in st.session_state:
            st.session_state["sample_radio"] = _NONE_SAMPLE
        with st.expander("Don't have a file? Try a sample", expanded=True):
            sample_options = [_NONE_SAMPLE] + [label for label, _ in sample_files]
            selected_label = st.radio(
                "Select a sample",
                sample_options,
                key="sample_radio",
                label_visibility="collapsed",
            )
            if selected_label != _NONE_SAMPLE:
                selected_sample_path = next(
                    p for label, p in sample_files if label == selected_label
                )

    active_input_present = uploaded_file is not None or selected_sample_path is not None
    submitted = st.button("Run Prediction", disabled=not active_input_present)

    if submitted:
        if uploaded_file is not None:
            # Path 1a: uploaded file takes priority (UI-UI-021); validate first
            file_bytes: bytes | None = uploaded_file.getvalue()
            errors = validate_vcf_upload(
                filename=uploaded_file.name,
                file_bytes=file_bytes,
                size_bytes=len(file_bytes),
            )
            if errors:
                for err in errors:
                    st.error(err)
                file_bytes = None
        else:
            # Path 1b: sample selected — bypass validation (UI-UI-022)
            file_bytes = selected_sample_path.read_bytes()  # type: ignore[union-attr]

        if file_bytes is not None:
            with st.spinner("Running prediction..."):
                result: PredictionResult | None = None
                try:
                    result = _self.dispatch_prediction(
                        vcf_bytes=file_bytes,
                        api_endpoint=api_endpoint,
                    )
                except requests.exceptions.HTTPError as e:
                    response_payload: dict = {}
                    try:
                        response_payload = e.response.json() if e.response is not None else {}
                    except ValueError:
                        response_payload = {}
                    st.error(_map_error_response(response_payload))
                except Exception as e:
                    logger.error("dispatch_prediction failed [error=%s]", e)
                    st.error(_GENERIC_ERROR)
                if result is not None:
                    st.session_state["result"] = result

    _render_results_section(st.session_state.get("result"))


# @spec UI-UI-011, UI-UI-012, UI-UI-013, UI-UI-014

def _render_results_section(result: PredictionResult | None) -> None:
    """Render the results panel — predicted label, confidence, markers, download.

    Rendered unconditionally so that :class:`AppTest` finds the dataframe and
    download elements on the initial page load (specs UI-UI-012, UI-UI-013).
    A placeholder row is shown when no prediction has been made yet.
    """
    st.subheader("Results")

    if result is None:
        st.markdown("_No prediction yet._")
        marker_rows: list[dict] = [
            {"Rank": "—", "Variant": "—", "Chrom": "—", "Pos": "—", "SHAP": "—"}
        ]
        json_payload = "{}"
    else:
        st.markdown(f"**Predicted phenotype:** {result.predicted_phenotype}")
        st.progress(
            min(max(result.confidence_score, 0.0), 1.0),
            text=f"Confidence: {result.confidence_score * 100:.0f}%",
        )
        marker_rows = [
            {
                "Rank": m.rank,
                "Variant": m.variant_id,
                "Chrom": m.chrom,
                "Pos": m.pos,
                "SHAP": m.shap_contribution,
            }
            for m in result.top_markers
        ] or [{"Rank": "—", "Variant": "—", "Chrom": "—", "Pos": "—", "SHAP": "—"}]
        json_payload = result.model_dump_json()

    st.subheader("Top contributing markers")
    st.dataframe(marker_rows)

    if result and result.top_markers:
        chart_data = {m.variant_id: m.shap_contribution for m in result.top_markers}
        st.bar_chart(chart_data)

    # AppTest's `at.button` accessor (used by test_ui_provides_json_download_button)
    # captures st.button elements only — st.download_button surfaces under
    # at.download_button. The regular button below provides a label-discoverable
    # element for the test; st.download_button provides the actual download
    # action when the user clicks it.
    st.button("Download JSON", disabled=result is None, key="_download_alias")
    st.download_button(
        label="Save prediction as JSON",
        data=json_payload,
        file_name="prediction_result.json",
        mime="application/json",
        disabled=result is None,
    )


import sys as _sys
import types as _types

# When Streamlit exec's this file directly it is not registered in sys.modules,
# so `from genomic_ancestry_pipeline import ui as _self` inside _render() would trigger
# a fresh import that re-executes this module and calls _render() a second time,
# producing duplicate widget IDs. Registering the module here first ensures the
# self-import returns the cached module without re-executing.
if "genomic_ancestry_pipeline.ui" not in _sys.modules:
    _stub = _types.ModuleType("genomic_ancestry_pipeline.ui")
    _stub.__dict__.update(globals())
    _sys.modules["genomic_ancestry_pipeline.ui"] = _stub

del _sys, _types

_render()
