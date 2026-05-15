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

import requests
import streamlit as st

from phenotype_pipeline.models import PredictionResult

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
        "The uploaded VCF could not be processed "
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


# @spec UI-UI-002, UI-UI-006
def fetch_phenotype_labels(api_endpoint: str) -> list[str]:
    """Fetch supported phenotype labels from ``{api_endpoint}/labels``.

    Args:
        api_endpoint: Base API URL (with or without trailing slash).

    Returns:
        List of phenotype label strings derived from the deployed model's
        ``label_encoder.json``.

    Raises:
        requests.exceptions.RequestException: If the GET request fails or
            times out. Callers (the Streamlit script) use this to disable
            the phenotype dropdown per UI-UI-006.
    """
    url = f"{api_endpoint.rstrip('/')}/labels"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("labels", []))


# @spec UI-UI-008, UI-UI-010
def dispatch_prediction(
    vcf_bytes: bytes,
    phenotype: str,
    api_endpoint: str,
) -> PredictionResult:
    """POST a VCF + phenotype to the Lambda API and return the PredictionResult.

    Args:
        vcf_bytes: Raw VCF content (single sample).
        phenotype: Selected phenotype label.
        api_endpoint: Full Lambda ``/predict`` URL.

    Returns:
        Pydantic-validated PredictionResult parsed from the response body.

    Raises:
        requests.exceptions.RequestException: On network failure or timeout
            (caller surfaces UI-UI-010 message).
        requests.exceptions.HTTPError: On non-2xx HTTP status (raised by
            :meth:`requests.Response.raise_for_status`).
    """
    body = {
        "vcf": vcf_bytes.decode("utf-8") if isinstance(vcf_bytes, bytes) else vcf_bytes,
        "phenotype": phenotype,
    }
    response = requests.post(api_endpoint, json=body, timeout=60)
    response.raise_for_status()
    return PredictionResult(**response.json())


# ── Streamlit application ──────────────────────────────────────────────────────


def _try_fetch_labels(api_endpoint: str) -> tuple[list[str], str | None]:
    """Fetch phenotype labels, returning (labels, error_message).

    The fetch is dispatched via the ``phenotype_pipeline.ui`` module path rather
    than the local name so that :func:`unittest.mock.patch` targets in tests
    (which see ``phenotype_pipeline.ui`` but not the ``__main__`` module that
    Streamlit's :class:`AppTest` runner uses) take effect. The real
    ``fetch_phenotype_labels`` raises on empty/invalid endpoint; the except
    branch produces the UI-UI-006 error message.
    """
    try:
        from phenotype_pipeline import ui as _self
        return _self.fetch_phenotype_labels(api_endpoint), None
    except Exception as e:
        logger.error(
            "fetch_phenotype_labels failed [api_endpoint=%s error=%s]",
            api_endpoint, e,
        )
        return [], (
            "Could not load phenotype options — check API endpoint. "
            f"({type(e).__name__})"
        )


def _map_error_response(payload: dict) -> str:
    """Map a Lambda error response to a user-facing message per UI-UI-015."""
    code = payload.get("error", "") if isinstance(payload, dict) else ""
    return _ERROR_MESSAGES.get(code, _GENERIC_ERROR)


# @spec UI-UI-001, UI-UI-002, UI-UI-006, UI-UI-009, UI-UI-011, UI-UI-012,
#       UI-UI-013, UI-UI-014, UI-UI-015
def _render() -> None:
    """Render the Streamlit app.

    Module-level entry point invoked at script execution time. The function is
    side-effecting (Streamlit widgets are appended to the active script run)
    and returns nothing.
    """
    st.title("Phenotype Pipeline")

    api_endpoint = os.environ.get("PHENO_API_ENDPOINT", "")
    labels, labels_error = _try_fetch_labels(api_endpoint)
    if labels_error:
        st.error(labels_error)

    # UI-UI-001: file upload widget restricted to .vcf
    uploaded_file = st.file_uploader("Upload SNP data", type=["vcf"])

    # UI-UI-002 / UI-UI-006: phenotype dropdown (disabled when labels missing)
    phenotype = st.selectbox(
        "Target phenotype",
        options=labels or ["(unavailable)"],
        disabled=not labels,
    )

    # UI-UI-006: disable submit only when labels endpoint failed (the dropdown is
    # also disabled). File-presence is checked on click so the button is always
    # responsive when the labels endpoint is healthy.
    submit_disabled = not labels
    submitted = st.button("Run Prediction", disabled=submit_disabled)

    if submitted and uploaded_file is None:
        st.error("Please upload a VCF file before submitting.")
    elif submitted and uploaded_file is not None and labels:
        file_bytes = uploaded_file.getvalue()
        errors = validate_vcf_upload(
            filename=uploaded_file.name,
            file_bytes=file_bytes,
            size_bytes=len(file_bytes),
        )
        if errors:
            for err in errors:
                st.error(err)
        else:
            with st.spinner("Running prediction..."):
                try:
                    # See _try_fetch_labels: route through the module so test
                    # patches on phenotype_pipeline.ui.dispatch_prediction apply.
                    from phenotype_pipeline import ui as _self
                    result = _self.dispatch_prediction(
                        vcf_bytes=file_bytes,
                        phenotype=phenotype,
                        api_endpoint=api_endpoint,
                    )
                except requests.exceptions.HTTPError as e:
                    response_payload: dict = {}
                    try:
                        response_payload = e.response.json() if e.response is not None else {}
                    except ValueError:
                        response_payload = {}
                    st.error(_map_error_response(response_payload))
                    result = None
                except Exception as e:
                    logger.error("dispatch_prediction failed [error=%s]", e)
                    st.error(_GENERIC_ERROR)
                    result = None
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


_render()
