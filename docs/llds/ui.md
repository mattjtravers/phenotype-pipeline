# UI

## Context and Design Philosophy

This component provides a Streamlit web interface for the Business Analyst persona. It is the human-facing entry point to the pipeline: the user uploads SNP data, submits a prediction request, and views the results — phenotype label, confidence score, and top contributing markers.

The UI is a demo interface, not a production web app. It runs locally (`streamlit run`) and optionally deploys to a cloud host. It does not own any ML logic; it calls the prediction component's Python API directly (local) or via the Lambda endpoint (cloud).

## User Flow

```
1. Upload SNP file (VCF or CSV)
        ↓
2. Select target phenotype (dropdown)
        ↓
3. Submit → prediction request dispatched
        ↓
4. Results page:
   - Predicted phenotype label
   - Confidence score (progress bar)
   - Top-20 contributing markers (table + bar chart)
   - Option to download full PredictionResult as JSON
```

## Layout

```
┌──────────────────────────────────────────────┐
│ Phenotype Pipeline                           │
├──────────────────────────────────────────────┤
│ Upload SNP data         [Browse files...]    │
│                                              │
│ Target phenotype        [Eye color      ▾]   │
│                                              │
│                         [Run Prediction]     │
├──────────────────────────────────────────────┤
│ Results                                      │
│                                              │
│ Predicted phenotype:  Brown                  │
│ Confidence:           ████████░░  82%        │
│                                              │
│ Top contributing markers                     │
│ ┌────────────────────────────────────────┐   │
│ │ Rank │ Variant        │ Chrom │ Score  │   │
│ │  1   │ 15_28513871... │  15   │  0.34  │   │
│ │  2   │ 15_28230318... │  15   │  0.21  │   │
│ │  ...                                   │   │
│ └────────────────────────────────────────┘   │
│                                              │
│              [Download JSON]                 │
└──────────────────────────────────────────────┘
```

## Execution Modes

The UI supports two execution modes, toggled by an environment variable `PHENO_MODE`:

| Mode | Value | Prediction source |
|---|---|---|
| Local | `local` (default) | Calls `pipeline.predict` Python function directly |
| Cloud | `cloud` | Posts to Lambda API Gateway endpoint (`PHENO_API_ENDPOINT` env var) |

In local mode, the Streamlit app and the prediction pipeline run in the same Python process. This is the primary mode for the SDD demo.

## Input Validation

The UI validates the uploaded file before submitting:
- File extension must be `.vcf` or `.csv`
- File size limit: 50 MB (browser-side guard; not a security boundary)
- At least one sample row must be present

Validation errors are shown inline as Streamlit `st.error()` messages.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|---|---|---|---|
| UI framework | Streamlit | Gradio, Flask + React, Jupyter widget | Streamlit requires minimal code and is locally runnable; no JavaScript needed; Gradio is comparable but Streamlit has wider adoption for data apps |
| Prediction integration | Direct Python call (local) / HTTP (cloud) | Always HTTP, always in-process | Direct call eliminates network overhead for local dev; HTTP preserves cloud compatibility without duplicating logic |
| File input format | VCF or CSV | VCF only | CSV is more accessible for non-bioinformatician Business Analyst users testing with small datasets |

## Open Questions & Future Decisions

### Resolved
1. ✅ Streamlit chosen over Gradio for wider ecosystem and demo flexibility

### Deferred
1. Cloud deployment target for the Streamlit app (EC2, Streamlit Community Cloud, ECS) — out of scope until the pipeline itself is complete
2. Authentication for the UI — no auth in the demo; add if deployed beyond localhost

## References

- `docs/llds/prediction.md` — Python API and `PredictionResult` schema consumed by the UI
- `docs/llds/deployment.md` — Lambda API Gateway endpoint used in cloud mode
