---
title: Xceler BI Chatbot POC
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
pinned: false
license: other
---

# Xceler BI Chatbot — POC

A governed, retrieval-grounded BI chatbot for the LBL/GBL Summary Report
(or any similarly-structured CTRM shipment report), built per the Xceler
BI Chatbot Technical Design doc. All arithmetic is computed deterministically
in Python; the LLM only narrates verified, pre-computed facts.

## Architecture (matches the design doc's 8 layers)

| Layer | Module | What it does |
|---|---|---|
| 1. BI Semantic Layer | `src/semantic_registry.py` | Business definitions, types, grain, aggregation rules per field |
| 2. Metric Computation Engine | `src/metric_engine.py` | All totals, rankings, variance, reconciliation — deterministic, auditable |
| 3. Schema RAG | `src/semantic_registry.py` + `src/vector_store.py` | Retrieves field/KPI definitions before answering |
| 4. Historical RAG | `src/historical_rag.py` + `src/vector_store.py` | Business-grain chunks: voyage, contract, route, counterparty/company/commodity rollups |
| 5. Embedding Layer | `src/embeddings.py` | IBM Granite embeddings (`granite-embedding-125m-english`), FAISS in-memory index |
| 6. IBM Granite | `src/llm_granite.py` | Local transformers pipeline (`granite-3.2-2b-instruct`), narrates computed facts |
| 7. Fine-Tuning | *(not in this POC)* | See `Step 8` of the design doc for the training-pair format to extend later |
| 8. Granite Guardian | `src/guardian.py` | Screens input (Step 2) and output (Step 9) for injection / unsafe / out-of-scope content |

`src/pipeline.py` orchestrates the full 10-step end-to-end runtime sequence
from the design doc, and `app.py` wraps it in a Gradio chat UI.

## Running on Hugging Face Spaces

1. **Hardware**: this Space loads three local models at once
   (LLM + Guardian + embeddings). Use at least a **T4 small** GPU Space —
   CPU-only will be very slow and a free CPU Space may time out on first
   load. Set this under Space settings → Hardware.
2. Push this folder's contents to your Space repo (`app.py`,
   `requirements.txt`, `README.md`, `src/`, `data/`), or use
   `huggingface-cli upload <space-id> . --repo-type=space`.
3. The default sample report (`data/LBL_and_GBL_Summary_Report.xlsx`)
   loads automatically on startup. Users can upload a different
   LBL/GBL-style report from the UI to re-run ingestion against it.
4. First request will be slow while the three models load and the report
   is embedded; subsequent requests reuse the loaded models and index.

### Known issue: `TypeError: argument of type 'bool' is not iterable` on startup

This is a version-compatibility bug between `gradio_client`'s OpenAPI
schema parser and newer `pydantic` releases, which started emitting
`additionalProperties: true` as a bare boolean instead of a schema object.
It throws on the `/info` (API docs) endpoint specifically — the chat UI
itself still comes up and works — but it's noisy and can break
programmatic API access to the Space. `requirements.txt` pins
`gradio==4.44.1`, `gradio_client==1.3.0`, `pydantic==2.10.6`, and
`huggingface_hub==0.25.2` together, which is the verified-working
combination (confirmed by reproducing and fixing the exact error locally).
If you bump any of these, re-test `demo.get_api_info()` doesn't throw
before deploying.

## Environment variables (optional overrides)

| Variable | Default | Purpose |
|---|---|---|
| `XCELER_EMBED_BACKEND` | `granite` | `granite` or `hash` (offline dev fallback, not semantically meaningful) |
| `XCELER_EMBED_MODEL` | `ibm-granite/granite-embedding-125m-english` | Embedding model id |
| `XCELER_LLM_BACKEND` | `granite` | `granite` or `mock` (offline dev fallback, templated output) |
| `XCELER_LLM_MODEL` | `ibm-granite/granite-3.2-2b-instruct` | Generation model id |
| `XCELER_GUARDIAN_BACKEND` | `granite` | `granite` or `rule_based` (regex pre-filter only, cheaper, less coverage) |
| `XCELER_GUARDIAN_MODEL` | `ibm-granite/granite-guardian-3.1-2b` | Guardian model id |

For a **fast first deploy / smoke test before committing GPU budget**, set
all three backends to their offline fallback values
(`XCELER_EMBED_BACKEND=hash`, `XCELER_LLM_BACKEND=mock`,
`XCELER_GUARDIAN_BACKEND=rule_based`) on a free CPU Space. This validates
ingestion, the metric engine, retrieval wiring, and the Gradio UI without
needing GPU hardware or Hub network access for model downloads. Switch
back to the `granite` backends once you're ready to validate real
generation quality.

## Local development

This was built and tested in a sandboxed environment without network
access to huggingface.co, using the offline fallback backends above
against the real uploaded report (`data/LBL_and_GBL_Summary_Report.xlsx`).
Verified end-to-end: ingestion (782 rows → 372 detail / 284 subtotal / 126
total rows correctly classified), reconciliation against the report's own
Sub Total / Total control rows (caught one real data discrepancy — contract
`0/MALAYSIA/S//24/2304` reports a Sub Total of 4,884 vs. a computed line
sum of 4,884,100, worth checking against source), schema + historical RAG
retrieval, the metric engine, and the Gradio chat loop. Swap the three
backend env vars to `granite` (the default) before relying on answer
quality — the offline fallbacks exist only to validate plumbing without
Hub access.

```bash
pip install -r requirements.txt
python app.py
```

## Known limitations of this POC

- **No date/period column** in the source report, so Historical RAG
  produces business-grain summaries (voyage/contract/route/dimension
  rollups) rather than the doc's monthly/quarterly time-series chunks. If
  a periodized version of the report becomes available, extend
  `src/historical_rag.py` with month-over-month and rolling-average
  chunks as shown in the design doc's Step 5 example.
- **NL → metric plan resolution** (`src/pipeline.py:resolve_metric_plan`)
  is a small, transparent rule-based resolver, not a full NLU system. It
  covers the POC's four target question types (ranking, concentration,
  route activity, data-quality gaps) — extend the keyword/dimension maps
  for more question patterns, or replace with a Granite-based
  question → metric-plan classifier per the doc's fine-tuning section
  once you have training pairs.

## Fine-tuning training pairs (Step 8 / Layer 7)

`src/finetune_pairs.py` generates training pairs in the doc's
`{question, metric_plan, expected_response}` format. Per the doc,
fine-tuning should only shift response style, terminology, and
intent-to-metric consistency — so `expected_response` text is **never**
hand-written or LLM-generated; it's rendered deterministically from the
real metric_engine output for that exact plan against the live report.
Regenerating this file after the source report changes keeps the
training set grounded and consistent automatically.

```bash
python src/finetune_pairs.py
# writes data/finetune_pairs.jsonl   (machine-readable training pairs)
#        data/finetune_pairs_review.md (human-readable review doc)
```

Review `finetune_pairs_review.md` before using the pairs for actual
fine-tuning — a human should sanity-check tone/phrasing even though the
numbers are guaranteed correct. Extend the template functions in that
file (`build_ranking_pairs`, `build_concentration_pairs`, etc.) to cover
more question patterns or dimensions as the chatbot's scope grows.
