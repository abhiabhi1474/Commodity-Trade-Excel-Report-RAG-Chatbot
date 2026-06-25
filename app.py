"""
Xceler BI Chatbot POC -- Gradio app for HF Spaces.

Implements the full 10-step end-to-end runtime sequence from the design
doc: ingest -> Guardian input screen -> intent -> Schema RAG -> Historical
RAG -> deterministic metric engine -> Granite narrative -> Guardian output
screen -> answer.

Local model weights are loaded directly into the Space process (transformers
pipelines), so this Space needs a GPU runtime (T4 small or larger) to run
the LLM + Guardian + embedding models together at reasonable latency.
"""
import os
import sys
import traceback

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import XcelerBIPipeline  # noqa: E402

pipeline = XcelerBIPipeline()
DEFAULT_REPORT_PATH = os.path.join(os.path.dirname(__file__), "data", "LBL_and_GBL_Summary_Report.xlsx")

EXAMPLE_QUESTIONS = [
    "Which counterparty shipped the most quantity?",
    "What is the concentration of crude palm oil by voyage?",
    "Which routes are most active?",
    "Where are the data quality gaps?",
]


def load_default_report():
    if os.path.exists(DEFAULT_REPORT_PATH):
        try:
            stats = pipeline.load_report(DEFAULT_REPORT_PATH)
            return _format_load_summary(stats)
        except Exception as e:  # noqa: BLE001
            return f"Failed to load default report: {e}"
    return "No report loaded yet. Upload an LBL/GBL-style Excel report below."


def _format_load_summary(stats: dict) -> str:
    rec = stats["reconciliation"]
    dq = stats["data_quality"]
    if rec["reconciled"]:
        rec_line = "✅ all matched"
    else:
        rec_line = f"⚠️ {len(rec['mismatches'])} mismatch(es) found"
    lines = [
        f"**Report ingested.** {stats['rows_ingested']} rows read, {stats['detail_rows']} detail (fact) rows.",
        f"Schema RAG chunks: {stats['schema_chunks']} | Historical RAG chunks: {stats['historical_chunks']}",
        f"Reconciliation vs. report Sub Total / Total control rows: {rec_line} "
        f"({rec['subtotal_rows_checked']} subtotals, {rec['total_rows_checked']} totals checked).",
        f"Average row completeness: {dq['avg_completeness']:.0%} | Duplicate lines flagged: {dq['duplicate_lines']}",
    ]
    return "\n\n".join(lines)


def handle_upload(file):
    if file is None:
        return "No file uploaded."
    try:
        stats = pipeline.load_report(file.name)
        return _format_load_summary(stats)
    except Exception as e:  # noqa: BLE001
        return f"Ingestion failed: {e}\n\n{traceback.format_exc()}"


def chat(message, history):
    if pipeline.df is None:
        return "Please load a report first (default report loads automatically, or upload your own above)."
    try:
        result = pipeline.ask(message)
    except Exception as e:  # noqa: BLE001
        return f"Pipeline error: {e}"

    if not result.allowed:
        return f"⚠️ Request blocked: {result.blocked_reason}"

    debug = (
        f"\n\n<sub>intent: `{result.intent}` · metric plan: `{result.metric_plan}` · "
        f"schema hits: {len(result.schema_context or [])} · history hits: {len(result.history_context or [])}</sub>"
    )
    return result.answer + debug


with gr.Blocks(title="Xceler BI Chatbot POC") as demo:
    gr.Markdown("# Xceler BI Chatbot — POC\nAsk natural-language questions about the LBL/GBL shipment report.")

    with gr.Row():
        upload = gr.File(label="Upload a different LBL/GBL-style Excel report", file_types=[".xlsx"])
    load_status = gr.Markdown()

    chatbot = gr.ChatInterface(
        fn=chat,
        examples=EXAMPLE_QUESTIONS,
        title=None,
    )

    demo.load(fn=load_default_report, outputs=load_status)
    upload.change(fn=handle_upload, inputs=upload, outputs=load_status)

if __name__ == "__main__":
    demo.launch()
