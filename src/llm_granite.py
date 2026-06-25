"""
Layer 6: IBM Granite (local transformers pipeline, per the chosen HF Space
setup -- a small instruct model loaded directly in the Space).

Two responsibilities, matching the end-to-end runtime sequence in the doc:
  - Step 3: identify BI intent (comparison, trend, variance, ranking,
    contribution, concentration, route, quality-check)
  - Step 7/8: convert verified, pre-computed facts into a grounded,
    analyst-style narrative. The model is never asked to do arithmetic --
    only to phrase numbers that the metric engine already produced.

XCELER_LLM_BACKEND env var selects:
  - "granite" (default): loads GRANITE_LLM_MODEL via transformers on the
    Space's GPU.
  - "mock": rule-based stand-in used for local pipeline testing without
    network access to the Hub. Produces a templated, clearly-labeled
    response so the rest of the pipeline (retrieval, metric engine,
    guardian) can be validated end-to-end before the real model is wired
    in on the deployed Space.
"""
from __future__ import annotations

import os
import json
import re
from typing import Dict, Any, List

LLM_BACKEND = os.environ.get("XCELER_LLM_BACKEND", "granite")
GRANITE_LLM_MODEL = os.environ.get("XCELER_LLM_MODEL", "ibm-granite/granite-3.2-2b-instruct")

_pipe = None

INTENT_KEYWORDS = {
    "ranking": ["most", "top", "highest", "rank", "which counterparty", "biggest"],
    "concentration": ["concentration", "share", "percentage of", "dominant"],
    "trend": ["trend", "over time", "growth", "increase", "decrease"],
    "variance": ["versus", "vs", "compared to", "variance", "difference"],
    "route_analysis": ["route", "lane", "load location", "unload location", "most active"],
    "quality_check": ["missing", "data quality", "gap", "duplicate", "incomplete"],
    "contribution": ["contribute", "contribution"],
}


INTENT_PRIORITY = ["quality_check", "route_analysis", "concentration", "variance", "trend", "ranking", "contribution"]


def classify_intent(question: str) -> str:
    """Step 3: lightweight keyword-based intent classifier. On the deployed
    Space this is cheap and reliable; Granite is reserved for the final
    narrative generation step rather than intent tagging, to keep latency
    and GPU load down for a 2B model. Checked in a fixed priority order so
    compound phrases like "most active routes" resolve to route_analysis
    rather than the more generic "ranking" keyword "most"."""
    q = question.lower()
    for intent in INTENT_PRIORITY:
        kws = INTENT_KEYWORDS[intent]
        if any(kw in q for kw in kws):
            return intent
    return "general_inquiry"


def _load_granite_pipeline():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline(
            "text-generation",
            model=GRANITE_LLM_MODEL,
            device_map="auto",
            torch_dtype="auto",
        )
    return _pipe


SYSTEM_PROMPT = (
    "You are the Xceler BI analyst assistant. You explain shipment and "
    "trading report data in clear business language. You NEVER perform or "
    "invent arithmetic -- you only narrate the numbers given to you in "
    "'Computed metrics'. If completeness or duplicate flags indicate weak "
    "data quality, mention that limitation. Keep responses to 2-4 sentences "
    "unless the question asks for a breakdown."
)


def _build_prompt(question: str, schema_context: List[Dict[str, Any]],
                   history_context: List[Dict[str, Any]], computed: Dict[str, Any]) -> List[Dict[str, str]]:
    schema_text = "\n".join(f"- {c['text']}" for c in schema_context) or "none"
    history_text = "\n".join(f"- {c['text']}" for c in history_context) or "none"
    user_content = (
        f"Question: {question}\n\n"
        f"Retrieved schema definitions:\n{schema_text}\n\n"
        f"Retrieved historical context:\n{history_text}\n\n"
        f"Computed metrics (already calculated, do not recompute):\n{json.dumps(computed, default=str, indent=2)}\n\n"
        "Write the analyst-style answer now."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _mock_generate(question: str, computed: Dict[str, Any]) -> str:
    total = computed.get("total")
    breakdown = computed.get("breakdown")
    completeness = computed.get("avg_completeness")
    parts = [f"[mock-llm backend, install transformers + set XCELER_LLM_BACKEND=granite for real generation]"]
    if total is not None:
        parts.append(f"Total for the scoped question: {total:,.2f}.")
    if breakdown:
        top = breakdown[0]
        parts.append(f"Top contributor: {top['group']} at {top['value']:,.2f} ({top['share_of_total']:.1%} share).")
    if completeness is not None and completeness < 0.8:
        parts.append(f"Note: average data completeness for these rows is {completeness:.0%}, so treat this with some caution.")
    return " ".join(parts)


def generate_insight(question: str, schema_context: List[Dict[str, Any]],
                      history_context: List[Dict[str, Any]], computed: Dict[str, Any]) -> str:
    """Step 7/8: turn verified facts into a grounded narrative."""
    if LLM_BACKEND == "mock":
        return _mock_generate(question, computed)

    pipe = _load_granite_pipeline()
    messages = _build_prompt(question, schema_context, history_context, computed)
    out = pipe(messages, max_new_tokens=300, do_sample=False)
    text = out[0]["generated_text"]
    if isinstance(text, list):  # chat-style pipelines return the full message list
        text = text[-1]["content"]
    return text.strip()


if __name__ == "__main__":
    print("LLM backend:", LLM_BACKEND)
    print("intent:", classify_intent("Which counterparty shipped the most crude palm oil?"))
    print(_mock_generate("test", {"total": 1000, "breakdown": [{"group": "AAA", "value": 600, "share_of_total": 0.6}], "avg_completeness": 0.7}))
