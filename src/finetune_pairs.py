"""
Step 8 / Layer 7: Fine-Tuning Strategy.

Generates training pairs in the doc's format:

    {
      "question": "...",
      "metric_plan": {...},
      "expected_response": "..."
    }

Per the doc, fine-tuning should only improve three things: response style,
Xceler-specific terminology, and intent-to-metric consistency. So pairs are
NOT hand-written or LLM-hallucinated -- every `expected_response` is
generated deterministically from the *actual* metric_engine output for that
exact metric_plan against the real report. This keeps the training set
grounded and auditable: if the source report changes, regenerating this
file regenerates correct, consistent expected_response text alongside it.

Question text is produced from templates parameterized with real dimension
values pulled from the ingested data (real counterparty/commodity/route
names), so the question surface forms are varied but still map to a small,
well-understood set of metric plans -- exactly the "intent-to-metric
consistency" the doc wants fine-tuning to reinforce.
"""
from __future__ import annotations

import json
import random
from typing import List, Dict, Any

import pandas as pd

from ingestion import ingest
from metric_engine import MetricPlan, execute_plan, data_quality_gaps

random.seed(7)


def _detail(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["row_type"] == "detail"].copy()


def _top_values(df: pd.DataFrame, col: str, n: int) -> List[str]:
    d = _detail(df)
    counts = d[col].dropna().value_counts()
    return counts.head(n).index.tolist()


def _fmt(n: float) -> str:
    return f"{n:,.2f}"


# ---- Response template renderers (deterministic, grounded in computed metrics) ----

def _render_ranking_response(computed: Dict[str, Any], group_label: str) -> str:
    if not computed.get("breakdown"):
        return f"No data was found for this {group_label} ranking."
    top = computed["breakdown"][0]
    others = computed["breakdown"][1:3]
    parts = [
        f"{top['group']} leads by quantity at {_fmt(top['value'])}, "
        f"representing {top['share_of_total']:.1%} of the {_fmt(computed['total'])} total scoped volume."
    ]
    if others:
        trail = ", ".join(f"{o['group']} ({_fmt(o['value'])}, {o['share_of_total']:.1%})" for o in others)
        parts.append(f"Next: {trail}.")
    if computed.get("avg_completeness") is not None and computed["avg_completeness"] < 0.85:
        parts.append(f"Average data completeness for these rows is {computed['avg_completeness']:.0%}, so treat the ranking with some caution.")
    return " ".join(parts)


def _render_concentration_response(computed: Dict[str, Any], commodity: str) -> str:
    if not computed.get("breakdown") or computed["total"] == 0:
        return f"No shipment volume was found for {commodity} in this scope."
    top = computed["breakdown"][0]
    return (
        f"For {commodity}, the largest concentration is in {top['group']} at {_fmt(top['value'])}, "
        f"{top['share_of_total']:.1%} of the {_fmt(computed['total'])} total {commodity} volume scoped."
    )


def _render_route_response(computed: Dict[str, Any]) -> str:
    if not computed.get("breakdown"):
        return "No route activity was found."
    top = computed["breakdown"][0]
    return (
        f"The most active route is {top['group']} with {int(top['value'])} shipment line(s), "
        f"{top['share_of_total']:.1%} of all {int(computed['total'])} scoped shipment lines."
    )


def _render_quality_response(dq: Dict[str, Any]) -> str:
    flags = dq["flag_counts"]
    worst = max(flags, key=flags.get)
    worst_label = worst.replace("flag_missing_", "missing ").replace("_", " ")
    return (
        f"Out of {dq['total_detail_rows']} detail rows, average completeness is {dq['avg_completeness']:.0%}. "
        f"The largest gap is {worst_label} ({flags[worst]} rows), and {dq['duplicate_lines']} line(s) "
        f"are flagged as possible duplicates."
    )


def _render_contribution_response(computed: Dict[str, Any], dimension_label: str, value: str) -> str:
    if not computed.get("breakdown"):
        return f"No contribution data found for {dimension_label} '{value}'."
    match = next((b for b in computed["breakdown"] if b["group"].lower() == value.lower()), None)
    if not match:
        return f"{value} was not found among the scoped {dimension_label} values."
    return (
        f"{value} contributes {_fmt(match['value'])} ({match['share_of_total']:.1%}) "
        f"to the {_fmt(computed['total'])} total scoped {dimension_label} volume."
    )


# ---- Pair builders ----

def build_ranking_pairs(df: pd.DataFrame, n_per_dim: int = 3) -> List[Dict[str, Any]]:
    pairs = []
    templates = {
        "Counterparty": "Which counterparty shipped the most quantity?",
        "Commodity": "What is the top commodity by shipped quantity?",
        "Company": "Which company has the highest shipped volume?",
    }
    for dim, question in templates.items():
        plan = MetricPlan(metric="Quantity_Load", aggregation="SUM", group_by=dim, top_n=5)
        computed = execute_plan(df, plan)
        pairs.append({
            "question": question,
            "metric_plan": {"metric": "shipped_quantity", "comparison": "ranking", "dimensions": [dim.lower()]},
            "expected_response": _render_ranking_response(computed, dim.lower()),
        })
    return pairs


def build_concentration_pairs(df: pd.DataFrame, n: int = 4) -> List[Dict[str, Any]]:
    pairs = []
    for commodity in _top_values(df, "Commodity", n):
        question = f"What is the concentration of {commodity} by voyage?"
        plan = MetricPlan(metric="Quantity_Load", aggregation="SUM", group_by="Voyage_Number_norm",
                           filters={"Commodity": commodity}, top_n=5)
        computed = execute_plan(df, plan)
        pairs.append({
            "question": question,
            "metric_plan": {"metric": "shipped_quantity", "comparison": "concentration",
                             "dimensions": ["voyage"], "filters": {"commodity": commodity}},
            "expected_response": _render_concentration_response(computed, commodity),
        })
    return pairs


def build_route_pairs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    plan = MetricPlan(metric="Quantity_Load", aggregation="COUNT", group_by="Route", top_n=5)
    computed = execute_plan(df, plan)
    return [{
        "question": "Which routes are most active?",
        "metric_plan": {"metric": "shipment_line_count", "comparison": "ranking", "dimensions": ["route"]},
        "expected_response": _render_route_response(computed),
    }]


def build_quality_pairs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    dq = data_quality_gaps(df)
    return [{
        "question": "Where are the data quality gaps in this report?",
        "metric_plan": {"metric": "completeness_and_duplicate_flags", "comparison": "quality_check", "dimensions": []},
        "expected_response": _render_quality_response(dq),
    }]


def build_contribution_pairs(df: pd.DataFrame, n: int = 4) -> List[Dict[str, Any]]:
    pairs = []
    plan = MetricPlan(metric="Quantity_Load", aggregation="SUM", group_by="Counterparty", top_n=None)
    computed = execute_plan(df, plan)
    for counterparty in _top_values(df, "Counterparty", n):
        question = f"How much does {counterparty} contribute to total shipped volume?"
        pairs.append({
            "question": question,
            "metric_plan": {"metric": "shipped_quantity", "comparison": "contribution",
                             "dimensions": ["counterparty"], "filters": {"counterparty": counterparty}},
            "expected_response": _render_contribution_response(computed, "counterparty", counterparty),
        })
    return pairs


def generate_training_pairs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    pairs = []
    pairs += build_ranking_pairs(df)
    pairs += build_concentration_pairs(df)
    pairs += build_route_pairs(df)
    pairs += build_quality_pairs(df)
    pairs += build_contribution_pairs(df)
    random.shuffle(pairs)
    return pairs


def write_jsonl(pairs: List[Dict[str, Any]], path: str):
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def write_markdown_review(pairs: List[Dict[str, Any]], path: str):
    """Human-readable export for analyst review before these pairs are used
    in actual fine-tuning -- per the doc, fine-tuning should only shift
    style/terminology/intent-mapping, so a human should sanity-check tone
    here before training."""
    lines = ["# Xceler BI Chatbot — Fine-Tuning Training Pairs (for review)\n",
              f"{len(pairs)} pairs generated from the live report via the metric engine "
              "(grounded, not hallucinated).\n"]
    for i, p in enumerate(pairs, 1):
        lines.append(f"## {i}. {p['question']}\n")
        lines.append(f"**metric_plan**: `{json.dumps(p['metric_plan'])}`\n")
        lines.append(f"**expected_response**: {p['expected_response']}\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import sys
    df = ingest("data/LBL_and_GBL_Summary_Report.xlsx")
    pairs = generate_training_pairs(df)
    write_jsonl(pairs, "data/finetune_pairs.jsonl")
    write_markdown_review(pairs, "data/finetune_pairs_review.md")
    print(f"{len(pairs)} training pairs written to data/finetune_pairs.jsonl and data/finetune_pairs_review.md")
    for p in pairs[:3]:
        print(json.dumps(p, indent=2))
