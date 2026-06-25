"""


 1. User submits a question
 2. Granite Guardian screens the prompt
 3. Granite identifies the BI intent
 4. Schema layer / Schema RAG resolves field meaning
 5. Historical RAG retrieves baselines / prior context
 6. Metric engine computes all numbers deterministically
 7. Granite converts verified facts into an insight
 8. (fine-tuning would happen offline, not part of runtime)
 9. Granite Guardian validates the output
10. Final answer is returned
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd

from ingestion import ingest
from semantic_registry import schema_chunks
from historical_rag import generate_all_historical_chunks
from vector_store import XcelerVectorStore
from metric_engine import MetricPlan, execute_plan, reconciliation_check, data_quality_gaps
import guardian
import llm_granite


KNOWN_DIMENSIONS = [
    "Counterparty", "Commodity", "Company", "Route",
    "Vessel_Name_norm", "Voyage_Number_norm", "Sell_Contract_ID",
]
DIMENSION_ALIASES = {
    "counterparty": "Counterparty", "counterparties": "Counterparty", "buyer": "Counterparty",
    "commodity": "Commodity", "commodities": "Commodity", "product": "Commodity",
    "company": "Company",
    "route": "Route", "routes": "Route", "lane": "Route",
    "vessel": "Vessel_Name_norm", "ship": "Vessel_Name_norm",
    "voyage": "Voyage_Number_norm",
    "contract": "Sell_Contract_ID",
}


def resolve_metric_plan(question: str, intent: str) -> MetricPlan:
    """Very small, transparent rule-based NL -> MetricPlan resolver. Keeps
    the metric engine fully deterministic and auditable: the LLM is never
    asked to decide the arithmetic, only (elsewhere) to phrase the result.
    Extend this resolver as more question patterns are needed."""
    q = question.lower()
    group_by = None
    for alias, col in DIMENSION_ALIASES.items():
        if alias in q:
            group_by = col
            break

    aggregation = "SUM"
    if intent == "route_analysis" and "most active" in q:
        aggregation = "COUNT"
        group_by = group_by or "Route"
    if intent == "ranking" and not group_by:
        group_by = "Counterparty"
    if intent == "concentration" and not group_by:
        group_by = "Commodity"

    top_n = 5
    m = re.search(r"top (\d+)", q)
    if m:
        top_n = int(m.group(1))

    filters = {}
    # naive commodity filter: look for known commodity-ish phrase after "of"/"for",
    # stopping before trailing clauses like "by voyage" or "by counterparty"
    m2 = re.search(r"(?:of|for) ([a-zA-Z ]+?)(?:\s+by\b|\?|$)", question)
    if m2 and intent == "concentration":
        filters["Commodity"] = m2.group(1).strip()

    return MetricPlan(
        metric="Quantity_Load",
        aggregation=aggregation,
        group_by=group_by,
        filters=filters,
        top_n=top_n if group_by else None,
    )


@dataclass
class PipelineResult:
    question: str
    allowed: bool
    intent: Optional[str] = None
    metric_plan: Optional[Dict[str, Any]] = None
    computed: Optional[Dict[str, Any]] = None
    schema_context: Optional[List[Dict[str, Any]]] = None
    history_context: Optional[List[Dict[str, Any]]] = None
    answer: str = ""
    blocked_reason: Optional[str] = None


class XcelerBIPipeline:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.store = XcelerVectorStore()
        self.reconciliation: Optional[Dict[str, Any]] = None
        self.quality: Optional[Dict[str, Any]] = None

    def load_report(self, path: str, sheet_name=0):
        """Steps 1-3 (ingest/classify/normalize) + Step 4-5 (build schema and
        historical RAG sources) + embedding into the vector store."""
        self.df = ingest(path, sheet_name=sheet_name)
        chunks_schema = schema_chunks()
        chunks_history = generate_all_historical_chunks(self.df)
        self.store.build(chunks_schema, chunks_history)
        self.reconciliation = reconciliation_check(self.df)
        self.quality = data_quality_gaps(self.df)
        return {
            "rows_ingested": int(len(self.df)),
            "detail_rows": int((self.df["row_type"] == "detail").sum()),
            "schema_chunks": len(chunks_schema),
            "historical_chunks": len(chunks_history),
            "reconciliation": self.reconciliation,
            "data_quality": self.quality,
        }

    def ask(self, question: str) -> PipelineResult:
        if self.df is None:
            return PipelineResult(question=question, allowed=False, blocked_reason="No report loaded yet.")

        # Step 2: Guardian screens input
        input_check = guardian.screen_input(question)
        if not input_check.allowed:
            return PipelineResult(question=question, allowed=False, blocked_reason=input_check.reason)

        # Step 3: Granite identifies BI intent
        intent = llm_granite.classify_intent(question)

        # Step 4: Schema RAG resolves field meaning
        schema_context = self.store.retrieve_schema(question, k=4)

        # Step 5: Historical RAG retrieves context
        history_context = self.store.retrieve_history(question, k=5)

        # Step 6: Metric engine computes all numbers deterministically
        plan = resolve_metric_plan(question, intent)
        computed = execute_plan(self.df, plan)

        # Step 7: Granite converts verified facts into a narrative
        answer = llm_granite.generate_insight(question, schema_context, history_context, computed)

        # Step 9: Guardian validates the output
        grounding = f"computed_metrics={computed}"
        output_check = guardian.screen_output(answer, grounding_context=grounding)
        if not output_check.allowed:
            return PipelineResult(
                question=question, allowed=False, intent=intent,
                metric_plan=plan.__dict__, computed=computed,
                schema_context=schema_context, history_context=history_context,
                blocked_reason=output_check.reason,
            )

        # Step 10: final answer
        return PipelineResult(
            question=question, allowed=True, intent=intent,
            metric_plan=plan.__dict__, computed=computed,
            schema_context=schema_context, history_context=history_context,
            answer=answer,
        )


if __name__ == "__main__":
    pipe = XcelerBIPipeline()
    stats = pipe.load_report("data/LBL_and_GBL_Summary_Report.xlsx")
    print("Load stats:", stats)
    for q in [
        "Which counterparty shipped the most quantity?",
        "What is the concentration of crude palm oil by voyage?",
        "Which routes are most active?",
        "Where are the data quality gaps?",
    ]:
        r = pipe.ask(q)
        print(f"\nQ: {q}\nintent={r.intent} allowed={r.allowed}\nA: {r.answer or r.blocked_reason}")
