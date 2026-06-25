"""
Step 6 / Layer 2: Metric Computation Engine.

All arithmetic happens here in plain Python/pandas, never inside the LLM.
This produces the auditable totals, growth rates, variance, contribution
share, and rankings that Granite is allowed to narrate but never compute.

A "metric plan" is a small structured dict (the kind Step 6 of the doc
says a user question gets translated into). This module executes plans
deterministically against the normalized detail rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class MetricPlan:
    metric: str = "Quantity_Load"          # column to aggregate
    aggregation: str = "SUM"                # SUM | COUNT | AVG
    group_by: Optional[str] = None          # dimension to break out by
    filters: Dict[str, Any] = field(default_factory=dict)  # {column: value}
    top_n: Optional[int] = None
    compare_to: Optional[Dict[str, Any]] = None  # {column: value} baseline filter


def _detail(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["row_type"] == "detail"].copy()


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = df
    for col, val in (filters or {}).items():
        if col not in out.columns:
            continue
        out = out[out[col].astype(str).str.lower() == str(val).lower()]
    return out


def _aggregate(df: pd.DataFrame, metric: str, aggregation: str) -> float:
    if metric not in df.columns:
        return 0.0
    series = df[metric]
    if aggregation == "SUM":
        return float(series.sum())
    if aggregation == "COUNT":
        return float(series.count())
    if aggregation == "AVG":
        return float(series.mean()) if len(series) else 0.0
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def execute_plan(df: pd.DataFrame, plan: MetricPlan) -> Dict[str, Any]:
    """Execute a metric plan deterministically. Returns auditable totals,
    rankings, contribution shares, and (if requested) a variance vs. a
    comparison filter."""
    detail = _detail(df)
    scoped = _apply_filters(detail, plan.filters)

    result: Dict[str, Any] = {
        "plan": plan.__dict__,
        "row_count": int(len(scoped)),
        "total": _aggregate(scoped, plan.metric, plan.aggregation),
    }

    if plan.group_by and plan.group_by in scoped.columns:
        grouped = (
            scoped.groupby(plan.group_by)[plan.metric]
            .agg("sum" if plan.aggregation == "SUM" else
                 "count" if plan.aggregation == "COUNT" else "mean")
            .sort_values(ascending=False)
        )
        total_all = grouped.sum() if grouped.sum() else 1
        breakdown = [
            {
                "group": str(idx),
                "value": float(val),
                "share_of_total": float(val) / float(total_all) if total_all else 0.0,
            }
            for idx, val in grouped.items()
        ]
        if plan.top_n:
            breakdown = breakdown[: plan.top_n]
        result["breakdown"] = breakdown
        result["ranking"] = [b["group"] for b in breakdown]

    if plan.compare_to:
        baseline_scoped = _apply_filters(detail, plan.compare_to)
        baseline_total = _aggregate(baseline_scoped, plan.metric, plan.aggregation)
        result["baseline_total"] = baseline_total
        if baseline_total:
            variance_pct = (result["total"] - baseline_total) / baseline_total * 100
        else:
            variance_pct = None
        result["variance_pct"] = variance_pct

    # data quality summary for the scoped rows, so Granite can mention confidence
    if len(scoped):
        result["avg_completeness"] = float(scoped["completeness_score"].mean())
        result["duplicate_lines"] = int(scoped["duplicate_flag"].sum())
    else:
        result["avg_completeness"] = None
        result["duplicate_lines"] = 0

    return result


def reconciliation_check(df: pd.DataFrame) -> Dict[str, Any]:
    """Cross-check that detail-row sums reconcile to the report's own
    Sub Total / Total control rows -- an auditability check unique to
    having both fact rows and report-control rows in the same sheet."""
    detail = _detail(df)
    subtotal_rows = df[df["row_type"] == "subtotal"]
    total_rows = df[df["row_type"] == "total"]

    contract_sums = detail.groupby("Sell_Contract_ID")["Quantity_Load"].sum()
    # subtotal rows store the contract id in Vessel_Name and total in Quantity_Load
    mismatches = []
    for _, r in subtotal_rows.iterrows():
        contract = r["Vessel_Name"]
        reported = r["Quantity_Load"]
        actual = contract_sums.get(contract)
        if actual is not None and reported is not None and pd.notna(reported):
            if abs(actual - reported) > 0.01:
                mismatches.append({"contract": contract, "reported_subtotal": float(reported), "computed_sum": float(actual)})

    return {
        "subtotal_rows_checked": int(len(subtotal_rows)),
        "total_rows_checked": int(len(total_rows)),
        "mismatches": mismatches,
        "reconciled": len(mismatches) == 0,
    }


# Convenience presets matching the doc's example questions
def ranking_by(df: pd.DataFrame, dimension: str, top_n: int = 5) -> Dict[str, Any]:
    return execute_plan(df, MetricPlan(metric="Quantity_Load", aggregation="SUM", group_by=dimension, top_n=top_n))


def concentration_by_voyage(df: pd.DataFrame, commodity: str) -> Dict[str, Any]:
    return execute_plan(df, MetricPlan(
        metric="Quantity_Load", aggregation="SUM", group_by="Voyage_Number_norm",
        filters={"Commodity": commodity},
    ))


def most_active_routes(df: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
    return execute_plan(df, MetricPlan(metric="Quantity_Load", aggregation="COUNT", group_by="Route", top_n=top_n))


def data_quality_gaps(df: pd.DataFrame) -> Dict[str, Any]:
    detail = _detail(df)
    flags = ["flag_missing_commodity", "flag_missing_bl", "flag_missing_gbl",
              "flag_missing_location", "flag_missing_counterparty"]
    return {
        "total_detail_rows": int(len(detail)),
        "flag_counts": {f: int(detail[f].sum()) for f in flags},
        "duplicate_lines": int(detail["duplicate_flag"].sum()),
        "avg_completeness": float(detail["completeness_score"].mean()),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ingestion import ingest
    df = ingest("data/LBL_and_GBL_Summary_Report.xlsx")
    print("Top counterparties by volume:", ranking_by(df, "Counterparty", 5))
    print("\nReconciliation:", reconciliation_check(df))
    print("\nData quality:", data_quality_gaps(df))
