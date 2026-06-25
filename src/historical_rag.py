"""
Step 5 / Layer 4: Generate historical summaries (Historical RAG).

Per the doc's chunking strategy, this is NOT fixed-window chunking. Chunks
are business-grain: one chunk per voyage, one per contract, one per route,
plus rollups by counterparty / company / commodity. Each chunk keeps the
underlying detail rows together with the subtotal/total so a chatbot
answer about volume, splits, or missing fields never loses context.

Note: the source report has no date/period column, so this module produces
business-grain summaries (voyage / contract / route / counterparty /
company / commodity) rather than monthly/quarterly time-series chunks. If
a periodized version of the report becomes available (a reporting date or
period column), extend `generate_period_summaries()` to add true
month-over-month / rolling-average baselines as shown in the design doc.
"""
from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd


def _detail(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["row_type"] == "detail"].copy()


def generate_voyage_chunks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    d = _detail(df)
    chunks = []
    for (vessel, voyage), g in d.groupby(["Vessel_Name_norm", "Voyage_Number_norm"], dropna=False):
        total_qty = g["Quantity_Load"].sum()
        commodities = sorted(c for c in g["Commodity"].dropna().unique())
        counterparties = sorted(c for c in g["Counterparty"].dropna().unique())
        routes = sorted(g["Route"].dropna().unique())
        avg_completeness = g["completeness_score"].mean()
        text = (
            f"Voyage {voyage} on vessel {vessel}: {len(g)} shipment lines, "
            f"total quantity {total_qty:.2f} across commodities {commodities or 'unspecified'}, "
            f"counterparties {counterparties or 'unspecified'}, routes {routes or 'unspecified'}. "
            f"Average data completeness {avg_completeness:.0%}."
        )
        chunks.append({
            "chunk_type": "voyage_summary",
            "vessel": vessel, "voyage": voyage,
            "total_quantity": float(total_qty), "line_count": int(len(g)),
            "commodities": commodities, "counterparties": counterparties, "routes": routes,
            "completeness": float(avg_completeness),
            "row_indices": g["source_row"].tolist(),
            "text": text,
        })
    return chunks


def generate_contract_chunks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    d = _detail(df)
    chunks = []
    for contract, g in d.groupby("Sell_Contract_ID", dropna=False):
        if pd.isna(contract):
            continue
        total_qty = g["Quantity_Load"].sum()
        split_count = len(g)
        bls = sorted(b for b in g["BL_Number"].dropna().unique())
        dup_flag = bool(g["duplicate_flag"].any())
        text = (
            f"Contract {contract}: {split_count} BL split line(s) ({bls or 'no BL recorded'}), "
            f"total quantity {total_qty:.2f}, counterparty {g['Counterparty'].dropna().unique().tolist()}, "
            f"commodity {g['Commodity'].dropna().unique().tolist()}. "
            f"{'Possible duplicate line(s) detected.' if dup_flag else 'No duplicates detected.'}"
        )
        chunks.append({
            "chunk_type": "contract_summary",
            "contract": contract, "total_quantity": float(total_qty),
            "split_count": split_count, "bl_numbers": bls, "duplicate_flag": dup_flag,
            "row_indices": g["source_row"].tolist(),
            "text": text,
        })
    return chunks


def generate_route_chunks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    d = _detail(df)
    chunks = []
    for route, g in d.groupby("Route", dropna=False):
        total_qty = g["Quantity_Load"].sum()
        commodities = sorted(c for c in g["Commodity"].dropna().unique())
        text = (
            f"Route {route}: {len(g)} shipment line(s), total quantity {total_qty:.2f}, "
            f"commodities moved: {commodities or 'unspecified'}."
        )
        chunks.append({
            "chunk_type": "route_summary",
            "route": route, "total_quantity": float(total_qty), "line_count": int(len(g)),
            "commodities": commodities,
            "row_indices": g["source_row"].tolist(),
            "text": text,
        })
    return chunks


def generate_dimension_rollups(df: pd.DataFrame, dimension: str) -> List[Dict[str, Any]]:
    """Generic rollup for Counterparty / Company / Commodity."""
    d = _detail(df)
    chunks = []
    for value, g in d.groupby(dimension, dropna=False):
        if pd.isna(value):
            continue
        total_qty = g["Quantity_Load"].sum()
        share = total_qty / d["Quantity_Load"].sum() if d["Quantity_Load"].sum() else 0
        text = (
            f"{dimension} '{value}': {len(g)} shipment line(s), total quantity {total_qty:.2f}, "
            f"representing {share:.1%} of total reported quantity."
        )
        chunks.append({
            "chunk_type": f"{dimension.lower()}_rollup",
            dimension: value, "total_quantity": float(total_qty),
            "line_count": int(len(g)), "share_of_total": float(share),
            "row_indices": g["source_row"].tolist(),
            "text": text,
        })
    return chunks


def generate_all_historical_chunks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    chunks = []
    chunks += generate_voyage_chunks(df)
    chunks += generate_contract_chunks(df)
    chunks += generate_route_chunks(df)
    for dim in ["Counterparty", "Company", "Commodity"]:
        chunks += generate_dimension_rollups(df, dim)
    return chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ingestion import ingest
    df = ingest("data/LBL_and_GBL_Summary_Report.xlsx")
    chunks = generate_all_historical_chunks(df)
    print(f"{len(chunks)} historical chunks generated")
    for c in chunks[:3]:
        print(c["text"])
