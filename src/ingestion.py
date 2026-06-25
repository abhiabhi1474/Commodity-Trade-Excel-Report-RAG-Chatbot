"""
Step 1-3 of the Xceler BI Chatbot POC: ingest the Excel report, classify rows
(detail / subtotal / total), and normalize fields.

Built against the real LBL & GBL Summary Report structure:
- Title row(s) before the real header (header is NOT assumed to be row 1)
- Columns: Vessel_Name, Voyage_Number, Sell contract ID, Counterparty,
  Commodity, Quantity Load, Uom, BL_Number, GBL_Number, Load_Location,
  Unload_Location, Company
- Subtotal rows: 'Sub Total' appears in the Counterparty column, the
  contract id is shifted into the Vessel_Name column, quantity is in
  Quantity Load
- Total rows: 'Total' appears in the Voyage_Number column
- Vessel_Name / Voyage_Number act like merged cells: a real value appears
  once then is blank or '0' on subsequent rows of the same group
"""
from __future__ import annotations

import pandas as pd
import numpy as np

EXPECTED_COLUMNS = [
    "Vessel_Name", "Voyage_Number", "Sell contract ID", "Counterparty",
    "Commodity", "Quantity Load", "Uom", "BL_Number", "GBL_Number",
    "Load_Location", "Unload_Location", "Company",
]

CANONICAL_RENAME = {
    "Sell contract ID": "Sell_Contract_ID",
    "Quantity Load": "Quantity_Load",
}

NULL_MARKERS = {"", "na", "n/a", "none", "null", "0", "nan", None}


def _is_null_marker(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip().lower() in NULL_MARKERS


def find_header_row(path: str, sheet_name=0, max_scan_rows: int = 15) -> int:
    """Locate the real header row instead of assuming row 0 is the header."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    for i, row in raw.iterrows():
        values = {str(v).strip() for v in row.tolist() if pd.notna(v)}
        if {"Vessel_Name", "Counterparty", "Commodity"}.issubset(values):
            return i
    raise ValueError("Could not locate header row containing expected report columns")


def load_report(path: str, sheet_name=0) -> pd.DataFrame:
    """Step 1: Ingest the Excel file with correct header detection."""
    header_row = find_header_row(path, sheet_name=sheet_name)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    df = df.dropna(how="all")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Report is missing expected columns: {missing}")
    df = df[EXPECTED_COLUMNS].rename(columns=CANONICAL_RENAME)
    df = df.reset_index(drop=True)
    df.insert(0, "source_row", df.index + header_row + 2)  # 1-indexed Excel row
    return df


def classify_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Classify each row as detail, subtotal, or total.

    Report-control patterns (confirmed against the real file):
    - total row:    Voyage_Number == 'Total'
    - subtotal row: Counterparty == 'Sub Total'
    - detail row:   everything else with a populated Sell_Contract_ID
    - blank row:    fully empty trailing rows, dropped
    """
    df = df.copy()

    def classify(r):
        voyage = str(r["Voyage_Number"]).strip() if pd.notna(r["Voyage_Number"]) else ""
        counterparty = str(r["Counterparty"]).strip() if pd.notna(r["Counterparty"]) else ""
        if voyage == "Total":
            return "total"
        if counterparty == "Sub Total":
            return "subtotal"
        if all(_is_null_marker(v) for v in r.drop(labels=["source_row"]).tolist()):
            return "blank"
        if not _is_null_marker(r.get("Sell_Contract_ID")):
            return "detail"
        return "unclassified"

    df["row_type"] = df.apply(classify, axis=1)
    df = df[df["row_type"] != "blank"].reset_index(drop=True)
    return df


def normalize_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Normalize fields, forward-fill grouped identifiers, derive
    Route, and raise data-quality flags."""
    df = df.copy()

    # Vessel_Name / Voyage_Number behave like merged cells on detail rows:
    # a real value appears once, then is blank/'0' until it changes.
    for col in ["Vessel_Name", "Voyage_Number"]:
        is_real_detail = (df["row_type"] == "detail") & df[col].apply(lambda v: not _is_null_marker(v))
        df[col + "_clean"] = df[col].where(is_real_detail)
        ffilled = df[col + "_clean"].ffill()
        df[col + "_norm"] = np.where(df["row_type"] == "detail", ffilled, df[col])
        df.drop(columns=[col + "_clean"], inplace=True)

    df["Quantity_Load"] = pd.to_numeric(df["Quantity_Load"], errors="coerce")

    def _clean_str(v):
        if _is_null_marker(v):
            return None
        return str(v).strip()

    for col in ["Counterparty", "Commodity", "Uom", "BL_Number", "GBL_Number",
                "Load_Location", "Unload_Location", "Company", "Sell_Contract_ID"]:
        df[col] = df[col].apply(_clean_str)

    def route(r):
        load = r["Load_Location"] if pd.notna(r["Load_Location"]) else "UNKNOWN"
        unload = r["Unload_Location"] if pd.notna(r["Unload_Location"]) else "UNKNOWN"
        return f"{load} -> {unload}"

    df["Route"] = df.apply(route, axis=1)

    df["flag_missing_commodity"] = df["Commodity"].isna() & (df["row_type"] == "detail")
    df["flag_missing_bl"] = df["BL_Number"].isna() & (df["row_type"] == "detail")
    df["flag_missing_gbl"] = df["GBL_Number"].isna() & (df["row_type"] == "detail")
    df["flag_missing_location"] = (
        (df["Load_Location"].isna() | df["Unload_Location"].isna()) & (df["row_type"] == "detail")
    )
    df["flag_missing_counterparty"] = df["Counterparty"].isna() & (df["row_type"] == "detail")
    df["completeness_score"] = 1 - df[[
        "flag_missing_commodity", "flag_missing_bl", "flag_missing_gbl",
        "flag_missing_location", "flag_missing_counterparty",
    ]].mean(axis=1)

    # duplicate flag: same contract + BL + GBL + quantity seen more than once among detail rows
    dup_cols = ["Sell_Contract_ID", "BL_Number", "GBL_Number", "Quantity_Load"]
    dup_key = df[dup_cols].apply(lambda c: c.map(lambda v: str(v))).agg("|".join, axis=1)
    df["duplicate_flag"] = dup_key.duplicated(keep=False) & (df["row_type"] == "detail")

    return df


def ingest(path: str, sheet_name=0) -> pd.DataFrame:
    """Full Step 1-3 pipeline: load, classify, normalize."""
    df = load_report(path, sheet_name=sheet_name)
    df = classify_rows(df)
    df = normalize_fields(df)
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/LBL_and_GBL_Summary_Report.xlsx"
    out = ingest(path)
    print(out["row_type"].value_counts())
    print(out[out["row_type"] == "detail"].head(10).to_string())
