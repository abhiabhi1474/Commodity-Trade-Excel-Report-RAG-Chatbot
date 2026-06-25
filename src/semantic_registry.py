"""
Step 4 / Layer 1+3: Build semantic metadata for each report field.

This is the Schema RAG source: a registry of business definitions, types,
grain, aggregation rules, and synonyms for every Xceler field, used by the
chatbot to interpret ambiguous business terminology (e.g. "closed
positions", "shipped volume") before any retrieval or computation happens.

Each entry becomes one Schema RAG chunk per the doc's chunking strategy:
"a schema chunk should contain a single business concept definition."
"""
from __future__ import annotations

from typing import List, Dict, Any

SEMANTIC_REGISTRY: List[Dict[str, Any]] = [
    {
        "column": "Vessel_Name",
        "type": "Dimension",
        "grain": "Voyage",
        "aggregation": "NONE",
        "businessMeaning": "The vessel carrying one or more shipments for a voyage. Multiple contracts/BLs can share one voyage on one vessel.",
        "synonyms": ["vessel", "ship", "carrier"],
        "depends_on": [],
    },
    {
        "column": "Voyage_Number",
        "type": "Dimension",
        "grain": "Voyage",
        "aggregation": "NONE",
        "businessMeaning": "Identifies a single voyage. A voyage is the reporting scope for multiple shipment/contract rows in the table.",
        "synonyms": ["voyage", "trip", "journey"],
        "depends_on": ["Vessel_Name"],
    },
    {
        "column": "Sell_Contract_ID",
        "type": "Dimension",
        "grain": "Contract",
        "aggregation": "NONE",
        "businessMeaning": "Unique identifier for the sell contract governing a shipment. A contract may have multiple BL/GBL split lines that roll up to one Sub Total.",
        "synonyms": ["contract", "contract id", "sale contract", "sell contract"],
        "depends_on": ["Voyage_Number"],
    },
    {
        "column": "Counterparty",
        "type": "Dimension",
        "grain": "Contract",
        "aggregation": "NONE",
        "businessMeaning": "The trading counterparty (buyer/seller) on the contract.",
        "synonyms": ["counterparty", "buyer", "client", "trading partner"],
        "depends_on": [],
    },
    {
        "column": "Commodity",
        "type": "Dimension",
        "grain": "Line item",
        "aggregation": "NONE",
        "businessMeaning": "The traded commodity for the shipment line, e.g. Crude Palm Oil, Cocoa, Corn, Soyabeans.",
        "synonyms": ["commodity", "product", "cargo type"],
        "depends_on": [],
    },
    {
        "column": "Quantity_Load",
        "type": "Measure",
        "grain": "Line item",
        "aggregation": "SUM",
        "businessMeaning": "Quantity loaded onto the vessel for this line. Additive across detail rows within a contract; reconciles to the contract Sub Total and voyage Total.",
        "synonyms": ["volume", "quantity", "qty", "load quantity", "tonnage"],
        "depends_on": ["Uom"],
    },
    {
        "column": "Uom",
        "type": "Dimension",
        "grain": "Line item",
        "aggregation": "NONE",
        "businessMeaning": "Unit of measure for Quantity_Load (e.g. MT, L). Must be standardized before cross-row aggregation since mixed units are not directly additive.",
        "synonyms": ["unit", "uom", "unit of measure"],
        "depends_on": [],
    },
    {
        "column": "BL_Number",
        "type": "Dimension",
        "grain": "Line item",
        "aggregation": "NONE",
        "businessMeaning": "Bill of Lading number for the shipment line. A contract can be split across multiple BLs.",
        "synonyms": ["bl", "bl number", "bill of lading"],
        "depends_on": ["Sell_Contract_ID"],
    },
    {
        "column": "GBL_Number",
        "type": "Dimension",
        "grain": "Line item",
        "aggregation": "NONE",
        "businessMeaning": "Government/Group Bill of Lading number associated with the line, used for shipment traceability.",
        "synonyms": ["gbl", "gbl number"],
        "depends_on": ["BL_Number"],
    },
    {
        "column": "Load_Location",
        "type": "Dimension",
        "grain": "Line item",
        "aggregation": "NONE",
        "businessMeaning": "Origin port/location where the commodity was loaded.",
        "synonyms": ["load port", "origin", "loading location"],
        "depends_on": [],
    },
    {
        "column": "Unload_Location",
        "type": "Dimension",
        "grain": "Line item",
        "aggregation": "NONE",
        "businessMeaning": "Destination port/location where the commodity was unloaded.",
        "synonyms": ["unload port", "destination", "discharge location"],
        "depends_on": [],
    },
    {
        "column": "Route",
        "type": "Derived Dimension",
        "grain": "Line item",
        "aggregation": "NONE",
        "businessMeaning": "Derived as Load_Location -> Unload_Location. Used to analyze route concentration and most-active trade lanes.",
        "synonyms": ["route", "trade lane", "shipping lane"],
        "depends_on": ["Load_Location", "Unload_Location"],
    },
    {
        "column": "Company",
        "type": "Dimension",
        "grain": "Contract",
        "aggregation": "NONE",
        "businessMeaning": "The reporting entity / internal company booking the contract.",
        "synonyms": ["company", "entity", "booking company"],
        "depends_on": [],
    },
    {
        "column": "Subtotal",
        "type": "KPI",
        "grain": "Contract",
        "aggregation": "SUM",
        "businessMeaning": "Sum of Quantity_Load across all detail rows sharing a Sell_Contract_ID. Appears in the report as a 'Sub Total' control row, not part of the fact table.",
        "synonyms": ["sub total", "contract total"],
        "depends_on": ["Quantity_Load", "Sell_Contract_ID"],
    },
    {
        "column": "Total",
        "type": "KPI",
        "grain": "Voyage",
        "aggregation": "SUM",
        "businessMeaning": "Sum of Quantity_Load across all contracts within a voyage. Appears as a 'Total' control row keyed by Voyage_Number == 'Total'.",
        "synonyms": ["voyage total", "grand total"],
        "depends_on": ["Quantity_Load", "Voyage_Number"],
    },
    {
        "column": "Completeness_Score",
        "type": "Derived Measure",
        "grain": "Line item",
        "aggregation": "AVG",
        "businessMeaning": "1 minus the share of data-quality flags raised on a detail row (missing commodity, BL, GBL, location, counterparty). Used to flag rows where reporting confidence is weaker.",
        "synonyms": ["data quality", "completeness", "confidence"],
        "depends_on": ["flag_missing_commodity", "flag_missing_bl", "flag_missing_gbl", "flag_missing_location", "flag_missing_counterparty"],
    },
]


def schema_chunks() -> List[Dict[str, Any]]:
    """Render the registry as retrievable Schema RAG chunks (one concept per chunk)."""
    chunks = []
    for entry in SEMANTIC_REGISTRY:
        text = (
            f"Field: {entry['column']}. Type: {entry['type']}. Grain: {entry['grain']}. "
            f"Aggregation rule: {entry['aggregation']}. "
            f"Business meaning: {entry['businessMeaning']} "
            f"Synonyms: {', '.join(entry['synonyms'])}. "
            f"Depends on: {', '.join(entry['depends_on']) if entry['depends_on'] else 'none'}."
        )
        chunks.append({
            "chunk_type": "schema_definition",
            "column": entry["column"],
            "text": text,
            "metadata": entry,
        })
    return chunks


if __name__ == "__main__":
    for c in schema_chunks()[:3]:
        print(c["text"], "\n")
