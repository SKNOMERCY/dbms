"""Small, independently testable helpers used by the Streamlit dashboard."""

import pandas as pd


_FORMULA_PREFIXES = ("=", "+", "-", "@")


def safe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a dataframe while neutralizing spreadsheet formulas in text cells."""
    safe_frame = frame.copy()
    text_columns = safe_frame.select_dtypes(include=["object", "string"]).columns

    def neutralize(value):
        if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(_FORMULA_PREFIXES):
            return "'" + value
        return value

    for column in text_columns:
        safe_frame[column] = safe_frame[column].map(neutralize)
    return safe_frame.to_csv(index=False).encode("utf-8-sig")


def database_error_message(error: Exception) -> str:
    """Return a safe, actionable dashboard message for expected DB failures."""
    error_text = str(error).lower()
    if "investigation_history" in error_text and "does not exist" in error_text:
        return (
            "The case-history table is missing from this database. Run `advanced_db.sql` "
            "against the same database configured for the dashboard, then reload the page."
        )
    return (
        "A database request failed. Check that PostgreSQL is running, the FinGuard schema is installed, "
        "and your local connection settings are correct. See the README troubleshooting steps."
    )
