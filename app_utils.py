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
