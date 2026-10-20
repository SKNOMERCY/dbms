import pandas as pd

from app_utils import safe_csv_bytes


def test_safe_csv_neutralizes_formula_like_text_and_keeps_numbers_numeric():
    frame = pd.DataFrame(
        {
            "merchant": ["=HYPERLINK(\"https://bad.example\")", "  +SUM(A1:A2)"],
            "amount": [-10.5, 20.0],
        }
    )

    exported = safe_csv_bytes(frame).decode("utf-8-sig")

    assert "'=HYPERLINK" in exported
    assert "'  +SUM" in exported
    assert "-10.5" in exported
    assert "20.0" in exported
