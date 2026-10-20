import pandas as pd

from app_utils import database_error_message, safe_csv_bytes


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


def test_missing_case_history_table_gets_actionable_migration_message():
    error = pd.errors.DatabaseError(
        'relation "investigation_history" does not exist'
    )

    message = database_error_message(error)

    assert "Run `advanced_db.sql`" in message
    assert "same database configured for the dashboard" in message


def test_generic_database_message_does_not_expose_exception_details():
    message = database_error_message(Exception("password=secret-value connection failed"))

    assert "database request failed" in message
    assert "secret-value" not in message
