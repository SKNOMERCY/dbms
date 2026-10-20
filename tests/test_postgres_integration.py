import os
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from dashboard_logic import build_filter_sql

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def isolated_schema_connection():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration checks.")

    engine = create_engine(database_url)
    schema_name = f"finguard_test_{uuid4().hex}"
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
                connection.execute(text(f'SET LOCAL search_path TO "{schema_name}"'))
                connection.exec_driver_sql((PROJECT_DIR / "schema.sql").read_text(encoding="utf-8"))
                # Exercise the compatibility path for databases created with the former timestamp name.
                connection.exec_driver_sql("ALTER TABLE investigation ADD COLUMN closed_at TIMESTAMP")

                advanced_sql = (PROJECT_DIR / "advanced_db.sql").read_text(encoding="utf-8")
                start = advanced_sql.index("-- 2. Show HIGH-risk transactions")
                end = advanced_sql.index("-- 6a. CTE:", start)
                connection.exec_driver_sql(advanced_sql[start:end])
                yield connection
            finally:
                transaction.rollback()
    finally:
        engine.dispose()


@pytest.fixture
def savepoint(isolated_schema_connection):
    transaction = isolated_schema_connection.begin_nested()
    try:
        yield isolated_schema_connection
    finally:
        transaction.rollback()


def seed_investigation(connection, create_case=True):
    home_location = connection.execute(
        text("INSERT INTO location (loc_type, latitude, longitude) VALUES ('HOME', 40, -73) RETURNING location_id")
    ).scalar_one()
    merchant_location = connection.execute(
        text("INSERT INTO location (loc_type, latitude, longitude) VALUES ('MERCHANT', 40, -73) RETURNING location_id")
    ).scalar_one()
    customer_id = connection.execute(
        text("""
            INSERT INTO customer (first_name, last_name, gender, dob, home_location_id)
            VALUES ('Test', 'Customer', 'F', DATE '1995-01-01', :location_id)
            RETURNING customer_id
        """),
        {"location_id": home_location},
    ).scalar_one()
    account_id = connection.execute(
        text("INSERT INTO account (customer_id, account_type, opened_date) VALUES (:customer_id, 'SAVINGS', CURRENT_DATE) RETURNING account_id"),
        {"customer_id": customer_id},
    ).scalar_one()
    merchant_id = connection.execute(
        text("INSERT INTO merchant (name, category) VALUES (:name, 'shopping_net') RETURNING merchant_id"),
        {"name": f"Test Merchant {uuid4().hex}"},
    ).scalar_one()
    device_id = connection.execute(
        text("INSERT INTO device (device_type) VALUES ('MOBILE') RETURNING device_id")
    ).scalar_one()
    transaction_id = connection.execute(
        text("""
            INSERT INTO transaction
                (trans_num, account_id, merchant_id, device_id, location_id, amount, txn_timestamp, is_fraud)
            VALUES (:trans_num, :account_id, :merchant_id, :device_id, :location_id, 25, CURRENT_TIMESTAMP, TRUE)
            RETURNING transaction_id
        """),
        {
            "trans_num": uuid4().hex,
            "account_id": account_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "location_id": merchant_location,
        },
    ).scalar_one()
    connection.execute(
        text("INSERT INTO risk_score (transaction_id, score_value, model_name) VALUES (:transaction_id, 90, 'test')"),
        {"transaction_id": transaction_id},
    )
    alert_id = connection.execute(
        text("INSERT INTO fraud_alert (transaction_id, risk_score) VALUES (:transaction_id, 90) RETURNING alert_id"),
        {"transaction_id": transaction_id},
    ).scalar_one()
    case_id = None
    if create_case:
        case_id = connection.execute(
            text("INSERT INTO investigation (alert_id) VALUES (:alert_id) RETURNING case_id"),
            {"alert_id": alert_id},
        ).scalar_one()
    return customer_id, transaction_id, alert_id, case_id


def test_schema_rejects_invalid_transaction_amount(savepoint):
    customer_id, _, _, _ = seed_investigation(savepoint)
    account_id = savepoint.execute(
        text("SELECT account_id FROM account WHERE customer_id = :customer_id"),
        {"customer_id": customer_id},
    ).scalar_one()
    merchant_id = savepoint.execute(text("SELECT merchant_id FROM merchant LIMIT 1")).scalar_one()
    device_id = savepoint.execute(text("SELECT device_id FROM device LIMIT 1")).scalar_one()
    location_id = savepoint.execute(text("SELECT location_id FROM location LIMIT 1")).scalar_one()

    with pytest.raises(IntegrityError):
        with savepoint.begin_nested():
            savepoint.execute(
                text("""
                    INSERT INTO transaction
                        (trans_num, account_id, merchant_id, device_id, location_id, amount, txn_timestamp, is_fraud)
                    VALUES (:trans_num, :account_id, :merchant_id, :device_id, :location_id, -1, CURRENT_TIMESTAMP, FALSE)
                """),
                {
                    "trans_num": uuid4().hex,
                    "account_id": account_id,
                    "merchant_id": merchant_id,
                    "device_id": device_id,
                    "location_id": location_id,
                },
            )


def test_customer_risk_function_and_resolution_trigger_stay_consistent(savepoint):
    customer_id, _, alert_id, case_id = seed_investigation(savepoint)

    initial_status = savepoint.execute(
        text("SELECT status FROM fraud_alert WHERE alert_id = :alert_id"),
        {"alert_id": alert_id},
    ).scalar_one()
    assert initial_status == "UNDER_REVIEW"

    duplicate_case = savepoint.execute(
        text("INSERT INTO investigation (alert_id) VALUES (:alert_id) ON CONFLICT (alert_id) DO NOTHING RETURNING case_id"),
        {"alert_id": alert_id},
    ).scalar_one_or_none()
    assert duplicate_case is None

    summary = savepoint.execute(
        text("SELECT * FROM get_customer_risk(:customer_id)"),
        {"customer_id": customer_id},
    ).mappings().one()
    assert summary["transaction_count"] == 1
    assert float(summary["total_amount"]) == 25.0
    assert float(summary["average_score_value"]) == 90.0
    assert summary["alert_count"] == 1

    savepoint.execute(
        text("UPDATE investigation SET outcome = 'FRAUD' WHERE case_id = :case_id"),
        {"case_id": case_id},
    )
    resolved_at, alert_status = savepoint.execute(
        text("SELECT i.resolved_at, fa.status FROM investigation i JOIN fraud_alert fa USING (alert_id) WHERE i.case_id = :case_id"),
        {"case_id": case_id},
    ).one()
    assert resolved_at is not None
    assert alert_status == "CONFIRMED_FRAUD"

    savepoint.execute(
        text("UPDATE investigation SET outcome = 'PENDING' WHERE case_id = :case_id"),
        {"case_id": case_id},
    )
    reopened_at, reopened_status = savepoint.execute(
        text("SELECT i.resolved_at, fa.status FROM investigation i JOIN fraud_alert fa USING (alert_id) WHERE i.case_id = :case_id"),
        {"case_id": case_id},
    ).one()
    assert reopened_at is None
    assert reopened_status == "UNDER_REVIEW"


def test_investigation_history_records_create_resolve_and_reopen(savepoint):
    _, _, alert_id, _ = seed_investigation(savepoint, create_case=False)

    savepoint.execute(
        text("""SELECT set_config('finguard.actor', :actor, true),
                     set_config('finguard.reason', :reason, true)"""),
        {"actor": "Demo Analyst", "reason": "Alert requires investigation"},
    )
    case_id = savepoint.execute(
        text("INSERT INTO investigation (alert_id) VALUES (:alert_id) RETURNING case_id"),
        {"alert_id": alert_id},
    ).scalar_one()
    savepoint.execute(
        text("""SELECT set_config('finguard.actor', :actor, true),
                     set_config('finguard.reason', :reason, true)"""),
        {"actor": "Demo Analyst", "reason": "Reviewed transaction evidence"},
    )
    savepoint.execute(
        text("UPDATE investigation SET outcome = 'FRAUD' WHERE case_id = :case_id"),
        {"case_id": case_id},
    )
    savepoint.execute(
        text("""SELECT set_config('finguard.actor', :actor, true),
                     set_config('finguard.reason', :reason, true)"""),
        {"actor": "Demo Analyst", "reason": "Additional information received"},
    )
    savepoint.execute(
        text("UPDATE investigation SET outcome = 'PENDING' WHERE case_id = :case_id"),
        {"case_id": case_id},
    )
    savepoint.execute(
        text("UPDATE investigation SET outcome = 'PENDING' WHERE case_id = :case_id"),
        {"case_id": case_id},
    )

    events = savepoint.execute(
        text("""SELECT event_type, old_outcome, new_outcome, changed_by, reason
               FROM investigation_history
               WHERE case_reference = :case_id
               ORDER BY history_id"""),
        {"case_id": case_id},
    ).mappings().all()
    alert_status = savepoint.execute(
        text("SELECT status FROM fraud_alert WHERE alert_id = :alert_id"),
        {"alert_id": alert_id},
    ).scalar_one()

    assert [(event["event_type"], event["old_outcome"], event["new_outcome"]) for event in events] == [
        ("CASE_CREATED", None, "PENDING"),
        ("CASE_RESOLVED", "PENDING", "FRAUD"),
        ("CASE_REOPENED", "FRAUD", "PENDING"),
    ]
    assert events[0]["changed_by"]
    assert events[0]["changed_by"] == "Demo Analyst"
    assert events[0]["reason"] == "Alert requires investigation"
    assert events[1]["changed_by"] == "Demo Analyst"
    assert events[1]["reason"] == "Reviewed transaction evidence"
    assert events[2]["reason"] == "Additional information received"
    assert alert_status == "UNDER_REVIEW"


def test_shared_filters_apply_inclusive_date_range_and_bound_statuses(savepoint):
    _, transaction_id, _, _ = seed_investigation(savepoint)
    transaction_time = savepoint.execute(
        text("SELECT txn_timestamp FROM transaction WHERE transaction_id = :transaction_id"),
        {"transaction_id": transaction_id},
    ).scalar_one()
    savepoint.execute(
        text("""
            INSERT INTO transaction (trans_num, account_id, merchant_id, device_id, location_id,
                                     amount, txn_timestamp, is_fraud)
            SELECT :trans_num, account_id, merchant_id, device_id, location_id,
                   amount, :next_day, is_fraud
            FROM transaction WHERE transaction_id = :transaction_id
        """),
        {
            "trans_num": uuid4().hex,
            "next_day": transaction_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1),
            "transaction_id": transaction_id,
        },
    )

    where, params = build_filter_sql(
        start_day=transaction_time.date(),
        end_day=transaction_time.date(),
        category="All categories",
        amount_range=(0, 100),
        score_range=(0, 100),
        risk_level="High",
        alert_status="Under review",
        case_outcome="Pending",
    )
    count = savepoint.execute(
        text(f"""SELECT COUNT(*) FROM transaction t
                  JOIN merchant m ON m.merchant_id = t.merchant_id
                  LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
                  LEFT JOIN fraud_alert fa ON fa.transaction_id = t.transaction_id
                  LEFT JOIN investigation i ON i.alert_id = fa.alert_id
                  WHERE {where}"""),
        params,
    ).scalar_one()
    assert count == 1
