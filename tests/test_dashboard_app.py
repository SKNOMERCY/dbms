from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_database_failure_is_actionable_and_does_not_expose_connection_details(monkeypatch):
    monkeypatch.setenv("PGPASSWORD", "test-only-placeholder")
    monkeypatch.setenv("PGDATABASE", "finguard_database_that_does_not_exist")

    dashboard_path = Path(__file__).resolve().parents[1] / "dashboard.py"
    app = AppTest.from_file(dashboard_path, default_timeout=30).run()

    assert not app.exception
    visible_errors = [element.value for element in app.error]
    assert any("could not connect to PostgreSQL" in message for message in visible_errors)
    assert all("test-only-placeholder" not in message for message in visible_errors)
    assert all("finguard_database_that_does_not_exist" not in message for message in visible_errors)
