from datetime import date, datetime

import pytest

from dashboard_logic import build_filter_sql, risk_band, threshold_metrics


@pytest.mark.parametrize(
    ("score", "expected"),
    [(None, "Not scored"), (float("nan"), "Not scored"), (0, "Low"), (49.99, "Low"),
     (50, "Medium"), (79.99, "Medium"), (80, "High"), (100, "High")],
)
def test_risk_band_threshold_boundaries(score, expected):
    assert risk_band(score) == expected


def test_global_filter_values_are_bound_and_date_end_is_inclusive():
    where, params = build_filter_sql(
        start_day=date(2020, 1, 1),
        end_day=date(2020, 1, 31),
        category="category' OR TRUE --",
        amount_range=(12.5, 500.0),
        score_range=(50, 100),
        risk_level="High",
        alert_status="Open",
        case_outcome="Pending",
    )

    assert "category = :category" in where
    assert "' OR TRUE" not in where
    assert params["category"] == "category' OR TRUE --"
    assert params["alert_status"] == "OPEN"
    assert params["case_outcome"] == "PENDING"
    assert params["start_time"] == datetime(2020, 1, 1)
    assert params["end_time"] == datetime(2020, 2, 1)


def test_global_filter_rejects_unknown_status_instead_of_ignoring_it():
    with pytest.raises(ValueError, match="Unsupported alert status"):
        build_filter_sql(
            start_day=date(2020, 1, 1),
            end_day=date(2020, 1, 1),
            category="All categories",
            amount_range=(0, 100),
            score_range=(0, 100),
            risk_level="All levels",
            alert_status="Broken status",
            case_outcome="All cases",
        )


def test_threshold_metrics_match_expected_confusion_and_auc():
    metrics = threshold_metrics([0, 0, 1, 1], [0.1, 0.8, 0.6, 0.9], 0.5)

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"].tolist() == [[1, 1], [0, 2]]


def test_threshold_metrics_are_defined_for_empty_or_one_class_inputs():
    empty = threshold_metrics([], [], 0.5)
    one_class = threshold_metrics([0, 0], [0.1, 0.2], 0.5)

    assert empty["precision"] is None
    assert empty["confusion_matrix"].tolist() == [[0, 0], [0, 0]]
    assert one_class["pr_auc"] is None
    assert one_class["roc_auc"] is None
