"""Pure dashboard rules shared by the Streamlit UI and its unit tests."""

from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOW_RISK_MAX = 50
HIGH_RISK_MIN = 80
ALERT_STATUSES = {
    "Open": "OPEN",
    "Under review": "UNDER_REVIEW",
    "Confirmed fraud": "CONFIRMED_FRAUD",
    "False positive": "FALSE_POSITIVE",
}
CASE_OUTCOMES = {"Pending": "PENDING", "Confirmed fraud": "FRAUD", "Not fraud": "NOT_FRAUD"}


def risk_band(score: Any) -> str:
    """Return the project's risk band for a stored 0–100 score."""
    if score is None:
        return "Not scored"
    try:
        if score != score:  # NaN
            return "Not scored"
        value = float(score)
    except (TypeError, ValueError):
        return "Not scored"
    if value < LOW_RISK_MAX:
        return "Low"
    if value < HIGH_RISK_MIN:
        return "Medium"
    return "High"


def threshold_metrics(y_true, scores, threshold: float) -> dict[str, Any]:
    """Calculate operating-point metrics; scores are not assumed calibrated."""
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1 or len(labels) != len(values):
        raise ValueError("Labels and scores must be one-dimensional arrays with equal length")
    if len(labels) == 0:
        return {"precision": None, "recall": None, "f1": None, "pr_auc": None, "roc_auc": None, "confusion_matrix": np.zeros((2, 2), dtype=int)}

    predicted = (values >= threshold).astype(int)
    has_positive = bool(np.any(labels == 1))
    has_negative = bool(np.any(labels == 0))
    return {
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "pr_auc": average_precision_score(labels, values) if has_positive else None,
        "roc_auc": roc_auc_score(labels, values) if has_positive and has_negative else None,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]),
    }


def build_filter_sql(
    *,
    start_day: date,
    end_day: date,
    category: str,
    amount_range: tuple[float, float],
    score_range: tuple[float, float],
    risk_level: str,
    alert_status: str,
    case_outcome: str,
) -> tuple[str, dict[str, Any]]:
    """Build parameterized shared transaction filters for aliases t, m, rs, fa, i."""
    clauses = ["t.txn_timestamp >= :start_time", "t.txn_timestamp < :end_time"]
    params: dict[str, Any] = {
        "start_time": datetime.combine(start_day, time.min),
        "end_time": datetime.combine(end_day + timedelta(days=1), time.min),
        "amount_min": float(amount_range[0]),
        "amount_max": float(amount_range[1]),
        "score_min": float(score_range[0]),
        "score_max": float(score_range[1]),
    }
    clauses.extend(
        [
            "t.amount BETWEEN :amount_min AND :amount_max",
            "COALESCE(rs.score_value, 0) BETWEEN :score_min AND :score_max",
        ]
    )

    if category != "All categories":
        clauses.append("m.category = :category")
        params["category"] = category

    if risk_level != "All levels":
        if risk_level not in {"Low", "Medium", "High", "Not scored"}:
            raise ValueError("Unsupported risk level")
        clauses.append(
            "CASE WHEN rs.score_value IS NULL THEN 'Not scored' "
            "WHEN rs.score_value >= 80 THEN 'High' "
            "WHEN rs.score_value >= 50 THEN 'Medium' ELSE 'Low' END = :risk_level"
        )
        params["risk_level"] = risk_level

    if alert_status == "No alert":
        clauses.append("fa.alert_id IS NULL")
    elif alert_status != "All alerts":
        if alert_status not in ALERT_STATUSES:
            raise ValueError("Unsupported alert status")
        clauses.append("fa.status = :alert_status")
        params["alert_status"] = ALERT_STATUSES[alert_status]

    if case_outcome != "All cases":
        if case_outcome not in CASE_OUTCOMES:
            raise ValueError("Unsupported case outcome")
        clauses.append("i.outcome = :case_outcome")
        params["case_outcome"] = CASE_OUTCOMES[case_outcome]

    return " AND ".join(clauses), params
