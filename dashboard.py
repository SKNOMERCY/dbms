"""FinGuard AI fraud intelligence and investigation workspace."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sklearn.metrics import precision_recall_curve, roc_curve

from app_utils import database_error_message, safe_csv_bytes
from dashboard_logic import ALERT_STATUSES, CASE_OUTCOMES, build_filter_sql, risk_band, threshold_metrics


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
ALERT_SCORE_THRESHOLD = 50
INVESTIGATION_SCORE_THRESHOLD = 80
COLORS = {
    "navy": "#90A8C4",
    "teal": "#39C6B0",
    "blue": "#6EA8FE",
    "red": "#F27D72",
    "amber": "#E9B45B",
    "muted": "#8DA0B8",
    "grid": "#26374B",
}

st.set_page_config(
    page_title="FinGuard AI | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(PROJECT_DIR / ".env", override=False)
DB_PASSWORD = os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD")


def connection_url():
    if not DB_PASSWORD:
        return None
    return URL.create(
        "postgresql+psycopg2",
        username=os.getenv("PGUSER", "postgres"),
        password=DB_PASSWORD,
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        database=os.getenv("PGDATABASE", "finguard"),
    )


@st.cache_resource
def get_engine(url):
    return create_engine(url, pool_pre_ping=True, pool_recycle=1800)


@st.cache_data(ttl=30, show_spinner=False)
def _read_query(query: str, params_items: tuple):
    with get_engine(connection_url()).connect() as connection:
        return pd.read_sql_query(text(query), connection, params=dict(params_items))


def read_query(query: str, params: dict | None = None) -> pd.DataFrame:
    try:
        return _read_query(query, tuple(sorted((params or {}).items())))
    except (SQLAlchemyError, pd.errors.DatabaseError) as exc:
        st.error(database_error_message(exc))
        st.stop()


def set_case_audit_context(connection, actor_name: str, reason: str):
    """Set transaction-local context consumed by the PostgreSQL history trigger."""
    connection.execute(
        text(
            "SELECT set_config('finguard.actor', :actor_name, true), "
            "set_config('finguard.reason', :reason, true)"
        ),
        {"actor_name": actor_name.strip(), "reason": reason.strip()},
    )


def safe_table(frame: pd.DataFrame, **kwargs):
    st.dataframe(frame, width="stretch", hide_index=True, **kwargs)


def show_chart(chart, *, height=280):
    themed = (
        chart.properties(height=height)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor=COLORS["muted"],
            titleColor=COLORS["navy"],
            gridColor=COLORS["grid"],
            domainColor=COLORS["grid"],
            tickColor=COLORS["grid"],
        )
        .configure_legend(labelColor=COLORS["navy"], titleColor=COLORS["navy"])
    )
    st.altair_chart(themed, width="stretch")


def metric_value(value, fmt="number"):
    if pd.isna(value):
        return "—"
    if fmt == "currency":
        return f"${float(value):,.0f}"
    if fmt == "percent":
        return f"{float(value):.2f}%"
    return f"{int(value):,}"


st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #091321; color: #e8eff7; }
    [data-testid="stHeader"] { background: rgba(9,19,33,.94); }
    [data-testid="stSidebar"] { background: #101e30; border-right: 1px solid #26374b; }
    [data-testid="stSidebar"] * { color: #dce7f2; }
    .stApp input, .stApp textarea, [data-baseweb="select"] > div {
        color: #e8eff7 !important; background: #132238 !important; border-color: #30445b !important;
    }
    [data-baseweb="select"] svg { fill: #9aacc1; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1560px; }
    .hero {
        padding: 1.25rem 1.55rem; border: 1px solid #26374b; border-radius: 14px;
        margin: .15rem 0 1.1rem; background: #101e30;
    }
    .hero .eyebrow { font-size: .68rem; font-weight: 750; letter-spacing: .15em; color: #39c6b0; margin-bottom: .38rem; }
    .hero h1 { font-size: 1.75rem; margin: 0 0 .3rem; color: #f4f7fb; }
    .hero p { margin: 0; color: #a8b8cb; font-size: .94rem; }
    [data-testid="stMetric"] { background: #101e30; border: 1px solid #26374b; border-radius: 11px; padding: .8rem .95rem; }
    [data-testid="stMetricLabel"] { color: #9aacc1; }
    [data-testid="stMetricValue"] { color: #f2f6fa; }
    [data-testid="stTabs"] button { color: #a8b8cb; font-weight: 650; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #39c6b0; }
    div[data-testid="stDataFrame"] { border: 1px solid #26374b; border-radius: 9px; }
    div[data-testid="stAlert"] { border-radius: 9px; }
    hr { border-color: #26374b; }
    </style>
    <div class="hero">
      <div class="eyebrow">FINGUARD AI · FRAUD INTELLIGENCE</div>
      <h1>Fraud Intelligence &amp; Investigation Workspace</h1>
      <p>Explore model risk signals, compare them with dataset labels, and manage cases with explicit links between scores, alerts, and outcomes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not DB_PASSWORD:
    st.error("PostgreSQL credentials are missing. Copy `.env.example` to `.env`, set `PGPASSWORD`, and restart the dashboard.")
    st.stop()

engine = get_engine(connection_url())
try:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
except SQLAlchemyError:
    st.error("FinGuard could not connect to PostgreSQL. Check the service, database name, and local `.env` settings.")
    st.stop()

date_bounds = read_query(
    """SELECT MIN(txn_timestamp)::date AS first_day, MAX(txn_timestamp)::date AS last_day,
              MIN(amount) AS min_amount, MAX(amount) AS max_amount
       FROM transaction"""
).iloc[0]
if pd.isna(date_bounds["first_day"]) or pd.isna(date_bounds["last_day"]):
    st.warning("The database is connected, but it contains no transactions. Load the sample before opening the workspace.")
    st.stop()

first_day = pd.Timestamp(date_bounds["first_day"]).date()
last_day = pd.Timestamp(date_bounds["last_day"]).date()
amount_min = float(date_bounds["min_amount"])
amount_max = float(date_bounds["max_amount"])
categories = read_query("SELECT DISTINCT category FROM merchant ORDER BY category")["category"].tolist()

model_report_path = OUTPUTS_DIR / "model_results.csv"
model_report_available = model_report_path.is_file()

with st.sidebar:
    st.markdown("### FinGuard AI")
    st.caption("Fraud intelligence workspace")
    st.markdown("**System status**")
    st.success("PostgreSQL connected")
    st.caption("Model evaluation report available" if model_report_available else "Model report not generated")
    st.divider()
    st.markdown("**Reporting filters**")
    date_preset = st.selectbox(
        "Reporting period",
        ["All sample dates", "Last 90 sample days", "Last 30 sample days", "Custom range"],
        key="date_preset",
    )
    if date_preset == "Custom range":
        selected_dates = st.date_input(
            "Transaction dates",
            value=(first_day, last_day),
            min_value=first_day,
            max_value=last_day,
            key="selected_dates",
        )
    elif date_preset == "Last 90 sample days":
        selected_dates = (max(first_day, last_day - timedelta(days=89)), last_day)
    elif date_preset == "Last 30 sample days":
        selected_dates = (max(first_day, last_day - timedelta(days=29)), last_day)
    else:
        selected_dates = (first_day, last_day)

    selected_category = st.selectbox("Merchant category", ["All categories", *categories], key="category_filter")
    selected_amount = st.slider(
        "Transaction amount",
        min_value=amount_min,
        max_value=amount_max,
        value=(amount_min, amount_max),
        format="$%.2f",
        key="amount_filter",
    )
    selected_score = st.slider("Model risk score", 0, 100, (0, 100), key="score_filter")
    selected_risk_band = st.selectbox("Risk level", ["All levels", "High", "Medium", "Low", "Not scored"], key="risk_filter")
    selected_alert_status = st.selectbox(
        "Alert status", ["All alerts", "No alert", *ALERT_STATUSES.keys()], key="alert_filter"
    )
    selected_case_outcome = st.selectbox(
        "Investigation outcome", ["All cases", *CASE_OUTCOMES.keys()], key="case_filter"
    )
    if st.button("Reset filters", key="reset_filters", use_container_width=True):
        for key in ("date_preset", "selected_dates", "category_filter", "amount_filter", "score_filter", "risk_filter", "alert_filter", "case_filter"):
            st.session_state.pop(key, None)
        st.rerun()
    st.divider()
    st.caption(f"Simulated sample · {first_day:%b %Y}–{last_day:%b %Y}")
    st.caption("Score ≥50 creates an alert; ≥80 opens a case.")

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_day, end_day = selected_dates
elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:
    start_day = end_day = selected_dates[0]
else:
    start_day = end_day = selected_dates
if start_day is None or end_day is None or start_day > end_day:
    st.error("Choose a valid start and end date.")
    st.stop()

filter_sql, query_params = build_filter_sql(
    start_day=start_day,
    end_day=end_day,
    category=selected_category,
    amount_range=selected_amount,
    score_range=selected_score,
    risk_level=selected_risk_band,
    alert_status=selected_alert_status,
    case_outcome=selected_case_outcome,
)
FILTER_JOINS = """
    JOIN merchant m ON m.merchant_id = t.merchant_id
    LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
    LEFT JOIN fraud_alert fa ON fa.transaction_id = t.transaction_id
    LEFT JOIN investigation i ON i.alert_id = fa.alert_id
"""

overview_kpis = read_query(
    f"""SELECT COUNT(*) AS transactions,
               COALESCE(SUM(t.amount), 0) AS transaction_value,
               COUNT(*) FILTER (WHERE fa.alert_id IS NOT NULL) AS flagged_transactions,
               COALESCE(SUM(t.amount) FILTER (WHERE fa.alert_id IS NOT NULL), 0) AS flagged_value,
               COUNT(*) FILTER (WHERE t.is_fraud) AS dataset_fraud,
               COALESCE(100.0 * AVG(CASE WHEN t.is_fraud THEN 1.0 ELSE 0.0 END), 0) AS labeled_fraud_rate,
               COUNT(DISTINCT i.case_id) FILTER (WHERE i.outcome = 'PENDING') AS open_cases,
               COUNT(DISTINCT i.case_id) FILTER (WHERE i.outcome = 'PENDING' AND rs.score_value >= 80) AS high_risk_open_cases,
               COUNT(DISTINCT i.case_id) FILTER (WHERE i.outcome = 'FRAUD') AS confirmed_cases
        FROM transaction t {FILTER_JOINS}
        WHERE {filter_sql}""",
    query_params,
).iloc[0]

st.caption(
    f"Reporting window: {start_day:%b %d, %Y} – {end_day:%b %d, %Y}"
    f" · {selected_category if selected_category != 'All categories' else 'All categories'}"
    " · Counts are filtered by all sidebar selections."
)
st.markdown("#### Activity at a glance")
kpi_row_one = st.columns(4)
kpi_row_one[0].metric("Transactions", metric_value(overview_kpis["transactions"]))
kpi_row_one[1].metric("Model-flagged", metric_value(overview_kpis["flagged_transactions"]), help="Transactions with a persisted fraud_alert record; this is a model workflow signal, not confirmed fraud.")
kpi_row_one[2].metric("Confirmed cases", metric_value(overview_kpis["confirmed_cases"]), help="Investigations whose outcome is FRAUD, based on recorded analyst disposition.")
kpi_row_one[3].metric("Dataset-labeled fraud rate", metric_value(overview_kpis["labeled_fraud_rate"], "percent"), help="Dataset labels marked fraud divided by all transactions matching the filters.")
kpi_row_two = st.columns(4)
kpi_row_two[0].metric("Transaction value", metric_value(overview_kpis["transaction_value"], "currency"))
kpi_row_two[1].metric("Flagged transaction value", metric_value(overview_kpis["flagged_value"], "currency"))
kpi_row_two[2].metric("Open investigations", metric_value(overview_kpis["open_cases"]))
kpi_row_two[3].metric("High-risk cases awaiting review", metric_value(overview_kpis["high_risk_open_cases"]), help="Pending investigation cases whose stored score is at least 80/100.")

overview_tab, transaction_tab, investigation_tab, customer_tab, model_tab = st.tabs(
    ["Overview", "Transactions", "Investigations", "Customer profile", "Model evaluation"]
)

with overview_tab:
    st.subheader("Portfolio overview")
    st.caption("Dataset labels, model alerts, and analyst-confirmed cases are separate measurements.")
    st.info(
        "Risk score is stored on a 0–100 scale. Scores of 50 or higher create model alerts; scores of 80 or higher open investigations. "
        "The score is not a calibrated probability and does not establish criminal intent."
    )
    daily = read_query(
        f"""SELECT DATE_TRUNC('day', t.txn_timestamp)::date AS day,
                   COUNT(*) AS transactions,
                   COUNT(*) FILTER (WHERE fa.alert_id IS NOT NULL) AS model_alerts,
                   COUNT(*) FILTER (WHERE t.is_fraud) AS dataset_labeled_fraud,
                   COALESCE(SUM(t.amount), 0) AS transaction_value
            FROM transaction t {FILTER_JOINS}
            WHERE {filter_sql}
            GROUP BY 1 ORDER BY 1""",
        query_params,
    )
    by_category = read_query(
        f"""SELECT m.category, COUNT(*) AS transactions,
                   COUNT(*) FILTER (WHERE t.is_fraud) AS labeled_fraud,
                   COALESCE(100.0 * AVG(CASE WHEN t.is_fraud THEN 1.0 ELSE 0.0 END), 0) AS labeled_fraud_rate,
                   COUNT(*) FILTER (WHERE fa.alert_id IS NOT NULL) AS model_alerts,
                   COALESCE(100.0 * AVG(CASE WHEN fa.alert_id IS NOT NULL THEN 1.0 ELSE 0.0 END), 0) AS model_alert_rate
            FROM transaction t {FILTER_JOINS}
            WHERE {filter_sql}
            GROUP BY m.category ORDER BY model_alerts DESC, m.category""",
        query_params,
    )
    risk_distribution = read_query(
        f"""SELECT CASE WHEN rs.score_value IS NULL THEN 'Not scored'
                         WHEN rs.score_value >= 80 THEN 'High'
                         WHEN rs.score_value >= 50 THEN 'Medium' ELSE 'Low' END AS risk_level,
                   COUNT(*) AS transactions
            FROM transaction t {FILTER_JOINS}
            WHERE {filter_sql}
            GROUP BY 1""",
        query_params,
    )
    status_breakdown = read_query(
        f"""SELECT COALESCE(fa.status, 'NO_ALERT') AS alert_status,
                   COUNT(*) AS transactions
            FROM transaction t {FILTER_JOINS}
            WHERE {filter_sql}
            GROUP BY 1 ORDER BY 2 DESC""",
        query_params,
    )
    case_breakdown = read_query(
        f"""SELECT COALESCE(i.outcome, 'NO_CASE') AS case_outcome, COUNT(*) AS transactions
            FROM transaction t {FILTER_JOINS}
            WHERE {filter_sql}
            GROUP BY 1 ORDER BY 2 DESC""",
        query_params,
    )
    scatter = read_query(
        f"""SELECT t.transaction_id, t.amount, rs.score_value AS risk_score,
                   t.is_fraud AS dataset_fraud, fa.status AS alert_status
            FROM transaction t {FILTER_JOINS}
            WHERE {filter_sql} AND rs.score_value IS NOT NULL
            ORDER BY t.txn_timestamp DESC LIMIT 1000""",
        query_params,
    )

    if daily.empty:
        st.info("No transactions match the current filter set.")
    else:
        top_left, top_right = st.columns([1.4, 1])
        with top_left:
            st.markdown("##### Transaction volume and value over time")
            volume_chart = alt.Chart(daily).transform_fold(
                ["transactions", "model_alerts", "dataset_labeled_fraud"],
                as_=["measure", "count"],
            ).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("day:T", title="Transaction date"),
                y=alt.Y("count:Q", title="Transaction count"),
                color=alt.Color("measure:N", title="Measure", scale=alt.Scale(domain=["transactions", "model_alerts", "dataset_labeled_fraud"], range=[COLORS["blue"], COLORS["amber"], COLORS["red"]])),
                tooltip=[alt.Tooltip("day:T", title="Date"), "measure:N", alt.Tooltip("count:Q", title="Count", format=",")],
            )
            show_chart(volume_chart, height=300)
            value_chart = alt.Chart(daily).mark_area(opacity=.18, line={"color": COLORS["teal"]}).encode(
                x=alt.X("day:T", title="Transaction date"),
                y=alt.Y("transaction_value:Q", title="Transaction value", axis=alt.Axis(format="$,.0f")),
                tooltip=[alt.Tooltip("day:T", title="Date"), alt.Tooltip("transaction_value:Q", title="Value", format="$,.2f")],
            )
            show_chart(value_chart, height=190)
        with top_right:
            st.markdown("##### Risk score distribution")
            if risk_distribution.empty:
                st.info("No model scores are available for this selection.")
            else:
                order = ["High", "Medium", "Low", "Not scored"]
                risk_chart = alt.Chart(risk_distribution).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y("risk_level:N", sort=order, title=None),
                    x=alt.X("transactions:Q", title="Transactions"),
                    color=alt.Color("risk_level:N", sort=order, scale=alt.Scale(domain=order, range=[COLORS["red"], COLORS["amber"], COLORS["teal"], COLORS["muted"]]), legend=None),
                    tooltip=["risk_level:N", alt.Tooltip("transactions:Q", format=",")],
                )
                show_chart(risk_chart, height=185)
            st.markdown("##### Alert status")
            if status_breakdown.empty:
                st.info("No alert records match these filters.")
            else:
                status_chart = alt.Chart(status_breakdown).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y("alert_status:N", sort="-x", title=None),
                    x=alt.X("transactions:Q", title="Transactions"),
                    color=alt.value(COLORS["blue"]),
                    tooltip=["alert_status:N", alt.Tooltip("transactions:Q", format=",")],
                )
                show_chart(status_chart, height=165)
            st.markdown("##### Investigation outcome")
            if case_breakdown.empty:
                st.info("No investigation records match these filters.")
            else:
                case_chart = alt.Chart(case_breakdown).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y("case_outcome:N", sort="-x", title=None),
                    x=alt.X("transactions:Q", title="Transactions"),
                    color=alt.value(COLORS["teal"]),
                    tooltip=["case_outcome:N", alt.Tooltip("transactions:Q", format=",")],
                )
                show_chart(case_chart, height=165)

        bottom_left, bottom_right = st.columns([1.15, 1])
        with bottom_left:
            st.markdown("##### Labeled fraud and model-alert rate by category")
            if by_category.empty:
                st.info("No category data match this selection.")
            else:
                category_chart = alt.Chart(by_category).transform_fold(
                    ["labeled_fraud_rate", "model_alert_rate"], as_=["signal", "rate"]
                ).mark_bar().encode(
                    y=alt.Y("category:N", sort="-x", title=None),
                    x=alt.X("rate:Q", title="Transactions in category (%)", axis=alt.Axis(format=".1f")),
                    color=alt.Color("signal:N", scale=alt.Scale(domain=["labeled_fraud_rate", "model_alert_rate"], range=[COLORS["red"], COLORS["teal"]]), title="Signal"),
                    tooltip=["category:N", "signal:N", alt.Tooltip("rate:Q", format=".2f"), alt.Tooltip("transactions:Q", format=","), alt.Tooltip("labeled_fraud:Q", title="Labeled fraud", format=","), alt.Tooltip("model_alerts:Q", title="Model alerts", format=",")],
                )
                show_chart(category_chart, height=320)
        with bottom_right:
            st.markdown("##### Transaction value vs. model score")
            if scatter.empty:
                st.info("No scored transactions match these filters.")
            else:
                scatter_chart = alt.Chart(scatter).mark_circle(size=45, opacity=.68).encode(
                    x=alt.X("risk_score:Q", title="Model risk score (0–100)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("amount:Q", title="Transaction amount", axis=alt.Axis(format="$,.0f")),
                    color=alt.Color("dataset_fraud:N", title="Dataset label", scale=alt.Scale(domain=[True, False], range=[COLORS["red"], COLORS["blue"]])),
                    tooltip=["transaction_id:N", alt.Tooltip("risk_score:Q", format=".1f"), alt.Tooltip("amount:Q", format="$,.2f"), "dataset_fraud:N", "alert_status:N"],
                )
                show_chart(scatter_chart, height=320)
                st.caption("Scatter plot shows the latest 1,000 matching scored transactions; red marks the dataset fraud label.")

        with st.expander("Category detail"):
            safe_table(by_category, column_config={
                "transactions": st.column_config.NumberColumn("Transactions", format="%d"),
                "labeled_fraud": st.column_config.NumberColumn("Dataset-labeled fraud", format="%d"),
                "labeled_fraud_rate": st.column_config.NumberColumn("Dataset-labeled rate", format="%.2f%%"),
                "model_alerts": st.column_config.NumberColumn("Model alerts", format="%d"),
                "model_alert_rate": st.column_config.NumberColumn("Model-alert rate", format="%.2f%%"),
            })

with transaction_tab:
    st.subheader("Transaction investigation")
    st.caption("Search uses bound parameters. Results are sorted by model score and loaded in bounded pages.")
    search_text = st.text_input("Search transaction ID, customer, merchant, or category", placeholder="Transaction ID or a merchant/category name")
    page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1)
    transaction_where = filter_sql
    transaction_params = dict(query_params)
    if search_text.strip():
        transaction_where += " AND (CAST(t.transaction_id AS TEXT) ILIKE :search OR CAST(c.customer_id AS TEXT) ILIKE :search OR (c.first_name || ' ' || c.last_name) ILIKE :search OR m.name ILIKE :search OR m.category ILIKE :search)"
        transaction_params["search"] = f"%{search_text.strip()}%"
    transaction_joins = """
        JOIN account a ON a.account_id = t.account_id
        JOIN customer c ON c.customer_id = a.customer_id
        JOIN merchant m ON m.merchant_id = t.merchant_id
        LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
        LEFT JOIN fraud_alert fa ON fa.transaction_id = t.transaction_id
        LEFT JOIN investigation i ON i.alert_id = fa.alert_id
    """
    matching_count = int(read_query(
        f"SELECT COUNT(*) AS total FROM transaction t {transaction_joins} WHERE {transaction_where}", transaction_params
    ).iloc[0]["total"])
    page_count = max(1, int(np.ceil(matching_count / page_size)))
    if st.session_state.get("transaction_page", 1) > page_count:
        st.session_state["transaction_page"] = page_count
    page_number = int(st.number_input("Page", min_value=1, max_value=page_count, step=1, key="transaction_page"))
    transaction_params.update({"page_limit": page_size, "page_offset": (page_number - 1) * page_size})
    transactions = read_query(
        f"""SELECT t.transaction_id, t.txn_timestamp, c.customer_id,
                   c.first_name || ' ' || c.last_name AS customer,
                   m.name AS merchant, m.category, t.amount, t.is_fraud AS dataset_label,
                   rs.score_value AS risk_score, rs.model_name, fa.alert_id,
                   fa.status AS alert_status, fa.reason AS alert_reason,
                   i.outcome AS investigation_outcome, i.opened_at, i.resolved_at
            FROM transaction t {transaction_joins}
            WHERE {transaction_where}
            ORDER BY rs.score_value DESC NULLS LAST, t.txn_timestamp DESC
            LIMIT :page_limit OFFSET :page_offset""",
        transaction_params,
    )
    st.caption(f"{matching_count:,} matching transactions · page {page_number} of {page_count}")
    if transactions.empty:
        st.info("No transactions match the current filters and search.")
    else:
        safe_table(transactions.drop(columns=["model_name", "alert_id", "alert_reason", "opened_at", "resolved_at"]), column_config={
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f"),
            "dataset_label": st.column_config.CheckboxColumn("Dataset fraud label", help="Ground-truth sample label; separate from analyst outcome."),
            "txn_timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
        })
        st.download_button(
            "Download current page as CSV",
            data=safe_csv_bytes(transactions),
            file_name=f"finguard_transactions_page_{page_number}.csv",
            mime="text/csv",
        )
        transaction_ids = transactions["transaction_id"].astype(int).tolist()
        detail_key = f"transaction_detail_page_{page_number}"
        if st.session_state.get(detail_key) not in transaction_ids:
            st.session_state.pop(detail_key, None)
        selected_transaction = st.selectbox("Inspect a transaction", transaction_ids, key=detail_key)
        details = transactions.loc[transactions["transaction_id"] == selected_transaction].iloc[0]
        with st.expander("Transaction evidence and linked workflow", expanded=True):
            detail_metrics = st.columns(4)
            detail_metrics[0].metric("Model score", f"{float(details['risk_score']):.1f}/100" if pd.notna(details["risk_score"]) else "Not scored")
            detail_metrics[1].metric("Risk band", risk_band(details["risk_score"]))
            detail_metrics[2].metric("Alert", str(details["alert_status"] or "No alert").replace("_", " ").title())
            detail_metrics[3].metric("Investigation", str(details["investigation_outcome"] or "No case").replace("_", " ").title())
            st.write(f"**Transaction:** {int(details['transaction_id'])} · **Customer:** {details['customer']} (#{int(details['customer_id'])})")
            st.write(f"**Merchant:** {details['merchant']} · **Category:** {details['category']} · **Amount:** ${float(details['amount']):,.2f}")
            st.write(f"**Timestamp:** {details['txn_timestamp']} · **Model:** {details['model_name'] or 'No score recorded'} · **Dataset label:** {'Fraud' if details['dataset_label'] else 'Not fraud'}")
            if pd.notna(details["alert_id"]):
                st.write(f"**Alert:** #{int(details['alert_id'])} · **Recorded reason:** {details['alert_reason'] or 'No reason stored'}")
            if pd.notna(details["investigation_outcome"]):
                st.write(f"**Case dates:** opened {details['opened_at']} · resolved {details['resolved_at'] if pd.notna(details['resolved_at']) else 'still pending'}")
            st.caption("The stored score is a model signal, not a calibrated probability. A score alone does not establish why a transaction occurred or whether it was criminal.")

with investigation_tab:
    st.subheader("Alert queue and case management")
    st.caption("Create a case from an alert, resolve pending cases, or reopen a resolved case. There is no separate audit-history table in this schema.")
    cases = read_query(
        f"""SELECT i.case_id, i.outcome, i.opened_at, i.resolved_at, fa.alert_id,
                   fa.status AS alert_status, t.transaction_id, t.txn_timestamp, t.amount,
                   rs.score_value AS risk_score, rs.model_name, m.category,
                   c.first_name || ' ' || c.last_name AS customer
            FROM investigation i
            JOIN fraud_alert fa ON fa.alert_id = i.alert_id
            JOIN transaction t ON t.transaction_id = fa.transaction_id
            JOIN merchant m ON m.merchant_id = t.merchant_id
            JOIN account a ON a.account_id = t.account_id
            JOIN customer c ON c.customer_id = a.customer_id
            LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
            WHERE {filter_sql}
            ORDER BY CASE WHEN i.outcome = 'PENDING' THEN 0 ELSE 1 END,
                     rs.score_value DESC NULLS LAST, i.opened_at DESC
            LIMIT 500""",
        query_params,
    )
    st.markdown("##### Cases in the selected view")
    if cases.empty:
        st.info("No investigation cases match the selected filters.")
    else:
        safe_table(cases.drop(columns=["alert_id"]), column_config={
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f"),
            "txn_timestamp": st.column_config.DatetimeColumn("Transaction time", format="YYYY-MM-DD HH:mm"),
        })
        case_options = cases["case_id"].astype(int).tolist()
        case_detail_key = "case_detail_selection"
        if st.session_state.get(case_detail_key) not in case_options:
            st.session_state.pop(case_detail_key, None)
        selected_case_id = int(st.selectbox("Case to update", case_options, key=case_detail_key))
        selected_case = cases.loc[cases["case_id"] == selected_case_id].iloc[0]
        current_outcome = selected_case["outcome"]
        next_outcome = "FRAUD" if current_outcome == "PENDING" else "PENDING"
        action_text = "Resolve as confirmed fraud" if next_outcome == "FRAUD" else "Reopen case for review"
        if current_outcome == "FRAUD":
            next_outcome = "PENDING"
            action_text = "Reopen case for review"
        elif current_outcome == "NOT_FRAUD":
            next_outcome = "PENDING"
            action_text = "Reopen case for review"
        st.info(
            f"Case **#{selected_case_id}** · Transaction **{int(selected_case['transaction_id'])}** · "
            f"Current outcome **{current_outcome.replace('_', ' ')}** · Alert **{selected_case['alert_status'].replace('_', ' ')}**"
        )
        with st.form("case_action_form"):
            if current_outcome == "PENDING":
                next_outcome = st.selectbox("Investigation outcome", ["FRAUD", "NOT_FRAUD"], format_func=lambda value: "Confirm fraud" if value == "FRAUD" else "Not fraud")
                action_text = "Save investigation outcome"
            action_actor = st.text_input("Analyst name", max_chars=120, key="case_action_actor")
            action_reason = st.text_area("Reason for this action", max_chars=500, key="case_action_reason")
            confirm_action = st.checkbox("I reviewed this case and confirm this database change.")
            submitted = st.form_submit_button(action_text, type="primary")
        if submitted and (not action_actor.strip() or not action_reason.strip()):
            st.warning("Enter an analyst name and a reason to record this case action.")
        elif submitted and not confirm_action:
            st.warning("Confirm the case change before saving.")
        elif submitted:
            expected_outcome = current_outcome
            try:
                with engine.begin() as connection:
                    set_case_audit_context(connection, action_actor, action_reason)
                    updated = connection.execute(
                        text("UPDATE investigation SET outcome = :outcome WHERE case_id = :case_id AND outcome = :expected RETURNING case_id"),
                        {"outcome": next_outcome, "case_id": selected_case_id, "expected": expected_outcome},
                    ).first()
                    if not updated:
                        st.session_state["case_action_notice"] = "This case changed in another session. Refresh the queue before retrying."
                    else:
                        st.session_state["case_action_notice"] = f"Case #{selected_case_id} updated to {next_outcome.replace('_', ' ')}."
                st.cache_data.clear()
                st.rerun()
            except SQLAlchemyError:
                st.error("The case could not be updated. The database change was rolled back; refresh the page and try again.")

    notice = st.session_state.pop("case_action_notice", None)
    if notice:
        st.success(notice)

    if not cases.empty:
        case_history = read_query(
            """SELECT event_type, old_outcome, new_outcome, changed_by, reason, changed_at
               FROM investigation_history
               WHERE case_reference = :case_id
               ORDER BY history_id DESC
               LIMIT 50""",
            {"case_id": selected_case_id},
        )
        st.markdown("##### Selected case history")
        if case_history.empty:
            st.caption("No case-history events are recorded for this case yet.")
        else:
            safe_table(case_history, column_config={
                "changed_at": st.column_config.DatetimeColumn("Changed at", format="YYYY-MM-DD HH:mm:ss"),
                "reason": st.column_config.TextColumn("Reason", width="large"),
            })

    st.divider()
    st.markdown("##### Alerts without an investigation")
    if selected_case_outcome != "All cases":
        eligible_alerts = pd.DataFrame()
        st.info("Set the investigation outcome filter to All cases to review alerts that do not have a case yet.")
    else:
        no_case_where, no_case_params = build_filter_sql(
            start_day=start_day, end_day=end_day, category=selected_category,
            amount_range=selected_amount, score_range=selected_score,
            risk_level=selected_risk_band, alert_status=selected_alert_status,
            case_outcome="All cases",
        )
        eligible_alerts = read_query(
            f"""SELECT fa.alert_id, fa.status, fa.created_at, fa.risk_score,
                       t.transaction_id, t.txn_timestamp, t.amount, m.category
                FROM fraud_alert fa
                JOIN transaction t ON t.transaction_id = fa.transaction_id
                JOIN merchant m ON m.merchant_id = t.merchant_id
                LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
                LEFT JOIN investigation i ON i.alert_id = fa.alert_id
                WHERE {no_case_where} AND i.case_id IS NULL
                ORDER BY fa.risk_score DESC, fa.created_at DESC LIMIT 250""",
            no_case_params,
        )
    if selected_case_outcome != "All cases":
        pass
    elif eligible_alerts.empty:
        st.caption("No unassigned alerts match the current filters.")
    else:
        safe_table(eligible_alerts, column_config={
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f"),
        })
        alert_ids = eligible_alerts["alert_id"].astype(int).tolist()
        with st.form("create_case_form"):
            selected_alert_id = int(st.selectbox("Alert to investigate", alert_ids))
            create_actor = st.text_input("Analyst name", max_chars=120, key="create_case_actor")
            create_reason = st.text_area("Reason for opening this case", max_chars=500, key="create_case_reason")
            confirm_create = st.checkbox("Create a pending investigation for this alert.")
            create_submitted = st.form_submit_button("Create investigation", type="primary")
        if create_submitted and (not create_actor.strip() or not create_reason.strip()):
            st.warning("Enter an analyst name and a reason to record this case action.")
        elif create_submitted and not confirm_create:
            st.warning("Confirm case creation before saving.")
        elif create_submitted:
            try:
                with engine.begin() as connection:
                    set_case_audit_context(connection, create_actor, create_reason)
                    created = connection.execute(
                        text("INSERT INTO investigation (alert_id) VALUES (:alert_id) ON CONFLICT (alert_id) DO NOTHING RETURNING case_id"),
                        {"alert_id": selected_alert_id},
                    ).scalar_one_or_none()
                st.cache_data.clear()
                if created:
                    st.session_state["case_action_notice"] = f"Created pending case #{created} for alert #{selected_alert_id}."
                else:
                    st.session_state["case_action_notice"] = "Another session already created a case for that alert."
                st.rerun()
            except SQLAlchemyError:
                st.error("The investigation could not be created. The database change was rolled back.")

with customer_tab:
    st.subheader("Customer profile")
    st.caption("Customer lookup is independent of the portfolio filters and shows aggregate stored records.")
    with st.form("customer_lookup"):
        customer_id = st.number_input("Customer ID", min_value=1, step=1, value=1)
        customer_submitted = st.form_submit_button("Open customer profile", type="primary")
    if customer_submitted:
        profile = read_query("SELECT * FROM get_customer_risk(:customer_id)", {"customer_id": int(customer_id)})
        if profile.empty:
            st.info("No customer was found for that ID.")
        else:
            customer = profile.iloc[0]
            st.markdown(f"##### {customer['customer_name']}")
            profile_metrics = st.columns(4)
            profile_metrics[0].metric("Transactions", f"{int(customer['transaction_count']):,}")
            profile_metrics[1].metric("Total value", f"${float(customer['total_amount']):,.2f}")
            avg_score = customer["average_score_value"]
            profile_metrics[2].metric("Average model score", f"{float(avg_score):.1f}/100" if pd.notna(avg_score) else "Not scored")
            profile_metrics[3].metric("Alerts", f"{int(customer['alert_count']):,}")
            customer_txns = read_query(
                """SELECT t.transaction_id, t.txn_timestamp, m.name AS merchant, m.category,
                          t.amount, rs.score_value AS risk_score, rs.model_name,
                          fa.status AS alert_status, i.outcome AS investigation_outcome
                   FROM transaction t
                   JOIN account a ON a.account_id = t.account_id
                   JOIN merchant m ON m.merchant_id = t.merchant_id
                   LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
                   LEFT JOIN fraud_alert fa ON fa.transaction_id = t.transaction_id
                   LEFT JOIN investigation i ON i.alert_id = fa.alert_id
                   WHERE a.customer_id = :customer_id
                   ORDER BY rs.score_value DESC NULLS LAST, t.txn_timestamp DESC LIMIT 50""",
                {"customer_id": int(customer_id)},
            )
            st.markdown("##### Highest-scored transactions (up to 50)")
            safe_table(customer_txns, column_config={"amount": st.column_config.NumberColumn("Amount", format="$%.2f"), "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f")})

with model_tab:
    st.subheader("Model evaluation")
    st.caption("Evaluation artifacts are read from disk; the dashboard never trains or silently reloads a model.")
    if not model_report_available:
        st.info("No model report is available. Run `python ml_pipeline.py --reports-only` after loading the sample database.")
    else:
        model_results = pd.read_csv(model_report_path)
        metadata_path = OUTPUTS_DIR / "evaluation_metadata.json"
        prediction_path = OUTPUTS_DIR / "evaluation_predictions.csv"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
        if "validation_pr_auc" not in model_results.columns or not model_results["selected"].astype(bool).any():
            st.warning("This report does not include a validation-selected model. Regenerate the evaluation artifacts with the current pipeline.")
        else:
            selected_row = model_results.loc[model_results["selected"].astype(bool)].iloc[0]
            eval_header = st.columns(3)
            eval_header[0].metric("Selected by validation PR-AUC", selected_row["model"])
            eval_header[1].metric("Validation PR-AUC", f"{float(selected_row['validation_pr_auc']):.3f}")
            eval_header[2].metric("Held-out test PR-AUC", f"{float(selected_row['pr_auc']):.3f}")
            if metadata:
                st.caption(f"Split: {metadata.get('split_description', 'metadata unavailable')} · Operating threshold in saved comparison: {metadata.get('threshold', 0.5):.2f} · Test rows: {metadata.get('test_rows', 'unknown')}")
            else:
                st.caption("Split: stratified random transaction-level 60/20/20 train/validation/test. Evaluation is on this simulated sample only.")
            st.warning("Isolation Forest produces anomaly scores scaled against its training-score range. Its scores are not probabilities and do not have the same interpretation as supervised scores; the saved table compares ranking metrics but threshold metrics are operating-point illustrations.")
            display_results = model_results.sort_values("validation_pr_auc", ascending=False)
            safe_table(display_results, column_config={
                "selected": st.column_config.CheckboxColumn("Selected by validation"),
                "validation_pr_auc": st.column_config.ProgressColumn("Validation PR-AUC", min_value=0, max_value=1, format="%.3f"),
                "precision": st.column_config.ProgressColumn("Test precision @ 0.50", min_value=0, max_value=1, format="%.3f"),
                "recall": st.column_config.ProgressColumn("Test recall @ 0.50", min_value=0, max_value=1, format="%.3f"),
                "f1": st.column_config.ProgressColumn("Test F1 @ 0.50", min_value=0, max_value=1, format="%.3f"),
                "roc_auc": st.column_config.ProgressColumn("Test ROC-AUC", min_value=0, max_value=1, format="%.3f"),
                "pr_auc": st.column_config.ProgressColumn("Test PR-AUC", min_value=0, max_value=1, format="%.3f"),
            })
            predictions = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()
            score_columns = [column for column in predictions.columns if column not in {"y_true", "transaction_id"}]
            model_names = [name for name in model_results["model"].tolist() if name in score_columns]
            if "y_true" in predictions.columns and not predictions.empty and model_names:
                selected_model = st.selectbox("Inspect held-out test model", model_names, index=model_names.index(selected_row["model"]) if selected_row["model"] in model_names else 0)
                threshold = st.slider("Illustrative operating threshold", 0.05, 0.95, 0.50, step=0.05, help="Changes test-set precision, recall, and confusion matrix for inspection only; it does not modify stored alerts.")
                y_true = predictions["y_true"].astype(int).to_numpy()
                scores = predictions[selected_model].astype(float).to_numpy()
                metrics = threshold_metrics(y_true, scores, threshold)
                threshold_cols = st.columns(4)
                threshold_cols[0].metric("Precision", f"{metrics['precision']:.3f}" if metrics["precision"] is not None else "N/A")
                threshold_cols[1].metric("Recall", f"{metrics['recall']:.3f}" if metrics["recall"] is not None else "N/A")
                threshold_cols[2].metric("F1", f"{metrics['f1']:.3f}" if metrics["f1"] is not None else "N/A")
                threshold_cols[3].metric("PR-AUC", f"{metrics['pr_auc']:.3f}" if metrics["pr_auc"] is not None else "N/A")
                cm = metrics["confusion_matrix"]
                matrix_df = pd.DataFrame(cm, index=["Actual normal", "Actual fraud"], columns=["Predicted normal", "Predicted fraud"])
                if len(np.unique(y_true)) < 2:
                    st.info("Precision–recall and ROC curves require both label classes in the saved test data.")
                else:
                    pr_precision, pr_recall, _ = precision_recall_curve(y_true, scores)
                    roc_fpr, roc_tpr, _ = roc_curve(y_true, scores)
                    curve_left, curve_right = st.columns(2)
                    with curve_left:
                        st.markdown("##### Precision–recall curve")
                        pr_df = pd.DataFrame({"recall": pr_recall, "precision": pr_precision})
                        pr_chart = alt.Chart(pr_df).mark_line(color=COLORS["teal"], strokeWidth=2).encode(x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])), y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])), tooltip=[alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")])
                        show_chart(pr_chart, height=260)
                    with curve_right:
                        st.markdown("##### ROC curve")
                        roc_df = pd.DataFrame({"false_positive_rate": roc_fpr, "true_positive_rate": roc_tpr})
                        roc_chart = alt.Chart(roc_df).mark_line(color=COLORS["blue"], strokeWidth=2).encode(x=alt.X("false_positive_rate:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])), y=alt.Y("true_positive_rate:Q", title="True positive rate", scale=alt.Scale(domain=[0, 1])), tooltip=[alt.Tooltip("false_positive_rate:Q", format=".3f"), alt.Tooltip("true_positive_rate:Q", format=".3f")])
                        show_chart(roc_chart, height=260)
                st.markdown(f"##### Confusion matrix at threshold {threshold:.2f}")
                safe_table(matrix_df.rename_axis("Actual / predicted").reset_index())
            else:
                st.info("The saved report has aggregate test metrics but no row-level test predictions. Run `python ml_pipeline.py --reports-only` to enable threshold inspection and interactive test-set curves.")

            customer_results_path = OUTPUTS_DIR / "customer_holdout_results.csv"
            customer_predictions_path = OUTPUTS_DIR / "customer_holdout_predictions.csv"
            customer_metadata_path = OUTPUTS_DIR / "customer_holdout_metadata.json"
            st.divider()
            st.markdown("##### Customer-held-out diagnostic")
            if customer_results_path.exists() and customer_metadata_path.exists():
                customer_results = pd.read_csv(customer_results_path)
                customer_metadata = json.loads(
                    customer_metadata_path.read_text(encoding="utf-8")
                )
                st.caption(
                    f"{customer_metadata.get('split_description', 'Customer-disjoint split')} "
                    f"Test rows: {customer_metadata.get('test_rows', 'unknown')} · "
                    f"Held-out customers: {customer_metadata.get('test_customers', 'unknown')}"
                )
                st.warning(
                    "This is a harder split of the same simulated sample, not an independent dataset "
                    "or evidence of real-world fraud performance."
                )
                safe_table(customer_results.sort_values("validation_pr_auc", ascending=False), column_config={
                    "selected": st.column_config.CheckboxColumn("Selected by group validation"),
                    "validation_pr_auc": st.column_config.ProgressColumn("Validation PR-AUC", min_value=0, max_value=1, format="%.3f"),
                    "precision": st.column_config.ProgressColumn("Test precision @ 0.50", min_value=0, max_value=1, format="%.3f"),
                    "recall": st.column_config.ProgressColumn("Test recall @ 0.50", min_value=0, max_value=1, format="%.3f"),
                    "f1": st.column_config.ProgressColumn("Test F1 @ 0.50", min_value=0, max_value=1, format="%.3f"),
                    "roc_auc": st.column_config.ProgressColumn("Test ROC-AUC", min_value=0, max_value=1, format="%.3f"),
                    "pr_auc": st.column_config.ProgressColumn("Test PR-AUC", min_value=0, max_value=1, format="%.3f"),
                })
                if customer_predictions_path.exists():
                    customer_predictions = pd.read_csv(customer_predictions_path)
                    group_selected = customer_metadata.get("selected_model")
                    if (
                        "y_true" in customer_predictions
                        and group_selected in customer_predictions
                        and len(np.unique(customer_predictions["y_true"])) > 1
                    ):
                        group_precision, group_recall, _ = precision_recall_curve(
                            customer_predictions["y_true"].astype(int),
                            customer_predictions[group_selected].astype(float),
                        )
                        group_pr_data = pd.DataFrame(
                            {"recall": group_recall, "precision": group_precision}
                        )
                        group_pr_chart = alt.Chart(group_pr_data).mark_line(
                            color=COLORS["teal"], strokeWidth=2
                        ).encode(
                            x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
                            tooltip=[
                                alt.Tooltip("recall:Q", format=".3f"),
                                alt.Tooltip("precision:Q", format=".3f"),
                            ],
                        )
                        st.caption(f"Selected on customer-held-out validation: {group_selected}")
                        st.markdown("###### Held-out customer precision–recall curve")
                        show_chart(group_pr_chart, height=260)
            else:
                st.info(
                    "This diagnostic has not been generated yet. Run the report-only pipeline "
                    "to add customer-disjoint metrics without changing database scores, alerts, or cases."
                )

            feature_importance_path = OUTPUTS_DIR / "feature_importance.png"
            if feature_importance_path.exists():
                with st.expander("Global Random Forest feature importance"):
                    st.image(str(feature_importance_path), width="stretch")
                    st.caption("Global tree-based feature importance summarizes the fitted model; it is not a per-transaction explanation or causal reason.")

st.divider()
st.caption("FinGuard AI · PostgreSQL-backed prototype · Simulated data · Model outputs support analyst review and do not establish criminal intent.")
