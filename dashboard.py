"""Streamlit dashboard for the FinGuard PostgreSQL database."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import URL


st.set_page_config(page_title="FinGuard AI · Risk dashboard", page_icon="🛡️", layout="wide")

DB_PASSWORD = os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    st.error("Set PGPASSWORD before starting the dashboard. DB_PASSWORD is also accepted for compatibility.")
    st.stop()

DB_URL = URL.create(
    "postgresql+psycopg2",
    username=os.getenv("PGUSER", "postgres"),
    password=DB_PASSWORD,
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    database=os.getenv("PGDATABASE", "finguard"),
)


@st.cache_resource
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


engine = get_engine()


def read_query(query, params=None):
    try:
        with engine.connect() as connection:
            return pd.read_sql(text(query), connection, params=params)
    except SQLAlchemyError as exc:
        st.error("The dashboard could not read the database. Check PostgreSQL, your connection settings, and the project setup steps in the README.")
        with st.expander("Database error details"):
            st.code(f"{type(exc).__name__}: {exc}")
        st.stop()


st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; color: #16243a; }
    [data-testid="stHeader"] { background: rgba(245, 247, 251, .92); }
    [data-testid="stSidebar"] { background: #10243b; }
    [data-testid="stSidebar"] * { color: #e7edf7; }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #16243a;
    }
    .hero {
        padding: 1.45rem 1.75rem;
        border-radius: 16px;
        margin: .2rem 0 1rem;
        background: linear-gradient(112deg, #10243b 0%, #153c54 65%, #11766e 100%);
        color: #f4f8ff;
        box-shadow: 0 10px 26px rgba(16, 36, 59, .12);
    }
    .hero .eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .12em; color: #85e1d2; margin-bottom: .35rem; }
    .hero h1 { font-size: 1.9rem; margin: 0 0 .35rem; color: #fff; }
    .hero p { margin: 0; color: #d2e0eb; font-size: 1rem; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e4eaf2; border-radius: 12px;
        padding: .85rem 1rem; box-shadow: 0 2px 9px rgba(16, 36, 59, .035);
    }
    [data-testid="stMetricLabel"] { color: #607089; }
    [data-testid="stTabs"] button { font-weight: 650; }
    div[data-testid="stDataFrame"] { border: 1px solid #e4eaf2; border-radius: 10px; }
    div[data-testid="stAlert"] { border-radius: 10px; }
    </style>
    <div class="hero">
      <div class="eyebrow">FINGUARD AI · FRAUD OPERATIONS</div>
      <h1>Transaction risk dashboard</h1>
      <p>Review labeled activity, prioritize model alerts, and document investigation outcomes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

date_bounds = read_query(
    "SELECT MIN(txn_timestamp)::date AS first_day, MAX(txn_timestamp)::date AS last_day FROM transaction"
).iloc[0]
if pd.isna(date_bounds["first_day"]) or pd.isna(date_bounds["last_day"]):
    st.warning("No transactions are available. Load the project data before using the dashboard.")
    st.stop()

first_day = pd.Timestamp(date_bounds["first_day"]).date()
last_day = pd.Timestamp(date_bounds["last_day"]).date()
categories = read_query("SELECT DISTINCT category FROM merchant ORDER BY category")["category"].tolist()

with st.sidebar:
    st.markdown("## FinGuard AI")
    st.caption("TRANSACTION RISK WORKSPACE")
    st.divider()
    st.markdown("### Reporting filters")
    date_preset = st.selectbox(
        "Time window",
        ["All sample dates", "Last 90 sample days", "Last 30 sample days", "Custom range"],
        help="Recent windows are based on the latest date in the loaded data.",
    )
    if date_preset == "Custom range":
        selected_dates = st.date_input(
            "Transaction date range",
            value=(first_day, last_day),
            min_value=first_day,
            max_value=last_day,
            help="Applies to overview, transaction search, and investigation queue.",
        )
    elif date_preset == "Last 90 sample days":
        selected_dates = (max(first_day, last_day - timedelta(days=89)), last_day)
    elif date_preset == "Last 30 sample days":
        selected_dates = (max(first_day, last_day - timedelta(days=29)), last_day)
    else:
        selected_dates = (first_day, last_day)
    selected_category = st.selectbox("Merchant category", ["All categories", *categories])
    st.divider()
    st.caption("Filters affect the overview, transactions, and case queue. Customer profiles are searched independently.")
    st.caption("Database: live PostgreSQL connection")

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_day, end_day = selected_dates
elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:
    start_day = end_day = selected_dates[0]
else:
    start_day = end_day = selected_dates

if start_day is None or end_day is None:
    st.error("Select both a start date and an end date.")
    st.stop()

if start_day > end_day:
    st.error("The start date must be on or before the end date.")
    st.stop()

filter_sql = "t.txn_timestamp >= :start_time AND t.txn_timestamp < :end_time"
query_params = {
    "start_time": datetime.combine(start_day, datetime.min.time()),
    "end_time": datetime.combine(end_day + timedelta(days=1), datetime.min.time()),
}
if selected_category != "All categories":
    filter_sql += " AND m.category = :category"
    query_params["category"] = selected_category

kpis = read_query(
    f"""
    SELECT COUNT(*) AS total_transactions,
           COALESCE(SUM(t.amount), 0) AS transaction_volume,
           COUNT(*) FILTER (WHERE t.is_fraud) AS known_fraud,
           COALESCE(100.0 * AVG(CASE WHEN t.is_fraud THEN 1.0 ELSE 0.0 END), 0) AS fraud_rate,
           COUNT(DISTINCT fa.alert_id) AS fraud_alerts,
           COUNT(DISTINCT i.case_id) FILTER (WHERE i.outcome = 'PENDING') AS open_investigations
    FROM transaction t
    JOIN merchant m ON m.merchant_id = t.merchant_id
    LEFT JOIN fraud_alert fa ON fa.transaction_id = t.transaction_id
    LEFT JOIN investigation i ON i.alert_id = fa.alert_id
    WHERE {filter_sql}
    """,
    query_params,
).iloc[0]

st.caption(
    f"Showing {start_day:%b %d, %Y} – {end_day:%b %d, %Y}"
    + (f" · {selected_category}" if selected_category != "All categories" else " · All merchant categories")
)

metric_columns = st.columns(6)
metric_columns[0].metric("Transactions", f"{int(kpis['total_transactions']):,}")
metric_columns[1].metric("Transaction volume", f"${float(kpis['transaction_volume']):,.0f}")
metric_columns[2].metric(
    "Dataset-labeled fraud", f"{int(kpis['known_fraud']):,}",
    help="Ground-truth labels provided by the sample dataset; these are not investigator-confirmed cases.",
)
metric_columns[3].metric(
    "Labeled fraud rate", f"{float(kpis['fraud_rate']):.2f}%",
    help="Share of transactions marked as fraud in the dataset.",
)
metric_columns[4].metric(
    "Model alerts", f"{int(kpis['fraud_alerts']):,}",
    help="Alerts created when the model score meets the configured threshold.",
)
metric_columns[5].metric("Cases awaiting review", f"{int(kpis['open_investigations']):,}")

overview_tab, transaction_tab, investigation_tab, customer_tab, model_tab = st.tabs(
    ["Overview", "Transaction search", "Case review", "Customer profile", "Model evaluation"]
)

with overview_tab:
    st.subheader("Portfolio overview")
    st.caption("Use the date and category filters to focus the reporting window. Dataset labels and model alerts describe different things.")
    st.info(
        "How to read this page: **labeled fraud** comes from the sample data. **Model alerts** are generated from risk scores. "
        "Scores below 50 are LOW, 50–79 are MEDIUM, and 80 or higher are HIGH; HIGH scores enter the investigation queue."
    )

    daily_data = read_query(
        f"""
        SELECT DATE_TRUNC('day', t.txn_timestamp)::date AS day,
               COUNT(*) AS transactions,
               COUNT(*) FILTER (WHERE t.is_fraud) AS known_fraud
        FROM transaction t
        JOIN merchant m ON m.merchant_id = t.merchant_id
        WHERE {filter_sql}
        GROUP BY 1
        ORDER BY 1
        """,
        query_params,
    ).set_index("day")

    category_data = read_query(
        f"""
        SELECT m.category,
               COUNT(*) AS transactions,
               COUNT(*) FILTER (WHERE t.is_fraud) AS known_fraud,
               COALESCE(100.0 * AVG(CASE WHEN t.is_fraud THEN 1.0 ELSE 0.0 END), 0) AS fraud_rate
        FROM transaction t
        JOIN merchant m ON m.merchant_id = t.merchant_id
        WHERE {filter_sql}
        GROUP BY m.category
        ORDER BY fraud_rate DESC
        """,
        query_params,
    )
    hour_data = read_query(
        f"""
        SELECT EXTRACT(HOUR FROM t.txn_timestamp)::INT AS hour_of_day,
               COUNT(*) FILTER (WHERE t.is_fraud) AS known_fraud,
               COUNT(*) AS transactions
        FROM transaction t
        JOIN merchant m ON m.merchant_id = t.merchant_id
        WHERE {filter_sql}
        GROUP BY 1
        ORDER BY 1
        """,
        query_params,
    ).set_index("hour_of_day")
    score_distribution = read_query(
        f"""
        SELECT CASE
                   WHEN rs.score_value IS NULL THEN 'Not scored'
                   WHEN rs.score_value >= 80 THEN 'High (80–100)'
                   WHEN rs.score_value >= 50 THEN 'Medium (50–79)'
                   ELSE 'Low (0–49)'
               END AS score_band,
               COUNT(*) AS transactions
        FROM transaction t
        JOIN merchant m ON m.merchant_id = t.merchant_id
        LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
        WHERE {filter_sql}
        GROUP BY 1
        ORDER BY 1
        """,
        query_params,
    )

    chart_left, chart_right = st.columns([1.55, 1])
    with chart_left:
        st.markdown("#### Transaction activity")
        if daily_data.empty:
            st.info("No transaction activity in this date range.")
        else:
            st.line_chart(daily_data[["transactions", "known_fraud"]], y_label="Transactions")
    with chart_right:
        st.markdown("#### Fraud rate by category")
        if category_data.empty:
            st.info("No category data for this selection.")
        else:
            st.bar_chart(category_data.set_index("category")[["fraud_rate"]], y_label="Fraud rate (%)")

    lower_left, lower_right = st.columns([1, 1.2])
    with lower_left:
        st.markdown("#### Known fraud by hour")
        if hour_data.empty:
            st.info("No hourly data for this selection.")
        else:
            st.bar_chart(hour_data[["known_fraud"]], y_label="Transactions")
    with lower_right:
        st.markdown("#### Model score distribution")
        if score_distribution.empty:
            st.info("No model scores are available for this selection.")
        else:
            st.bar_chart(score_distribution.set_index("score_band")["transactions"], y_label="Transactions")

    with st.expander("View category detail"):
        st.dataframe(
            category_data,
            width="stretch",
            hide_index=True,
            column_config={
                "fraud_rate": st.column_config.NumberColumn("Labeled fraud rate", format="%.2f%%"),
                "transactions": st.column_config.NumberColumn("Transactions", format="%d"),
                "known_fraud": st.column_config.NumberColumn("Dataset-labeled fraud", format="%d"),
            },
        )

with transaction_tab:
    st.subheader("Risk transaction explorer")
    st.caption("Search the selected reporting window, narrow the results, and export the rows shown below.")
    score_column, label_column, status_column, row_column = st.columns([1.2, 1.1, 1.3, 1])
    with score_column:
        minimum_score = st.slider("Minimum risk score", min_value=0, max_value=100, value=50)
    with label_column:
        label_filter = st.selectbox("Dataset label", ["All labels", "Fraud-labeled", "Not fraud-labeled"])
    with status_column:
        alert_filter = st.selectbox(
            "Model alert status",
            ["All alert statuses", "Open", "Under review", "Confirmed fraud", "False positive", "No alert"],
        )
    with row_column:
        row_limit = st.select_slider("Maximum rows", options=[25, 50, 100, 250, 500], value=100)
    transaction_search = st.text_input(
        "Search transaction ID, customer, or merchant",
        placeholder="For example: 1204 or grocery_pos",
    )

    transaction_sql = f"""
        SELECT t.transaction_id, t.txn_timestamp, c.customer_id,
               c.first_name || ' ' || c.last_name AS customer,
               m.name AS merchant, m.category, t.amount, t.is_fraud AS dataset_label,
               rs.score_value AS risk_score, rs.model_name,
               fa.status AS alert_status
        FROM transaction t
        JOIN account a ON a.account_id = t.account_id
        JOIN customer c ON c.customer_id = a.customer_id
        JOIN merchant m ON m.merchant_id = t.merchant_id
        LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
        LEFT JOIN fraud_alert fa ON fa.transaction_id = t.transaction_id
        WHERE {filter_sql}
          AND COALESCE(rs.score_value, 0) >= :minimum_score
    """
    transaction_params = {**query_params, "minimum_score": minimum_score, "row_limit": row_limit}
    if label_filter == "Fraud-labeled":
        transaction_sql += " AND t.is_fraud = TRUE"
    elif label_filter == "Not fraud-labeled":
        transaction_sql += " AND t.is_fraud = FALSE"
    if alert_filter == "No alert":
        transaction_sql += " AND fa.alert_id IS NULL"
    elif alert_filter != "All alert statuses":
        alert_status = {
            "Open": "OPEN",
            "Under review": "UNDER_REVIEW",
            "Confirmed fraud": "CONFIRMED_FRAUD",
            "False positive": "FALSE_POSITIVE",
        }[alert_filter]
        transaction_sql += " AND fa.status = :alert_status"
        transaction_params["alert_status"] = alert_status
    if transaction_search.strip():
        transaction_sql += """
          AND (CAST(t.transaction_id AS TEXT) ILIKE :search
               OR CAST(c.customer_id AS TEXT) ILIKE :search
               OR (c.first_name || ' ' || c.last_name) ILIKE :search
               OR m.name ILIKE :search OR m.category ILIKE :search)
        """
        transaction_params["search"] = f"%{transaction_search.strip()}%"
    transaction_sql += " ORDER BY rs.score_value DESC NULLS LAST, t.txn_timestamp DESC LIMIT :row_limit"
    transactions = read_query(transaction_sql, transaction_params)

    if transactions.empty:
        st.info("No transactions match the selected filters.")
    else:
        st.caption(f"Showing the highest-risk {len(transactions):,} matching transactions.")
        st.dataframe(
            transactions,
            width="stretch",
            hide_index=True,
            column_config={
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "risk_score": st.column_config.ProgressColumn(
                    "Risk score", min_value=0, max_value=100, format="%.1f"
                ),
                "dataset_label": st.column_config.CheckboxColumn(
                    "Dataset fraud label",
                    help="Ground-truth label from the sample data; it is not an investigation result.",
                ),
                "txn_timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
            },
        )
        st.download_button(
            "Download filtered transactions (CSV)",
            data=transactions.to_csv(index=False).encode("utf-8"),
            file_name="finguard_filtered_transactions.csv",
            mime="text/csv",
        )

with investigation_tab:
    st.subheader("Investigation queue")
    st.caption("Pending cases appear first. Resolved cases stay visible for review and cannot be changed from this screen.")
    case_update_notice = st.session_state.pop("case_update_notice", None)
    if case_update_notice:
        st.success(case_update_notice)
    outcome_filter = st.selectbox(
        "Filter cases by outcome",
        ["PENDING", "All outcomes", "FRAUD", "NOT_FRAUD"],
        format_func=lambda value: "All outcomes" if value == "All outcomes" else value.replace("_", " ").title(),
    )
    investigation_sql = f"""
        SELECT i.case_id, i.outcome, i.opened_at, i.resolved_at, i.alert_id,
               fa.status AS alert_status, t.transaction_id, t.txn_timestamp, t.amount,
               rs.score_value AS risk_score, m.category,
               c.first_name || ' ' || c.last_name AS customer
        FROM investigation i
        JOIN fraud_alert fa ON fa.alert_id = i.alert_id
        JOIN transaction t ON t.transaction_id = fa.transaction_id
        JOIN merchant m ON m.merchant_id = t.merchant_id
        JOIN account a ON a.account_id = t.account_id
        JOIN customer c ON c.customer_id = a.customer_id
        LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
        WHERE {filter_sql}
    """
    investigation_params = dict(query_params)
    if outcome_filter != "All outcomes":
        investigation_sql += " AND i.outcome = :outcome"
        investigation_params["outcome"] = outcome_filter
    investigation_sql += " ORDER BY CASE WHEN i.outcome = 'PENDING' THEN 0 ELSE 1 END, rs.score_value DESC NULLS LAST, i.opened_at DESC"
    investigations = read_query(investigation_sql, investigation_params)
    if investigations.empty:
        st.info("No investigations match the selected filters.")
    else:
        pending_cases = investigations[investigations["outcome"] == "PENDING"]
        resolved_count = int((investigations["outcome"] != "PENDING").sum())
        case_metrics = st.columns(3)
        case_metrics[0].metric("Cases in this view", f"{len(investigations):,}")
        case_metrics[1].metric("Awaiting review", f"{len(pending_cases):,}")
        case_metrics[2].metric("Resolved", f"{resolved_count:,}")
        st.dataframe(
            investigations.drop(columns=["alert_id"]),
            width="stretch",
            hide_index=True,
            column_config={
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f"),
            },
        )
        if pending_cases.empty:
            st.info("There are no pending cases to update in this view.")
        else:
            st.markdown("#### Resolve a pending case")
            case_options = pending_cases["case_id"].tolist()
            selected_case = st.selectbox(
                "Pending case",
                case_options,
                format_func=lambda case_id: (
                    f"Case #{case_id} · transaction "
                    f"{int(pending_cases.loc[pending_cases['case_id'] == case_id, 'transaction_id'].iloc[0])}"
                ),
            )
            selected_row = pending_cases.loc[pending_cases["case_id"] == selected_case].iloc[0]
            score_text = (
                f"{float(selected_row['risk_score']):.1f}/100"
                if pd.notna(selected_row["risk_score"]) else "No model score"
            )
            st.info(
                f"Customer: **{selected_row['customer']}** · Merchant category: **{selected_row['category']}** · "
                f"Amount: **${float(selected_row['amount']):,.2f}** · Model score: **{score_text}**"
            )
            with st.form("resolve_investigation"):
                new_outcome = st.selectbox(
                    "Investigation result",
                    options=["FRAUD", "NOT_FRAUD"],
                    format_func=lambda value: "Confirm fraud" if value == "FRAUD" else "Mark as not fraud",
                    help="This updates the investigation and its linked alert in PostgreSQL.",
                )
                confirm_update = st.checkbox("I reviewed this case and confirm the database update.")
                submitted = st.form_submit_button("Save case result", type="primary")
            if submitted and not confirm_update:
                st.warning("Confirm the case result before saving.")
            elif submitted:
                with engine.begin() as connection:
                    connection.execute(
                        text("UPDATE investigation SET outcome = :outcome WHERE case_id = :case_id"),
                        {"outcome": new_outcome, "case_id": int(selected_case)},
                    )
                    updated = connection.execute(
                        text(
                            """
                            SELECT i.outcome, i.resolved_at, fa.status AS alert_status
                            FROM investigation i
                            JOIN fraud_alert fa ON fa.alert_id = i.alert_id
                            WHERE i.case_id = :case_id
                            """
                        ),
                        {"case_id": int(selected_case)},
                    ).mappings().one()
                st.session_state["case_update_notice"] = (
                    f"Saved: {updated['outcome']} · linked alert is {updated['alert_status']} · "
                    f"resolved at {updated['resolved_at']}."
                )
                st.rerun()

with customer_tab:
    st.subheader("Customer risk profile")
    st.caption("Review an aggregate risk summary and the customer's highest-scored activity.")
    with st.form("customer_lookup"):
        customer_id = st.number_input("Customer ID", min_value=1, step=1, value=1)
        customer_submitted = st.form_submit_button("Look up customer", type="primary")
    if customer_submitted:
        customer_risk = read_query(
            "SELECT * FROM get_customer_risk(:customer_id)",
            {"customer_id": int(customer_id)},
        )
        if customer_risk.empty:
            st.info("No customer found for that ID.")
        else:
            customer_row = customer_risk.iloc[0]
            st.markdown(f"#### {customer_row['customer_name']}")
            customer_metrics = st.columns(4)
            customer_metrics[0].metric("Transactions", f"{int(customer_row['transaction_count']):,}")
            customer_metrics[1].metric("Total amount", f"${float(customer_row['total_amount']):,.2f}")
            average_score = customer_row["average_score_value"]
            customer_metrics[2].metric(
                "Average model score",
                f"{float(average_score):.1f}/100" if pd.notna(average_score) else "Not scored",
                help="Average model risk score across this customer's scored transactions.",
            )
            customer_metrics[3].metric("Model alerts", f"{int(customer_row['alert_count']):,}")
            customer_transactions = read_query(
                """
                SELECT t.transaction_id, t.txn_timestamp, m.name AS merchant, m.category,
                       t.amount, rs.score_value AS risk_score, fa.status AS alert_status
                FROM transaction t
                JOIN account a ON a.account_id = t.account_id
                JOIN merchant m ON m.merchant_id = t.merchant_id
                LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
                LEFT JOIN fraud_alert fa ON fa.transaction_id = t.transaction_id
                WHERE a.customer_id = :customer_id
                ORDER BY rs.score_value DESC NULLS LAST, t.txn_timestamp DESC
                LIMIT 25
                """,
                {"customer_id": int(customer_id)},
            )
            st.markdown("#### Highest-risk transactions")
            st.dataframe(
                customer_transactions,
                width="stretch",
                hide_index=True,
                column_config={
                    "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                    "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f"),
                },
            )

with model_tab:
    st.subheader("Model performance")
    st.caption("Validation PR-AUC selects the model; the metrics below come from the separate test set.")
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    results_path = outputs_dir / "model_results.csv"
    if results_path.exists():
        model_results = pd.read_csv(results_path)
        if "validation_pr_auc" not in model_results.columns:
            st.warning("This saved report predates the current validation/test split. Rerun ml_pipeline.py to refresh it.")
        selection_metric = "validation_pr_auc" if "validation_pr_auc" in model_results.columns else "pr_auc"
        model_results = model_results.sort_values(selection_metric, ascending=False)
        if not model_results.empty:
            selected_rows = model_results[model_results["selected"]] if "selected" in model_results.columns else model_results.iloc[0:0]
            best_model = selected_rows.iloc[0] if not selected_rows.empty else model_results.iloc[0]
            metric_label = "Selected model by validation PR-AUC" if selection_metric == "validation_pr_auc" else "Top model in saved report by PR-AUC"
            metric_value = best_model[selection_metric]
            st.metric(metric_label, best_model["model"], f"{metric_value:.3f} PR-AUC")
        st.dataframe(
            model_results,
            width="stretch",
            hide_index=True,
            column_config={
                "selected": st.column_config.CheckboxColumn("Selected by validation"),
                "precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1, format="%.3f"),
                "recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.3f"),
                "f1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f"),
                "roc_auc": st.column_config.ProgressColumn("ROC-AUC", min_value=0, max_value=1, format="%.3f"),
                "pr_auc": st.column_config.ProgressColumn("PR-AUC", min_value=0, max_value=1, format="%.3f"),
                "validation_pr_auc": st.column_config.ProgressColumn("Validation PR-AUC", min_value=0, max_value=1, format="%.3f"),
            },
        )
    else:
        st.info("No model report found yet. Run ml_pipeline.py to generate evaluation results.")

    available_charts = [
        ("roc_curve.png", "ROC curve"),
        ("precision_recall_curve.png", "Precision-recall curve"),
        ("feature_importance.png", "Feature importance"),
        ("fraud_by_category.png", "Fraud by category"),
        ("fraud_by_hour.png", "Fraud by hour"),
    ]
    for chart_index in range(0, len(available_charts), 2):
        chart_columns = st.columns(2)
        for column, (filename, title) in zip(chart_columns, available_charts[chart_index:chart_index + 2]):
            chart_path = outputs_dir / filename
            if chart_path.exists():
                with column:
                    st.markdown(f"#### {title}")
                    st.image(str(chart_path), width="stretch")

st.divider()
st.caption("FinGuard AI · PostgreSQL-backed fraud analytics · Scores and operational actions are subject to your database state.")
