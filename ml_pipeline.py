"""
FinGuard AI - ML pipeline
=========================
Steps:
  1. Read transactions (joined with customer + merchant) from PostgreSQL
  2. Build features with Pandas
  3. Train & compare 3 models: Logistic Regression, Random Forest, Isolation Forest
  4. Write results into risk_score, fraud_alert, investigation

Install:  pip install pandas numpy scikit-learn sqlalchemy psycopg2-binary matplotlib
Run:      python ml_pipeline.py [--reports-only]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve, roc_curve)

# ======================================================================
# CONFIG
# ======================================================================
DB_PASSWORD = os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise RuntimeError("Set PGPASSWORD before running this script. DB_PASSWORD is also accepted for compatibility.")

DB_URL = URL.create(
    "postgresql+psycopg2",
    username=os.getenv("PGUSER", "postgres"),
    password=DB_PASSWORD,
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    database=os.getenv("PGDATABASE", "finguard"),
)
# Coordinates are joined from location to return the model's expected columns.
#   transaction_id, customer_id, amount, trans_time, is_fraud,
#   cust_lat, cust_long, merch_lat, merch_long, category, dob
EXTRACT_SQL = """
SELECT  t.transaction_id,
        c.customer_id,
        t.amount,
        t.txn_timestamp AS trans_time,
        t.is_fraud,
        home.latitude   AS cust_lat,
        home.longitude  AS cust_long,
        merchant_loc.latitude  AS merch_lat,
        merchant_loc.longitude AS merch_long,
        m.category,
        c.dob
FROM transaction t
JOIN account  a ON a.account_id  = t.account_id
JOIN customer c ON c.customer_id = a.customer_id
JOIN merchant m ON m.merchant_id = t.merchant_id
JOIN location home ON home.location_id = c.home_location_id
JOIN location merchant_loc ON merchant_loc.location_id = t.location_id
"""

# Output columns match schema.sql; risk_level is derived when reporting.
# ======================================================================

RANDOM_STATE = 42
ALERT_THRESHOLD = 0.50   # risk score >= this creates a fraud_alert
HIGH_THRESHOLD = 0.80    # risk score >= this also opens an investigation

engine = create_engine(DB_URL)


# ---------------------------------------------------------------- 1. LOAD
def load_data():
    df = pd.read_sql(EXTRACT_SQL, engine)
    df["trans_time"] = pd.to_datetime(df["trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    print(f"Loaded {len(df)} transactions, fraud rate = {df['is_fraud'].mean():.2%}")
    return df


# ------------------------------------------------------- 2. FEATURE BUILD
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/long points, in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_features(df):
    f = pd.DataFrame(index=df.index)
    f["amount"] = df["amount"]
    f["log_amount"] = np.log1p(df["amount"])              # tames huge amounts
    f["hour"] = df["trans_time"].dt.hour
    f["is_night"] = ((f["hour"] >= 22) | (f["hour"] <= 3)).astype(int)  # EDA finding
    f["day_of_week"] = df["trans_time"].dt.dayofweek
    f["distance_km"] = haversine_km(df["cust_lat"], df["cust_long"],
                                    df["merch_lat"], df["merch_long"])
    f["age"] = (df["trans_time"] - df["dob"]).dt.days / 365.25

    # Merchant category -> one-hot columns
    f = f.join(pd.get_dummies(df["category"], prefix="cat", dtype=int))
    return f


# ------------------------------------------------------------ 3. TRAINING
def minmax(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)


def train_and_compare(X, y):
    # Keep a final test set untouched during model selection.
    X_dev, X_te, y_dev, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=RANDOM_STATE)

    models = {
        # Baseline: simple, explainable. Scaling matters for LR.
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000)),
        # Handles non-linear patterns; class_weight fixes 93/7 imbalance.
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        # Unsupervised: never sees labels, flags "unusual" transactions.
        "IsolationForest": IsolationForest(
            n_estimators=200, contamination="auto",
            random_state=RANDOM_STATE),
    }

    validation_results = []
    for name, model in models.items():
        if name == "IsolationForest":
            model.fit(X_tr)                                # no labels used
            raw_tr = -model.score_samples(X_tr)            # higher = more anomalous
            lo, hi = raw_tr.min(), raw_tr.max()
            scorer = lambda d, m=model, lo=lo, hi=hi: minmax(-m.score_samples(d), lo, hi)
        else:
            model.fit(X_tr, y_tr)
            scorer = lambda d, m=model: m.predict_proba(d)[:, 1]

        scores = scorer(X_val)
        validation_results.append({
            "model": name,
            "pr_auc": average_precision_score(y_val, scores),
        })

    validation = pd.DataFrame(validation_results).set_index("model")
    best = validation["pr_auc"].idxmax()
    print("\n=== Model selection (validation split) ===")
    print(validation.round(3), "\n")
    print(f"Selected by validation PR-AUC: {best}")

    # Refit each candidate on train + validation data, then evaluate once on
    # the untouched test split. The test results are reporting only.
    results, scorers, test_scores = [], {}, {}
    for name, model in models.items():
        if name == "IsolationForest":
            model.fit(X_dev)
            raw_dev = -model.score_samples(X_dev)
            lo, hi = raw_dev.min(), raw_dev.max()
            scorer = lambda d, m=model, lo=lo, hi=hi: minmax(-m.score_samples(d), lo, hi)
        else:
            model.fit(X_dev, y_dev)
            scorer = lambda d, m=model: m.predict_proba(d)[:, 1]

        scores = scorer(X_te)
        pred = (scores >= ALERT_THRESHOLD).astype(int)
        results.append({
            "model": name,
            "selected": name == best,
            "validation_pr_auc": validation.loc[name, "pr_auc"],
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_te, scores),
            "pr_auc": average_precision_score(y_te, scores),
        })
        scorers[name] = scorer
        test_scores[name] = scores

    res = pd.DataFrame(results).set_index("model").round(3)
    print("\n=== Final model comparison (untouched 20% test set) ===")
    print(res, "\n")
    evaluation = {"y_test": y_te, "scores": test_scores}
    return best, scorers[best], models[best], res, models, evaluation


# ------------------------------------------------------------- 3B. OUTPUTS
def save_outputs(results, models, evaluation):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    results.rename_axis("model").reset_index().to_csv(
        os.path.join(output_dir, "model_results.csv"), index=False
    )

    y_test = evaluation["y_test"]
    scores_by_model = evaluation["scores"]
    for name, scores in scores_by_model.items():
        predictions = (scores >= ALERT_THRESHOLD).astype(int)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set(
            title=f"{name} — Confusion Matrix (Held-Out Test Set)",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
        )
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center")
        fig.colorbar(image, ax=ax, label="Transactions")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"confusion_matrix_{name}.png"), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, scores in scores_by_model.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, scores)
        ax.plot(false_positive_rate, true_positive_rate, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set(title="ROC Curves (Held-Out Test Set)", xlabel="False Positive Rate",
           ylabel="True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ax.plot(recall, precision, label=name)
    ax.set(title="Precision-Recall Curves (Held-Out Test Set)", xlabel="Recall",
           ylabel="Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=150)
    plt.close(fig)

    random_forest = models["RandomForest"]
    importance = pd.Series(random_forest.feature_importances_, index=random_forest.feature_names_in_)
    top_features = importance.nlargest(10).sort_values()
    top_features.rename_axis("feature").rename("importance").reset_index().sort_values(
        "importance", ascending=False
    ).to_csv(os.path.join(output_dir, "top_features.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features.index, top_features.values, color="steelblue")
    ax.set(title="Top 10 Random Forest Features", xlabel="Feature Importance",
           ylabel="Feature")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)

    category_data = pd.read_sql(
        """
        SELECT m.category,
               100.0 * AVG(CASE WHEN t.is_fraud THEN 1.0 ELSE 0.0 END) AS fraud_rate
        FROM transaction t
        JOIN merchant m ON m.merchant_id = t.merchant_id
        GROUP BY m.category
        ORDER BY fraud_rate DESC
        """,
        engine,
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(category_data["category"], category_data["fraud_rate"], color="darkorange")
    ax.set(title="Fraud Rate by Merchant Category", xlabel="Merchant Category",
           ylabel="Fraud Rate (%)")
    ax.tick_params(axis="x", labelrotation=60)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fraud_by_category.png"), dpi=150)
    plt.close(fig)

    hour_data = pd.read_sql(
        """
        SELECT EXTRACT(HOUR FROM txn_timestamp)::INT AS hour_of_day,
               100.0 * AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END) AS fraud_rate
        FROM transaction
        GROUP BY 1
        ORDER BY 1
        """,
        engine,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(hour_data["hour_of_day"], hour_data["fraud_rate"], color="seagreen")
    ax.set(title="Fraud Rate by Hour of Day", xlabel="Hour of Day",
           ylabel="Fraud Rate (%)", xticks=range(24))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fraud_by_hour.png"), dpi=150)
    plt.close(fig)

    print(f"Saved output files to {output_dir}")


# -------------------------------------------------------- 4. WRITE RESULTS
def level(score):
    return "HIGH" if score >= HIGH_THRESHOLD else "MEDIUM" if score >= ALERT_THRESHOLD else "LOW"


def write_results(df, X, best_name, scorer, best_model):
    if best_name == "IsolationForest":
        scores = scorer(X)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_predict(
            best_model, X, df["is_fraud"], cv=cv, method="predict_proba"
        )[:, 1]
    out = pd.DataFrame({"transaction_id": df["transaction_id"], "score": scores})
    out["risk_level"] = out["score"].apply(level)
    out["risk_score"] = (out["score"] * 100).round(2)
    out["model_name"] = best_name

    with engine.begin() as conn:                       # one transaction: all or nothing
        # Clear old results so the script can be re-run safely
        conn.execute(text("DELETE FROM investigation"))
        conn.execute(text("DELETE FROM fraud_alert"))
        conn.execute(text("DELETE FROM risk_score"))

        # risk_score: one row per transaction
        out[["transaction_id", "risk_score", "model_name"]] \
            .rename(columns={"risk_score": "score_value"}) \
            .to_sql("risk_score", conn, if_exists="append", index=False, chunksize=2000)

        # fraud_alert: only risky transactions
        alerts = out[out["score"] >= ALERT_THRESHOLD].copy()
        alerts["reason"] = f"Flagged by {best_name}"
        alerts["status"] = "OPEN"
        alerts[["transaction_id", "risk_score", "status", "reason"]] \
            .to_sql("fraud_alert", conn, if_exists="append", index=False)

        # investigation: only HIGH alerts; need alert_id generated by the DB
        ids = pd.read_sql(text("SELECT alert_id, transaction_id FROM fraud_alert"), conn)
        high = alerts[alerts["risk_level"] == "HIGH"].merge(ids, on="transaction_id")
        inv = pd.DataFrame({"alert_id": high["alert_id"],
                            "outcome": "PENDING"})
        inv.to_sql("investigation", conn, if_exists="append", index=False)

    print(f"Wrote {len(out)} risk scores, {len(alerts)} alerts, {len(inv)} investigations")


# -------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud models and generate evaluation reports.")
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Refresh evaluation CSVs and charts without changing database scores, alerts, or cases.",
    )
    args = parser.parse_args()

    data = load_data()
    features = build_features(data)
    best_name, best_scorer, best_model, results, models, evaluation = train_and_compare(
        features, data["is_fraud"]
    )
    save_outputs(results, models, evaluation)
    if args.reports_only:
        print("Reports refreshed; database risk scores, alerts, and investigations were not changed.")
    else:
        write_results(data, features, best_name, best_scorer, best_model)
