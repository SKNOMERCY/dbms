# FinGuard AI portfolio guide

Use this guide to present the project accurately in a demo, interview, resume, or LinkedIn post. The project is an educational prototype evaluated on one simulated 10,000-row dataset.

## Five-minute demo script

### 0:00–0:30 — Problem and scope

“FinGuard AI is a DBMS and analytics prototype for exploring transaction risk. It combines a PostgreSQL relational model, a reproducible Python scoring pipeline, and a Streamlit analyst dashboard. The dataset is simulated, so the results demonstrate the workflow rather than real-world fraud performance.”

### 0:30–1:30 — Relational model

Open `ER_DIAGRAM.md`, then run the row-count query at the top of `queries.sql`. Point out the customer-to-account-to-transaction relationships, foreign keys, constraints, and the separate risk score, alert, and investigation records. Show one four-table join and the category summary view.

### 1:30–2:30 — Database behavior

Open `advanced_db.sql`. Explain that the high-risk views and customer-risk function support analyst queries. Show the investigation trigger: resolving a case stamps `resolved_at` and updates the linked alert; reopening it clears the timestamp and returns the alert to review. The trigger demonstration is wrapped in a transaction and rolled back.

### 2:30–3:30 — Modeling

Open `outputs/model_results.csv`. Explain that model choice uses validation PR-AUC, while final comparison uses a separate held-out test split. The recorded Random Forest result is 0.948 validation PR-AUC and 0.962 test PR-AUC on this sample. Explain that PR-AUC is useful for an imbalanced label, and that transaction-level splitting does not prove performance on unseen customers or future data.

### 3:30–4:30 — Dashboard

Open the dashboard and show its shared date/category/amount/risk/alert/case filters, KPI definitions, and interactive overview. In **Transactions**, search and page through records, inspect actual score, model, alert, and case fields, and demonstrate CSV export. In **Investigations**, show an alert without a case, then create, resolve, or reopen one on a demo/test database. In **Model evaluation**, move the operating threshold and explain that it changes test-set metrics only; it does not rewrite stored alerts.

### 4:30–5:00 — Close

“The main engineering focus was data integrity and an explainable end-to-end flow: relational constraints, isolated evaluation, traceable case status, and a dashboard that lets an analyst move from aggregate trends to a specific transaction. The next step would be evaluation on a second time-separated dataset and repeated index benchmarks at a realistic scale.”

## Resume bullet

Built FinGuard AI, an end-to-end PostgreSQL, Python, scikit-learn, and Streamlit fraud analytics prototype over 10,000 simulated transactions; designed a 12-table schema with constraints, views, indexes, a SQL function, and trigger-driven case resolution; implemented leakage-aware model comparison and an analyst dashboard, with Random Forest achieving 0.962 held-out test PR-AUC on the sample.

## LinkedIn project post

I built FinGuard AI, a fraud analytics prototype that brings database design, machine learning, and an analyst dashboard into one project.

The project uses PostgreSQL for a 12-table relational model, Python and scikit-learn to compare transaction-risk models, and Streamlit for transaction exploration and investigation review. I also added SQL views, a customer-risk function, trigger-driven case updates, safe CSV export, and tests for key workflow behavior.

On the included 10,000-row simulated dataset, the Random Forest reached 0.948 validation PR-AUC and 0.962 held-out test PR-AUC. These are sample-specific results, not a claim about production fraud detection.

This project helped me practice connecting database integrity, model evaluation, and a useful analyst workflow. Feedback is welcome.

#PostgreSQL #Python #MachineLearning #DataAnalytics #Streamlit #DBMS

## Dashboard screenshot checklist

The dashboard script passed Streamlit AppTest, including a rerun with the High risk filter, against the configured PostgreSQL database. It was not visually inspected in a browser in this environment, so no screenshot is included. After launching the app locally and loading the sample database, capture real screenshots under `docs/screenshots/` and link them from `README.md`:

1. Overview with the shared filter context, KPI rows, and chart titles visible.
2. Transactions with search, page controls, a result table, and the evidence panel.
3. Customer profile with a risk summary and scored transaction history.
4. Investigations showing the alert queue and the pending-case workflow.
5. Model evaluation showing the split description, threshold metrics, and precision–recall curve.

Use sample records only. Do not include local database credentials, terminal history, or private account information in screenshots.

## Interview questions and suggested answers

1. **Why use PostgreSQL?** It gives the project durable relational storage, foreign-key and check-constraint enforcement, SQL analytics, views, transactions, and trigger support.
2. **Why separate risk scores from alerts?** A score is a model output; an alert represents a review action. Keeping them separate preserves the scoring record even when alert thresholds or investigation outcomes change.
3. **How is referential integrity enforced?** Primary and foreign keys connect entities, unique constraints prevent duplicate relationships such as multiple payments for the same loan due date, and checks constrain valid amounts and outcomes.
4. **What does the investigation trigger do?** It stamps resolution time and synchronizes the linked alert status when an investigation is resolved; reopening resets the timestamp and returns the alert to review.
5. **Why use a validation set and a test set?** Validation selects the candidate model; the untouched test split estimates final performance after that choice, reducing optimistic reporting from selecting on the test results.
6. **Why report PR-AUC?** Fraud labels are imbalanced. Precision-recall summarizes the trade-off around the positive class more directly than accuracy alone.
7. **What does the 0.962 test PR-AUC mean?** It describes ranking performance on the held-out portion of this simulated sample under the selected split. It does not establish production performance or generalization to new customers.
8. **How do you reduce leakage in database scoring?** The pipeline uses out-of-fold predictions for supervised-model scores written to the database, and customer aggregates calculated across the full dataset are excluded from those features.
9. **What is the role of Isolation Forest?** It provides an unsupervised anomaly-detection baseline, which is compared with supervised models using the same held-out labels for evaluation.
10. **Why use a composite index?** The query filters by merchant and timestamp together. The composite index can apply both predicates in the index; the measured single-run latency was noisy and did not prove a speedup on 10,000 rows.
11. **How would you evaluate this for production?** Use time-based splits, a second independent dataset, customer-level holdouts, calibration and threshold analysis, cost-weighted metrics, drift monitoring, and human review controls.
12. **How do you prevent accidental data deletion?** The loader requires `--reset` and an explicit typed confirmation. Automation requires the additional `--yes` flag.
13. **What happens if a database load fails?** Schema recreation and data inserts run in one database transaction; the connection context rolls back on failure and the connection closes in `finally`.
14. **Why neutralize CSV values?** Spreadsheet applications may interpret cells beginning with formula markers as executable formulas. Text cells are prefixed before export; numeric values remain numeric.
15. **What would you improve next?** Add a time-separated external evaluation, capture dashboard screenshots, benchmark indexes repeatedly on a larger dataset, and add deployment configuration and monitoring only after validating the workflow.
