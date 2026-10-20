# FinGuard AI — likely faculty questions

## 1. Why did you use PR-AUC to choose the model?

Fraud is the minority class: the included sample has 700 fraud transactions out of 10,000. PR-AUC focuses on precision and recall for that positive class, which makes it useful alongside ROC-AUC for this imbalanced task.

## 2. How is the selected model chosen?

The pipeline selects among the models using validation PR-AUC, then compares the refitted models on a separate untouched test split. The latest saved report selects Random Forest with validation PR-AUC 0.948; the model comparison uses the held-out test split.

## 3. What is the difference between `score_value` and a risk level?

`risk_score.score_value` is a numeric score constrained by the schema to the range 0–100. The pipeline derives LOW, MEDIUM, or HIGH from its thresholds; `risk_level` is not a column in the base table.

## 4. What does the investigation trigger do?

When an investigation is inserted as `PENDING`, the advanced trigger moves its linked alert to `UNDER_REVIEW`. Changing the outcome to `FRAUD` or `NOT_FRAUD` stamps `resolved_at` and changes the alert to `CONFIRMED_FRAUD` or `FALSE_POSITIVE`. Reopening a case clears the timestamp and moves the alert back to `UNDER_REVIEW`. Installing the advanced SQL also reconciles alerts linked to cases created before the trigger existed.

## 5. Why does the investigation table use `outcome`, not `status`?

That is the real column in `schema.sql`, with allowed values `PENDING`, `FRAUD`, and `NOT_FRAUD`. The alert table separately has a `status` column with its own allowed values.

## 6. What do the indexes do?

Indexes support the joins and filters used by the project, including transaction lookups by account, merchant, or timestamp and alerts by status. `risk_score.transaction_id` is already indexed through its UNIQUE constraint. In the local merchant-plus-timestamp demonstration, the existing timestamp index was used before the composite index and the composite index applied both predicates afterward; the single-run timings were too small and noisy to claim a speedup.

## 7. How was normalization applied?

The schema separates entities into tables and relates them with primary and foreign keys. For example, transactions reference account, merchant, device, and location instead of storing all those entities' descriptive columns repeatedly on every transaction row; the schema comments identify the design as 3NF.

## 8. Which data is synthetic?

According to `load_data.py`, income, accounts, card type/expiry, devices, loans, and loan payments are generated. Transaction, customer, location, merchant, and card source data comes from the Sparkov CSV, with account/card and location-related values transformed or associated by the loader.

## 9. How did you avoid data leakage?

The supervised model's stored scores use five-fold stratified out-of-fold predictions, so the model scoring a row was not fitted on that row's label. Customer count and average-spend aggregates from the full dataset were removed because they could share information across folds. The evaluation still uses a random transaction-level split from the same sampled customer population, so it does not measure generalization to unseen customers or a separate dataset.

## 10. How was the held-out comparison performed?

`ml_pipeline.py` makes stratified 60/20/20 train, validation, and test splits with random state 42. Validation PR-AUC selects the model; the reported comparison metrics and charts are calculated on the separate test set.

## 11. What are the current model results?

On the refreshed report, Random Forest was selected with validation PR-AUC 0.948. On the separate held-out test split it recorded precision 0.915, recall 0.929, F1 0.922, ROC-AUC 0.992, and PR-AUC 0.962. These results describe transaction-level splits of the included 10,000-row sample, not performance on a second dataset or unseen customers.

## 12. How many records are in the database?

The loader builds the relational data from the included 10,000-row sample and generates synthetic account, device, and loan data. For exact row counts, run the queries in `queries.sql` against the database currently loaded for the demo.

## 13. How are model scores turned into alerts?

The pipeline creates an alert when the model probability is at least 0.50 and scales the stored numeric score to 0–100. A score at least 0.80 is classified HIGH and receives an investigation.

## 14. What does the dashboard show?

It displays transaction volume and dataset-labeled fraud separately from model-generated alerts; charts by date, category, hour, and model-score band; transaction search with alert-status filters and CSV export; a pending-case workflow with read-only resolved history; customer profiles; and model evaluation reports.

## 15. What happens if you run the loader again?

The loader executes `schema.sql`, which drops and recreates the project tables and base views before loading the CSV. It requires `--reset` and a typed `RESET` confirmation; non-interactive use also requires `--yes`. This replaces existing database data and removes pipeline and advanced objects, so the ML pipeline and advanced SQL setup must be run again afterward.
