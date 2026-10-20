-- 1. Compare the transaction lookup plan before and after the composite index.
-- RUN: Explain before composite index
DROP INDEX IF EXISTS idx_advanced_txn_merchant_timestamp;
EXPLAIN ANALYZE
SELECT transaction_id, amount
FROM transaction
WHERE merchant_id = 1
  AND txn_timestamp >= TIMESTAMP '2020-06-21 00:00:00'
  AND txn_timestamp < TIMESTAMP '2020-06-22 00:00:00';
-- RUN: Create requested indexes
CREATE INDEX IF NOT EXISTS idx_txn_merchant ON transaction(merchant_id);
CREATE INDEX IF NOT EXISTS idx_txn_account ON transaction(account_id);
CREATE INDEX IF NOT EXISTS idx_txn_timestamp ON transaction(txn_timestamp);
CREATE INDEX IF NOT EXISTS idx_alert_status ON fraud_alert(status);
CREATE INDEX IF NOT EXISTS idx_advanced_txn_merchant_timestamp ON transaction(merchant_id, txn_timestamp);
-- risk_score.transaction_id already has a UNIQUE-constraint index in schema.sql.
-- RUN: Explain after composite index
EXPLAIN ANALYZE
SELECT transaction_id, amount
FROM transaction
WHERE merchant_id = 1
  AND txn_timestamp >= TIMESTAMP '2020-06-21 00:00:00'
  AND txn_timestamp < TIMESTAMP '2020-06-22 00:00:00';

-- 2. Show HIGH-risk transactions with their customer, merchant, score, and alert.
-- RUN: Create high-risk transaction view
CREATE OR REPLACE VIEW v_high_risk_transactions AS
SELECT t.transaction_id, t.txn_timestamp, t.amount, t.is_fraud,
       c.customer_id, c.first_name, c.last_name,
       m.merchant_id, m.name AS merchant_name, m.category,
       rs.score_value, fa.alert_id, fa.status AS alert_status
FROM transaction t
JOIN account a ON a.account_id = t.account_id
JOIN customer c ON c.customer_id = a.customer_id
JOIN merchant m ON m.merchant_id = t.merchant_id
JOIN risk_score rs ON rs.transaction_id = t.transaction_id
LEFT JOIN fraud_alert fa ON fa.transaction_id = t.transaction_id
WHERE rs.score_value >= 80;

-- 3. Aggregate transaction and risk metrics without multiplying transaction rows.
-- RUN: Create advanced customer risk summary view
CREATE OR REPLACE VIEW v_customer_risk_summary_advanced AS
SELECT c.customer_id,
       c.first_name || ' ' || c.last_name AS customer_name,
       COUNT(DISTINCT t.transaction_id) AS transaction_count,
       COALESCE(SUM(t.amount), 0::numeric) AS total_amount,
       ROUND(AVG(rs.score_value), 2) AS average_score_value,
       COUNT(DISTINCT fa.alert_id) AS alert_count
FROM customer c
LEFT JOIN account a ON a.customer_id = c.customer_id
LEFT JOIN transaction t ON t.account_id = a.account_id
LEFT JOIN risk_score rs ON rs.transaction_id = t.transaction_id
LEFT JOIN fraud_alert fa ON fa.transaction_id = t.transaction_id
GROUP BY c.customer_id, c.first_name, c.last_name;

-- 4. Return the advanced risk summary row for one customer.
-- RUN: Create customer risk function
CREATE OR REPLACE FUNCTION get_customer_risk(p_customer_id INTEGER)
RETURNS TABLE (
    customer_id INTEGER,
    customer_name TEXT,
    transaction_count BIGINT,
    total_amount NUMERIC,
    average_score_value NUMERIC,
    alert_count BIGINT
)
LANGUAGE sql
STABLE
AS $function$
    SELECT s.customer_id, s.customer_name, s.transaction_count,
           s.total_amount, s.average_score_value, s.alert_count
    FROM v_customer_risk_summary_advanced s
    WHERE s.customer_id = p_customer_id;
$function$;
-- RUN: Test customer risk function
SELECT * FROM get_customer_risk((SELECT MIN(customer_id) FROM customer));

-- 5. Keep investigation creation, resolution, and reopening synchronized with alerts.
-- RUN: Ensure investigation resolved timestamp exists (already in fresh schema)
ALTER TABLE investigation ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMP;
-- Migrate the legacy timestamp column when installing onto an older project schema.
DO $migration$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'investigation'
          AND column_name = 'closed_at'
    ) THEN
        EXECUTE 'UPDATE investigation SET resolved_at = closed_at '
                'WHERE resolved_at IS NULL AND closed_at IS NOT NULL';
    END IF;
END;
$migration$;
-- RUN: Create investigation outcome trigger function
CREATE OR REPLACE FUNCTION sync_investigation_resolution()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.outcome IN ('FRAUD', 'NOT_FRAUD') THEN
            NEW.resolved_at := COALESCE(NEW.resolved_at, CURRENT_TIMESTAMP);
        ELSE
            NEW.resolved_at := NULL;
        END IF;
        UPDATE fraud_alert
        SET status = CASE NEW.outcome
                         WHEN 'FRAUD' THEN 'CONFIRMED_FRAUD'
                         WHEN 'NOT_FRAUD' THEN 'FALSE_POSITIVE'
                         ELSE 'UNDER_REVIEW'
                     END
        WHERE alert_id = NEW.alert_id;
    ELSIF NEW.outcome IN ('FRAUD', 'NOT_FRAUD')
       AND NEW.outcome IS DISTINCT FROM OLD.outcome THEN
        NEW.resolved_at := COALESCE(NEW.resolved_at, CURRENT_TIMESTAMP);
        UPDATE fraud_alert
        SET status = CASE NEW.outcome
                         WHEN 'FRAUD' THEN 'CONFIRMED_FRAUD'
                         ELSE 'FALSE_POSITIVE'
                     END
        WHERE alert_id = NEW.alert_id;
    ELSIF NEW.outcome = 'PENDING' AND OLD.outcome IS DISTINCT FROM NEW.outcome THEN
        NEW.resolved_at := NULL;
        UPDATE fraud_alert
        SET status = 'UNDER_REVIEW'
        WHERE alert_id = NEW.alert_id;
    END IF;
    RETURN NEW;
END;
$function$;
-- RUN: Create investigation outcome trigger
DROP TRIGGER IF EXISTS trg_sync_investigation_resolution ON investigation;
CREATE TRIGGER trg_sync_investigation_resolution
BEFORE INSERT OR UPDATE OF outcome ON investigation
FOR EACH ROW
EXECUTE FUNCTION sync_investigation_resolution();

-- Reconcile cases created before this trigger was installed (for example by ml_pipeline.py).
UPDATE fraud_alert fa
SET status = CASE i.outcome
                 WHEN 'FRAUD' THEN 'CONFIRMED_FRAUD'
                 WHEN 'NOT_FRAUD' THEN 'FALSE_POSITIVE'
                 ELSE 'UNDER_REVIEW'
             END
FROM investigation i
WHERE i.alert_id = fa.alert_id
  AND fa.status IS DISTINCT FROM CASE i.outcome
                                    WHEN 'FRAUD' THEN 'CONFIRMED_FRAUD'
                                    WHEN 'NOT_FRAUD' THEN 'FALSE_POSITIVE'
                                    ELSE 'UNDER_REVIEW'
                                END;

-- Keep a durable event log for case creation, resolution, and reopening.
-- case_reference and alert_id remain available if one investigation row is deleted; a full loader reset removes the log table.
CREATE TABLE IF NOT EXISTS investigation_history (
    history_id BIGSERIAL PRIMARY KEY,
    case_id INTEGER REFERENCES investigation(case_id) ON DELETE SET NULL,
    case_reference INTEGER NOT NULL,
    alert_id INTEGER NOT NULL,
    event_type VARCHAR(16) NOT NULL
        CHECK (event_type IN ('CASE_CREATED', 'CASE_RESOLVED', 'CASE_REOPENED')),
    old_outcome VARCHAR(20)
        CHECK (old_outcome IS NULL OR old_outcome IN ('PENDING', 'FRAUD', 'NOT_FRAUD')),
    new_outcome VARCHAR(20) NOT NULL
        CHECK (new_outcome IN ('PENDING', 'FRAUD', 'NOT_FRAUD')),
    changed_by VARCHAR(120) NOT NULL,
    reason TEXT,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS idx_investigation_history_case_time
    ON investigation_history(case_reference, changed_at DESC, history_id DESC);

CREATE OR REPLACE FUNCTION log_investigation_history()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
DECLARE
    previous_outcome VARCHAR(20);
    actor_name TEXT;
    event_reason TEXT;
BEGIN
    IF TG_OP = 'UPDATE' AND NEW.outcome IS NOT DISTINCT FROM OLD.outcome THEN
        RETURN NEW;
    END IF;

    IF TG_OP = 'UPDATE' THEN
        previous_outcome := OLD.outcome;
    END IF;
    actor_name := COALESCE(
        NULLIF(BTRIM(current_setting('finguard.actor', true)), ''),
        session_user
    );
    event_reason := NULLIF(BTRIM(current_setting('finguard.reason', true)), '');

    INSERT INTO investigation_history (
        case_id, case_reference, alert_id, event_type,
        old_outcome, new_outcome, changed_by, reason
    ) VALUES (
        NEW.case_id,
        NEW.case_id,
        NEW.alert_id,
        CASE
            WHEN TG_OP = 'INSERT' THEN 'CASE_CREATED'
            WHEN NEW.outcome = 'PENDING' THEN 'CASE_REOPENED'
            ELSE 'CASE_RESOLVED'
        END,
        previous_outcome,
        NEW.outcome,
        actor_name,
        event_reason
    );
    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS trg_log_investigation_history ON investigation;
CREATE TRIGGER trg_log_investigation_history
AFTER INSERT OR UPDATE OF outcome ON investigation
FOR EACH ROW
EXECUTE FUNCTION log_investigation_history();

-- 6a. CTE: show the ten highest model scores.
-- RUN: Demo CTE query
WITH top_risk AS (
    SELECT transaction_id, customer_id, score_value
    FROM v_high_risk_transactions
)
SELECT transaction_id, customer_id, score_value
FROM top_risk
ORDER BY score_value DESC, transaction_id
LIMIT 10;
-- 6b. Window: rank each customer's transactions by risk score.
-- RUN: Demo window query
SELECT customer_id, transaction_id, score_value,
       RANK() OVER (PARTITION BY customer_id ORDER BY score_value DESC) AS customer_risk_rank
FROM v_high_risk_transactions
ORDER BY customer_id, customer_risk_rank
LIMIT 10;
-- 6c. Group and HAVING: customers with multiple alerts.
-- RUN: Demo GROUP BY HAVING query
SELECT customer_id, customer_name, alert_count
FROM v_customer_risk_summary_advanced
GROUP BY customer_id, customer_name, alert_count
HAVING alert_count > 1
ORDER BY alert_count DESC, customer_id
LIMIT 10;
-- 6d. Subquery: transactions above the average transaction amount.
-- RUN: Demo subquery
SELECT transaction_id, amount
FROM transaction
WHERE amount > (SELECT AVG(amount) FROM transaction)
ORDER BY amount DESC
LIMIT 10;
-- 6e. LEFT JOIN: include customers with no fraud alerts.
-- RUN: Demo LEFT JOIN query
SELECT c.customer_id, c.first_name, c.last_name, COUNT(fa.alert_id) AS alert_count
FROM customer c
LEFT JOIN account a ON a.customer_id = c.customer_id
LEFT JOIN transaction t ON t.account_id = a.account_id
LEFT JOIN fraud_alert fa ON fa.transaction_id = t.transaction_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY alert_count DESC, c.customer_id
LIMIT 10;

-- 7a. Commit example uses a temporary table so project data stays unchanged.
-- RUN: Begin commit example
BEGIN;
-- RUN: Create temporary transaction demo table
CREATE TEMP TABLE IF NOT EXISTS advanced_txn_demo (step_name TEXT PRIMARY KEY);
-- RUN: Clear temporary transaction demo table
TRUNCATE advanced_txn_demo;
-- RUN: Insert committed temporary row
INSERT INTO advanced_txn_demo VALUES ('committed');
-- RUN: Show row before commit
SELECT step_name FROM advanced_txn_demo;
-- RUN: Commit temporary transaction demo
COMMIT;
-- 7b. Rollback example removes only the temporary demonstration row.
-- RUN: Begin rollback example
BEGIN;
-- RUN: Insert row to roll back
INSERT INTO advanced_txn_demo VALUES ('rolled_back');
-- RUN: Roll back temporary transaction demo
ROLLBACK;
-- RUN: Verify rollback example
SELECT step_name FROM advanced_txn_demo ORDER BY step_name;

-- 8a. Update one investigation inside a transaction to demonstrate the trigger.
-- RUN: Begin trigger test and update outcome
BEGIN;
UPDATE investigation
SET outcome = CASE WHEN outcome = 'FRAUD' THEN 'NOT_FRAUD' ELSE 'FRAUD' END
WHERE case_id = (SELECT case_id FROM investigation ORDER BY case_id LIMIT 1);
-- 8b. Show the trigger-set timestamp and linked alert status before undoing the test.
-- RUN: Show trigger test result
SELECT i.case_id, i.outcome, i.resolved_at, fa.status AS alert_status
FROM investigation i
JOIN fraud_alert fa ON fa.alert_id = i.alert_id
WHERE i.case_id = (SELECT MIN(case_id) FROM investigation);
-- 8c. Roll back the trigger test so all project rows are restored.
-- RUN: Roll back trigger test
ROLLBACK;
-- RUN: Verify trigger test rollback
SELECT i.case_id, i.outcome, i.resolved_at, fa.status AS alert_status
FROM investigation i
JOIN fraud_alert fa ON fa.alert_id = i.alert_id
WHERE i.case_id = (SELECT MIN(case_id) FROM investigation);
