-- =====================================================================
-- FinGuard AI - demo queries (run these in pgAdmin / VS Code during review)
-- =====================================================================

-- 1. Row counts of every table (shows the database is populated)
SELECT 'customer' AS tbl, COUNT(*) FROM customer UNION ALL
SELECT 'account', COUNT(*) FROM account UNION ALL
SELECT 'card', COUNT(*) FROM card UNION ALL
SELECT 'merchant', COUNT(*) FROM merchant UNION ALL
SELECT 'device', COUNT(*) FROM device UNION ALL
SELECT 'location', COUNT(*) FROM location UNION ALL
SELECT 'transaction', COUNT(*) FROM transaction UNION ALL
SELECT 'loan', COUNT(*) FROM loan UNION ALL
SELECT 'loan_payment', COUNT(*) FROM loan_payment UNION ALL
SELECT 'risk_score', COUNT(*) FROM risk_score UNION ALL
SELECT 'fraud_alert', COUNT(*) FROM fraud_alert UNION ALL
SELECT 'investigation', COUNT(*) FROM investigation;

-- 2. JOIN across 4 tables: full detail of the 10 largest fraud transactions
SELECT t.transaction_id, c.first_name, c.last_name, m.name AS merchant,
       m.category, t.amount, t.txn_timestamp
FROM transaction t
JOIN account  a ON a.account_id  = t.account_id
JOIN customer c ON c.customer_id = a.customer_id
JOIN merchant m ON m.merchant_id = t.merchant_id
WHERE t.is_fraud
ORDER BY t.amount DESC
LIMIT 10;

-- 3. GROUP BY via view: fraud rate per merchant category
SELECT * FROM v_fraud_by_category ORDER BY fraud_pct DESC;

-- 4. Fraud by hour of day (fraud tends to cluster at night)
SELECT EXTRACT(HOUR FROM txn_timestamp)::INT AS hour_of_day,
       COUNT(*) AS txns,
       SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_txns
FROM transaction
GROUP BY 1
ORDER BY 1;

-- 5. Subquery: customers whose average spend is above the overall average
SELECT customer_name, txn_count, total_spent
FROM v_customer_risk_summary
WHERE total_spent / NULLIF(txn_count, 0) >
      (SELECT AVG(amount) FROM transaction)
ORDER BY total_spent DESC
LIMIT 10;

-- 6. Window function: rank each customer's transactions by amount
SELECT customer_id, transaction_id, amount,
       RANK() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS rank_in_customer
FROM (SELECT a.customer_id, t.transaction_id, t.amount
      FROM transaction t JOIN account a ON a.account_id = t.account_id) x
ORDER BY customer_id, rank_in_customer
LIMIT 15;

-- 7. Loan risk: customers with late or missed repayments
SELECT c.customer_id, c.first_name, c.last_name, l.loan_id, l.status,
       COUNT(*) FILTER (WHERE p.paid_date IS NULL)          AS missed,
       COUNT(*) FILTER (WHERE p.paid_date > p.due_date)     AS late
FROM loan l
JOIN customer c      ON c.customer_id = l.customer_id
JOIN loan_payment p  ON p.loan_id = l.loan_id
GROUP BY c.customer_id, c.first_name, c.last_name, l.loan_id, l.status
HAVING COUNT(*) FILTER (WHERE p.paid_date IS NULL) > 0
ORDER BY missed DESC
LIMIT 10;

-- 8. Integrity demo: each of these must FAIL (shows constraints working)
--    Run one at a time.
-- INSERT INTO transaction (trans_num, account_id, merchant_id, device_id, location_id, amount, txn_timestamp, is_fraud)
-- VALUES (repeat('x',32), 1, 1, 1, 1, -50, NOW(), FALSE);                       -- CHECK: amount > 0
-- INSERT INTO account (customer_id, account_type, balance, opened_date)
-- VALUES (999999, 'SAVINGS', 100, CURRENT_DATE);                                -- FOREIGN KEY violation
-- INSERT INTO fraud_alert (transaction_id, risk_score, status) VALUES (1, 90, 'OPEN');
-- INSERT INTO fraud_alert (transaction_id, risk_score, status) VALUES (1, 95, 'OPEN');  -- UNIQUE: one alert per txn

-- 9. ACID demo: a transfer that either fully happens or not at all
BEGIN;
UPDATE account SET balance = balance - 500 WHERE account_id = 1;
UPDATE account SET balance = balance + 500 WHERE account_id = 2;
SELECT account_id, balance FROM account WHERE account_id IN (1, 2);
ROLLBACK;   -- change to COMMIT to make it permanent
