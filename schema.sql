-- =====================================================================
-- FinGuard AI  |  DBMS Project - Review 1  |  PostgreSQL schema
-- 12 tables in 3NF, with entity / referential / domain integrity
-- Run order: this file is executed by load_data.py. It drops and recreates
-- all project tables and views; running the loader resets the project data.
-- =====================================================================

DROP VIEW  IF EXISTS v_fraud_by_category, v_customer_risk_summary, v_open_alerts CASCADE;
DROP TABLE IF EXISTS loan_payment, loan, investigation, risk_score, fraud_alert,
                     transaction, device, location, merchant, card, account, customer CASCADE;

-- ---------------------------------------------------------------------
-- Reference / parent tables
-- ---------------------------------------------------------------------
CREATE TABLE location (
    location_id  SERIAL PRIMARY KEY,
    loc_type     VARCHAR(10)  NOT NULL CHECK (loc_type IN ('HOME', 'MERCHANT')),
    street       VARCHAR(120),
    city         VARCHAR(80),
    state        CHAR(2),
    zip          VARCHAR(10),
    country      VARCHAR(40)  NOT NULL DEFAULT 'USA',
    latitude     NUMERIC(9,6) NOT NULL CHECK (latitude  BETWEEN  -90 AND  90),
    longitude    NUMERIC(9,6) NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    city_pop     INTEGER      CHECK (city_pop >= 0)
);

CREATE TABLE customer (
    customer_id      SERIAL PRIMARY KEY,
    first_name       VARCHAR(50) NOT NULL,
    last_name        VARCHAR(50) NOT NULL,
    gender           CHAR(1)     NOT NULL CHECK (gender IN ('M', 'F')),
    dob              DATE        NOT NULL,
    occupation       VARCHAR(100),
    income           NUMERIC(12,2) CHECK (income >= 0),       -- synthetic
    home_location_id INTEGER     NOT NULL REFERENCES location(location_id)
);

CREATE TABLE merchant (
    merchant_id  SERIAL PRIMARY KEY,
    name         VARCHAR(120) NOT NULL UNIQUE,
    category     VARCHAR(40)  NOT NULL
);

CREATE TABLE device (                                        -- synthetic
    device_id    SERIAL PRIMARY KEY,
    device_type  VARCHAR(20) NOT NULL
                 CHECK (device_type IN ('MOBILE', 'DESKTOP', 'TABLET', 'POS_TERMINAL', 'ATM'))
);

-- ---------------------------------------------------------------------
-- Customer-owned tables
-- ---------------------------------------------------------------------
CREATE TABLE account (
    account_id    SERIAL PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customer(customer_id) ON DELETE CASCADE,
    account_type  VARCHAR(10) NOT NULL CHECK (account_type IN ('SAVINGS', 'CURRENT', 'CREDIT')),
    balance       NUMERIC(14,2) NOT NULL DEFAULT 0,
    opened_date   DATE NOT NULL
);

CREATE TABLE card (
    card_id      SERIAL PRIMARY KEY,
    account_id   INTEGER NOT NULL REFERENCES account(account_id) ON DELETE CASCADE,
    card_number  BIGINT  NOT NULL UNIQUE,                    -- cc_num from dataset
    card_type    VARCHAR(10) NOT NULL CHECK (card_type IN ('DEBIT', 'CREDIT')),
    expiry_date  DATE NOT NULL
);

-- ---------------------------------------------------------------------
-- Transactions
-- ---------------------------------------------------------------------
CREATE TABLE transaction (
    transaction_id  BIGSERIAL PRIMARY KEY,
    trans_num       CHAR(32)      NOT NULL UNIQUE,           -- id from dataset
    account_id      INTEGER       NOT NULL REFERENCES account(account_id),
    merchant_id     INTEGER       NOT NULL REFERENCES merchant(merchant_id),
    device_id       INTEGER       NOT NULL REFERENCES device(device_id),
    location_id     INTEGER       NOT NULL REFERENCES location(location_id),
    amount          NUMERIC(12,2) NOT NULL CHECK (amount > 0),
    txn_timestamp   TIMESTAMP     NOT NULL,
    is_fraud        BOOLEAN       NOT NULL                   -- ground-truth label for ML
);

CREATE TABLE risk_score (                                    -- 1 : 0/1 with transaction
    score_id        SERIAL PRIMARY KEY,
    transaction_id  BIGINT NOT NULL UNIQUE REFERENCES transaction(transaction_id) ON DELETE CASCADE,
    score_value     NUMERIC(5,2) NOT NULL CHECK (score_value BETWEEN 0 AND 100),
    model_name      VARCHAR(40),
    computed_at     TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE fraud_alert (                                   -- 1 : 0/1 with transaction
    alert_id        SERIAL PRIMARY KEY,
    transaction_id  BIGINT NOT NULL UNIQUE REFERENCES transaction(transaction_id) ON DELETE CASCADE,
    risk_score      NUMERIC(5,2) NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    status          VARCHAR(20) NOT NULL DEFAULT 'OPEN'
                    CHECK (status IN ('OPEN', 'UNDER_REVIEW', 'CONFIRMED_FRAUD', 'FALSE_POSITIVE')),
    reason          VARCHAR(200),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE investigation (                                 -- 1 : 0/1 with fraud_alert
    case_id      SERIAL PRIMARY KEY,
    alert_id     INTEGER NOT NULL UNIQUE REFERENCES fraud_alert(alert_id) ON DELETE CASCADE,
    opened_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    resolved_at  TIMESTAMP,
    outcome      VARCHAR(20) NOT NULL DEFAULT 'PENDING'
                 CHECK (outcome IN ('PENDING', 'FRAUD', 'NOT_FRAUD')),
    CHECK (resolved_at IS NULL OR resolved_at >= opened_at),
    CHECK ((outcome = 'PENDING' AND resolved_at IS NULL)
        OR (outcome IN ('FRAUD', 'NOT_FRAUD') AND resolved_at IS NOT NULL))
);

-- ---------------------------------------------------------------------
-- Loans (synthetic)
-- ---------------------------------------------------------------------
CREATE TABLE loan (
    loan_id        SERIAL PRIMARY KEY,
    customer_id    INTEGER NOT NULL REFERENCES customer(customer_id) ON DELETE CASCADE,
    principal      NUMERIC(12,2) NOT NULL CHECK (principal > 0),
    interest_rate  NUMERIC(5,2)  NOT NULL CHECK (interest_rate >= 0),
    term_months    INTEGER       NOT NULL CHECK (term_months > 0),
    issued_date    DATE          NOT NULL,
    status         VARCHAR(10)   NOT NULL CHECK (status IN ('ACTIVE', 'CLOSED', 'DEFAULTED'))
);

CREATE TABLE loan_payment (
    payment_id  SERIAL PRIMARY KEY,
    loan_id     INTEGER NOT NULL REFERENCES loan(loan_id) ON DELETE CASCADE,
    amount      NUMERIC(12,2) NOT NULL CHECK (amount > 0),
    due_date    DATE NOT NULL,
    paid_date   DATE,                                        -- NULL = not paid yet
    UNIQUE (loan_id, due_date)
);

-- ---------------------------------------------------------------------
-- Indexes (speed up the joins and filters used by analytics)
-- ---------------------------------------------------------------------
CREATE INDEX idx_txn_account    ON transaction(account_id);
CREATE INDEX idx_txn_merchant   ON transaction(merchant_id);
CREATE INDEX idx_txn_timestamp  ON transaction(txn_timestamp);
CREATE INDEX idx_txn_fraud      ON transaction(is_fraud);
CREATE INDEX idx_account_cust   ON account(customer_id);
CREATE INDEX idx_loan_cust      ON loan(customer_id);
CREATE INDEX idx_payment_loan   ON loan_payment(loan_id);
CREATE INDEX idx_alert_status   ON fraud_alert(status);

-- ---------------------------------------------------------------------
-- Views (used by the dashboard / reporting module)
-- ---------------------------------------------------------------------
CREATE VIEW v_fraud_by_category AS
SELECT m.category,
       COUNT(*)                                        AS total_txns,
       SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)     AS fraud_txns,
       ROUND(100.0 * SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_pct,
       ROUND(AVG(t.amount), 2)                         AS avg_amount
FROM transaction t
JOIN merchant m ON m.merchant_id = t.merchant_id
GROUP BY m.category;

CREATE VIEW v_customer_risk_summary AS
SELECT c.customer_id,
       c.first_name || ' ' || c.last_name              AS customer_name,
       COUNT(t.transaction_id)                         AS txn_count,
       ROUND(SUM(t.amount), 2)                         AS total_spent,
       SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)     AS fraud_txns
FROM customer c
JOIN account a     ON a.customer_id = c.customer_id
LEFT JOIN transaction t ON t.account_id = a.account_id
GROUP BY c.customer_id, c.first_name, c.last_name;

CREATE VIEW v_open_alerts AS
SELECT fa.alert_id, fa.transaction_id, fa.risk_score, fa.status, fa.reason,
       t.amount, t.txn_timestamp, m.name AS merchant
FROM fraud_alert fa
JOIN transaction t ON t.transaction_id = fa.transaction_id
JOIN merchant m    ON m.merchant_id = t.merchant_id
WHERE fa.status IN ('OPEN', 'UNDER_REVIEW');
