# FinGuard AI ER diagram

The diagram below follows the 12 base tables and foreign keys defined in `schema.sql`. Investigation resolution time is stored in `investigation.resolved_at`; the advanced SQL installs the trigger that maintains it alongside the linked alert status.

```mermaid
erDiagram
    LOCATION {
        int location_id PK
        varchar loc_type
        varchar street
        varchar city
        char state
        varchar zip
        varchar country
        numeric latitude
        numeric longitude
        int city_pop
    }
    CUSTOMER {
        int customer_id PK
        varchar first_name
        varchar last_name
        char gender
        date dob
        varchar occupation
        numeric income
        int home_location_id FK
    }
    MERCHANT {
        int merchant_id PK
        varchar name UK
        varchar category
    }
    DEVICE {
        int device_id PK
        varchar device_type
    }
    ACCOUNT {
        int account_id PK
        int customer_id FK
        varchar account_type
        numeric balance
        date opened_date
    }
    CARD {
        int card_id PK
        int account_id FK
        bigint card_number UK
        varchar card_type
        date expiry_date
    }
    TRANSACTION {
        bigint transaction_id PK
        char trans_num UK
        int account_id FK
        int merchant_id FK
        int device_id FK
        int location_id FK
        numeric amount
        timestamp txn_timestamp
        boolean is_fraud
    }
    RISK_SCORE {
        int score_id PK
        bigint transaction_id FK, UK
        numeric score_value
        varchar model_name
        timestamp computed_at
    }
    FRAUD_ALERT {
        int alert_id PK
        bigint transaction_id FK, UK
        numeric risk_score
        varchar status
        varchar reason
        timestamp created_at
    }
    INVESTIGATION {
        int case_id PK
        int alert_id FK, UK
        timestamp opened_at
        timestamp resolved_at
        varchar outcome
    }
    LOAN {
        int loan_id PK
        int customer_id FK
        numeric principal
        numeric interest_rate
        int term_months
        date issued_date
        varchar status
    }
    LOAN_PAYMENT {
        int payment_id PK
        int loan_id FK
        numeric amount
        date due_date
        date paid_date
    }

    LOCATION ||--o{ CUSTOMER : "home location"
    CUSTOMER ||--o{ ACCOUNT : owns
    ACCOUNT ||--o{ CARD : has
    ACCOUNT ||--o{ TRANSACTION : funds
    MERCHANT ||--o{ TRANSACTION : receives
    DEVICE ||--o{ TRANSACTION : used_for
    LOCATION ||--o{ TRANSACTION : occurs_at
    TRANSACTION ||--o| RISK_SCORE : scored_by
    TRANSACTION ||--o| FRAUD_ALERT : may_raise
    FRAUD_ALERT ||--o| INVESTIGATION : may_open
    CUSTOMER ||--o{ LOAN : borrows
    LOAN ||--o{ LOAN_PAYMENT : has
```

The `transaction_id` columns in `RISK_SCORE` and `FRAUD_ALERT`, and the `alert_id` in `INVESTIGATION`, are both foreign keys and unique in the schema. This makes each relationship optional one-to-one rather than one-to-many.
