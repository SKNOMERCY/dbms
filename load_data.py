"""
FinGuard AI - load_data.py
Creates the PostgreSQL schema and loads the Sparkov sample into it.

Real data (from Sparkov CSV): customers, locations, merchants, cards, transactions
Synthetic data (generated here, seed=42): income, accounts, card type/expiry,
                                          devices, loans, loan payments

Usage:
    pip install pandas numpy psycopg2-binary
    python load_data.py --reset
"""
import os
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

# ------------------------- CONFIG (edit these) -------------------------
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=False)
CSV_PATH = PROJECT_DIR / "sparkov_sample_10000.csv"
SCHEMA_PATH = PROJECT_DIR / "schema.sql"
SEED = 42
# ------------------------------------------------------------------------

rng = np.random.default_rng(SEED)


def insert(cur, table, cols, rows, returning=None):
    """Bulk insert rows (list of tuples)."""
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s"
    if returning:
        sql += f" RETURNING {returning}"
    execute_values(cur, sql, rows, page_size=2000)


def database_config():
    password = os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD")
    if not password:
        raise RuntimeError("Set PGPASSWORD in the environment or in a local .env file before loading data.")
    return {
        "host": os.getenv("PGHOST", "localhost"),
        "port": int(os.getenv("PGPORT", 5432)),
        "dbname": os.getenv("PGDATABASE", "finguard"),
        "user": os.getenv("PGUSER", "postgres"),
        "password": password,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create/reset the FinGuard schema and load the sample dataset.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Explicitly allow dropping and recreating all FinGuard tables and views.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive RESET confirmation (for deliberate automation only).",
    )
    args = parser.parse_args(argv)
    if not args.reset:
        parser.error("This loader resets the project schema. Pass --reset to proceed.")
    if not args.yes:
        if not sys.stdin.isatty():
            parser.error("A non-interactive reset also requires --yes.")
        confirmation = input("This drops and recreates FinGuard tables. Type RESET to continue: ")
        if confirmation != "RESET":
            print("Cancelled; database was not changed.")
            return 1

    # ---------------------------- 1. read + clean ----------------------------
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]          # drop index column if present
    df["merchant"] = df["merchant"].str.replace("^fraud_", "", regex=True)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.drop_duplicates(subset="trans_num")
    df = df[df["amt"] > 0].reset_index(drop=True)
    print(f"Loaded {len(df):,} transactions, {df['is_fraud'].sum()} fraud")

    conn = psycopg2.connect(**database_config())
    try:
        with conn:
            with conn.cursor() as cur:

                # ---------------------------- 2. create schema ---------------------------
                with open(SCHEMA_PATH, encoding="utf-8") as f:
                    cur.execute(f.read())
                print("Schema created (12 tables, 3 views)")

                # ---------------------------- 3. customers + home locations --------------
                cust = (df.drop_duplicates("cc_num")
                          [["cc_num", "first", "last", "gender", "dob", "job",
                            "street", "city", "state", "zip", "lat", "long", "city_pop"]]
                          .reset_index(drop=True))

                home_rows = [("HOME", r.street, r.city, r.state, str(r.zip).zfill(5), r.lat, r.long, int(r.city_pop))
                             for r in cust.itertuples()]
                home_ids = [x[0] for x in insert_returning(cur, "location",
                            ["loc_type", "street", "city", "state", "zip", "latitude", "longitude", "city_pop"],
                            home_rows, "location_id")]

                incomes = np.round(np.clip(rng.lognormal(mean=10.9, sigma=0.45, size=len(cust)), 18000, 250000), 2)
                cust_rows = [(r.first, r.last, r.gender, r.dob, r.job, float(incomes[i]), home_ids[i])
                             for i, r in enumerate(cust.itertuples())]
                cust_ids = [x[0] for x in insert_returning(cur, "customer",
                            ["first_name", "last_name", "gender", "dob", "occupation", "income", "home_location_id"],
                            cust_rows, "customer_id")]
                cc_to_customer = dict(zip(cust["cc_num"], cust_ids))
                print(f"customers: {len(cust_ids)}")

                # ---------------------------- 4. accounts + cards (synthetic) ------------
                acct_types = rng.choice(["SAVINGS", "CURRENT", "CREDIT"], size=len(cust), p=[0.35, 0.45, 0.20])
                acct_rows = [(cid, acct_types[i], float(round(rng.uniform(500, 60000), 2)),
                              date(2015, 1, 1) + timedelta(days=int(rng.integers(0, 2000))))
                             for i, cid in enumerate(cust_ids)]
                acct_ids = [x[0] for x in insert_returning(cur, "account",
                            ["customer_id", "account_type", "balance", "opened_date"], acct_rows, "account_id")]
                cc_to_account = dict(zip(cust["cc_num"], acct_ids))

                card_rows = []
                for i, r in enumerate(cust.itertuples()):
                    ctype = "CREDIT" if acct_types[i] == "CREDIT" else "DEBIT"
                    card_rows.append((acct_ids[i], int(r.cc_num), ctype,
                                      date(2026, 1, 1) + timedelta(days=int(rng.integers(0, 1800)))))
                insert(cur, "card", ["account_id", "card_number", "card_type", "expiry_date"], card_rows)
                print(f"accounts: {len(acct_ids)}, cards: {len(card_rows)}")

                # ---------------------------- 5. merchants -------------------------------
                merch = df.drop_duplicates("merchant")[["merchant", "category"]]
                merch_ids = [x[0] for x in insert_returning(cur, "merchant", ["name", "category"],
                             list(merch.itertuples(index=False, name=None)), "merchant_id")]
                merchant_to_id = dict(zip(merch["merchant"], merch_ids))
                print(f"merchants: {len(merch_ids)}")

                # ---------------------------- 6. devices (synthetic) ---------------------
                dev_types = ["MOBILE", "DESKTOP", "TABLET", "POS_TERMINAL", "ATM"]
                dev_rows = [(rng.choice(dev_types, p=[0.30, 0.15, 0.05, 0.40, 0.10]),) for _ in range(300)]
                dev_ids = [x[0] for x in insert_returning(cur, "device", ["device_type"], dev_rows, "device_id")]

                # ---------------------------- 7. merchant-side locations -----------------
                mloc = df[["merch_lat", "merch_long"]].drop_duplicates().reset_index(drop=True)
                mloc_ids = [x[0] for x in insert_returning(cur, "location",
                            ["loc_type", "latitude", "longitude"],
                            [("MERCHANT", r.merch_lat, r.merch_long) for r in mloc.itertuples()], "location_id")]
                coord_to_loc = {(r.merch_lat, r.merch_long): mloc_ids[i] for i, r in enumerate(mloc.itertuples())}
                print(f"merchant locations: {len(mloc_ids)}")

                # ---------------------------- 8. transactions ----------------------------
                dev_choice = rng.choice(dev_ids, size=len(df))
                txn_rows = [(r.trans_num, cc_to_account[r.cc_num], merchant_to_id[r.merchant],
                             int(dev_choice[i]), coord_to_loc[(r.merch_lat, r.merch_long)],
                             float(r.amt), r.trans_date_trans_time.to_pydatetime(), bool(r.is_fraud))
                            for i, r in enumerate(df.itertuples())]
                insert(cur, "transaction",
                       ["trans_num", "account_id", "merchant_id", "device_id", "location_id",
                        "amount", "txn_timestamp", "is_fraud"], txn_rows)
                print(f"transactions: {len(txn_rows):,}")

                # ---------------------------- 9. loans + payments (synthetic) ------------
                as_of = date(2020, 12, 31)
                borrowers = rng.choice(cust_ids, size=int(len(cust_ids) * 0.40), replace=False)
                loan_rows = []
                for cid in borrowers:
                    for _ in range(int(rng.choice([1, 2], p=[0.8, 0.2]))):
                        term = int(rng.choice([6, 12, 18, 24]))
                        issued = date(2019, 6, 1) + timedelta(days=int(rng.integers(0, 540)))
                        loan_rows.append((int(cid), float(round(rng.uniform(5000, 80000), -2)),
                                          float(round(rng.uniform(7, 18), 2)), term, issued))
                loan_ids = [x[0] for x in insert_returning(cur, "loan",
                            ["customer_id", "principal", "interest_rate", "term_months", "issued_date", "status"],
                            [l + ("ACTIVE",) for l in loan_rows], "loan_id")]

                pay_rows, status_by_loan = [], {}
                for lid, (cid, principal, rate, term, issued) in zip(loan_ids, loan_rows):
                    emi = round(principal * (1 + rate / 100 * term / 12) / term, 2)
                    risky = rng.random() < 0.12                                  # ~12% poor payers
                    missed = 0
                    for k in range(1, term + 1):
                        due = issued + timedelta(days=30 * k)
                        if due > as_of:
                            break
                        p_late = 0.45 if risky else 0.08
                        p_miss = 0.25 if risky else 0.01
                        roll = rng.random()
                        if roll < p_miss:
                            paid, missed = None, missed + 1
                        elif roll < p_miss + p_late:
                            paid = due + timedelta(days=int(rng.integers(3, 25)))
                        else:
                            paid = due - timedelta(days=int(rng.integers(0, 4)))
                        if paid and paid > as_of:
                            paid = None
                        pay_rows.append((lid, emi, due, paid))
                    last_due = issued + timedelta(days=30 * term)
                    status_by_loan[lid] = ("DEFAULTED" if missed >= 3
                                           else "CLOSED" if last_due <= as_of else "ACTIVE")
                insert(cur, "loan_payment", ["loan_id", "amount", "due_date", "paid_date"], pay_rows)
                execute_values(cur, "UPDATE loan l SET status = v.s FROM (VALUES %s) AS v(id, s) WHERE l.loan_id = v.id",
                               list(status_by_loan.items()))
                print(f"loans: {len(loan_ids)}, loan payments: {len(pay_rows):,}")

        print("\nDone. Database is ready.")
    finally:
        conn.close()


def insert_returning(cur, table, cols, rows, returning):
    """Insert and return the generated ids IN INPUT ORDER."""
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s RETURNING {returning}"
    return execute_values(cur, sql, rows, page_size=2000, fetch=True)


if __name__ == "__main__":
    raise SystemExit(main())
