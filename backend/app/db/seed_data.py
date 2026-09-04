"""
==============================================================================
OmniBrain Synthetic Financial Data Seeder
==============================================================================
DISCLAIMER: All financial records, stock prices, quarterly figures, and analyst
targets in this file are SYNTHETIC DEMONSTRATION DATA generated solely for
functional testing and evaluation of the Text-to-SQL agent and RAG pipeline.
They do NOT represent certified corporate filings, live market feeds, or
investment advice. All rows include the data_source = 'SYNTHETIC_DEMO' marker.
==============================================================================
"""

from datetime import datetime, timedelta
import random
from typing import List, Tuple
from .database import FinancialDatabase

def seed_database(db: FinancialDatabase, force: bool = False):
    """
    Populates the database with labeled synthetic demo data for AAPL, NVDA, MSFT, TSLA.
    """
    with db.get_connection(read_only=False) as conn:
        cursor = conn.cursor()

        # Check if already seeded unless forced
        if not force:
            cursor.execute("SELECT COUNT(*) FROM companies")
            if cursor.fetchone()[0] > 0:
                return

        # 1. Seed Companies (Synthetic Demo Records)
        companies_data = [
            ("NVDA", "NVIDIA Corporation", "Semiconductors", 3250.5, "SYNTHETIC_DEMO"),
            ("AAPL", "Apple Inc.", "Consumer Electronics", 3420.0, "SYNTHETIC_DEMO"),
            ("MSFT", "Microsoft Corporation", "Software & Cloud", 3120.0, "SYNTHETIC_DEMO"),
            ("TSLA", "Tesla Inc.", "Automotive & Clean Energy", 780.0, "SYNTHETIC_DEMO"),
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO companies (ticker, name, sector, market_cap, data_source)
            VALUES (?, ?, ?, ?, ?);
        """, companies_data)

        # 2. Seed Quarterly Financials (Synthetic Demo Records)
        quarterly_data = [
            # NVDA (Revenue, Net Income, Gross Margin %, EPS, Operating Cash Flow in Billions)
            ("NVDA", "2024-Q3", 35.08, 19.31, 74.6, 0.78, 14.50, "SYNTHETIC_DEMO"),
            ("NVDA", "2024-Q2", 30.04, 16.60, 75.1, 0.67, 13.48, "SYNTHETIC_DEMO"),
            ("NVDA", "2024-Q1", 26.04, 14.88, 78.4, 0.59, 15.34, "SYNTHETIC_DEMO"),
            ("NVDA", "2023-Q4", 22.10, 12.29, 76.0, 0.49, 11.22, "SYNTHETIC_DEMO"),

            # AAPL
            ("AAPL", "2024-Q4", 94.93, 14.74, 46.2, 0.97, 26.80, "SYNTHETIC_DEMO"),
            ("AAPL", "2024-Q3", 85.78, 21.45, 46.3, 1.40, 28.90, "SYNTHETIC_DEMO"),
            ("AAPL", "2024-Q2", 90.75, 23.64, 46.6, 1.53, 22.70, "SYNTHETIC_DEMO"),
            ("AAPL", "2024-Q1", 119.58, 33.92, 45.9, 2.18, 39.90, "SYNTHETIC_DEMO"),

            # MSFT
            ("MSFT", "2024-Q3", 65.59, 24.67, 69.4, 3.30, 19.30, "SYNTHETIC_DEMO"),
            ("MSFT", "2024-Q2", 64.73, 22.04, 69.6, 2.95, 23.30, "SYNTHETIC_DEMO"),
            ("MSFT", "2024-Q1", 61.86, 21.94, 70.1, 2.94, 20.96, "SYNTHETIC_DEMO"),
            ("MSFT", "2023-Q4", 62.02, 21.87, 68.4, 2.93, 9.10, "SYNTHETIC_DEMO"),

            # TSLA
            ("TSLA", "2024-Q3", 25.18, 2.17, 19.8, 0.72, 2.74, "SYNTHETIC_DEMO"),
            ("TSLA", "2024-Q2", 25.50, 1.48, 18.0, 0.52, 1.34, "SYNTHETIC_DEMO"),
            ("TSLA", "2024-Q1", 21.30, 1.13, 17.4, 0.45, -2.53, "SYNTHETIC_DEMO"),
            ("TSLA", "2023-Q4", 25.17, 7.93, 17.6, 2.27, 2.06, "SYNTHETIC_DEMO"),
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO quarterly_financials 
            (ticker, quarter, revenue, net_income, gross_margin, eps, operating_cash_flow, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, quarterly_data)

        # 3. Seed Analyst Estimates (Synthetic Demo Records)
        estimates_data = [
            ("NVDA", 165.0, "Strong Buy", 42, "SYNTHETIC_DEMO"),
            ("AAPL", 245.0, "Buy", 38, "SYNTHETIC_DEMO"),
            ("MSFT", 480.0, "Strong Buy", 36, "SYNTHETIC_DEMO"),
            ("TSLA", 260.0, "Hold", 32, "SYNTHETIC_DEMO"),
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO analyst_estimates 
            (ticker, consensus_target, rating, num_analysts, data_source)
            VALUES (?, ?, ?, ?, ?);
        """, estimates_data)

        # 4. Seed Stock Price History (~1 Year of Daily Prices: 250 trading days)
        base_configs = {
            "NVDA": {"base_price": 140.0, "trend": 0.002, "vol": 3.5, "base_vol": 45000000},
            "AAPL": {"base_price": 225.0, "trend": 0.0008, "vol": 2.2, "base_vol": 35000000},
            "MSFT": {"base_price": 415.0, "trend": 0.0006, "vol": 2.8, "base_vol": 22000000},
            "TSLA": {"base_price": 320.0, "trend": 0.0012, "vol": 6.5, "base_vol": 70000000},
        }

        # Deterministic pseudo-random seed for repeatable synthetic data
        rng = random.Random(42)
        start_date = datetime(2024, 1, 2)
        
        stock_rows: List[Tuple] = []

        for ticker, cfg in base_configs.items():
            current_price = cfg["base_price"] * 0.7  # Start 1 year ago at a lower level
            prices_series = []

            # Generate 250 trading days
            for day_idx in range(250):
                date_val = (start_date + timedelta(days=int(day_idx * 1.45))).strftime("%Y-%m-%d")
                
                # Daily return with drift
                pct_change = cfg["trend"] + rng.gauss(0, 0.015)
                current_price = max(10.0, current_price * (1 + pct_change))
                
                open_p = round(current_price * (1 + rng.uniform(-0.008, 0.008)), 2)
                high_p = round(max(open_p, current_price) * (1 + rng.uniform(0.002, 0.015)), 2)
                low_p = round(min(open_p, current_price) * (1 - rng.uniform(0.002, 0.015)), 2)
                close_p = round(current_price, 2)
                vol = int(cfg["base_vol"] * rng.uniform(0.7, 1.4))

                prices_series.append(close_p)

                # Compute synthetic MA-50 and MA-200
                ma_50 = round(sum(prices_series[-50:]) / len(prices_series[-50:]), 2) if len(prices_series) >= 50 else round(close_p, 2)
                ma_200 = round(sum(prices_series[-200:]) / len(prices_series[-200:]), 2) if len(prices_series) >= 200 else round(close_p * 0.95, 2)

                stock_rows.append((
                    ticker, date_val, open_p, high_p, low_p, close_p, vol, ma_50, ma_200, "SYNTHETIC_DEMO"
                ))

        cursor.executemany("""
            INSERT OR REPLACE INTO stock_history 
            (ticker, date, open, high, low, close, volume, ma_50, ma_200, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, stock_rows)

        conn.commit()

if __name__ == "__main__":
    db = FinancialDatabase()
    seed_database(db, force=True)
    print("Synthetic financial database successfully seeded.")
