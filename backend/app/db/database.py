import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..config import settings

class FinancialDatabase:
    """
    SQLite Connection Manager and Schema Manager for OmniBrain financial tables.
    Provides read-write connection for initialization/seeding and dedicated read-only
    connections for SQL query execution.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or settings.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self, read_only: bool = False) -> sqlite3.Connection:
        """
        Returns a SQLite connection. If read_only is True, opens the connection
        using SQLite's URI read-only mode (mode=ro).
        """
        if read_only:
            uri_path = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
            conn = sqlite3.connect(uri_path, uri=True)
        else:
            conn = sqlite3.connect(str(self.db_path))

        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self):
        """Creates the core structured financial database tables if they do not exist."""
        with self.get_connection(read_only=False) as conn:
            cursor = conn.cursor()

            # 1. Companies Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                ticker TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                sector TEXT NOT NULL,
                market_cap REAL,
                data_source TEXT DEFAULT 'SYNTHETIC_DEMO'
            );
            """)

            # 2. Stock Price History Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                ma_50 REAL,
                ma_200 REAL,
                data_source TEXT DEFAULT 'SYNTHETIC_DEMO',
                FOREIGN KEY (ticker) REFERENCES companies(ticker)
            );
            """)

            # 3. Quarterly Financials Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS quarterly_financials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                quarter TEXT NOT NULL,
                revenue REAL NOT NULL,
                net_income REAL NOT NULL,
                gross_margin REAL,
                eps REAL,
                operating_cash_flow REAL,
                data_source TEXT DEFAULT 'SYNTHETIC_DEMO',
                FOREIGN KEY (ticker) REFERENCES companies(ticker)
            );
            """)

            # 4. Analyst Estimates Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyst_estimates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                consensus_target REAL NOT NULL,
                rating TEXT NOT NULL,
                num_analysts INTEGER NOT NULL,
                data_source TEXT DEFAULT 'SYNTHETIC_DEMO',
                FOREIGN KEY (ticker) REFERENCES companies(ticker)
            );
            """)

            conn.commit()

    def get_schema_info(self) -> str:
        """
        Returns a formatted schema description used to ground the Text-to-SQL LLM prompt.
        """
        return """
Database Schema (SQLite - READ-ONLY):

Table: companies
- ticker (TEXT, PRIMARY KEY, e.g. 'NVDA', 'AAPL', 'MSFT', 'TSLA')
- name (TEXT, e.g. 'NVIDIA Corporation', 'Apple Inc.')
- sector (TEXT, e.g. 'Semiconductors', 'Consumer Electronics')
- market_cap (REAL, Market capitalization in billions USD)
- data_source (TEXT, e.g. 'SYNTHETIC_DEMO')

Table: stock_history
- ticker (TEXT, Foreign Key -> companies.ticker)
- date (TEXT, YYYY-MM-DD format)
- open (REAL, Opening price USD)
- high (REAL, High price USD)
- low (REAL, Low price USD)
- close (REAL, Closing price USD)
- volume (INTEGER, Daily trading volume)
- ma_50 (REAL, 50-day simple moving average price)
- ma_200 (REAL, 200-day simple moving average price)
- data_source (TEXT, e.g. 'SYNTHETIC_DEMO')

Table: quarterly_financials
- ticker (TEXT, Foreign Key -> companies.ticker)
- quarter (TEXT, format 'YYYY-QX', e.g. '2024-Q3', '2024-Q2')
- revenue (REAL, Revenue in billions USD)
- net_income (REAL, Net income in billions USD)
- gross_margin (REAL, Gross margin percentage, e.g. 74.6)
- eps (REAL, Diluted earnings per share USD)
- operating_cash_flow (REAL, Operating cash flow in billions USD)
- data_source (TEXT, e.g. 'SYNTHETIC_DEMO')

Table: analyst_estimates
- ticker (TEXT, Foreign Key -> companies.ticker)
- consensus_target (REAL, Wall Street mean target price USD)
- rating (TEXT, e.g. 'Strong Buy', 'Buy', 'Hold', 'Neutral')
- num_analysts (INTEGER, Number of covering Wall Street analysts)
- data_source (TEXT, e.g. 'SYNTHETIC_DEMO')
""".strip()

    def get_tables_metadata(self) -> List[Dict[str, Any]]:
        """Returns structured metadata for all tables for the SQL Sandbox & Schema Explorer."""
        tables = ["companies", "stock_history", "quarterly_financials", "analyst_estimates"]
        metadata_list = []

        with self.get_connection(read_only=True) as conn:
            cursor = conn.cursor()
            for t in tables:
                cursor.execute(f"PRAGMA table_info({t})")
                cols = [
                    {
                        "name": row["name"],
                        "type": row["type"],
                        "is_primary_key": bool(row["pk"])
                    }
                    for row in cursor.fetchall()
                ]

                cursor.execute(f"SELECT COUNT(*) FROM {t}")
                count = cursor.fetchone()[0]

                cursor.execute(f"SELECT * FROM {t} LIMIT 3")
                sample_rows = [dict(r) for r in cursor.fetchall()]

                metadata_list.append({
                    "table_name": t,
                    "columns": cols,
                    "row_count": count,
                    "sample_rows": sample_rows
                })

        return metadata_list

    def get_stats(self) -> Dict[str, int]:
        """Returns total record count per table."""
        with self.get_connection(read_only=True) as conn:
            cursor = conn.cursor()
            stats = {}
            for t in ["companies", "stock_history", "quarterly_financials", "analyst_estimates"]:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {t}")
                    stats[t] = cursor.fetchone()[0]
                except Exception:
                    stats[t] = 0
            return stats
