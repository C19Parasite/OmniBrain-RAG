import re
import time
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple
from ..config import settings
from ..db.database import FinancialDatabase

class SQLValidationError(Exception):
    """Raised when a generated SQL query fails read-only security and validation checks."""
    pass

class TextToSQLAgent:
    """
    Isolated Text-to-SQL Agent for Financial Analysis.
    - Generates SQL queries from natural language using LLM (Gemini / OpenAI or fallback).
    - Validates queries against strict read-only allowlists (single SELECT statement only).
    - Executes validated queries on a dedicated read-only SQLite connection.
    - Returns rows, columns, execution time, and SQL citation.
    """

    def __init__(self, db: Optional[FinancialDatabase] = None):
        self.db = db or FinancialDatabase()
        self.schema_info = self.db.get_schema_info()

    def run(self, question: str) -> Dict[str, Any]:
        """
        Main entrypoint: takes a natural language question, generates SQL,
        validates, executes on read-only connection, and returns result dictionary.
        """
        start_time = time.time()
        
        # 1. Generate SQL from question via LLM
        generated_sql = self._generate_sql(question)

        # 2. Validate SQL query (allowlist check)
        try:
            validated_sql = self.validate_sql(generated_sql)
        except SQLValidationError as ve:
            return {
                "question": question,
                "generated_sql": generated_sql,
                "executed_sql": None,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                "is_valid": False,
                "error": f"Validation Error: {str(ve)}",
                "data_source": "SYNTHETIC_DEMO"
            }

        # 3. Execute on Read-Only connection
        exec_result = self._execute_read_only(validated_sql)
        total_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "question": question,
            "generated_sql": generated_sql,
            "executed_sql": validated_sql,
            "columns": exec_result.get("columns", []),
            "rows": exec_result.get("rows", []),
            "row_count": exec_result.get("row_count", 0),
            "execution_time_ms": total_time_ms,
            "is_valid": exec_result.get("error") is None,
            "error": exec_result.get("error"),
            "data_source": "SYNTHETIC_DEMO"
        }

    def validate_sql(self, sql: str) -> str:
        """
        Validates SQL against strict safety rules:
        - Must not be empty.
        - Single statement only (no multiple statements via semicolons).
        - Must start with SELECT or WITH.
        - No destructive or unauthorized keywords (DROP, DELETE, INSERT, UPDATE, PRAGMA, ATTACH, etc.).
        - No SQL comments that could hide payload.
        """
        if not sql or not sql.strip():
            raise SQLValidationError("Query is empty.")

        clean_sql = sql.strip()

        # Remove trailing semicolon if present
        if clean_sql.endswith(";"):
            clean_sql = clean_sql[:-1].strip()

        # Check for multiple statements / semicolon chaining
        if ";" in clean_sql:
            raise SQLValidationError("Multiple SQL statements / semicolon chaining detected. Only single statements are permitted.")

        # Check for comments that could obscure commands
        if "--" in clean_sql or "/*" in clean_sql or "*/" in clean_sql:
            raise SQLValidationError("SQL comments (-- or /* */) are not permitted.")

        upper_sql = clean_sql.upper()

        # Must start with SELECT or WITH
        if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
            raise SQLValidationError(f"Invalid statement type: '{clean_sql.split()[0]}'. Only SELECT queries are permitted.")

        # Disallowed keywords list
        disallowed_keywords = [
            "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
            "TRUNCATE", "REPLACE", "PRAGMA", "ATTACH", "DETACH", "VACUUM",
            "REINDEX", "EXEC", "EXECUTE", "INTO", "GRANT", "REVOKE",
            "COMMIT", "ROLLBACK", "SAVEPOINT", "LOCK", "EXPLAIN"
        ]

        for kw in disallowed_keywords:
            pattern = r'\b' + kw + r'\b'
            if re.search(pattern, upper_sql):
                raise SQLValidationError(f"Disallowed keyword '{kw}' detected. Write, administrative, and multi-step commands are forbidden.")

        return clean_sql

    def _generate_sql(self, question: str) -> str:
        """
        Generates a SQL query for the given question using configured LLM provider,
        or deterministic semantic rule mapper if no API key is provided.
        """
        # Try LLM providers if keys configured
        if settings.GEMINI_API_KEY:
            try:
                return self._call_gemini(question)
            except Exception as e:
                print(f"[SQLAgent] Gemini LLM call failed: {e}, falling back to semantic parser.")
        elif settings.OPENAI_API_KEY:
            try:
                return self._call_openai(question)
            except Exception as e:
                print(f"[SQLAgent] OpenAI LLM call failed: {e}, falling back to semantic parser.")

        # Fallback semantic generator (also handles destructive prompt simulation for safety tests)
        return self._semantic_fallback_generator(question)

    def _call_gemini(self, question: str) -> str:
        """Call Google Gemini API to generate SQL."""
        model = getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.GEMINI_API_KEY}"
        
        system_prompt = f"""You are an expert financial Text-to-SQL translator for SQLite.
Given the following database schema, generate a single read-only SQLite SELECT query that directly answers the question.
Output ONLY the raw SQL query with no explanation, markdown code blocks, or comments.

{self.schema_info}
"""
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt},
                        {"text": f"Question: {question}\nSQL Query:"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": settings.TEMPERATURE,
                "maxOutputTokens": 400
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            parts = res_data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            raw_text = "".join([p.get("text", "") for p in parts if "text" in p]).strip()
            return self._clean_llm_sql_output(raw_text)

    def _call_openai(self, question: str) -> str:
        """Call OpenAI API to generate SQL."""
        url = "https://api.openai.com/v1/chat/completions"
        system_prompt = f"""You are an expert financial Text-to-SQL translator for SQLite.
Given the database schema below, generate a single read-only SQLite SELECT query.
Output ONLY the raw SQL query.

{self.schema_info}
"""
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": settings.TEMPERATURE
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            raw_text = res_data["choices"][0]["message"]["content"]
            return self._clean_llm_sql_output(raw_text)

    def _clean_llm_sql_output(self, text: str) -> str:
        """Cleans Markdown markdown code fences e.g. ```sql ... ```"""
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _semantic_fallback_generator(self, question: str) -> str:
        """
        High-precision financial semantic parser for offline execution and testing.
        Also maps destructive intents directly to their SQL commands so the safety validator
        can test rejection.
        """
        q = question.lower().strip()

        # Destructive prompts (to verify validator rejection)
        if "drop" in q:
            match = re.search(r'drop\s+(?:table\s+)?(\w+)', q)
            table_name = match.group(1) if match else "stock_history"
            return f"DROP TABLE {table_name}"
        if "delete from" in q or ("delete" in q and "table" in q):
            return "DELETE FROM stock_history WHERE ticker = 'NVDA'"
        if "insert into" in q:
            return "INSERT INTO companies (ticker, name, sector) VALUES ('FAKE', 'Fake Corp', 'Tech')"
        if "update" in q and "set" in q:
            return "UPDATE quarterly_financials SET revenue = 999 WHERE ticker = 'NVDA'"
        if "pragma" in q:
            return "PRAGMA table_info(companies)"





        # Detect ticker
        ticker = "NVDA"
        if "aapl" in q or "apple" in q:
            ticker = "AAPL"
        elif "msft" in q or "microsoft" in q:
            ticker = "MSFT"
        elif "tsla" in q or "tesla" in q:
            ticker = "TSLA"

        # 1. Moving average trend
        if any(term in q for term in ["moving average", "ma_50", "ma_200", "50-day", "50 day", "200-day", "trend", "price history"]):
            return f"SELECT ticker, date, close, ma_50, ma_200 FROM stock_history WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT 30;"

        # 2. Revenue last quarter or quarterly financials
        if any(term in q for term in ["revenue", "last quarter", "quarter", "net income", "gross margin", "eps", "cash flow"]):
            return f"SELECT ticker, quarter, revenue, net_income, gross_margin, eps, operating_cash_flow FROM quarterly_financials WHERE ticker = '{ticker}' ORDER BY quarter DESC LIMIT 1;"

        # 3. Analyst estimates / consensus
        if any(term in q for term in ["analyst", "consensus", "target", "rating", "wall street"]):
            return f"SELECT ticker, consensus_target, rating, num_analysts FROM analyst_estimates WHERE ticker = '{ticker}';"

        # 4. Company profile / market cap
        if any(term in q for term in ["company", "market cap", "sector", "profile"]):
            return f"SELECT ticker, name, sector, market_cap FROM companies WHERE ticker = '{ticker}';"

        # Default fallback
        return f"SELECT ticker, quarter, revenue, net_income FROM quarterly_financials WHERE ticker = '{ticker}' ORDER BY quarter DESC LIMIT 4;"


    def _execute_read_only(self, sql: str) -> Dict[str, Any]:
        """
        Executes the query using a dedicated read-only connection (mode=ro).
        """
        try:
            with self.db.get_connection(read_only=True) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                raw_rows = cursor.fetchall()
                rows = [list(row) for row in raw_rows]
                return {
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "error": None
                }
        except Exception as e:
            return {
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": f"Database Execution Error: {str(e)}"
            }
