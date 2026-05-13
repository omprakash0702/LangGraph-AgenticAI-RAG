import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


DB_PATH = Path("agent_metrics.db")


@dataclass
class AgentMetric:
    agent_name: str
    model: str
    temperature: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    timestamp: str
    success: bool
    query_preview: str = ""
    error: Optional[str] = None


class MetricsDB:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            cls._instance._create_table()
        return cls._instance

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                model TEXT,
                temperature REAL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                latency_ms REAL,
                timestamp TEXT,
                success INTEGER DEFAULT 1,
                query_preview TEXT,
                error TEXT
            )
        """)
        self._conn.commit()

    def insert(self, metric: AgentMetric):
        data = asdict(metric)
        data["success"] = int(data["success"])
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        self._conn.execute(
            f"INSERT INTO agent_metrics ({cols}) VALUES ({placeholders})",
            list(data.values()),
        )
        self._conn.commit()

    def get_all(self, limit: int = 500) -> list:
        cur = self._conn.execute(
            "SELECT * FROM agent_metrics ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_summary(self) -> list:
        cur = self._conn.execute("""
            SELECT
                agent_name,
                COUNT(*) as total_calls,
                ROUND(AVG(latency_ms), 1) as avg_latency_ms,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(input_tokens), 0) as total_input,
                COALESCE(SUM(output_tokens), 0) as total_output,
                ROUND(AVG(temperature), 2) as avg_temperature,
                SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successes
            FROM agent_metrics
            GROUP BY agent_name
            ORDER BY total_calls DESC
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_latency_timeline(self) -> list:
        cur = self._conn.execute("""
            SELECT agent_name, timestamp, latency_ms, total_tokens
            FROM agent_metrics
            WHERE success = 1
            ORDER BY timestamp ASC
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def clear(self):
        self._conn.execute("DELETE FROM agent_metrics")
        self._conn.commit()
