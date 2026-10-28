"""Simple SQLite persistence for the Autonomous Research Assistant.

Uses Python's standard library `sqlite3` module to record completed research tasks.
This keeps persistence lightweight, beginner-friendly, and completely dependency-free.
"""

import json
from pathlib import Path
import sqlite3
from typing import Any

from config import DATABASE_PATH


def get_connection(db_path: str = DATABASE_PATH) -> sqlite3.Connection:
    """Creates a connection to the SQLite database and ensures the parent folder exists."""
    if db_path != ":memory:":
        Path(db_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enables column access by name
    return conn


def init_db(db_path: str = DATABASE_PATH) -> None:
    """Initializes the SQLite database schema if the table does not yet exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS research_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                search_queries TEXT NOT NULL,
                final_report TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_research(
    question: str,
    search_queries: list[str],
    final_report: str,
    db_path: str = DATABASE_PATH,
) -> int:
    """Saves a completed research execution into the research_history table.

    Returns the inserted record's auto-generated integer ID.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO research_history (question, search_queries, final_report)
            VALUES (?, ?, ?)
            """,
            (
                question,
                json.dumps(search_queries),
                final_report,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_research_history(
    limit: int = 10,
    db_path: str = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Retrieves the most recent research records, deserializing search queries."""
    init_db(db_path)
    # Bound limit to a reasonable maximum (e.g., between 1 and 100)
    safe_limit = max(1, min(limit, 100))

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, question, search_queries, final_report, created_at
            FROM research_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        )
        rows = cursor.fetchall()

    history: list[dict[str, Any]] = []
    for row in rows:
        try:
            queries = json.loads(row["search_queries"])
        except Exception:
            queries = []

        history.append({
            "id": row["id"],
            "question": row["question"],
            "search_queries": queries,
            "final_report": row["final_report"],
            "created_at": row["created_at"],
        })
    return history

