"""
Safe read-only SQL execution tool.

The SQL tool is intentionally restricted to SELECT-style queries.
Destructive SQL operations are blocked before execution.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text

from src.data.database import get_engine
from src.utils.config import get_settings


BLOCKED_SQL_KEYWORDS = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
}


class SQLSafetyError(ValueError):
    """Raised when SQL violates safety rules."""


def validate_read_only_sql(query: str) -> str:
    """
    Validate that a SQL query is read-only.

    Returns the normalized query if valid.
    """

    if not query or not query.strip():
        raise SQLSafetyError("SQL query cannot be empty.")

    normalized = query.strip()

    if normalized.endswith(";"):
        normalized = normalized[:-1].strip()

    # Only one statement is allowed.
    if ";" in normalized:
        raise SQLSafetyError(
            "Multiple SQL statements are not allowed."
        )

    # Remove SQL comments for keyword inspection.
    without_comments = re.sub(
        r"--.*?$",
        "",
        normalized,
        flags=re.MULTILINE,
    )

    without_comments = re.sub(
        r"/\*.*?\*/",
        "",
        without_comments,
        flags=re.DOTALL,
    )

    tokens = set(
        re.findall(
            r"\b[A-Za-z_]+\b",
            without_comments.upper(),
        )
    )

    blocked = tokens.intersection(BLOCKED_SQL_KEYWORDS)

    if blocked:
        raise SQLSafetyError(
            "Blocked SQL operation detected: "
            + ", ".join(sorted(blocked))
        )

    first_token_match = re.search(
        r"^\s*(SELECT|WITH)\b",
        without_comments,
        flags=re.IGNORECASE,
    )

    if not first_token_match:
        raise SQLSafetyError(
            "Only SELECT or WITH queries are allowed."
        )

    return normalized


def execute_sql_query(
    query: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute a validated read-only SQL query.

    Returns structured results suitable for the agent.
    """

    safe_query = validate_read_only_sql(query)

    settings = get_settings()

    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            text(safe_query),
            parameters or {},
        )

        rows = result.mappings().fetchmany(
            settings.max_sql_rows
        )

        return {
            "success": True,
            "columns": list(result.keys()),
            "rows": [dict(row) for row in rows],
            "row_count": len(rows),
            "query": safe_query,
        }