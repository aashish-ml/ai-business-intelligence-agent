"""
Database connection and initialization utilities.
"""

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils.config import get_settings


def get_engine() -> Engine:
    """Create and return the SQLAlchemy database engine."""

    settings = get_settings()

    if settings.database_url.startswith("sqlite:///"):
        db_path = settings.database_url.replace("sqlite:///", "", 1)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    return create_engine(
        settings.database_url,
        future=True,
    )


def _load_schema_statements(schema_path: Path) -> list[str]:
    """Load SQL schema and split it into executable statements."""

    schema_sql = schema_path.read_text(encoding="utf-8")

    # Remove SQL comment lines before splitting statements.
    cleaned_lines = [
        line
        for line in schema_sql.splitlines()
        if not line.strip().startswith("--")
    ]

    cleaned_sql = "\n".join(cleaned_lines)

    return [
        statement.strip()
        for statement in cleaned_sql.split(";")
        if statement.strip()
    ]


def initialize_database() -> None:
    """Create all database tables and indexes from schema.sql."""

    engine = get_engine()

    schema_path = Path("sql/schema.sql")

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Database schema not found: {schema_path}"
        )

    statements = _load_schema_statements(schema_path)

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully.")