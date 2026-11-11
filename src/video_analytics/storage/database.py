import sqlite3
from pathlib import Path


EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    object_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    direction TEXT,
    metadata TEXT
);
"""


class Database:
    """
    Manage the SQLite database connection and schema initialization.

    This class is responsible only for database setup and connections.
    Event-specific persistence logic belongs in the repository layer.
    """

    def __init__(self, database_path):
        self.database_path = Path(database_path)

    def connect(self) -> sqlite3.Connection:
        """
        Open and return a SQLite connection.
        """

        return sqlite3.connect(self.database_path)

    def initialize(self) -> None:
        """
        Create the database parent directory if necessary and ensure
        that the required database schema exists.
        """

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.connect() as connection:
            connection.execute(EVENTS_TABLE_SQL)
            connection.commit()