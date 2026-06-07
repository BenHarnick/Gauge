"""SQLite-backed session store.

Provides the same public interface as :class:`InMemorySessionStore` but
persists sessions to a SQLite database so they survive server restarts.

Sessions are stored as JSON blobs (one row per session).  All Pydantic
models in the session graph serialise cleanly via ``model_dump_json`` /
``model_validate_json``.

Thread safety is achieved with a single ``threading.Lock`` combined with
``check_same_thread=False`` on the SQLite connection and WAL journal mode,
which gives safe concurrent reads and serialised writes.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from gauge.session.models import Session


class SqliteSessionStore:
    """SQLite-backed store of active sessions, keyed by session ID.

    Parameters
    ----------
    db_path : Path or str
        Path to the SQLite database file.  Created (along with any missing
        parent directories) if it does not already exist.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            data       TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """

    def __init__(self, db_path: Path | str) -> None:
        """Open (or create) the database and ensure the sessions table exists.

        Parameters
        ----------
        db_path : Path or str
            Path to the SQLite ``.db`` file.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(self._CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public interface (mirrors InMemorySessionStore)
    # ------------------------------------------------------------------

    def create(self, session: Session) -> None:
        """Insert a new session into the store.

        Parameters
        ----------
        session : Session
            The session to persist.  Any existing session with the same
            ``session_id`` is silently overwritten.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, data, created_at)"
                " VALUES (?, ?, ?)",
                (
                    session.session_id,
                    session.model_dump_json(),
                    session.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def get(self, session_id: str) -> Session | None:
        """Return the session for ``session_id``, or ``None`` if absent.

        Parameters
        ----------
        session_id : str
            The identifier returned when the session was created.

        Returns
        -------
        Session or None
            The deserialised session, or ``None`` if not found.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return Session.model_validate_json(row[0])

    def update(self, session: Session) -> None:
        """Overwrite a stored session with an updated copy.

        Parameters
        ----------
        session : Session
            Updated session.  Must already exist in the store.

        Raises
        ------
        KeyError
            If no session with ``session.session_id`` exists in the
            database.
        """
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE sessions SET data = ? WHERE session_id = ?",
                (session.model_dump_json(), session.session_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(session.session_id)
            self._conn.commit()

    def delete(self, session_id: str) -> bool:
        """Remove a session from the store.

        Parameters
        ----------
        session_id : str
            The session to remove.

        Returns
        -------
        bool
            ``True`` if the session existed and was removed, ``False`` if
            it was not found.
        """
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def close(self) -> None:
        """Close the underlying database connection.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self._lock:
            self._conn.close()
