"""Session store interface and thread-safe in-memory implementation.

The public contract is expressed as :class:`SessionStore` — a structural
Protocol that both :class:`InMemorySessionStore` and
:class:`~gauge.session.sqlite_store.SqliteSessionStore` satisfy.
Callers should annotate with ``SessionStore`` rather than a concrete class
so either implementation can be injected transparently.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from gauge.session.models import Session


@runtime_checkable
class SessionStore(Protocol):
    """Structural interface for session persistence.

    Any object that exposes ``create``, ``get``, ``update``, and ``delete``
    with these exact signatures satisfies the protocol regardless of its
    concrete type.
    """

    def create(self, session: Session) -> None:
        """Persist a new session.

        Parameters
        ----------
        session : Session
            The session to store.
        """
        ...

    def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID.

        Parameters
        ----------
        session_id : str
            The session identifier.

        Returns
        -------
        Session or None
            The stored session, or ``None`` if not found.
        """
        ...

    def update(self, session: Session) -> None:
        """Overwrite a stored session.

        Parameters
        ----------
        session : Session
            Updated session.  Must already exist.

        Raises
        ------
        KeyError
            If the session does not exist.
        """
        ...

    def delete(self, session_id: str) -> bool:
        """Remove a session.

        Parameters
        ----------
        session_id : str
            The session to remove.

        Returns
        -------
        bool
            ``True`` if the session was found and removed.
        """
        ...


class InMemorySessionStore:
    """Thread-safe in-memory store of active sessions, keyed by session ID.

    Sessions are lost when the server restarts.  That is intentional for
    this prototype -- no persistence layer is in scope.  A database-backed
    implementation would slot in cleanly behind the same public interface.
    """

    def __init__(self) -> None:
        """Initialise an empty, thread-safe session store."""
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, session: Session) -> None:
        """Insert a new session into the store.

        Parameters
        ----------
        session : Session
            The session to persist.  Any existing session with the same
            ``session_id`` is silently overwritten.
        """
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Session | None:
        """Return the session for ``session_id``, or ``None`` if absent.

        Parameters
        ----------
        session_id : str
            The identifier returned when the session was created.

        Returns
        -------
        Session or None
            The session object, or ``None`` if not found.
        """
        with self._lock:
            return self._sessions.get(session_id)

    def update(self, session: Session) -> None:
        """Overwrite a stored session with an updated copy.

        Parameters
        ----------
        session : Session
            Updated session.  Must already exist in the store.

        Raises
        ------
        KeyError
            If no session with ``session.session_id`` exists.
        """
        with self._lock:
            if session.session_id not in self._sessions:
                raise KeyError(session.session_id)
            self._sessions[session.session_id] = session

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
            return self._sessions.pop(session_id, None) is not None
