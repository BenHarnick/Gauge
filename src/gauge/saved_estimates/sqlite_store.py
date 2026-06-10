"""SQLite-backed saved-estimate store.

Provides the same public interface as ``InMemorySavedEstimateStore`` but
persists snapshots to SQLite so they survive server restarts.

All nested Pydantic models (``PredictionFeatures``, ``CostPrediction``,
``Plan``, ``OopInterval``) are stored as JSON blobs and round-tripped via
``model_validate_json`` / ``model_dump_json``.  This keeps the schema simple
and avoids a proliferation of joined tables for the nested structures.

Schema
------
``saved_estimates``
    One row per snapshot.  ``user_id`` is indexed for fast per-user list
    queries.  ``plan_json`` and ``oop_interval_json`` are nullable because
    a session may be saved before a plan is confirmed.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from gauge.benefits.models import Plan
from gauge.predictor.annual_cost import OopInterval
from gauge.predictor.model import CostPrediction
from gauge.predictor.schemas import PredictionFeatures
from gauge.saved_estimates.models import SavedEstimate

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS saved_estimates (
        id                 TEXT PRIMARY KEY,
        user_id            TEXT NOT NULL,
        label              TEXT NOT NULL,
        features_json      TEXT NOT NULL,
        prediction_json    TEXT NOT NULL,
        plan_json          TEXT,
        oop_interval_json  TEXT,
        created_at         TEXT NOT NULL
    )
"""

_CREATE_USER_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_saved_estimates_user_id
    ON saved_estimates (user_id)
"""


class SqliteSavedEstimateStore:
    """SQLite-backed store of named estimate snapshots, scoped by user ID.

    Parameters
    ----------
    db_path : Path or str
        Path to the SQLite database file.  Created (along with any missing
        parent directories) if it does not already exist.
    """

    def __init__(self, db_path: Path | str) -> None:
        """Open (or create) the database and ensure the table and index exist.

        Parameters
        ----------
        db_path : Path or str
            Path to the SQLite ``.db`` file.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_USER_INDEX)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public interface (mirrors InMemorySavedEstimateStore)
    # ------------------------------------------------------------------

    def save(
        self,
        user_id: str,
        label: str,
        features: PredictionFeatures,
        prediction: CostPrediction,
        plan: Plan | None,
        oop_interval: OopInterval | None,
    ) -> SavedEstimate:
        """Persist a new named estimate snapshot owned by ``user_id``.

        Parameters
        ----------
        user_id : str
            Anonymous identity of the requesting user.
        label : str
            Human-readable name for this snapshot.
        features : PredictionFeatures
            Demographics used for the prediction.
        prediction : CostPrediction
            Raw ML prediction.
        plan : Plan or None
            Confirmed plan, or ``None``.
        oop_interval : OopInterval or None
            Conformal OOP interval, or ``None``.

        Returns
        -------
        SavedEstimate
            The newly created snapshot.
        """
        estimate = SavedEstimate(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            label=label,
            features=features,
            prediction=prediction,
            plan=plan,
            oop_interval=oop_interval,
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO saved_estimates"
                " (id, user_id, label, features_json, prediction_json,"
                "  plan_json, oop_interval_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    estimate.id,
                    estimate.user_id,
                    estimate.label,
                    estimate.features.model_dump_json(),
                    estimate.prediction.model_dump_json(),
                    estimate.plan.model_dump_json() if estimate.plan else None,
                    estimate.oop_interval.model_dump_json() if estimate.oop_interval else None,
                    estimate.created_at.isoformat(),
                ),
            )
            self._conn.commit()
        return estimate

    def list(self, user_id: str) -> list[SavedEstimate]:
        """Return all estimates owned by ``user_id``, newest first.

        Parameters
        ----------
        user_id : str
            Caller's anonymous identity.

        Returns
        -------
        list[SavedEstimate]
            Snapshots owned by ``user_id``, sorted by ``created_at`` descending.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, user_id, label, features_json, prediction_json,"
                "       plan_json, oop_interval_json, created_at"
                " FROM saved_estimates"
                " WHERE user_id = ?"
                " ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [_row_to_estimate(row) for row in rows]

    def get(self, estimate_id: str) -> SavedEstimate | None:
        """Retrieve a single saved estimate by ID regardless of owner.

        Parameters
        ----------
        estimate_id : str
            The ``id`` of the snapshot to fetch.

        Returns
        -------
        SavedEstimate or None
            The deserialised snapshot, or ``None`` if not found.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id, user_id, label, features_json, prediction_json,"
                "       plan_json, oop_interval_json, created_at"
                " FROM saved_estimates WHERE id = ?",
                (estimate_id,),
            ).fetchone()
        return _row_to_estimate(row) if row else None

    def rename(self, estimate_id: str, user_id: str, label: str) -> SavedEstimate:
        """Update the label of an estimate the caller owns.

        Parameters
        ----------
        estimate_id : str
            The ``id`` of the snapshot to rename.
        user_id : str
            Caller's anonymous identity.  Must match the snapshot's owner.
        label : str
            New label to assign.

        Returns
        -------
        SavedEstimate
            Updated snapshot.

        Raises
        ------
        KeyError
            If no snapshot with ``estimate_id`` exists.
        PermissionError
            If ``user_id`` does not match the snapshot's owner.
        """
        existing = self.get(estimate_id)
        if existing is None:
            raise KeyError(estimate_id)
        if existing.user_id != user_id:
            raise PermissionError(estimate_id)
        with self._lock:
            self._conn.execute(
                "UPDATE saved_estimates SET label = ? WHERE id = ?",
                (label, estimate_id),
            )
            self._conn.commit()
        return existing.model_copy(update={"label": label})

    def delete(self, estimate_id: str, user_id: str) -> None:
        """Remove an estimate the caller owns.

        Parameters
        ----------
        estimate_id : str
            The ``id`` of the snapshot to delete.
        user_id : str
            Caller's anonymous identity.  Must match the snapshot's owner.

        Raises
        ------
        KeyError
            If no snapshot with ``estimate_id`` exists.
        PermissionError
            If ``user_id`` does not match the snapshot's owner.
        """
        existing = self.get(estimate_id)
        if existing is None:
            raise KeyError(estimate_id)
        if existing.user_id != user_id:
            raise PermissionError(estimate_id)
        with self._lock:
            self._conn.execute(
                "DELETE FROM saved_estimates WHERE id = ?",
                (estimate_id,),
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection.

        Returns
        -------
        None
        """
        with self._lock:
            self._conn.close()


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _row_to_estimate(row: tuple[str, ...]) -> SavedEstimate:
    """Deserialise a database row into a ``SavedEstimate``.

    Parameters
    ----------
    row : tuple
        A row from the ``saved_estimates`` table in column order:
        ``(id, user_id, label, features_json, prediction_json,
          plan_json, oop_interval_json, created_at)``.

    Returns
    -------
    SavedEstimate
        The deserialised snapshot.
    """
    (
        est_id,
        user_id,
        label,
        features_json,
        prediction_json,
        plan_json,
        oop_interval_json,
        created_at_str,
    ) = row

    try:
        created_at = datetime.fromisoformat(created_at_str)
    except ValueError:
        created_at = datetime.now(timezone.utc)

    return SavedEstimate(
        id=est_id,
        user_id=user_id,
        label=label,
        features=PredictionFeatures.model_validate_json(features_json),
        prediction=CostPrediction.model_validate_json(prediction_json),
        plan=Plan.model_validate_json(plan_json) if plan_json else None,
        oop_interval=OopInterval.model_validate_json(oop_interval_json)
        if oop_interval_json
        else None,
        created_at=created_at,
    )
