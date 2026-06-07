"""In-memory storage and the protocol that the API depends on.

The repository is split into a protocol and a concrete implementation so
that the API does not depend on a particular storage choice. Swapping in a
database-backed implementation later only requires conforming to the
protocol.
"""

from __future__ import annotations

from typing import Protocol

from gauge.benefits.models import Member, Plan, Procedure


class CatalogRepository(Protocol):
    """Read-only lookups for plans, members, and procedures."""

    def get_plan(self, plan_id: str) -> Plan | None:
        """Return the plan with ``plan_id``, or ``None`` if absent."""
        ...

    def get_member(self, member_id: str) -> Member | None:
        """Return the member with ``member_id``, or ``None`` if absent."""
        ...

    def get_procedure(self, code: str) -> Procedure | None:
        """Return the procedure with ``code``, or ``None`` if absent."""
        ...


class InMemoryRepository:
    """Trivial in-memory implementation used by the prototype."""

    def __init__(
        self,
        plans: list[Plan],
        members: list[Member],
        procedures: list[Procedure],
    ) -> None:
        """Build the repository from pre-constructed domain objects.

        Parameters
        ----------
        plans : list[Plan]
            Plans to make queryable by ``plan_id``.
        members : list[Member]
            Members to make queryable by ``member_id``.
        procedures : list[Procedure]
            Procedures to make queryable by ``code``.
        """
        self._plans: dict[str, Plan] = {p.plan_id: p for p in plans}
        self._members: dict[str, Member] = {m.member_id: m for m in members}
        self._procedures: dict[str, Procedure] = {
            p.code: p for p in procedures
        }

    def get_plan(self, plan_id: str) -> Plan | None:
        """Return the plan with ``plan_id``, or ``None`` if absent."""
        return self._plans.get(plan_id)

    def get_member(self, member_id: str) -> Member | None:
        """Return the member with ``member_id``, or ``None`` if absent."""
        return self._members.get(member_id)

    def get_procedure(self, code: str) -> Procedure | None:
        """Return the procedure with ``code``, or ``None`` if absent."""
        return self._procedures.get(code)
