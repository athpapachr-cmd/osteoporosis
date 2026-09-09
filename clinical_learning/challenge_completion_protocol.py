from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID


CompletionState = Literal[
    "IN_PROGRESS",
    "DEBRIEF_COMPLETE_HANDOFF_PENDING",
    "HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW",
    "HANDOFF_FAILED_MANUAL_FALLBACK_READY",
]


@dataclass(frozen=True)
class HandoffDecision:
    state: CompletionState
    receipt: dict[str, str] | None = None
    reason: str | None = None


def _uuid_text(value: Any) -> str | None:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def classify_handoff_receipt(
    receipt: Any,
    *,
    transport_attempted: bool,
    debrief_complete: bool,
) -> HandoffDecision:
    """Classify only transport evidence; never infer Cockpit success from invocation.

    The Cockpit transport is considered successful only when the returned receipt
    proves creation of a pending-review import with stable identifiers. This helper
    intentionally does not inspect or mutate patient, Foundation, Signal or
    reference-verification state.
    """
    if not debrief_complete:
        return HandoffDecision(state="IN_PROGRESS")

    if not transport_attempted:
        return HandoffDecision(
            state="DEBRIEF_COMPLETE_HANDOFF_PENDING",
            reason="trusted_transport_not_attempted",
        )

    if not isinstance(receipt, dict):
        return HandoffDecision(
            state="HANDOFF_FAILED_MANUAL_FALLBACK_READY",
            reason="transport_receipt_missing",
        )

    if receipt.get("state") != "pending_review":
        return HandoffDecision(
            state="HANDOFF_FAILED_MANUAL_FALLBACK_READY",
            reason="transport_receipt_state_invalid",
        )

    import_id = _uuid_text(receipt.get("import_id"))
    source_event_id = _uuid_text(receipt.get("source_event_id"))
    source_format = str(receipt.get("source_format") or "").strip()
    if not import_id or not source_event_id:
        return HandoffDecision(
            state="HANDOFF_FAILED_MANUAL_FALLBACK_READY",
            reason="transport_receipt_identity_invalid",
        )
    if source_format not in {"canonical_challenge_v1", "rich_challenge_export_v1"}:
        return HandoffDecision(
            state="HANDOFF_FAILED_MANUAL_FALLBACK_READY",
            reason="transport_receipt_source_format_invalid",
        )

    return HandoffDecision(
        state="HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW",
        receipt={
            "state": "pending_review",
            "import_id": import_id,
            "source_event_id": source_event_id,
            "source_format": source_format,
        },
    )
