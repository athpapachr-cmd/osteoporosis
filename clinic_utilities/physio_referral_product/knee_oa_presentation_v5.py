"""Bounded post-v4 Knee-OA usability/prose presentation polish.

This layer changes only final presented Greek referral prose. It does not alter
structured clinical state, CU-1 safety/validation, evidence, suggestions,
jurisdiction semantics or treatment selection.
"""
from __future__ import annotations

import copy
from typing import Any

from clinic_utilities.physio_referral_product.knee_oa_presentation_v4 import (
    present_project_result as present_v4_project_result,
)


def _reconcile_pain_overlap(text: str, state: dict[str, Any]) -> str:
    """Prevent a richer pain qualifier from fusing with legacy joint-line prose.

    The clinician-selected structured state is left untouched. This is only a
    display reconciliation for the known overlap between the detailed pain
    qualifier and the older broad ``joint_line_pain`` output phrase.
    """

    qualifiers = state.get("qualifiers") or {}
    locations = set(qualifiers.get("pain_locations") or [])
    findings = set(state.get("findings") or [])
    if not locations or "joint_line_pain" not in findings:
        return text

    marker = " στη μεσάρθρια γραμμή"
    if marker not in text:
        return text

    # If the richer qualifier already identifies a medial/lateral joint-line
    # location, the broad suffix is redundant. Otherwise preserve the separately
    # selected broad joint-line finding as a separate pain phrase rather than
    # fusing it grammatically with another location such as pes anserine.
    if locations & {"medial_joint_line", "lateral_joint_line"}:
        return text.replace(marker, "", 1)
    return text.replace(marker, " και πόνο στη μεσάρθρια γραμμή", 1)


def _improve_readability(text: str) -> str:
    text = text.replace(
        "Επιπλέον στόχος: ",
        "Επιπρόσθετη λειτουργική προτεραιότητα: ",
    ).replace(
        "Επιπλέον στόχοι: ",
        "Επιπρόσθετες λειτουργικές προτεραιότητες: ",
    )

    marker = " Παρακαλώ για φυσιοθεραπευτική αξιολόγηση"
    if marker in text:
        text = text.replace(marker, "\n\nΠαρακαλώ για φυσιοθεραπευτική αξιολόγηση", 1)
    return text


def present_project_result(result: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
    presented = present_v4_project_result(result, request_payload)
    text = presented.get("text")
    if not isinstance(text, str) or not text:
        return presented

    state = copy.deepcopy(request_payload.get("state") or {})
    text = _reconcile_pain_overlap(text, state)
    text = _improve_readability(text)
    presented["text"] = text
    return presented


__all__ = ["present_project_result"]
