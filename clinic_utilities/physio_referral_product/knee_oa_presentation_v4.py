"""Post-release Knee-OA presentation polish shared by both transports.

This module changes only final Greek display prose. Structured clinical state,
CU-1 validation/safety, evidence states, suggestion policy and readiness remain
owned by the existing shared projection.
"""
from __future__ import annotations

import copy
from typing import Any


def present_project_result(result: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
    presented = copy.deepcopy(result)
    text = presented.get("text")
    if not isinstance(text, str) or not text:
        return presented

    # Multiple selected pain locations are unordered. Do not invent a primary
    # location by inserting "κυρίως" unless the state explicitly carries rank.
    text = text.replace("πόνο κυρίως ", "πόνο ", 1)

    qualifiers = ((request_payload.get("state") or {}).get("qualifiers") or {})
    if qualifiers.get("weakness_detail") == "quadriceps_exam":
        for phrase in (
            "αδυναμία τετρακεφάλου",
            "μυϊκή αδυναμία",
        ):
            if phrase in text:
                text = text.replace(phrase, "αδυναμία του τετρακεφάλου κατά την εξέταση", 1)
                break

    # Keep the clinician-to-physiotherapist sentence continuous and natural.
    # The intervention list remains exactly the reviewed clinician-selected list.
    text = text.replace(
        ", με ενδεικτικές προτεραιότητες ",
        ", με έμφαση σε ",
        1,
    ).replace(
        ", με αρχικές προτεραιότητες ",
        ", με έμφαση σε ",
        1,
    )

    presented["text"] = text
    return presented


__all__ = ["present_project_result"]
