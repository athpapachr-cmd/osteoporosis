from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from clinic_utilities.physio_referral_runtime import (
    CU1ContractBundle,
    CU1ContractError,
    _canonical_id_from_selection,
    _deep_get,
    _is_empty,
    _normalize_whitespace,
)


_GREEK_RE = re.compile(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊϋΐΰ]")
_MACHINE_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)+$")


class CU1GreekReferralFormatter:
    """Natural-language Greek formatter for a validated normalized ReferralDraftV1.

    Validation, route ownership and safety remain owned by CU1Engine. This class
    only renders already-validated semantic state and must never infer missing
    diagnoses/findings/restrictions.
    """

    def __init__(self, bundle: CU1ContractBundle):
        self.bundle = bundle
        language = bundle.artifacts.get("referral_language_el")
        if not isinstance(language, Mapping):
            raise CU1ContractError("CU-1 Greek referral language artifact is missing")
        if language.get("language") != "el":
            raise CU1ContractError("CU-1 referral language artifact must be Greek")
        self.language: Mapping[str, Any] = language
        self._profile_route_labels = self._build_profile_route_labels()

    def format(self, draft: Mapping[str, Any], mode: str) -> str:
        if mode not in {"short", "detailed"}:
            raise CU1ContractError(f"Unsupported CU-1 formatter mode: {mode}")
        if mode == "short":
            text = self._format_short(draft)
        else:
            text = self._format_detailed(draft)
        self._assert_no_machine_id_leak(text)
        return text.rstrip() + "\n"

    # ------------------------------------------------------------------
    # Route/problem labels
    # ------------------------------------------------------------------

    def _build_profile_route_labels(self) -> Dict[str, Dict[str, str]]:
        labels: Dict[str, Dict[str, str]] = {}
        profiles = self.bundle.registry.get("profiles", {})
        if not isinstance(profiles, Mapping):
            return labels
        for profile_id, spec in profiles.items():
            if not isinstance(spec, Mapping):
                continue
            source = spec.get("source")
            if not isinstance(source, str):
                continue
            labels[str(profile_id)] = self._extract_route_labels(self.bundle.root / source)
        return labels

    def _extract_route_labels(self, profile_path: Path) -> Dict[str, str]:
        if not profile_path.exists():
            return {}
        text = profile_path.read_text(encoding="utf-8")
        labels: Dict[str, str] = {}

        # Most frozen profiles use a compact fenced form:
        # key: knee_osteoarthritis
        # display: Οστεοαρθρίτιδα γόνατος
        compact = re.compile(
            r"(?im)^\s*(?:key|structured\s+key)\s*:\s*([a-z0-9_]+)\s*$"
            r"(?:(?!^\s*```\s*$).){0,1200}?"
            r"^\s*(?:default\s+display|display)\s*:\s*([^\n`]+)\s*$"
        )
        for match in compact.finditer(text):
            label = _normalize_whitespace(match.group(2).lstrip("> "))
            if label:
                labels[match.group(1)] = label

        # Some profiles use a prose heading followed by a key code block and a
        # blockquote display line.
        structured = re.compile(
            r"(?is)(?:Structured\s+key|Structured\s+key:)\s*\n\s*```(?:text)?\s*\n\s*([a-z0-9_]+)\s*\n\s*```"
            r"(?:(?!\n## ).){0,1200}?"
            r"(?:Default\s+display|Display)\s*:\s*\n\s*>\s*([^\n]+)"
        )
        for match in structured.finditer(text):
            label = _normalize_whitespace(match.group(2))
            if label:
                labels[match.group(1)] = label

        return labels

    def _route_label(self, profile_id: str, route_id: str) -> str:
        label = self._profile_route_labels.get(profile_id, {}).get(route_id)
        if not label:
            # The older runtime parser may have found a label; use it only if it
            # is already Greek and not a machine-id fallback.
            label = self.bundle.profile_route_labels.get(profile_id, {}).get(route_id)
        if not label or not self._is_greek_clinician_phrase(label):
            raise CU1ContractError(
                f"Missing Greek clinician-facing route label for {profile_id}.{route_id}"
            )
        return label

    def _problem_label(self, problem: Mapping[str, Any], *, include_subtype: bool) -> str:
        profile_id = str(problem.get("profile_id") or "")
        route_id = str(problem.get("route_id") or "")
        label = self._route_label(profile_id, route_id)
        laterality = self._optional_label("laterality", str(problem.get("laterality") or ""))
        if laterality:
            label = f"{label} ({laterality})"
        subtype = problem.get("subtype_id_optional")
        if include_subtype and subtype:
            subtype_label = self._optional_label("route_detail_labels", str(subtype))
            if subtype_label:
                label += f" — {subtype_label}"
        return label

    # ------------------------------------------------------------------
    # Label/selection rendering
    # ------------------------------------------------------------------

    def _required_label(self, section: str, canonical_id: str) -> str:
        value = _deep_get(self.language, f"{section}.{canonical_id}")
        if not isinstance(value, str) or not value.strip():
            raise CU1ContractError(
                f"Missing Greek clinician-facing label for {section}.{canonical_id}"
            )
        value = _normalize_whitespace(value)
        if not self._is_greek_clinician_phrase(value):
            raise CU1ContractError(
                f"Non-Greek clinician-facing label for {section}.{canonical_id}: {value}"
            )
        return value

    def _optional_label(self, section: str, canonical_id: str) -> Optional[str]:
        if not canonical_id or canonical_id in {"not_stated", "not_assessed", "not_applicable"}:
            return None
        value = _deep_get(self.language, f"{section}.{canonical_id}")
        if not isinstance(value, str) or not value.strip():
            return None
        return _normalize_whitespace(value)

    def _selection_labels(self, draft: Mapping[str, Any], key: str, language_section: str) -> List[str]:
        values = draft.get(key, [])
        if not isinstance(values, list):
            return []
        rendered: List[str] = []
        for item in values:
            canonical_id = _canonical_id_from_selection(item)
            if not canonical_id:
                continue
            rendered.append(self._required_label(language_section, canonical_id))
        return rendered

    def _restriction_labels(self, draft: Mapping[str, Any], *, detailed: bool) -> List[str]:
        values = draft.get("explicit_restrictions", [])
        if not isinstance(values, list):
            return []
        rendered: List[str] = []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            canonical_id = _canonical_id_from_selection(item)
            if not canonical_id:
                continue
            text = self._required_label("restrictions", canonical_id)
            state = item.get("state_or_value")
            if isinstance(state, str) and state not in {"not_stated", "not_assessed"}:
                state_label = self._optional_label("context_values", state)
                if state_label:
                    text += f": {state_label}"
                elif not _MACHINE_ID_RE.match(state):
                    text += f": {_normalize_whitespace(state)}"
            notes = item.get("notes_optional")
            if detailed and isinstance(notes, str) and notes.strip():
                text += f" ({_normalize_whitespace(notes)})"
            rendered.append(text)
        return rendered

    def _measurement_labels(self, draft: Mapping[str, Any]) -> List[str]:
        values = draft.get("measurements", [])
        if not isinstance(values, list):
            return []
        rendered: List[str] = []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            canonical_id = _canonical_id_from_selection(item)
            if not canonical_id:
                continue
            label = self._required_label("measurements", canonical_id)
            value = item.get("value")
            unit = item.get("unit_optional")
            if value is not None:
                label += f": {value}"
                if isinstance(unit, str) and unit.strip():
                    label += f" {unit.strip()}"
            rendered.append(label)
        return rendered

    def _adjunct_labels(self, draft: Mapping[str, Any], *, detailed: bool) -> List[str]:
        values = draft.get("adjunct_options", [])
        if not isinstance(values, list):
            return []
        rendered: List[str] = []
        for item in values:
            canonical_id = _canonical_id_from_selection(item)
            if not canonical_id:
                continue
            text = self._required_label("adjuncts", canonical_id)
            if detailed and isinstance(item, Mapping) and item.get("provenance") == "therapist_proposed_context":
                text += " (ως επιλογή που προτάθηκε από τον φυσιοθεραπευτή)"
            rendered.append(text)
        return rendered

    # ------------------------------------------------------------------
    # Natural prose composition
    # ------------------------------------------------------------------

    def _format_short(self, draft: Mapping[str, Any]) -> str:
        problem = draft.get("primary_problem", {}) if isinstance(draft.get("primary_problem"), Mapping) else {}
        problem_label = self._problem_label(problem, include_subtype=False)
        sentences: List[str] = [
            f"Παραπέμπεται για φυσιοθεραπευτική αποκατάσταση με κύριο πρόβλημα: {problem_label}."
        ]

        findings = self._selection_labels(draft, "findings", "findings")
        function = self._selection_labels(draft, "functional_impairments", "functional_impairments")
        if findings and function:
            sentences.append(
                f"Η κλινική εικόνα περιλαμβάνει {self._join_greek(findings)}, με λειτουργικό περιορισμό σε {self._join_greek(function)}."
            )
        elif findings:
            sentences.append(f"Η κλινική εικόνα περιλαμβάνει {self._join_greek(findings)}.")
        elif function:
            sentences.append(f"Λειτουργικά καταγράφεται {self._join_greek(function)}.")

        restrictions = self._restriction_labels(draft, detailed=False)
        precautions = self._selection_labels(draft, "precautions", "restrictions")
        all_limits = restrictions + precautions
        if all_limits:
            sentences.append(f"Να τηρηθούν οι εξής περιορισμοί/προφυλάξεις: {self._join_greek(all_limits)}.")

        directions = self._selection_labels(draft, "rehab_directions", "rehab_directions")
        goals = self._selection_labels(draft, "goals", "goals")
        if directions and goals:
            sentences.append(
                f"Παρακαλώ για {self._join_greek(directions)}, με στόχο {self._join_greek(goals)}."
            )
        elif directions:
            sentences.append(f"Παρακαλώ για {self._join_greek(directions)}.")
        elif goals:
            sentences.append(f"Στόχοι αποκατάστασης: {self._join_greek(goals)}.")

        adjuncts = self._adjunct_labels(draft, detailed=False)
        if adjuncts:
            sentences.append(f"Επιπλέον έχει επιλεγεί {self._join_greek(adjuncts)}.")

        note = draft.get("clinician_free_text_optional")
        if isinstance(note, str) and note.strip():
            sentences.append(_normalize_whitespace(note).rstrip(".") + ".")

        return " ".join(sentences)

    def _format_detailed(self, draft: Mapping[str, Any]) -> str:
        problem = draft.get("primary_problem", {}) if isinstance(draft.get("primary_problem"), Mapping) else {}
        problem_label = self._problem_label(problem, include_subtype=True)
        sections: List[str] = ["ΠΑΡΑΠΟΜΠΗ ΓΙΑ ΦΥΣΙΟΘΕΡΑΠΕΙΑ"]

        clinical_sentences: List[str] = [
            f"Παραπέμπεται για φυσιοθεραπευτική αποκατάσταση με κύριο πρόβλημα: {problem_label}."
        ]
        findings = self._selection_labels(draft, "findings", "findings")
        function = self._selection_labels(draft, "functional_impairments", "functional_impairments")
        if findings:
            clinical_sentences.append(f"Στην κλινική εικόνα καταγράφονται {self._join_greek(findings)}.")
        if function:
            clinical_sentences.append(f"Λειτουργικά καταγράφεται {self._join_greek(function)}.")

        secondary = draft.get("secondary_problems", [])
        if isinstance(secondary, list):
            labels = [self._problem_label(item, include_subtype=True) for item in secondary if isinstance(item, Mapping)]
            if labels:
                clinical_sentences.append(f"Συνυπάρχει επίσης {self._join_greek(labels)}.")

        sections.append("Κλινική εικόνα\n" + " ".join(clinical_sentences))

        restrictions = self._restriction_labels(draft, detailed=True)
        precautions = self._selection_labels(draft, "precautions", "restrictions")
        if restrictions or precautions:
            text = []
            if restrictions:
                text.append(f"Ισχύουν οι εξής περιορισμοί: {self._join_greek(restrictions)}.")
            if precautions:
                text.append(f"Προφυλάξεις: {self._join_greek(precautions)}.")
            sections.append("Περιορισμοί / προφυλάξεις\n" + " ".join(text))

        goals = self._selection_labels(draft, "goals", "goals")
        directions = self._selection_labels(draft, "rehab_directions", "rehab_directions")
        adjuncts = self._adjunct_labels(draft, detailed=True)
        plan: List[str] = []
        if goals:
            plan.append(f"Στόχοι αποκατάστασης: {self._join_greek(goals)}.")
        if directions:
            plan.append(f"Παρακαλώ για {self._join_greek(directions)}.")
        if adjuncts:
            plan.append(f"Συμπληρωματικά: {self._join_greek(adjuncts)}.")
        if plan:
            sections.append("Στόχοι και κατευθύνσεις αποκατάστασης\n" + " ".join(plan))

        extra = self._detailed_context_sentences(problem)
        measurements = self._measurement_labels(draft)
        if measurements:
            extra.append(f"Μετρήσεις: {self._join_greek(measurements)}.")
        note = draft.get("clinician_free_text_optional")
        if isinstance(note, str) and note.strip():
            extra.append(f"Κλινική σημείωση: {_normalize_whitespace(note).rstrip('.')}.")
        disposition = _deep_get(draft, "safety.clinician_disposition")
        disposition_label = self._optional_label("dispositions", str(disposition or ""))
        if disposition_label:
            extra.append(f"Καταγεγραμμένη ενέργεια: {disposition_label}.")
        if extra:
            sections.append("Πρόσθετα κλινικά στοιχεία\n" + " ".join(extra))

        return "\n\n".join(sections)

    def _detailed_context_sentences(self, problem: Mapping[str, Any]) -> List[str]:
        context = problem.get("context", {})
        if not isinstance(context, Mapping):
            return []
        rendered: List[str] = []
        for key, value in context.items():
            if key == "neurological_screen" or _is_empty(value):
                continue
            key_label = self._optional_label("context_keys", str(key))
            if not key_label:
                continue
            value_label = self._context_value_label(value)
            if value_label:
                rendered.append(f"{key_label}: {value_label}.")

        neuro = context.get("neurological_screen")
        if isinstance(neuro, Mapping):
            parts: List[str] = []
            for key, value in neuro.items():
                if value in {None, "not_assessed", "not_stated"}:
                    continue
                key_label = self._optional_label("neurological_keys", str(key))
                value_label = self._optional_label("context_values", str(value))
                if key_label and value_label:
                    parts.append(f"{key_label} {value_label}")
            if parts:
                rendered.append(f"Νευρολογικός έλεγχος: {self._join_greek(parts)}.")
        return rendered

    def _context_value_label(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            return "ναι" if value else "όχι"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            if value in {"not_stated", "not_assessed", "not_applicable", "unknown"}:
                return None
            for section in ("context_values", "route_detail_labels"):
                label = self._optional_label(section, value)
                if label:
                    return label
            # Explicit clinician-entered/free-text values can be rendered when
            # they are clearly not canonical snake_case machine identifiers.
            if not _MACHINE_ID_RE.match(value):
                return _normalize_whitespace(value)
            return None
        return None

    @staticmethod
    def _join_greek(items: Iterable[str]) -> str:
        values = [item.strip() for item in items if isinstance(item, str) and item.strip()]
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} και {values[1]}"
        return ", ".join(values[:-1]) + f" και {values[-1]}"

    @staticmethod
    def _is_greek_clinician_phrase(value: str) -> bool:
        # Standard abbreviations/proper names may coexist, but the phrase must
        # contain Greek text rather than being a raw English machine token.
        return bool(_GREEK_RE.search(value)) and "_" not in value

    def _assert_no_machine_id_leak(self, text: str) -> None:
        if "_" in text:
            raise CU1ContractError("Generated CU-1 referral contains an underscore/machine-id leak")
        # Assert that known selectable canonical ids never appear verbatim even
        # if they happen not to contain underscores.
        sections = (
            "findings",
            "functional_impairments",
            "goals",
            "rehab_directions",
            "adjuncts",
            "measurements",
            "restrictions",
        )
        for section in sections:
            mapping = self.language.get(section, {})
            if not isinstance(mapping, Mapping):
                continue
            for canonical_id in mapping:
                if canonical_id and str(canonical_id) in text:
                    raise CU1ContractError(
                        f"Generated CU-1 referral leaked machine id: {canonical_id}"
                    )


__all__ = ["CU1GreekReferralFormatter"]
