from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Set, Tuple

from clinic_utilities.physio_referral_runtime import (
    CU1ContractBundle,
    CU1ContractError,
    _canonical_id_from_selection,
    _deep_get,
    _normalize_whitespace,
)
from clinic_utilities.physio_rich_referral import CU1RichReferralRenderer


@dataclass(frozen=True)
class _ComposedFacts:
    phrases: List[str]
    residual_ids: List[str]


class CU1ClinicalContextComposer:
    """Deterministic selected-facts-only composer for rich-referral clinical context."""

    def __init__(self, bundle: CU1ContractBundle, rich_renderer: CU1RichReferralRenderer):
        self.bundle = bundle
        self.rich_renderer = rich_renderer
        payload = bundle.artifacts.get("clinical_composition_el")
        if not isinstance(payload, Mapping):
            raise CU1ContractError("CU-1 clinical composition artifact is missing")
        if payload.get("language") != "el" or payload.get("runtime_authorized") is not True:
            raise CU1ContractError("CU-1 clinical composition artifact is not active Greek runtime authority")
        self.payload = payload
        self.templates = payload.get("templates") if isinstance(payload.get("templates"), Mapping) else {}
        self.finding_rules = self._rules("finding_fusions")
        self.functional_rules = self._rules("functional_fusions")

    def compose(
        self,
        draft: Mapping[str, Any],
        *,
        detailed: bool,
        route: Tuple[str, str, Optional[str], Mapping[str, Any]],
        fallback_problem_label: str,
    ) -> List[str]:
        del detailed
        problem = draft.get("primary_problem") if isinstance(draft.get("primary_problem"), Mapping) else {}
        finding_ids = self._selected_ids(draft.get("findings"))
        function_ids = self._selected_ids(draft.get("functional_impairments"))

        composed_findings = self._compose_facts(finding_ids, self.finding_rules)
        composed_functions = self._compose_facts(function_ids, self.functional_rules)

        problem_phrase = self._route_problem_phrase(route)
        sentences: List[str] = []
        if problem_phrase:
            laterality = str(problem.get("laterality") or "")
            laterality_suffix = self._laterality_suffix(laterality)
            if composed_findings.phrases:
                template = self._template(
                    "patient_problem_with_findings_el",
                    "Ασθενής με {problem}{laterality}, με {findings}",
                )
                sentences.append(
                    template.format(
                        problem=problem_phrase,
                        laterality=laterality_suffix,
                        findings=self._join_greek(composed_findings.phrases),
                    )
                )
            else:
                template = self._template("patient_problem_only_el", "Ασθενής με {problem}{laterality}")
                sentences.append(template.format(problem=problem_phrase, laterality=laterality_suffix))
        else:
            sentences.append(fallback_problem_label)
            if composed_findings.phrases:
                sentences.append(self._residual_finding_sentence(composed_findings.phrases))

        if composed_findings.residual_ids:
            residual_labels = [self._required_label("findings", item) for item in composed_findings.residual_ids]
            sentences.append(self._residual_finding_sentence(residual_labels))

        function_phrases = list(composed_functions.phrases)
        function_phrases.extend(
            self._required_label("functional_impairments", item) for item in composed_functions.residual_ids
        )
        if function_phrases:
            template = self._template("functional_sentence_el", "Λειτουργικά αναφέρεται {function}")
            sentences.append(template.format(function=self._join_greek(function_phrases)))

        return [_normalize_whitespace(sentence).rstrip(".") for sentence in sentences if sentence.strip()]

    def _route_problem_phrase(
        self,
        route: Tuple[str, str, Optional[str], Mapping[str, Any]],
    ) -> Optional[str]:
        spec = self.rich_renderer.route_spec(
            profile_id=route[0],
            route_id=route[1],
            subtype_id=route[2],
            context=route[3],
        )
        labels = spec.get("clinical_context_problem_phrase_el_by_wording_mode")
        if not isinstance(labels, Mapping):
            return None
        wording_mode = route[3].get("__wording_mode")
        candidate = labels.get(wording_mode)
        if not isinstance(candidate, str) or not candidate.strip():
            return None
        return _normalize_whitespace(candidate)

    def _compose_facts(self, selected_ids: Sequence[str], rules: Sequence[Mapping[str, Any]]) -> _ComposedFacts:
        remaining: Set[str] = set(selected_ids)
        phrases: List[str] = []
        for rule in rules:
            required = self._string_list(rule.get("require_all"), field="require_all", rule=rule)
            consume = self._string_list(rule.get("consume"), field="consume", rule=rule)
            suppress = self._string_list(
                rule.get("suppress_if_matched") or [], field="suppress_if_matched", rule=rule
            )
            if not set(consume).issubset(set(required)):
                raise CU1ContractError(f"Clinical composition consume must be a subset of require_all: {rule.get('rule_id')}")
            if not set(required).issubset(remaining):
                continue
            phrase = rule.get("phrase_el")
            if not isinstance(phrase, str) or not phrase.strip():
                raise CU1ContractError(f"Clinical composition rule missing phrase_el: {rule.get('rule_id')}")
            phrases.append(_normalize_whitespace(phrase))
            remaining.difference_update(consume)
            remaining.difference_update(suppress)

        residual_ids = [item for item in selected_ids if item in remaining]
        return _ComposedFacts(phrases=phrases, residual_ids=residual_ids)

    def _rules(self, key: str) -> List[Mapping[str, Any]]:
        values = self.payload.get(key)
        if not isinstance(values, list):
            raise CU1ContractError(f"CU-1 clinical composition {key} must be a list")
        rules: List[Mapping[str, Any]] = []
        seen_ids: Set[str] = set()
        for item in values:
            if not isinstance(item, Mapping):
                raise CU1ContractError(f"Invalid CU-1 clinical composition rule in {key}")
            rule_id = item.get("rule_id")
            priority = item.get("priority")
            if not isinstance(rule_id, str) or not rule_id or rule_id in seen_ids:
                raise CU1ContractError(f"Invalid/duplicate CU-1 clinical composition rule_id in {key}")
            if not isinstance(priority, int):
                raise CU1ContractError(f"CU-1 clinical composition rule priority must be integer: {rule_id}")
            seen_ids.add(rule_id)
            rules.append(item)
        return sorted(rules, key=lambda item: int(item.get("priority") or 0), reverse=True)

    @staticmethod
    def _selected_ids(values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        result: List[str] = []
        seen: Set[str] = set()
        for item in values:
            canonical_id = _canonical_id_from_selection(item)
            if canonical_id and canonical_id not in seen:
                seen.add(canonical_id)
                result.append(canonical_id)
        return result

    @staticmethod
    def _string_list(value: Any, *, field: str, rule: Mapping[str, Any]) -> List[str]:
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise CU1ContractError(f"Clinical composition rule {rule.get('rule_id')} has invalid {field}")
        return list(value)

    def _required_label(self, section: str, canonical_id: str) -> str:
        language = self.bundle.artifacts.get("referral_language_el")
        corrections = self.bundle.artifacts.get("referral_language_el_corrections")
        value = _deep_get(corrections, f"{section}.{canonical_id}")
        if not isinstance(value, str) or not value.strip():
            value = _deep_get(language, f"{section}.{canonical_id}")
        if not isinstance(value, str) or not value.strip():
            raise CU1ContractError(f"Missing Greek clinician-facing label for {section}.{canonical_id}")
        return _normalize_whitespace(value)

    def _laterality_suffix(self, laterality: str) -> str:
        mapping = self.templates.get("laterality_suffix_el")
        value = mapping.get(laterality) if isinstance(mapping, Mapping) else None
        if not isinstance(value, str):
            return ""
        return value

    def _residual_finding_sentence(self, phrases: Sequence[str]) -> str:
        template = self._template("residual_findings_sentence_el", "Στην κλινική εικόνα καταγράφονται {findings}")
        return template.format(findings=self._join_greek(list(phrases)))

    def _template(self, key: str, default: str) -> str:
        value = self.templates.get(key) if isinstance(self.templates, Mapping) else None
        return _normalize_whitespace(value) if isinstance(value, str) and value.strip() else default

    @staticmethod
    def _join_greek(items: Sequence[str]) -> str:
        clean = [item.strip() for item in items if isinstance(item, str) and item.strip()]
        if not clean:
            return ""
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]} και {clean[1]}"
        return ", ".join(clean[:-1]) + f" και {clean[-1]}"


__all__ = ["CU1ClinicalContextComposer"]
