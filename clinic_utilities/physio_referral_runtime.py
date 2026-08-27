from __future__ import annotations

import copy
import os
import re
import secrets
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


CONTRACT_VERSION = "cu1_referral_draft_v1"
SEVERITY_ORDER = {
    "info": 0,
    "soft_warning": 1,
    "hard_warning_ack_required": 2,
    "block_until_disposition": 3,
    "urgent_reassessment": 4,
}
ALLOWED_URGENT_DISPOSITIONS = {
    "urgent_or_same_day_assessment_arranged",
    "routine_physiotherapy_deferred",
}


class CU1ContractError(RuntimeError):
    pass


class CU1ValidationError(BaseModel):
    error_id: str
    error_class: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CU1SafetyResult(BaseModel):
    rule_id: str
    severity: str
    message_key: str
    acknowledgement_required: bool
    acknowledged: bool
    disposition_required: bool
    formatter_blocked: bool
    clinician_disposition: str
    source_profile_id: str
    source_route_id_optional: Optional[str] = None


class CU1ValidationResponse(BaseModel):
    normalized_draft: Dict[str, Any]
    validation_errors: List[CU1ValidationError]
    safety_results: List[CU1SafetyResult]
    highest_severity: Optional[str] = None
    formatter_blocked: bool


class CU1GenerateRequest(BaseModel):
    draft: Dict[str, Any]
    mode: str = Field(default="short", pattern=r"^(short|detailed)$")


class CU1GenerateResponse(CU1ValidationResponse):
    mode: str
    text: Optional[str] = None


class CU1ValidateRequest(BaseModel):
    draft: Dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise CU1ContractError(f"Missing normative CU-1 artifact: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive contract loader boundary
        raise CU1ContractError(f"Unable to parse normative CU-1 artifact: {path}") from exc
    if not isinstance(data, dict):
        raise CU1ContractError(f"Normative CU-1 artifact must be a mapping: {path}")
    return data


def _deep_get(data: Any, path: str, default: Any = None) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def _deep_set(data: MutableMapping[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: MutableMapping[str, Any] = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, MutableMapping):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip() == "not_stated"
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _missing(data: Mapping[str, Any], path: str) -> bool:
    sentinel = object()
    value = _deep_get(data, path, sentinel)
    if value is sentinel or value is None:
        return True
    return isinstance(value, str) and value == "not_stated"


def _canonical_id_from_selection(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        candidate = value.get("id") or value.get("finding_id") or value.get("restriction_id") or value.get("adjunct_id") or value.get("measurement_id")
        return str(candidate) if candidate is not None else None
    return None


def _contains(value: Any, canonical_id: str) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_canonical_id_from_selection(item) == canonical_id for item in value)
    return _canonical_id_from_selection(value) == canonical_id


def _humanize_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.replace("_", " ")


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _read_profile_route_labels(profile_path: Path) -> Dict[str, str]:
    """Extract clinician-facing route labels only.

    Frozen profile Markdown remains clinical/content authority. Runtime validation,
    routing, safety and precedence never depend on this parser; failures here only
    cause a display-label fallback to the canonical route id.
    """

    if not profile_path.exists():
        return {}
    text = profile_path.read_text(encoding="utf-8")
    labels: Dict[str, str] = {}
    pattern = re.compile(
        r"(?:Structured key:|Structured key\s*)\s*\n\s*```text\s*\n\s*([a-zA-Z0-9_]+)\s*\n\s*```"
        r"(?:(?!\n## ).){0,900}?"
        r"(?:Default display:|Display:)\s*\n\s*>\s*([^\n]+)",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        labels[match.group(1)] = _normalize_whitespace(match.group(2))
    return labels


class CU1ContractBundle:
    def __init__(self, root: Optional[Path] = None):
        self.root = root or _repo_root()
        self.manifest_path = self.root / "clinic_utilities/contracts/cu1_contract_manifest_v1.yaml"
        self.manifest = _load_yaml(self.manifest_path)
        self.artifacts: Dict[str, Dict[str, Any]] = {}
        self._load_normative_artifacts()
        self._apply_route_requirements_correction()
        self.profile_route_labels = self._load_profile_labels()
        self._validate_contract_integrity()

    def _load_normative_artifacts(self) -> None:
        normative = self.manifest.get("normative_machine_artifacts")
        if not isinstance(normative, Mapping):
            raise CU1ContractError("Manifest missing normative_machine_artifacts")
        required_yaml_keys = {
            "typed_supplement",
            "route_registry",
            "route_detail_catalog",
            "option_and_safety_catalog",
            "structured_option_scope",
            "id_normalization",
            "context_value_sets",
            "rule_catalog",
            "route_requirements",
            "route_requirements_correction",
            "validation_error_policy",
            "semantic_fixtures",
            "r1_r2_semantic_fixtures",
        }
        missing_keys = sorted(required_yaml_keys.difference(normative))
        if missing_keys:
            raise CU1ContractError(f"Manifest missing normative YAML keys: {missing_keys}")
        for key, rel_path in normative.items():
            if not isinstance(rel_path, str):
                raise CU1ContractError(f"Invalid manifest artifact path for {key}")
            path = self.root / rel_path
            if rel_path.endswith(".yaml") or rel_path.endswith(".yml"):
                self.artifacts[key] = _load_yaml(path)
            elif not path.exists():
                raise CU1ContractError(f"Missing normative CU-1 artifact: {path}")

    def _apply_route_requirements_correction(self) -> None:
        base = copy.deepcopy(self.artifacts["route_requirements"])
        correction = self.artifacts["route_requirements_correction"]
        if correction.get("applies_to") != base.get("version"):
            raise CU1ContractError("Route requirements correction applies_to mismatch")
        corrections = correction.get("corrections", {})
        if not isinstance(corrections, Mapping):
            raise CU1ContractError("Route requirements correction must contain a mapping")
        for path, payload in corrections.items():
            if not isinstance(payload, Mapping) or "replace_with" not in payload:
                raise CU1ContractError(f"Invalid route requirements correction at {path}")
            _deep_set(base, path, copy.deepcopy(payload["replace_with"]))
        self.artifacts["effective_route_requirements"] = base

    def _load_profile_labels(self) -> Dict[str, Dict[str, str]]:
        registry = self.artifacts["route_registry"]
        profiles = registry.get("profiles", {})
        result: Dict[str, Dict[str, str]] = {}
        if not isinstance(profiles, Mapping):
            return result
        for profile_id, profile in profiles.items():
            source = profile.get("source") if isinstance(profile, Mapping) else None
            if isinstance(source, str):
                result[str(profile_id)] = _read_profile_route_labels(self.root / source)
        return result

    def _validate_contract_integrity(self) -> None:
        if self.manifest.get("runtime_may_read_profile_markdown_for_trigger_or_validation_logic") is not False:
            raise CU1ContractError("Manifest must forbid profile Markdown trigger/validation interpretation")
        registry = self.artifacts["route_registry"]
        rules = self.artifacts["rule_catalog"]
        options = self.artifacts["option_and_safety_catalog"]
        if not isinstance(registry.get("profiles"), Mapping):
            raise CU1ContractError("Registry profiles missing")
        safety_ids = set(_deep_get(rules, "safety_input_flags.allowed_ids", []) or [])
        if not safety_ids:
            raise CU1ContractError("Closed safety input flag namespace missing")
        severity_refs = set((options.get("safety_rules") or {}).keys())
        for rule_id, rule in (rules.get("rules") or {}).items():
            if not isinstance(rule, Mapping):
                raise CU1ContractError(f"Invalid rule definition: {rule_id}")
            if rule.get("severity_ref") not in severity_refs:
                raise CU1ContractError(f"Rule {rule_id} references unknown severity_ref")

    @property
    def registry(self) -> Dict[str, Any]:
        return self.artifacts["route_registry"]

    @property
    def route_requirements(self) -> Dict[str, Any]:
        return self.artifacts["effective_route_requirements"]

    @property
    def context_values(self) -> Dict[str, Any]:
        return self.artifacts["context_value_sets"]

    @property
    def id_normalization(self) -> Dict[str, Any]:
        return self.artifacts["id_normalization"]

    @property
    def rules(self) -> Dict[str, Any]:
        return self.artifacts["rule_catalog"]

    @property
    def options(self) -> Dict[str, Any]:
        return self.artifacts["option_and_safety_catalog"]

    @property
    def validation_policy(self) -> Dict[str, Any]:
        return self.artifacts["validation_error_policy"]

    @property
    def route_detail_catalog(self) -> Dict[str, Any]:
        return self.artifacts["route_detail_catalog"]

    def contract_payload(self) -> Dict[str, Any]:
        profiles: Dict[str, Any] = {}
        for profile_id, profile in (self.registry.get("profiles") or {}).items():
            routes: Dict[str, Any] = {}
            for route_id, route in (profile.get("routes") or {}).items():
                routes[route_id] = {
                    "visibility": route.get("visibility"),
                    "wording_modes": route.get("wording_modes", []),
                    "owner_class": route.get("owner_class"),
                    "display": self.profile_route_labels.get(profile_id, {}).get(route_id, _humanize_id(route_id)),
                }
            profiles[profile_id] = {
                "display": PROFILE_LABELS.get(profile_id, _humanize_id(profile_id)),
                "routes": routes,
            }
        return {
            "contract_version": CONTRACT_VERSION,
            "profiles": profiles,
            "subtypes": self.registry.get("subtypes", {}),
            "gateways": self.registry.get("gateways", {}),
            "laterality_values": _deep_get(self.route_requirements, "base_requirements.laterality_values", []),
            "common_findings": self.options.get("common_findings", []),
            "profile_findings": self.options.get("profile_findings", {}),
            "functional_impairments": self.options.get("common_functional_impairments", []),
            "goals": self.options.get("common_goal_ids", []),
            "rehab_directions": self.options.get("common_rehab_direction_ids", []),
            "adjuncts": self.options.get("adjunct_ids", []),
            "measurements": self.options.get("measurement_ids", []),
            "restrictions": self.options.get("restriction_ids", []),
            "safety_input_flags": _deep_get(self.rules, "safety_input_flags.allowed_ids", []),
            "context_value_sets": {
                "shared_fracture": self.context_values.get("shared_fracture", {}),
                "shared_muscle_myotendinous": self.context_values.get("shared_muscle_myotendinous", {}),
                "shared_deconditioning_balance_gait": self.context_values.get("shared_deconditioning_balance_gait", {}),
                "postoperative_common": self.context_values.get("postoperative_common", {}),
            },
        }


PROFILE_LABELS = {
    "cervical": "Αυχενική μοίρα",
    "lumbar": "Οσφυϊκή μοίρα",
    "shoulder": "Ώμος",
    "elbow": "Αγκώνας",
    "wrist_hand": "Καρπός / Χέρι",
    "knee": "Γόνατο",
    "hip_groin": "Ισχίο / Βουβωνική χώρα",
    "ankle_foot": "Ποδοκνημική / Άκρος πόδας",
    "shared_fracture": "Κάταγμα / Μετά ακινητοποίηση",
    "shared_muscle_myotendinous": "Μυϊκή / μυοτενόντια κάκωση",
    "shared_deconditioning_balance_gait": "Αποδυνάμωση / Ισορροπία / Βάδιση",
}

LATERALITY_LABELS = {
    "left": "αριστερά",
    "right": "δεξιά",
    "bilateral": "αμφοτερόπλευρα",
    "midline": "μέση γραμμή",
    "not_applicable": "",
    "not_stated": "",
}

SECTION_LABELS = {
    "findings": "Επιλεγμένα ευρήματα",
    "functional_impairments": "Λειτουργικοί περιορισμοί",
    "explicit_restrictions": "Περιορισμοί",
    "precautions": "Προφυλάξεις",
    "goals": "Στόχοι",
    "rehab_directions": "Κατευθύνσεις αποκατάστασης",
    "adjunct_options": "Συμπληρωματικές επιλογές",
    "measurements": "Μετρήσεις",
}


class CU1Engine:
    def __init__(self, bundle: Optional[CU1ContractBundle] = None):
        self.bundle = bundle or CU1ContractBundle()

    def normalize(self, draft: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = copy.deepcopy(dict(draft))
        normalized.setdefault("contract_version", CONTRACT_VERSION)
        aliases = self.bundle.id_normalization.get("aliases", {}) or {}

        def norm_id(value: Any) -> Any:
            if isinstance(value, str):
                return aliases.get(value, value)
            return value

        normalized["body_region"] = norm_id(normalized.get("body_region"))
        for key in ("primary_problem",):
            problem = normalized.get(key)
            if isinstance(problem, MutableMapping):
                self._normalize_problem(problem, norm_id)
        secondary = normalized.get("secondary_problems")
        if isinstance(secondary, list):
            for problem in secondary:
                if isinstance(problem, MutableMapping):
                    self._normalize_problem(problem, norm_id)
        self._normalize_selection_list(normalized, "findings", ("finding_id", "id"), norm_id)
        self._normalize_selection_list(normalized, "functional_impairments", ("id",), norm_id)
        self._normalize_selection_list(normalized, "precautions", ("id",), norm_id)
        self._normalize_selection_list(normalized, "explicit_restrictions", ("restriction_id", "id"), norm_id)
        self._normalize_selection_list(normalized, "goals", ("id",), norm_id)
        self._normalize_selection_list(normalized, "rehab_directions", ("id",), norm_id)
        self._normalize_selection_list(normalized, "adjunct_options", ("adjunct_id", "id"), norm_id)
        self._normalize_selection_list(normalized, "measurements", ("measurement_id", "id"), norm_id)

        safety = normalized.setdefault("safety", {})
        if isinstance(safety, MutableMapping):
            flags = safety.get("input_flags", [])
            if isinstance(flags, list):
                safety["input_flags"] = [norm_id(item) for item in flags]
            acknowledgements = safety.get("acknowledged_rule_ids", [])
            if isinstance(acknowledgements, list):
                safety["acknowledged_rule_ids"] = [norm_id(item) for item in acknowledgements]

        self._resolve_shared_target(normalized)
        return normalized

    def _normalize_problem(self, problem: MutableMapping[str, Any], norm_id) -> None:
        for field in ("profile_id", "route_id", "subtype_id_optional", "source_route_optional"):
            if field in problem:
                problem[field] = norm_id(problem.get(field))
        context = problem.setdefault("context", {})
        if isinstance(context, MutableMapping):
            self._normalize_context(context, norm_id)
        target = problem.get("shared_target_optional")
        if isinstance(target, MutableMapping):
            for field in ("profile_id", "route_id", "subtype_or_site_id_optional"):
                if field in target:
                    target[field] = norm_id(target.get(field))

    def _normalize_selection_list(self, draft: MutableMapping[str, Any], key: str, candidate_keys: Tuple[str, ...], norm_id) -> None:
        values = draft.get(key)
        if not isinstance(values, list):
            return
        for index, item in enumerate(values):
            if isinstance(item, str):
                values[index] = norm_id(item)
            elif isinstance(item, MutableMapping):
                for candidate in candidate_keys:
                    if candidate in item:
                        item[candidate] = norm_id(item.get(candidate))
                        break

    def _normalize_context(self, context: MutableMapping[str, Any], norm_id) -> None:
        requirement_aliases = self.bundle.route_requirements.get("context_aliases", {}) or {}
        for alias_key, alias_spec in list(requirement_aliases.items()):
            if alias_key not in context or not isinstance(alias_spec, Mapping):
                continue
            canonical = alias_spec.get("canonical")
            if not canonical:
                continue
            raw_value = context.pop(alias_key)
            if isinstance(raw_value, bool):
                mapped = alias_spec.get("true_value") if raw_value else alias_spec.get("false_value")
            else:
                mapped = raw_value
            if canonical not in context:
                context[str(canonical)] = mapped

        for key, value in list(context.items()):
            canonical_key = norm_id(key)
            if canonical_key != key:
                context.pop(key)
                if canonical_key not in context:
                    context[canonical_key] = value
            if isinstance(value, str):
                context[canonical_key] = self._normalize_context_value(canonical_key, norm_id(value))
            elif isinstance(value, MutableMapping):
                for sub_key, sub_value in list(value.items()):
                    new_sub_key = norm_id(sub_key)
                    if new_sub_key != sub_key:
                        value.pop(sub_key)
                    value[new_sub_key] = norm_id(sub_value) if isinstance(sub_value, str) else sub_value

    def _normalize_context_value(self, key: str, value: str) -> str:
        groups = self.bundle.context_values
        for group_name in ("shared_fracture", "shared_muscle_myotendinous", "postoperative_common", "shared_deconditioning_balance_gait"):
            group = groups.get(group_name, {})
            alias_map = group.get(f"{key}_aliases", {}) if isinstance(group, Mapping) else {}
            if isinstance(alias_map, Mapping) and value in alias_map:
                return str(alias_map[value])
        return value

    def _resolve_shared_target(self, draft: MutableMapping[str, Any]) -> None:
        problem = draft.get("primary_problem")
        if not isinstance(problem, MutableMapping):
            return
        target = problem.get("shared_target_optional")
        if not isinstance(target, Mapping):
            return
        target_profile = target.get("profile_id")
        target_route = target.get("route_id")
        if not target_profile or not target_route:
            return
        source_route = problem.get("route_id")
        if source_route and not problem.get("source_route_optional"):
            problem["source_route_optional"] = source_route
        problem["profile_id"] = target_profile
        problem["route_id"] = target_route
        subtype_or_site = target.get("subtype_or_site_id_optional")
        if subtype_or_site:
            context = problem.setdefault("context", {})
            if target_profile == "shared_fracture":
                context.setdefault("fracture_site", subtype_or_site)
            elif target_profile == "shared_muscle_myotendinous":
                context.setdefault("muscle_group", subtype_or_site)
            elif target_profile == "shared_deconditioning_balance_gait":
                context.setdefault("functional_route_id", subtype_or_site)
        draft["body_region"] = target_profile

    def validate(self, raw_draft: Mapping[str, Any]) -> CU1ValidationResponse:
        draft = self.normalize(raw_draft)
        errors: List[CU1ValidationError] = []
        self._validate_contract_version(draft, errors)
        self._validate_primary_route(draft, errors)
        self._validate_options(draft, errors)
        self._validate_safety_flags(draft, errors)
        safety_results = self._evaluate_rules(draft, errors)
        hard_validation_block = any(error.error_class == "validation_error" for error in errors)
        formatter_blocked = hard_validation_block or any(item.formatter_blocked for item in safety_results)
        highest = None
        if safety_results:
            highest = max(safety_results, key=lambda item: SEVERITY_ORDER.get(item.severity, -1)).severity
        return CU1ValidationResponse(
            normalized_draft=draft,
            validation_errors=errors,
            safety_results=safety_results,
            highest_severity=highest,
            formatter_blocked=formatter_blocked,
        )

    def generate(self, raw_draft: Mapping[str, Any], mode: str = "short") -> CU1GenerateResponse:
        result = self.validate(raw_draft)
        text = None if result.formatter_blocked else self._format(result.normalized_draft, mode)
        return CU1GenerateResponse(**result.model_dump(), mode=mode, text=text)

    def _validate_contract_version(self, draft: Mapping[str, Any], errors: List[CU1ValidationError]) -> None:
        if draft.get("contract_version") != CONTRACT_VERSION:
            self._add_error(errors, "invalid_route_or_subtype", {"path": "contract_version", "expected": CONTRACT_VERSION})

    def _validate_primary_route(self, draft: Mapping[str, Any], errors: List[CU1ValidationError]) -> None:
        req = self.bundle.route_requirements
        for path in _deep_get(req, "base_requirements.required_fields", []) or []:
            if _missing(draft, path):
                self._add_error(errors, "required_field_missing", {"path": path})

        problem = draft.get("primary_problem")
        if not isinstance(problem, Mapping):
            return
        profile_id = problem.get("profile_id")
        route_id = problem.get("route_id")
        registry_profile = _deep_get(self.bundle.registry, f"profiles.{profile_id}") if profile_id else None
        route = _deep_get(self.bundle.registry, f"profiles.{profile_id}.routes.{route_id}") if profile_id and route_id else None
        if not isinstance(registry_profile, Mapping) or not isinstance(route, Mapping):
            self._add_error(errors, "invalid_route_or_subtype", {"profile_id": profile_id, "route_id": route_id})
            return

        body_region = draft.get("body_region")
        if body_region and body_region != profile_id:
            self._add_error(errors, "invalid_route_or_subtype", {"path": "body_region", "expected_profile_id": profile_id, "value": body_region})

        wording_mode = problem.get("wording_mode")
        allowed_modes = route.get("wording_modes", [])
        if wording_mode not in allowed_modes:
            self._add_error(errors, "invalid_route_or_subtype", {"path": "primary_problem.wording_mode", "allowed": allowed_modes, "value": wording_mode})

        laterality = problem.get("laterality")
        laterality_values = _deep_get(req, "base_requirements.laterality_values", []) or []
        if laterality not in laterality_values:
            self._add_error(errors, "invalid_context_enum_value", {"path": "primary_problem.laterality", "value": laterality})

        route_override = _deep_get(req, f"route_overrides.{route_id}", {}) or {}
        wording_policy = _deep_get(req, f"wording_mode_requirements.{wording_mode}", {}) or {}
        formal_policy = route_override.get("formal_assertion_policy") or wording_policy.get("formal_assertion_policy")
        if wording_mode == "formal_diagnosis" and formal_policy != "context_based_by_functional_route":
            if problem.get("formal_assertion_state_optional") != "yes":
                self._add_error(errors, "formal_diagnosis_assertion_required", {"path": "primary_problem.formal_assertion_state_optional"})

        self._validate_subtype(problem, route_id, wording_mode, errors)
        self._apply_policy_block(draft, wording_policy, errors)
        self._apply_policy_block(draft, route_override, errors)
        self._validate_shared_context(draft, profile_id, route_id, errors)
        self._validate_context_keys_and_values(draft, profile_id, route_id, errors)

    def _validate_subtype(self, problem: Mapping[str, Any], route_id: str, wording_mode: Any, errors: List[CU1ValidationError]) -> None:
        policy = _deep_get(self.bundle.route_requirements, f"subtype_policies.{route_id}")
        subtype = problem.get("subtype_id_optional")
        if not isinstance(policy, Mapping):
            return
        policy_name = policy.get("policy")
        if policy_name == "required" and not subtype:
            self._add_error(errors, str(policy.get("error_if_missing", "subtype_required")), {"path": "primary_problem.subtype_id_optional"})
        if policy_name == "optional_unless_formal_or_structural_wording" and wording_mode in {"formal_diagnosis", "established_structural_diagnosis"} and not subtype:
            conditional = policy.get("conditional_requirements", []) or []
            error_id = "subtype_required"
            if conditional and isinstance(conditional[0], Mapping):
                error_id = str(conditional[0].get("error", error_id))
            self._add_error(errors, error_id, {"path": "primary_problem.subtype_id_optional"})
        if subtype:
            allowed = _deep_get(self.bundle.registry, f"subtypes.{route_id}")
            if isinstance(allowed, list) and subtype not in allowed:
                self._add_error(errors, "invalid_route_or_subtype", {"path": "primary_problem.subtype_id_optional", "value": subtype})

    def _apply_policy_block(self, draft: Mapping[str, Any], policy: Mapping[str, Any], errors: List[CU1ValidationError]) -> None:
        if not isinstance(policy, Mapping):
            return
        if policy.get("apply_policy") == "postoperative_context":
            self._apply_postoperative_context(draft, errors)
        require = policy.get("require")
        if isinstance(require, list):
            error_map = policy.get("errors", {}) if isinstance(policy.get("errors"), Mapping) else {}
            for path in require:
                if _missing(draft, path):
                    if path.endswith("established_diagnosis_source"):
                        error_id = error_map.get("diagnosis_source_missing", "required_field_missing")
                    elif path.endswith("management_context"):
                        error_id = error_map.get("management_missing", "required_field_missing")
                    else:
                        error_id = policy.get("error", "required_field_missing")
                    self._add_error(errors, str(error_id), {"path": path})
        if "allowed_established_diagnosis_sources" in policy:
            value = _deep_get(draft, "primary_problem.context.established_diagnosis_source")
            if value is not None and value not in policy["allowed_established_diagnosis_sources"]:
                self._add_error(errors, "invalid_context_enum_value", {"path": "primary_problem.context.established_diagnosis_source", "value": value})
        if "allowed_management_context" in policy:
            value = _deep_get(draft, "primary_problem.context.management_context")
            if value is not None and value not in policy["allowed_management_context"]:
                self._add_error(errors, "invalid_context_enum_value", {"path": "primary_problem.context.management_context", "value": value})
        for forbidden in policy.get("forbidden_context_values", []) or []:
            if not isinstance(forbidden, Mapping):
                continue
            value = _deep_get(draft, str(forbidden.get("path", "")))
            if value in (forbidden.get("values") or []):
                self._add_error(errors, str(policy.get("error", "invalid_route_or_subtype")), {"path": forbidden.get("path"), "value": value})
        for conditional in policy.get("conditional_requirements", []) or []:
            if not isinstance(conditional, Mapping):
                continue
            when = conditional.get("when")
            if isinstance(when, Mapping) and self._eval_expr(when, draft, errors):
                if conditional.get("apply_policy") == "postoperative_context":
                    self._apply_postoperative_context(draft, errors)
                for path in conditional.get("require", []) or []:
                    if _missing(draft, path):
                        self._add_error(errors, str(conditional.get("error", "required_field_missing")), {"path": path})
                allowed_values = conditional.get("allowed_values")
                if isinstance(allowed_values, list):
                    for path in conditional.get("require", []) or []:
                        value = _deep_get(draft, path)
                        if value is not None and value not in allowed_values:
                            self._add_error(errors, "invalid_context_enum_value", {"path": path, "value": value})

    def _apply_postoperative_context(self, draft: Mapping[str, Any], errors: List[CU1ValidationError]) -> None:
        postop = self.bundle.route_requirements.get("postoperative_context", {}) or {}
        for path in postop.get("required_fields", []) or []:
            if _missing(draft, path):
                self._add_error(errors, "required_field_missing", {"path": path})
        for conditional in postop.get("conditional_requirements", []) or []:
            if not isinstance(conditional, Mapping):
                continue
            when = conditional.get("when")
            if not isinstance(when, Mapping) or not self._eval_expr(when, draft, errors):
                continue
            for path in conditional.get("require", []) or []:
                if _missing(draft, path):
                    self._add_error(errors, str(conditional.get("error", "required_field_missing")), {"path": path})
            required_flag = conditional.get("require_safety_input_flag")
            if required_flag:
                flags = _deep_get(draft, "safety.input_flags", []) or []
                if required_flag not in flags:
                    self._add_error(errors, str(conditional.get("error", "required_field_missing")), {"required_safety_input_flag": required_flag})

    def _validate_shared_context(self, draft: Mapping[str, Any], profile_id: str, route_id: str, errors: List[CU1ValidationError]) -> None:
        shared = _deep_get(self.bundle.route_requirements, f"shared_context_requirements.{profile_id}")
        if not isinstance(shared, Mapping) or shared.get("route_id") != route_id:
            return
        for path in shared.get("required_fields", []) or []:
            if _missing(draft, path):
                self._add_error(errors, "required_field_missing", {"path": path})
        for conditional in shared.get("conditional_requirements", []) or []:
            if not isinstance(conditional, Mapping):
                continue
            applies = False
            site_group = conditional.get("when_site_group")
            if site_group:
                site = _deep_get(draft, "primary_problem.context.fracture_site")
                applies = site in _deep_get(shared, f"site_groups.{site_group}", [])
            elif isinstance(conditional.get("when"), Mapping):
                applies = self._eval_expr(conditional["when"], draft, errors)
            if applies:
                for path in conditional.get("require", []) or []:
                    if _missing(draft, path):
                        self._add_error(errors, str(conditional.get("error", "required_field_missing")), {"path": path})

        if profile_id == "shared_deconditioning_balance_gait":
            route_value = _deep_get(draft, "primary_problem.context.functional_route_id")
            wording = _deep_get(draft, "primary_problem.wording_mode")
            route_wording = _deep_get(shared, f"route_specific_wording.{route_value}")
            if isinstance(route_wording, Mapping):
                allowed = route_wording.get("allowed_wording_modes", [])
                if wording not in allowed:
                    self._add_error(errors, "invalid_route_or_subtype", {"path": "primary_problem.wording_mode", "allowed": allowed, "value": wording})
                for path in route_wording.get("require", []) or []:
                    if _missing(draft, path):
                        self._add_error(errors, str(route_wording.get("error", "required_field_missing")), {"path": path})
                required_values = route_wording.get("require_value", {})
                if isinstance(required_values, Mapping):
                    for path, expected in required_values.items():
                        if _deep_get(draft, path) != expected:
                            self._add_error(errors, str(route_wording.get("error", "invalid_context_enum_value")), {"path": path, "expected": expected})

    def _validate_context_keys_and_values(self, draft: Mapping[str, Any], profile_id: str, route_id: str, errors: List[CU1ValidationError]) -> None:
        context = _deep_get(draft, "primary_problem.context", {})
        if not isinstance(context, Mapping):
            self._add_error(errors, "invalid_context_key", {"path": "primary_problem.context"})
            return
        canonical = self.bundle.route_requirements.get("canonical_context_keys", {}) or {}
        allowed_keys = set(canonical.get("common", []) or [])
        if profile_id in canonical:
            allowed_keys.update(canonical.get(profile_id, []) or [])
        for key in context:
            if key not in allowed_keys:
                self._add_error(errors, "invalid_context_key", {"path": f"primary_problem.context.{key}"})

        shared = _deep_get(self.bundle.route_requirements, f"shared_context_requirements.{profile_id}")
        if isinstance(shared, Mapping) and shared.get("route_id") == route_id:
            for key, source in (shared.get("value_sets") or {}).items():
                if key not in context or _is_empty(context.get(key)):
                    continue
                allowed = self._resolve_value_set_source(str(source))
                if allowed is not None and context.get(key) not in allowed:
                    self._add_error(errors, "invalid_context_enum_value", {"path": f"primary_problem.context.{key}", "value": context.get(key)})

        if _deep_get(self.bundle.route_requirements, f"route_overrides.{route_id}.apply_policy") == "postoperative_context" or _deep_get(draft, "primary_problem.wording_mode") == "postoperative":
            postop = self.bundle.route_requirements.get("postoperative_context", {}) or {}
            for key, source in (postop.get("value_sets") or {}).items():
                value = context.get(key)
                if value is None:
                    continue
                allowed = self._resolve_value_set_source(str(source))
                if allowed is not None and value not in allowed:
                    self._add_error(errors, "invalid_context_enum_value", {"path": f"primary_problem.context.{key}", "value": value})

    def _resolve_value_set_source(self, source: str) -> Optional[List[str]]:
        if source.startswith("cu1_context_value_sets_v1."):
            path = source.split(".", 1)[1]
            value = _deep_get(self.bundle.context_values, path)
            return value if isinstance(value, list) else None
        if source == "cu1_route_detail_catalog_v1.shared_fracture_site_ids":
            value = self.bundle.route_detail_catalog.get("shared_fracture_site_ids")
            return value if isinstance(value, list) else None
        if source == "cu1_route_detail_catalog_v1.shared_muscle_group_ids":
            value = self.bundle.route_detail_catalog.get("shared_muscle_group_ids")
            return value if isinstance(value, list) else None
        if source == "cu1_route_detail_catalog_v1.shared_deconditioning_route_ids":
            value = self.bundle.route_detail_catalog.get("shared_deconditioning_route_ids")
            return value if isinstance(value, list) else None
        return None

    def _validate_options(self, draft: Mapping[str, Any], errors: List[CU1ValidationError]) -> None:
        options = self.bundle.options
        profile_id = _deep_get(draft, "primary_problem.profile_id")
        valid_findings = set(options.get("common_findings", []) or [])
        valid_findings.update(_deep_get(options, f"profile_findings.{profile_id}", []) or [])
        self._validate_selection_ids(draft.get("findings", []), valid_findings, errors, "findings")
        self._validate_selection_ids(draft.get("functional_impairments", []), set(options.get("common_functional_impairments", []) or []), errors, "functional_impairments")
        self._validate_selection_ids(draft.get("goals", []), set(options.get("common_goal_ids", []) or []), errors, "goals")
        self._validate_selection_ids(draft.get("rehab_directions", []), set(options.get("common_rehab_direction_ids", []) or []), errors, "rehab_directions")
        self._validate_selection_ids(draft.get("adjunct_options", []), set(options.get("adjunct_ids", []) or []), errors, "adjunct_options")
        self._validate_selection_ids(draft.get("measurements", []), set(options.get("measurement_ids", []) or []), errors, "measurements")
        self._validate_selection_ids(draft.get("explicit_restrictions", []), set(options.get("restriction_ids", []) or []), errors, "explicit_restrictions")

    def _validate_selection_ids(self, values: Any, allowed: set[str], errors: List[CU1ValidationError], path: str) -> None:
        if values is None:
            return
        if not isinstance(values, list):
            self._add_error(errors, "invalid_route_or_subtype", {"path": path})
            return
        for index, item in enumerate(values):
            canonical_id = _canonical_id_from_selection(item)
            if not canonical_id or canonical_id not in allowed:
                self._add_error(errors, "invalid_route_or_subtype", {"path": f"{path}[{index}]", "value": canonical_id})

    def _validate_safety_flags(self, draft: Mapping[str, Any], errors: List[CU1ValidationError]) -> None:
        flags = _deep_get(draft, "safety.input_flags", []) or []
        allowed = set(_deep_get(self.bundle.rules, "safety_input_flags.allowed_ids", []) or [])
        if not isinstance(flags, list):
            self._add_error(errors, "invalid_safety_input_flag", {"path": "safety.input_flags"})
            return
        for flag in flags:
            if flag not in allowed:
                self._add_error(errors, "invalid_safety_input_flag", {"value": flag})

    def _evaluate_rules(self, draft: Mapping[str, Any], errors: List[CU1ValidationError]) -> List[CU1SafetyResult]:
        results: List[CU1SafetyResult] = []
        profile_id = str(_deep_get(draft, "primary_problem.profile_id", ""))
        route_id = str(_deep_get(draft, "primary_problem.route_id", ""))
        safety = draft.get("safety", {}) if isinstance(draft.get("safety"), Mapping) else {}
        acknowledged_ids = set(safety.get("acknowledged_rule_ids", []) or [])
        disposition = str(safety.get("clinician_disposition", "none_recorded") or "none_recorded")
        for rule_id, rule in (self.bundle.rules.get("rules") or {}).items():
            if not isinstance(rule, Mapping) or not self._rule_applies(rule, profile_id, route_id):
                continue
            trigger = rule.get("trigger")
            if not isinstance(trigger, Mapping) or not self._eval_expr(trigger, draft, errors):
                continue
            severity_ref = str(rule.get("severity_ref"))
            severity_spec = _deep_get(self.bundle.options, f"safety_rules.{severity_ref}")
            if not isinstance(severity_spec, Mapping):
                raise CU1ContractError(f"Unknown severity_ref {severity_ref}")
            severity = str(severity_spec.get("severity"))
            disposition_required = bool(severity_spec.get("disposition_required", False))
            acknowledgement_required = severity in {"hard_warning_ack_required", "block_until_disposition", "urgent_reassessment"}
            acknowledged = rule_id in acknowledged_ids
            blocked = self._safety_result_blocked(severity, acknowledged, disposition_required, disposition)
            results.append(
                CU1SafetyResult(
                    rule_id=str(rule_id),
                    severity=severity,
                    message_key=str(rule_id),
                    acknowledgement_required=acknowledgement_required,
                    acknowledged=acknowledged,
                    disposition_required=disposition_required,
                    formatter_blocked=blocked,
                    clinician_disposition=disposition,
                    source_profile_id=profile_id,
                    source_route_id_optional=route_id or None,
                )
            )
        return results

    def _rule_applies(self, rule: Mapping[str, Any], profile_id: str, route_id: str) -> bool:
        applies = rule.get("applies_to", {})
        if not isinstance(applies, Mapping):
            return False
        profiles = applies.get("profiles", "all")
        routes = applies.get("routes", "all")
        profile_ok = profiles == "all" or profile_id in (profiles or [])
        route_ok = routes == "all" or route_id in (routes or [])
        return profile_ok and route_ok

    def _eval_expr(self, expr: Mapping[str, Any], draft: Mapping[str, Any], errors: Sequence[CU1ValidationError]) -> bool:
        if "all" in expr:
            return all(self._eval_expr(item, draft, errors) for item in expr["all"])
        if "any" in expr:
            return any(self._eval_expr(item, draft, errors) for item in expr["any"])
        if "not" in expr:
            return not self._eval_expr(expr["not"], draft, errors)
        if "eq" in expr:
            spec = expr["eq"]
            return _deep_get(draft, str(spec.get("path"))) == spec.get("value")
        if "in" in expr:
            spec = expr["in"]
            return _deep_get(draft, str(spec.get("path"))) in (spec.get("values") or [])
        if "contains" in expr:
            spec = expr["contains"]
            canonical_id = spec.get("canonical_id")
            return _contains(_deep_get(draft, str(spec.get("path"))), str(canonical_id))
        if "missing" in expr:
            return _missing(draft, str(expr["missing"]))
        if "empty" in expr:
            return _is_empty(_deep_get(draft, str(expr["empty"])))
        if "validation_error" in expr:
            target = str(expr["validation_error"])
            return any(error.error_id == target for error in errors)
        if "validation_error_class" in expr:
            target = str(expr["validation_error_class"])
            return any(error.error_class == target for error in errors)
        raise CU1ContractError(f"Unsupported CU-1 expression DSL: {expr}")

    def _safety_result_blocked(self, severity: str, acknowledged: bool, disposition_required: bool, disposition: str) -> bool:
        if severity in {"info", "soft_warning"}:
            return False
        if severity == "hard_warning_ack_required":
            return not acknowledged
        if severity == "block_until_disposition":
            return disposition == "none_recorded"
        if severity == "urgent_reassessment":
            return disposition not in ALLOWED_URGENT_DISPOSITIONS
        if disposition_required:
            return disposition == "none_recorded"
        return False

    def _add_error(self, errors: List[CU1ValidationError], error_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        error_class = _deep_get(self.bundle.validation_policy, f"error_to_class.{error_id}", "validation_error")
        candidate = CU1ValidationError(error_id=error_id, error_class=str(error_class), metadata=metadata or {})
        if candidate.model_dump() not in [existing.model_dump() for existing in errors]:
            errors.append(candidate)

    def _problem_label(self, problem: Mapping[str, Any]) -> str:
        profile_id = str(problem.get("profile_id", ""))
        route_id = str(problem.get("route_id", ""))
        label = self.bundle.profile_route_labels.get(profile_id, {}).get(route_id, _humanize_id(route_id))
        laterality = LATERALITY_LABELS.get(str(problem.get("laterality", "")), "")
        if laterality:
            label = f"{label} — {laterality}"
        subtype = problem.get("subtype_id_optional")
        if subtype:
            label += f" ({_humanize_id(subtype)})"
        return label

    def _format(self, draft: Mapping[str, Any], mode: str) -> str:
        problem = draft.get("primary_problem", {}) if isinstance(draft.get("primary_problem"), Mapping) else {}
        lines: List[str] = ["Παραπομπή για φυσιοθεραπεία", "", f"Κύριο πρόβλημα: {self._problem_label(problem)}"]
        self._append_selection_line(lines, draft, "findings", SECTION_LABELS["findings"], detailed=mode == "detailed")
        self._append_selection_line(lines, draft, "functional_impairments", SECTION_LABELS["functional_impairments"], detailed=True)
        self._append_selection_line(lines, draft, "explicit_restrictions", SECTION_LABELS["explicit_restrictions"], detailed=True)
        self._append_selection_line(lines, draft, "precautions", SECTION_LABELS["precautions"], detailed=True)
        self._append_selection_line(lines, draft, "goals", SECTION_LABELS["goals"], detailed=True)
        self._append_selection_line(lines, draft, "rehab_directions", SECTION_LABELS["rehab_directions"], detailed=True)
        self._append_selection_line(lines, draft, "adjunct_options", SECTION_LABELS["adjunct_options"], detailed=True)

        if mode == "detailed":
            context = problem.get("context", {}) if isinstance(problem.get("context"), Mapping) else {}
            visible_context = [(key, value) for key, value in context.items() if not _is_empty(value) and key != "neurological_screen"]
            if visible_context:
                lines.append("Κλινικό / δομικό πλαίσιο: " + "; ".join(f"{_humanize_id(key)}: {_humanize_id(value)}" for key, value in visible_context))
            neuro = context.get("neurological_screen")
            if isinstance(neuro, Mapping):
                rendered = [f"{_humanize_id(key)}: {_humanize_id(value)}" for key, value in neuro.items() if value not in {None, "not_assessed", "not_stated"}]
                if rendered:
                    lines.append("Νευρολογικός έλεγχος: " + "; ".join(rendered))
            self._append_selection_line(lines, draft, "measurements", SECTION_LABELS["measurements"], detailed=True)
            secondary = draft.get("secondary_problems", [])
            if isinstance(secondary, list) and secondary:
                labels = [self._problem_label(item) for item in secondary if isinstance(item, Mapping)]
                if labels:
                    lines.append("Δευτερεύοντα προβλήματα: " + "; ".join(labels))

        clinician_free_text = draft.get("clinician_free_text_optional")
        if isinstance(clinician_free_text, str) and clinician_free_text.strip():
            lines.append("Σημείωση κλινικού: " + _normalize_whitespace(clinician_free_text))

        disposition = _deep_get(draft, "safety.clinician_disposition")
        if disposition and disposition != "none_recorded":
            lines.append("Καταγεγραμμένη διάθεση/ενέργεια ασφάλειας: " + _humanize_id(disposition))

        return "\n".join(line for line in lines if line is not None).strip() + "\n"

    def _append_selection_line(self, lines: List[str], draft: Mapping[str, Any], key: str, label: str, detailed: bool) -> None:
        values = draft.get(key, [])
        if not isinstance(values, list) or not values:
            return
        rendered: List[str] = []
        for item in values:
            if isinstance(item, str):
                rendered.append(_humanize_id(item))
                continue
            if not isinstance(item, Mapping):
                continue
            canonical_id = _canonical_id_from_selection(item)
            if not canonical_id:
                continue
            text = _humanize_id(canonical_id)
            if key == "measurements":
                value = item.get("value")
                unit = item.get("unit_optional")
                if value is not None:
                    text += f": {value}{' ' + str(unit) if unit else ''}"
            elif key == "explicit_restrictions":
                state = item.get("state_or_value")
                source = item.get("source")
                if state is not None:
                    text += f": {_humanize_id(state)}"
                if detailed and source:
                    text += f" [{_humanize_id(source)}]"
            elif key == "adjunct_options" and item.get("provenance") == "therapist_proposed_context":
                text += " [πρόταση/πλαίσιο φυσιοθεραπευτή]"
            notes = item.get("notes_optional") or item.get("free_text_optional")
            if detailed and isinstance(notes, str) and notes.strip():
                text += f" — {_normalize_whitespace(notes)}"
            rendered.append(text)
        if rendered:
            lines.append(f"{label}: " + "; ".join(rendered))


_BUNDLE: Optional[CU1ContractBundle] = None
_ENGINE: Optional[CU1Engine] = None


def get_cu1_bundle() -> CU1ContractBundle:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = CU1ContractBundle()
    return _BUNDLE


def get_cu1_engine() -> CU1Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CU1Engine(get_cu1_bundle())
    return _ENGINE


def _require_clinical_key(
    x_clinical_key: Optional[str] = Header(default=None, alias="X-Clinical-Key"),
) -> None:
    expected = os.environ.get("CLINICAL_DATA_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Clinical data access is disabled until CLINICAL_DATA_KEY is configured.")
    if not x_clinical_key or not secrets.compare_digest(x_clinical_key, expected):
        raise HTTPException(status_code=401, detail="Invalid clinical data key")


def build_cu1_physio_referral_router() -> APIRouter:
    router = APIRouter(prefix="/clinical/clinic-utilities/physio-referral", tags=["cu1-physio-referral"])
    protected = [Depends(_require_clinical_key)]
    page_path = _repo_root() / "static/clinic-utilities/physio-referral/index.html"

    @router.get("", include_in_schema=False, dependencies=protected)
    def physio_referral_page() -> FileResponse:
        if not page_path.exists():
            raise HTTPException(status_code=500, detail="CU-1 utility page is unavailable")
        return FileResponse(page_path)

    @router.get("/api/contract", dependencies=protected)
    def cu1_contract() -> Dict[str, Any]:
        return get_cu1_bundle().contract_payload()

    @router.post("/api/validate", response_model=CU1ValidationResponse, dependencies=protected)
    def cu1_validate(req: CU1ValidateRequest) -> CU1ValidationResponse:
        try:
            return get_cu1_engine().validate(req.draft)
        except CU1ContractError as exc:
            raise HTTPException(status_code=500, detail=f"CU-1 contract error: {exc}") from exc

    @router.post("/api/generate", response_model=CU1GenerateResponse, dependencies=protected)
    def cu1_generate(req: CU1GenerateRequest) -> CU1GenerateResponse:
        try:
            return get_cu1_engine().generate(req.draft, req.mode)
        except CU1ContractError as exc:
            raise HTTPException(status_code=500, detail=f"CU-1 contract error: {exc}") from exc

    return router


__all__ = [
    "CONTRACT_VERSION",
    "CU1ContractBundle",
    "CU1Engine",
    "CU1ContractError",
    "build_cu1_physio_referral_router",
    "get_cu1_bundle",
    "get_cu1_engine",
]
