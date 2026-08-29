from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from clinic_utilities.physio_referral_runtime import (
    CU1ContractBundle,
    CU1Engine,
    CU1ValidationError,
    _deep_get,
    _is_empty,
)


class CU1RouteContextEngine(CU1Engine):
    """CU-1 engine extension for closed, data-driven route-specific clinician context.

    The base engine remains authoritative for routing, shared/postoperative context, options and
    safety. This subclass only widens the *closed* context namespace for the selected route using
    cu1_route_context_intake_v1 and validates those values exactly. It never infers a context value.
    """

    @property
    def route_context_intake(self) -> Mapping[str, Any]:
        value = self.bundle.artifacts.get("route_context_intake", {})
        return value if isinstance(value, Mapping) else {}

    def _route_context_spec(self, route_id: str) -> Mapping[str, Any]:
        routes = self.route_context_intake.get("routes", {})
        route = routes.get(route_id) if isinstance(routes, Mapping) else None
        return route if isinstance(route, Mapping) else {}

    def _validate_context_keys_and_values(
        self,
        draft: Mapping[str, Any],
        profile_id: str,
        route_id: str,
        errors: list[CU1ValidationError],
    ) -> None:
        context = _deep_get(draft, "primary_problem.context", {})
        if not isinstance(context, Mapping):
            self._add_error(errors, "invalid_context_key", {"path": "primary_problem.context"})
            return

        canonical = self.bundle.route_requirements.get("canonical_context_keys", {}) or {}
        allowed_keys = set(canonical.get("common", []) or [])
        if profile_id in canonical:
            allowed_keys.update(canonical.get(profile_id, []) or [])

        route_context = self._route_context_spec(route_id)
        route_fields = route_context.get("fields", {}) if isinstance(route_context, Mapping) else {}
        if isinstance(route_fields, Mapping):
            allowed_keys.update(str(key) for key in route_fields)

        for key in context:
            if key not in allowed_keys:
                self._add_error(errors, "invalid_context_key", {"path": f"primary_problem.context.{key}"})

        # Preserve all existing shared-context enum validation.
        shared = _deep_get(self.bundle.route_requirements, f"shared_context_requirements.{profile_id}")
        if isinstance(shared, Mapping) and shared.get("route_id") == route_id:
            for key, source in (shared.get("value_sets") or {}).items():
                if key not in context or _is_empty(context.get(key)):
                    continue
                allowed = self._resolve_value_set_source(str(source))
                if allowed is not None and context.get(key) not in allowed:
                    self._add_error(
                        errors,
                        "invalid_context_enum_value",
                        {"path": f"primary_problem.context.{key}", "value": context.get(key)},
                    )

        if (
            _deep_get(self.bundle.route_requirements, f"route_overrides.{route_id}.apply_policy") == "postoperative_context"
            or _deep_get(draft, "primary_problem.wording_mode") == "postoperative"
        ):
            postop = self.bundle.route_requirements.get("postoperative_context", {}) or {}
            for key, source in (postop.get("value_sets") or {}).items():
                value = context.get(key)
                if value is None:
                    continue
                allowed = self._resolve_value_set_source(str(source))
                if allowed is not None and value not in allowed:
                    self._add_error(
                        errors,
                        "invalid_context_enum_value",
                        {"path": f"primary_problem.context.{key}", "value": value},
                    )

        # Route-specific context values are self-contained and route-scoped.
        if isinstance(route_fields, Mapping):
            for key, field in route_fields.items():
                if not isinstance(field, Mapping):
                    continue
                value = context.get(key)
                if value is None or _is_empty(value):
                    if field.get("required") is True:
                        self._add_error(
                            errors,
                            "required_field_missing",
                            {"path": f"primary_problem.context.{key}"},
                        )
                    continue
                field_type = field.get("type")
                if field_type == "enum":
                    allowed = field.get("values", [])
                    if not isinstance(allowed, list) or value not in allowed:
                        self._add_error(
                            errors,
                            "invalid_context_enum_value",
                            {"path": f"primary_problem.context.{key}", "value": value},
                        )


def route_context_contract_payload(bundle: CU1ContractBundle) -> Dict[str, Any]:
    artifact = bundle.artifacts.get("route_context_intake", {})
    if not isinstance(artifact, Mapping):
        return {"routes": {}}
    routes = artifact.get("routes", {})
    return {"routes": dict(routes) if isinstance(routes, Mapping) else {}}


__all__ = ["CU1RouteContextEngine", "route_context_contract_payload"]
