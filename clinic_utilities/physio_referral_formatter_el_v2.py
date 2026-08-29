from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional

from clinic_utilities.physio_referral_formatter_el import CU1GreekReferralFormatter as _BaseGreekFormatter
from clinic_utilities.physio_referral_runtime import CU1ContractBundle, CU1ContractError, _normalize_whitespace


_LATERAL_ELBOW_ROUTE = "lateral_elbow_tendinopathy"
_MAX_REFERRAL_CHARS = 2000


class CU1GreekReferralFormatter(_BaseGreekFormatter):
    """Greek formatter v2 plus bounded rich-referral prototype for lateral elbow."""

    def __init__(self, bundle: CU1ContractBundle):
        super().__init__(bundle)
        base_language = copy.deepcopy(bundle.artifacts.get("referral_language_el"))
        corrections = bundle.artifacts.get("referral_language_el_corrections")
        if not isinstance(base_language, dict) or not isinstance(corrections, Mapping):
            raise CU1ContractError("CU-1 Greek referral language composition artifacts are missing")
        for section, values in corrections.items():
            if section in {"version", "language", "status"} or not isinstance(values, Mapping):
                continue
            target = base_language.setdefault(str(section), {})
            if not isinstance(target, dict):
                raise CU1ContractError(f"Invalid Greek language section: {section}")
            for key, value in values.items():
                target[str(key)] = copy.deepcopy(value)
        self.language = base_language

        route_artifact = bundle.artifacts.get("route_labels_el")
        routes = route_artifact.get("routes") if isinstance(route_artifact, Mapping) else None
        if not isinstance(routes, Mapping):
            raise CU1ContractError("CU-1 explicit Greek route-label artifact is missing")
        self.explicit_route_labels: Mapping[str, Any] = routes

    def _route_label(self, profile_id: str, route_id: str) -> str:
        profile = self.explicit_route_labels.get(profile_id)
        label = profile.get(route_id) if isinstance(profile, Mapping) else None
        if not isinstance(label, str) or not label.strip():
            raise CU1ContractError(f"Missing explicit Greek route label for {profile_id}.{route_id}")
        label = _normalize_whitespace(label)
        if not self._is_greek_clinician_phrase(label):
            raise CU1ContractError(f"Invalid Greek route label for {profile_id}.{route_id}: {label}")
        return label

    def contract_route_labels(self) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        profiles = self.bundle.registry.get("profiles", {})
        if not isinstance(profiles, Mapping):
            raise CU1ContractError("CU-1 route registry is missing")
        for profile_id, profile_spec in profiles.items():
            if not isinstance(profile_spec, Mapping):
                continue
            result[str(profile_id)] = {}
            for route_id in (profile_spec.get("routes") or {}):
                result[str(profile_id)][str(route_id)] = self._route_label(str(profile_id), str(route_id))
        return result

    def _format_short(self, draft: Mapping[str, Any]) -> str:
        if self._is_lateral_elbow(draft):
            return self._format_lateral_elbow_short(draft)
        return super()._format_short(draft)

    def _format_detailed(self, draft: Mapping[str, Any]) -> str:
        if self._is_lateral_elbow(draft):
            return self._format_lateral_elbow_detailed(draft)
        return super()._format_detailed(draft)

    @staticmethod
    def _is_lateral_elbow(draft: Mapping[str, Any]) -> bool:
        problem = draft.get("primary_problem")
        return isinstance(problem, Mapping) and problem.get("route_id") == _LATERAL_ELBOW_ROUTE

    def _format_lateral_elbow_short(self, draft: Mapping[str, Any]) -> str:
        problem = draft.get("primary_problem", {}) if isinstance(draft.get("primary_problem"), Mapping) else {}
        problem_label = self._problem_label(problem, include_subtype=False)
        findings = self._selection_labels(draft, "findings", "findings")
        function = self._selection_labels(draft, "functional_impairments", "functional_impairments")
        work_context = self._explicit_work_or_sport_context(draft)
        restrictions = self._restriction_labels(draft, detailed=False)
        precautions = self._optional_selection_labels(draft, "precautions", "precautions")

        parts: List[str] = [f"{problem_label}."]
        if findings:
            parts.append(self._join_greek(findings).capitalize() + ".")
        if function:
            parts.append(self._join_greek(function).capitalize() + ".")
        if work_context:
            parts.append(f"Πλαίσιο δραστηριότητας: {work_context}.")
        if restrictions or precautions:
            parts.append(f"Περιορισμοί/προφυλάξεις: {self._join_greek(restrictions + precautions)}.")

        parts.append(
            "Αρχικά έλεγχος ερεθιστικότητας/φορτίου, ενεργητική κινητοποίηση και ανεκτή ισομετρική φόρτιση "
            "εκτεινόντων καρπού· στη συνέχεια προοδευτική ομόκεντρη/έκκεντρη φόρτιση, δύναμη λαβής και αντοχή."
        )
        parts.append(
            "Τελικός στόχος η λειτουργική επανένταξη. Παθητικά μέσα μόνο επικουρικά· όχι αντί ενεργητικής "
            "αποκατάστασης. Πρόοδος βάσει κλινικής/λειτουργικής ανταπόκρισης, χωρίς καθολικά αριθμητικά κριτήρια."
        )
        return self._bounded_lateral_elbow_text(" ".join(parts))

    def _format_lateral_elbow_detailed(self, draft: Mapping[str, Any]) -> str:
        problem = draft.get("primary_problem", {}) if isinstance(draft.get("primary_problem"), Mapping) else {}
        problem_label = self._problem_label(problem, include_subtype=True)
        findings = self._selection_labels(draft, "findings", "findings")
        function = self._selection_labels(draft, "functional_impairments", "functional_impairments")
        work_context = self._explicit_work_or_sport_context(draft)
        restrictions = self._restriction_labels(draft, detailed=True)
        precautions = self._optional_selection_labels(draft, "precautions", "precautions")

        clinical: List[str] = [problem_label + "."]
        if findings:
            clinical.append(self._join_greek(findings).capitalize() + ".")
        if function:
            clinical.append(self._join_greek(function).capitalize() + ".")
        if work_context:
            clinical.append(f"Πλαίσιο δραστηριότητας: {work_context}.")
        if restrictions or precautions:
            clinical.append(f"Περιορισμοί/προφυλάξεις: {self._join_greek(restrictions + precautions)}.")

        sections = [
            "ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ\n" + " ".join(clinical),
            (
                "ΣΤΑΔΙΟ 1 — ΕΡΕΘΙΣΤΙΚΟΤΗΤΑ / ΑΡΧΙΚΗ ΦΟΡΤΙΣΗ\n"
                "Στόχοι: μείωση πόνου/ερεθιστικότητας, λειτουργική κινητικότητα, έναρξη φόρτισης εκτεινόντων καρπού.\n"
                "Κατευθύνσεις: εκπαίδευση/διαχείριση φορτίου, εργονομική προσαρμογή όπου χρειάζεται, ενεργητική "
                "κινητοποίηση, ανεκτή ισομετρική φόρτιση. Επικουρικά κρυοθεραπεία/TENS ή αναλγητικές/κινητοποιητικές τεχνικές.\n"
                "Πρόοδος: μικρότερη ερεθιστικότητα, καλύτερη ανοχή σε κίνηση, σύλληψη και αρχική φόρτιση."
            ),
            (
                "ΣΤΑΔΙΟ 2 — ΔΥΝΑΜΗ / ΑΝΤΟΧΗ / ΦΟΡΤΙΣΗ\n"
                "Στόχοι: δύναμη λαβής, μυϊκή αντοχή, ανοχή σε επαναλαμβανόμενη χρήση.\n"
                "Κατευθύνσεις: προοδευτική αντιστασιακή φόρτιση εκτεινόντων με ισομετρική, ομόκεντρη και έκκεντρη "
                "κατεύθυνση· σταδιακή αύξηση μηχανικής απαίτησης. Ώμος/ωμοπλάτη μόνο αν υπάρχει σχετικό έλλειμμα.\n"
                "Πρόοδος: καλύτερη δύναμη/αντοχή, φόρτιση και λειτουργία χωρίς παρατεταμένη έξαρση."
            ),
            (
                "ΣΤΑΔΙΟ 3 — ΛΕΙΤΟΥΡΓΙΚΗ ΕΠΑΝΕΝΤΑΞΗ\n"
                "Στόχοι: επάνοδος στις καθημερινές, επαγγελματικές/αθλητικές απαιτήσεις και αυτοδιαχείριση φορτίου.\n"
                "Κατευθύνσεις: προοδευτική έκθεση στις πραγματικές απαιτήσεις, λειτουργική χρήση/λαβή, εργονομική "
                "προσαρμογή και αυτοδιαχείριση.\n"
                "Πρόοδος/ολοκλήρωση: ουσιαστική λειτουργική βελτίωση και ανοχή στο συνήθες φορτίο, χωρίς καθολικά αριθμητικά κριτήρια."
            ),
            (
                "Παθητικά μέσα μόνο επικουρικά· όχι ως υποκατάστατο ενεργητικής προοδευτικής αποκατάστασης. "
                "Ιατρική επανεκτίμηση σε άτυπη πορεία ή νέο/προοδευτικό νευρολογικό, ριζιτικό, μηχανικό ή τραυματικό/ασταθές εύρημα."
            ),
        ]
        return self._bounded_lateral_elbow_text("\n\n".join(sections))

    @staticmethod
    def _bounded_lateral_elbow_text(text: str) -> str:
        text = text.rstrip()
        if len(text) <= _MAX_REFERRAL_CHARS:
            return text
        clipped = text[: _MAX_REFERRAL_CHARS - 1].rstrip()
        for boundary in ("\n\n", ". ", "; "):
            cut = clipped.rfind(boundary)
            if cut >= int(_MAX_REFERRAL_CHARS * 0.8):
                clipped = clipped[: cut + (1 if boundary == ". " else 0)].rstrip()
                break
        return clipped + "…"

    @staticmethod
    def _explicit_work_or_sport_context(draft: Mapping[str, Any]) -> Optional[str]:
        patient_context = draft.get("patient_context")
        if not isinstance(patient_context, Mapping):
            return None
        value = patient_context.get("sport_or_work_demand_optional")
        if not isinstance(value, str) or not value.strip() or "_" in value:
            return None
        value = _normalize_whitespace(value)
        return value if len(value) <= 120 else value[:117].rstrip() + "…"


__all__ = ["CU1GreekReferralFormatter"]