from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional

from clinic_utilities.physio_referral_formatter_el import CU1GreekReferralFormatter as _BaseGreekFormatter
from clinic_utilities.physio_referral_runtime import CU1ContractBundle, CU1ContractError, _normalize_whitespace


_LATERAL_ELBOW_ROUTE = "lateral_elbow_tendinopathy"


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
        restrictions = self._restriction_labels(draft, detailed=False)
        precautions = self._optional_selection_labels(draft, "precautions", "precautions")

        sentences: List[str] = [f"Παραπέμπεται για φυσιοθεραπευτική αποκατάσταση λόγω {problem_label}."]
        if findings or function:
            parts: List[str] = []
            if findings:
                parts.append(f"Κλινικά καταγράφονται {self._join_greek(findings)}")
            if function:
                parts.append(f"με λειτουργικό περιορισμό σε {self._join_greek(function)}")
            sentences.append(" ".join(parts).rstrip(".") + ".")

        sentences.append(
            "Η αποκατάσταση να εξελιχθεί ενεργητικά και προοδευτικά: αρχικά έλεγχος ερεθιστικότητας και "
            "τροποποίηση επιβαρυντικού φορτίου, διατήρηση ή αποκατάσταση λειτουργικού εύρους κίνησης και "
            "έναρξη ανεκτής ενεργοποίησης/ισομετρικής φόρτισης των εκτεινόντων του καρπού."
        )
        sentences.append(
            "Στη συνέχεια να προχωρήσει σε προοδευτική αντιστασιακή φόρτιση με ομόκεντρη και έκκεντρη "
            "κατεύθυνση, αποκατάσταση δύναμης λαβής, μυϊκής αντοχής και ανοχής σε επαναλαμβανόμενη χρήση, "
            "με προσθήκη ώμου/ωμοπλάτης μόνο εφόσον διαπιστωθεί σχετικό έλλειμμα."
        )
        sentences.append(
            "Τελικός στόχος είναι η σταδιακή επανένταξη στις καταγεγραμμένες καθημερινές, επαγγελματικές ή "
            "αθλητικές απαιτήσεις. Κρυοθεραπεία/TENS και άλλα κατάλληλα παθητικά μέσα μπορούν να χρησιμοποιούνται "
            "επικουρικά για συμπτωματική ανακούφιση, αλλά δεν υποκαθιστούν την ενεργητική προοδευτική αποκατάσταση."
        )
        if restrictions or precautions:
            sentences.append(
                f"Να τηρηθούν οι καταγεγραμμένοι περιορισμοί/προφυλάξεις: {self._join_greek(restrictions + precautions)}."
            )
        sentences.append(
            "Η πρόοδος να αξιολογείται με την κλινική ανταπόκριση, τη δύναμη/αντοχή λαβής, τη λειτουργική "
            "ικανότητα και κατάλληλα εργαλεία έκβασης, χωρίς προκαθορισμένα καθολικά αριθμητικά κριτήρια μετάβασης."
        )
        return " ".join(sentences)

    def _format_lateral_elbow_detailed(self, draft: Mapping[str, Any]) -> str:
        problem = draft.get("primary_problem", {}) if isinstance(draft.get("primary_problem"), Mapping) else {}
        problem_label = self._problem_label(problem, include_subtype=True)
        findings = self._selection_labels(draft, "findings", "findings")
        function = self._selection_labels(draft, "functional_impairments", "functional_impairments")
        restrictions = self._restriction_labels(draft, detailed=True)
        precautions = self._optional_selection_labels(draft, "precautions", "precautions")
        selected_adjuncts = self._adjunct_labels(draft, detailed=True)
        measurements = self._measurement_labels(draft)
        work_context = self._explicit_work_or_sport_context(draft)

        sections: List[str] = ["ΠΑΡΑΠΟΜΠΗ ΓΙΑ ΦΥΣΙΟΘΕΡΑΠΕΙΑ"]
        clinical: List[str] = [f"Κύριο πρόβλημα: {problem_label}."]
        if findings:
            clinical.append(f"Κλινικά ευρήματα: {self._join_greek(findings)}.")
        if function:
            clinical.append(f"Λειτουργικός περιορισμός: {self._join_greek(function)}.")
        if work_context:
            clinical.append(f"Καταγεγραμμένο επαγγελματικό/αθλητικό πλαίσιο: {work_context}.")
        clinical.extend(self._detailed_context_sentences(problem))
        sections.append("ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ\n" + " ".join(clinical))

        if restrictions or precautions:
            limits: List[str] = []
            if restrictions:
                limits.append(f"Περιορισμοί: {self._join_greek(restrictions)}.")
            if precautions:
                limits.append(f"Προφυλάξεις: {self._join_greek(precautions)}.")
            sections.append("ΠΕΡΙΟΡΙΣΜΟΙ / ΠΡΟΦΥΛΑΞΕΙΣ\n" + " ".join(limits))

        sections.append(
            "ΣΤΑΔΙΟ 1 — ΕΛΕΓΧΟΣ ΕΡΕΘΙΣΤΙΚΟΤΗΤΑΣ, ΚΙΝΗΤΙΚΟΤΗΤΑ ΚΑΙ ΑΡΧΙΚΗ ΦΟΡΤΙΣΗ\n"
            "Στόχοι:\n"
            "• Μείωση πόνου και ερεθιστικότητας ώστε να είναι εφικτή η ενεργητική αποκατάσταση.\n"
            "• Διατήρηση ή αποκατάσταση λειτουργικά επαρκούς ενεργητικού εύρους κίνησης αγκώνα, αντιβραχίου και καρπού, όταν υπάρχει σχετικό έλλειμμα.\n"
            "• Έναρξη αποκατάστασης της ικανότητας φόρτισης των εκτεινόντων του καρπού.\n"
            "• Αναγνώριση και κατάλληλη τροποποίηση των δραστηριοτήτων/φορτίων που επιδεινώνουν τα συμπτώματα, χωρίς περιττή ακινητοποίηση.\n\n"
            "Κατευθύνσεις για την επίτευξη των στόχων:\n"
            "• Εκπαίδευση, διαχείριση φορτίου και εργονομική προσαρμογή όπου είναι σχετική με το πραγματικό λειτουργικό πρόβλημα.\n"
            "• Ενεργητική κινητοποίηση/ROM αγκώνα, αντιβραχίου και καρπού και αντιμετώπιση τυχόν περιορισμένης κινητικότητας ή ευκαμψίας.\n"
            "• Αρχική χαμηλής απαίτησης ενεργοποίηση των εκτεινόντων του καρπού, με ισομετρική φόρτιση ως πρώιμη επιλογή όταν είναι ανεκτή.\n"
            "• Για βραχυπρόθεσμη συμπτωματική ανακούφιση μπορούν, όταν ενδείκνυνται, να χρησιμοποιηθούν κρυοθεραπεία/TENS και επιλεγμένες αναλγητικές ή τεχνικές κινητοποίησης. Αυτές λειτουργούν επικουρικά και δεν υποκαθιστούν την ενεργητική θεραπεία.\n\n"
            "Δείκτες προόδου:\n"
            "• Μείωση της ερεθιστικότητας και καλύτερη ανοχή στις βασικές καθημερινές δραστηριότητες.\n"
            "• Λειτουργικά επαρκές ή βελτιούμενο ενεργητικό εύρος κίνησης, όταν υπήρχε περιορισμός.\n"
            "• Καλύτερη ανοχή στην ενεργητική/ισομετρική φόρτιση και στη σύλληψη χωρίς κλινικά σημαντική παρατεταμένη επιδείνωση.\n\n"
            "Κατεύθυνση μετάβασης:\n"
            "Όταν η κλινική ανταπόκριση το επιτρέπει, η θεραπεία εξελίσσεται προς προοδευτικά μεγαλύτερη αντιστασιακή φόρτιση, δύναμη και αντοχή."
        )

        sections.append(
            "ΣΤΑΔΙΟ 2 — ΑΠΟΚΑΤΑΣΤΑΣΗ ΔΥΝΑΜΗΣ, ΑΝΤΟΧΗΣ ΚΑΙ ΙΚΑΝΟΤΗΤΑΣ ΦΟΡΤΙΣΗΣ\n"
            "Στόχοι:\n"
            "• Προοδευτική αποκατάσταση της ικανότητας των εκτεινόντων του καρπού να δέχονται φορτίο.\n"
            "• Βελτίωση δύναμης λαβής και ανοχής σε επαναλαμβανόμενη χρήση του άνω άκρου.\n"
            "• Αποκατάσταση μυϊκής αντοχής για τις πραγματικές καθημερινές, επαγγελματικές ή αθλητικές απαιτήσεις.\n"
            "• Αντιμετώπιση σχετικών εγγύς ελλειμμάτων μόνο εφόσον αυτά έχουν διαπιστωθεί στην αξιολόγηση.\n\n"
            "Κατευθύνσεις για την επίτευξη των στόχων:\n"
            "• Προοδευτική αντιστασιακή φόρτιση των εκτεινόντων, με εξέλιξη από την αρχική ενεργοποίηση/ισομετρική εργασία προς ομόκεντρη και έκκεντρη φόρτιση, σύμφωνα με την ανοχή και την κλινική ανταπόκριση.\n"
            "• Προοδευτική αποκατάσταση δύναμης λαβής και μυϊκής αντοχής του άνω άκρου.\n"
            "• Σταδιακή αύξηση της μηχανικής απαίτησης και της επαναλαμβανόμενης χρήσης, χωρίς καθολική προκαθορισμένη δοσολογία.\n"
            "• Ενδυνάμωση/νευρομυϊκός έλεγχος ώμου και ωμοπλάτης μόνο όταν υπάρχει αντίστοιχο κλινικό έλλειμμα.\n"
            "• Τεχνικές κινητοποίησης, άκαμπτη περίδεση, dry needling ή άλλα επιλεγμένα συμπληρωματικά μέσα μπορούν να χρησιμοποιηθούν όταν είναι κατάλληλα, αλλά δεν αποτελούν το κύριο περιεχόμενο του σταδίου.\n\n"
            "Δείκτες προόδου:\n"
            "• Βελτίωση της δύναμης/αντοχής και της ικανότητας λαβής σε σχέση με την αρχική αξιολόγηση.\n"
            "• Αυξανόμενη ανοχή στην προοδευτική ομόκεντρη/έκκεντρη φόρτιση.\n"
            "• Μικρότερος λειτουργικός περιορισμός και καλύτερη ανοχή σε επαναλαμβανόμενη χρήση χωρίς δυσανάλογη ή παρατεταμένη έξαρση.\n"
            "• Βελτίωση των δραστηριοτήτων που έχουν καταγραφεί ως σημαντικές για τον ασθενή.\n\n"
            "Κατεύθυνση μετάβασης:\n"
            "Η αποκατάσταση εξελίσσεται προς μεγαλύτερη διάρκεια, μεγαλύτερη λειτουργική απαίτηση και σταδιακή επανένταξη στις πραγματικές απαιτήσεις του ασθενούς."
        )

        stage3_goal = (
            "στις συγκεκριμένες καταγεγραμμένες επαγγελματικές/αθλητικές απαιτήσεις"
            if work_context
            else "στις συγκεκριμένες καθημερινές ή άλλες λειτουργικές απαιτήσεις που έχουν καταγραφεί ως περιορισμένες"
        )
        sections.append(
            "ΣΤΑΔΙΟ 3 — ΛΕΙΤΟΥΡΓΙΚΗ ΚΑΙ ΕΠΑΓΓΕΛΜΑΤΙΚΗ/ΑΘΛΗΤΙΚΗ ΕΠΑΝΕΝΤΑΞΗ\n"
            "Στόχοι:\n"
            f"• Προοδευτική επάνοδος {stage3_goal}.\n"
            "• Αποκατάσταση ανοχής σε παρατεταμένη και επαναλαμβανόμενη χρήση, σύλληψη και φόρτιση του άνω άκρου.\n"
            "• Επαρκής δύναμη και αντοχή για το πραγματικό λειτουργικό φορτίο του ασθενούς.\n"
            "• Αυτοδιαχείριση φορτίου και στρατηγική πρόληψης υποτροπής.\n\n"
            "Κατευθύνσεις για την επίτευξη των στόχων:\n"
            "• Προοδευτικά υψηλότερη μηχανική και λειτουργική απαίτηση, με μεγαλύτερη διάρκεια και επαναληπτικότητα όπου απαιτείται.\n"
            "• Λειτουργική εκπαίδευση λαβής και άνω άκρου και σταδιακή έκθεση στις πραγματικές δραστηριότητες του ασθενούς.\n"
            "• Εργονομική/τεχνική προσαρμογή όπου απαιτείται και σχέδιο ανεξάρτητης συνέχισης/αυτοδιαχείρισης.\n\n"
            "Δείκτες ολοκλήρωσης/προόδου:\n"
            "• Ουσιαστική βελτίωση ή επάνοδος στις δραστηριότητες που αποτελούν προτεραιότητα για τον ασθενή.\n"
            "• Βελτιωμένη δύναμη, αντοχή και λειτουργική ανοχή στη φόρτιση σε σχέση με την αρχική αξιολόγηση.\n"
            "• Το συνήθες λειτουργικό φορτίο δεν προκαλεί δυσανάλογη ή παρατεταμένη επιδείνωση.\n"
            "• Ο ασθενής μπορεί να διαχειρίζεται σταδιακά το φορτίο και πιθανές υποτροπές συμπτωμάτων.\n\n"
            "Ολοκλήρωση:\n"
            "Δεν εφαρμόζεται αυτόματο καθολικό αριθμητικό κριτήριο εξόδου. Η ολοκλήρωση βασίζεται στη λειτουργική αποκατάσταση και στις πραγματικές απαιτήσεις του συγκεκριμένου ασθενούς."
        )

        monitoring = (
            "Η πρόοδος να παρακολουθείται σε σχέση με την αρχική αξιολόγηση με κατάλληλα εργαλεία, όπως PRTEE ή DASH, "
            "PSFS/δραστηριο-ειδικό εργαλείο όταν υπάρχει απαιτητική λειτουργία, καθώς και δύναμη λαβής χωρίς πόνο/μέγιστη "
            "δύναμη λαβής και εύρος κίνησης όταν είναι σχετικά με το κλινικό έλλειμμα. Οι μετρήσεις δεν αποτελούν από μόνες "
            "τους καθολικά κριτήρια μετάβασης ή εξόδου."
        )
        if measurements:
            monitoring += f" Καταγεγραμμένες μετρήσεις στο παραπεμπτικό: {self._join_greek(measurements)}."
        sections.append("ΠΑΡΑΚΟΛΟΥΘΗΣΗ ΠΡΟΟΔΟΥ\n" + monitoring)

        adjunct_text = (
            "Παθητικές ή συμπτωματικές παρεμβάσεις μπορεί να χρησιμοποιούνται επιλεκτικά ως συμπληρωματικά μέσα. "
            "Κρυοθεραπεία/TENS μπορούν να χρησιμοποιηθούν για βραχυπρόθεσμη ανακούφιση σε κατάλληλο κλινικό πλαίσιο, "
            "ενώ το θεραπευτικό υπερηχογράφημα δεν πρέπει να αντιμετωπίζεται ως επαρκής αυτόνομη θεραπεία. "
            "Το βασικό πλάνο αποκατάστασης παραμένει ενεργητικό, προοδευτικό και λειτουργικά προσανατολισμένο."
        )
        if selected_adjuncts:
            adjunct_text += f" Επιπλέον επιλεγμένα συμπληρωματικά μέσα: {self._join_greek(selected_adjuncts)}."
        sections.append("ΡΟΛΟΣ ΣΥΜΠΛΗΡΩΜΑΤΙΚΩΝ ΜΕΣΩΝ\n" + adjunct_text)

        sections.append(
            "ΕΠΑΝΕΚΤΙΜΗΣΗ\n"
            "Απαιτείται ιατρική/διαγνωστική επανεκτίμηση όταν η πορεία είναι ουσιαστικά ασύμβατη με την αναμενόμενη "
            "λειτουργική βελτίωση ή όταν εμφανιστούν προοδευτικό αντικειμενικό νευρολογικό έλλειμμα, σαφές αυχενικό/ριζιτικό "
            "πρότυπο, σημαντικός μηχανικός αποκλεισμός, ουσιώδης τραυματική/ασταθής εικόνα ή άλλο εύρημα που δεν συμβαδίζει "
            "με τη συνήθη αποκατάσταση έξω επικονδυλαλγίας/τενοντοπάθειας."
        )

        note = draft.get("clinician_free_text_optional")
        if isinstance(note, str) and note.strip():
            sections.append("ΙΑΤΡΙΚΗ ΣΗΜΕΙΩΣΗ\n" + _normalize_whitespace(note).rstrip(".") + ".")
        return "\n\n".join(sections)

    @staticmethod
    def _explicit_work_or_sport_context(draft: Mapping[str, Any]) -> Optional[str]:
        patient_context = draft.get("patient_context")
        if not isinstance(patient_context, Mapping):
            return None
        value = patient_context.get("sport_or_work_demand_optional")
        if not isinstance(value, str) or not value.strip() or "_" in value:
            return None
        return _normalize_whitespace(value)


__all__ = ["CU1GreekReferralFormatter"]
