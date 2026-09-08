from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

from clinical_learning.models import AssessmentMethod


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "static" / "clinical-learning" / "index.html"
JS_PATH = ROOT / "static" / "clinical-learning" / "app.js"


class _LearningHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.evidence_methods: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        classes = set(str(values.get("class") or "").split())
        if tag == "input" and "evidence-enabled" in classes and values.get("value"):
            self.evidence_methods.append(str(values["value"]))


class ClinicalLearningL1UiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = HTML_PATH.read_text(encoding="utf-8")
        cls.js = JS_PATH.read_text(encoding="utf-8")
        parser = _LearningHtmlParser()
        parser.feed(cls.html)
        cls.ids = parser.ids
        cls.evidence_methods = parser.evidence_methods

    def test_all_literal_dollar_id_references_exist_in_html(self) -> None:
        referenced = set(re.findall(r"\$\('([^']+)'\)", self.js))
        missing = sorted(referenced - self.ids)
        self.assertEqual(missing, [], f"JavaScript references missing DOM ids: {missing}")
        self.assertNotIn("assessmentJson", self.js)

    def test_history_has_real_new_revision_workflow(self) -> None:
        self.assertIn("$('newRevision').addEventListener('click'", self.js)
        self.assertIn("candidate.supersedes_revision = Number(latest.revision)", self.js)
        self.assertIn("method: revising ? 'PUT' : 'POST'", self.js)
        self.assertIn("revision-choice", self.js)
        self.assertIn("Immutable revision payload", self.js)

    def test_foundation_form_uses_only_frozen_assessment_methods(self) -> None:
        allowed = set(AssessmentMethod.__args__)
        self.assertTrue(self.evidence_methods)
        self.assertEqual(sorted(set(self.evidence_methods) - allowed), [])
        self.assertIn("unaided_explanation", self.evidence_methods)
        self.assertIn("mechanistic_explanation", self.evidence_methods)
        self.assertIn("novel_case_transfer", self.evidence_methods)
        self.assertIn("boundary_or_exception_recognition", self.evidence_methods)
        self.assertIn("evidence_directness_calibration", self.evidence_methods)

    def test_foundation_form_captures_explicit_state_result_and_notes(self) -> None:
        for required_id in (
            "assessmentProposedState",
            "assessmentState",
            "assessmentDue",
            "assessmentEvidence",
            "assessmentNote",
            "previewAssessment",
            "saveAssessment",
        ):
            self.assertIn(required_id, self.ids)
        self.assertIn("clinician_reviewed: true", self.js)
        self.assertIn("source_artifact_type: 'foundation_assessment'", self.js)
        self.assertIn("result,", self.js)
        self.assertIn("note,", self.js)

    def test_fact_ledger_exposes_progressive_disclosure_provenance(self) -> None:
        self.assertIn("introduced via:", self.js)
        self.assertIn("fact.introduced_via", self.js)
        self.assertIn("fact.introduced_at_stage", self.js)


if __name__ == "__main__":
    unittest.main()
