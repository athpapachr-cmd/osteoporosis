from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

from clinical_learning.models import AssessmentMethod


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "static" / "clinical-learning" / "index.html"
JS_PATH = ROOT / "static" / "clinical-learning" / "app.js"
L1B_JS_PATH = ROOT / "static" / "clinical-learning" / "l1b.js"


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
        cls.l1b_js = L1B_JS_PATH.read_text(encoding="utf-8")
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

    def test_clipboard_handoff_is_visible_and_explicit_user_gesture_only(self) -> None:
        self.assertIn("Quick Challenge Handoff", self.l1b_js)
        self.assertIn("Paste & Send Challenge", self.l1b_js)
        self.assertIn("navigator.clipboard?.readText", self.l1b_js)
        self.assertIn("$('clipboardImport')?.addEventListener('click', importFromClipboard)", self.l1b_js)
        self.assertIn("installClipboardHandoffSurface();", self.l1b_js)

    def test_clipboard_handoff_reuses_pending_import_and_preserves_source_event_id(self) -> None:
        self.assertIn("const envelope = { episode: parsed.episode }", self.l1b_js)
        self.assertIn("envelope.source_event_id = parsed.source_event_id", self.l1b_js)
        self.assertIn("envelope.source_event_id = episode.source_event_id.trim()", self.l1b_js)
        self.assertIn("delete episode.source_event_id", self.l1b_js)
        self.assertIn("const body = await api('/api/imports'", self.l1b_js)
        self.assertIn("source_event_id: body.source_event_id", self.l1b_js)
        self.assertIn("openInboxItem(body.import_id)", self.l1b_js)
        self.assertIn("Challenge imported · pending review", self.l1b_js)

    def test_clipboard_handoff_accepts_fenced_json_and_has_manual_fallback(self) -> None:
        self.assertIn("matchAll(/```(?:json)?", self.l1b_js)
        self.assertIn("clipboardFallback", self.l1b_js)
        self.assertIn("clipboardManualJson", self.l1b_js)
        self.assertIn("Send pasted Challenge", self.l1b_js)
        self.assertIn("importFromManualPaste", self.l1b_js)

    def test_clipboard_handoff_does_not_persist_clipboard_content_in_browser_storage(self) -> None:
        storage_operation = r"(?:window\.)?(?:localStorage|sessionStorage)\s*\.\s*(?:setItem|getItem|removeItem|clear|key)\s*\("
        self.assertNotRegex(self.l1b_js, storage_operation)
        self.assertNotIn("CLINICAL_LEARNING_INGEST_KEY", self.l1b_js)


if __name__ == "__main__":
    unittest.main()
