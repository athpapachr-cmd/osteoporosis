from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from clinic_utilities.clinical_documents.report_models import (
    EvidenceItemV1,
    MedicalReportAnalysisV1,
    WorkAbsenceIntervalV1,
)
from clinic_utilities.clinical_documents.report_sources import (
    build_clinician_context_source,
    deterministic_warnings,
    validate_analysis_references,
)


def _source():
    return build_clinician_context_source("Συνθετικές αναρρωτικές περίοδοι για deterministic test")


def _evidence(source_id: str):
    return [
        EvidenceItemV1(
            evidence_id="ev-a",
            source_id=source_id,
            page_numbers=[1],
            evidence_type="work_absence",
            statement="Συνθετική αναρρωτική περίοδος Α",
        ),
        EvidenceItemV1(
            evidence_id="ev-b",
            source_id=source_id,
            page_numbers=[1],
            evidence_type="work_absence",
            statement="Συνθετική αναρρωτική περίοδος Β",
        ),
        EvidenceItemV1(
            evidence_id="ev-c",
            source_id=source_id,
            page_numbers=[1],
            evidence_type="work_absence",
            statement="Συνθετική αναρρωτική περίοδος Γ",
        ),
    ]


def test_work_absence_interval_rejects_inverted_dates():
    with pytest.raises(ValidationError):
        WorkAbsenceIntervalV1(
            interval_id="leave-bad",
            leave_from=date(2026, 9, 10),
            leave_to=date(2026, 9, 9),
        )


def test_work_absence_overlap_and_gap_are_deterministically_flagged():
    source = _source()
    analysis = MedicalReportAnalysisV1(
        evidence_items=_evidence(source.source_id),
        work_absence_intervals=[
            WorkAbsenceIntervalV1(
                interval_id="leave-a",
                leave_from=date(2026, 1, 1),
                leave_to=date(2026, 1, 10),
                source_ids=[source.source_id],
                evidence_ids=["ev-a"],
            ),
            WorkAbsenceIntervalV1(
                interval_id="leave-b",
                leave_from=date(2026, 1, 10),
                leave_to=date(2026, 1, 15),
                source_ids=[source.source_id],
                evidence_ids=["ev-b"],
            ),
            WorkAbsenceIntervalV1(
                interval_id="leave-c",
                leave_from=date(2026, 1, 20),
                leave_to=date(2026, 1, 25),
                source_ids=[source.source_id],
                evidence_ids=["ev-c"],
            ),
        ],
    )
    validate_analysis_references(analysis, [source])
    warnings = deterministic_warnings(analysis, [source])
    assert any("Επικάλυψη αναρρωτικών" in item for item in warnings)
    assert any("Κενό 4 ημερών" in item for item in warnings)


def test_work_absence_references_fail_closed():
    source = _source()
    analysis = MedicalReportAnalysisV1(
        evidence_items=_evidence(source.source_id),
        work_absence_intervals=[
            WorkAbsenceIntervalV1(
                interval_id="leave-forged",
                leave_from=date(2026, 1, 1),
                leave_to=date(2026, 1, 2),
                source_ids=["missing-source"],
                evidence_ids=["ev-a"],
            )
        ],
    )
    with pytest.raises(ValueError, match="Unknown work-absence source"):
        validate_analysis_references(analysis, [source])
