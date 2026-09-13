from __future__ import annotations

from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ReportType = Literal["accident_medical_report", "medico_legal_expert_report"]
ReportIdentityType = Literal["ADT", "ARC", "OTHER"]
SourceType = Literal[
    "gesy_visit", "clinician_note_self", "clinician_note_other", "specialist_report",
    "hospital_record", "emergency_record", "admission_note", "discharge_summary",
    "procedure_note", "imaging_report", "lab_report", "physiotherapy_report",
    "prior_medical_report", "sick_leave_certificate", "prescription", "imaging_referral",
    "specialist_referral", "lab_or_service_referral", "heidi_transcript", "other",
]
EvidenceType = Literal[
    "patient_reported", "clinician_observed", "specialist_opinion", "imaging_finding",
    "lab_finding", "procedure_performed", "treatment_given", "medication",
    "diagnosis_recorded", "functional_limitation", "work_absence", "pre_existing_condition",
    "causation_opinion", "prognostic_opinion", "recommended_future_care", "other",
]
SectionId = Literal[
    "patient_case_details", "purpose_instructions", "sources_reviewed", "incident_history",
    "initial_management", "clinical_course", "current_symptoms", "objective_findings",
    "investigations", "treatment_procedures", "diagnoses", "pre_existing_conditions",
    "causation", "functional_consequences", "work_incapacity", "prognosis",
    "future_needs", "summary_opinion",
]
ExtractionMethod = Literal["text", "visual_ai", "clinician_text", "none"]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class MedicalReportCaseV1(StrictModel):
    report_type: ReportType = "accident_medical_report"
    patient_name: str = Field(min_length=1, max_length=160)
    id_type: ReportIdentityType = "ADT"
    id_number: str = Field(default="", max_length=64)
    birth_date: date | None = None
    occupation: str = Field(default="", max_length=160)
    incident_date: date | None = None
    report_date: date
    instructing_party: str = Field(default="", max_length=200)
    instructing_reference: str = Field(default="", max_length=120)
    purpose_and_questions: str = Field(default="", max_length=6000)
    clinician_context: str = Field(default="", max_length=20000)

    @field_validator("patient_name", "id_number", "occupation", "instructing_party", "instructing_reference", "purpose_and_questions", "clinician_context", mode="before")
    @classmethod
    def strip_text(cls, value):
        return str(value or "").strip()

class SourcePageV1(StrictModel):
    page_number: int = Field(ge=1, le=5000)
    text: str = Field(default="", max_length=120000)

class ReportSourceV1(StrictModel):
    source_id: str = Field(min_length=1, max_length=80)
    filename: str = Field(min_length=1, max_length=255)
    source_type: SourceType = "other"
    status: Literal["extracted", "visual_extracted", "no_extractable_text"]
    extraction_method: ExtractionMethod = "text"
    review_required: bool = False
    page_count: int = Field(ge=1, le=5000)
    character_count: int = Field(ge=0, le=350000)
    pages: list[SourcePageV1] = Field(default_factory=list)
    structured_notes: list[str] = Field(default_factory=list)

class SourceSummaryV1(StrictModel):
    source_id: str = Field(min_length=1, max_length=80)
    proposed_source_type: SourceType = "other"
    document_date: date | None = None
    date_text: str = Field(default="", max_length=120)
    author: str = Field(default="", max_length=200)
    specialty: str = Field(default="", max_length=160)
    institution: str = Field(default="", max_length=200)
    summary: str = Field(default="", max_length=5000)
    warnings: list[str] = Field(default_factory=list)

class EvidenceItemV1(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=100)
    source_id: str = Field(min_length=1, max_length=80)
    page_numbers: list[int] = Field(default_factory=list)
    event_date: date | None = None
    date_text: str = Field(default="", max_length=120)
    evidence_type: EvidenceType
    statement: str = Field(min_length=1, max_length=6000)
    certainty: Literal["documented", "probable", "possible", "uncertain"] = "documented"
    conflict_key: str = Field(default="", max_length=120)
    requires_clinician_review: bool = True

class TimelineEventV1(StrictModel):
    event_id: str = Field(min_length=1, max_length=100)
    event_date: date | None = None
    date_text: str = Field(default="", max_length=120)
    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=6000)
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    conflict_flags: list[str] = Field(default_factory=list)

class WorkAbsenceIntervalV1(StrictModel):
    interval_id: str = Field(min_length=1, max_length=100)
    leave_from: date
    leave_to: date
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = Field(default="", max_length=1000)
    requires_clinician_review: bool = True

    @model_validator(mode="after")
    def validate_interval(self):
        if self.leave_to < self.leave_from:
            raise ValueError("Work-absence interval ends before it starts")
        return self

class ClinicianResolutionV1(StrictModel):
    resolution_id: str = Field(min_length=1, max_length=100)
    topic_or_conflict_key: str = Field(default="", max_length=160)
    clinician_statement: str = Field(min_length=1, max_length=6000)
    related_evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["confirmed"] = "confirmed"

class DiagnosisAnalysisV1(StrictModel):
    diagnosis: str = Field(min_length=1, max_length=500)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    pre_existing_discussion: str = Field(default="", max_length=5000)
    causation_draft: str = Field(default="", max_length=6000)
    alternative_causes: list[str] = Field(default_factory=list)
    uncertainty: str = Field(default="", max_length=3000)
    requires_clinician_review: Literal[True] = True

class ReportSectionDraftV1(StrictModel):
    section_id: SectionId
    title: str = Field(min_length=1, max_length=160)
    draft_text: str = Field(default="", max_length=30000)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    requires_clinician_review: bool = True

class PrognosisQuestionV1(StrictModel):
    diagnosis_or_problem: str = Field(min_length=1, max_length=500)
    question: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(default="", max_length=1500)

class FutureNeedDraftV1(StrictModel):
    problem: str = Field(min_length=1, max_length=500)
    suggested_need: str = Field(min_length=1, max_length=3000)
    basis_evidence_ids: list[str] = Field(default_factory=list)
    uncertainty: str = Field(default="", max_length=1500)
    requires_clinician_review: Literal[True] = True

class MedicalReportAnalysisV1(StrictModel):
    source_summaries: list[SourceSummaryV1] = Field(default_factory=list)
    evidence_items: list[EvidenceItemV1] = Field(default_factory=list)
    timeline: list[TimelineEventV1] = Field(default_factory=list)
    work_absence_intervals: list[WorkAbsenceIntervalV1] = Field(default_factory=list)
    clinician_resolutions: list[ClinicianResolutionV1] = Field(default_factory=list)
    diagnosis_analyses: list[DiagnosisAnalysisV1] = Field(default_factory=list)
    report_sections: list[ReportSectionDraftV1] = Field(default_factory=list)
    prognosis_questions: list[PrognosisQuestionV1] = Field(default_factory=list)
    future_needs: list[FutureNeedDraftV1] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class ProviderUsageV1(StrictModel):
    provider: str = Field(default="openai", max_length=80)
    model: str = Field(default="", max_length=120)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    web_search_calls: int = Field(default=0, ge=0)

class VisualPageExtractionV1(StrictModel):
    page_number: int = Field(ge=1, le=20)
    text: str = Field(default="", max_length=30000)

class VisualExtractionResultV1(StrictModel):
    pages: list[VisualPageExtractionV1] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class ResearchCitationV1(StrictModel):
    title: str = Field(default="", max_length=500)
    url: str = Field(min_length=1, max_length=3000)

class MedicalReportResearchRequestV1(StrictModel):
    case: MedicalReportCaseV1
    analysis: MedicalReportAnalysisV1

class MedicalReportResearchResultV1(StrictModel):
    research_text: str = Field(default="", max_length=30000)
    citations: list[ResearchCitationV1] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    usage: ProviderUsageV1

class MedicalReportRefinementRequestV1(StrictModel):
    case: MedicalReportCaseV1
    analysis: MedicalReportAnalysisV1
    clinician_message: str = Field(min_length=1, max_length=6000)

    @field_validator("clinician_message", mode="before")
    @classmethod
    def strip_message(cls, value):
        return str(value or "").strip()

class MedicalReportRefinementResultV1(StrictModel):
    assistant_reply: str = Field(min_length=1, max_length=8000)
    updated_analysis: MedicalReportAnalysisV1
    proposed_resolutions: list[ClinicianResolutionV1] = Field(default_factory=list)
    usage: ProviderUsageV1

class FinalMedicalReportV1(StrictModel):
    case: MedicalReportCaseV1
    sections: list[ReportSectionDraftV1] = Field(default_factory=list)
    research_text: str = Field(default="", max_length=30000)
    citations: list[ResearchCitationV1] = Field(default_factory=list)
    declaration_text: str = Field(default="", max_length=8000)
    clinician_confirmed: bool = False

    @model_validator(mode="after")
    def require_confirmation(self):
        if not self.clinician_confirmed:
            raise ValueError("Απαιτείται ρητή επιβεβαίωση του ιατρού πριν την τελική έκθεση")
        if not any(section.draft_text.strip() for section in self.sections):
            raise ValueError("Η τελική έκθεση δεν περιέχει κείμενο")
        return self
