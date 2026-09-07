from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ChallengeMode = Literal["synthetic", "deidentified_real_case", "mixed"]
FactScope = Literal[
    "real_deidentified_case_fact",
    "synthetic_case_fact",
    "clinician_hypothesis",
    "ai_inference",
    "counterfactual_teaching_point",
]
FactIntroducedVia = Literal[
    "initial_case",
    "progressive_disclosure",
    "clinician_entry",
    "heidi_transcript_candidate",
    "accepted_encounter_data",
    "evidence_review",
]
FactStatus = Literal["active", "corrected", "withdrawn"]
Certainty = Literal["high", "moderate", "low", "unknown"]
ObservationCategory = Literal[
    "strength",
    "clear_error",
    "defensible_disagreement",
    "evidence_gap",
    "blind_spot",
    "reasoning_pattern",
    "clinical_insight",
    "uncertainty",
    "missed_opportunity",
    "communication_pattern",
]
ObservationImportance = Literal["low", "moderate", "high", "safety_critical"]
ClinicianDisposition = Literal["pending", "accepted", "modified", "dismissed"]
RecordReviewState = Literal["imported_pending_review", "clinician_reviewed"]
GapClass = Literal["knowledge", "reasoning", "execution", "communication_system"]
EvidenceType = Literal[
    "randomized_trial",
    "systematic_review_meta_analysis",
    "guideline",
    "consensus_position_statement",
    "cohort",
    "case_control",
    "diagnostic_accuracy",
    "narrative_review",
    "mechanistic_or_translational",
    "other",
]
EvidenceRelation = Literal["supports", "challenges", "contextualizes", "mixed"]
ReferenceVerificationState = Literal[
    "unverified", "verified_locator", "verified_content", "invalid_or_unresolved"
]
LearningActionType = Literal[
    "foundation_block",
    "targeted_reading",
    "retrieval_test",
    "transfer_case",
    "clinical_challenge",
    "red_team",
    "deliberate_practice",
    "workflow_change",
    "communication_exercise",
    "reassessment",
    "other",
]
LearningActionStatus = Literal["planned", "completed", "dismissed"]
FoundationState = Literal[
    "FORMAL_SOLID", "INTUITIVE_UNSTRUCTURED", "FRAGMENTED", "UNKNOWN_UNTESTED"
]
AssessmentMethod = Literal[
    "unaided_explanation",
    "mechanistic_explanation",
    "novel_case_transfer",
    "boundary_or_exception_recognition",
    "evidence_directness_calibration",
    "retention_retrieval",
    "real_case_execution",
    "self_rating_only",
]
AssessmentResult = Literal["demonstrated", "partial", "not_demonstrated", "not_assessed"]
RetentionState = Literal[
    "not_scheduled", "scheduled", "due", "retained", "needs_refresh", "insufficient_evidence"
]
DueStatus = Literal[
    "not_scheduled", "scheduled", "due", "overdue", "completed", "deferred", "not_applicable"
]
DeliveryMode = Literal["visible", "shadow_hidden", "safety_only"]


class LearningFactV1(StrictModel):
    fact_id: UUID
    statement: str = Field(min_length=1)
    fact_scope: FactScope
    introduced_via: FactIntroducedVia
    authoritative_for_patient: Literal[False]
    source: str = Field(min_length=1)
    certainty: Certainty | None = None
    introduced_at_stage: str = Field(min_length=1)
    status: FactStatus
    supersedes_fact_id: UUID | None = None


class ProgressiveDisclosureV1(StrictModel):
    disclosure_id: UUID
    sequence: int = Field(ge=1)
    label: str | None = None
    narrative: str | None = None
    fact_ids: list[UUID] = Field(default_factory=list)
    released_after_response_id: UUID | None = None


class ClinicianReasoningResponseV1(StrictModel):
    response_id: UUID
    stage: Literal["initial", "followup", "final"]
    text: str = Field(min_length=1)
    confidence_percent: int | None = Field(default=None, ge=0, le=100)
    created_at: datetime | None = None


class LearningReferenceV1(StrictModel):
    reference_id: UUID
    title: str = Field(min_length=1)
    evidence_type: EvidenceType
    framework_or_guideline: str | None = None
    pmid: str | None = None
    doi: str | None = None
    url: str | None = None
    relation: EvidenceRelation
    verification_state: ReferenceVerificationState = "unverified"
    verification_note: str | None = None


class LearningObservationV1(StrictModel):
    observation_id: UUID
    category: ObservationCategory
    statement: str = Field(min_length=1)
    importance: ObservationImportance
    linked_fact_ids: list[UUID] = Field(default_factory=list)
    linked_reference_ids: list[UUID] = Field(default_factory=list)
    gap_classes: list[GapClass] = Field(default_factory=list)
    clinician_disposition: ClinicianDisposition
    clinician_modified_statement: str | None = None
    disposition_note: str | None = None


class LearningActionV1(StrictModel):
    action_id: UUID
    action_type: LearningActionType
    title: str = Field(min_length=1)
    rationale: str | None = None
    foundation_node_ids: list[str] = Field(default_factory=list)
    reference_ids: list[UUID] = Field(default_factory=list)
    due_on: date | None = None
    status: LearningActionStatus
    completed_at: datetime | None = None


class ChallengePrivacyV1(StrictModel):
    contains_direct_identifiers: Literal[False]
    deidentification_attested: bool
    source_case_deidentified: bool | None = None


class ClinicalLearningChallengeV1(StrictModel):
    challenge_id: UUID
    schema_version: Literal["clinical_learning_challenge_v1"]
    revision: int = Field(ge=1)
    supersedes_revision: int | None = None
    module: Literal["osteoporosis"]
    created_at: datetime
    title: str = Field(min_length=1)
    challenge_mode: ChallengeMode
    topics: list[str] = Field(min_length=1)
    foundation_node_ids: list[str] = Field(default_factory=list)
    difficulty_label: str | None = None
    initial_case: str = Field(min_length=1)
    fact_ledger: list[LearningFactV1]
    progressive_disclosures: list[ProgressiveDisclosureV1] = Field(default_factory=list)
    reasoning_responses: list[ClinicianReasoningResponseV1] = Field(min_length=1)
    final_clinician_decision: str | None = None
    observations: list[LearningObservationV1]
    references: list[LearningReferenceV1]
    gap_classes: list[GapClass] = Field(default_factory=list)
    learning_actions: list[LearningActionV1]
    next_challenge_topic: str | None = None
    spaced_repetition_due: date | None = None
    linked_signal_ids: list[str] = Field(default_factory=list)
    record_review_state: RecordReviewState
    reviewed_at: datetime | None = None
    privacy: ChallengePrivacyV1


class FoundationAssessmentEvidenceV1(StrictModel):
    evidence_id: UUID
    method: AssessmentMethod
    result: AssessmentResult
    clinician_reviewed: bool
    note: str | None = None
    source_artifact_type: Literal[
        "foundation_assessment", "challenge", "daily_case_review", "practice_review"
    ]
    source_artifact_id: str | None = None


class FoundationAssessmentAttemptV1(StrictModel):
    attempt_id: UUID
    schema_version: Literal["foundation_assessment_attempt_v1"]
    module: Literal["osteoporosis"]
    foundation_node_id: str = Field(min_length=1)
    assessed_at: datetime
    evidence: list[FoundationAssessmentEvidenceV1] = Field(min_length=1)
    proposed_state: FoundationState
    clinician_final_state: FoundationState
    clinician_note: str | None = None


class ReferenceVerificationRequest(StrictModel):
    verification_state: ReferenceVerificationState
    verification_note: str | None = None


class ChallengeSaveEnvelope(StrictModel):
    challenge: ClinicalLearningChallengeV1
    confirm_save: bool


class ChallengePreviewEnvelope(StrictModel):
    challenge: ClinicalLearningChallengeV1


class DeleteConfirmation(StrictModel):
    confirm_delete: bool


class FoundationAssessmentEnvelope(StrictModel):
    attempt: FoundationAssessmentAttemptV1
    next_review_due: date | None = None
    confirm_save: bool = False


class FoundationDomainStateV1(StrictModel):
    foundation_node_id: str
    schema_version: Literal["foundation_domain_state_v1"] = "foundation_domain_state_v1"
    module: Literal["osteoporosis"] = "osteoporosis"
    state: FoundationState
    retention_state: RetentionState
    evidence_attempt_ids: list[UUID] = Field(default_factory=list)
    last_assessed_at: datetime | None = None
    next_review_due: date | None = None
    linked_signal_ids: list[str] = Field(default_factory=list)
    clinician_note: str | None = None


class LearningDueStateV1(StrictModel):
    due_item_id: UUID
    schema_version: Literal["learning_due_state_v1"] = "learning_due_state_v1"
    item_type: Literal[
        "challenge_repetition", "foundation_reassessment", "daily_case_review", "progress_review", "learning_action"
    ]
    target_id: str | None = None
    due_on: date | None = None
    due_status: DueStatus
    delivery_mode: DeliveryMode
    reason_code: str = Field(min_length=1)
    completed_at: datetime | None = None
    deferred_until: date | None = None
