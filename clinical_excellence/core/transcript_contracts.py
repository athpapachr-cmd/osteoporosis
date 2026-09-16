from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

SemanticType = Literal[
    "patient_history_fact", "objective_result", "clinician_interpretation",
    "option_discussed", "clinician_recommendation", "patient_preference",
    "final_decision", "patient_accepted", "patient_declined", "patient_undecided",
    "followup_task", "uncertain_needs_review",
]
Speaker = Literal["patient", "clinician", "third_party", "unclear"]
Polarity = Literal["positive", "negative", "not_applicable", "unclear"]
Temporality = Literal["current", "past", "planned", "future", "relative", "unclear"]
Certainty = Literal["explicit", "probable", "uncertain"]
WarningCode = Literal[
    "LOW_SOURCE_CLARITY", "UNMAPPED_CANDIDATE", "AMBIGUOUS_TARGET",
    "EVIDENCE_NOT_VERIFIABLE", "PARTIAL_EXTRACTION",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranscriptContextV1(StrictModel):
    encounter_archetype: str | None = Field(default=None, max_length=120)


class TranscriptExtractRequestV1(StrictModel):
    schema_version: Literal["clinical_transcript_extract_request_v1"]
    source_type: Literal["heidi_transcript"]
    module: str = Field(min_length=1, max_length=80)
    encounter_phase: Literal["during_visit", "post_visit"] = "during_visit"
    language: Literal["el", "en"] = "el"
    transcript: str = Field(max_length=120_000)
    context: TranscriptContextV1 = Field(default_factory=TranscriptContextV1)


class TextValueV1(StrictModel):
    kind: Literal["text"]
    text: str = Field(min_length=1, max_length=4000)


class CodeValueV1(StrictModel):
    kind: Literal["code"]
    code: str = Field(min_length=1, max_length=160)


class NumberValueV1(StrictModel):
    kind: Literal["number"]
    value: float


class IntegerValueV1(StrictModel):
    kind: Literal["integer"]
    value: int


class BooleanValueV1(StrictModel):
    kind: Literal["boolean"]
    value: bool


class QuantityValueV1(StrictModel):
    kind: Literal["quantity"]
    value: float
    unit: str = Field(min_length=1, max_length=80)


class DateValueV1(StrictModel):
    kind: Literal["date"]
    normalized: str | None = Field(default=None, max_length=10)
    precision: Literal["day", "month", "year", "relative", "unclear"]
    date_text: str = Field(default="", max_length=160)

    @model_validator(mode="after")
    def validate_precision(self):
        value = self.normalized or ""
        if self.precision == "day" and not _matches_date(value, 10, (4, 7), "-"):
            raise ValueError("day precision requires valid YYYY-MM-DD")
        if self.precision == "month" and not _matches_date(value, 7, (4,), "-"):
            raise ValueError("month precision requires valid YYYY-MM")
        if self.precision == "year" and not _matches_year(value):
            raise ValueError("year precision requires valid YYYY")
        if self.precision in {"relative", "unclear"} and self.normalized is not None:
            raise ValueError("relative/unclear date must not contain normalized exact date")
        return self


def _matches_date(value: str, length: int, separator_positions: tuple[int, ...], separator: str) -> bool:
    if len(value) != length:
        return False
    for index, char in enumerate(value):
        if index in separator_positions:
            if char != separator:
                return False
        elif not char.isdigit():
            return False
    try:
        candidate = value if length == 10 else f"{value}-01"
        date.fromisoformat(candidate)
    except ValueError:
        return False
    return True


def _matches_year(value: str) -> bool:
    if len(value) != 4 or not value.isdigit():
        return False
    year = int(value)
    return 1 <= year <= 9999


CandidateValueV1 = Annotated[
    Union[TextValueV1, CodeValueV1, NumberValueV1, IntegerValueV1, BooleanValueV1, QuantityValueV1, DateValueV1],
    Field(discriminator="kind"),
]


class CandidateComponentV1(StrictModel):
    concept_key: str = Field(min_length=1, max_length=160)
    value: CandidateValueV1
    qualifier_code: str | None = Field(default=None, max_length=120)


class SourceAssertionV1(StrictModel):
    speaker: Speaker = "unclear"
    polarity: Polarity = "unclear"
    temporality: Temporality = "unclear"
    normalized_date: str | None = Field(default=None, max_length=10)
    date_precision: Literal["day", "month", "year"] | None = None
    date_text: str = Field(default="", max_length=160)
    certainty: Certainty = "uncertain"

    @model_validator(mode="after")
    def validate_date(self):
        if self.normalized_date is None:
            if self.date_precision is not None:
                raise ValueError("date_precision requires normalized_date")
            return self
        if self.date_precision == "day" and not _matches_date(self.normalized_date, 10, (4, 7), "-"):
            raise ValueError("invalid day date")
        if self.date_precision == "month" and not _matches_date(self.normalized_date, 7, (4,), "-"):
            raise ValueError("invalid month date")
        if self.date_precision == "year" and not _matches_year(self.normalized_date):
            raise ValueError("invalid year date")
        if self.date_precision is None:
            raise ValueError("normalized_date requires date_precision")
        if self.temporality == "relative":
            raise ValueError("relative timing must not be normalized into an exact date")
        return self


class ProviderCandidateV1(StrictModel):
    semantic_type: SemanticType
    components: list[CandidateComponentV1] = Field(min_length=1, max_length=20)
    source_assertion: SourceAssertionV1
    evidence_snippet: str = Field(default="", max_length=320)
    confidence: Literal["high", "medium", "low"] = "medium"

    @model_validator(mode="after")
    def validate_unique_concept_keys(self):
        keys = [component.concept_key for component in self.components]
        if len(keys) != len(set(keys)):
            raise ValueError("concept_key values must be unique within one provider candidate")
        return self


class ProviderTranscriptExtractionV1(StrictModel):
    schema_version: Literal["clinical_transcript_provider_output_v1"] = "clinical_transcript_provider_output_v1"
    candidates: list[ProviderCandidateV1] = Field(default_factory=list, max_length=100)
    warnings: list[Literal["LOW_SOURCE_CLARITY", "PARTIAL_EXTRACTION"]] = Field(default_factory=list, max_length=20)


class TargetMappingV1(StrictModel):
    component_keys: list[str] = Field(default_factory=list)
    target_path: str | None = Field(default=None, max_length=240)
    status: Literal["mapped", "ambiguous", "unmapped"]
    reason_code: str = Field(min_length=1, max_length=120)
    proposed_value: Any = None


class TranscriptCandidateV1(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=80)
    semantic_type: SemanticType
    components: list[CandidateComponentV1]
    source_assertion: SourceAssertionV1
    evidence_snippet: str = Field(default="", max_length=320)
    confidence: Literal["high", "medium", "low"]
    target_mappings: list[TargetMappingV1] = Field(default_factory=list)
    warnings: list[WarningCode] = Field(default_factory=list)
    requires_clinician_review: Literal[True] = True
    status: Literal["proposed"] = "proposed"


class TranscriptResponseMetaV1(StrictModel):
    processing_mode: Literal["ephemeral_preview"] = "ephemeral_preview"
    candidate_count: int = Field(ge=0, le=100)
    raw_persisted: Literal[False] = False
    candidates_persisted: Literal[False] = False
    authoritative_write: Literal[False] = False


class TranscriptCandidatesResponseV1(StrictModel):
    schema_version: Literal["clinical_transcript_candidates_v1"] = "clinical_transcript_candidates_v1"
    request_id: str = Field(min_length=1, max_length=80)
    source_type: Literal["heidi_transcript"] = "heidi_transcript"
    module: str = Field(min_length=1, max_length=80)
    encounter_phase: Literal["during_visit", "post_visit"]
    language: Literal["el", "en"]
    candidates: list[TranscriptCandidateV1]
    warnings: list[WarningCode] = Field(default_factory=list)
    meta: TranscriptResponseMetaV1
