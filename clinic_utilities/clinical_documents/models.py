from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


IdentityType = Literal["ADT", "ARC"]
RelationType = Literal["extension", "new_leave_same_patient"]


@dataclass(frozen=True)
class ClinicianProfile:
    name: str
    specialty: str
    phone: str
    email: str
    clinic: str = ""
    address: str = ""

    @classmethod
    def from_environment(cls) -> "ClinicianProfile":
        # Preferred future owner for all Clinical Documents clinician identity.
        # During Phase 1, retain backwards compatibility with the already
        # configured RF doctor profile so no new production config is required.
        raw = os.getenv("CLINICAL_DOCUMENTS_CLINICIAN_PROFILE_JSON", "").strip()
        if not raw:
            raw = os.getenv("RF_DOCTOR_PROFILE_JSON", "").strip()
        if not raw:
            raise ValueError("Clinical Documents clinician profile is not configured")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Clinical Documents clinician profile is not valid JSON") from exc
        required = ("name", "specialty", "phone", "email")
        values = {key: str(data.get(key) or "").strip() for key in required}
        if any(not value for value in values.values()):
            raise ValueError("Clinical Documents clinician profile is incomplete")
        return cls(
            **values,
            clinic=str(data.get("clinic") or data.get("medical_center") or "").strip(),
            address=str(data.get("address") or "").strip(),
        )


class PatientIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_name: str = Field(min_length=1, max_length=160)
    id_type: IdentityType
    id_number: str = Field(min_length=1, max_length=64)

    @field_validator("patient_name", "id_number", mode="before")
    @classmethod
    def _strip_text(cls, value):
        return str(value or "").strip()


class SickLeaveDraftV1(PatientIdentity):
    diagnosis: str = Field(min_length=1, max_length=600)
    leave_from: date
    leave_to: date
    issued_on: date
    derived_from_document_id: str = Field(default="", max_length=80)
    relation: RelationType | None = None

    @field_validator("diagnosis", "derived_from_document_id", mode="before")
    @classmethod
    def _strip_more_text(cls, value):
        return str(value or "").strip()

    @model_validator(mode="after")
    def _validate_dates(self):
        if self.leave_to < self.leave_from:
            raise ValueError("Η λήξη της άδειας δεν μπορεί να προηγείται της έναρξης")
        if self.relation and not self.derived_from_document_id:
            raise ValueError("Η σχέση με προηγούμενη άδεια απαιτεί document id")
        return self

    @property
    def inclusive_duration_days(self) -> int:
        return (self.leave_to - self.leave_from).days + 1


class SickLeaveDocumentMetadataV1(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_type: Literal["sick_leave_certificate_v1"] = Field(default="sick_leave_certificate_v1", alias="schema")
    version: Literal["1.0"] = "1.0"
    document_id: str = Field(min_length=36, max_length=80)
    patient_name: str = Field(min_length=1, max_length=160)
    id_type: IdentityType
    id_number: str = Field(min_length=1, max_length=64)
    diagnosis: str = Field(min_length=1, max_length=600)
    leave_from: date
    leave_to: date
    issued_on: date
    derived_from_document_id: str = Field(default="", max_length=80)
    relation: RelationType | None = None

    @field_validator(
        "document_id",
        "patient_name",
        "id_number",
        "diagnosis",
        "derived_from_document_id",
        mode="before",
    )
    @classmethod
    def _strip_metadata_text(cls, value):
        return str(value or "").strip()
