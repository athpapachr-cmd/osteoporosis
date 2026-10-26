"""Private Priority 3 tables. Production DDL is installed by an admin migration."""

from sqlalchemy import Column, DateTime, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase


class OfficeCaseBase(DeclarativeBase):
    pass


class SurgeryDetailORM(OfficeCaseBase):
    __tablename__ = "clinical_surgery_details"

    # Foreign keys are established by the reviewed PostgreSQL migration. These
    # existing owner tables live in other SQLAlchemy metadata registries.
    surgery_id = Column(String, primary_key=True)
    rf_application_id = Column(String, unique=True)
    room_entry_time = Column(String(5))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class AccidentCaseORM(OfficeCaseBase):
    __tablename__ = "clinical_accident_cases"

    id = Column(String(36), primary_key=True)
    patient_name = Column(String(200), nullable=False)
    phone = Column(String(40), nullable=False)
    accident_date = Column(String(10), nullable=False)
    visit_date = Column(String(10))
    writing_started_date = Column(String(10))
    report_sent_date = Column(String(10))
    amount_due_cents = Column(Integer)
    revision = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class AccidentCaseAuditORM(OfficeCaseBase):
    __tablename__ = "clinical_accident_case_audit"

    id = Column(String(36), primary_key=True)
    case_id = Column(String(36), nullable=False, index=True)
    revision = Column(Integer, nullable=False)
    action = Column(String(30), nullable=False)
    before_json = Column(JSON, nullable=False)
    after_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False)


class AccidentPaymentORM(OfficeCaseBase):
    __tablename__ = "clinical_accident_payments"
    __table_args__ = (UniqueConstraint("case_id", "payment_request_id", name="uq_accident_payment_request"),)

    id = Column(String(36), primary_key=True)
    case_id = Column(String(36), nullable=False, index=True)
    payment_request_id = Column(String(36), nullable=False)
    amount_cents = Column(Integer, nullable=False)
    paid_on = Column(String(10), nullable=False)
    payer = Column(String(200), nullable=False)
    created_at = Column(DateTime, nullable=False)
    voided_at = Column(DateTime)
    void_reason = Column(String(500))
