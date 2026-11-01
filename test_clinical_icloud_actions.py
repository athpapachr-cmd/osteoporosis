"""Real pilot API contracts tested with entirely synthetic IMAP/clinical fixtures."""
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import clinical_icloud_mail as imap
from clinical_icloud_actions import (
    MailDecisionORM, MailDecisionEventORM, MailDecisionBase,
    build_icloud_actions_router,
)
from clinical_data import ClinicalBase, PatientORM, EncounterORM, ClinicalPendingORM, build_clinical_router
from clinical_calendar import CalendarBase, ClinicalAppointmentORM
from test_clinical_icloud_mail import FakeMail, mail

ALIAS = "practice.alias@icloud.com"
PATIENT = "synthetic-test-person-a"
PENDING = "synthetic-test-pending-a"
VISIT = "synthetic-test-visit-a"
BOOKING = "synthetic-test-appointment-b"
NOW = datetime(2026, 10, 30)
AUTH = {"X-Clinical-Key": "synthetic-clinical-key"}
ORIGIN = {"Origin": "https://testserver"}


@pytest.fixture
def scenario(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", AUTH["X-Clinical-Key"])
    monkeypatch.setenv("ICLOUD_CLINICAL_READ_ENABLED", "1")
    monkeypatch.setenv("ICLOUD_CLINICAL_IMAP_USERNAME", "synthetic-icloud-account")
    monkeypatch.setenv("ICLOUD_CLINICAL_APP_PASSWORD", "synthetic-test-password")
    monkeypatch.setenv("ICLOUD_CLINICAL_ALIAS", ALIAS)
    mailbox = FakeMail()
    mailbox.rows["10"] = mail(
        subject="MRI Synthetic Patient Alpha",
        sender="Radiology Unit <radiology@example.invalid>",
    )
    mailbox.rows["11"] = mail(
        subject="Booking notice", sender="hello@cal.com",
    )
    mailbox.rows["12"] = mail(
        subject="Laboratory results for Synthetic Patient Alpha",
        sender="lab@example.invalid",
    )
    @contextmanager
    def pretend(_username, _password):
        yield mailbox, "765"
    monkeypatch.setattr(imap, "_session", pretend)
    engine = create_engine("sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread":False}, poolclass=StaticPool)
    ClinicalBase.metadata.create_all(engine)
    CalendarBase.metadata.create_all(engine)
    # Synthetic fixture DDL is explicit; production router startup is read-only.
    MailDecisionBase.metadata.create_all(engine)
    with Session(engine) as db, db.begin():
        db.add(PatientORM(
            patient_id=PATIENT, demographics_json={
                "full_name":"Synthetic Patient Alpha", "phone":"+35799000000",
            }, created_at=NOW, updated_at=NOW,
        ))
        db.add(PatientORM(
            patient_id="synthetic-test-person-b",
            demographics_json={"full_name":"Synthetic Patient Beta"},
            created_at=NOW, updated_at=NOW,
        ))
        db.add(EncounterORM(
            id=VISIT, patient_id=PATIENT, encounter_date="2026-10-30",
            status="completed", payload_json={"_visit_capture_v1":{
                "signed":True, "signed_at":"2026-10-30T10:00:00Z",
                "pending_ids":[PENDING],
                "candidate":{"pending":[{"item_type":"imaging"}]},
            }}, created_at=NOW, updated_at=NOW,
        ))
        db.add(ClinicalPendingORM(
            id=PENDING, patient_id=PATIENT, source_encounter_id=VISIT,
            item_type="imaging", description="MRI Synthetic Test",
            trigger_text="Await MRI", status="awaiting_result",
            responsible_role="patient", provenance_json=["synthetic-fixture"],
            created_at=NOW, updated_at=NOW,
        ))
        db.add(ClinicalAppointmentORM(
            id=BOOKING, source="test", source_appointment_id=BOOKING,
            start_at=NOW, end_at=NOW, duration_minutes=30,
            clinic="Synthetic clinic", category="other",
            patient_display_name="Synthetic Patient Beta", phone_e164="+35799111111",
            linked_patient_id=None, label="test",
            comment="",status="scheduled",updated_at=NOW,
        ))
    app=FastAPI()
    app.include_router(imap.build_icloud_mail_router(engine))
    app.include_router(build_icloud_actions_router(engine))
    app.include_router(build_clinical_router(engine))
    with TestClient(app, base_url="https://testserver") as client:
        yield client,mailbox,engine
    engine.dispose()


def endpoint(item="765-10"):
    return "/clinical/icloud-actions/items/"+item


def get(client, url):
    return client.get(url,headers=AUTH)


def post(client,item,action,version,**extra):
    return client.post(endpoint(item),
        headers={**AUTH,**ORIGIN},
        json={"action":action,"expected_version":version,
              "request_id":str(uuid4()),**extra})


def test_read_and_candidates_are_private_and_nonmutating(scenario):
    client,mailbox,dbengine=scenario
    assert client.get(endpoint()).status_code==401
    data=get(client,endpoint())
    assert data.status_code==200
    assert data.json()["category"]=="clinical"
    assert data.json()["patient_id"] is None
    result=get(client,endpoint()+"/candidates")
    assert result.status_code==200
    assert any(x["id"]==PATIENT and x["kind"]=="patient"
               for x in result.json()["items"])
    q=get(client,endpoint()+"/candidates?q=Beta")
    assert q.status_code==200
    assert any(x["id"]==BOOKING for x in q.json()["items"])
    assert get(client,endpoint()+"/candidates?q=x").status_code==422
    with Session(dbengine) as db:
        assert db.scalar(select(func.count()).select_from(MailDecisionORM))==0
        assert db.scalar(select(func.count()).select_from(MailDecisionEventORM))==0
    assert not any(op in ("STORE","EXPUNGE") for op,_ in mailbox.operations)


def test_full_one_doctor_mri_followthrough_does_not_resolve_original_pending(scenario):
    client,mailbox,dbengine=scenario
    original=get(client,"/clinical/icloud-mail/items/765-10")
    assert original.status_code==200
    assert original.json()["item"]["attachments"]==[]
    r=post(client,"765-10","link",0,patient_id=PATIENT,pending_id=PENDING)
    assert r.status_code==200,r.text
    assert r.json()["patient_id"]==PATIENT and r.json()["pending_id"]==PENDING
    assert r.json()["version"]==1
    chart=get(client,"/clinical/clinical-updates?scope=real&active_only=true")
    assert chart.status_code==200,chart.text
    linked=[x for x in chart.json()["items"] if x["pending_id"]==PENDING]
    assert len(linked)==1 and linked[0]["mail_followthrough"][0]["item_id"]=="765-10"
    r=post(client,"765-10","review_notify",1)
    assert r.status_code==200 and r.json()["contact_status"]=="to_contact"
    r=post(client,"765-10","contact_attempt",2)
    assert r.status_code==200 and r.json()["contact_status"]=="attempted"
    assert r.json()["callback_complete"] is False
    r=post(client,"765-10","patient_notified",3)
    assert r.status_code==200 and r.json()["contact_status"]=="notified"
    assert r.json()["callback_complete"] is False
    r=post(client,"765-10","callback_complete",4)
    assert r.status_code==200 and r.json()["callback_complete"] is True
    result=get(client,endpoint()).json()
    assert result["version"]==5 and result["callback_complete"] is True
    with Session(dbengine) as db:
        p=db.get(ClinicalPendingORM,PENDING)
        e=db.get(EncounterORM,VISIT)
        assert p.status=="awaiting_result"
        assert e.status=="completed" and e.payload_json["_visit_capture_v1"]["signed"] is True
        events=db.scalars(select(MailDecisionEventORM)).all()
        assert len(events)==5
        raw=" ".join(str(vars(event)) for event in events)
        assert "MRI Synthetic Patient Alpha" not in raw
        assert "synthetic-test-password" not in raw
    assert not any(op in ("STORE","EXPUNGE") for op,_ in mailbox.operations)


def test_manual_admin_classification_persists_and_does_not_create_patient(scenario):
    client,mailbox,engine=scenario
    before=get(client,endpoint("765-11")).json()
    assert before["category"]=="administrative"
    r=post(client,"765-11","classify",0,category="other")
    assert r.status_code==200 and r.json()["category"]=="other"
    assert get(client,endpoint("765-11")).json()["category"]=="other"
    assert "765-11" not in {
        x["id"] for x in get(client,"/clinical/icloud-mail?view=clinical").json()["items"]
    }
    assert any(x["id"]=="765-11" and x["category"]=="other"
               for x in get(client,"/clinical/icloud-mail?view=all").json()["items"])
    with Session(engine) as db:
        item=db.scalar(select(MailDecisionORM).where(MailDecisionORM.message_ref=="765-11"))
        assert item is not None and item.patient_id is None
        assert db.scalar(select(func.count()).select_from(PatientORM))==2


def test_stale_and_conflicting_replays_fail_closed(scenario):
    client,mailbox,engine=scenario
    payload={"action":"classify","expected_version":0,
             "request_id":str(uuid4()),"category":"clinical"}
    first=client.post(endpoint(),json=payload,headers={**AUTH,**ORIGIN})
    assert first.status_code==200 and first.json()["version"]==1
    same=client.post(endpoint(),json=payload,headers={**AUTH,**ORIGIN})
    assert same.status_code==200 and same.json()==first.json()
    altered={**payload,"category":"administrative"}
    assert client.post(endpoint(),json=altered,headers={**AUTH,**ORIGIN}).status_code==409
    assert post(client,"765-10","classify",0,category="administrative").status_code==409
    assert get(client,endpoint()).json()["category"]=="clinical"
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(MailDecisionEventORM))==1


def test_weak_cross_patient_link_is_denied_and_unlinked_review_cannot_close(scenario):
    client,_,engine=scenario
    assert post(client,"765-10","link",0,
                patient_id="synthetic-test-person-b",pending_id=PENDING).status_code==409
    assert post(client,"765-10","review_notify",0).status_code==409
    r=post(client,"765-10","link",0,patient_id=PATIENT)
    assert r.status_code==200
    assert post(client,"765-10","patient_notified",1).status_code==409
    assert post(client,"765-10","callback_complete",1).status_code==409
    assert post(client,"765-10","review_no_contact",1).status_code==200
    assert post(client,"765-10","review_notify",2).status_code==409


def test_real_appointment_without_chart_never_fabricates_one(scenario):
    client,_,engine=scenario
    r=post(client,"765-12","link",0,booking_id=BOOKING)
    assert r.status_code==200
    assert r.json()["patient_id"] is None and r.json()["pending_id"] is None
    r=post(client,"765-12","review_no_contact",1)
    assert r.status_code==200
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM))==1
        assert db.scalar(select(func.count()).select_from(PatientORM))==2


def test_explicit_origin_uid_and_mailbox_alias_must_qualify(scenario):
    client,mailbox,engine=scenario
    response=client.post(endpoint(),json={"action":"classify",
        "expected_version":0,"request_id":str(uuid4()),"category":"clinical"},headers=AUTH)
    assert response.status_code==403
    assert get(client,endpoint("999-10")).status_code==409
    assert get(client,endpoint("bad-input")).status_code==404
    mailbox.rows["13"]=mail(to="someoneelse@icloud.com",subject="MRI")
    assert get(client,endpoint("765-13")).status_code==404
    assert post(client,"765-13","classify",0,category="clinical").status_code==404


def test_cancelled_pending_revokes_prior_action_authority(scenario):
    client,mailbox,engine=scenario
    assert post(client,"765-10","link",0,patient_id=PATIENT,pending_id=PENDING).status_code==200
    with Session(engine) as db,db.begin():
        db.get(ClinicalPendingORM,PENDING).status="cancelled"
    r=get(client,endpoint())
    assert r.status_code==200
    assert r.json()["source_valid"] is False
    assert r.json()["review_status"]=="source_invalidated"
    assert post(client,"765-10","review_notify",1).status_code==409


def test_phone_and_reversed_name_in_attachment_are_suggestions_not_auto_links(scenario):
    from email.message import EmailMessage
    client, mailbox, engine = scenario
    msg=EmailMessage()
    msg["From"]="Radiology Unit <radiology@example.invalid>"
    msg["To"]=ALIAS
    msg["Subject"]=""
    msg["Date"]="Fri, 30 Oct 2026 12:00:00 +0200"
    msg.set_content("Contact number +35799000000")
    msg.add_attachment(b"synthetic JPG",maintype="image",subtype="jpeg",
                       filename="Alpha_Patient_Synthetic.jpg")
    mailbox.rows["21"]=msg.as_bytes()
    result=get(client,endpoint("765-21")+"/candidates")
    assert result.status_code==200,result.text
    suggested=[x for x in result.json()["items"] if x["id"]==PATIENT]
    assert len(suggested)==1
    assert suggested[0]["evidence"]=="name_phone"
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(MailDecisionORM))==0


def test_manual_reclassification_revokes_prior_review_and_notification(scenario):
    client, _, engine = scenario
    assert post(client,"765-10","link",0,patient_id=PATIENT,pending_id=PENDING).status_code==200
    assert post(client,"765-10","review_notify",1).status_code==200
    assert post(client,"765-10","patient_notified",2).status_code==200
    corrected=post(client,"765-10","classify",3,category="administrative")
    assert corrected.status_code==200
    assert corrected.json()["review_status"]=="not_reviewed"
    assert corrected.json()["contact_status"]=="not_requested"
    assert corrected.json()["callback_complete"] is False
    assert post(client,"765-10","callback_complete",4).status_code==409
    assert post(client,"765-10","review_notify",4).status_code==409
    assert get(client,endpoint()).json()["category"]=="administrative"


def test_signed_pending_membership_amendment_invalidates_mail_link(scenario):
    client, _, engine = scenario
    assert post(client,"765-10","link",0,patient_id=PATIENT,pending_id=PENDING).status_code==200
    with Session(engine) as db,db.begin():
        visit=db.get(EncounterORM,VISIT)
        payload=dict(visit.payload_json)
        signed=dict(payload["_visit_capture_v1"])
        signed["pending_ids"]=[]
        signed["candidate"]={"pending":[]}
        payload["_visit_capture_v1"]=signed
        visit.payload_json=payload
    refreshed=get(client,endpoint())
    assert refreshed.status_code==200
    assert refreshed.json()["source_valid"] is False
    assert post(client,"765-10","review_notify",1).status_code==409


def test_booking_identity_change_invalidates_link(scenario):
    client, _, engine = scenario
    assert post(client, "765-12", "link", 0, booking_id=BOOKING).status_code == 200
    with Session(engine) as db, db.begin():
        booking = db.get(ClinicalAppointmentORM, BOOKING)
        booking.patient_display_name = "New Synthetic Booking Name"
    assert get(client, endpoint("765-12")).json()["source_valid"] is False
    assert post(client, "765-12", "review_no_contact", 1).status_code == 409


def test_unsigned_membership_fails_closed_while_chart_link_stays_usable(scenario):
    client, _, engine = scenario
    with Session(engine) as db, db.begin():
        visit = db.get(EncounterORM, VISIT)
        payload = dict(visit.payload_json)
        signed = dict(payload["_visit_capture_v1"])
        signed.pop("pending_ids")
        payload["_visit_capture_v1"] = signed
        visit.payload_json = payload
    assert post(client, "765-10", "link", 0, patient_id=PATIENT, pending_id=PENDING).status_code == 409
    result = post(client, "765-10", "link", 0, patient_id=PATIENT)
    assert result.status_code == 200
    assert result.json()["pending_id"] is None


def test_doctor_can_correct_review_without_erasing_original_contact_history(scenario):
    client, _, engine = scenario
    assert post(client, "765-10", "link", 0, patient_id=PATIENT, pending_id=PENDING).status_code == 200
    assert post(client, "765-10", "review_notify", 1).status_code == 200
    assert post(client, "765-10", "patient_notified", 2).status_code == 200
    corrected = post(client, "765-10", "reopen_review", 3)
    assert corrected.status_code == 200
    assert corrected.json()["review_status"] == "not_reviewed"
    assert corrected.json()["contact_status"] == "not_requested"
    assert corrected.json()["callback_complete"] is False
    assert post(client, "765-10", "callback_complete", 4).status_code == 409
    assert post(client, "765-10", "review_no_contact", 4).status_code == 200
    with Session(engine) as db:
        history = db.scalars(select(MailDecisionEventORM).order_by(MailDecisionEventORM.version)).all()
        assert [e.action for e in history] == [
            "link", "review_notify", "patient_notified", "reopen_review", "review_no_contact"
        ]


def test_cancelled_pending_allows_doctor_to_correct_mail_link(scenario):
    client, _, engine = scenario
    assert post(client, "765-10", "link", 0, patient_id=PATIENT, pending_id=PENDING).status_code == 200
    with Session(engine) as db, db.begin():
        db.get(ClinicalPendingORM, PENDING).status = "cancelled"
    assert get(client, endpoint()).json()["source_valid"] is False
    correction = post(client, "765-10", "unlink", 1)
    assert correction.status_code == 200, correction.text
    assert correction.json()["source_valid"] is True
    assert correction.json()["pending_id"] is None
    assert post(client, "765-10", "link", 2, patient_id=PATIENT).status_code == 200


def test_resolved_signed_pending_remains_historical_email_source(scenario):
    client, _, engine = scenario
    assert post(client, "765-10", "link", 0, patient_id=PATIENT, pending_id=PENDING).status_code == 200
    assert post(client, "765-10", "review_notify", 1).status_code == 200
    with Session(engine) as db, db.begin():
        db.get(ClinicalPendingORM, PENDING).status = "resolved"
    current = get(client, endpoint())
    assert current.status_code == 200
    assert current.json()["source_valid"] is True
    assert current.json()["review_status"] == "reviewed_notify"
    assert post(client, "765-10", "patient_notified", 2).status_code == 200
