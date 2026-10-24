"""Phase-A synthetic ClinicalAttention contract tests. No external providers or PHI."""
from uuid import uuid4
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select,func
from clinical_data import ClinicalPendingORM
from test_cockpit_synthetic_161 import setup,application,login,PREFIX,SYNTHETIC_161_PATIENT_IDS


def token():return str(uuid4())


def test_synthetic_inbox_requires_existing_key_and_scopes_accounts(setup):
    app,engine=setup
    anonymous=TestClient(app,base_url="https://testserver")
    assert anonymous.get(PREFIX+"/attention").status_code==401
    assert anonymous.get(PREFIX+"/attention/SYN-ATT-MRI").status_code==401
    assert anonymous.post(PREFIX+"/attention/SYN-ATT-MRI/action",json={
        "action":"mark_reviewed","expected_version":0,"idempotency_key":token()
    }).status_code==401
    c=login(app);data=c.get(PREFIX+"/attention").json()
    assert data["mode"]=="synthetic_only" and not data["live_provider_connected"]
    assert data["coverage"]=="synthetic_fixture_only"
    assert len(data["accounts"])==3 and len(data["items"])==3
    assert len({i["logical_source_item_key"] for i in data["items"]})==3
    assert len({i["provider_native_message_id"] for i in data["items"]})==1
    mixed=c.get(PREFIX+"/attention/SYN-ATT-MIXED").json()["item"]
    assert mixed["source_subject_scope"]=="mixed_subjects"
    assert mixed["candidate_summary"] is None and mixed["patient_id"] is None
    # A mixed document cannot be promoted to one patient just by changing its scope.
    assert c.post(PREFIX+"/attention/SYN-ATT-MIXED/synthetic-revision",json={
        "expected_version":0,"idempotency_key":token(),"new_revision":"r2",
        "subject_scope":"single_subject"
    }).status_code==409
    assert c.get(PREFIX+"/attention/brief/outside-synthetic-scope").status_code==404
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(ClinicalPendingORM))==0


def test_synthetic_review_link_revision_replay_and_mixed_document_fail_closed(setup):
    app,engine=setup
    a,b=login(app),login(app)
    assert a.post(PREFIX+"/prepare").status_code==200
    endpoint=PREFIX+"/attention/SYN-ATT-MRI"
    payload={"action":"confirm_link","expected_version":0,"idempotency_key":token(),
             "patient_id":SYNTHETIC_161_PATIENT_IDS[1],
             "verified_dob":"1972-01-01","independently_verified":True}
    assert a.post(endpoint+"/action",json={**payload,"idempotency_key":token(),
                                            "verified_dob":"1970-01-01"}).status_code==409
    result=a.post(endpoint+"/action",json=payload)
    assert result.status_code==200,result.text
    before=result.json()["item"]
    assert before["patient_id"]==SYNTHETIC_161_PATIENT_IDS[1] and before["version"]==1
    assert a.post(endpoint+"/action",json=payload).json()["replayed"] is True
    assert a.post(endpoint+"/action",json={**payload,"verified_dob":"1971-01-01"}).status_code==409
    assert b.post(endpoint+"/action",json={"action":"mark_reviewed","expected_version":0,
                                           "idempotency_key":token()}).status_code==409
    mark={"action":"mark_reviewed","expected_version":1,"idempotency_key":token()}
    assert b.post(endpoint+"/action",json=mark).status_code==200
    assert b.get(PREFIX+"/attention/brief/"+SYNTHETIC_161_PATIENT_IDS[1]).json()["items"][0]["review_state"]=="reviewed"
    assert b.get(PREFIX+"/attention/brief/"+SYNTHETIC_161_PATIENT_IDS[0]).json()["items"]==[]
    assert b.post(PREFIX+"/attention/SYN-ATT-MIXED/action",json={
        **payload,"idempotency_key":token()}).status_code==409
    revision={"expected_version":2,"idempotency_key":token(),
              "new_revision":"r2","subject_scope":"mixed_subjects"}
    changed=a.post(endpoint+"/synthetic-revision",json=revision)
    assert changed.status_code==200,changed.text
    row=changed.json()["item"]
    assert row["item_id"]==before["item_id"]
    assert row["logical_source_item_key"]==before["logical_source_item_key"]
    assert row["source_receipt_version_key"]!=before["source_receipt_version_key"]
    assert row["review_state"]=="needs_re_review" and row["patient_id"] is None
    assert row["candidate_summary"] is None and row["disposition"]=="undecided"
    assert a.post(endpoint+"/synthetic-revision",json=revision).json()["replayed"]
    assert a.post(endpoint+"/action",json={
        "action":"no_communication","expected_version":2,"idempotency_key":token()
    }).status_code==409
    assert a.get(PREFIX+"/attention/brief/"+SYNTHETIC_161_PATIENT_IDS[1]).json()["items"]==[]
    with Session(engine) as db:
        assert [p.status for p in db.scalars(select(ClinicalPendingORM)).all()]==["awaiting_patient"]*3


def test_synthetic_attention_is_ephemeral_no_new_persistence(setup):
    app,engine=setup
    a=login(app)
    assert a.post(PREFIX+"/attention/SYN-ATT-LAB/action",json={
        "action":"mark_reviewed","expected_version":0,"idempotency_key":token()
    }).status_code==200
    other=login(application(engine))
    clean=other.get(PREFIX+"/attention/SYN-ATT-LAB").json()["item"]
    assert clean["version"]==0 and clean["review_state"]=="unreviewed"
