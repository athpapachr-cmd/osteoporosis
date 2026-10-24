"""Phase-A synthetic ClinicalAttention contract tests. No external providers or PHI."""
from uuid import uuid4
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select,func
from clinical_data import ClinicalPendingORM
from test_cockpit_synthetic_161 import setup,application,login,PREFIX,BOOKINGS,SYNTHETIC_161_PATIENT_IDS
from clinical_attention_phase_a import SyntheticAttentionStore,AttentionAction


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
    assert len(data["accounts"])==3 and len(data["items"])==5
    assert len({i["logical_source_item_key"] for i in data["items"]})==5
    assert len({i["provider_native_message_id"] for i in data["items"]})==1
    mixed=c.get(PREFIX+"/attention/SYN-ATT-MIXED").json()["item"]
    assert mixed["source_subject_scope"]=="mixed_subjects"
    assert mixed["candidate_summary"] is None and mixed["patient_id"] is None
    gesy=c.get(PREFIX+"/attention/SYN-ATT-GESY").json()
    assert gesy["item"]["source_type"]=="gesy_notice"
    assert "αποτελέσματα διαθέσιμα" in gesy["item"]["candidate_summary"]
    assert gesy["item"]["review_state"]=="unreviewed"
    assert gesy["verified_source_match"]["patient_id"]==SYNTHETIC_161_PATIENT_IDS[1]
    assert gesy["verified_source_match"]["evidence_kind"]=="verified_referral"
    assert c.get(PREFIX+"/attention/SYN-ATT-NAME").json()["verified_source_match"] is None
    assert c.get(PREFIX+"/attention/SYN-ATT-MIXED").json()["verified_source_match"] is None
    assert "dob" not in c.get(PREFIX+"/attention/eligible-patients").json()["patients"][0]
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
             "evidence_kind":"verified_referral","evidence_ref":"syn-proof-referral-mri-2",
             "independently_verified":True}
    assert a.post(endpoint+"/action",json={**payload,"idempotency_key":token(),
                                            "evidence_ref":"wrong-proof"}).status_code==409
    assert a.post(endpoint+"/action",json={**payload,"idempotency_key":token(),
                                            "patient_id":SYNTHETIC_161_PATIENT_IDS[0]}).status_code==409
    assert a.post(endpoint+"/action",json={**payload,"idempotency_key":token(),
                                            "verified_dob":"1972-01-01"}).status_code==422
    assert a.post(endpoint+"/action",json={**payload,"idempotency_key":token(),
                                            "independently_verified":False}).status_code==422
    assert a.post(PREFIX+"/attention/SYN-ATT-NAME/action",json={
        **payload,"idempotency_key":token()}).status_code==409
    result=a.post(endpoint+"/action",json=payload)
    assert result.status_code==200,result.text
    before=result.json()["item"]
    assert before["patient_id"]==SYNTHETIC_161_PATIENT_IDS[1] and before["version"]==1
    assert a.post(endpoint+"/action",json=payload).json()["replayed"] is True
    assert a.post(endpoint+"/action",json={**payload,"evidence_ref":"other-proof"}).status_code==409
    assert b.post(endpoint+"/action",json={"action":"mark_reviewed","expected_version":0,
                                           "idempotency_key":token()}).status_code==409
    mark={"action":"mark_reviewed","expected_version":1,"idempotency_key":token()}
    assert b.post(endpoint+"/action",json=mark).status_code==200
    assert b.get(PREFIX+"/attention/brief/"+SYNTHETIC_161_PATIENT_IDS[1]).json()["items"][0]["review_state"]=="reviewed"
    # Existing booking owner consumes the SAME attention item, not a copied Inbox record.
    visit=b.post(PREFIX+"/open-booking",json={"booking_uid":BOOKINGS[1]}).json()
    assert visit["brief"]["clinical_attention"][0]["item_id"]=="SYN-ATT-MRI"
    assert visit["brief"]["clinical_attention"][0]["review_state"]=="reviewed"
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
    after=a.post(PREFIX+"/open-booking",json={"booking_uid":BOOKINGS[1]}).json()
    assert after["brief"]["clinical_attention"]==[]
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


def test_qualified_synthetic_source_relink_clears_prior_patient_review(setup):
    """A requalified protected synthetic source never reuses an old decision."""
    app,engine=setup
    c=login(app)
    assert c.post(PREFIX+"/prepare").status_code==200
    ids=SYNTHETIC_161_PATIENT_IDS
    store=SyntheticAttentionStore(ids)
    original=store.verified_proofs["SYN-ATT-MRI"]
    # Test-only replacement of the fictional evidence oracle, never a public
    # API or an unqualified name-to-patient claim.
    store.verified_proofs["SYN-ATT-MRI"]={
        **original,"patient_id":ids[0],"evidence_ref":"syn-proof-referral-mri-1"}
    def verify(patient_id,dob=None):
        assert dob is None and patient_id in ids
    def action(kind,version,**kwargs):
        return store.act("SYN-ATT-MRI",AttentionAction(
            action=kind,expected_version=version,idempotency_key=token(),**kwargs
        ),"synthetic-test-actor",verify)
    prior=action("confirm_link",0,patient_id=ids[0],
                 evidence_kind="verified_referral",
                 evidence_ref="syn-proof-referral-mri-1",independently_verified=True)
    action("mark_reviewed",1)
    action("no_communication",2)
    assert prior["item"]["patient_id"]==ids[0]
    assert store.brief(ids[0])[0]["disposition"]=="no_communication"
    with store.lock:
        store.verified_proofs["SYN-ATT-MRI"]=original
    from fastapi import HTTPException
    try:
        action("confirm_link",2,patient_id=ids[1],evidence_kind="verified_referral",
               evidence_ref="syn-proof-referral-mri-2",independently_verified=True)
        assert False,"Stale CAS should reject relink"
    except HTTPException as e:
        assert e.status_code==409
    corrected=action("confirm_link",3,patient_id=ids[1],
                     evidence_kind="verified_referral",
                     evidence_ref="syn-proof-referral-mri-2",independently_verified=True)["item"]
    assert corrected["version"]==4 and corrected["item_id"]=="SYN-ATT-MRI"
    assert corrected["patient_id"]==ids[1]
    assert corrected["review_state"]=="needs_re_review"
    assert corrected["review_actor"] is None and corrected["reviewed_at"] is None
    assert corrected["disposition"]=="undecided" and corrected["candidate_summary"] is None
    assert store.brief(ids[0])==[]
    assert store.brief(ids[1])[0]["review_state"]=="needs_re_review"
    try:
        action("no_communication",4)
        assert False,"New review must precede patient-specific decision"
    except HTTPException as e:
        assert e.status_code==409
    action("mark_reviewed",4)
    action("no_communication",5)
    with Session(engine) as db:
        assert [p.status for p in db.scalars(select(ClinicalPendingORM)).all()]==["awaiting_patient"]*3


def test_adt_proof_qualified_only_for_original_synthetic_subject(setup):
    app,engine=setup
    c=login(app)
    assert c.post(PREFIX+"/prepare").status_code==200
    proof=c.get(PREFIX+"/attention/SYN-ATT-LAB").json()["verified_source_match"]
    assert proof["evidence_kind"]=="verified_adt"
    assert proof["patient_id"]==SYNTHETIC_161_PATIENT_IDS[0]
    payload={"action":"confirm_link","expected_version":0,"idempotency_key":token(),
             "patient_id":proof["patient_id"],"independently_verified":True,
             "evidence_kind":proof["evidence_kind"],"evidence_ref":proof["evidence_ref"]}
    assert c.post(PREFIX+"/attention/SYN-ATT-MRI/action",json=payload).status_code==409
    assert c.post(PREFIX+"/attention/SYN-ATT-LAB/action",json=payload).status_code==200
    brief=c.get(PREFIX+"/attention/brief/"+proof["patient_id"]).json()["items"]
    assert len(brief)==1 and brief[0]["item_id"]=="SYN-ATT-LAB"
    assert c.post(PREFIX+"/attention/SYN-ATT-GESY/action",json={
        **payload,"idempotency_key":token()}).status_code==409
