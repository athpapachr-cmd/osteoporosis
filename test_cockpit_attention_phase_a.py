"""Phase-A synthetic ClinicalAttention contract tests. No external providers or PHI."""
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select,func
from clinical_data import ClinicalPendingORM
from test_cockpit_synthetic_161 import setup,application,login,PREFIX,BOOKINGS,SYNTHETIC_161_PATIENT_IDS


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
    assert len(data["accounts"])==3 and len(data["items"])==4
    assert len({i["logical_source_item_key"] for i in data["items"]})==4
    assert len({i["provider_native_message_id"] for i in data["items"]})==2
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
             "evidence_kind":"verified_adt","independently_verified":True}
    # A name, a different patient, a DOB or an unchecked assertion is never
    # equivalent to an issuer-qualified clinical registry identity match.
    assert a.post(endpoint+"/action",json={**payload,"idempotency_key":token(),
              "patient_id":SYNTHETIC_161_PATIENT_IDS[0]}).status_code==409
    assert a.post(endpoint+"/action",json={**payload,"idempotency_key":token(),
              "independently_verified":False}).status_code==422
    assert a.post(endpoint+"/action",json={**payload,"idempotency_key":token(),
              "verified_dob":"1972-01-01"}).status_code==422
    result=a.post(endpoint+"/action",json=payload)
    assert result.status_code==200,result.text
    before=result.json()["item"]
    assert before["patient_id"]==SYNTHETIC_161_PATIENT_IDS[1] and before["version"]==1
    assert before["link_evidence_kind"]=="verified_adt"
    assert "source_identity_ref" not in before and "link_evidence_ref" not in before
    assert a.post(endpoint+"/action",json=payload).json()["replayed"] is True
    assert a.post(endpoint+"/action",json={**payload,"evidence_kind":"verified_referral"}).status_code==409
    assert b.post(endpoint+"/action",json={"action":"mark_reviewed","expected_version":0,
                                           "idempotency_key":token()}).status_code==409
    mark={"action":"mark_reviewed","expected_version":1,"idempotency_key":token()}
    assert b.post(endpoint+"/action",json=mark).status_code==200
    assert b.get(PREFIX+"/attention/brief/"+SYNTHETIC_161_PATIENT_IDS[1]).json()["items"][0]["review_state"]=="reviewed"
    # Existing booking owner consumes the SAME attention item, not a second Inbox record.
    visit=b.post(PREFIX+"/open-booking",json={"booking_uid":BOOKINGS[1]}).json()
    assert visit["brief"]["clinical_attention"][0]["item_id"]=="SYN-ATT-MRI"
    assert visit["brief"]["clinical_attention"][0]["review_state"]=="reviewed"
    assert b.get(PREFIX+"/attention/brief/"+SYNTHETIC_161_PATIENT_IDS[0]).json()["items"]==[]
    assert b.post(PREFIX+"/attention/SYN-ATT-MIXED/action",json={
        **payload,"idempotency_key":token()}).status_code==409
    # Weak source, matching patient names, and free-form unverified tokens fail closed.
    weak=PREFIX+"/attention/SYN-ATT-LAB"
    assert a.post(weak+"/action",json={
        **payload,"idempotency_key":token()}).status_code==409
    assert a.post(weak+"/action",json={**payload,"idempotency_key":token(),
        "evidence_kind":"clinician_source_comparison",
        "evidence_ref":"UNVERIFIED-NAME"}).status_code==409
    assert a.get(weak).json()["item"]["verified_link_candidate"] is None
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


def test_corrected_source_relink_invalidates_old_patient_review_and_disposition(setup):
    app,engine=setup
    c=login(app)
    assert c.post(PREFIX+"/prepare").status_code==200
    endpoint=PREFIX+"/attention/SYN-ATT-LAB"
    old_id,new_id=SYNTHETIC_161_PATIENT_IDS[:2]

    def act(action,expected_version,**fields):
        return c.post(endpoint+"/action",json={
            "action":action,"expected_version":expected_version,
            "idempotency_key":token(),**fields
        })

    old_link=act("confirm_link",0,patient_id=old_id,
                 evidence_kind="clinician_source_comparison",
                 evidence_ref="SYN-LAB-ORDER-P1",independently_verified=True)
    assert old_link.status_code==200,old_link.text
    assert act("mark_reviewed",1).status_code==200
    assert act("no_communication",2).status_code==200
    old_brief=c.get(PREFIX+"/attention/brief/"+old_id).json()["items"]
    assert len(old_brief)==1 and old_brief[0]["disposition"]=="no_communication"
    # A second patient cannot be attached on the basis of a different name,
    # a new referral key, or an invented verification checkbox.
    assert act("confirm_link",3,patient_id=new_id,
               evidence_kind="clinician_source_comparison",
               evidence_ref="SYN-LAB-ORDER-P2",independently_verified=True).status_code==409

    correction={"expected_version":3,"idempotency_key":token(),
                "new_revision":"r2","subject_scope":"single_subject"}
    changed=c.post(endpoint+"/synthetic-revision",json=correction)
    assert changed.status_code==200,changed.text
    assert changed.json()["item"]["patient_id"] is None
    assert changed.json()["item"]["review_state"]=="needs_re_review"
    assert changed.json()["item"]["disposition"]=="undecided"
    assert c.get(PREFIX+"/attention/brief/"+old_id).json()["items"]==[]
    assert c.post(endpoint+"/synthetic-revision",json=correction).json()["replayed"]
    assert act("confirm_link",3,patient_id=new_id,
               evidence_kind="clinician_source_comparison",
               evidence_ref="SYN-LAB-ORDER-P2",independently_verified=True).status_code==409
    assert act("confirm_link",4,patient_id=new_id,
               evidence_kind="clinician_source_comparison",
               evidence_ref="SYN-LAB-ORDER-P1",independently_verified=True).status_code==409
    relink={"action":"confirm_link","expected_version":4,
            "idempotency_key":token(),"patient_id":new_id,
            "evidence_kind":"clinician_source_comparison",
            "evidence_ref":"SYN-LAB-ORDER-P2","independently_verified":True}
    result=c.post(endpoint+"/action",json=relink)
    assert result.status_code==200,result.text
    item=result.json()["item"]
    assert item["item_id"]=="SYN-ATT-LAB" and item["version"]==5
    assert item["patient_id"]==new_id and item["identity_state"]=="clinician_confirmed"
    assert item["review_state"]=="needs_re_review"
    assert item["review_actor"] is None and item["reviewed_at"] is None
    assert item["disposition"]=="undecided" and item["candidate_summary"] is None
    assert c.post(endpoint+"/action",json=relink).json()["replayed"] is True
    assert c.get(endpoint).json()["item"]["audit_event_count"]==5
    assert c.get(PREFIX+"/attention/brief/"+old_id).json()["items"]==[]
    new_brief=c.get(PREFIX+"/attention/brief/"+new_id).json()["items"]
    assert len(new_brief)==1 and new_brief[0]["item_id"]==item["item_id"]
    assert new_brief[0]["review_state"]=="needs_re_review"
    assert c.post(PREFIX+"/open-booking",json={"booking_uid":BOOKINGS[0]}).json()["brief"]["clinical_attention"]==[]
    assert c.post(PREFIX+"/open-booking",json={"booking_uid":BOOKINGS[1]}).json()["brief"]["clinical_attention"][0]["disposition"]=="undecided"
    assert act("no_communication",5).status_code==409
    assert act("mark_reviewed",5).status_code==200
    assert act("no_communication",6).status_code==200
    with Session(engine) as db:
        assert [p.status for p in db.scalars(select(ClinicalPendingORM)).all()]==["awaiting_patient"]*3


def test_prior_verified_gesy_referral_is_linked_without_dob_not_result_or_task(setup):
    app,engine=setup
    c=login(app)
    assert c.post(PREFIX+"/prepare").status_code==200
    endpoint=PREFIX+"/attention/SYN-ATT-GESY"
    item=c.get(endpoint).json()["item"]
    assert item["source_type"]=="gesy_notice"
    assert item["source_status"]=="results_available"
    assert item["candidate_summary"] is None and item["review_state"]=="unreviewed"
    assert item["identity_state"]=="clinician_confirmed"
    assert item["link_evidence_kind"]=="verified_referral"
    assert item["patient_id"]==SYNTHETIC_161_PATIENT_IDS[2]
    assert item["verified_link_candidate"] is None
    assert "source_identity_ref" not in item
    clinical=c.post(PREFIX+"/open-booking",json={"booking_uid":BOOKINGS[2]}).json()
    assert clinical["brief"]["clinical_attention"][0]["item_id"]=="SYN-ATT-GESY"
    assert clinical["brief"]["clinical_attention"][0]["review_state"]=="unreviewed"
    result=c.post(endpoint+"/action",json={
        "action":"mark_reviewed","expected_version":0,"idempotency_key":token()})
    assert result.status_code==200
    assert result.json()["item"]["source_status"]=="results_available"
    assert result.json()["item"]["candidate_summary"] is None
    assert result.json()["item"]["disposition"]=="undecided"
    with Session(engine) as db:
        assert [p.status for p in db.scalars(select(ClinicalPendingORM)).all()]==["awaiting_patient"]*3


def test_qualified_evidence_revocation_blocks_stale_patient_projection():
    import pytest
    from fastapi import HTTPException
    from clinical_attention_phase_a import SyntheticAttentionStore, AttentionAction

    store=SyntheticAttentionStore()
    # Protected synthetic patient setup, not a raw source-claimed patient ID.
    store.establish_previously_verified_referral(lambda patient_id,dob=None: None)
    patient_id=SYNTHETIC_161_PATIENT_IDS[2]
    assert store.brief(patient_id)[0]["item_id"]=="SYN-ATT-GESY"
    assert store.act("SYN-ATT-GESY",AttentionAction(action="mark_reviewed",
        expected_version=0,idempotency_key=token()),"synthetic-tester",lambda *args:None)
    store.verified_referrals["SYN-REF-GESY-003"]["state"]="revoked"
    assert store.brief(patient_id)==[]
    unsafe=store.detail("SYN-ATT-GESY")["item"]
    assert unsafe["identity_state"]=="conflict" and unsafe["patient_id"] is None
    with pytest.raises(HTTPException) as err:
        store.act("SYN-ATT-GESY",AttentionAction(action="no_communication",
            expected_version=1,idempotency_key=token()),"synthetic-tester",lambda *args:None)
    assert err.value.status_code==409
    # Likewise a revoked or conflicting issuer ID cannot keep the MRI in a Brief.
    mri=store.get("SYN-ATT-MRI")
    mri.update(identity_state="clinician_confirmed",
               patient_id=SYNTHETIC_161_PATIENT_IDS[1],
               link_evidence_kind="verified_adt",link_evidence_ref="SYN-ADT-002",
               linked_revision="r1")
    assert len(store.brief(SYNTHETIC_161_PATIENT_IDS[1]))==1
    store.verified_adts["SYN-ADT-002"]["state"]="conflict"
    assert store.brief(SYNTHETIC_161_PATIENT_IDS[1])==[]
    assert store.detail("SYN-ATT-MRI")["item"]["identity_state"]=="conflict"


@pytest.mark.parametrize("item_id,kind,patient_index", [
    ("SYN-ATT-MRI","verified_adt",1),
    ("SYN-ATT-GESY","verified_referral",2),
])
def test_same_subject_corrected_revision_revokes_strong_link_and_old_replays(
        setup,item_id,kind,patient_index):
    """P1: unchanged subject_scope cannot attest identity continuity across revisions."""
    app,engine=setup
    c=login(app)
    assert c.post(PREFIX+"/prepare").status_code==200
    pid=SYNTHETIC_161_PATIENT_IDS[patient_index]
    endpoint=PREFIX+"/attention/"+item_id
    initial=c.get(endpoint).json()["item"]
    assert initial["source_subject_scope"]=="single_subject"
    if kind=="verified_adt":
        assert initial["verified_link_candidate"]["patient_id"]==pid
        confirmed=c.post(endpoint+"/action",json={
            "action":"confirm_link","expected_version":0,"idempotency_key":token(),
            "patient_id":pid,"evidence_kind":kind,"independently_verified":True})
        assert confirmed.status_code==200,confirmed.text
        version=1
    else:
        assert initial["identity_state"]=="clinician_confirmed"
        assert initial["link_evidence_kind"]=="verified_referral"
        version=0

    assert c.post(endpoint+"/action",json={
        "action":"mark_reviewed","expected_version":version,
        "idempotency_key":token()}).status_code==200
    version+=1
    prior_disposition={"action":"no_communication","expected_version":version,
                       "idempotency_key":token()}
    approved=c.post(endpoint+"/action",json=prior_disposition)
    assert approved.status_code==200,approved.text
    version+=1
    linked=c.get(PREFIX+"/attention/brief/"+pid).json()["items"]
    assert len([x for x in linked if x["item_id"]==item_id])==1
    assert c.get(endpoint).json()["item"]["disposition"]=="no_communication"

    revision={"expected_version":version,"idempotency_key":token(),
              "new_revision":"r2","subject_scope":"single_subject"}
    updated=c.post(endpoint+"/synthetic-revision",json=revision)
    assert updated.status_code==200,updated.text
    item=updated.json()["item"]
    assert item["item_id"]==item_id
    assert item["active_source_revision"]=="r2" and item["version"]==version+1
    assert item["source_subject_scope"]=="single_subject"
    assert item["identity_state"]=="unlinked" and item["patient_id"] is None
    assert item["verified_link_candidate"] is None
    assert item["source_identity_kind"] is None and item["link_evidence_kind"] is None
    assert item["linked_revision"] is None
    assert item["review_state"]=="needs_re_review" and item["review_actor"] is None
    assert item["reviewed_at"] is None and item["disposition"]=="undecided"
    assert "source_identity_ref" not in item and "link_evidence_ref" not in item
    assert c.get(PREFIX+"/attention/brief/"+pid).json()["items"]==[]
    booking=c.post(PREFIX+"/open-booking",json={
        "booking_uid":BOOKINGS[patient_index]}).json()
    assert all(x["item_id"]!=item_id for x in booking["brief"]["clinical_attention"])

    # The old qualified identifier and previously approved disposition are
    # bound to r1. Neither a fresh action nor an old idempotency replay may
    # reattach/approve them on the corrected r2 document.
    stale=c.post(endpoint+"/action",json=prior_disposition)
    assert stale.status_code==409,stale.text
    attempt=c.post(endpoint+"/action",json={
        "action":"confirm_link","expected_version":version+1,
        "idempotency_key":token(),"patient_id":pid,
        "evidence_kind":kind,"independently_verified":True})
    assert attempt.status_code==409,attempt.text
    assert c.post(endpoint+"/action",json={
        "action":"no_communication","expected_version":version+1,
        "idempotency_key":token()}).status_code==409
    assert c.post(endpoint+"/synthetic-revision",json=revision).json()["replayed"]

    # Repeated preparation cannot silently restore the prior GESY referral
    # after the source has been corrected and its evidence invalidated.
    assert c.post(PREFIX+"/prepare").status_code==200
    again=c.get(endpoint).json()["item"]
    assert again["patient_id"] is None and again["verified_link_candidate"] is None
    assert c.get(PREFIX+"/attention/brief/"+pid).json()["items"]==[]
    with Session(engine) as db:
        assert [p.status for p in db.scalars(select(ClinicalPendingORM)).all()]==[
            "awaiting_patient"]*3


def test_corrected_source_audit_keeps_sanitized_prior_link_provenance():
    """Revision clears current evidence but retains minimal append-only audit."""
    from clinical_attention_phase_a import SyntheticAttentionStore, SyntheticRevision, AttentionAction

    store=SyntheticAttentionStore()
    patient_id=SYNTHETIC_161_PATIENT_IDS[1]
    store.act("SYN-ATT-MRI",AttentionAction(
        action="confirm_link",expected_version=0,idempotency_key=token(),
        patient_id=patient_id,evidence_kind="verified_adt",
        independently_verified=True),"synthetic-clinician",lambda *_:None)
    confirmed=store.get("SYN-ATT-MRI")
    assert confirmed["linked_revision"]=="r1"
    store.revise("SYN-ATT-MRI",SyntheticRevision(
        expected_version=1,idempotency_key=token(),new_revision="r2",
        subject_scope="single_subject"),"synthetic-clinician")
    now=store.get("SYN-ATT-MRI")
    assert now["source_identity_kind"] is None and now["source_identity_ref"] is None
    assert now["link_evidence_ref"] is None and now["linked_revision"] is None
    assert store.brief(patient_id)==[]
    audit=now["audit"][-1]
    assert audit["kind"]=="source_revised"
    assert audit["previous_link_evidence_kind"]=="verified_adt"
    assert audit["previous_source_revision"]=="r1"
    assert audit["identity_recheck_required"] is True
    assert "SYN-ADT-002" not in str(audit)
