"""Synthetic-only Phase A attention owner; no providers, schema or live clinical writes."""
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Literal
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from clinical_data import SYNTHETIC_161_PATIENT_IDS

ACCOUNTS=(("syn-gmail","gmail","Δοκιμή Gmail"),
          ("syn-icloud","icloud_mail","Δοκιμή iCloud"),
          ("syn-outlook","microsoft_outlook","Δοκιμή Outlook"))


class AttentionAction(BaseModel):
    model_config=ConfigDict(extra="forbid")
    action:Literal["confirm_link","mark_reviewed","needs_review","no_communication"]
    expected_version:int=Field(ge=0)
    idempotency_key:UUID
    patient_id:str|None=Field(default=None,max_length=128)
    evidence_kind:Literal["verified_adt","verified_referral","clinician_source_comparison"]|None=None
    evidence_ref:str|None=Field(default=None,max_length=64)
    independently_verified:bool=False


class SyntheticRevision(BaseModel):
    model_config=ConfigDict(extra="forbid")
    expected_version:int=Field(ge=0)
    idempotency_key:UUID
    new_revision:str=Field(min_length=1,max_length=36,pattern=r"^[A-Za-z0-9_-]+$")
    subject_scope:Literal["single_subject","unknown","mixed_subjects"]="single_subject"


def fixture(item_id,account,kind,subject_scope,title,summary,*,
            source_identity_kind=None,source_identity_ref=None,
            linked_patient=None,source_status=None,native="syn-msg-01"):
    # Default native key is deliberately repeated across DIFFERENT accounts.
    logical="/".join((account,native,"part-1","subject-1"))
    return dict(item_id=item_id,origin_kind="synthetic_fixture",
                mailbox_account_ref=account,provider_native_message_id=native,
                logical_source_item_key=logical,source_receipt_version_key=logical+"/r1",
                active_source_revision="r1",source_type=kind,source_status=source_status,
                source_identity_kind=source_identity_kind,source_identity_ref=source_identity_ref,
                source_subject_scope=subject_scope,source_title=title,
                candidate_summary=summary if subject_scope=="single_subject" else None,
                source_time="2026-10-23T09:17:00+03:00",
                review_state="unreviewed",disposition="undecided",
                identity_state="clinician_confirmed" if linked_patient else "unlinked",
                patient_id=linked_patient,
                link_evidence_kind="verified_referral" if linked_patient else None,
                link_evidence_ref=source_identity_ref if linked_patient else None,
                linked_revision="r1" if linked_patient else None,
                review_actor=None,reviewed_at=None,version=0,audit=[])


class SyntheticAttentionStore:
    """A single in-memory source of truth per #161 test router, not a real database."""
    def __init__(self):
        self.lock=RLock()
        # Invented issuer-qualified evidence. NEVER a live ADT/beneficiary registry.
        self.verified_adts={"SYN-ADT-002":{
            "patient_id":SYNTHETIC_161_PATIENT_IDS[1],"issuer":"SYN-CY-CIVIL",
            "state":"verified","provenance":"synthetic_clinician_verification"}}
        self.verified_referrals={"SYN-REF-GESY-003":{
            "patient_id":SYNTHETIC_161_PATIENT_IDS[2],"issuer":"SYN-GESY",
            "item_id":"SYN-ATT-GESY","state":"verified",
            "provenance":"synthetic_prior_referral_confirmation"}}
        # Fictional independent original-order evidence scoped to item AND revision.
        self.chart_evidence={
            ("SYN-ATT-LAB","r1","SYN-LAB-ORDER-P1"):SYNTHETIC_161_PATIENT_IDS[0],
            ("SYN-ATT-LAB","r2","SYN-LAB-ORDER-P2"):SYNTHETIC_161_PATIENT_IDS[1],
        }
        items=(
            fixture("SYN-ATT-MRI","syn-gmail","imaging_mri","single_subject",
                    "Συνθετική γνωμάτευση MRI","Δοκιμαστική γνωμάτευση γόνατος — όχι ιατρική ερμηνεία.",
                    source_identity_kind="verified_adt",source_identity_ref="SYN-ADT-002"),
            fixture("SYN-ATT-LAB","syn-icloud","lab","single_subject",
                    "Συνθετική εργαστηριακή αναφορά","Δοκιμαστική αναφορά — μη αποδεκτά κλινικά αποτελέσματα."),
            fixture("SYN-ATT-MIXED","syn-outlook","imaging_xray","mixed_subjects",
                    "Συνθετικό PDF πολλών ασθενών",None),
            fixture("SYN-ATT-GESY","syn-gmail","gesy_notice","single_subject",
                    "Συνθετική ειδοποίηση ΓεΣΥ — αποτελέσματα διαθέσιμα",None,
                    source_identity_kind="verified_referral",
                    source_identity_ref="SYN-REF-GESY-003",
                    linked_patient=SYNTHETIC_161_PATIENT_IDS[2],
                    source_status="results_available",native="syn-msg-gesy-01"),
        )
        self.items={row["item_id"]:row for row in items}
        self.replays={}

    def get(self,item_id):
        item=self.items.get(item_id)
        if not item:raise HTTPException(404,"Unknown synthetic attention item")
        return item

    def _strong_link(self,item):
        """Match only issuer-qualified, unrevoked, source-scoped synthetic evidence."""
        if item["source_subject_scope"]!="single_subject":
            return None
        kind,ref=item.get("source_identity_kind"),item.get("source_identity_ref")
        if kind=="verified_adt":
            record=self.verified_adts.get(ref)
            if record and record.get("issuer")=="SYN-CY-CIVIL" and record.get("state")=="verified":
                return record["patient_id"],kind
        elif kind=="verified_referral":
            record=self.verified_referrals.get(ref)
            if (record and record.get("issuer")=="SYN-GESY"
                    and record.get("state")=="verified" and record.get("item_id")==item["item_id"]):
                return record["patient_id"],kind
        return None

    def _link_current(self,item):
        if item["identity_state"]!="clinician_confirmed" or not item["patient_id"]:
            return False
        kind=item.get("link_evidence_kind")
        if kind in ("verified_adt","verified_referral"):
            return (self._strong_link(item)==(item["patient_id"],kind)
                    and item.get("link_evidence_ref")==item.get("source_identity_ref"))
        if kind=="clinician_source_comparison":
            return self.chart_evidence.get((
                item["item_id"],item["active_source_revision"],
                item.get("link_evidence_ref")))==item["patient_id"]
        return False

    def projection(self,item):
        # Raw equality keys (even fictional ones) do not belong in the routine UI.
        visible={k:v for k,v in item.items() if k not in (
            "audit","source_identity_ref","link_evidence_ref")}
        if item["source_subject_scope"]!="single_subject":
            visible["candidate_summary"]=None
            visible["patient_id"]=None
        strong=self._strong_link(item)
        visible["verified_link_candidate"]=(
            {"patient_id":strong[0],"evidence_kind":strong[1],
             "reason":("Επαληθευμένος συνθετικός ΑΔΤ" if strong[1]=="verified_adt"
                       else "Παραπεμπτικό επιβεβαιωμένο σε προηγούμενη επίσκεψη")}
            if strong and item["identity_state"]!="clinician_confirmed" else None)
        if item["identity_state"]=="clinician_confirmed" and not self._link_current(item):
            visible.update(identity_state="conflict",patient_id=None,
                           verified_link_candidate=None,
                           review_state="needs_re_review",disposition="undecided",
                           review_actor=None,reviewed_at=None,candidate_summary=None)
        visible["audit_event_count"]=len(item["audit"])
        return deepcopy(visible)

    def list_items(self):
        with self.lock:
            return {"mode":"synthetic_only","live_provider_connected":False,
                    "coverage":"synthetic_fixture_only",
                    "accounts":[{"mailbox_account_id":a,"provider_family":p,
                                 "label":label,"coverage":"synthetic_fixture_only"}
                                for a,p,label in ACCOUNTS],
                    "items":[self.projection(x) for x in self.items.values()]}

    def detail(self,item_id):
        with self.lock:return {"mode":"synthetic_only","item":self.projection(self.get(item_id))}

    def brief(self,patient_id):
        with self.lock:
            return [self.projection(i) for i in self.items.values()
                    if self._link_current(i)
                    and i["patient_id"]==patient_id and i["source_subject_scope"]=="single_subject"]

    def _replayed(self,item,actor,key,payload):
        previous=self.replays.get((item["item_id"],actor,str(key)))
        if previous:
            if previous["payload"]!=payload:
                raise HTTPException(409,"Idempotency key reused with new payload")
            return {**deepcopy(previous["result"]),"replayed":True}

    def _save(self,item,actor,key,payload,kind):
        item["version"]+=1
        item["audit"].append({"kind":kind,"version":item["version"],
                              "actor":actor,"at":datetime.now(timezone.utc).isoformat(),
                              "evidence_kind":item.get("link_evidence_kind")})
        result={"item":self.projection(item),"replayed":False}
        self.replays[(item["item_id"],actor,str(key))]={
            "payload":payload,"result":deepcopy(result)}
        return result

    def act(self,item_id,req,actor,verify):
        payload=req.model_dump(mode="json")
        with self.lock:
            item=self.get(item_id)
            replay=self._replayed(item,actor,req.idempotency_key,payload)
            if replay is not None:return replay
            if req.expected_version!=item["version"]:
                raise HTTPException(409,"Stale attention version")
            if req.action=="confirm_link":
                if item["source_subject_scope"]!="single_subject":
                    raise HTTPException(409,"Mixed/unknown subject is Inbox-only")
                if not req.independently_verified or not req.patient_id or not req.evidence_kind:
                    raise HTTPException(422,"Evidence-backed clinician confirmation required")
                # The existing protected synthetic patient must actually be present.
                verify(req.patient_id,None)
                strong=self._strong_link(item)
                if req.evidence_kind in ("verified_adt","verified_referral"):
                    if strong!=(req.patient_id,req.evidence_kind):
                        raise HTTPException(409,"Unverified or conflicting source identity")
                    evidence_ref=item.get("source_identity_ref")
                elif req.evidence_kind=="clinician_source_comparison":
                    evidence_ref=req.evidence_ref
                    if (not evidence_ref or self.chart_evidence.get((
                            item_id,item["active_source_revision"],evidence_ref))!=req.patient_id
                            or (strong and strong[0]!=req.patient_id)):
                        raise HTTPException(409,"No independently verified scoped source evidence")
                else:
                    raise HTTPException(422,"Unsupported verification evidence")
                # Name/date/DOB or a checked box alone never authorizes linkage.
                if (item["identity_state"]=="clinician_confirmed"
                        and item["patient_id"]!=req.patient_id):
                    item.update(review_state="needs_re_review",review_actor=None,
                                reviewed_at=None,disposition="undecided",
                                candidate_summary=None)
                item.update(patient_id=req.patient_id,identity_state="clinician_confirmed",
                            link_evidence_kind=req.evidence_kind,
                            link_evidence_ref=evidence_ref,
                            linked_revision=item["active_source_revision"])
            elif req.action=="mark_reviewed":
                item["review_state"]="reviewed"
                item["review_actor"]=actor
                item["reviewed_at"]=datetime.now(timezone.utc).isoformat()
            elif req.action=="needs_review":
                item.update(review_state="needs_re_review",review_actor=None,
                            reviewed_at=None,disposition="undecided")
            elif req.action=="no_communication":
                if item["review_state"]!="reviewed" or not self._link_current(item):
                    raise HTTPException(409,"Verified patient and review required")
                item["disposition"]="no_communication"
            return self._save(item,actor,req.idempotency_key,payload,req.action)

    def revise(self,item_id,req,actor):
        payload=req.model_dump(mode="json")
        with self.lock:
            item=self.get(item_id)
            replay=self._replayed(item,actor,req.idempotency_key,payload)
            if replay is not None:return replay
            if req.expected_version!=item["version"] or req.new_revision==item["active_source_revision"]:
                raise HTTPException(409,"Stale or duplicate source revision")
            # A mixed/unknown document must never become single-subject merely
            # because a synthetic adapter claims so; safe segmentation needs
            # separately reviewed evidence and explicit clinician verification.
            if item["source_subject_scope"]!="single_subject" and req.subject_scope=="single_subject":
                raise HTTPException(409,"Independent subject segmentation required")
            if (req.subject_scope!=item["source_subject_scope"]
                    or item.get("link_evidence_kind")=="clinician_source_comparison"):
                item.update(patient_id=None,identity_state="unlinked",link_evidence_kind=None,
                            link_evidence_ref=None,linked_revision=None)
            item.update(active_source_revision=req.new_revision,
                        source_receipt_version_key=item["logical_source_item_key"]+"/"+req.new_revision,
                        source_subject_scope=req.subject_scope,candidate_summary=None,
                        review_state="needs_re_review",review_actor=None,
                        reviewed_at=None,disposition="undecided")
            return self._save(item,actor,req.idempotency_key,payload,"source_revised")


def build_synthetic_attention_router(store,require_test_access,verify,patients):
    router=APIRouter(prefix="/attention",tags=["synthetic-attention"],
                     dependencies=[Depends(require_test_access)])

    @router.get("")
    def all_items():return store.list_items()

    @router.get("/eligible-patients")
    def eligible():return {"mode":"synthetic_only","patients":patients()}

    @router.get("/brief/{patient_id}")
    def brief(patient_id:str):
        verify(patient_id,None)
        return {"mode":"synthetic_only","items":store.brief(patient_id)}

    @router.get("/{item_id}")
    def detail(item_id:str):return store.detail(item_id)

    @router.post("/{item_id}/action")
    def action(item_id:str,req:AttentionAction,actor=Depends(require_test_access)):
        return store.act(item_id,req,actor.principal_id,verify)

    @router.post("/{item_id}/synthetic-revision")
    def revision(item_id:str,req:SyntheticRevision,actor=Depends(require_test_access)):
        return store.revise(item_id,req,actor.principal_id)

    return router
