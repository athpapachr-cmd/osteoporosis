"""Synthetic-only Phase A attention owner; no providers, schema or live clinical writes."""
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Literal
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ACCOUNTS=(("syn-gmail","gmail","Δοκιμή Gmail"),
          ("syn-icloud","icloud_mail","Δοκιμή iCloud"),
          ("syn-outlook","microsoft_outlook","Δοκιμή Outlook"))


class AttentionAction(BaseModel):
    model_config=ConfigDict(extra="forbid")
    action:Literal["confirm_link","mark_reviewed","needs_review","no_communication"]
    expected_version:int=Field(ge=0)
    idempotency_key:UUID
    patient_id:str|None=Field(default=None,max_length=128)
    verified_dob:str|None=Field(default=None,max_length=10)
    independently_verified:bool=False


class SyntheticRevision(BaseModel):
    model_config=ConfigDict(extra="forbid")
    expected_version:int=Field(ge=0)
    idempotency_key:UUID
    new_revision:str=Field(min_length=1,max_length=36,pattern=r"^[A-Za-z0-9_-]+$")
    subject_scope:Literal["single_subject","unknown","mixed_subjects"]="single_subject"


def fixture(item_id,account,kind,subject_scope,title,summary):
    native="syn-msg-01"  # deliberately repeated across different accounts
    logical="/".join((account,native,"part-1","subject-1"))
    return dict(item_id=item_id,origin_kind="synthetic_fixture",
                mailbox_account_ref=account,provider_native_message_id=native,
                logical_source_item_key=logical,source_receipt_version_key=logical+"/r1",
                active_source_revision="r1",source_type=kind,
                source_subject_scope=subject_scope,source_title=title,
                candidate_summary=summary if subject_scope=="single_subject" else None,
                source_time="2026-10-23T09:17:00+03:00",
                review_state="unreviewed",disposition="undecided",
                identity_state="unlinked",patient_id=None,link_evidence_kind=None,
                review_actor=None,reviewed_at=None,version=0,audit=[])


class SyntheticAttentionStore:
    """A single in-memory source of truth per #161 test router, not a real database."""
    def __init__(self):
        self.lock=RLock()
        items=(
            fixture("SYN-ATT-MRI","syn-gmail","imaging_mri","single_subject",
                    "Συνθετική γνωμάτευση MRI","Δοκιμαστική γνωμάτευση γόνατος — όχι ιατρική ερμηνεία."),
            fixture("SYN-ATT-LAB","syn-icloud","lab","single_subject",
                    "Συνθετική εργαστηριακή αναφορά","Δοκιμαστική αναφορά — μη αποδεκτά κλινικά αποτελέσματα."),
            fixture("SYN-ATT-MIXED","syn-outlook","imaging_xray","mixed_subjects",
                    "Συνθετικό PDF πολλών ασθενών",None),
        )
        self.items={row["item_id"]:row for row in items}
        self.replays={}

    def get(self,item_id):
        item=self.items.get(item_id)
        if not item:raise HTTPException(404,"Unknown synthetic attention item")
        return item

    @staticmethod
    def projection(item):
        visible={k:v for k,v in item.items() if k!="audit"}
        if item["source_subject_scope"]!="single_subject":
            visible["candidate_summary"]=None
            visible["patient_id"]=None
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
                    if i["identity_state"]=="clinician_confirmed"
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
                              "actor":actor,"at":datetime.now(timezone.utc).isoformat()})
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
                if not req.independently_verified or not req.patient_id or not req.verified_dob:
                    raise HTTPException(422,"Independent identifier verification required")
                verify(req.patient_id,req.verified_dob)
                item.update(patient_id=req.patient_id,identity_state="clinician_confirmed",
                            link_evidence_kind="synthetic_exact_id_dob")
            elif req.action=="mark_reviewed":
                item["review_state"]="reviewed"
                item["review_actor"]=actor
                item["reviewed_at"]=datetime.now(timezone.utc).isoformat()
            elif req.action=="needs_review":
                item.update(review_state="needs_re_review",review_actor=None,
                            reviewed_at=None,disposition="undecided")
            elif req.action=="no_communication":
                if item["review_state"]!="reviewed" or item["identity_state"]!="clinician_confirmed":
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
            if req.subject_scope!=item["source_subject_scope"]:
                item.update(patient_id=None,identity_state="unlinked",link_evidence_kind=None)
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
