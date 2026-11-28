"""Explicit audited correction of an existing signed visit, never an overwrite API."""
from copy import deepcopy
from datetime import date
import hashlib
import json
from uuid import uuid4
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field, UUID4
from sqlalchemy import select, update, delete
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, OperationalError
from clinical_data import EncounterORM, PatientORM, ClinicalPendingORM, VisitCaptureContextORM, VisitCaptureCandidateV1, _validate_visit_capture_candidate, _render_visit_capture, utcnow
from clinical_data import grounded_heidi_visit_focus, heidi_focus_fact_groups
from clinical_pending_one_tap import actor_for, installed, project_lifecycle, cancel_pending, pending_write_session

class Amendment(BaseModel):
    model_config={"extra":"forbid"}
    context_id: str=Field(min_length=1,max_length=120)
    expected_token: str=Field(pattern=r'^[a-f0-9]{64}$')
    request_id: UUID4
    candidate: VisitCaptureCandidateV1
    pending_bindings: list[str | None]=Field(max_length=60)

class SavedVisitReview(BaseModel):
    model_config = {"extra": "forbid"}
    expected_token: str = Field(pattern=r'^[a-f0-9]{64}$')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),default=str).encode()).hexdigest()

def signed_versions(row):
    signed=(row.payload_json or {}).get('_visit_capture_v1',{})
    history=signed.get('amendment_history',[])
    return [(h['previous']['date'],h['previous']['signed']) for h in history]+[(row.encounter_date,signed)]

def source_item(row,pending_id):
    for _,signed in reversed(signed_versions(row)):
        ids=signed.get('pending_ids',[])
        if pending_id in ids:
            items=signed.get('candidate',{}).get('pending',[]);i=ids.index(pending_id)
            if i<len(items):return deepcopy(items[i])
    return None

def token(db,row,pending):
    return digest({'date':row.encounter_date,'status':row.status,'updated_at':row.updated_at,'payload':row.payload_json,
        'pending':[{'id':p.id,'status':p.status,'updated_at':p.updated_at,'description':p.description,
            'trigger':p.trigger_text,'type':p.item_type,'role':p.responsible_role,'dependency':p.external_dependency_json,
            'review_date':p.review_date,'provenance':p.provenance_json,'resolution':p.resolution_event_id,
            'revision':project_lifecycle(db,p)['revision']} for p in pending]})

def attach_visit_amendments(router,engine,protected,context_for,ensure_patient):
    def load(db,pid,eid,request):
        patient=ensure_patient(db,pid);actor=actor_for(request,patient)
        row=db.scalar(select(EncounterORM).where(EncounterORM.id==eid,EncounterORM.patient_id==pid).with_for_update())
        if not row:raise HTTPException(404,'Visit not found')
        signed=(row.payload_json or {}).get('_visit_capture_v1',{})
        if row.status not in ('completed','amended') or not signed.get('signed') or not signed.get('signed_at') or not signed.get('candidate'):
            raise HTTPException(409,'This visit is not available for amendment')
        if not installed(engine):raise HTTPException(503,'Checkpoint storage is not installed')
        pending=db.scalars(select(ClinicalPendingORM).where(ClinicalPendingORM.source_encounter_id==eid).order_by(ClinicalPendingORM.id).with_for_update()).all()
        if any(p.patient_id!=pid for p in pending):raise HTTPException(409,'Invalid source ownership')
        return patient,actor,row,signed,pending

    @router.get('/patient/{patient_id}/visit/{encounter_id}/edit',dependencies=protected)
    def edit(patient_id:str,encounter_id:str,request:Request,response:Response):
        response.headers['Cache-Control']='no-store'
        with Session(engine) as db:
            _,_,row,signed,pending=load(db,patient_id,encounter_id,request)
            candidate=deepcopy(signed['candidate']);items=[];bindings=[]
            for p in pending:
                if p.status in ('cancelled','deleted'):continue
                item=source_item(row,p.id)
                if item is None:raise HTTPException(409,'Missing task source')
                items.append(item);bindings.append(p.id)
            candidate['pending']=items
            omitted_visit_focus=0
            if candidate.get('next_visit_focus') and any(binding.get('role')=='heidi_today' for binding in candidate['source_bindings']):
                draft=VisitCaptureCandidateV1.model_validate(candidate)
                focus=grounded_heidi_visit_focus(draft)
                omitted_visit_focus=len(draft.next_visit_focus)-len(focus)
                if omitted_visit_focus:
                    candidate['next_visit_focus']=[item.model_dump(mode='json') for item in focus]
            return {'patient_id':patient_id,'encounter_id':row.id,'candidate':candidate,'pending_bindings':bindings,
                'omitted_visit_focus':omitted_visit_focus,
                'expected_token':token(db,row,pending),'revision':signed.get('revision',0)}

    @router.post('/patient/{patient_id}/visit/{encounter_id}/review',dependencies=protected)
    async def review_saved_visit(patient_id:str,encounter_id:str,request:Request,response:Response):
        from clinical_capture_cp1 import _visit_focus_from_facts
        from clinical_visit_preparation import connected_call
        response.headers['Cache-Control']='no-store'
        try:
            raw=bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw)>4096:raise ValueError('request too large')
            body=SavedVisitReview.model_validate_json(bytes(raw))
        except Exception:
            raise HTTPException(422,'Έλεγξε την επίσκεψη.') from None
        with Session(engine) as db:
            _,_,row,signed,pending=load(db,patient_id,encounter_id,request)
            if token(db,row,pending)!=body.expected_token:
                raise HTTPException(409,'Visit changed; reopen editing')
            candidate=VisitCaptureCandidateV1.model_validate(signed['candidate'])
            if candidate.source_identity_state!='consistent' or candidate.reason_is_unspecified or not any(
                    binding.role=='heidi_today' for binding in candidate.source_bindings):
                raise HTTPException(422,'Η επίσκεψη δεν περιέχει κατάλληλη καταγραφή για έλεγχο.')
            facts=[];seen=set()
            for kind,group in heidi_focus_fact_groups(candidate):
                if kind=='reason':continue
                for fact in group:
                    if fact.certainty!='certain' or fact.text in seen:continue
                    seen.add(fact.text)
                    facts.append({'source_id':len(facts)+1,'kind':kind,'excerpt':fact.text,
                                  'draft_text':fact.text,'verification':fact.verification})
        status,suggestions=await connected_call(request,_visit_focus_from_facts(
            candidate,facts,'\n'.join(f['excerpt'] for f in facts),'saved_visit'))
        with Session(engine) as db:
            _,_,row,_,pending=load(db,patient_id,encounter_id,request)
            if token(db,row,pending)!=body.expected_token:
                raise HTTPException(409,'Visit changed; reopen editing')
        return {'patient_id':patient_id,'encounter_id':encounter_id,'expected_token':body.expected_token,
                'status':status,'suggestions':[item.model_dump(mode='json') for item in suggestions]}

    @router.post('/patient/{patient_id}/visit/{encounter_id}/amend',dependencies=protected)
    async def amend(patient_id:str,encounter_id:str,request:Request,response:Response):
        response.headers['Cache-Control']='no-store';raw=bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw)>1000000:raise HTTPException(413,'Visit draft is too large')
        try:
            req=Amendment.model_validate_json(bytes(raw));date.fromisoformat(req.candidate.encounter_date)
            _validate_visit_capture_candidate(req.candidate)
        except Exception:raise HTTPException(422,'Έλεγξε τα πεδία και την ημερομηνία της επίσκεψης.') from None
        if req.candidate.source_identity_state!='consistent':raise HTTPException(409,'Resolve source identity before saving')
        if len(req.pending_bindings)!=len(req.candidate.pending):raise HTTPException(422,'Invalid task bindings')
        fingerprint=req.model_dump(mode='json')
        for suggestion in fingerprint['candidate']['next_visit_focus']:
            if suggestion['evidence_basis']=='source_note':
                del suggestion['evidence_basis']
        identity=digest({'patient':patient_id,'visit':encounter_id,'request':fingerprint})
        with pending_write_session(engine) as db:
            patient,actor,row,signed,pending=load(db,patient_id,encounter_id,request)
            history=deepcopy(signed.get('amendment_history',[]))
            for h in history:
                if h['request_id']==str(req.request_id):
                    if h['digest']!=identity or h['actor']!=[actor.principal_id,actor.role]:raise HTTPException(409,'Amendment request conflict')
                    return {'encounter_id':row.id,'revision':signed.get('revision',0),'replayed':True}
            if token(db,row,pending)!=req.expected_token:raise HTTPException(409,'Visit or obligations changed; reopen editing')
            # Advice created from a saved visit must remain grounded in that signed version.
            saved_advice=[s for s in req.candidate.next_visit_focus if s.evidence_basis=='saved_visit']
            if saved_advice:
                signed_candidate=VisitCaptureCandidateV1.model_validate(signed['candidate'])
                signed_facts={fact.text for kind,group in heidi_focus_fact_groups(signed_candidate)
                              if kind!='reason' for fact in group if fact.certainty=='certain'}
                draft_facts={fact.text for kind,group in heidi_focus_fact_groups(req.candidate)
                             if kind!='reason' for fact in group if fact.certainty=='certain'}
                for suggestion in saved_advice:
                    if (req.candidate.reason_is_unspecified or req.candidate.source_identity_state!='consistent'
                            or not any(b.role=='heidi_today' for b in req.candidate.source_bindings)
                            or suggestion.purpose!=req.candidate.reason_for_visit.text
                            or not all(excerpt in draft_facts for excerpt in suggestion.source_excerpts)):
                        raise HTTPException(422,'Planning suggestion source changed')
                    if suggestion in signed_candidate.next_visit_focus:continue
                    if (signed_candidate.reason_is_unspecified or signed_candidate.source_identity_state!='consistent'
                            or not any(b.role=='heidi_today' for b in signed_candidate.source_bindings)
                            or suggestion.purpose!=signed_candidate.reason_for_visit.text
                            or not all(excerpt in signed_facts for excerpt in suggestion.source_excerpts)):
                        raise HTTPException(422,'Planning suggestion source changed')
            context=context_for(db,req.context_id)
            if context.patient_id!=patient_id:raise HTTPException(409,'Invalid patient context')
            ids=[v for v in req.pending_bindings if v is not None]
            eligible={p.id:p for p in pending if p.status not in ('cancelled','deleted')}
            if len(ids)!=len(set(ids)) or any(i not in eligible for i in ids):raise HTTPException(409,'Invalid or obsolete task binding')
            pairs=[];keep=set()
            normalized=req.candidate.model_dump(mode='json',exclude_none=False)
            for item,binding in zip(normalized['pending'],req.pending_bindings):
                same=binding is not None and item==source_item(row,binding)
                if not same and item['status'] not in ('open','awaiting_patient','awaiting_result'):raise HTTPException(422,'New tasks must be unfulfilled')
                if same:keep.add(binding)
                pairs.append((item,binding if same else None))
            now=utcnow();old_updated=row.updated_at
            # SELECT FOR UPDATE is effective on PostgreSQL; this CAS also protects SQLite.
            try:cas=db.execute(update(EncounterORM).where(EncounterORM.id==row.id,EncounterORM.updated_at==old_updated).values(updated_at=now))
            except OperationalError:raise HTTPException(409,'Visit changed; reopen editing') from None
            if cas.rowcount!=1:raise HTTPException(409,'Visit changed')
            consumed=db.execute(delete(VisitCaptureContextORM).where(VisitCaptureContextORM.id==context.id))
            if consumed.rowcount!=1:raise HTTPException(409,'Edit context already consumed')
            previous={'date':row.encounter_date,'status':row.status,'updated_at':old_updated.isoformat(),
                'signed':{k:deepcopy(v) for k,v in signed.items() if k!='amendment_history'}}
            for i,p in enumerate(pending):
                if p.id in eligible and p.id not in keep:
                    cancel_pending(db,p,actor,project_lifecycle(db,p)['revision'],'amend-'+str(req.request_id)+'-'+str(i))
            new_ids=[]
            for item,binding in pairs:
                if binding:new_ids.append(binding);continue
                p=ClinicalPendingORM(id=str(uuid4()),patient_id=patient_id,source_encounter_id=row.id,
                    item_type=item['item_type'],description=item['description'],trigger_text=item['trigger'],status=item['status'],
                    responsible_role=item['responsible_role'],external_dependency_json=item['external_dependency'],review_date=item['review_date'],
                    provenance_json=item['provenance'],resolution_event_id=None,created_at=now,updated_at=now)
                db.add(p);new_ids.append(p.id)
            history.append({'previous':previous,'amended_at':now.isoformat(),'actor':[actor.principal_id,actor.role],
                'request_id':str(req.request_id),'digest':identity})
            row.payload_json={**row.payload_json,'_visit_capture_v1':{**signed,'candidate':normalized,'pending_ids':new_ids,
                'signed_at':now.isoformat(),'revision':signed.get('revision',0)+1,'amendment_history':history}}
            row.encounter_date=normalized['encounter_date'];row.status='amended';row.updated_at=now;patient.updated_at=now
            try:db.commit()
            except IntegrityError:db.rollback();raise HTTPException(409,'Visit or obligations changed') from None
            return {'encounter_id':row.id,'revision':signed.get('revision',0)+1,'replayed':False}
