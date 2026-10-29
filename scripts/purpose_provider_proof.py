"""Invented Greek TTS pipeline check; no clinical encounter writes or real source."""
import json, os, time, unicodedata
from pathlib import Path
from uuid import uuid4
import httpx

BASE = 'https://ortho-reception-backend.onrender.com'
def norm(value):
    return ''.join(c for c in unicodedata.normalize('NFD', value.lower()) if unicodedata.category(c) != 'Mn')

def require(condition, code):
    if not condition: raise RuntimeError(code)

result = {'source': 'invented Greek TTS; no actual patient note or recording', 'clinical_writes': 0}
try:
    with httpx.Client(base_url=BASE, timeout=650, follow_redirects=False) as client:
        r=client.post('/clinical/login', json={'key': os.environ['CLINICAL_DATA_KEY']})
        require(r.status_code == 200, 'authentication_failed')
        r=client.get('/clinical/patient/cockpit-cp1-test/chart')
        require(r.status_code == 200, 'synthetic_chart_unavailable')
        require(r.json()['demographics'].get('synthetic_scenario') == 'cockpit-cp1-v1', 'synthetic_marker_missing')
        r=client.post('/clinical/visit-capture/context', json={'patient_id':'cockpit-cp1-test', 'client_session_id':'purpose-proof-'+uuid4().hex})
        require(r.status_code == 200, 'transient_context_unavailable')
        context=r.json()['context_id']
        started=time.monotonic()
        r=client.post('/clinical/visit-capture/audio/draft', headers={
            'X-Visit-Context':context, 'X-Encounter-Date':'2026-10-29',
            'X-Visit-Retention-Acknowledged':'standard', 'Content-Type':'audio/mp4'},
            content=Path('scripts/purpose-synthetic.m4a').read_bytes())
        result['elapsed_seconds']=round(time.monotonic()-started,2)
        result['http_status']=r.status_code
        require(r.status_code == 200, 'audio_http_failed')
        data=r.json()
        result['processing_status']=data.get('processing_status')
        result['contextual_status']=data.get('contextual_considerations_status')
        result['failure_stage']=(data.get('processing_failure') or {}).get('stage')
        # Synthetic only. Do not persist recovery capability, context or authentication data.
        result['transcript']=data.get('transcript')
        result['candidate']=data.get('candidate')
        require(data.get('processing_status') == 'complete', 'audio_not_complete')
        candidate=data['candidate'];raw=norm(json.dumps(candidate,ensure_ascii=False))
        focus=candidate.get('next_visit_focus',[])
        focus_text=norm(' '.join(v['text']+' '+v['rationale'] for v in focus))
        active=norm(json.dumps([v for v in candidate.get('pending',[]) if v.get('status','open') in ('open','awaiting_result','awaiting_patient')],ensure_ascii=False))
        checks={
          'purpose_established':not candidate.get('reason_is_unspecified',False) and 'οστεοπορ' in norm(candidate['reason_for_visit']['text']),
          'no_fracture_preserved':'δεν αναγνωριζεται σπονδυλικο καταγμα' in raw,
          'no_neural_compression_preserved':'σημαντικη νευρικη συμπιεση' in raw,
          'dexa_followthrough_preserved':'ιανουαριο' in raw and 'οστικης πυκνοτητας' in raw,
          'endocrine_followthrough_preserved':'ενδοκρινολογ' in raw,
          'conditional_therapy_preserved':'δεν αποφασιστηκε' in raw,
          'incidental_details_retained':'δισκικου υλικου' in raw and 'νεφρικη' in raw,
          'one_to_three_grounded_suggestions':1<=len(focus)<=3 and all(v['origin']=='ai_suggested' and v['status']=='suggestion' and v['source_excerpts'] for v in focus),
          'relevant_followthrough_focus':('dexa' in focus_text or 'οστικ' in focus_text or 'ενδοκρινολογ' in focus_text),
          'incidental_not_promoted':not any(v in focus_text for v in ['νεφρ','δισκικου υλικου','εξοδος δισκ']),
          'completed_referral_not_active':'παραπεμπτικ' not in active,
          'no_new_task_for_completed_advice_or_scheduled_followup':candidate.get('pending',[])==[],
          'short_complete_questions':all(len(v['text'].split())<=28 and v['text'].rstrip().endswith(('?',';',';')) for v in focus),
          'greek_latin_letters_only':all(not ch.isalpha() or unicodedata.name(ch,'').startswith(('GREEK','LATIN')) for v in focus for ch in v['text']),
        }
        result['checks']=checks
        require(all(checks.values()), 'synthetic_acceptance_failed')
        result['verdict']='PASS'
except Exception as exc:
    # Exception text can contain HTTP request details; record only controlled codes.
    result['verdict']='FAIL'
    result['failure_code']=str(exc) if type(exc) is RuntimeError else type(exc).__name__
finally:
    Path('purpose-provider-proof.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('transcript','candidate')},ensure_ascii=False))
require(result['verdict']=='PASS','provider_proof_failed')
