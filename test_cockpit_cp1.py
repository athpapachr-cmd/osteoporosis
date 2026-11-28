"""CP1 contract + real browser/DB workflow; provider mocked only in local tests."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from clinical_data import EncounterORM, ClinicalPendingORM, VisitCaptureCandidateV1
from clinical_capture_cp1 import HeidiVisitFacts, TEST_ID, extract_candidate
from test_cockpit_synthetic_161 import setup, login, KEY

TEXT='Επανέλεγχος γόνατος. Ήπιος πόνος στην κάμψη. Να προσκομιστεί το CD της μαγνητικής. Να σταλεί το πόρισμα.'
HEIDI_REASON = 'Επανέλεγχος κινητικότητας γόνατος.'
HEIDI_FINDING = 'Παραμένει δυσφορία όταν κατεβαίνει σκαλιά.'
HEIDI_TEXT = f'Λόγος επίσκεψης:\n{HEIDI_REASON}\nΕυρήματα:\n{HEIDI_FINDING}\nΣημειώσεις:\nΣυνθετική πηγή δοκιμής.'
HEIDI_QUESTION = 'Έχει αλλάξει η δυσκολία στα σκαλιά στην καθημερινότητα;'


@pytest.fixture
def heidi_provider(monkeypatch):
    import openai

    fact = lambda text: {"text": text, "provenance": ["heidi_today"]}
    record = HeidiVisitFacts.model_validate({
        "schema_version": "visit_capture_candidate_v1", "source_identity_state": "consistent",
        "encounter_date": "2026-10-18", "reason_for_visit": fact(HEIDI_REASON),
        "source_bindings": [{"source_ref": "heidi_today", "role": "heidi_today"}],
        "comparison_basis": {"type": "unavailable"}, "findings": [fact(HEIDI_FINDING)],
    })
    hint = {
        "category": "history_question", "text": HEIDI_QUESTION,
        "rationale": "Βοηθά να εκτιμηθεί η λειτουργικότητα στην επόμενη επαφή.",
        "source_ids": [1], "confidence": "moderate", "purpose_source_ids": [0],
        "role_scope": "within_role", "impact": "continuity",
    }
    topic_only = {**hint, "text": "Έχει αλλάξει ο σκοπός της επίσκεψης;", "source_ids": [0]}
    state = {"record": record, "calls": [],
             "advisory": json.dumps({"contextual_clinical_considerations": [{"invalid": True}, topic_only, hint, hint]})}

    def parse(**kwargs):
        state["calls"].append(("facts", kwargs))
        return SimpleNamespace(status="completed", output_parsed=state["record"])

    class Advisor:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(create=self.create)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def create(self, **kwargs):
            state["calls"].append(("advice", kwargs))
            return SimpleNamespace(status="completed", output_text=state["advisory"])

    monkeypatch.setenv("CLINICAL_DOCUMENTS_AI_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-only")
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    monkeypatch.setattr(openai, "AsyncOpenAI", Advisor)
    return state

def draft():
    fact=lambda s:{'text':s,'provenance':['dia_summary']}
    return VisitCaptureCandidateV1.model_validate({
        'schema_version':'visit_capture_candidate_v1','source_identity_state':'consistent','encounter_date':'2026-10-18',
        'reason_for_visit':fact('Επανέλεγχος γόνατος'),
        'source_bindings':[{'source_ref':'dia_summary','role':'dia_summary'}],
        'next_contact':{'purpose':'Κλινικός επανέλεγχος','assess':['Κάμψη γόνατος'],'provenance':['dia_summary']},
        'comparison_basis':{'type':'unavailable'},'findings':[fact('Ήπιος πόνος στην κάμψη')],
        'pending':[{'item_type':'investigation','description':s,'status':'awaiting_patient','responsible_role':'patient',
                    'provenance':['dia_summary'],'source_excerpt':s} for s in ['Να προσκομιστεί το CD της μαγνητικής.','Να σταλεί το πόρισμα.']]})

def ctx(c,patient=TEST_ID):
    return c.post('/clinical/visit-capture/context',json={'patient_id':patient,'client_session_id':'cp1-browser-test-session'}).json()['context_id']


def test_heidi_advice_is_generated_from_evidence_and_stays_out_of_tasks(setup, heidi_provider):
    app, engine = setup
    c = login(app)
    c.post('/clinical/visit-capture/test-patient')
    context = ctx(c)
    body = {"context_id": context, "text": HEIDI_TEXT, "encounter_date": "2026-10-18", "source_kind": "heidi"}
    raw = c.post('/clinical/visit-capture/extract', json=body)
    assert raw.status_code == 200
    result = raw.json()
    candidate = result["candidate"]
    assert result["visit_focus_status"] == "ready"
    assert len(candidate["next_visit_focus"]) == 1
    assert candidate["next_visit_focus"][0]["text"] == HEIDI_QUESTION and HEIDI_QUESTION not in HEIDI_TEXT
    assert candidate["next_visit_focus"][0]["source_excerpts"] == [HEIDI_FINDING]
    assert not candidate["pending"] and not candidate["recommendations"]
    (facts_kind, facts), (advice_kind, advice) = heidi_provider["calls"]
    assert (facts_kind, advice_kind) == ("facts", "advice")
    assert facts["store"] is False and advice["store"] is False and facts["model"] == advice["model"]
    schema = facts["text_format"].model_json_schema()["properties"]
    assert schema["next_visit_focus"]["maxItems"] == schema["pending"]["maxItems"] == 0
    evidence = advice["input"][1]["content"]
    assert HEIDI_REASON in evidence and HEIDI_FINDING in evidence
    source_payload = json.loads(evidence)
    assert source_payload["review_topic"] == {"source_id": 0, "label": HEIDI_REASON}
    assert [fact["excerpt"] for fact in source_payload["validated_facts"]] == [HEIDI_FINDING]
    assert "selected_identity" not in evidence and "Συνθετική πηγή δοκιμής" not in evidence
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM)) == 0

    heidi_provider["advisory"] = "malformed"
    recovered = c.post('/clinical/visit-capture/extract', json={
        **body, "text": HEIDI_TEXT + "\nΠροτάσεις AI:\nΣυνθετική παλαιότερη σύσταση."
        "\nΠροτάσεις του θεράποντος:\nΣυνθετική πρόταση γιατρού."
        "\nΠροτάσεις ασθενούς:\nΣυνθετική πρόταση ασθενούς.",
    }).json()
    assert recovered["visit_focus_status"] == "unavailable"
    assert recovered["candidate"]["findings"] == candidate["findings"]
    assert recovered["candidate"]["next_visit_focus"] == []
    facts_source = heidi_provider["calls"][-2][1]["input"][1]["content"]
    assert "Συνθετική παλαιότερη σύσταση" not in facts_source
    assert "Συνθετική πρόταση γιατρού" in facts_source and "Συνθετική πρόταση ασθενούς" in facts_source

    negative = c.post('/clinical/visit-capture/extract', json={
        **body, "text": HEIDI_TEXT.replace(HEIDI_FINDING, "Δεν ισχύει πλέον ότι: " + HEIDI_FINDING),
    }).json()
    assert negative["visit_focus_status"] == "insufficient_evidence"
    assert negative["candidate"]["next_visit_focus"] == []

    grouped = c.post('/clinical/visit-capture/extract', json={
        **body, "text": HEIDI_TEXT.replace(HEIDI_FINDING, "Δεν ισχύει πλέον\n\n- " + HEIDI_FINDING),
    }).json()
    assert grouped["visit_focus_status"] == "insufficient_evidence"
    assert grouped["candidate"]["next_visit_focus"] == []

    heidi_provider["record"].source_identity_state = "conflict"
    unmatched = c.post('/clinical/visit-capture/extract', json=body).json()
    assert unmatched["candidate"]["source_identity_state"] == "conflict"
    assert [name for name, _ in heidi_provider["calls"]] == ["facts", "advice", "facts", "advice", "facts", "facts", "facts"]

    heidi_provider["record"].source_identity_state = "consistent"
    bad_task = {"item_type": "investigation", "description": "επανάληψη της εξέτασης",
                "status": "open", "responsible_role": "unspecified",
                "provenance": ["heidi_today"], "source_excerpt": "επανάληψη της εξέτασης"}
    heidi_provider["record"] = VisitCaptureCandidateV1.model_validate(
        {**heidi_provider["record"].model_dump(), "pending": [bad_task]})
    assert c.post('/clinical/visit-capture/extract', json={
        **body, "text": HEIDI_TEXT + "\nΔεν συμφωνήθηκε επανάληψη της εξέτασης.",
    }).status_code == 502
    assert [name for name, _ in heidi_provider["calls"]][-2:] == ["facts", "facts"]

    stale = deepcopy(candidate)
    stale["findings"][0]["text"] = "Δεν ισχύει πλέον ότι " + HEIDI_FINDING
    assert c.post('/clinical/visit-capture/preview', json={"context_id": context, "candidate": stale}).status_code == 422
    uncertain = deepcopy(candidate)
    uncertain["findings"][0]["certainty"] = "uncertain"
    assert c.post('/clinical/visit-capture/save', json={"context_id": context, "candidate": uncertain}).status_code == 422
    assert c.post('/clinical/visit-capture/save', json={"context_id": context, "candidate": candidate}).status_code == 200
    chart = c.get('/clinical/patient/' + TEST_ID + '/chart').json()
    assert chart["visits"][0]["clinical"]["next_visit_focus"][0]["text"] == HEIDI_QUESTION
    assert chart["pending"] == []


def test_heidi_bold_bullet_headings_allow_grounded_advice_with_paraphrased_purpose(setup, heidi_provider):
    app, _ = setup
    c = login(app)
    c.post('/clinical/visit-capture/test-patient')
    context = ctx(c)
    excerpt = HEIDI_FINDING.replace("όταν κατεβαίνει", "όταν  κατεβαίνει").rstrip(".") + "."
    source = f"* **Συμπτώματα:**\n* Συζητήθηκε η βάδιση.\n\n* **Εξετάσεις/Απεικονίσεις:**\n* {excerpt}"
    assert HEIDI_REASON not in source
    body = {"context_id": context, "encounter_date": "2026-10-18", "source_kind": "heidi"}
    result = c.post('/clinical/visit-capture/extract', json={**body, "text": source}).json()
    assert result["visit_focus_status"] == "ready"
    assert result["candidate"]["next_visit_focus"][0]["source_excerpts"] == [HEIDI_FINDING]
    sent = json.loads(heidi_provider["calls"][-1][1]["input"][1]["content"])
    assert sent["validated_facts"][0]["excerpt"] == excerpt

    nested_headings = ("  **Συμπτώματα:**", "  ### Συμπτώματα", "  Συμπτώματα:")
    for qualified in (
        source.replace(excerpt, f"Δεν αναφέρει πλέον π.χ. {excerpt}"),
        source.replace("* Συζητήθηκε η βάδιση.", "Δεν τεκμηριώνεται το ακόλουθο"),
        source.replace("* Συζητήθηκε η βάδιση.", "* Δεν ανέφερε τα εξής:\n  - Πρόσφατη πτώση."),
        *(f"Ευρήματα:\n* Δεν ανέφερε τα εξής:\n{heading}\n  - {excerpt}" for heading in nested_headings),
    ):
        result = c.post('/clinical/visit-capture/extract', json={**body, "text": qualified}).json()
        assert result["visit_focus_status"] == "insufficient_evidence"
        assert not result["candidate"]["next_visit_focus"]
    assert [name for name, _ in heidi_provider["calls"]] == ["facts", "advice"] + ["facts"] * 6


def test_signed_legacy_heidi_advice_does_not_hide_visit_or_block_an_amendment(setup, heidi_provider):
    from clinical_pending_one_tap import PendingEventBase
    from clinical_visit_preparation import _signed_candidates

    app, engine = setup
    PendingEventBase.metadata.create_all(engine)
    c = login(app)
    c.post('/clinical/visit-capture/test-patient')
    context = ctx(c)
    candidate = c.post('/clinical/visit-capture/extract', json={
        "context_id": context, "encounter_date": "2026-10-18", "source_kind": "heidi", "text": HEIDI_TEXT,
    }).json()["candidate"]
    saved = c.post('/clinical/visit-capture/save', json={"context_id": context, "candidate": candidate})
    encounter = saved.json()["encounter"]["encounter_id"]
    path = f'/clinical/patient/{TEST_ID}/visit/{encounter}'
    legacy = deepcopy(candidate["next_visit_focus"][0])
    legacy["source_excerpts"].append(HEIDI_REASON)
    with Session(engine) as db:
        row = db.get(EncounterORM, encounter)
        payload = deepcopy(row.payload_json)
        payload["_visit_capture_v1"]["candidate"]["next_visit_focus"].append(legacy)
        row.payload_json = payload
        db.commit()
        visits, partial = _signed_candidates(db, TEST_ID, datetime.now(timezone.utc) + timedelta(seconds=10))
        assert not partial and len(visits) == 1
        assert visits[0][3].findings[0].text == HEIDI_FINDING

    edit = c.get(path + '/edit').json()
    assert edit["omitted_visit_focus"] == 1
    assert edit["candidate"]["next_visit_focus"] == candidate["next_visit_focus"]
    edited = c.post(path + '/amend', json={
        "context_id": ctx(c), "expected_token": edit["expected_token"], "request_id": str(uuid4()),
        "candidate": edit["candidate"], "pending_bindings": edit["pending_bindings"],
    })
    assert edited.status_code == 200, edited.text
    with Session(engine) as db:
        current = db.get(EncounterORM, encounter).payload_json["_visit_capture_v1"]
        assert current["amendment_history"][0]["previous"]["signed"]["candidate"]["next_visit_focus"][-1] == legacy


def test_saved_heidi_review_creates_only_unsaved_grounded_suggestions(setup, heidi_provider):
    from clinical_pending_one_tap import PendingEventBase
    app, engine = setup
    PendingEventBase.metadata.create_all(engine)
    c = login(app)
    c.post('/clinical/visit-capture/test-patient')
    context = ctx(c)
    candidate = c.post('/clinical/visit-capture/extract', json={
        "context_id": context, "encounter_date": "2026-10-18", "source_kind": "heidi", "text": HEIDI_TEXT,
    }).json()["candidate"]
    candidate["next_visit_focus"] = []
    encounter = c.post('/clinical/visit-capture/save', json={"context_id": context, "candidate": candidate}).json()["encounter"]["encounter_id"]
    path = f'/clinical/patient/{TEST_ID}/visit/{encounter}'
    edit = c.get(path + '/edit').json()
    before = c.get('/clinical/patient/' + TEST_ID + '/chart').json()
    result = c.post(path + '/review', json={'expected_token': edit['expected_token']})
    assert result.status_code == 200
    proposal = result.json()['suggestions'][0]
    assert proposal['text'] == HEIDI_QUESTION
    assert proposal['evidence_basis'] == 'saved_visit'
    assert proposal['source_excerpts'] == [HEIDI_FINDING]
    assert c.get('/clinical/patient/' + TEST_ID + '/chart').json() == before
    bad = c.post(path + '/review', json={'expected_token': '0' * 64})
    assert bad.status_code == 409
    amendment = {'context_id': ctx(c), 'expected_token': edit['expected_token'], 'request_id': str(uuid4()),
                 'candidate': edit['candidate'], 'pending_bindings': edit['pending_bindings']}
    amendment['candidate']['next_visit_focus'] = [proposal]
    assert c.post(path + '/amend', json=amendment).status_code == 200
    stored = c.get('/clinical/patient/' + TEST_ID + '/chart').json()
    assert stored['visits'][0]['clinical']['next_visit_focus'][0]['evidence_basis'] == 'saved_visit'


def test_proposal_approval_atomic_persistence_and_exact_origin(setup,monkeypatch):
    app,engine=setup;c=login(app)
    assert c.post('/clinical/visit-capture/test-patient').status_code==200
    assert TEST_ID not in [p['patient_id'] for p in c.get('/clinical/patients').json()]
    monkeypatch.setattr('clinical_capture_cp1.extract_candidate',lambda *a:draft())
    context=ctx(c)
    result=c.post('/clinical/visit-capture/extract',json={'context_id':context,'text':TEXT,'encounter_date':'2026-10-18'})
    assert result.status_code==200
    with Session(engine) as db: assert db.scalar(select(func.count()).select_from(ClinicalPendingORM))==0
    candidate=result.json()['candidate'];candidate['pending'].pop();candidate['findings'][0]['text']='Διορθωμένο εύρημα'
    body={'context_id':context,'candidate':candidate}
    saved=c.post('/clinical/visit-capture/save',json=body);assert saved.status_code==200,saved.text
    assert c.post('/clinical/visit-capture/save',json=body).status_code==409
    second=login(app)
    chart=second.get('/clinical/patient/'+TEST_ID+'/chart').json()
    assert chart['visits'][0]['clinical']['findings'][0]['text']=='Διορθωμένο εύρημα'
    assert len(chart['pending'])==1 and chart['pending'][0]['source_excerpt']=='Να προσκομιστεί το CD της μαγνητικής.'
    assert chart['pending'][0]['source_encounter_id']==saved.json()['encounter']['encounter_id']
    assert second.get('/clinical/clinical-updates').json()['items']==[]
    assert second.get('/clinical/clinical-updates?scope=test').json()['items'][0]['pending_id']==chart['pending'][0]['pending_id']
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM))==1
        payload=db.scalar(select(EncounterORM)).payload_json
        assert TEXT not in str(payload)


def test_stale_after_provider_and_protected_routes(setup,monkeypatch):
    app,_=setup;c=login(app);c.post('/clinical/visit-capture/test-patient');context=ctx(c)
    def mutate(*args):
        c.post('/clinical/patients',json={'patient_id':TEST_ID,'demographics':{'full_name':'Changed'}})
        return draft()
    monkeypatch.setattr('clinical_capture_cp1.extract_candidate',mutate)
    assert c.post('/clinical/visit-capture/extract',json={'context_id':context,'text':TEXT,'encounter_date':'2026-10-18'}).status_code==409
    invalid=c.post('/clinical/visit-capture/extract',json={'context_id':context,'text':TEXT,'encounter_date':'bad'})
    assert invalid.status_code==422 and TEXT not in invalid.text
    oversized=c.post('/clinical/visit-capture/extract',json={'context_id':context,'text':TEXT*1000,'encounter_date':'2026-10-18'})
    assert oversized.status_code in (413,422) and TEXT not in oversized.text
    anonymous=TestClient(app)
    assert anonymous.get('/clinical/patient/'+TEST_ID+'/chart').status_code==401
    assert anonymous.get('/clinical/clinical-updates').status_code==401
    assert anonymous.post('/clinical/visit-capture/test-patient').status_code==401
    assert anonymous.post('/clinical/visit-capture/extract',json={'context_id':context,'text':TEXT,'encounter_date':'2026-10-18'}).status_code==401


def test_provider_boundary_rejects_fabricated_excerpt_completion_and_failure(monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException
    import openai
    monkeypatch.setenv('CLINICAL_DOCUMENTS_AI_ENABLED','true');monkeypatch.setenv('OPENAI_API_KEY','fake-local')
    monkeypatch.setenv('CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED','true')
    current=draft();calls=[]
    def parse(**kw):calls.append(kw);return SimpleNamespace(status='completed',output_parsed=current)
    monkeypatch.setattr(openai,'OpenAI',lambda **kw:SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    assert extract_candidate(TEXT,'2026-10-18',{}).pending
    assert calls[0]['store'] is False and 'tools' not in calls[0]
    current.pending[0].source_excerpt='invented'
    with pytest.raises(HTTPException) as error:extract_candidate(TEXT,'2026-10-18',{})
    assert error.value.status_code==502 and 'invented' not in str(error.value.detail)
    current=draft();current.pending[0].status='resolved'
    with pytest.raises(HTTPException):extract_candidate(TEXT,'2026-10-18',{})
    class Exhausted(Exception):
        code='credit_balance_exhausted'
    def exhausted(**kw):raise Exhausted('DO_NOT_ECHO_SOURCE_OR_PROVIDER_ERROR')
    monkeypatch.setattr(openai,'OpenAI',lambda **kw:SimpleNamespace(responses=SimpleNamespace(parse=exhausted)))
    with pytest.raises(HTTPException) as error:extract_candidate(TEXT,'2026-10-18',{})
    assert error.value.status_code==503 and error.value.detail['code']=='provider_budget_exhausted'
    assert 'DO_NOT_ECHO' not in str(error.value.detail)
    monkeypatch.setenv('CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED','false')
    with pytest.raises(HTTPException) as error:extract_candidate(TEXT,'2026-10-18',{})
    assert error.value.status_code==503


def test_browser_heidi_paste_runs_one_processing_action_and_edits_grounding(setup, heidi_provider):
    import shutil
    import socket
    import threading

    from playwright.sync_api import expect, sync_playwright
    import uvicorn

    app, engine = setup
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=shutil.which("google-chrome") or shutil.which("chromium"))
            page = browser.new_page(viewport={"width": 390, "height": 950})
            page.set_default_timeout(8000)
            page.goto(f"http://localhost:{sock.getsockname()[1]}/static/cockpit/index.html")
            page.locator('#ceGlobalKey').click()
            page.locator('#ceGlobalKeyInput').fill(KEY)
            page.locator('#ceGlobalLoginForm button[type=submit]').click()
            expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες')
            page.keyboard.press('Escape')
            page.locator('#captureTestPatient').click()
            page.locator('#captureOpen').click()
            page.get_by_role("button", name="Σύνοψη Heidi με AI", exact=True).click()
            expect(page.locator(".capture-input-help")).to_contain_text("στην OpenAI")
            page.locator('#captureDate').fill('2026-10-18')
            page.locator('#captureText').fill(HEIDI_TEXT)
            expect(page.locator('#captureSave')).to_be_hidden()
            assert not heidi_provider["calls"]

            action = page.get_by_role("button", name="Επεξεργασία σύνοψης Heidi", exact=True)
            heidi_provider["record"].source_identity_state = "uncertain"
            action.click()
            expect(page.locator(".capture-identity-review")).to_contain_text("Χρειάζεται έλεγχος της καρτέλας")
            expect(page.locator("#captureText")).to_have_value(HEIDI_TEXT)
            expect(page.locator("#captureReview")).to_be_hidden()
            expect(page.locator("#captureReview textarea")).to_have_count(0)
            expect(page.locator("#captureSave")).to_be_hidden()
            assert [name for name, _ in heidi_provider["calls"]] == ["facts"]

            heidi_provider["record"].source_identity_state = "consistent"
            action.click()
            summary = page.locator("#captureAudioSummary")
            focus = summary.locator(".next-visit-focus")
            expect(focus).to_contain_text(HEIDI_QUESTION)
            expect(focus.get_by_text("Βοηθά να εκτιμηθεί η λειτουργικότητα στην επόμενη επαφή.")).to_be_visible()
            expect(summary.get_by_role("heading", name="Υποχρεώσεις", exact=True)).to_have_count(0)
            expect(page.locator("#captureSave")).to_have_text("Αποθήκευση επίσκεψης")
            focus.get_by_text("Σύνδεση με τον σκοπό", exact=True).click()
            expect(focus.locator("blockquote")).to_have_text(HEIDI_FINDING)
            assert [name for name, _ in heidi_provider["calls"]] == ["facts", "facts", "advice"]
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.set_viewport_size({"width": 1440, "height": 960})
            clinical_box = summary.locator(".capture-result-clinical").bounding_box()
            focus_box = focus.bounding_box()
            assert clinical_box["y"] == focus_box["y"] and clinical_box["x"] < focus_box["x"]
            with Session(engine) as db:
                assert db.scalar(select(func.count()).select_from(EncounterORM)) == 0
            summary.get_by_role("button", name="Διόρθωση · Ευρήματα", exact=True).click()
            editor = page.locator("#captureReview")
            editor.get_by_text("Λεπτομέρειες", exact=True).click()
            editor.get_by_label("Βεβαιότητα").select_option("uncertain")
            editor.get_by_role("button", name="Επιστροφή στην καταγραφή", exact=True).click()
            expect(focus.locator("article")).to_have_count(0)
            page.locator("#captureSave").click()
            expect(page.locator("#captureComposer")).to_be_hidden()

            saved_chart = page.locator(".patient-chart-overview")
            snapshot = saved_chart.locator(".patient-chart-snapshot")
            full = saved_chart.locator(":scope > .clinical-full-record")
            expect(snapshot.locator(".next-visit-focus")).to_contain_text(
                "Δεν αποθηκεύτηκαν προτάσεις AI για αυτή την επίσκεψη.")
            expect(snapshot.locator(".patient-chart-obligations")).to_have_count(0)
            expect(full).not_to_have_attribute("open", "")
            assert abs(snapshot.bounding_box()["width"] - snapshot.locator(".patient-chart-card").bounding_box()["width"]) < 1

            page.locator("#captureOpen").click()
            page.get_by_role("button", name="Σύνοψη Heidi με AI", exact=True).click()
            heidi_provider["record"].encounter_date = "2026-10-19"
            page.locator("#captureDate").fill("2026-10-19")
            page.locator("#captureText").fill(HEIDI_TEXT)
            action.click()
            expect(focus).to_contain_text(HEIDI_QUESTION)
            page.locator("#captureSave").click()
            expect(page.locator("#captureComposer")).to_be_hidden()

            advice_card = snapshot.locator(".patient-chart-next-visit")
            expect(advice_card.get_by_text(HEIDI_QUESTION)).to_be_visible()
            expect(advice_card.get_by_role("button", name="Αφαίρεση πρότασης")).to_have_count(0)
            for width in (1440, 390):
                page.set_viewport_size({"width": width, "height": 950})
                overview_box = snapshot.locator(".patient-chart-card").first.bounding_box()
                plan_box = advice_card.bounding_box()
                if width == 1440:
                    assert abs(overview_box["y"] - plan_box["y"]) < 1 and plan_box["x"] > overview_box["x"]
                else:
                    assert plan_box["y"] >= overview_box["y"] + overview_box["height"]
                expect(full).not_to_have_attribute("open", "")
                full.locator(":scope > summary").click()
                expect(full.get_by_text(HEIDI_FINDING, exact=True)).to_be_visible()
                assert abs(snapshot.bounding_box()["width"] - full.bounding_box()["width"]) < 1
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                full.locator(":scope > summary").click()
            browser.close()
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM)) == 2
        assert db.scalar(select(func.count()).select_from(ClinicalPendingORM)) == 0


def test_browser_saved_heidi_visit_can_generate_and_save_advice(setup, heidi_provider):
    import shutil
    import socket
    import threading
    import uvicorn
    from playwright.sync_api import expect, sync_playwright
    from clinical_pending_one_tap import PendingEventBase

    app, engine = setup
    PendingEventBase.metadata.create_all(engine)
    c = login(app)
    c.post('/clinical/visit-capture/test-patient')
    context = ctx(c)
    candidate = c.post('/clinical/visit-capture/extract', json={
        "context_id": context, "encounter_date": "2026-10-18", "source_kind": "heidi", "text": HEIDI_TEXT,
    }).json()["candidate"]
    candidate['next_visit_focus'] = []
    encounter = c.post('/clinical/visit-capture/save', json={
        'context_id': context, 'candidate': candidate,
    }).json()['encounter']['encounter_id']
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium'))
            page = browser.new_page(viewport={'width': 1280, 'height': 950})
            page.goto(f'http://localhost:{sock.getsockname()[1]}/static/cockpit/index.html')
            page.locator('#ceGlobalKey').click()
            page.locator('#ceGlobalKeyInput').fill(KEY)
            page.locator('#ceGlobalLoginForm button[type=submit]').click()
            expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες')
            page.keyboard.press('Escape')
            page.evaluate('([id,eid])=>window.CockpitStoredChart.openVisit(id,eid)', [TEST_ID, encounter])
            page.get_by_role('button', name='Επεξεργασία επίσκεψης', exact=True).click()
            page.get_by_role('button', name='Έλεγχος επίσκεψης με AI', exact=True).click()
            focus = page.locator('#captureAudioSummary .next-visit-focus')
            expect(focus).to_contain_text(HEIDI_QUESTION)
            focus.get_by_text('Σύνδεση με τον σκοπό').click()
            expect(focus).to_contain_text('Από την αποθηκευμένη επίσκεψη')
            assert c.get('/clinical/patient/' + TEST_ID + '/chart').json()['visits'][0]['clinical']['next_visit_focus'] == []
            page.locator('#captureSave').click()
            expect(page.locator('#captureComposer')).to_be_hidden()
            assert c.get('/clinical/patient/' + TEST_ID + '/chart').json()['visits'][0]['clinical']['next_visit_focus'][0]['text'] == HEIDI_QUESTION
            browser.close()
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()


def test_browser_capture_edit_reject_save_refresh_and_second_session(setup,monkeypatch,tmp_path):
    import socket,threading,shutil
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    app,engine=setup
    started=threading.Event();release=threading.Event();first=[True]
    def provider(*args):
        if first[0]:
            first[0]=False;started.set();release.wait(15)
        return draft()
    monkeypatch.setattr('clinical_capture_cp1.extract_candidate',provider)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    url=f'http://localhost:{port}/static/cockpit/index.html'
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium') or '/Applications/Dia.app/Contents/MacOS/Dia')
            for width in [1280,390]:
                context=browser.new_context(viewport={'width':width,'height':950},permissions=['clipboard-read','clipboard-write']);page=context.new_page();errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)));page.goto(url)
                page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(KEY)
                page.locator('#ceGlobalLoginForm button[type=submit]').click()
                expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες')
                page.keyboard.press('Escape')
                page.locator('#captureTestPatient').click();expect(page.locator('#captureOpen')).to_be_visible()
                page.locator('#captureOpen').click();page.locator("#captureModeAI").click()
                page.locator('#captureCopyDiaPrompt').click()
                expect(page.locator('#captureDiaPromptStatus')).to_contain_text('Αντιγράφηκε')
                copied=page.evaluate('navigator.clipboard.readText()')
                assert copied==page.locator('#captureDiaPromptText').text_content()
                assert 'ενεργής επίσκεψης' in copied and 'Δοκιμαστικός ασθενής' not in copied and '1972-01-01' not in copied
                page.locator('#captureText').fill('Το πρόχειρο παραμένει μετά την αντιγραφή')
                page.evaluate("()=>{navigator.clipboard.writeText=async()=>{throw new Error('Clipboard denied')};}")
                page.locator('#captureCopyDiaPrompt').click()
                expect(page.locator('#captureDiaPromptText')).to_be_visible()
                expect(page.locator('#captureDiaPromptStatus')).to_contain_text('Επίλεξε και αντέγραψε')
                expect(page.locator('#captureText')).to_have_value('Το πρόχειρο παραμένει μετά την αντιγραφή')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                page.locator('.capture-dia-prompt:visible').screenshot(path=str(tmp_path/f'dia-prompt-{width}.png'))
                page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
                if width==1280:
                    page.locator('#captureOpen').click();page.locator("#captureModeAI").click();page.locator('#captureText').fill(TEXT)
                    page.get_by_role('button',name='Προετοιμασία καταγραφής',exact=True).click()
                    assert started.wait(5)
                    page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
                    page.locator('#captureOpen').click();page.locator("#captureModeAI").click()
                    with page.expect_response('**/visit-capture/extract'):
                        release.set()
                    expect(page.locator('#captureSave')).to_be_hidden()
                    expect(page.locator('#captureReview')).to_be_empty()
                    page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
                    page.locator('#captureOpen').click();page.locator("#captureModeAI").click();page.locator('#captureText').fill(TEXT)
                    page.get_by_role('button',name='Προετοιμασία καταγραφής',exact=True).click()
                    expect(page.locator('#captureSave')).to_be_visible(timeout=15000)
                    expect(page.locator('#captureReview')).to_be_hidden()
                    composer=page.locator('#captureComposer')
                    result=page.locator('#captureAudioSummary')
                    editor=page.locator('#captureReview')
                    expect(result).to_contain_text('Να προσκομιστεί το CD')
                    expect(result.locator('.clinical-full-record')).not_to_have_attribute('open','')
                    expect(page.locator('#captureText')).to_be_hidden()
                    expect(page.locator('#captureSave')).to_be_in_viewport()
                    held_previews=[]
                    page.route('**/visit-capture/preview',lambda route: held_previews.append(route))
                    with page.expect_request('**/visit-capture/preview'):
                        page.locator('#captureSave').click()
                    page.wait_for_function("() => document.querySelector('#captureSave').disabled")
                    composer.get_by_role('button',name='Κλείσιμο',exact=True).click()
                    page.locator('#captureOpen').click()
                    page.locator('#captureModeAI').click()
                    page.locator('#captureText').fill(TEXT)
                    composer.get_by_role('button',name='Προετοιμασία καταγραφής',exact=True).click()
                    expect(result).to_be_visible()
                    with page.expect_response('**/visit-capture/preview'):
                        held_previews[0].fulfill(json={'can_save': True})
                    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
                    expect(page.locator('#captureStatus')).to_contain_text('Η καταγραφή είναι έτοιμη')
                    page.unroute('**/visit-capture/preview')
                    def edit_section(label):
                        result.get_by_role('button',name='Διόρθωση · '+label,exact=True).click()
                        expect(editor).to_be_visible()
                    def finish_section():
                        editor.get_by_role('button',name='Επιστροφή στην καταγραφή',exact=True).click()
                        expect(editor).to_be_hidden()
                    edit_section('Ευρήματα')
                    expect(editor.locator('section[data-field]')).to_have_count(1)
                    editor.locator('textarea').fill('Διορθωμένο εύρημα από τον γιατρό')
                    finish_section()
                    expect(result.get_by_role('button',name='Διόρθωση · Ευρήματα')).to_be_focused()
                    edit_section('Υποχρεώσεις')
                    editor.locator('section[data-field=pending]').get_by_role('button',name='Αφαίρεση',exact=True).last.click()
                    finish_section()
                    edit_section('Επόμενη επαφή')
                    assess=editor.locator('section[data-field=assess]')
                    assess.locator('input').fill('Αφαιρεθέν στοιχείο')
                    assess.get_by_role('button',name='Αφαίρεση').click()
                    finish_section()
                    expect(result).to_contain_text('Διορθωμένο εύρημα από τον γιατρό')
                    page.locator('#captureSave').click();expect(page.locator('#captureComposer')).to_be_hidden()
                expect(page.locator('#patientChart')).to_contain_text('Ενεργή υποχρέωση: Να προσκομιστεί το CD')
                page.reload();expect(page.locator('#patientChart')).to_contain_text('Να προσκομιστεί το CD')
                page.locator('.patient-journey-point').first.click()
                page.get_by_role('button',name='Άνοιγμα',exact=True).click()
                expect(page.locator('#patientChart')).to_contain_text('Διορθωμένο εύρημα από τον γιατρό')
                assert 'Να σταλεί το πόρισμα' not in page.locator('#patientChart').inner_text()
                assert 'Αφαιρεθέν στοιχείο' not in page.locator('#patientChart').inner_text()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.screenshot(path=str(tmp_path/f'cp1-{width}.png'),full_page=True)
                assert not errors,errors
                context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(timeout=10);sock.close()
    with Session(engine) as db:assert db.scalar(select(func.count()).select_from(EncounterORM))==1


@pytest.mark.parametrize("coding_state", [None, False, True])
def test_complete_user_case_visible_before_save_and_after_reload(setup, monkeypatch, tmp_path, coding_state):
    """The actual final chart card, not just the editable input, must retain each fact."""
    import socket, threading, shutil
    import uvicorn
    from playwright.sync_api import sync_playwright, expect
    app, engine = setup
    fact = lambda s: {"text": s, "provenance": ["dia_summary"], "verification": "source_stated"}
    fracture = "Κάταγμα πηχεοκαρπικής πριν δύο χρόνια."
    pain = "Πόνος στη θωρακική μοίρα εδώ και τρεις μήνες, κυρίως σε όρθια στάση και βάδιση, χωρίς νευρολογική σημειολογία."
    tscore = "T-score -2,8 στη σπονδυλική στήλη."
    therapy = "Σύσταση για έναρξη alendronate μετά από εκτίμηση από οδοντίατρο."
    case = VisitCaptureCandidateV1.model_validate({
        "schema_version": "visit_capture_candidate_v1", "source_identity_state": "consistent",
        "encounter_date": "2026-10-28", "reason_for_visit": fact(pain),
        "history": [fact(fracture)], "findings": [fact(tscore)],
        "safety_net": [fact('Επανεπικοινωνία για νέο σύμπτωμα.'), fact('Επανεπικοινωνία αν η κατάσταση επιδεινωθεί.')],
        "recommendations": [{**fact(therapy), "type": "therapy"}],
        "source_bindings": [{"source_ref": "dia_summary", "role": "dia_summary"}],
        "comparison_basis": {"type": "unavailable"},
        "pending": [{"item_type": "investigation", "description": "Ζητήθηκαν ακτινογραφίες.",
                     "source_excerpt": "Ζητήθηκαν ακτινογραφίες.", "provenance": ["dia_summary"],
                     "status": "open", "responsible_role": "unspecified"}]})
    monkeypatch.setattr("clinical_capture_cp1.extract_candidate", lambda *a: case)
    coding_text = "SYN · CODE-1 · Συνθετικός κωδικός δοκιμής"
    if coding_state is not None:
        from clinical_data import VisitCaptureCodingItem
        case.coding.gesy_coded.append(VisitCaptureCodingItem.model_validate({
            "system": "SYN", "code": "CODE-1", "title": "Συνθετικός κωδικός δοκιμής",
            "provenance": ["dia_summary"]}))
        case.coding.coding_complete = coding_state
    expected_texts = (fracture, pain, tscore, therapy) + ((coding_text, "Κωδικοποίηση: πλήρης." if coding_state else "Κωδικοποίηση: δεν έχει δηλωθεί πλήρης.") if coding_state is not None else ())
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True); thread.start()
    url = f"http://localhost:{port}/static/cockpit/index.html"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=shutil.which("google-chrome") or shutil.which("chromium") or "/Applications/Dia.app/Contents/MacOS/Dia")
            page = browser.new_page(viewport={"width": 1280, "height": 1000})
            page.goto(url)
            page.locator("#ceGlobalKey").click(); page.locator("#ceGlobalKeyInput").fill(KEY)
            page.locator("#ceGlobalLoginForm button[type=submit]").click()
            expect(page.locator("#ceGlobalAuthStatus")).to_contain_text("Συνδέθηκες"); page.keyboard.press("Escape")
            page.locator("#captureTestPatient").click(); page.locator("#captureOpen").click();page.locator("#captureModeAI").click()
            page.locator("#captureText").fill(" ".join([fracture, pain, tscore, therapy, "Ζητήθηκαν ακτινογραφίες."]))
            page.get_by_role("button", name="Προετοιμασία καταγραφής", exact=True).click()
            summary = page.locator("#captureAudioSummary")
            expect(summary).to_be_visible()
            expect(page.locator("#captureReview")).to_be_hidden()
            for text in expected_texts:
                expect(summary).to_contain_text(text)
            expect(summary.locator('.clinical-compact-overview').get_by_text('Επανεπικοινωνία αν η κατάσταση επιδεινωθεί.',exact=True)).to_be_visible()
            with Session(engine) as db:
                assert db.scalar(select(func.count()).select_from(EncounterORM)) == 0
            summary.screenshot(path=str(tmp_path / "complete-draft-1280.png"))
            page.locator("#captureSave").click()
            expect(page.locator("#captureComposer")).to_be_hidden()
            for width in (1280, 390):
                page.set_viewport_size({"width": width, "height": 1000})
                page.reload()
                saved_chart = page.locator(".patient-chart-overview")
                card = saved_chart.locator(".patient-chart-snapshot .patient-chart-card").first
                full = saved_chart.locator(":scope > .clinical-full-record")
                expect(card).to_be_visible()
                expect(full).not_to_have_attribute("open", "")
                full.locator(":scope > summary").click()
                for text in expected_texts:
                    expect(full.get_by_text(text, exact=True)).to_be_visible()
                expect(full).not_to_contain_text("Ξεκίνησε alendronate")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.locator(".patient-chart-overview .patient-chart-grid").screenshot(path=str(tmp_path / f"complete-chart-{width}.png"))
            data = page.request.get(f"http://localhost:{port}/clinical/patient/{TEST_ID}/chart").json()
            stored = data["visits"][0]["clinical"]
            assert stored["history"][0]["text"] == fracture
            assert stored["findings"][0]["text"] == tscore
            assert stored["recommendations"][0]["text"] == therapy
            page.evaluate("([id,eid])=>window.CockpitStoredChart.openVisit(id,eid)", [TEST_ID, data["visits"][0]["encounter_id"]])
            reader = page.locator(".patient-visit-reader")
            expect(reader).to_be_visible()
            for text in expected_texts:
                expect(reader).to_contain_text(text)
            browser.close()
    finally:
        server.should_exit = True; thread.join(timeout=10); sock.close()
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM)) == 1
