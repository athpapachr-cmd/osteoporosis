'use strict';

const fs = require('fs');
const assert = require('assert');

const app = fs.readFileSync('static/clinic-utilities/rf/app.js', 'utf8');
const html = fs.readFileSync('static/clinic-utilities/rf/index.html', 'utf8');
const css = fs.readFileSync('static/clinic-utilities/rf/styles.css', 'utf8');
const main = fs.readFileSync('main.py', 'utf8');

assert(app.includes("function numberOrNull(id)"), 'RF UI must preserve blank numeric fields as null');
assert(app.includes("download.download = 'RF_'"), 'PDF creation must trigger browser download');
assert(!app.includes("window.open(url, '_blank', 'noopener')"), 'PDF creation must not depend on popup window');
assert(html.includes('id="rfOutcomeCopy"'), 'Selected RF must expose a copy-to-new-PDF button');
assert(html.includes('id="rfOutcomeDelete"'), 'Selected RF must expose exact-ID deletion');
assert(app.includes('/api/reuse/issued-copy'), 'Issued form restore must use a protected request body');
assert(app.includes('expected_revision:item.revision'), 'Exact-ID deletion must check optimistic revision');
assert(app.includes('confirmation_code:code'), 'Deletion must confirm the chosen RF identifier');
assert(!app.includes('localStorage.setItem'), 'RF copy/delete identity must remain volatile');
for (const field of [
  'age',
  'painOnsetVas',
  'lastAssessmentVas',
  'interventionVasBefore',
  'interventionVasAfter',
  'legacyVasBefore',
  'legacyVasAfter',
  'legacyFollowupVas',
]) {
  assert(app.includes(`numberOrNull('${field}')`), `RF UI must use numberOrNull for ${field}`);
  assert(!app.includes(`Number($('${field}').value)`), `RF UI must not coerce blank ${field} to zero`);
}

assert(app.includes('/clinical/clinic-utilities/rf'), 'RF UI must use the native same-origin protected route');
assert(!app.includes('ortho-reception-backend-v2.onrender.com'), 'RF UI must not call the old RF service directly');
assert(html.includes('Clinical Excellence'), 'RF utility must use the Clinical Excellence shell');
assert(html.includes('Νέα θεραπεία'), 'A1 workflow must be visible');
assert(html.includes('Συνέχιση θεραπείας'), 'A2 workflow must be visible');
assert(!html.includes('Καρκινικός πόνος'), 'Category G workflow must not be exposed in this clinician-specific utility');
assert(!html.includes('Νευροπαθητικός πόνος'), 'Category B workflow must not be exposed in this clinician-specific utility');
for (const token of [
  'unknownMedicationPanel',
  'unknownMedicationRows',
  'addNsaidBtn',
  'addOtherBtn',
  'learnedMedicationDetails',
  '/api/medication-dictionary',
  'ΜΣΑΦ + μάθηση',
  'Άλλο + μάθηση',
  '✓ learned · clinician-confirmed',
]) {
  assert(html.includes(token) || app.includes(token), `RF medication-learning UI missing ${token}`);
}
assert(!app.includes('localStorage'), 'RF learned medication dictionary must not use localStorage');
assert(!app.includes('sessionStorage'), 'RF learned medication dictionary must not use sessionStorage');

assert(main.includes('app.include_router(build_rf_router(engine))'), 'main.py must mount the native RF router');
assert(!main.includes('app.include_router(build_rf_gateway_router())'), 'legacy RF gateway must not be mounted');

console.log('RF v2 UI integrity regression: PASS');


// Exercise the actual browser handlers with a small deterministic DOM shim.
const vm = require('vm');
const nodes = {};
const revoked = [];
let blobId = 0;
function fakeElement(tag) {
  return {
    tag, children: [], listeners: {}, className: '', textContent: '',
    get value() { return this._value ?? ''; },
    set value(next) { this._value = String(next ?? ''); }, // native input.value is always text
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    addEventListener(type, fn) { this.listeners[type] = fn; },
    querySelector() { return {value: '', onclick: null}; }, // synthetic adverse-row button
    querySelectorAll() { return []; }, // real DOM collection method for progress preview
    setAttribute() {},
    closest() { return {hidden: false}; },
  };
}
const doc = {
  getElementById(id) { return nodes[id] ||= fakeElement(id); },
  createElement: fakeElement,
  querySelectorAll() { return []; },
  addEventListener() {}, // install but do not run unrelated page bootstrap
};
const instrumented = app.replace(/\}\)\(\);\s*$/, '\n  globalThis.__rfTest = {normalizeRfIdentity, addHorosImages, releaseHorosPreviews, loadContract, saveDoctor, editDoctor, createPdf, populateRfFields, savedDraftFields, restoreRfDraft, loadRfSource, setPathway, resolveSourceName, invalidateSourceNameConfirmation, unresolvedSourceNameConflict, buildDraft, loadRfOutcomes, selectRfOutcome, invalidateRfOutcomeSelection, renderSelectedOutcome, applyRfOutcome, beginRfCorrection, outcomeAvailableActions, renderCorrectionContext};\n})();');
assert.notStrictEqual(instrumented, app, 'RF handler instrumentation point must be present');
const browser = {
  document: doc,
  URL: {
    createObjectURL() { return 'blob:rf-test-' + (++blobId); },
    revokeObjectURL(value) { revoked.push(value); },
  },
  window: {confirm:()=>true},
  history: {pushState(){}},
};
vm.runInNewContext(instrumented, browser, {filename: 'rf-app.js'});
assert.strictEqual(browser.__rfTest.normalizeRfIdentity('AB-W123'), 'ABW123');
assert.notStrictEqual(browser.__rfTest.normalizeRfIdentity('ABW123'),
                      browser.__rfTest.normalizeRfIdentity('AB123'));
assert.strictEqual(browser.__rfTest.normalizeRfIdentity('RF-CORR'),
                   browser.__rfTest.normalizeRfIdentity('RFCORR'));
const jpg = name => ({name, size: 1024, type: 'image/jpeg'});
const picker = doc.getElementById('horosJpegs');
picker.files = [jpg('first.jpg'), jpg('second.jpeg')];
browser.__rfTest.addHorosImages();
let grid = doc.getElementById('horosImageGrid');
assert.deepStrictEqual(grid.children.map(x => x.children[1].textContent), ['first.jpg', 'second.jpeg']);
assert.strictEqual(doc.getElementById('horosImageStatus').textContent, '2 εικόνες έτοιμες για επισύναψη.');
assert(grid.children.every(x => x.children[0].src.startsWith('blob:rf-test-')), 'thumbnails need blob URLs');
const removed = grid.children[0].children[0].src;
grid.children[0].children[2].listeners.click();
assert.deepStrictEqual(grid.children.map(x => x.children[1].textContent), ['second.jpeg']);
assert.strictEqual(doc.getElementById('horosImageStatus').textContent, '1 εικόνα έτοιμη για επισύναψη.');
assert(revoked.includes(removed), 'removal must revoke its blob URL');
picker.files = [jpg('third.jpg')];
browser.__rfTest.addHorosImages();
assert.deepStrictEqual(grid.children.map(x => x.children[1].textContent), ['second.jpeg', 'third.jpg']);
picker.files = [{name:'not-an-image.png', size:100, type:'image/png'}];
browser.__rfTest.addHorosImages();
assert.strictEqual(grid.children.length, 2);
assert(doc.getElementById('horosImageStatus').textContent.includes('JPEG'), 'error should be displayed');
browser.__rfTest.releaseHorosPreviews();
assert.strictEqual(revoked.length, 3, 'all preview URLs must be released');
assert(html.includes('id="horosJpegs"') && html.includes('multiple'), 'multiple file selection is required');
assert(html.includes('class="upload full-width horos-upload"') && html.includes('class="horos-picker-action"') && html.includes('Προσθήκη εικόνων'), 'Horos picker must have a visible Greek upload action');
assert(html.includes('id="horosJpegs" class="horos-file-input" type="file"') && css.includes('.horos-upload input.horos-file-input') && css.includes('clip-path:inset(50%)'), 'native Horos file input must remain available but visually hidden');
assert(html.includes('id="doctorSelect"'), 'doctor dropdown is required');
for (const token of ['doctorManagement','doctorDirectoryList','doctorForm','doctorSave','doctorCancel']) {
  assert(html.includes('id="' + token + '"'), 'managed doctor UI missing '+token);
}
assert(html.includes('id="rfSourceSearch"') && html.includes('id="rfPreviousPdf"'), 'A2 must offer one-field search and old RF PDF upload');
const typeAt = html.indexOf('data-pathway="A2"');
const sourceAt = html.indexOf('id="a2SourceFlow"');
assert(typeAt >= 0 && sourceAt > typeAt &&
       sourceAt < html.indexOf('id="rfDraftWorkspace"') &&
       sourceAt < html.indexOf('id="patientName"') &&
       sourceAt < html.indexOf('id="indicationSelect"') &&
       sourceAt < html.indexOf('id="a2Flow"'),
       'A2 prior-RF source search must render directly after pathway, before patient/diagnosis');
assert(html.includes('id="rfNameConflict"') &&
       html.includes('id="rfConfirmCurrentNameBtn"') &&
       html.includes('id="rfUsePreviousNameBtn"'), 'name/identity mismatch requires explicit choices');
assert(css.includes('.builder>section[hidden]{display:none}'),
  'hidden A1/A2 sections must not be forced visible by their display:grid rule');
assert(html.includes('id="rfDraftSaveBtn"') && html.includes('id="rfDraftList"'), 'A1/A2 saved drafts need Save and Resume controls');
assert(app.includes("'/api/reuse/search?query='") && app.includes("'/api/reuse/pdf'"), 'A2 history must reuse the protected RF API');
assert(app.includes("'/api/drafts'") && app.includes('saveRfDraft') && app.includes('restoreRfDraft'), 'saved drafts must be server-resumable');
assert(app.includes("state.draftId") && app.includes("draft_revision"), 'final RF create must link revisioned draft');
assert(!html.includes('Συμπλήρωσε ταυτότητα, ένδειξη και μία πλευρά (δεξιά ή αριστερά) πριν την αναζήτηση.'), 'no three-field gate before old RF search');
assert(app.includes('/api/doctors'), 'doctor directory must use the protected native RF API');
assert(!app.includes('RF_DOCTOR_MAXOUTLOU_PROFILE_JSON'), 'doctor CRUD must not depend on Render variables');
assert(!app.includes('localStorage') && !app.includes('sessionStorage'), 'RF doctors must not use browser storage');
assert(app.includes("form.append('horos_jpegs'") || app.includes("form.append('horos_jpegs',"), 'image files must reach multipart upload');
console.log('RF v2 Horos browser selection/preview/removal: PASS');

// P1-01: the actual edit/add/status handlers must preserve the issuer of the
// in-progress RF and all unrelated draft fields. No text-token-only oracle.
async function testIssuerSelectionAfterDirectoryChanges() {
  const sample = (key, name, gesy_code) => ({
    key, name, gesy_code, specialty: "Synthetic Orthopaedics",
    medical_center: "Synthetic Clinic", phone: "55501234",
    email: key + "@example.invalid", active: true, revision: 1,
  });
  const physicians = [sample("papachristou", "Synthetic Doctor A", "SYN-A"),
                      sample("physicianB", "Synthetic Doctor B", "SYN-B")];
  const physician = key => physicians.find(x => x.key === key);
  const form = doc.getElementById("doctorForm");
  const fields = ["name", "gesy_code", "specialty", "medical_center", "phone", "email"];
  const fieldControls = Object.fromEntries(fields.map(k => [k, {value: ""}]));
  form.elements = {namedItem(name) { return fieldControls[name]; }};
  form.reset = () => fields.forEach(k => {fieldControls[k].value = "";});
  for (const id of ["doctorSelect","productSelect","indicationSelect","lateralitySelect"]) {
    const select = doc.getElementById(id);
    select.dataset = {};
    select.add = option => select.children.push(option);
  }
  browser.Option = function(label, value) {this.label = label; this.value = value;};
  const received = [];
  browser.fetch = async (url, init = {}) => {
    const path = url.replace("/clinical/clinic-utilities/rf", "");
    received.push({path, method: init.method || "GET"});
    const respond = (data, status = 200) =>
      ({ok: status < 400, status, json: async () => JSON.parse(JSON.stringify(data))});
    if (path === "/api/contract") return respond({
      doctors: Object.fromEntries(physicians.map(x => [x.key, {label:x.name, configured:x.active}])),
      products: {medikey:{label:"Synthetic product",configured:true}},
      indications: {KNEE_OA_KL34:{label:"Synthetic indication"}},
      laterality: {left:"Αριστερά",right:"Δεξιά"}, rf_reasons: {},
    });
    if (path === "/api/doctors" && (!init.method || init.method === "GET")) return respond({doctors:physicians});
    if (path.startsWith("/api/doctors/") && init.method === "PUT") {
      const x = physician(path.split("/")[3]);
      assert(x, "editing existing doctor");
      const input = JSON.parse(init.body);
      assert.strictEqual(input.revision, x.revision);
      for (const k of fields) x[k] = input[k];
      x.revision++;
      return respond(x);
    }
    if (path === "/api/doctors" && init.method === "POST") {
      const input = JSON.parse(init.body), row = {...input, key:"physicianC", active:true, revision:1};
      physicians.push(row);
      return respond(row,201);
    }
    if (path.endsWith("/status") && init.method === "POST") {
      const x = physician(path.split("/")[3]), input = JSON.parse(init.body);
      assert.strictEqual(input.revision, x.revision);
      x.active = input.active; x.revision++;
      return respond(x);
    }
    throw Error("Unexpected RF network request " + (init.method || "GET") + " " + path);
  };

  await browser.__rfTest.loadContract();
  const get = id => doc.getElementById(id);
  assert.strictEqual(get("doctorSelect").value, "papachristou", "original default");
  get("patientName").value = "Synthetic Patient";
  get("productSelect").value = "medikey";
  get("indicationSelect").value = "KNEE_OA_KL34";
  get("lateralitySelect").value = "left";
  get("exactLocation").value = "Synthetic site";
  get("additionalNotes").value = "Unrelated draft text";
  const optionsBefore = {
    products: get("productSelect").children.length,
    indications: get("indicationSelect").children.length,
    laterality: get("lateralitySelect").children.length,
  };
  const unchangedDraft = () => {
    assert.strictEqual(get("patientName").value, "Synthetic Patient");
    assert.strictEqual(get("productSelect").value, "medikey");
    assert.strictEqual(get("indicationSelect").value, "KNEE_OA_KL34");
    assert.strictEqual(get("lateralitySelect").value, "left");
    assert.strictEqual(get("exactLocation").value, "Synthetic site");
    assert.strictEqual(get("additionalNotes").value, "Unrelated draft text");
    assert.deepStrictEqual({
      products: get("productSelect").children.length,
      indications: get("indicationSelect").children.length,
      laterality: get("lateralitySelect").children.length,
    }, optionsBefore, "directory reload must not duplicate options or clear side");
  };

  // Editing colleague B while preparing the RF as doctor A must not switch A -> B.
  browser.__rfTest.editDoctor({...physician("physicianB")});
  fieldControls.medical_center.value = "Edited Synthetic Clinic";
  await browser.__rfTest.saveDoctor({preventDefault(){}});
  assert.strictEqual(get("doctorSelect").value, "papachristou");
  assert(get("reviewSummary").innerHTML.includes("Synthetic Doctor A"),
    "visible RF review summary must still show issuer A");
  unchangedDraft();

  // Adding a third doctor similarly does not select the new physician.
  for (const k of fields) fieldControls[k].value = physician("physicianB")[k];
  fieldControls.name.value = "Synthetic Doctor C";
  fieldControls.gesy_code.value = "SYN-C";
  await browser.__rfTest.saveDoctor({preventDefault(){}});
  assert.strictEqual(get("doctorSelect").value, "papachristou");
  assert(get("doctorSelect").children.some(o => o.value === "physicianC"));
  unchangedDraft();

  // Explicitly select B; editing A should preserve B (no silent fallback).
  get("doctorSelect").value = "physicianB";
  browser.__rfTest.editDoctor({...physician("papachristou")});
  fieldControls.phone.value = "55509999";
  await browser.__rfTest.saveDoctor({preventDefault(){}});
  assert.strictEqual(get("doctorSelect").value, "physicianB");
  assert(get("reviewSummary").innerHTML.includes("Synthetic Doctor B"));
  unchangedDraft();

  // Deactivating B (the selected issuer) must leave no selected issuer,
  // not silently reselect default A; API /create cannot be reached.
  await browser.__rfTest.loadContract();
  await browser.__rfTest.loadContract();
  const list = get("doctorDirectoryList");
  const itemB = list.children.find(node => node.children[0].children[0].textContent === "Synthetic Doctor B");
  assert(itemB,"doctor B must be available in managed directory");
  await itemB.children[1].children[1].listeners.click();
  assert.strictEqual(get("doctorSelect").value, "",
    "deactivated selected doctor requires explicit new choice");
  assert(get("reviewSummary").innerHTML.includes("Ιατρός: —"));
  assert(get("doctorDirectoryStatus").textContent.includes("Επίλεξε ρητά"));
  assert(get("doctorSelect").children.some(o => o.value === ""),
    "explicit placeholder remains selected");
  unchangedDraft();
  await browser.__rfTest.createPdf();
  assert(get("validationPanel").textContent.includes("Επίλεξε ρητά"));
  assert(!received.some(x => x.path === "/api/create"), "cannot submit without issuer");
}
testIssuerSelectionAfterDirectoryChanges().then(async () => {
  console.log("RF v2 managed-doctor issuer-selection/retained-draft regression: PASS");
  await testReusableRFFieldsAndDraftRestore();
  console.log("RF v2 A2 prefill and saved-draft browser handlers: PASS");
  await testOldRFNameMismatchResolution();
  console.log("RF v2 A2 explicit patient-name conflict resolution: PASS");
  await testRfOutcomeActionsAndCorrection();
  console.log("RF v2 real outcome handlers and linked correction: PASS");
  await testRfStaleSelectionAndProvenance();
  console.log("RF v2 stale A/B selection, protected write and audit provenance: PASS");
}).catch(error => { console.error(error); process.exitCode = 1; });


async function testReusableRFFieldsAndDraftRestore() {
  doc.querySelectorAll = () => [];
  const get = id => doc.getElementById(id);
  get('adverseRows').querySelectorAll = () => [];
  get('doctorSelect').value = 'papachristou';
  get('patientName').value = 'Typed patient name';
  get('identityNumber').value = 'PAT-ONE';
  get('age').value = '';
  get('indicationSelect').value = '';
  get('lateralitySelect').value = '';
  const conflicts = browser.__rfTest.populateRfFields({
    patient_name:'Older stored name', identity_number:'PAT-ONE',
    gesy_number:'SYN-GESY', age:67, indication_code:'KNEE_OA_KL34',
    laterality:'left', exact_location:'Historical knee'
  });
  assert(conflicts.includes('patient_name'), 'typed name must not be silently overwritten');
  assert.strictEqual(get('doctorSelect').value, 'papachristou', 'prefill cannot switch issuer');
  assert.strictEqual(get('identityNumber').value, 'PAT-ONE');
  assert.strictEqual(get('indicationSelect').value, 'KNEE_OA_KL34');
  assert.strictEqual(get('lateralitySelect').value, 'left');
  assert.throws(() => browser.__rfTest.populateRfFields({
    identity_number:'OTHER-IDENTITY', patient_name:'Other person'
  }), /άλλη ταυτότητα/);
  assert.strictEqual(get('identityNumber').value, 'PAT-ONE');
  const saved = browser.__rfTest.savedDraftFields();
  assert(!Object.hasOwn(saved, 'imaging_review_confirmed'), 'imaging confirmation is not saved');
  assert(!Object.hasOwn(saved, 'imaging_report') && !Object.hasOwn(saved, 'horos_jpegs'),
    'draft cannot include binary attachment fields');
  get('imagingReport').value = 'old.pdf';
  get('imagingStatus').hidden = false;
  await browser.__rfTest.restoreRfDraft({
    pathway:'A2', doctor_key:'physicianB', patient_name:'Saved patient',
    identity_number:'DRAFT-001', age:68, gesy_number:'SYN-GESY',
    indication_code:'KNEE_OA_KL34', laterality:'left',
    exact_location:'Saved site', additional_notes:'Saved note',
    a2_current_followup_date:'2026-09-05', a2_current_followup_vas:5,
    legacy_history:{actual_procedure_date:'2026-06-01', vas_before:8, vas_after:3,
                    last_followup_date:'2026-07-01',last_followup_vas:4},
    legacy_confirmed:true, source_origin:'pdf'
  });
  assert.strictEqual(get('doctorSelect').value, '', 'inactive issuer requires explicit selection');
  assert.strictEqual(get('patientName').value, 'Saved patient');
  assert.strictEqual(get('age').value, '68');
  assert.strictEqual(get('imagingReport').value, '', 'attached files must be reattached');
  assert.strictEqual(get('imagingStatus').hidden, true);
  assert.strictEqual(get('rfCurrentFollowupVas').value, '5');
  assert.strictEqual(get('rfActualConfirm').checked, true);
  assert.strictEqual(get('additionalNotes').value, 'Saved note');
}


async function testOldRFNameMismatchResolution() {
  const get = id => doc.getElementById(id);
  get('patientName').value = 'Patient B';
  get('identityNumber').value = '';
  get('indicationSelect').value = '';
  get('lateralitySelect').value = '';
  get('exactLocation').value = '';
  const issued = [];
  browser.fetch = async (url, init = {}) => {
    const path = url.replace('/clinical/clinic-utilities/rf', '');
    issued.push(path);
    const result = path === '/api/reuse/application/old-A'
      ? {fields:{patient_name:'Patient A',identity_number:'PAT-A',
                 gesy_number:'GESY-A', indication_code:'KNEE_OA_KL34',
                 laterality:'left',exact_location:'Αριστερό γόνατο'},
         source:{kind:'application',id:'old-A'}} :
      path === '/api/history' ? {found:false,procedures:[]} :
      null;
    if (!result) throw Error('Unexpected protected RF request '+path);
    return {ok:true,status:200,json:async()=>result};
  };
  await browser.__rfTest.loadRfSource({kind:'application',id:'old-A'});
  assert.strictEqual(get('patientName').value,'Patient B',
    'typed current name should not be overwritten by old source');
  assert.strictEqual(get('identityNumber').value,'PAT-A',
    'selected prior patient identity is filled');
  assert.strictEqual(get('rfNameConflict').hidden,false,
    'name mismatch requires an explicit, visible resolution');
  assert(browser.__rfTest.unresolvedSourceNameConflict());
  await browser.__rfTest.createPdf();
  assert(get('validationPanel').textContent.includes('Επίλυσε ρητά'),
    'no RF PDF can be issued while selected source name disagrees');
  assert(!issued.some(s=>s==='/api/create'), 'blocked mismatch cannot reach create');
  browser.__rfTest.resolveSourceName('current');
  assert.strictEqual(browser.__rfTest.unresolvedSourceNameConflict(),false);
  assert.strictEqual(browser.__rfTest.buildDraft().source_name_confirmed,true,
    'keeping the new name must be an explicit clinician confirmation');
  browser.__rfTest.invalidateSourceNameConfirmation();
  assert.strictEqual(browser.__rfTest.unresolvedSourceNameConflict(),true,
    'editing identity/name invalidates the prior confirmation');
  browser.__rfTest.resolveSourceName('previous');
  assert.strictEqual(get('patientName').value,'Patient A');
  assert.strictEqual(get('rfNameConflict').hidden,true);
  assert.strictEqual(browser.__rfTest.buildDraft().source_name_confirmed,false);
  assert.strictEqual(get('a2SourceFlow').hidden,false,
    'A2 source selector must be visible in A2 mode');
  browser.__rfTest.setPathway('A1');
  assert.strictEqual(get('a2SourceFlow').hidden,true,
    'source selector hidden for A1; original workflow remains');
}


async function testRfOutcomeActionsAndCorrection() {
  for (const id of ['rfOutcomeAction', 'rfOutcomeScope']) {
    const node = doc.getElementById(id);
    node.add = value => {node.children.push(value); node.value = value.value;};
  }
  const el = id => doc.getElementById(id);
  const original = {
    id: 'synthetic-application-rejected', issued_date: '2026-10-01',
    patient_name: 'Synthetic RF Patient', identity_number: 'RF-CORR',
    indication_code: 'KNEE_OA_KL34', laterality: 'left', doctor_key: 'physicianB',
    issuer_name: 'Synthetic Doctor B', stage: 'rejected', revision: 1,
    submitted_date: null, decision_date: '2026-10-10',
    rejection_reason: 'Synthetic rejection source', decision: 'rejected',
    mine_to_follow: false, correction_of_application_id: '',
  };
  let card = {...original};
  const requests = [];
  browser.fetch = async (url, options = {}) => {
    const path = url.replace('/clinical/clinic-utilities/rf', '');
    const request = options.body ? JSON.parse(options.body) : {};
    requests.push({path, request, method: options.method});
    let data;
    if (path === '/api/outcomes/search') {
      data = {applications:[card],counts:{pending:0,approved:0,rejected:1,
        mine_actionable:0},denominator:1,truncated:false,has_more:false};
    } else if (path.endsWith('/events')) {
      assert.strictEqual(request.revision, card.revision);
      assert.strictEqual(request.action, 'correct_decision');
      assert.strictEqual(request.reported_source, 'colleague');
      assert.strictEqual(request.rejection_reason, 'Synthetic corrected reason');
      card = {...card,revision:card.revision+1,
        decision_date:request.event_date,rejection_reason:request.rejection_reason};
      data = card;
     } else if (path === '/api/reuse/rejected-correction') {
      assert.strictEqual(options.method, 'POST');
      assert.strictEqual(request.application_id, original.id);
      assert.strictEqual(request.expected_revision, card.revision);
      data = {application_id: original.id, revision: card.revision, source:{kind:'application',pathway:'A1'},fields:{
        pathway:'A1',patient_name:'Synthetic RF Patient',identity_number:'RF-CORR',
        gesy_number:'SYN-GESY',age:67,indication_code:'KNEE_OA_KL34',
        laterality:'left',exact_location:'Αριστερό γόνατο',product_key:'medikey',
      }};
    } else throw Error('Unexpected protected RF request: '+path);
    return {ok:true,status:200,json:async()=>data};
  };
  el('rfTrackingView').hidden = true; // Match the real initial HTML hidden state.
  el('rfOutcomeScope').value = 'all';
  el('rfOutcomeSearch').value = '';
  el('rfOutcomeAction').add = option => {
    el('rfOutcomeAction').children.push(option);
    if (el('rfOutcomeAction').children.length===1)
      el('rfOutcomeAction').value = option.value;
  };
  await browser.__rfTest.loadRfOutcomes();
  assert.strictEqual(el('rfOutcomeResults').children.length,1);
  assert(el('rfOutcomeCounts').textContent.includes('Απορρίψεις 1'));
  browser.__rfTest.renderSelectedOutcome(card);
  assert(el('rfOutcomeCorrection').hidden === false);
  assert.strictEqual(el('rfOutcomeAction').children[0].value,'correct_decision');
  el('rfOutcomeSource').value = 'colleague';
  el('rfOutcomeAction').value = 'correct_decision';
  el('rfOutcomeReason').value = 'Synthetic corrected reason';
  el('rfOutcomeDate').value = '2026-10-11';
  await browser.__rfTest.applyRfOutcome();
  assert.strictEqual(card.rejection_reason,'Synthetic corrected reason');
  assert.strictEqual(card.mine_to_follow,false);
  assert.strictEqual(el('rfOutcomeTitle').textContent.includes('Απορρίφθηκε'),true);
  assert(requests.some(x=>x.path.endsWith('/events')));
  assert(!requests.some(x=>x.path.includes('/api/create')),
         'editing outcome cannot create official RF or assign a colleague obligation');

  // A deliberately requested correction stays A1 and links only the rejected ID;
  // it may not silently select B or declare any prior procedure as performed.
  for (const id of ['identityNumber','patientName','indicationSelect','lateralitySelect',
                    'exactLocation','gesyNumber','productSelect','additionalNotes'])
    el(id).value = '';
  await browser.__rfTest.beginRfCorrection();
  assert.strictEqual(el('patientName').value,'Synthetic RF Patient',
    'guarded correction prefill failed: ' + JSON.stringify({
      error: el('validationPanel').textContent, requests: requests.slice(-3),
      title: el('rfOutcomeTitle').textContent,
      correctionContext: el('rfCorrectionContext').textContent
    }));
  assert.strictEqual(el('identityNumber').value,'RF-CORR');
  assert.strictEqual(el('rfCorrectionContext').hidden,false);
  assert.strictEqual(browser.__rfTest.buildDraft().correction_of_application_id,
                     original.id, 'new issued attempt must link original rejection');
  assert.strictEqual(browser.__rfTest.buildDraft().pathway,'A1',
                     'rejected application does not imply real A2 treatment');
  assert.strictEqual(el('imagingReport').value,'',
                     'new official RF requires new imaging attachment');
  assert.strictEqual(el('doctorSelect').value,'',
                     'correction requires an explicit fresh issuer selection');

  // A saved draft must never re-arm correction after reload.
  assert.strictEqual(browser.__rfTest.savedDraftFields().correction_of_application_id,
                      undefined);
  assert.strictEqual(browser.__rfTest.buildDraft().correction_source_expected_revision,
                      card.revision);
}


async function testRfStaleSelectionAndProvenance() {
  const el = id => doc.getElementById(id);
  el('rfOutcomeSearch').value = '';
  el('rfOutcomeScope').value = 'all';
  let resolveA, resolveB;
  const detailA = new Promise(resolve => {resolveA = resolve;});
  const detailB = new Promise(resolve => {resolveB = resolve;});
  const a = {
    id:'synthetic-A', patient_name:'Patient A', stage:'rejected',
    issued_date:'2026-10-01', decision:'rejected', decision_date:'2026-10-09',
    mine_to_follow:false, revision:1, rejection_reason:'Old rejected A',
  };
  const b = {
    id:'synthetic-B', patient_name:'Patient B', stage:'rejected',
    issued_date:'2026-10-01', decision:'rejected', decision_date:'2026-10-10',
    mine_to_follow:false, revision:1,
    rejection_reason:'<img src=x onerror="window.evil=true">',
    last_decision_report: {
      reported_source:'colleague', event_date:'2026-10-10',
      recorded_at:'2026-10-11T12:30:00Z',
    },
  };
  const eventPaths = [];
  browser.fetch = async (url, options = {}) => {
    const path = url.replace('/clinical/clinic-utilities/rf', '');
    if (path === '/api/outcomes/' + a.id) return detailA;
    if (path === '/api/outcomes/' + b.id) return detailB;
    if (path.endsWith('/events')) {
      eventPaths.push(path);
      const body = JSON.parse(options.body);
      assert.strictEqual(body.action,'claim');
      assert.strictEqual(body.revision,b.revision);
      assert.strictEqual(path, '/api/outcomes/' + b.id + '/events',
        'protected event must address the actually selected application');
      return {ok:true,status:200,json:async()=>({...b,revision:2,mine_to_follow:true})};
    }
    if (path === '/api/outcomes/search')
      return {ok:true,status:200,json:async()=>({
        applications:[b],counts:{},denominator:1,truncated:false,has_more:false
      })};
    throw Error('Unexpected stale selection request '+path);
  };

  const first = browser.__rfTest.selectRfOutcome(a);
  const second = browser.__rfTest.selectRfOutcome(b);
  // The previously selected item is not actionable while B is loading.
  el('rfOutcomeAction').value = 'claim';
  await browser.__rfTest.applyRfOutcome();
  assert.strictEqual(eventPaths.length,0,'no event while a different selection is pending');
  resolveB({ok:true,status:200,json:async()=>b});
  await second;
  assert(el('rfOutcomeTitle').textContent.includes('Patient B'));
  resolveA({ok:true,status:200,json:async()=>a});
  await first;
  assert(el('rfOutcomeTitle').textContent.includes('Patient B'),
    'late A response must not replace newer B');
  assert(el('rfOutcomeFacts').textContent.includes('ενημέρωση συναδέλφου'),
    'selected RF must show its real last-decision audit provenance');
  assert(el('rfOutcomeFacts').textContent.includes('Καταχώριση στο Cockpit:'),
    'the audit recorded time must be visibly distinct from decision date');
  assert(el('rfOutcomeFacts').textContent.includes('<img src=x onerror='),
    'free text shown literally, not parsed as HTML');
  assert.strictEqual(el('rfOutcomeFacts').innerHTML,undefined,
    'selected facts must use textContent instead of innerHTML');
  el('rfOutcomeAction').value = 'claim';
  await browser.__rfTest.applyRfOutcome();
  assert.deepStrictEqual(eventPaths,['/api/outcomes/synthetic-B/events']);
  assert(el('rfOutcomeFacts').textContent.includes('έχω αναλάβει'));

  // A change of search context makes even a previously loaded B unactionable.
  el('rfOutcomeSearch').value = 'different patient';
  await browser.__rfTest.applyRfOutcome();
  assert.strictEqual(eventPaths.length,1,
    'changed search must block an event against previously selected B');
  browser.__rfTest.invalidateRfOutcomeSelection();
  assert.strictEqual(el('rfOutcomeSelected').hidden,true);
  await browser.__rfTest.applyRfOutcome();
  assert.strictEqual(eventPaths.length,1,'invalidated selection cannot post');
  const historical = {...b,last_decision_report:null};
  browser.__rfTest.renderSelectedOutcome(historical);
  assert(el('rfOutcomeFacts').textContent.includes('Προέλευση απόφασης: άγνωστη'));
  assert(el('rfOutcomeFacts').textContent.includes('Καταχώριση στο Cockpit: άγνωστος'),
    'unknown historical audit provenance must never be invented');
}
