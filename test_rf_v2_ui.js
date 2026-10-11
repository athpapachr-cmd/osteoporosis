'use strict';

const fs = require('fs');
const assert = require('assert');

const app = fs.readFileSync('static/clinic-utilities/rf/app.js', 'utf8');
const html = fs.readFileSync('static/clinic-utilities/rf/index.html', 'utf8');
const css = fs.readFileSync('static/clinic-utilities/rf/styles.css', 'utf8');
const main = fs.readFileSync('main.py', 'utf8');

assert(app.includes("function numberOrNull(id)"), 'RF UI must preserve blank numeric fields as null');
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
    tag, children: [], listeners: {}, className: '', textContent: '', value: '',
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    addEventListener(type, fn) { this.listeners[type] = fn; },
    setAttribute() {},
  };
}
const doc = {
  getElementById(id) { return nodes[id] ||= fakeElement(id); },
  createElement: fakeElement,
  addEventListener() {}, // install but do not run unrelated page bootstrap
};
const instrumented = app.replace(/\}\)\(\);\s*$/, '\n  globalThis.__rfTest = {addHorosImages, releaseHorosPreviews, loadContract, saveDoctor, editDoctor, createPdf, populateRfFields, savedDraftFields, restoreRfDraft};\n})();');
assert.notStrictEqual(instrumented, app, 'RF handler instrumentation point must be present');
const browser = {
  document: doc,
  URL: {
    createObjectURL() { return 'blob:rf-test-' + (++blobId); },
    revokeObjectURL(value) { revoked.push(value); },
  },
  window: {},
};
vm.runInNewContext(instrumented, browser, {filename: 'rf-app.js'});
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
