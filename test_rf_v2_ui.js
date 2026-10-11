'use strict';

const fs = require('fs');
const assert = require('assert');

const app = fs.readFileSync('static/clinic-utilities/rf/app.js', 'utf8');
const html = fs.readFileSync('static/clinic-utilities/rf/index.html', 'utf8');
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
const instrumented = app.replace(/\}\)\(\);\s*$/, '\n  globalThis.__rfTest = {addHorosImages, releaseHorosPreviews};\n})();');
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
assert(grid.children.every(x => x.children[0].src.startsWith('blob:rf-test-')), 'thumbnails need blob URLs');
const removed = grid.children[0].children[0].src;
grid.children[0].children[2].listeners.click();
assert.deepStrictEqual(grid.children.map(x => x.children[1].textContent), ['second.jpeg']);
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
assert(html.includes('id="doctorSelect"'), 'doctor dropdown is required');
assert(app.includes("form.append('horos_jpegs'") || app.includes("form.append('horos_jpegs',"), 'image files must reach multipart upload');
console.log('RF v2 Horos browser selection/preview/removal: PASS');
