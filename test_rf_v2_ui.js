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
assert(main.includes('app.include_router(build_rf_router(engine))'), 'main.py must mount the native RF router');
assert(!main.includes('app.include_router(build_rf_gateway_router())'), 'legacy RF gateway must not be mounted');

console.log('RF v2 UI integrity regression: PASS');
