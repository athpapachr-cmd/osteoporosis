'use strict';

const fs = require('fs');
const assert = require('assert');

const app = fs.readFileSync('static/clinic-utilities/rf/app.js', 'utf8');
const html = fs.readFileSync('static/clinic-utilities/rf/index.html', 'utf8');

assert(app.includes('function applySuggestedLocation()'), 'RF UI must derive exact location from indication + side');
assert(app.includes("lateralitySelect').addEventListener('change', applySuggestedLocation)"), 'side change must refresh exact location');
assert(app.includes("item.location_labels?.[side]"), 'RF UI must use server-provided location labels');
assert(app.includes("exact.readOnly = item?.location_mode === 'derived'"), 'fixed RF locations must be system-derived and read-only');
assert(app.includes("!['left','right'].includes($('lateralitySelect').value)"), 'RF UI must fail closed unless side is left/right');
assert(html.includes('Μία αίτηση = μία πλευρά και μία εντόπιση'), 'RF UI must explain one-side/one-target rule');
assert(html.includes('Συμπληρώνεται από ένδειξη + πλευρά'), 'exact-location field must explain automatic suggestion');
assert(app.includes("api('/api/validate-imaging'"), 'RF UI must preview-check imaging attachment type');
assert(app.includes('imaging_review_confirmed'), 'RF draft must carry explicit ambiguous-attachment confirmation');
assert(app.includes("imagingReport').addEventListener('change', validateImaging)"), 'changing imaging file must reset/re-run semantic validation');
assert(html.includes('Επιβεβαιώνω ότι το επιλεγμένο PDF είναι η απεικονιστική έκθεση'), 'ambiguous attachment must expose explicit clinician confirmation');
assert(html.includes('έως 3'), 'medication copy must reflect maximum capacity, not a minimum requirement');

console.log('RF unilateral target UI regression: PASS');
