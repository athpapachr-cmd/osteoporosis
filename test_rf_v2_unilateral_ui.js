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

console.log('RF unilateral target UI regression: PASS');
