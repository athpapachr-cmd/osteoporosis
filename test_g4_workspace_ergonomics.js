"use strict";

const assert = require("assert");
const fs = require("fs");

const app = fs.readFileSync("static/baseline-audit/app.js", "utf8");
const helper = fs.readFileSync("static/baseline-audit/g4-workspace-ergonomics.js", "utf8");
const css = fs.readFileSync("static/baseline-audit/progressive-guidance.css", "utf8");

// Bootstrap ownership: G4 decorates the already-established G3 surfaces; it does not replace them.
const g3Guard = app.indexOf('loadScript("./g3-production-visibility-guard.js")');
const g4 = app.indexOf('loadScript("./g4-workspace-ergonomics.js")');
const finishOwner = app.indexOf('loadScript("./pilot-completion.js")');
assert(g3Guard >= 0 && g4 > g3Guard, "G4 workspace helper must load after G3 visibility guard");
assert(finishOwner > g4, "G4 helper must not replace authoritative Finish owner");

// Both requested dynamic surfaces are enhanced in place.
assert(helper.includes('#patientLongitudinalSummary'), "patient summary root must be collapsible");
assert(helper.includes('.patient-summary-head'), "patient summary header must own its collapse control");
assert(helper.includes('#progressiveGuidanceSummary'), "current-flow root must be collapsible");
assert(helper.includes('.progressive-guidance-head'), "current-flow header must own its collapse control");
assert(helper.includes('aria-expanded'), "collapse state must be exposed accessibly");
assert(helper.includes('aria-controls'), "collapse control must identify its target");
assert(helper.includes('button.type = "button"'), "collapse controls must use native buttons");
assert(helper.includes('sessionStorage'), "collapse preference may be retained as per-browser UI state");
assert(!helper.includes('localStorage'), "G4 UI preference must not enter legacy clinical working-case persistence");
assert(!/\bfetch\s*\(/.test(helper), "G4 workspace helper must not own clinical network I/O");
assert(!helper.includes('/clinical/encounter/'), "G4 workspace helper must not write encounter state");

// Sticky/collapse presentation is CSS-only and keeps the heading visible.
assert(/\.patient-longitudinal-summary\s*\{[\s\S]*position:\s*sticky/.test(css), "patient summary must be sticky");
assert(/\.patient-longitudinal-summary\s*\{[\s\S]*top:\s*8px/.test(css), "sticky summary top offset missing");
assert(css.includes('.patient-longitudinal-summary.g4-collapsed > :not(.patient-summary-head)'), "collapsed summary must retain its header");
assert(css.includes('.progressive-guidance-summary.g4-collapsed > :not(.progressive-guidance-head)'), "collapsed flow must retain its header");
assert(css.includes('.g4-collapse-control'), "collapse-control styling missing");

// Clinic Utilities integration: RF browser navigation stays same-origin and protected.
assert(helper.includes('/clinical/clinic-utilities/rf'), "RF utility target must use the protected same-origin gateway");
assert(!helper.includes('https://ortho-reception-backend-v2.onrender.com/rf'), "browser must not navigate directly to the protected RF backend");
assert(!helper.includes('RF_ACCESS_KEY'), "RF credential must never be embedded in browser JavaScript");
assert(!helper.includes('?key='), "RF credential must never be passed through a browser query string");
assert(helper.includes('/clinical/clinic-utilities/physio-referral'), "existing CU-1 utility should remain reachable alongside RF");
assert(helper.includes('Ραδιοκύματα — PDF'), "RF utility must be clinician-visible in Cockpit navigation");
assert(helper.includes('target = "_blank"'), "RF utility should still open without replacing the active encounter workspace");
assert(helper.includes('noopener noreferrer'), "new-tab utility link must use safe opener isolation");

// Dynamic G3 re-renders must be tolerated by re-applying controls, not by creating a second summary renderer.
assert(helper.includes('MutationObserver'), "G4 must tolerate G3 DOM re-renders");
assert(!helper.includes('Σύνοψη ασθενούς'), "G4 must not render a second patient-summary content owner");
assert(!helper.includes('Γιατί τώρα'), "G4 must not own guidance content");

console.log("G4 workspace ergonomics and RF authenticated-gateway integration regressions: PASS");
