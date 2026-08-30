"use strict";

const assert = require("assert");
const fs = require("fs");

const app = fs.readFileSync("static/baseline-audit/app.js", "utf8");
const ui = fs.readFileSync("static/baseline-audit/progressive-guidance-ui.js", "utf8");
const css = fs.readFileSync("static/baseline-audit/adaptive-applicability.css", "utf8");

function pos(text) {
  const i = app.indexOf(text);
  assert(i >= 0, `missing bootstrap entry: ${text}`);
  return i;
}

const adaptive = pos('./adaptive-applicability.js');
const core = pos('./progressive-guidance-core.js');
const finalization = pos('./finalization-coordinator.js');
const registry = pos('./patient-registry.js');
const guidanceUi = pos('./progressive-guidance-ui.js');
const pilot = pos('./pilot-completion.js');

assert(adaptive < core, "guidance core must load after existing adaptive applicability");
assert(core < finalization, "guidance core must not disturb finalization coordinator ownership");
assert(finalization < registry && registry < pilot, "authoritative Finish load order must remain coordinator -> registry -> pilot");
assert(registry < guidanceUi && guidanceUi < pilot, "guidance UI should load after registry and before pilot completion without taking Finish ownership");

assert(ui.includes('/clinical/patient/${encodeURIComponent(patientId)}/encounters'), "guidance UI must use protected historical encounter endpoint");
assert(ui.includes('credentials: "same-origin"'), "historical fetch must stay within authenticated same-origin clinical route");
assert(ui.includes('#quickNotes'), "first-page free-text context must be consumed as context");
assert(!ui.includes('classList.remove("adaptive-collapsed"'), "guidance must not mutate coarse applicability ownership");
assert(css.includes('adaptive-collapsed:not(.guidance-surfaced)'), "CSS must visually override collapsed state only while higher-priority guidance is active");
assert(ui.includes('textContent = "Σημερινή ροή"'), "guidance summary surface missing");
assert(ui.includes('document.createTextNode(item.why_now'), "WHY NOW text must be rendered as text, not injected HTML");

console.log("progressive guidance wiring regression: OK");
