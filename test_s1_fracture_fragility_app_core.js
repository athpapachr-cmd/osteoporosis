"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync("static/baseline-audit/app-core.js", "utf8");
const startup = "  bindChoiceButtons(); bindStep1Inputs(); bindStep2Inputs(); bindNavigation(); setupPrivacy(); restoreActiveCase(); syncUiFromState(); renderCaseList();";
assert(source.includes(startup), "app-core startup seam changed; S1 harness must be reviewed");

const instrumented = source.replace(startup, `  window.__S1FractureHooks = {
    normalizeLoadedCase,
    renderFractureEvents,
    collectFractureEventsFromDom,
    setCurrentCase(value) { currentCase = value; },
    getCurrentCase() { return currentCase; },
    fractureRoot: el.fractureEvents
  };`);

let uuidCalls = 0;
function makeStub() {
  return {
    value: "",
    checked: false,
    hidden: false,
    innerHTML: "",
    textContent: "",
    dataset: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute() {},
    getAttribute() { return null; },
    addEventListener() {},
    querySelectorAll() { return []; },
    closest() { return null; },
    scrollIntoView() {}
  };
}

const elements = new Map();
function elementFor(selector) {
  if (!elements.has(selector)) elements.set(selector, makeStub());
  return elements.get(selector);
}
const fractureRoot = elementFor("#fractureEvents");
fractureRoot._rows = [];
fractureRoot.querySelectorAll = selector => selector === ".fracture-event" ? fractureRoot._rows : [];

const documentStub = {
  head: { appendChild() {} },
  querySelector(selector) { return elementFor(selector); },
  querySelectorAll() { return []; },
  createElement() { return makeStub(); }
};

const localStore = new Map();
const sandbox = {
  console,
  document: documentStub,
  localStorage: {
    getItem(key) { return localStore.has(key) ? localStore.get(key) : null; },
    setItem(key, value) { localStore.set(key, String(value)); }
  },
  window: {
    crypto: { randomUUID() { uuidCalls += 1; return `uuid-${uuidCalls}`; } },
    addEventListener() {},
    scrollTo() {},
    confirm() { return true; },
    alert() {}
  },
  Intl,
  Date,
  Math,
  JSON,
  Array,
  Number,
  String,
  Object,
  Set,
  Map
};
vm.createContext(sandbox);
vm.runInContext(instrumented, sandbox, { filename: "app-core.js" });

const hooks = sandbox.window.__S1FractureHooks;
assert(hooks, "S1 app-core hooks missing");

// 1. Legacy prior=true + zero events stays zero through normalize/load + render; render creates no fracture UUID/event.
{
  const legacy = {
    internal_uuid: "legacy-case",
    case_sequence_number: 1,
    risk_context: {
      prior_fragility_fracture: true,
      last_fracture_site: "hip",
      last_fracture_month: "2024-10"
    },
    fracture_history: { events: [] }
  };
  const loaded = hooks.normalizeLoadedCase(legacy);
  assert.strictEqual(loaded.fracture_history.events.length, 0);
  hooks.setCurrentCase(loaded);
  const beforeRenderUuidCalls = uuidCalls;
  hooks.renderFractureEvents();
  assert.strictEqual(hooks.getCurrentCase().fracture_history.events.length, 0);
  assert.strictEqual(uuidCalls, beforeRenderUuidCalls, "render must not synthesize a fracture event UUID");
}

// 2. Writer updates raw event fields only; no mechanism value auto-writes compatibility fragility fields.
{
  const mechanisms = ["no", "uncertain", "", "yes"];
  mechanisms.forEach((mechanism, index) => {
    const id = `s1-writer-${index}`;
    const current = {
      fracture_history: { events: [{ id, site: "hip", month: "2026-08", low_trauma: "", occurred_on_treatment: "" }] },
      risk_context: { prior_fragility_fracture: false, last_fracture_site: "", last_fracture_month: "" }
    };
    hooks.setCurrentCase(current);
    fractureRoot._rows = [{
      dataset: { eventId: id },
      querySelectorAll(selector) {
        if (selector !== "[data-event-field]") return [];
        return [
          { dataset: { eventField: "site" }, value: "hip" },
          { dataset: { eventField: "month" }, value: "2026-08" },
          { dataset: { eventField: "low_trauma" }, value: mechanism },
          { dataset: { eventField: "occurred_on_treatment" }, value: "no" }
        ];
      }
    }];
    hooks.collectFractureEventsFromDom();
    const after = hooks.getCurrentCase();
    assert.strictEqual(after.fracture_history.events.length, 1);
    assert.strictEqual(after.fracture_history.events[0].id, id);
    assert.strictEqual(after.fracture_history.events[0].low_trauma, mechanism);
    assert.strictEqual(after.risk_context.prior_fragility_fracture, false, "event collection must not auto-promote compatibility prior fragility");
    assert.strictEqual(after.risk_context.last_fracture_site, "", "generic event must not overwrite compatibility last-fragility site");
    assert.strictEqual(after.risk_context.last_fracture_month, "", "generic event must not overwrite compatibility last-fragility month");
  });
}

// 3. Legacy event.fragility is preserved raw and never copied into canonical low_trauma during load.
{
  const loaded = hooks.normalizeLoadedCase({
    internal_uuid: "legacy-fragility-field",
    case_sequence_number: 2,
    risk_context: { prior_fragility_fracture: true },
    fracture_history: { events: [{ id: "stable-legacy", site: "hip", month: "2025-01", fragility: "yes" }] }
  });
  assert.strictEqual(loaded.fracture_history.events.length, 1);
  assert.strictEqual(loaded.fracture_history.events[0].id, "stable-legacy");
  assert.strictEqual(loaded.fracture_history.events[0].fragility, "yes");
  assert.strictEqual(loaded.fracture_history.events[0].low_trauma, undefined);
}

// 4. The implicit render-time fracture seeding owner is gone; only explicit add-event flow may create a new event.
{
  assert(!source.includes("ensureStep1FractureSeed"), "render/load must not contain legacy fracture seed synthesis");
  const renderStart = source.indexOf("  function renderFractureEvents()");
  const collectStart = source.indexOf("  function collectFractureEventsFromDom()", renderStart);
  assert(renderStart >= 0 && collectStart > renderStart);
  const renderBody = source.slice(renderStart, collectStart);
  assert(!renderBody.includes("makeFractureEvent("), "render path must not create factual fracture events");

  const collectEnd = source.indexOf("  function updateStep2ContextNote()", collectStart);
  const collectBody = source.slice(collectStart, collectEnd);
  assert(!/prior_fragility_fracture\s*=\s*true/.test(collectBody));
  assert(!/last_fracture_site\s*=/.test(collectBody));
  assert(!/last_fracture_month\s*=/.test(collectBody));
}

console.log("S1 app-core fracture/fragility writer + load/render regressions: PASS");
