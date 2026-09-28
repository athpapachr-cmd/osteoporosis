"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync("static/baseline-audit/app-core.js", "utf8");
const g2Source = fs.readFileSync("static/baseline-audit/osteoporosis-evidence-guidance-core.js", "utf8");
const startup = "  bindChoiceButtons(); bindStep1Inputs(); bindStep2Inputs(); bindNavigation(); setupPrivacy(); restoreActiveCase(); syncUiFromState(); renderCaseList();";
assert(source.includes(startup), "app-core startup seam changed; S1 harness must be reviewed");

const instrumented = source.replace(startup, `  window.__S1FractureHooks = {
    normalizeLoadedCase,
    renderFractureEvents,
    collectFractureEventsFromDom,
    saveDraft,
    loadCase,
    getStore,
    syncUiFromState,
    bindStep2Inputs,
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
    style: {},
    _listeners: Object.create(null),
    textContent: "",
    dataset: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute() {},
    getAttribute() { return null; },
    addEventListener(type, handler) { (this._listeners[type] ||= []).push(handler); },
    dispatch(type, target) { (this._listeners[type] || []).forEach(handler => handler({ target: target || this })); },
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
hooks.bindStep2Inputs();

const semanticSandbox = { window: {}, console };
vm.createContext(semanticSandbox);
vm.runInContext(g2Source, semanticSandbox, { filename: "osteoporosis-evidence-guidance-core.js" });
const g2 = semanticSandbox.window.BaselineOsteoporosisEvidenceGuidance;
assert(g2, "G2 evidence core missing");

const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
const CANONICAL_LOW_TRAUMA = new Set(["yes", "no", "uncertain", ""]);

function semanticFragility(raw) {
  const context = g2.buildEvidenceContext({
    internal_uuid: "semantic-current",
    encounter_date: "2026-09-28",
    fracture_history: {
      interval_fracture_status: "yes",
      events: [{ id: "semantic-fracture", site: "hip", month: "2026-09", low_trauma: raw }]
    },
    anthropometrics: {},
    risk_context: {},
    risk_assessment: {},
    step3: { dxa: {}, secondary: {} },
    step4: { administrations: [], treatment_episodes: [], decision: {}, transition: {} }
  }, {}, { new_events: {} }, { historicalEncounters: [] });
  return context.current_fragility_fracture;
}

function makeStoredCase(id, raw, extra = {}) {
  return {
    internal_uuid: id,
    case_sequence_number: 1,
    encounter_date: "2026-09-28",
    risk_context: {
      prior_fragility_fracture: Boolean(extra.prior_fragility_fracture),
      last_fracture_site: extra.last_fracture_site || "",
      last_fracture_month: extra.last_fracture_month || ""
    },
    fracture_history: {
      reviewed: "yes",
      review_scope: "full_history",
      interval_fracture_status: "yes",
      events: extra.zero_events ? [] : [{
        id: `${id}-fracture`,
        site: "hip",
        month: "2026-08",
        low_trauma: raw,
        occurred_on_treatment: "no",
        vertebral_level: ""
      }]
    }
  };
}

function makeRenderedRow(event, displayedLowTrauma) {
  const row = { dataset: { eventId: event.id }, querySelectorAll(selector) { return selector === "[data-event-field]" ? fields : []; } };
  const makeField = (eventField, value) => ({
    dataset: { eventField },
    value,
    closest(selector) { return selector === ".fracture-event" ? row : null; }
  });
  const fields = [
    makeField("site", event.site || ""),
    makeField("month", event.month || ""),
    makeField("low_trauma", displayedLowTrauma),
    makeField("occurred_on_treatment", event.occurred_on_treatment || ""),
    makeField("vertebral_level", event.vertebral_level || "")
  ];
  return { row, lowTraumaField: fields[2] };
}

function loadRenderSaveNoEdit(raw, id) {
  localStore.set(STORAGE_KEY, JSON.stringify([makeStoredCase(id, raw)]));
  fractureRoot._rows = [];
  hooks.loadCase(id);
  const event = hooks.getCurrentCase().fracture_history.events[0];
  const displayed = CANONICAL_LOW_TRAUMA.has(raw) ? raw : "";
  const rendered = makeRenderedRow(event, displayed);
  fractureRoot._rows = [rendered.row];
  const renderHtml = fractureRoot.innerHTML;
  hooks.saveDraft(false);
  const saved = hooks.getStore().find(item => item.internal_uuid === id);
  return { raw: saved.fracture_history.events[0].low_trauma, renderHtml };
}

function loadRenderEditSave(raw, next, id) {
  localStore.set(STORAGE_KEY, JSON.stringify([makeStoredCase(id, raw)]));
  fractureRoot._rows = [];
  hooks.loadCase(id);
  const event = hooks.getCurrentCase().fracture_history.events[0];
  const rendered = makeRenderedRow(event, CANONICAL_LOW_TRAUMA.has(raw) ? raw : "");
  fractureRoot._rows = [rendered.row];
  rendered.lowTraumaField.value = next;
  fractureRoot.dispatch("change", rendered.lowTraumaField);
  hooks.saveDraft(false);
  const saved = hooks.getStore().find(item => item.internal_uuid === id);
  return saved.fracture_history.events[0].low_trauma;
}

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
          { dataset: { eventField: "low_trauma", lowTraumaEdited: "true" }, value: mechanism },
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


// 5. Actual load/render/save path preserves exact raw low_trauma when the clinician does not edit the control.
{
  const cases = [
    ["yes", "raw-yes"],
    ["no", "raw-no"],
    ["uncertain", "raw-uncertain"],
    ["", "raw-empty"],
    ["unknown", "raw-unknown"],
    [" YES ", "raw-spaced-yes"],
    ["legacy-unrecognised-token", "raw-other"]
  ];
  cases.forEach(([raw, id]) => {
    const saved = loadRenderSaveNoEdit(raw, id);
    assert.strictEqual(saved.raw, raw, `no-edit save must preserve exact raw low_trauma ${JSON.stringify(raw)}`);
  });
}

// 6. A noncanonical raw value can render as the blank select option without being rewritten by save.
{
  const saved = loadRenderSaveNoEdit(" YES ", "render-blank-preserve");
  assert(!saved.renderHtml.includes('option value="yes" selected'), "noncanonical raw should not be silently presented as a canonical selection");
  assert.strictEqual(saved.raw, " YES ");
}

// 7. Explicit clinician edits replace the preserved legacy raw value with the chosen canonical option.
{
  assert.strictEqual(loadRenderEditSave("unknown", "yes", "explicit-to-yes"), "yes");
  assert.strictEqual(loadRenderEditSave(" YES ", "no", "explicit-to-no"), "no");
}

// 8. Normalized clinical interpretation is stable across a no-edit save; source preservation and semantic interpretation stay separate.
{
  [" YES ", "unknown"].forEach((raw, index) => {
    const before = semanticFragility(raw);
    const saved = loadRenderSaveNoEdit(raw, `semantic-stability-${index}`).raw;
    const after = semanticFragility(saved);
    assert.strictEqual(after, before, `semantic interpretation changed across no-edit save for ${JSON.stringify(raw)}`);
  });
  assert.strictEqual(semanticFragility(" YES "), true, "accepted normalizer should continue to interpret trimmed/case-folded yes as positive");
  assert.strictEqual(semanticFragility("unknown"), false, "unknown raw value must remain fail-closed");
}

// 9. Legacy prior=true + zero structured events stays zero through actual load/render/save.
{
  const id = "legacy-zero-events-save";
  localStore.set(STORAGE_KEY, JSON.stringify([makeStoredCase(id, "", {
    prior_fragility_fracture: true,
    last_fracture_site: "hip",
    last_fracture_month: "2024-10",
    zero_events: true
  })]));
  fractureRoot._rows = [];
  const beforeUuidCalls = uuidCalls;
  hooks.loadCase(id);
  hooks.saveDraft(false);
  const saved = hooks.getStore().find(item => item.internal_uuid === id);
  assert.strictEqual(saved.fracture_history.events.length, 0);
  assert.strictEqual(saved.risk_context.prior_fragility_fracture, true);
  assert.strictEqual(uuidCalls, beforeUuidCalls, "load/render/save must not synthesize a fracture event UUID");
}

assert(source.includes('fieldName === "low_trauma" && field.dataset.lowTraumaEdited !== "true"'), "raw-preservation writer guard missing");
assert((source.match(/dataset\.lowTraumaEdited = "true"/g) || []).length === 2, "both fracture input/change paths must mark explicit low-trauma edits");

console.log("S1 app-core fracture/fragility writer + load/render regressions: PASS");
