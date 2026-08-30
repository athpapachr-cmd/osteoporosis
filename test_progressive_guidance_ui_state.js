"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync("static/baseline-audit/progressive-guidance-ui.js", "utf8");

const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
const ACTIVE_PATIENT_KEY = "osteoporosis.clinical.activePatient.v1";

function createStorage(initial = {}) {
  const store = new Map(Object.entries(initial));
  return {
    getItem(key) { return store.has(key) ? store.get(key) : null; },
    setItem(key, value) { store.set(key, String(value)); },
    removeItem(key) { store.delete(key); }
  };
}

function control(value = "") {
  return {
    value,
    addEventListener() {},
    closest() { return null; },
    matches() { return false; }
  };
}

function fractureRoot(events = []) {
  return {
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll(selector) {
      if (selector !== ".fracture-event") return [];
      return events.map(event => ({
        dataset: { eventId: event.id || "" },
        querySelectorAll(fieldSelector) {
          if (fieldSelector !== "[data-event-field]") return [];
          return Object.entries(event)
            .filter(([key]) => key !== "id")
            .map(([key, value]) => ({ dataset: { eventField: key }, value }));
        }
      }));
    }
  };
}

function genericElement() {
  return {
    dataset: {},
    className: "",
    textContent: "",
    innerHTML: "",
    hidden: false,
    classList: { add() {}, remove() {} },
    style: { removeProperty() {} },
    appendChild() {},
    append() {},
    prepend() {},
    insertAdjacentElement() {},
    remove() {},
    setAttribute() {},
    removeAttribute() {},
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
    closest() { return null; },
    matches() { return false; }
  };
}

function createHarness({ persistedCase, nodes = {}, fetchImpl = async () => ({ ok: true, json: async () => [] }), patientId = "" }) {
  const localStorage = createStorage({
    [STORAGE_KEY]: JSON.stringify([persistedCase]),
    [ACTIVE_KEY]: persistedCase.internal_uuid
  });
  const sessionStorage = createStorage(patientId ? { [ACTIVE_PATIENT_KEY]: patientId } : {});

  const document = {
    head: { appendChild() {} },
    querySelector(selector) { return nodes[selector] || null; },
    querySelectorAll(selector) {
      if (selector === ".step-tab" || selector === "article.card.guidance-surfaced") return [];
      return [];
    },
    createElement() { return genericElement(); },
    createTextNode(value) { return { textContent: String(value) }; },
    addEventListener() {}
  };

  let timerId = 0;
  const context = {
    console,
    JSON,
    Object,
    Array,
    String,
    Number,
    Boolean,
    Error,
    Promise,
    encodeURIComponent,
    localStorage,
    sessionStorage,
    document,
    fetch: fetchImpl,
    clearTimeout() {},
    setTimeout() { timerId += 1; return timerId; }
  };
  context.window = context;
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(source, context, { filename: "progressive-guidance-ui.js" });
  return { context, ui: context.window.ProgressiveGuidanceUI, localStorage, sessionStorage };
}

function baseCase() {
  return {
    internal_uuid: "case-1",
    encounter_archetype: "other",
    encounter_date: "2026-08-30",
    quick_notes: "persisted note",
    fracture_history: {
      interval_fracture_status: "yes",
      events: [{ id: "old-fracture", occurred_on_treatment: "yes" }]
    }
  };
}

async function testHistoryUnavailableIsNotZeroHistory() {
  const h = createHarness({
    persistedCase: baseCase(),
    patientId: "SYNTH-P1",
    fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) })
  });

  await h.ui.refresh();
  const state = h.ui.getHistoryLoadState();
  assert.strictEqual(state.status, "unavailable");
  assert.strictEqual(state.patient_id, "SYNTH-P1");
  assert.strictEqual(state.encounter_count, 0);
  assert.strictEqual(state.error_present, true);

  const text = h.ui.getLongitudinalMetaText({ prior_encounter_count: 0 });
  assert.match(text, /δεν ήταν δυνατή/i);
  assert.match(text, /Μην θεωρήσεις/i);
  assert.doesNotMatch(text, /^0 προηγούμενες/);
}

async function testSuccessfulEmptyHistoryIsExplicitlyLoadedZero() {
  const h = createHarness({
    persistedCase: baseCase(),
    patientId: "SYNTH-EMPTY",
    fetchImpl: async () => ({ ok: true, status: 200, json: async () => [] })
  });

  await h.ui.refresh();
  const state = h.ui.getHistoryLoadState();
  assert.strictEqual(state.status, "loaded");
  assert.strictEqual(state.encounter_count, 0);
  assert.match(h.ui.getLongitudinalMetaText({ prior_encounter_count: 0 }), /^0 προηγούμενες/);
}

async function testNewPatientLoadClearsOldHistoryBeforeAwaitingResponse() {
  let resolveSecond;
  const h = createHarness({
    persistedCase: baseCase(),
    patientId: "SYNTH-P1",
    fetchImpl: async (url) => {
      if (url.includes("SYNTH-P1")) {
        return { ok: true, status: 200, json: async () => [{ encounter_id: "e-old" }] };
      }
      if (url.includes("SYNTH-P2")) {
        return new Promise(resolve => {
          resolveSecond = () => resolve({ ok: true, status: 200, json: async () => [] });
        });
      }
      throw new Error("unexpected patient");
    }
  });

  await h.ui.refresh();
  assert.strictEqual(h.ui.getHistoryLoadState().status, "loaded");
  assert.strictEqual(h.ui.getHistoryLoadState().encounter_count, 1);

  h.sessionStorage.setItem(ACTIVE_PATIENT_KEY, "SYNTH-P2");
  const pending = h.ui.refresh();
  const loadingState = h.ui.getHistoryLoadState();
  assert.strictEqual(loadingState.status, "loading");
  assert.strictEqual(loadingState.patient_id, "SYNTH-P2");
  assert.strictEqual(loadingState.encounter_count, 0, "old patient history must be cleared before the new fetch resolves");

  resolveSecond();
  await pending;
  assert.strictEqual(h.ui.getHistoryLoadState().status, "loaded");
  assert.strictEqual(h.ui.getHistoryLoadState().encounter_count, 0);
}

function testLiveBlankControlsOwnCurrentSnapshot() {
  const h = createHarness({
    persistedCase: baseCase(),
    nodes: {
      "#encounterArchetype": control(""),
      "#encounterDate": control(""),
      "#quickNotes": control(""),
      "#intervalFractureStatus": control(""),
      "#fractureEvents": fractureRoot([])
    }
  });

  const snapshot = h.ui.getCurrentCaseSnapshot();
  assert.strictEqual(snapshot.encounter_archetype, "");
  assert.strictEqual(snapshot.encounter_date, "");
  assert.strictEqual(snapshot.quick_notes, "");
  assert.strictEqual(snapshot.fracture_history.interval_fracture_status, "");
  assert.deepStrictEqual(Array.from(snapshot.fracture_history.events), []);
}

function testPersistedFallbackOnlyWhenLiveControlsAreAbsent() {
  const h = createHarness({ persistedCase: baseCase(), nodes: {} });
  const snapshot = h.ui.getCurrentCaseSnapshot();
  assert.strictEqual(snapshot.encounter_archetype, "other");
  assert.strictEqual(snapshot.encounter_date, "2026-08-30");
  assert.strictEqual(snapshot.quick_notes, "persisted note");
  assert.strictEqual(snapshot.fracture_history.interval_fracture_status, "yes");
  assert.strictEqual(snapshot.fracture_history.events.length, 1);
  assert.strictEqual(snapshot.fracture_history.events[0].id, "old-fracture");
}

(async () => {
  await testHistoryUnavailableIsNotZeroHistory();
  await testSuccessfulEmptyHistoryIsExplicitlyLoadedZero();
  await testNewPatientLoadClearsOldHistoryBeforeAwaitingResponse();
  testLiveBlankControlsOwnCurrentSnapshot();
  testPersistedFallbackOnlyWhenLiveControlsAreAbsent();
  console.log("progressive guidance UI state regressions: OK");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});