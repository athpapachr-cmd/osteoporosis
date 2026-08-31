"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync("static/baseline-audit/progressive-guidance-ui.js", "utf8");
const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";

function createStorage(initial = {}) {
  const store = new Map(Object.entries(initial));
  return {
    getItem(key) { return store.has(key) ? store.get(key) : null; },
    setItem(key, value) { store.set(key, String(value)); },
    removeItem(key) { store.delete(key); }
  };
}

function classList(selected = false) {
  return {
    contains(name) { return name === "selected" && selected; },
    add() {},
    remove() {}
  };
}

function control(value = "", options = {}) {
  return {
    value,
    checked: Boolean(options.checked),
    type: options.type || "text",
    dataset: options.dataset || {},
    classList: classList(Boolean(options.selected)),
    getAttribute(name) { return name === "aria-pressed" && options.selected ? "true" : null; },
    addEventListener() {},
    closest() { return null; },
    matches() { return false; }
  };
}

function choice(field, value, selected = false) {
  return control("", { selected, dataset: { field, value } });
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

function repeatRoot(rows, rowSelector, idAttribute) {
  return {
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll(selector) {
      if (selector !== rowSelector) return [];
      return rows.map(item => ({
        dataset: {},
        getAttribute(name) { return name === idAttribute ? item.id || "" : null; },
        querySelectorAll(fieldSelector) {
          if (fieldSelector !== "[data-k]") return [];
          return Object.entries(item)
            .filter(([key]) => key !== "id")
            .map(([key, value]) => control(value === null ? "" : String(value), {
              type: typeof value === "number" ? "number" : "text",
              dataset: { k: key }
            }));
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

function createHarness({ persistedCase, nodes = {}, lists = {} }) {
  const localStorage = createStorage({
    [STORAGE_KEY]: JSON.stringify([persistedCase]),
    [ACTIVE_KEY]: persistedCase.internal_uuid
  });
  const sessionStorage = createStorage();
  const document = {
    head: { appendChild() {} },
    querySelector(selector) { return nodes[selector] || null; },
    querySelectorAll(selector) {
      if (lists[selector]) return lists[selector];
      if (selector === ".step-tab" || selector === "article.card.guidance-surfaced") return [];
      return [];
    },
    createElement() { return genericElement(); },
    createTextNode(value) { return { textContent: String(value) }; },
    addEventListener() {}
  };
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
    fetch: async () => ({ ok: true, json: async () => [] }),
    clearTimeout() {},
    setTimeout() { return 1; }
  };
  context.window = context;
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(source, context, { filename: "progressive-guidance-ui.js" });
  return context.window.ProgressiveGuidanceUI;
}

function persistedCase() {
  return {
    internal_uuid: "case-g2",
    encounter_archetype: "treatment_start",
    encounter_date: "2026-08-31",
    age_years: 72,
    sex: "female",
    menopause_status: "postmenopausal",
    patient_relationship_status: "established_patient",
    osteoporosis_status: "osteoporosis",
    quick_notes: "persisted",
    anthropometrics: { current_height_cm: 155, reference_height_cm: 160, derived_height_loss_cm: 5 },
    risk_context: {
      glucocorticoids: true,
      glucocorticoid_prednisolone_mg_day: 10,
      glucocorticoid_duration_months: 12,
      falls_last_12_months: 3
    },
    risk_assessment: { formal_indicated: "yes", declared_framework: "nogg_2024", resulting_risk_category: "very_high" },
    fracture_history: { interval_fracture_status: "yes", events: [{ id: "old", site: "vertebral", month: "2026-01", low_trauma: "yes" }] },
    step3: {
      dxa: { used: "yes", spine_t: -3.8, total_hip_t: -3.1, femoral_neck_t: -3.2 },
      secondary: { prior_workup_adequate: "yes" }
    },
    step4: {
      treatment_episodes: [{ id: "old-ep", agent: "denosumab", status: "active", start_date: "2024-01-01" }],
      administrations: [{ id: "old-ad", agent: "denosumab", actual_date: "2026-02-28", status: "done" }],
      decision: { type: "start", selected_agent: "denosumab" },
      transition: { relevant: "yes", type: "denosumab_exit", next_agent: "zoledronate", prior_end_date: "2026-02-28" }
    }
  };
}

// Live blank/deleted state must own today's snapshot over stale persisted values.
{
  const nodes = {
    "#encounterArchetype": control(""),
    "#encounterDate": control(""),
    "#ageYears": control(""),
    "#quickNotes": control(""),
    "#currentHeightCm": control(""),
    "#referenceHeightCm": control(""),
    "#glucocorticoids": control("", { checked: false, type: "checkbox" }),
    "#gcDoseMg": control("", { type: "number" }),
    "#gcDurationMonths": control("", { type: "number" }),
    "#fallsLast12m": control("", { type: "number" }),
    "#formalRiskIndicated": control(""),
    "#declaredRiskFramework": control(""),
    "#resultingRiskCategory": control(""),
    "#intervalFractureStatus": control(""),
    "#fractureEvents": fractureRoot([]),
    "#s3DxaUsed": control(""),
    "#s3SpineT": control("", { type: "number" }),
    "#s3TotalHipT": control("", { type: "number" }),
    "#s3FnT": control("", { type: "number" }),
    "#s3PriorWorkupAdequate": control(""),
    "#s4Episodes": repeatRoot([], "[data-episode-id]", "data-episode-id"),
    "#s4Administrations": repeatRoot([], "[data-admin-id]", "data-admin-id"),
    "#s4DecisionType": control(""),
    "#s4SelectedAgent": control(""),
    "#s4TransitionRelevant": control(""),
    "#s4TransitionType": control(""),
    "#s4PriorAgentEnd": control(""),
    "#s4NextAgent": control(""),
    "#s4NextAgentDate": control("")
  };
  const lists = {
    '[data-field="sex"][data-value]': [choice("sex", "female"), choice("sex", "male")],
    '[data-field="menopause_status"][data-value]': [choice("menopause_status", "postmenopausal")],
    '[data-field="patient_relationship_status"][data-value]': [choice("patient_relationship_status", "established_patient")],
    '[data-field="osteoporosis_status"][data-value]': [choice("osteoporosis_status", "osteoporosis")]
  };
  const ui = createHarness({ persistedCase: persistedCase(), nodes, lists });
  const snapshot = ui.getCurrentCaseSnapshot();
  assert.strictEqual(snapshot.encounter_archetype, "");
  assert.strictEqual(snapshot.encounter_date, "");
  assert.strictEqual(snapshot.age_years, null);
  assert.strictEqual(snapshot.sex, "");
  assert.strictEqual(snapshot.menopause_status, "");
  assert.strictEqual(snapshot.osteoporosis_status, "");
  assert.strictEqual(snapshot.quick_notes, "");
  assert.strictEqual(snapshot.anthropometrics.derived_height_loss_cm, null);
  assert.strictEqual(snapshot.risk_context.glucocorticoids, false);
  assert.strictEqual(snapshot.risk_context.glucocorticoid_prednisolone_mg_day, null);
  assert.strictEqual(snapshot.risk_context.glucocorticoid_duration_months, null);
  assert.strictEqual(snapshot.risk_context.falls_last_12_months, null);
  assert.strictEqual(snapshot.risk_assessment.formal_indicated, "");
  assert.strictEqual(snapshot.risk_assessment.declared_framework, "");
  assert.strictEqual(snapshot.risk_assessment.resulting_risk_category, "");
  assert.deepStrictEqual(Array.from(snapshot.fracture_history.events), []);
  assert.strictEqual(snapshot.step3.dxa.used, "");
  assert.strictEqual(snapshot.step3.dxa.spine_t, null);
  assert.strictEqual(snapshot.step3.secondary.prior_workup_adequate, "");
  assert.deepStrictEqual(Array.from(snapshot.step4.treatment_episodes), []);
  assert.deepStrictEqual(Array.from(snapshot.step4.administrations), []);
  assert.strictEqual(snapshot.step4.decision.type, "");
  assert.strictEqual(snapshot.step4.decision.selected_agent, "");
  assert.strictEqual(snapshot.step4.transition.type, "");
  assert.strictEqual(snapshot.step4.transition.next_agent, "");
}

// Live changed values and repeated rows must be projected without waiting for persistence.
{
  const nodes = {
    "#encounterArchetype": control("treatment_continuation_or_due_monitoring"),
    "#encounterDate": control("2026-08-31"),
    "#ageYears": control("65", { type: "number" }),
    "#quickNotes": control("live"),
    "#currentHeightCm": control("156", { type: "number" }),
    "#referenceHeightCm": control("160", { type: "number" }),
    "#glucocorticoids": control("", { checked: true, type: "checkbox" }),
    "#gcDoseMg": control("7.5", { type: "number" }),
    "#gcDurationMonths": control("3", { type: "number" }),
    "#fallsLast12m": control("1", { type: "number" }),
    "#formalRiskIndicated": control("yes"),
    "#declaredRiskFramework": control("nogg_2024"),
    "#resultingRiskCategory": control("high"),
    "#intervalFractureStatus": control("no"),
    "#fractureEvents": fractureRoot([]),
    "#s3DxaUsed": control("yes"),
    "#s3SpineT": control("-3.6", { type: "number" }),
    "#s3TotalHipT": control("-2.8", { type: "number" }),
    "#s3FnT": control("-2.9", { type: "number" }),
    "#s3PriorWorkupAdequate": control("no"),
    "#s4Episodes": repeatRoot([{ id: "live-ep", agent: "alendronate", status: "active", start_date: "2026-06-01", duration_years: 0.2 }], "[data-episode-id]", "data-episode-id"),
    "#s4Administrations": repeatRoot([{ id: "live-ad", agent: "zoledronate", scheduled_date: "2026-08-31", actual_date: "", status: "planned" }], "[data-admin-id]", "data-admin-id"),
    "#s4DecisionType": control("continue"),
    "#s4SelectedAgent": control("alendronate"),
    "#s4TransitionRelevant": control("no"),
    "#s4TransitionType": control(""),
    "#s4PriorAgentEnd": control(""),
    "#s4NextAgent": control(""),
    "#s4NextAgentDate": control("")
  };
  const lists = {
    '[data-field="sex"][data-value]': [choice("sex", "female"), choice("sex", "male", true)],
    '[data-field="menopause_status"][data-value]': [choice("menopause_status", "postmenopausal")],
    '[data-field="patient_relationship_status"][data-value]': [choice("patient_relationship_status", "established_patient", true)],
    '[data-field="osteoporosis_status"][data-value]': [choice("osteoporosis_status", "osteoporosis", true)]
  };
  const ui = createHarness({ persistedCase: persistedCase(), nodes, lists });
  const snapshot = ui.getCurrentCaseSnapshot();
  assert.strictEqual(snapshot.age_years, 65);
  assert.strictEqual(snapshot.sex, "male");
  assert.strictEqual(snapshot.anthropometrics.derived_height_loss_cm, 4);
  assert.strictEqual(snapshot.risk_assessment.formal_indicated, "yes");
  assert.strictEqual(snapshot.step3.dxa.spine_t, -3.6);
  assert.strictEqual(snapshot.step4.treatment_episodes.length, 1);
  assert.strictEqual(snapshot.step4.treatment_episodes[0].agent, "alendronate");
  assert.strictEqual(snapshot.step4.administrations.length, 1);
  assert.strictEqual(snapshot.step4.administrations[0].actual_date, "");
  assert.strictEqual(snapshot.step4.decision.type, "continue");
  assert.strictEqual(snapshot.step4.decision.selected_agent, "alendronate");
}

console.log("G2 guidance live-state regressions: OK");
