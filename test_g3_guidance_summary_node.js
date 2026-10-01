"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const g1Source = fs.readFileSync("static/baseline-audit/progressive-guidance-core.js", "utf8");
const g2Source = fs.readFileSync("static/baseline-audit/osteoporosis-evidence-guidance-core.js", "utf8");
const g3Source = fs.readFileSync("static/baseline-audit/osteoporosis-longitudinal-summary-core.js", "utf8");
const sandbox = { window: {}, console };
vm.createContext(sandbox);
vm.runInContext(g1Source, sandbox, { filename: "progressive-guidance-core.js" });
vm.runInContext(g2Source, sandbox, { filename: "osteoporosis-evidence-guidance-core.js" });
vm.runInContext(g3Source, sandbox, { filename: "osteoporosis-longitudinal-summary-core.js" });

const g1 = sandbox.window.BaselineProgressiveGuidanceCore;
const g2 = sandbox.window.BaselineOsteoporosisEvidenceGuidance;
const g3 = sandbox.window.BaselineOsteoporosisLongitudinalSummary;
assert(g1 && g2 && g3, "G1/G2/G3 core missing");

function encounter(id, date, payload, status = "completed", updated = `${date}T12:00:00`) {
  return { encounter_id: id, encounter_date: date, status, payload, updated_at: updated };
}

function baseCase(overrides = {}) {
  const base = {
    internal_uuid: "current-case",
    encounter_archetype: "routine_followup_stable",
    encounter_date: "2026-09-01",
    age_years: 72,
    sex: "female",
    menopause_status: "postmenopausal",
    osteoporosis_status: "osteoporosis",
    patient_relationship_status: "established_patient",
    anthropometrics: { current_height_cm: 160, reference_height_cm: 163.9, derived_height_loss_cm: 3.9 },
    risk_context: { glucocorticoids: false, falls_last_12_months: 0 },
    risk_assessment: { formal_indicated: "no", declared_framework: "nogg_2024", resulting_risk_category: "high" },
    fracture_history: { interval_fracture_status: "no", events: [] },
    step3: { dxa: { used: "no" }, secondary: { prior_workup_adequate: "yes" } },
    step4: { treatment_episodes: [], administrations: [], decision: { type: "", selected_agent: "" }, transition: {} }
  };
  return {
    ...base,
    ...overrides,
    anthropometrics: { ...base.anthropometrics, ...(overrides.anthropometrics || {}) },
    risk_context: { ...base.risk_context, ...(overrides.risk_context || {}) },
    risk_assessment: { ...base.risk_assessment, ...(overrides.risk_assessment || {}) },
    fracture_history: { ...base.fracture_history, ...(overrides.fracture_history || {}) },
    step3: { ...base.step3, ...(overrides.step3 || {}), dxa: { ...base.step3.dxa, ...(overrides.step3?.dxa || {}) } },
    step4: { ...base.step4, ...(overrides.step4 || {}), decision: { ...base.step4.decision, ...(overrides.step4?.decision || {}) } }
  };
}

function planFor(current, historical = []) {
  const projection = g1.buildLongitudinalProjection(historical, { currentInternalUuid: current.internal_uuid });
  const context = g1.buildEncounterContext(current, projection);
  const basePlan = g1.buildVisitPlan(context);
  const evidenceContext = g2.buildEvidenceContext(current, projection, context, { historicalEncounters: historical });
  const contributions = g2.evaluateEvidenceGuidance(evidenceContext);
  return { projection, plan: g2.mergeEvidenceContributions(basePlan, contributions), contributions };
}

// 1. Initial plan establishes a baseline and never marks all visible guidance as new.
{
  const initial = planFor(baseCase());
  const state = g3.advanceSalienceState({ items: initial.plan.ordered_cards, initialize: true });
  assert.deepStrictEqual(Array.from(state.newly_surfaced_domains), []);
}

// 2. Exact product-owner example: live height loss crossing from <4 cm to >=4 cm newly surfaces VFA.
{
  const before = planFor(baseCase({ anthropometrics: { derived_height_loss_cm: 3.9 } }));
  assert(!before.contributions.some(item => item.rule_id === "OST_G2_R02_VFA_STRUCTURED_TRIGGER"));
  const baseline = g3.advanceSalienceState({ items: before.plan.ordered_cards, initialize: true });

  const after = planFor(baseCase({ anthropometrics: { derived_height_loss_cm: 4.0 } }));
  assert(after.contributions.some(item => item.rule_id === "OST_G2_R02_VFA_STRUCTURED_TRIGGER"));
  const changed = g3.advanceSalienceState({
    previousDomains: baseline.current_domains,
    retainedNewDomains: baseline.newly_surfaced_domains,
    items: after.plan.ordered_cards
  });
  assert(changed.newly_surfaced_domains.includes("vfa"), "VFA should be marked newly surfaced after crossing 4 cm");

  const stable = g3.advanceSalienceState({
    previousDomains: changed.current_domains,
    retainedNewDomains: changed.newly_surfaced_domains,
    items: after.plan.ordered_cards
  });
  assert(stable.newly_surfaced_domains.includes("vfa"), "new salience should persist while the newly surfaced guidance remains applicable");

  const reverted = planFor(baseCase({ anthropometrics: { derived_height_loss_cm: 3.9 } }));
  const cleared = g3.advanceSalienceState({
    previousDomains: stable.current_domains,
    retainedNewDomains: stable.newly_surfaced_domains,
    items: reverted.plan.ordered_cards
  });
  assert(!cleared.newly_surfaced_domains.includes("vfa"), "new salience should clear when the item stops applying");
}

// 3. Pure archetype/base-flow additions are not eligible for salience noise without evidence/event/due/treatment context.
{
  const state = g3.advanceSalienceState({
    previousDomains: [],
    retainedNewDomains: [],
    items: [{ card_id: "documentation_capture", reason_codes: ["VISIT_TYPE_CORE"], evidence_rules: [] }]
  });
  assert.deepStrictEqual(Array.from(state.newly_surfaced_domains), []);
}

// 4. Longitudinal summary uses completed/amended history, excludes current UUID, and keeps first/latest chronology.
{
  const encounters = [
    encounter("e1", "2025-01-10", { internal_uuid: "old-1" }),
    encounter("e2", "2025-06-10", { internal_uuid: "old-2" }, "amended"),
    encounter("e-current", "2026-09-01", { internal_uuid: "current-case" })
  ];
  const projection = g1.buildLongitudinalProjection(encounters, { currentInternalUuid: "current-case" });
  const summary = g3.buildSummary({ encounters, projection, currentCase: baseCase(), historyStatus: "loaded" });
  assert.strictEqual(summary.course.encounter_count, 2);
  assert.strictEqual(summary.course.first_date, "2025-01-10");
  assert.strictEqual(summary.course.latest_date, "2025-06-10");
  assert.strictEqual(summary.current_visit.state, "current_non_historical");
}

// 5. A later blank encounter does not erase the latest prior authoritative risk or DXA state.
{
  const encounters = [
    encounter("e1", "2025-01-10", {
      internal_uuid: "old-1",
      risk_assessment: { resulting_risk_category: "very_high", declared_framework: "nogg_2024", frax_mof_percent: 28, frax_hip_percent: 12 },
      step3: { dxa: { used: "yes", date: "2024-12-20", spine_t: -3.1, total_hip_t: -2.7, femoral_neck_t: -2.8 } }
    }),
    encounter("e2", "2025-07-10", { internal_uuid: "old-2", risk_assessment: {}, step3: { dxa: { used: "no" } } })
  ];
  const projection = g1.buildLongitudinalProjection(encounters);
  const summary = g3.buildSummary({ encounters, projection, currentCase: baseCase(), historyStatus: "loaded" });
  assert.strictEqual(summary.risk.category, "very_high");
  assert.strictEqual(summary.risk.source_encounter_id, "e1");
  assert.strictEqual(summary.dxa.date, "2024-12-20");
  assert.strictEqual(summary.dxa.spine_t, -3.1);
}

// 6. Repeated snapshots of the same stable fracture event are not double-counted.
{
  const fracture = { id: "fx-1", site: "vertebral", month: "2025-03", low_trauma: "yes" };
  const encounters = [
    encounter("e1", "2025-03-20", { internal_uuid: "old-1", fracture_history: { events: [fracture] }, risk_context: { prior_fragility_fracture: true } }),
    encounter("e2", "2025-06-20", { internal_uuid: "old-2", fracture_history: { events: [fracture] }, risk_context: { prior_fragility_fracture: true } })
  ];
  const summary = g3.buildSummary({ encounters, projection: g1.buildLongitudinalProjection(encounters), currentCase: baseCase(), historyStatus: "loaded" });
  assert.strictEqual(summary.fractures.documented_count, 1);
  assert.strictEqual(summary.fractures.prior_fragility_fracture, true);
  assert.strictEqual(summary.fractures.confirmed_fragility_count, 1);
  assert.strictEqual(summary.fractures.fragility_confirmation, "confirmed_event");
  assert.strictEqual(summary.fractures.most_recent.month, "2025-03");
}

// 7. Treatment summary reuses G1 actual-administration semantics: scheduled-only never appears as an actual dose.
{
  const encounters = [
    encounter("e1", "2025-01-01", { internal_uuid: "old-1", step4: { treatment_episodes: [{ id: "ep", agent: "denosumab", status: "active", start_date: "2025-01-01" }], administrations: [{ id: "a1", agent: "denosumab", actual_date: "2025-01-01", status: "done" }] } }),
    encounter("e2", "2025-07-01", { internal_uuid: "old-2", step4: { treatment_episodes: [{ id: "ep", agent: "denosumab", status: "active", start_date: "2025-01-01" }], administrations: [{ id: "planned", agent: "denosumab", scheduled_date: "2025-07-01", status: "planned" }] } })
  ];
  const projection = g1.buildLongitudinalProjection(encounters);
  const summary = g3.buildSummary({ encounters, projection, currentCase: baseCase(), historyStatus: "loaded" });
  assert.strictEqual(summary.treatment.actual_event_count, 1);
  assert.strictEqual(summary.treatment.latest_actual.actual_date, "2025-01-01");
  assert.strictEqual(summary.treatment.administration_count_by_agent.denosumab, 1);
}

// 8. Treatment/admin conflict is surfaced, never silently reconciled.
{
  const encounters = [
    encounter("e1", "2025-01-01", { internal_uuid: "old-1", step4: { administrations: [{ id: "same", agent: "denosumab", actual_date: "2025-01-01" }] } }),
    encounter("e2", "2025-02-01", { internal_uuid: "old-2", step4: { administrations: [{ id: "same", agent: "denosumab", actual_date: "2025-02-01" }] } })
  ];
  const projection = g1.buildLongitudinalProjection(encounters);
  const summary = g3.buildSummary({ encounters, projection, currentCase: baseCase(), historyStatus: "loaded" });
  assert.strictEqual(summary.treatment.state, "conflicting");
  assert(summary.unresolved.conflicts.length > 0);
}

// 9. Latest protected lab snapshot and latest explicit management decision are selected deterministically.
{
  const encounters = [
    encounter("e1", "2025-01-01", { internal_uuid: "old-1", step4: { decision: { type: "start", selected_agent: "alendronate" } } }),
    encounter("e2", "2026-01-01", { internal_uuid: "old-2", step4: { decision: { type: "switch", selected_agent: "denosumab" } } })
  ];
  const labs = [
    { lab_date: "2025-01-02", values: { ca: 9.2, vitamin_d: 31 } },
    { lab_date: "2026-02-02", values: { ca: 9.5, vitamin_d: 42, egfr: 78, ctx: 0.18 } }
  ];
  const summary = g3.buildSummary({ encounters, labs, projection: g1.buildLongitudinalProjection(encounters), currentCase: baseCase(), historyStatus: "loaded" });
  assert.strictEqual(summary.decision.type, "switch");
  assert.strictEqual(summary.decision.selected_agent, "denosumab");
  assert.strictEqual(summary.labs.date, "2026-02-02");
  assert(summary.labs.values.some(item => item.key === "ctx" && item.value === 0.18));
}

// 10. Unavailable history is explicit and never represented as zero prior history.
{
  const summary = g3.buildSummary({ encounters: [], labs: [], projection: {}, currentCase: baseCase(), historyStatus: "unavailable" });
  assert.strictEqual(summary.state, "unavailable");
  assert(!Object.prototype.hasOwnProperty.call(summary, "course"));
}

// 11. S1: generic traumatic/uncertain/missing fractures remain documented without becoming confirmed fragility.
{
  const mechanisms = ["no", "uncertain", "", undefined];
  mechanisms.forEach((mechanism, index) => {
    const event = { id: `s1-g3-generic-${index}`, site: "hip", month: "2025-04" };
    if (mechanism !== undefined) event.low_trauma = mechanism;
    const encounters = [
      encounter(`s1-g3-e-${index}`, "2025-04-20", {
        fracture_history: { events: [event] },
        risk_context: { prior_fragility_fracture: false }
      })
    ];
    const summary = g3.buildSummary({ encounters, projection: g1.buildLongitudinalProjection(encounters), currentCase: baseCase(), historyStatus: "loaded" });
    assert.strictEqual(summary.fractures.documented_count, 1);
    assert.strictEqual(summary.fractures.prior_fragility_fracture, false);
    assert.strictEqual(summary.fractures.confirmed_fragility_count, 0);
    assert.strictEqual(summary.fractures.fragility_confirmation, "not_confirmed");
  });
}

// 12. S1: legacy compatibility prior=true with zero structured events stays zero and is not upgraded to confirmed fragility.
{
  const encounters = [
    encounter("s1-g3-legacy-zero", "2025-05-20", {
      risk_context: {
        prior_fragility_fracture: true,
        last_fracture_site: "hip",
        last_fracture_month: "2024-10"
      },
      fracture_history: { events: [] }
    })
  ];
  const summary = g3.buildSummary({ encounters, projection: g1.buildLongitudinalProjection(encounters), currentCase: baseCase(), historyStatus: "loaded" });
  assert.strictEqual(summary.fractures.documented_count, 0);
  assert.strictEqual(summary.fractures.prior_fragility_fracture, false);
  assert.strictEqual(summary.fractures.confirmed_fragility_count, 0);
  assert.strictEqual(summary.fractures.fragility_confirmation, "legacy_claim_unconfirmed");
  assert.strictEqual(summary.fractures.most_recent.site, "hip");
}

// 13. S1: a structured low_trauma=yes event controls derived fragility even if compatibility prior=false.
{
  const encounters = [
    encounter("s1-g3-structured-yes", "2025-06-20", {
      risk_context: { prior_fragility_fracture: false },
      fracture_history: { events: [{ id: "s1-g3-yes", site: "distal_radius", month: "2025-06", low_trauma: "yes" }] }
    })
  ];
  const summary = g3.buildSummary({ encounters, projection: g1.buildLongitudinalProjection(encounters), currentCase: baseCase(), historyStatus: "loaded" });
  assert.strictEqual(summary.fractures.documented_count, 1);
  assert.strictEqual(summary.fractures.prior_fragility_fracture, true);
  assert.strictEqual(summary.fractures.confirmed_fragility_count, 1);
  assert.strictEqual(summary.fractures.fragility_confirmation, "confirmed_event");
}

// 14. S1: stale event.fragility is non-canonical and cannot upgrade an event without low_trauma.
{
  const encounters = [
    encounter("s1-g3-stale-field", "2025-07-20", {
      risk_context: { prior_fragility_fracture: false },
      fracture_history: { events: [{ id: "s1-g3-stale", site: "hip", month: "2025-07", fragility: "yes" }] }
    })
  ];
  const summary = g3.buildSummary({ encounters, projection: g1.buildLongitudinalProjection(encounters), currentCase: baseCase(), historyStatus: "loaded" });
  assert.strictEqual(summary.fractures.documented_count, 1);
  assert.strictEqual(summary.fractures.prior_fragility_fracture, false);
  assert.strictEqual(summary.fractures.confirmed_fragility_count, 0);
}

// 15. S1: conflicting explicit mechanism snapshots for one stable ID fail closed without duplicating the fracture.
{
  const encounters = [
    encounter("s1-g3-conflict-1", "2025-08-01", {
      fracture_history: { events: [{ id: "s1-g3-conflict", site: "vertebral", month: "2025-07", low_trauma: "yes" }] }
    }),
    encounter("s1-g3-conflict-2", "2025-08-15", {
      fracture_history: { events: [{ id: "s1-g3-conflict", site: "vertebral", month: "2025-07", low_trauma: "no" }] }
    })
  ];
  const summary = g3.buildSummary({ encounters, projection: g1.buildLongitudinalProjection(encounters), currentCase: baseCase(), historyStatus: "loaded" });
  assert.strictEqual(summary.fractures.documented_count, 1);
  assert.strictEqual(summary.fractures.prior_fragility_fracture, false);
  assert.strictEqual(summary.fractures.confirmed_fragility_count, 0);
  assert.strictEqual(summary.fractures.count_reliability, "partial");
}

// 16. S1: legacy prior=true plus uncertain/missing structured mechanism remains unconfirmed rather than upgraded.
{
  ["uncertain", ""].forEach((mechanism, index) => {
    const event = { id: `s1-g3-legacy-uncertain-${index}`, site: "hip", month: "2025-09" };
    if (mechanism) event.low_trauma = mechanism;
    const encounters = [
      encounter(`s1-g3-legacy-u-${index}`, "2025-09-20", {
        risk_context: { prior_fragility_fracture: true },
        fracture_history: { events: [event] }
      })
    ];
    const summary = g3.buildSummary({ encounters, projection: g1.buildLongitudinalProjection(encounters), currentCase: baseCase(), historyStatus: "loaded" });
    assert.strictEqual(summary.fractures.documented_count, 1);
    assert.strictEqual(summary.fractures.prior_fragility_fracture, false);
    assert.strictEqual(summary.fractures.fragility_confirmation, "legacy_claim_unconfirmed");
  });
}

console.log("G3 guidance salience + longitudinal summary regressions: PASS");
