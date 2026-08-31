"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const g1Source = fs.readFileSync("static/baseline-audit/progressive-guidance-core.js", "utf8");
const g2Source = fs.readFileSync("static/baseline-audit/osteoporosis-evidence-guidance-core.js", "utf8");
const sandbox = { window: {}, console };
vm.createContext(sandbox);
vm.runInContext(g1Source, sandbox, { filename: "progressive-guidance-core.js" });
vm.runInContext(g2Source, sandbox, { filename: "osteoporosis-evidence-guidance-core.js" });

const g1 = sandbox.window.BaselineProgressiveGuidanceCore;
const g2 = sandbox.window.BaselineOsteoporosisEvidenceGuidance;
assert(g1, "G1 core missing");
assert(g2, "G2 core missing");

function encounter(id, date, payload, status = "completed", updated = `${date}T12:00:00`) {
  return { encounter_id: id, encounter_date: date, status, payload, updated_at: updated };
}

function baseCase(overrides = {}) {
  const base = {
    internal_uuid: "current-case",
    encounter_archetype: "routine_followup_stable",
    encounter_date: "2026-08-31",
    age_years: 72,
    sex: "female",
    menopause_status: "postmenopausal",
    osteoporosis_status: "osteoporosis",
    patient_relationship_status: "established_patient",
    anthropometrics: {
      current_height_cm: 160,
      reference_height_cm: 160,
      derived_height_loss_cm: 0
    },
    risk_context: {
      glucocorticoids: false,
      glucocorticoid_prednisolone_mg_day: null,
      glucocorticoid_duration_months: null,
      falls_last_12_months: 0
    },
    risk_assessment: {
      formal_indicated: "no",
      declared_framework: "nogg_2024",
      resulting_risk_category: "high"
    },
    fracture_history: { interval_fracture_status: "no", events: [] },
    step3: {
      dxa: { used: "no", spine_t: null, total_hip_t: null, femoral_neck_t: null },
      secondary: { prior_workup_adequate: "yes" }
    },
    step4: {
      treatment_episodes: [],
      administrations: [],
      decision: { type: "", selected_agent: "" },
      transition: { relevant: "", type: "", next_agent: "" }
    }
  };
  return {
    ...base,
    ...overrides,
    anthropometrics: { ...base.anthropometrics, ...(overrides.anthropometrics || {}) },
    risk_context: { ...base.risk_context, ...(overrides.risk_context || {}) },
    risk_assessment: { ...base.risk_assessment, ...(overrides.risk_assessment || {}) },
    fracture_history: { ...base.fracture_history, ...(overrides.fracture_history || {}) },
    step3: {
      ...base.step3,
      ...(overrides.step3 || {}),
      dxa: { ...base.step3.dxa, ...(overrides.step3?.dxa || {}) },
      secondary: { ...base.step3.secondary, ...(overrides.step3?.secondary || {}) }
    },
    step4: {
      ...base.step4,
      ...(overrides.step4 || {}),
      decision: { ...base.step4.decision, ...(overrides.step4?.decision || {}) },
      transition: { ...base.step4.transition, ...(overrides.step4?.transition || {}) },
      treatment_episodes: overrides.step4?.treatment_episodes || base.step4.treatment_episodes,
      administrations: overrides.step4?.administrations || base.step4.administrations
    }
  };
}

function evaluate(current, historical = []) {
  const projection = g1.buildLongitudinalProjection(historical, { currentInternalUuid: current.internal_uuid || "" });
  const baseContext = g1.buildEncounterContext(current, projection);
  const evidenceContext = g2.buildEvidenceContext(current, projection, baseContext, { historicalEncounters: historical });
  const contributions = g2.evaluateEvidenceGuidance(evidenceContext);
  const basePlan = g1.buildVisitPlan(baseContext);
  const mergedPlan = g2.mergeEvidenceContributions(basePlan, contributions);
  return { projection, baseContext, evidenceContext, contributions, mergedPlan, ids: contributions.map(item => item.rule_id) };
}

function contribution(result, id) {
  return result.contributions.find(item => item.rule_id === id);
}

// 1. Initial product flow must not imply R01 without explicit indication.
{
  const current = baseCase({ encounter_archetype: "initial_assessment_new_or_uncertain_diagnosis" });
  const result = evaluate(current);
  assert(!result.ids.includes("OST_G2_R01_INITIAL_FORMAL_RISK"));
  assert(result.mergedPlan.ordered_cards.some(item => item.card_id === "formal_risk"), "G1 product flow should still surface Formal Risk");
}

// 2. NOGG scope + explicit formal-risk indication activates R01.
{
  const current = baseCase({ risk_assessment: { formal_indicated: "yes", declared_framework: "nogg_2024" } });
  assert(evaluate(current).ids.includes("OST_G2_R01_INITIAL_FORMAL_RISK"));
}

// 3. Explicitly non-NOGG framework suppresses NOGG-specific R01/R08 threshold labeling.
{
  const current = baseCase({
    risk_assessment: { formal_indicated: "yes", declared_framework: "other_framework", resulting_risk_category: "very_high" },
    step3: { dxa: { used: "yes", spine_t: -4.0 } }
  });
  const result = evaluate(current);
  assert(!result.ids.includes("OST_G2_R01_INITIAL_FORMAL_RISK"));
  assert(!result.ids.includes("OST_G2_R08_EXPLICIT_VERY_HIGH_RISK_REVIEW"));
}

// 4. Height loss >=4 cm activates VFA guidance.
{
  const current = baseCase({ anthropometrics: { derived_height_loss_cm: 4.2 } });
  assert(evaluate(current).ids.includes("OST_G2_R02_VFA_STRUCTURED_TRIGGER"));
}

// 5. New low-trauma fracture triggers prompt reassessment and treatment-plan review.
{
  const current = baseCase({
    encounter_archetype: "post_fragility_fracture",
    fracture_history: {
      interval_fracture_status: "yes",
      events: [{ id: "f1", site: "hip", month: "2026-08", low_trauma: "yes", occurred_on_treatment: "no" }]
    }
  });
  const result = evaluate(current);
  assert(result.ids.includes("OST_G2_R05_NEW_FRAGILITY_FRACTURE_PROMPT_REASSESSMENT"));
  assert(result.ids.includes("OST_G2_R06_NEW_FRAGILITY_FRACTURE_TREATMENT_PLAN"));
}

// 6. Fracture on treatment triggers review but never an automatic failure/switch command.
{
  const current = baseCase({
    encounter_archetype: "fracture_on_treatment",
    fracture_history: { interval_fracture_status: "yes", events: [{ id: "f2", site: "vertebral", month: "2026-08", low_trauma: "yes", occurred_on_treatment: "yes" }] }
  });
  const result = evaluate(current);
  const r07 = contribution(result, "OST_G2_R07_FRACTURE_ON_TREATMENT_REVIEW");
  assert(r07);
  assert.strictEqual(r07.forbidden_output, "automatic_treatment_failure_or_switch");
}

// 7. Denosumab six-month due derives from reliable exact actual date and remains ephemeral.
{
  const current = baseCase({
    encounter_archetype: "treatment_continuation_or_due_monitoring",
    step4: { treatment_episodes: [{ id: "ep", agent: "denosumab", status: "active", start_date: "2025-02-28" }] }
  });
  const historical = [encounter("e1", "2026-02-28", { step4: { administrations: [{ id: "d1", agent: "denosumab", actual_date: "2026-02-28", status: "done" }] } })];
  const result = evaluate(current, historical);
  const r12 = contribution(result, "OST_G2_R12_DENOSUMAB_EVIDENCE_DUE");
  assert(r12);
  assert.strictEqual(r12.evidence_expected_due_date, "2026-08-28");
  assert.strictEqual(r12.persistence, "ephemeral_only");
}

// 8. Scheduled-only denosumab does not count and cannot create R12/R24.
{
  const current = baseCase({
    encounter_archetype: "treatment_continuation_or_due_monitoring",
    step4: {
      treatment_episodes: [{ id: "ep", agent: "denosumab", status: "active", start_date: "2026-02-28" }],
      administrations: [{ id: "scheduled", agent: "denosumab", scheduled_date: "2026-08-28", status: "planned" }]
    }
  });
  const result = evaluate(current);
  assert.strictEqual(result.evidenceContext.administration_timeline.count_by_agent.denosumab, undefined);
  assert(!result.ids.includes("OST_G2_R12_DENOSUMAB_EVIDENCE_DUE"));
  assert(!result.ids.includes("OST_G2_R24_DENOSUMAB_GT7M_REBOUND_ESCALATION"));
}

// 9. >7 months after one reliable dose does not activate the specific >=2-dose R24 warning.
{
  const current = baseCase({
    encounter_date: "2026-08-02",
    step4: { treatment_episodes: [{ id: "ep", agent: "denosumab", status: "active", start_date: "2026-01-01" }] }
  });
  const historical = [encounter("e1", "2026-01-01", { step4: { administrations: [{ id: "d1", agent: "denosumab", actual_date: "2026-01-01" }] } })];
  const result = evaluate(current, historical);
  assert.strictEqual(result.evidenceContext.administration_timeline.count_by_agent.denosumab, 1);
  assert(!result.ids.includes("OST_G2_R24_DENOSUMAB_GT7M_REBOUND_ESCALATION"));
}

// 10. >7 months after >=2 reliable actual doses activates R24.
{
  const current = baseCase({
    encounter_date: "2026-08-02",
    step4: { treatment_episodes: [{ id: "ep", agent: "denosumab", status: "active", start_date: "2025-07-01" }] }
  });
  const historical = [
    encounter("e1", "2025-07-01", { step4: { administrations: [{ id: "d1", agent: "denosumab", actual_date: "2025-07-01" }] } }),
    encounter("e2", "2026-01-01", { step4: { administrations: [{ id: "d2", agent: "denosumab", actual_date: "2026-01-01" }] } })
  ];
  const result = evaluate(current, historical);
  assert.strictEqual(result.evidenceContext.administration_timeline.count_by_agent.denosumab, 2);
  assert(result.ids.includes("OST_G2_R24_DENOSUMAB_GT7M_REBOUND_ESCALATION"));
}

// 11. Conflicting denosumab history suppresses exact timing milestones.
{
  const current = baseCase({
    encounter_date: "2026-08-31",
    step4: { treatment_episodes: [{ id: "ep", agent: "denosumab", status: "active", start_date: "2025-01-01" }] }
  });
  const historical = [
    encounter("e1", "2026-01-01", { step4: { administrations: [{ id: "same", agent: "denosumab", actual_date: "2026-01-01" }] } }),
    encounter("e2", "2026-02-01", { step4: { administrations: [{ id: "same", agent: "denosumab", actual_date: "2026-02-01" }] } })
  ];
  const result = evaluate(current, historical);
  assert.strictEqual(result.evidenceContext.administration_timeline.reliability_by_agent.denosumab, "conflicting");
  assert(!result.ids.includes("OST_G2_R12_DENOSUMAB_EVIDENCE_DUE"));
  assert(!result.ids.includes("OST_G2_R24_DENOSUMAB_GT7M_REBOUND_ESCALATION"));
}

// 12. Denosumab exit guidance is pure and never writes selected-agent state.
{
  const current = baseCase({
    encounter_archetype: "treatment_change_or_transition",
    step4: {
      treatment_episodes: [{ id: "ep", agent: "denosumab", status: "active", start_date: "2024-01-01" }],
      decision: { type: "stop", selected_agent: "" },
      transition: { relevant: "yes", type: "denosumab_exit", next_agent: "" }
    }
  });
  const historical = [encounter("e1", "2026-03-01", { step4: { administrations: [{ id: "d1", agent: "denosumab", actual_date: "2026-03-01" }] } })];
  const before = JSON.stringify(current);
  const result = evaluate(current, historical);
  const r14 = contribution(result, "OST_G2_R14_DENOSUMAB_EXIT_6M_SEQUENTIAL");
  assert(r14);
  assert.strictEqual(r14.forbidden_output, "automatic_selected_agent");
  assert.strictEqual(JSON.stringify(current), before);
}

// 13. R15/R16 remain blocked and never appear in runtime output.
{
  assert.deepStrictEqual(Array.from(g2.BLOCKED_RULE_IDS), ["OST_G2_R15_DENOSUMAB_EXIT_CTX_FOLLOWUP", "OST_G2_R16_DENOSUMAB_EXIT_NO_CTX_OPTION"]);
  const result = evaluate(baseCase({ encounter_archetype: "treatment_change_or_transition" }));
  g2.BLOCKED_RULE_IDS.forEach(id => assert(!result.ids.includes(id)));
}

// 14. Medication safety rules are checklist-only, never clearance state.
{
  const agents = [
    ["zoledronate", "OST_G2_R17_ZOLEDRONATE_START_SAFETY"],
    ["romosozumab", "OST_G2_R18_ROMOSOZUMAB_START_SAFETY"],
    ["teriparatide", "OST_G2_R19_TERIPARATIDE_START_SAFETY"],
    ["alendronate", "OST_G2_R25_ORAL_BISPHOSPHONATE_START_SAFETY_USE"]
  ];
  agents.forEach(([agent, ruleId]) => {
    const current = baseCase({ encounter_archetype: "treatment_start", step4: { decision: { type: "start", selected_agent: agent } } });
    const rule = contribution(evaluate(current), ruleId);
    assert(rule, `${ruleId} missing`);
    assert.strictEqual(rule.activation_mode, "checklist_only");
    assert.notStrictEqual(rule.clearance, true);
  });
}

// 15. Oral-BP early review requires exact reliable start date; approximate duration is insufficient.
{
  const exact = baseCase({
    encounter_date: "2026-08-31",
    step4: { treatment_episodes: [{ id: "ep", agent: "alendronate", status: "active", start_date: "2026-06-01" }] }
  });
  assert(evaluate(exact).ids.includes("OST_G2_R26_ORAL_BISPHOSPHONATE_EARLY_REVIEW"));

  const approximateOnly = baseCase({
    encounter_date: "2026-08-31",
    step4: { treatment_episodes: [{ id: "ep", agent: "alendronate", status: "active", duration_years: 0.25 }] }
  });
  assert(!evaluate(approximateOnly).ids.includes("OST_G2_R26_ORAL_BISPHOSPHONATE_EARLY_REVIEW"));
}

// 16. Oral-BP >=5y and zoledronate >=3y require exact exposure.
{
  const oral = baseCase({ step4: { treatment_episodes: [{ id: "ep", agent: "alendronate", status: "active", start_date: "2021-08-31" }] } });
  assert(evaluate(oral).ids.includes("OST_G2_R21_ORAL_BISPHOSPHONATE_5Y_REASSESS"));
  const zol = baseCase({ step4: { treatment_episodes: [{ id: "ep", agent: "zoledronate", status: "active", start_date: "2023-08-31" }] } });
  assert(evaluate(zol).ids.includes("OST_G2_R22_ZOLEDRONATE_3Y_REASSESS"));
  const approximate = baseCase({ step4: { treatment_episodes: [{ id: "ep", agent: "zoledronate", status: "active", duration_years: 3.5 }] } });
  assert(!evaluate(approximate).ids.includes("OST_G2_R22_ZOLEDRONATE_3Y_REASSESS"));
}

// 17. Post-romosozumab/teriparatide consolidation requires explicit transition or reliable exact course exposure.
{
  const explicit = baseCase({
    encounter_archetype: "treatment_completion_or_consolidation",
    step4: { transition: { relevant: "yes", type: "post_romosozumab", next_agent: "" } }
  });
  assert(evaluate(explicit).ids.includes("OST_G2_R20_POST_ANABOLIC_CONSOLIDATION"));

  const exactCourse = baseCase({
    encounter_date: "2026-08-31",
    step4: { treatment_episodes: [{ id: "ep", agent: "teriparatide", status: "active", start_date: "2024-08-31" }] }
  });
  assert(evaluate(exactCourse).ids.includes("OST_G2_R20_POST_ANABOLIC_CONSOLIDATION"));
}

// 18. NOGG very-high-risk rule activates only with NOGG framework/scope.
{
  const nogg = baseCase({ risk_assessment: { declared_framework: "nogg_2024", resulting_risk_category: "very_high" } });
  assert(evaluate(nogg).ids.includes("OST_G2_R08_EXPLICIT_VERY_HIGH_RISK_REVIEW"));
  const nonNogg = baseCase({ risk_assessment: { declared_framework: "other_framework", resulting_risk_category: "very_high" } });
  assert(!evaluate(nonNogg).ids.includes("OST_G2_R08_EXPLICIT_VERY_HIGH_RISK_REVIEW"));
}

// 19. No CTX 280/300 automatic command and no generic 4th/8th/10th Prolia milestone exist in runtime source.
{
  assert(!/CTX\s*[>=]+\s*(280|300)/i.test(g2Source));
  assert(!/(4th|8th|10th)\s*(dose|prolia)/i.test(g2Source));
}

// 20. Same evidence context is deterministic and merge preserves G1 reasons plus G2 provenance.
{
  const current = baseCase({
    encounter_archetype: "post_fragility_fracture",
    fracture_history: { interval_fracture_status: "yes", events: [{ id: "f", site: "hip", month: "2026-08", low_trauma: "yes" }] }
  });
  const first = evaluate(current);
  const second = evaluate(current);
  assert.strictEqual(JSON.stringify(first.contributions), JSON.stringify(second.contributions));
  assert.strictEqual(JSON.stringify(first.mergedPlan), JSON.stringify(second.mergedPlan));
  const formalRisk = first.mergedPlan.ordered_cards.find(item => item.card_id === "formal_risk");
  assert(formalRisk.reason_codes.includes("NEW_EVENT"));
  assert(formalRisk.reason_codes.some(code => code.startsWith("G2:")));
  assert(formalRisk.evidence_rules.length > 0);
}

console.log("G2 evidence guidance core regressions: OK");
