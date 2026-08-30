"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync("static/baseline-audit/progressive-guidance-core.js", "utf8");
const sandbox = { window: {} };
vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: "progressive-guidance-core.js" });

const core = sandbox.window.BaselineProgressiveGuidanceCore;
assert(core, "BaselineProgressiveGuidanceCore export missing");

function encounter(id, date, payload, status = "completed", updated = `${date}T12:00:00`) {
  return { encounter_id: id, encounter_date: date, status, payload, updated_at: updated };
}

function card(plan, id) {
  return plan.ordered_cards.find(item => item.card_id === id);
}

// 1. First assessment: broad core flow, but no treatment decision automation.
{
  const projection = core.buildLongitudinalProjection([]);
  const context = core.buildEncounterContext({
    internal_uuid: "current-1",
    encounter_archetype: "initial_assessment_new_or_uncertain_diagnosis",
    patient_relationship_status: "new_to_service",
    encounter_date: "2026-08-30",
    quick_notes: "πρώτη εκτίμηση",
    fracture_history: { interval_fracture_status: "", events: [] }
  }, projection);
  const plan = core.buildVisitPlan(context);
  assert(card(plan, "fracture_history"));
  assert(card(plan, "dxa"));
  assert(card(plan, "secondary_causes"));
  assert(card(plan, "treatment_decision"));
  assert.strictEqual(plan.visit_context_text, "πρώτη εκτίμηση");
}

// 2. Other + free text remains context only; no forced inferred classification.
{
  const projection = core.buildLongitudinalProjection([]);
  const context = core.buildEncounterContext({
    internal_uuid: "current-2",
    encounter_archetype: "other",
    encounter_date: "2026-08-30",
    quick_notes: "επανάληψη συνταγογράφησης",
    fracture_history: { interval_fracture_status: "", events: [] }
  }, projection);
  const plan = core.buildVisitPlan(context);
  assert.strictEqual(plan.encounter_archetype, "other");
  assert.strictEqual(plan.visit_context_text, "επανάληψη συνταγογράφησης");
  assert.strictEqual(plan.ordered_cards.length, 0, "free text must not silently create structured guidance rules");
}

// 3. Repeated representation of the same actual administration deduplicates by exact agent + actual date.
{
  const rows = [
    encounter("e1", "2026-01-01", { step4: { administrations: [{ id: "a1", agent: "denosumab", actual_date: "2026-01-01", status: "done" }] } }),
    encounter("e2", "2026-02-01", { step4: { administrations: [{ id: "different-local-id", agent: "denosumab", actual_date: "2026-01-01", status: "done" }, { id: "planned", agent: "denosumab", scheduled_date: "2026-07-01", status: "planned" }] } })
  ];
  const projection = core.buildLongitudinalProjection(rows);
  assert.strictEqual(projection.administration_projection.unique_actual_events.length, 1);
  assert.strictEqual(projection.administration_projection.administration_count_by_agent.denosumab, 1);
  assert.strictEqual(projection.administration_projection.unique_actual_events[0].identity_basis, "agent_plus_exact_actual_date");
}

// 4. Scheduled/planned without actual date never increments administered count.
{
  const rows = [encounter("e1", "2026-01-01", { step4: { administrations: [{ id: "a1", agent: "denosumab", scheduled_date: "2026-01-01", status: "done" }] } })];
  const projection = core.buildLongitudinalProjection(rows);
  assert.strictEqual(projection.administration_projection.unique_actual_events.length, 0);
  assert.strictEqual(projection.administration_projection.administration_count_by_agent.denosumab, undefined);
}

// 5. Conflicting next-due facts for the same actual administration remain explicit.
{
  const rows = [
    encounter("e1", "2026-01-01", { step4: { administrations: [{ id: "a1", agent: "denosumab", actual_date: "2026-01-01", next_due_date: "2026-07-01" }] } }),
    encounter("e2", "2026-02-01", { step4: { administrations: [{ id: "a1", agent: "denosumab", actual_date: "2026-01-01", next_due_date: "2026-07-08" }] } })
  ];
  const projection = core.buildLongitudinalProjection(rows);
  assert(projection.conflict_records.some(x => x.summary_code === "CONFLICTING_NEXT_DUE_FOR_SAME_ACTUAL_EVENT"));
  assert.strictEqual(projection.administration_projection.unique_actual_events[0].next_due_date, null);
}

// 6. Planned prior task resurfaces; same semantic task explicitly completed later does not.
{
  const task = { type: "lab", due_date: "2026-08-01", timeframe_text: "", status: "planned" };
  const rows = [encounter("e1", "2026-07-01", { step4: { tasks: [task], close: { unresolved_critical: "no" } } })];
  let projection = core.buildLongitudinalProjection(rows);
  assert.strictEqual(projection.unresolved_task_projection.length, 1);

  rows.push(encounter("e2", "2026-08-10", { step4: { tasks: [{ ...task, status: "already_done" }], close: { unresolved_critical: "no" } } }));
  projection = core.buildLongitudinalProjection(rows);
  assert.strictEqual(projection.unresolved_task_projection.length, 0);
}

// 7. Explicit due dates are respected; no due state is invented from an actual date alone.
{
  const withoutDue = core.buildLongitudinalProjection([
    encounter("e1", "2026-01-01", { step4: { administrations: [{ agent: "denosumab", actual_date: "2026-01-01", status: "done" }] } })
  ]);
  let context = core.buildEncounterContext({
    encounter_archetype: "treatment_continuation_or_due_monitoring",
    encounter_date: "2026-08-30",
    fracture_history: { events: [] }
  }, withoutDue);
  assert.deepStrictEqual(Array.from(context.explicit_due_agents), []);
  assert.deepStrictEqual(Array.from(context.explicit_overdue_agents), []);

  const withDue = core.buildLongitudinalProjection([
    encounter("e1", "2026-01-01", { step4: { administrations: [{ agent: "denosumab", actual_date: "2026-01-01", next_due_date: "2026-08-30", status: "done" }] } })
  ]);
  context = core.buildEncounterContext({
    encounter_archetype: "treatment_continuation_or_due_monitoring",
    encounter_date: "2026-08-30",
    fracture_history: { events: [] }
  }, withDue);
  assert.deepStrictEqual(Array.from(context.explicit_due_agents), ["denosumab"]);
}

// 8. New fracture overrides the concise routine flow.
{
  const projection = core.buildLongitudinalProjection([
    encounter("e1", "2026-01-01", { step4: { treatment_episodes: [{ id: "t1", agent: "denosumab", status: "active" }] } })
  ]);
  const context = core.buildEncounterContext({
    encounter_archetype: "treatment_continuation_or_due_monitoring",
    encounter_date: "2026-08-30",
    fracture_history: {
      interval_fracture_status: "yes",
      events: [{ occurred_on_treatment: "yes" }]
    }
  }, projection);
  const plan = core.buildVisitPlan(context);
  assert.strictEqual(card(plan, "fracture_history").priority, core.REASON_PRIORITY.NEW_EVENT);
  assert.strictEqual(card(plan, "transition_safety").priority, core.REASON_PRIORITY.NEW_EVENT);
  assert(card(plan, "formal_risk").reason_codes.includes("NEW_EVENT"));
}

// 9. Draft/current server rows do not contaminate historical projection.
{
  const rows = [
    encounter("historical", "2026-01-01", { internal_uuid: "old", step4: { administrations: [{ agent: "denosumab", actual_date: "2026-01-01" }] } }),
    encounter("draft", "2026-08-30", { internal_uuid: "current", step4: { administrations: [{ agent: "denosumab", actual_date: "2026-08-30" }] } }, "draft")
  ];
  const projection = core.buildLongitudinalProjection(rows, { currentInternalUuid: "current" });
  assert.strictEqual(projection.prior_encounter_count, 1);
  assert.strictEqual(projection.administration_projection.administration_count_by_agent.denosumab, 1);
}

// 10. Same structured context produces deterministic plan ordering/reasons.
{
  const context = {
    encounter_archetype: "treatment_continuation_or_due_monitoring",
    active_treatment_agents: ["denosumab"],
    explicit_due_agents: [],
    explicit_overdue_agents: [],
    new_events: { fracture: "no", fracture_on_treatment: "no" },
    unresolved_prior_items: [{ task_type: "lab" }],
    prior_unresolved_critical: false,
    projection_conflicts: [],
    visit_context_text: ""
  };
  assert.strictEqual(JSON.stringify(core.buildVisitPlan(context)), JSON.stringify(core.buildVisitPlan(context)));
}

console.log("progressive guidance core regression: OK");
