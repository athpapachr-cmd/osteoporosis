"use strict";

const assert = require("assert");
const fs = require("fs");

const app = fs.readFileSync("static/baseline-audit/app.js", "utf8");
const ui = fs.readFileSync("static/baseline-audit/progressive-guidance-ui.js", "utf8");
const g2 = fs.readFileSync("static/baseline-audit/osteoporosis-evidence-guidance-core.js", "utf8");

function indexOfOrFail(source, needle, label) {
  const index = source.indexOf(needle);
  assert(index >= 0, `${label} missing: ${needle}`);
  return index;
}

// 1. Bootstrap order preserves generic G1 core, then G2 content, then the existing finalization/persistence/render owners.
{
  const g1Core = indexOfOrFail(app, 'loadScript("./progressive-guidance-core.js")', "G1 core load");
  const g2Core = indexOfOrFail(app, 'loadScript("./osteoporosis-evidence-guidance-core.js")', "G2 core load");
  const finalization = indexOfOrFail(app, 'loadScript("./finalization-coordinator.js")', "finalization coordinator load");
  const registry = indexOfOrFail(app, 'loadScript("./patient-registry.js")', "patient registry load");
  const renderer = indexOfOrFail(app, 'loadScript("./progressive-guidance-ui.js")', "progressive guidance UI load");
  const finishOwner = indexOfOrFail(app, 'loadScript("./pilot-completion.js")', "pilot completion load");
  assert(g1Core < g2Core && g2Core < finalization && finalization < registry && registry < renderer && renderer < finishOwner,
    "G2 must be a pure content layer before the existing single renderer and must not disrupt Finish ownership");
}

// 2. G2 core is pure browser-domain logic: no DOM, storage, fetch or listener ownership.
{
  ["document.", "localStorage", "sessionStorage", "fetch(", "addEventListener", "MutationObserver"].forEach(forbidden => {
    assert(!g2.includes(forbidden), `G2 core must not own browser IO/render state: found ${forbidden}`);
  });
  assert(g2.includes("window.BaselineOsteoporosisEvidenceGuidance"), "G2 export missing");
}

// 3. Existing progressive-guidance UI remains the single composition/render owner.
{
  assert(ui.includes("g2.buildEvidenceContext(current, projection, context, { historicalEncounters })"), "UI must build G2 evidence context from G1 projection + live current snapshot");
  assert(ui.includes("g2.evaluateEvidenceGuidance(lastEvidenceContext)"), "UI must evaluate deterministic G2 rules");
  assert(ui.includes("g2.mergeEvidenceContributions(basePlan, lastEvidenceContributions)"), "UI must merge G2 contributions into G1 plan");
  assert(ui.includes("applyPlanToCards(plan)"), "existing UI must remain card render owner");
  assert(ui.includes("renderSummary(context, plan, projection)"), "existing UI must remain summary render owner");
}

// 4. Live-over-cache support covers the reviewed G2 trigger seams including repeated Step-4 rows.
{
  [
    '#formalRiskIndicated', '#declaredRiskFramework', '#resultingRiskCategory',
    '#s3DxaUsed', '#s3SpineT', '#s3TotalHipT', '#s3FnT', '#s3PriorWorkupAdequate',
    '#s4Episodes', '#s4Administrations', '#s4DecisionType', '#s4SelectedAgent', '#s4TransitionType', '#s4NextAgent'
  ].forEach(selector => assert(ui.includes(selector), `live G2 selector missing from snapshot/wiring: ${selector}`));
  assert(ui.includes('repeatRowsFromDom("#s4Episodes"'), "live treatment episodes must own current snapshot");
  assert(ui.includes('repeatRowsFromDom("#s4Administrations"'), "live administrations must own current snapshot");
}

// 5. Evidence provenance and checklist-not-clearance semantics are visible in the only renderer.
{
  assert(ui.includes("Safety checklist — απαιτεί κλινική επιβεβαίωση, όχι automatic clearance."), "checklist disclaimer missing");
  assert(ui.includes("Τεκμηρίωση:"), "evidence provenance label missing");
  assert(ui.includes("rule.activation_mode === \"checklist_only\""), "checklist activation mode not recognized by UI");
}

// 6. Frozen blocked rules are declared, but no runtime evaluator branch activates them.
{
  assert(g2.includes('"OST_G2_R15_DENOSUMAB_EXIT_CTX_FOLLOWUP"'));
  assert(g2.includes('"OST_G2_R16_DENOSUMAB_EXIT_NO_CTX_OPTION"'));
  const evaluatorStart = indexOfOrFail(g2, "function evaluateEvidenceGuidance(context)", "G2 evaluator");
  const evaluatorEnd = indexOfOrFail(g2, "function mergeEvidenceContributions", "G2 merge function");
  const evaluator = g2.slice(evaluatorStart, evaluatorEnd);
  assert(!evaluator.includes("OST_G2_R15_DENOSUMAB_EXIT_CTX_FOLLOWUP"), "R15 must remain inactive");
  assert(!evaluator.includes("OST_G2_R16_DENOSUMAB_EXIT_NO_CTX_OPTION"), "R16 must remain inactive");
}

// 7. No prohibited automatic treatment semantics are introduced into UI wiring.
{
  assert(!ui.includes("automatic_selected_agent"), "UI must not own an automatic selected-agent action");
  assert(!ui.includes("automatic_treatment_failure_or_switch"), "UI must not auto-label treatment failure/switch");
}

console.log("G2 evidence guidance wiring regressions: OK");
