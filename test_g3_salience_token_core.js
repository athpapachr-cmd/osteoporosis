"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const context = { window: {} };
vm.createContext(context);
vm.runInContext(fs.readFileSync("static/baseline-audit/g3-salience-token-core.js", "utf8"), context);
const core = context.window.G3SalienceTokenCore;
assert(core, "G3 salience token core must load");

const baseVfa = {
  card_id: "vfa",
  reason_codes: ["VISIT_TYPE_CORE"],
  evidence_rules: []
};

let state = core.advance({
  previousTokens: null,
  retainedNewTokens: [],
  items: [baseVfa],
  initialize: true
});
assert.deepStrictEqual(Array.from(state.newly_surfaced_domains), [], "initial/base-flow VFA must not be marked new");

const triggeredVfa = {
  card_id: "vfa",
  reason_codes: ["VISIT_TYPE_CORE"],
  evidence_rules: [{ rule_id: "OST_G2_R02_VFA_STRUCTURED_TRIGGER" }]
};

state = core.advance({
  previousTokens: Array.from(state.current_tokens),
  retainedNewTokens: Array.from(state.retained_new_tokens),
  items: [triggeredVfa],
  initialize: false
});
assert.deepStrictEqual(Array.from(state.newly_surfaced_domains), ["vfa"], "new R02 evidence on an already-visible VFA card must mark VFA as new");
assert(state.retained_new_tokens.includes("E|vfa|OST_G2_R02_VFA_STRUCTURED_TRIGGER"), "R02 salience token must be retained while applicable");

const stableState = core.advance({
  previousTokens: Array.from(state.current_tokens),
  retainedNewTokens: Array.from(state.retained_new_tokens),
  items: [triggeredVfa],
  initialize: false
});
assert.deepStrictEqual(Array.from(stableState.newly_surfaced_domains), ["vfa"], "new marker may remain while the newly surfaced trigger is still applicable");

const removedState = core.advance({
  previousTokens: Array.from(stableState.current_tokens),
  retainedNewTokens: Array.from(stableState.retained_new_tokens),
  items: [baseVfa],
  initialize: false
});
assert.deepStrictEqual(Array.from(removedState.newly_surfaced_domains), [], "new marker must clear when the material trigger no longer applies");

console.log("G3 salience material-trigger token regressions: PASS");
