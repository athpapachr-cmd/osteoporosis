# SLICE_PLAN_CURRENT.md — G-2 Evidence-backed Guidance Runtime v1

> **STATUS:** IMPLEMENTED / TESTED — RELEASE REVIEW REQUIRED BEFORE PR/MERGE/DEPLOY.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** `M01-G2-EVIDENCE-GUIDANCE-RUNTIME-v1`.
> **Fresh base main:** `5182d250e244b2ed9e086138cb3b2edcdb967e25`.
> **Design-complete ancestry:** `0395e52ed75f835d49713504df3df4ce51183edf`.
> **Implementation branch:** `feat/module01-g2-evidence-guidance-runtime-2026-08-31`.
> **Exact tested runtime head:** `e0657ba5924db87b38a0e05514613fbadf45bcd9`.
> **Runtime test workflow:** `G2 evidence guidance runtime` run `33403182604` — SUCCESS.
> **Runtime writer:** NONE after implementation/test closeout.

---

# 1. Objective / result

The reviewed G-2 evidence content is now implemented over the production-proven G-1 guidance mechanics while preserving one render owner and the existing finalization/state-integrity boundaries.

Runtime composition:

```text
G-1 LongitudinalGuidanceProjection
+ live current encounter state
→ OsteoporosisEvidenceContextV1
→ deterministic G-2 evidence-rule evaluation
→ evidence contributions per existing domain/card
→ deterministic merge with G-1 Visit Plan
→ existing Σημερινή ροή / Γιατί τώρα presentation
```

No automatic treatment decision or authoritative treatment write was introduced.

---

# 2. Normative design authority preserved

Implementation conforms to:

```text
schemas/osteoporosis_guidance_contract_manifest_v1.yaml
schemas/osteoporosis_guidance_evidence_registry_v1.yaml
schemas/osteoporosis_guidance_rules_v1.yaml
schemas/osteoporosis_guidance_profiles_v1.yaml
schemas/osteoporosis_therapy_milestones_v1.yaml
M01_G2_EVIDENCE_GUIDANCE_REVIEW_V1.md
```

The frozen design remains the clinical-content authority. Runtime implementation did not reopen evidence thresholds/policy.

---

# 3. Implemented runtime

Primary runtime file:

```text
static/baseline-audit/osteoporosis-evidence-guidance-core.js
```

Export:

```text
window.BaselineOsteoporosisEvidenceGuidance
```

The G-2 core is pure deterministic browser-domain logic and does not own DOM rendering, local/session storage, network fetch or Finish listeners.

Implemented capabilities include:

- G-2 evidence-context projection from current structured state + G-1 longitudinal projection;
- exact calendar-month/year calculations;
- current/historical actual-administration timeline with fail-closed conflict handling;
- NOGG scope/framework guards;
- evidence contributions carrying rule ID, rule class, priority, domains, WHY NOW, objective, provenance, strength and activation mode;
- deterministic merge into existing G-1 card states and rule trace;
- evidence-derived timing values kept ephemeral.

---

# 4. G-1 integration / single render owner

Bootstrap order is now:

```text
adaptive applicability
→ progressive-guidance-core.js
→ osteoporosis-evidence-guidance-core.js
→ finalization-coordinator.js
→ patient-registry.js
→ progressive-guidance-ui.js
→ pilot-completion.js
```

`progressive-guidance-ui.js` remains the only guidance renderer/order owner.

Its compute path is:

```text
G1 projection
→ G1 encounter context
→ G1 base plan
→ G2 evidence context
→ G2 deterministic contributions
→ merged Visit Plan
→ existing card + Σημερινή ροή rendering
```

C1 Finish ownership remains unchanged.

---

# 5. Live-state invariant implemented

For G-2 trigger seams the current in-memory snapshot now reads live controls/roots before persisted browser cache.

Covered current-state seams include:

- encounter archetype/date/age;
- sex / menopause / patient relationship / osteoporosis status segmented choices;
- current/reference height with live derived height loss;
- glucocorticoid flag/dose/duration;
- falls count;
- formal-risk indication/framework/resulting category;
- live fracture-event rows;
- Step-3 DXA use/T-scores and prior-workup adequacy;
- live Step-4 treatment episodes;
- live Step-4 administrations;
- decision type/selected agent;
- transition type/next agent/timing fields.

Hard behavior:

```text
LIVE CURRENT CONTROL/ROOT, INCLUDING BLANK OR EMPTY LIST
>
PERSISTED BROWSER CACHE
```

The protected longitudinal projection remains a distinct authoritative prior-data source and is not silently rewritten by current local UI state.

---

# 6. Timeline semantics implemented/tested

- actual administration requires exact `actual_date`;
- scheduled/planned-only events never count as actual doses;
- duplicate actual events use existing G-1 identity semantics;
- conflicting administration history suppresses exact G-2 denosumab timing derivation;
- denosumab evidence due = 6 calendar months from reliable last actual dose, ephemeral only;
- specific denosumab >7-month rebound escalation requires reliable actual count ≥2;
- treatment-exposure milestones require exact reliable episode start date;
- approximate duration does not create exact milestones;
- oral-bisphosphonate early review = 12–16 weeks from exact start;
- oral-BP ≥5y and zoledronate ≥3y use calendar anniversaries;
- administration count and elapsed exposure remain distinct.

---

# 7. Framework / safety semantics implemented/tested

NOGG-specific rules are guarded against silent application under an explicitly non-NOGG framework.

Medication-specific safety content classified `checklist_only` is visibly presented as:

```text
Safety checklist — requires clinical confirmation, not automatic clearance.
```

The UI also surfaces concise evidence provenance such as NOGG/EMA source labels.

The runtime does not infer:

```text
safe
cleared
no contraindication
ready for administration
```

from missing/stale data.

---

# 8. Blocked / forbidden semantics preserved

Not activated:

```text
OST_G2_R15_DENOSUMAB_EXIT_CTX_FOLLOWUP
OST_G2_R16_DENOSUMAB_EXIT_NO_CTX_OPTION
```

Still explicitly absent:

```text
CTX >=280 ng/L → automatic second zoledronate
CTX >=300 ng/L → automatic second zoledronate
CTX at 3 months → mandatory retreatment command
Prolia 4th/8th/10th dose → generic milestone
fracture on treatment → automatic failure/switch
romosozumab → automatic cardiology/vascular referral without approved clinic policy
automatic selected-agent mutation
```

---

# 9. Test evidence

New focused tests:

```text
test_g2_evidence_guidance_node.js
test_g2_guidance_live_state.js
test_g2_evidence_guidance_wiring.js
```

Runtime workflow:

```text
G2 evidence guidance runtime
run: 33403182604
head: e0657ba5924db87b38a0e05514613fbadf45bcd9
status: COMPLETED
conclusion: SUCCESS
```

The single `g2-runtime` job passed all steps, including:

- JavaScript syntax checks;
- frozen G-2 contract validation;
- G-2 evidence-core regressions;
- G-2 live-state regressions;
- G-2 wiring/ownership regressions;
- inherited G-1 core regression;
- inherited G-1 wiring regression;
- inherited G-1 UI-state regressions;
- inherited G-1 WHY-NOW presentation regression;
- authoritative Finish browser regression;
- server finalization lifecycle regression.

---

# 10. Exact-head review

At tested head `e0657ba5…`:

```text
main 5182d250… → tested head
status: ahead
merge base: exactly main
behind: 0
```

From design-complete head `0395e52e…` to tested runtime head there are 10 commits and only expected G-2 runtime/test/canonical files.

No physiotherapy/RF, PR-1 or PR-2 leakage was found.

Source review confirmed:

- R15/R16 have no evaluator activation branch;
- G-2 core has no DOM/storage/fetch/listener ownership;
- single G-1 render owner is preserved;
- C1 Finish owner/load order is preserved;
- checklist≠clearance is visible in UI;
- no prohibited automatic treatment semantics were introduced.

---

# 11. Completion matrix

```text
G-2 evidence/content design                DESIGN-COMPLETE / REVIEWED
G-2 runtime implementation                 YES
G-2 focused runtime tests                  PASS
G-2 frozen contract validation             PASS
Inherited G-1 regressions                  PASS
Inherited C1 regressions                   PASS
Exact-head source/delta review             PASS
Product-owner release review               NO
PR opened                                  NO
Merged                                     NO
Deployed                                   NO
Production-smoke-verified                  NO
Pilot-validated                            NO
```

`IMPLEMENTED / TESTED` does not imply release readiness has been authorized by the product owner.

---

# 12. Next action / stop gate

This bounded implementation slice is closed at its authorized stop gate.

Next possible action requires a **separate fresh release-readiness bootstrap/review and explicit product-owner release authority** before any release PR/merge/deploy action.

Until then:

```text
NO ACTIVE RUNTIME WRITER
NO RELEASE PR
NO MERGE
NO DEPLOY
NO PRODUCTION-SMOKE CLAIM
```

PR-1/PR-2 and parked physiotherapy/RF work remain outside this completed slice.
