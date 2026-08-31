# SLICE_PLAN_CURRENT.md — G-2 Evidence-backed Guidance Runtime v1

> **STATUS:** ACTIVE IMPLEMENTATION.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** `M01-G2-EVIDENCE-GUIDANCE-RUNTIME-v1`.
> **Fresh base main:** `5182d250e244b2ed9e086138cb3b2edcdb967e25`.
> **Design-complete ancestry:** `0395e52ed75f835d49713504df3df4ce51183edf`.
> **Implementation branch:** `feat/module01-g2-evidence-guidance-runtime-2026-08-31`.
> **Runtime writer:** THIS SESSION.

---

# 1. Objective

Activate the reviewed G-2 evidence content over the production-proven G-1 guidance mechanics while preserving one render owner and all existing finalization/state-integrity invariants.

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

No treatment decision is made automatically.

---

# 2. Normative design authority

Implementation must conform to:

```text
schemas/osteoporosis_guidance_contract_manifest_v1.yaml
schemas/osteoporosis_guidance_evidence_registry_v1.yaml
schemas/osteoporosis_guidance_rules_v1.yaml
schemas/osteoporosis_guidance_profiles_v1.yaml
schemas/osteoporosis_therapy_milestones_v1.yaml
M01_G2_EVIDENCE_GUIDANCE_REVIEW_V1.md
```

The frozen design is `DESIGN-COMPLETE`; this slice must not reopen evidence content merely for convenience.

---

# 3. In scope

- pure deterministic browser G-2 evidence-guidance core;
- projection of currently available structured fields into an evidence context;
- exact date/timeline derivation only from reliable exact facts;
- evidence contributions carrying rule ID, class, priority, domain, reason, objective, source refs, strength and activation mode;
- deterministic merge into current G-1 ordered cards;
- live-DOM snapshot support for G-2 trigger fields;
- checklist-only medication safety presentation without automated clearance;
- focused synthetic/unit/wiring tests;
- inherited G-1/C1 regression preservation;
- CI and canonical closeout.

---

# 4. Out of scope / forbidden

- R15 post-denosumab-exit CTX rule activation;
- R16 no-CTX fallback activation;
- any automatic CTX 280/300 retreatment rule;
- arbitrary Prolia 4th/8th/10th milestone;
- automatic treatment failure/switch after fracture;
- automatic selected-agent mutation;
- automatic cardiology/vascular referral for romosozumab without approved clinic policy;
- treating missing/stale data as safety clearance;
- parsing free text into authoritative G-2 trigger facts;
- new persistence/schema migration unless a REPLAN trigger is formally accepted;
- PR-1 / PR-2;
- physiotherapy / RF;
- release PR / merge / deploy.

---

# 5. Runtime ownership

## G-1 remains generic owner of

- historical encounter projection;
- base encounter context;
- product-flow/archetype guidance;
- unresolved prior-item resurfacing;
- existing Visit Plan/card state mechanics.

## G-2 owns

- osteoporosis-specific evidence context;
- reviewed evidence rules and therapy milestones;
- evidence provenance/activation mode;
- evidence-derived ephemeral timing states.

## UI ownership

`progressive-guidance-ui.js` remains the single guidance render owner. G-2 must not add a second independent renderer/listener competing for card ordering or `Σημερινή ροή`.

---

# 6. Live-state invariant

For every G-2 field with a live control/root:

```text
current live value, including explicit blank
>
persisted browser/cache fallback
```

Fallback to persisted state is allowed only when the corresponding live control/root is absent.

This must be regression-tested for at least formal-risk/framework and treatment/agent fields, plus any repeated Step-4 timeline structures used for current-visit evaluation.

---

# 7. Timeline semantics

- actual administration requires exact `actual_date`;
- scheduled/planned-only events never count as actual doses;
- duplicate representations of the same actual event are deduplicated under existing G-1 semantics;
- conflicting administration history suppresses exact G-2 milestone derivation;
- denosumab evidence due = 6 calendar months from reliable last actual dose, ephemeral only;
- specific denosumab >7-month rebound escalation requires reliable actual administration count ≥2;
- treatment exposure milestones require an exact reliable episode start date; approximate duration is insufficient;
- oral-bisphosphonate early review = 12–16 weeks from exact reliable start;
- oral-BP ≥5y and zoledronate ≥3y use calendar-anniversary logic, not fixed day approximations;
- administration count and elapsed exposure remain separate.

---

# 8. Framework/scope semantics

NOGG-specific threshold rules must not silently apply under another declared framework.

For first runtime:

- R01 requires NOGG scope and explicit formal-risk indication; if an explicitly non-NOGG framework is declared, product flow remains but NOGG evidence labeling is suppressed;
- R08 exact NOGG very-high-risk threshold semantics require NOGG framework/scope;
- generic new-fracture/event guidance remains independent of those threshold labels.

---

# 9. Safety checklist semantics

Rules classified `checklist_only` may surface statements such as:

```text
verify
review
confirm
address before treatment
```

They must never output or imply:

```text
safe
cleared
no contraindication
ready for administration
```

unless a later separately reviewed contract explicitly proves all required current inputs/freshness.

---

# 10. Minimum test matrix

1. Initial visit product flow does not assert R01 merely because visit is initial.
2. NOGG scope + explicit formal-risk indication activates R01.
3. Non-NOGG declared framework suppresses NOGG threshold labeling.
4. ≥4 cm height loss activates VFA guidance.
5. New fragility fracture surfaces event reassessment/treatment-plan guidance.
6. Fracture on treatment triggers reassessment but never automatic failure/switch.
7. Denosumab 6-month expected due derives from reliable exact actual date.
8. Scheduled-only denosumab administration does not count or create an exact due milestone.
9. >7 months after one reliable denosumab dose does not activate R24.
10. >7 months after ≥2 reliable denosumab doses activates R24.
11. Conflicting denosumab history suppresses exact milestone derivation.
12. Denosumab exit guidance does not write selected agent.
13. R15 and R16 remain inactive.
14. Zoledronate/romosozumab/teriparatide/oral-BP safety outputs remain checklist-only.
15. Oral-BP 12–16-week review requires exact reliable start date.
16. Oral-BP ≥5y and zoledronate ≥3y require reliable exact exposure.
17. Post-romosozumab/teriparatide consolidation only from explicit/reliable completion context.
18. No CTX 280/300 automatic rule exists.
19. No generic Prolia 4th/8th/10th milestone exists.
20. Live blank/changed controls outrank persisted cache.
21. Same evidence context produces deterministic same contributions/merged plan.
22. Existing G-1 tests remain green.
23. Existing authoritative Finish browser/server regressions remain green.

All fixtures must be synthetic and non-identifiable.

---

# 11. Acceptance gate

The slice may reach:

```text
IMPLEMENTED / TESTED
```

only when:

- exact implementation head passes G-2 focused tests;
- design contract validation passes;
- inherited G-1/C1 regression suite passes;
- no runtime rule violates activation/blocking classification;
- no unexpected unrelated file enters the branch delta;
- canonicals record exact evidence.

Then STOP.

No PR/merge/deploy is authorized by this slice.

---

# 12. REPLAN triggers

Stop implementation and replan before further mutation if inspection demonstrates any of:

- a required trigger cannot obey live-over-cache ownership without changing source ownership;
- a reviewed rule requires data that current schema cannot distinguish safely;
- exact treatment-event linkage assumed by an active rule is not reliable;
- integration would require a second guidance render owner;
- a safety checklist would necessarily be presented as automated clearance;
- current runtime target/card IDs materially differ from the reviewed contract;
- a DB/schema migration becomes necessary.
