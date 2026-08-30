# SLICE_PLAN_CURRENT.md — Progressive Guidance Foundations v1

> **STATUS:** DESIGN REFINEMENT COMPLETE / PRE-RUNTIME STOP.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-G0.1-PROGRESSIVE-GUIDANCE-v1.
> **Verified remote main:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent design:** `design/module01-dynamic-guided-visit-replan-2026-08-30` @ `6aadc8ef55719be98233afa6a80a179f43512c1d`.
> **Current design branch:** `design/module01-progressive-guidance-foundations-2026-08-30`.
> **Runtime writer:** NONE.
> **Runtime mutation:** NOT AUTHORIZED by this refinement.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product-owner clarification

The dynamic consultation architecture is intentionally **progressive**, not a requirement to fully classify every osteoporosis encounter before real use.

The immediate goal is to install extensible foundations and then improve the visit taxonomy/cards from actual encounters and transcripts.

Do not front-load a complete ontology of first/second/4th/8th/10th visits or every possible treatment state.

---

# 2. Minimum first-use encounter classification

The existing first-page encounter dropdown remains the primary clinician-supplied **coarse visit label**.

Useful coarse intents include, conceptually:

```text
first / initial assessment
results or work-up review / management decision
treatment start
treatment administration / continuation
repeat prescription / routine treatment maintenance
treatment change / transition
fracture / fracture-on-treatment
adverse effect / intolerance
other
```

The exact persisted enum does not need to contain every one of these before first pilot use. Existing compatible archetypes may be reused.

The first-page free-text/quick-notes field acts as a fallback **visit summary/context** when the dropdown is too coarse or no option fits.

Rules:

- dropdown label = clinician-declared visit intent, not inferred clinical truth;
- free text may clarify what happened but must not silently become a new authoritative enum;
- transcript extraction may later propose a visit-classification candidate, but clinician confirmation remains required before reclassification;
- `other` plus concise summary is preferable to forcing a wrong category.

---

# 3. Minimal G-1 foundation

The first runtime guidance slice should prove only enough mechanics to support progressive refinement:

```text
coarse encounter intent
+ concise current-visit context
+ read-only prior encounter context where available
+ basic deterministic card applicability / ordering
+ event/safety override plumbing
+ unresolved-prior plumbing
+ treatment/due context plumbing
+ WHY NOW explanation
```

G-1 does **not** require:

- a complete osteoporosis visit taxonomy;
- all medication-specific cards;
- all treatment milestones;
- complete Prolia dose-number logic;
- predictive/AI auto-classification;
- a new patient-level treatment database;
- Practice Review runtime;
- transcript/provider runtime unless separately authorized.

---

# 4. Progressive taxonomy rule

The visit taxonomy and card library should mature from real-use evidence rather than speculative completeness.

Canonical refinement checkpoints:

```text
initial foundation
→ 5 real system-assisted encounters
→ one deliberate usability/classification refinement
→ 30-case system-assisted baseline
→ second evidence-from-use refinement
→ later periodic/milestone reviews when enough new data accumulate
```

At each checkpoint review:

- which dropdown categories were actually used;
- how often `other` was selected;
- recurring free-text visit summaries that suggest a missing category;
- transcript patterns that repeatedly imply the same visit purpose;
- cards repeatedly opened despite being initially hidden;
- cards repeatedly irrelevant despite being surfaced;
- clinically important omissions;
- duplicate questioning/data entry;
- unresolved items that recur across visits;
- treatment/event patterns that justify a new evidence-backed milestone rule.

Do not change taxonomy automatically from frequency alone. Proposed category/card/rule changes remain clinician-reviewed and versioned.

---

# 5. Transcript role in future classification

Heidi transcript data may later help classify encounters more accurately, but first priority remains capture of clinical facts.

Possible later flow:

```text
clinician chooses coarse visit label
+ optional short visit summary
+ transcript extraction identifies what actually occurred
→ system compares these sources
→ proposes classification refinement only when useful
→ clinician confirms/edits
```

Examples of useful signals:

- repeated transcript pattern: review of pending tests followed by treatment decision;
- repeated routine administration visits with very small common information set;
- repeat-prescription visits distinct from administration visits;
- fracture events that override a routine visit label;
- recurring treatment-transition pattern not represented well by current dropdown.

`Transcript pattern != automatic taxonomy mutation`.

---

# 6. Relationship to G-0 design

The G-0 architecture remains valid as **capability**, including:

```text
EncounterContextV1
LongitudinalGuidanceProjectionV1
GuidanceRuleV1
VisitPlanV1
GuidedCardStateV1
TherapyMilestoneProfileV1
GuidanceExposureV1
```

This refinement changes the **implementation ambition/order**, not those extension seams.

The first implementation may use only a small subset. Richer rules are added only after evidence from real use or reviewed clinical guidance justifies them.

---

# 7. Five-case pilot purpose after this refinement

The five cases are now explicitly a test of the intended lightweight system-assisted workflow, including:

- whether the coarse visit label was adequate;
- whether free-text fallback captured missing context;
- whether cards shown were relevant;
- whether important cards/items were missing;
- whether Heidi extraction reduced manual entry;
- whether transcript omissions/errors were easy to correct;
- completion time and cognitive burden;
- persistence/finalization integrity.

The five cases are **not** expected to validate a complete visit taxonomy.

After all five, make one deliberate refinement rather than changing the taxonomy after each case unless safety/data-integrity demands it.

---

# 8. Thirty-case role

The 30-case system-assisted baseline is also a richer product-learning dataset.

Without contaminating KPI/performance feedback methodology, it may be used after baseline lock to review:

- distribution of visit types;
- repeated `other`/free-text clusters;
- transcript-derived recurring encounter patterns;
- stable versus unnecessary cards;
- treatment-specific/milestone patterns;
- guidance exposure and unresolved-item patterns where reliable.

This can support a second, better-informed taxonomy/card revision.

---

# 9. Exact next action

STOP design mutation after this clarification.

If runtime work is separately authorized after fresh bootstrap, G-1 should implement the **minimum progressive foundation**, not the full G-0 capability surface:

```text
1. use existing encounter dropdown as coarse intent;
2. preserve/use short first-page visit summary/free text as fallback context;
3. derive only the longitudinal context required for first useful card relevance;
4. render basic dynamic card ordering/applicability + WHY NOW;
5. support safety/new-event and unresolved-prior overrides;
6. leave richer taxonomy, therapy milestones and card specialization for evidence-from-use refinement;
7. keep extensibility compatible with the frozen G-0 contracts.
```

C1 authoritative Finish merge/deploy remains a separate release decision. PR-1/PR-2 remain required before the five-case real pilot unless separately replanned.
