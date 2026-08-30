# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-0 DYNAMIC GUIDED VISIT DESIGN-COMPLETE / PRE-RUNTIME STOP.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main` at G-0 bootstrap:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Completed design branch:** `design/module01-dynamic-guided-visit-replan-2026-08-30`.
> **Parent tested runtime ancestry:** `fix/module01-c1-authoritative-finish-2026-08-30` @ `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **Current major phase:** dynamic guided osteoporosis consultation + Heidi-first capture before real pilot.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE — G-0 design is complete.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **Runtime mutation:** NOT AUTHORIZED by G-0 completion.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product truth now frozen

The clinician-facing Module 01 product exists primarily to:

```text
improve the current osteoporosis visit
+
reduce duplicate/manual capture
+
review whether what was said/reasoned/decided was appropriate
+
improve future clinician performance longitudinally
```

The Baseline Audit is an underlying measurement/storage foundation, not the intended clinician-facing workflow.

The current largely manual Steps 1–6 flow is already known to impose unacceptable burden for routine intended use and will not be the workflow tested in the five-case real pilot.

---

# 2. G-0 methodology replan complete

Approved sequence:

```text
C1 authoritative Finish merge/deploy/smoke
→ G-1 dynamic Clinical Guidance mechanics
→ evidence-backed osteoporosis guidance profiles
→ PR-1 Heidi transcript extraction
→ PR-2 inline provisional population / clinician review
→ guided clinical-card UX sufficient for real use
→ 5 real system-assisted pilot encounters
→ one deliberate refinement + freeze
→ Quick Practice Review shadow capability
→ 30-case scored system-assisted baseline
→ baseline lock
→ reviewed Signals/intervention
→ re-measure
→ final Module 01 closure review
```

During the 30-case baseline:

- stable Clinical Guidance stays active;
- transcript-assisted capture stays active;
- routine KPI/performance feedback stays hidden;
- routine clinician-facing Practice Review stays hidden by default;
- safety-critical feedback is allowed;
- guidance exposure is recorded where technically reliable;
- cohort is labelled **system-assisted baseline**.

---

# 3. Frozen architecture

System functions:

```text
Clinical Guidance
!= Transcript-assisted Capture
!= Audit / Measurement
!= Clinical Practice Review
```

Normative machine entrypoint:

```text
schemas/dynamic_guided_visit_contract_manifest_v1.yaml
```

Normative contracts:

```text
schemas/dynamic_guided_visit_v1.yaml
schemas/longitudinal_guidance_projection_v1.yaml
```

Frozen objects:

```text
EncounterContextV1
LongitudinalGuidanceProjectionV1
ProjectionConflictV1
GuidanceRuleV1
VisitPlanV1
GuidedCardStateV1
TherapyMilestoneProfileV1
GuidanceExposureV1
```

Exact design review:

```text
M01_G0_DYNAMIC_GUIDANCE_DESIGN_REVIEW_V1.md
classification: DESIGN-COMPLETE
```

---

# 4. Dynamic visit rule

Coarse `encounter_archetype` remains **visit intent**, not the whole Visit Plan.

The Visit Plan is derived from:

```text
visit intent
+ prior authoritative longitudinal state
+ active treatment/agent
+ actual administrations
+ elapsed exposure
+ reliable administration count
+ due/overdue state
+ monitoring due state
+ new fracture/adverse event/safety trigger
+ unresolved prior tasks/prerequisites
+ patient-specific context
+ transcript uncertainty/conflict later
```

Frozen priority:

```text
critical safety / urgent event
→ unresolved prior critical item
→ treatment/agent-specific requirement
→ evidence-defined milestone/due item
→ archetype base flow
→ contextual item
```

Every non-obvious surfaced card should be able to answer `WHY NOW?`.

---

# 5. Repeated-treatment / Prolia design

Do not create one hard-coded form per ordinal dose.

Use treatment timeline + milestone/event rules.

Count and elapsed exposure remain separate. Scheduled/planned events do not count as actual administrations. Delays may make count and elapsed duration diverge and that divergence must remain visible.

No exact 4th/8th/10th Prolia clinical rule is active from G-0. Exact milestone content requires reviewed evidence or approved clinic-policy provenance.

---

# 6. Longitudinal storage seam — verified / no G-1 migration required

There is no separate patient-level treatment timeline table today.

However the protected endpoint:

```text
GET /clinical/patient/{patient_id}/encounters
```

returns all historical encounters with full `payload`.

G-0 therefore froze a read-only `LongitudinalGuidanceProjectionV1` that:

- derives context from completed/amended historical encounters;
- does not let a later blank snapshot erase prior history;
- preserves material conflicts;
- counts only reliable unique actual administration events;
- never reconstructs missing doses from expected cadence;
- resurfaces unresolved prior tasks where deterministically supported;
- does not create a second source of truth.

Preferred G-1 boundary: ephemeral projection, no DB migration by default.

---

# 7. Heidi / PR-2 position

Archived corrected PR-1 v3 semantic/privacy design remains the extraction starting point.

New product UX:

```text
raw Heidi transcript
→ ephemeral semantic extraction
→ deterministic Module 01 target mapping
→ provisional values in destination cards
→ clinician Accept / Edit / Reject
→ authoritative write only after review
```

No silent overwrite. `Not mentioned` does not mean negative.

---

# 8. C1 authoritative Finish state preserved

```text
branch: fix/module01-c1-authoritative-finish-2026-08-30
head:   a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871
state:  IMPLEMENTED / TESTED
CI:     exact-head run 33323204227 SUCCESS
MERGED: NO
DEPLOYED: NO
PRODUCTION-SMOKE: NO
```

G-0 inherits this code in ancestry but did not merge/deploy it.

---

# 9. Physiotherapy remains parked/preserved

```text
feat/cu1-rich-referral-global-evidence-2026-08-29
@ bdd23b83a8252405f5aa5a0c0b67f303ccfcef5f
IMPLEMENTED / TESTED / PRODUCT-OWNER REVIEWED
MERGED NO / DEPLOYED NO
```

Do not mutate/merge/deploy during Module 01 work without separate authorization.

---

# 10. Status matrix

```text
G-0 PRODUCT/METHODOLOGY REPLAN               COMPLETE
G-0 MACHINE CONTRACT                         COMPLETE
G-0 LONGITUDINAL PROJECTION CONTRACT         COMPLETE
G-0 EXACT DESIGN REVIEW                      PASS
G-0 DESIGN-COMPLETE                          YES
G-0 RUNTIME MUTATION                         NO
ACTIVE WRITER                                NONE
C1 AUTHORITATIVE FINISH IMPLEMENTED/TESTED   YES
C1 MERGED                                    NO
C1 DEPLOYED                                  NO
G-1 IMPLEMENTED                              NO
PR-1 IMPLEMENTED                             NO
PR-2 IMPLEMENTED                             NO
5-CASE SYSTEM-ASSISTED PILOT                 NOT STARTED
30-CASE SYSTEM-ASSISTED BASELINE             NOT STARTED
MODULE 01 CLOSED                             NO
```

---

# 11. Exact next action

STOP G-0.

If the product owner authorizes implementation, a new session must fresh-bootstrap and open a bounded **G-1 runtime slice** limited to:

```text
1. read-only LongitudinalGuidanceProjectionV1;
2. EncounterContextV1 resolver;
3. generic deterministic GuidanceRuleV1 evaluator / priority resolution;
4. VisitPlanV1 + GuidedCardStateV1;
5. `why now` rendering;
6. synthetic tests for event override, unresolved-prior and treatment/due plumbing;
7. no invented agent-specific clinical milestone content;
8. no PR-1/provider/PR-2 work unless explicitly included in a later approved slice.
```

C1 merge/deploy remains a separate release decision and must be distinguished from G-1 runtime authorization.
