# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-0 DYNAMIC GUIDED VISIT REPLAN ACTIVE / PRE-RUNTIME DESIGN.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Current design branch:** `design/module01-dynamic-guided-visit-replan-2026-08-30`.
> **Parent tested runtime ancestry:** `fix/module01-c1-authoritative-finish-2026-08-30` @ `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **Current major phase:** dynamic guided osteoporosis consultation + Heidi-first capture before real pilot.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/module01-dynamic-guided-visit-replan-2026-08-30`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **Runtime mutation:** NOT AUTHORIZED in G-0.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product-owner decision now governing Module 01

The product owner clarified that the system exists primarily to:

```text
improve the current osteoporosis visit
+
reduce duplicate/manual capture
+
review whether what was said/reasoned/decided was appropriate
+
improve future clinician performance longitudinally
```

The current largely manual post-visit form is already known to impose unacceptable burden for intended routine use. Therefore a five-case real pilot must test the **intended system-assisted product**, not the known manual predecessor.

The product owner also clarified that visit content is intrinsically dynamic across first assessment, results review/decision visits, repeated treatment administrations/milestones, fracture events, treatment transitions and other clinical states.

---

# 2. Methodology REPLAN accepted

The prior order:

```text
5 manual pilot cases
→ freeze form
→ transcript extraction later
→ adaptive visible workflow much later
```

is superseded.

Current intended order:

```text
C1 authoritative Finish merge/deploy/smoke
→ dynamic guided-visit engine
→ PR-1 transcript extraction
→ PR-2 inline provisional population / clinician review
→ guided clinical-card UX
→ 5 real system-assisted pilot cases
→ one deliberate refinement + freeze
→ Quick Practice Review shadow capability
→ 30-case scored system-assisted baseline
→ baseline lock
→ Signals/intervention
→ re-measure
→ final Module 01 closure review
```

During the scored baseline, stable Clinical Guidance remains active; routine KPI/performance feedback and routine clinician-facing Practice Review remain hidden by default. The cohort must be labelled **system-assisted baseline**.

---

# 3. Current G-0 design architecture

The design now separates four functions:

```text
Clinical Guidance
!= Transcript-assisted Capture
!= Audit / Measurement
!= Clinical Practice Review
```

New design objects:

```text
EncounterContextV1
VisitPlanV1
GuidanceRuleV1
GuidedCardStateV1
GuidanceExposureV1
TherapyMilestoneProfileV1
```

The existing runtime coarse encounter archetypes remain useful as visit intent, but they are combined with longitudinal triggers rather than treated as the full applicability model.

---

# 4. Dynamic rule hierarchy

Frozen design priority:

```text
critical safety / urgent event
→ unresolved prior critical item
→ treatment/agent-specific requirement
→ evidence-defined milestone/due item
→ archetype base flow
→ patient-specific contextual item
```

Every dynamically surfaced non-obvious item should be able to answer `WHY NOW?`.

Repeated therapy must use actual treatment history, elapsed exposure, reliable administration count, due/overdue state and reviewed milestone rules rather than a separate hard-coded form for each ordinal visit.

Exact Prolia/other treatment milestone content is NOT frozen in G-0 and must not be invented without reviewed evidence or approved clinic-policy provenance.

---

# 5. Actual runtime seams already inspected

Read-only inspection confirms the current runtime already contains:

- coarse `encounterArchetype` values in `static/baseline-audit/index.html`;
- `adaptive-applicability.js` with archetype→domain `applicable/uncertain/not_applicable` mapping;
- Step 4 structured treatment episodes;
- Step 4 administration events/due dates;
- treatment decision / transition / tasks / Close state;
- protected encounter persistence in `clinical_encounters.payload_json`;
- archived corrected PR-1 v3 actual-runtime target mapping.

Therefore G-0 does **not** require a new patient-data model from scratch. It must define a deterministic presentation/guidance layer over current authoritative longitudinal data, adding only the minimum structured state later proven necessary.

Known current limitation: the coarse applicability map cannot yet express `why now`, due/milestone/event/unresolved-prior reasons, or treatment-timeline-derived relevance.

---

# 6. C1 authoritative Finish preserved

The parent runtime branch remains:

```text
branch: fix/module01-c1-authoritative-finish-2026-08-30
head:   a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871
state:  IMPLEMENTED / TESTED
MERGED: NO
DEPLOYED: NO
PRODUCTION-SMOKE: NO
```

Exact-head GitHub Actions run `33323204227` succeeded.

G-0 inherits this tested code in branch ancestry but does not grant merge/deploy authority.

---

# 7. Physiotherapy remains parked

Preserved rich-referral branch:

```text
feat/cu1-rich-referral-global-evidence-2026-08-29
@ bdd23b83a8252405f5aa5a0c0b67f303ccfcef5f
IMPLEMENTED / TESTED / PRODUCT-OWNER REVIEWED
MERGED NO / DEPLOYED NO
```

Do not mutate/merge/deploy it during Module 01 G-0.

---

# 8. G-0 status matrix

```text
PRODUCT PURPOSE RECONCILED                    YES
METHODOLOGY REPLAN                            YES
AGENTS UPDATED                                YES
TODO REBASED                                  YES
CLINICAL_EXCELLENCE_PLAN v3                   YES
SLICE PLAN G-0                                YES
MACHINE DYNAMIC-GUIDANCE CONTRACT             NOT YET
EXACT DESIGN-COMPLETENESS REVIEW              NOT YET
G-0 DESIGN-COMPLETE                           NO
RUNTIME MUTATION                              NO
C1 MERGED                                     NO
C1 DEPLOYED                                   NO
REAL PILOT STARTED                            NO
```

---

# 9. Exact next authorized action

```text
1. add machine-readable dynamic-guidance contract/schema for:
   EncounterContextV1
   GuidanceRuleV1
   VisitPlanV1
   GuidedCardStateV1
   GuidanceExposureV1
   TherapyMilestoneProfileV1;
2. verify the contract against actual current persisted/runtime paths;
3. run exact G-0 design-completeness review;
4. if PASS, update CURRENT_OPERATIONAL to G-0 DESIGN-COMPLETE and release canonical writer;
5. STOP before runtime implementation.
```

A separate product-owner/runtime authorization is required before creating G-1 runtime code. C1 merge/deploy also remains a separate release decision.
