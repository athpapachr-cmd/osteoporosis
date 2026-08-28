# SLICE_PLAN_CURRENT.md — CU-1 history + evidence + timeline design hardening v1

> **STATUS:** ACTIVE PRE-RUNTIME DESIGN HARDENING.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1 history-evidence-timeline v1.
> **Base:** `ad8045657616cd306b66d3becbda271f00c7fbbc`.
> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **Runtime writer:** NONE.
> **Clinical taxonomy:** frozen and preserved unless a specific evidence conflict requires a narrow reviewed correction.
> **Runtime implementation:** NOT AUTHORIZED in this slice until design-complete review passes.
> **CU-2:** not authorized.
> **PR-1:** remains paused.

---

# 1. Problem

The current CU-1 can now produce Greek prose and dynamically relevant fields, but clinician review demonstrates that it is still not a sufficiently evidence-grounded referral system.

Three gaps are structural:

```text
A. HISTORY is under-modelled and therefore under-rendered
B. goals have no explicit time horizon / reassessment semantics
C. route recommendations and timelines do not resolve to explicit literature claims
```

The active objective is therefore not more formatter polish. It is to add the missing clinical-composition and evidence architecture.

---

# 2. Objective

Build a referral contract that supports:

```text
clinical presentation / asserted diagnosis
+
coherent structured history
+
actual examination/findings
+
functional impact
+
referral request
+
selected goals with explicit timing semantics
+
evidence-aware rehabilitation directions
+
reassessment plan
```

The resulting system must preserve clinician autonomy and must never turn evidence metadata into autonomous diagnosis or treatment selection.

---

# 3. New typed objects

Normative candidate design is defined in:

```text
clinic_utilities/contracts/CU1_HISTORY_EVIDENCE_TIMELINE_V1.md
```

Required objects:

```text
ReferralHistoryV2
RouteHistorySelection
GoalPlanV1
ReassessmentPlanV1
EvidenceSource
EvidenceClaim
RouteEvidenceProfile
```

No patient identifiers are introduced.

---

# 4. History contract

History must be capable of carrying, when explicitly supplied:

```text
onset date or approximate duration
onset pattern
mechanism / trigger
symptom course
prior episodes
prior treatment + response
relevant investigations
aggravating/easing factors
work/sport/activity context
patient-priority activity
route-specific history items
```

Rules:

```text
approximate duration != inferred exact date
mechanism != causal diagnosis
missing history != negative history
route-specific history questions are dynamically scoped
nothing is auto-selected
```

Detailed formatter must include a coherent `ΙΣΤΟΡΙΚΟ` section when data exist.

---

# 5. Timing contract

Timing must distinguish:

```text
reassessment_window
expected_progress_window
goal_achievement_target
safety-triggered earlier reassessment
```

A time window carries provenance:

```text
clinician_entered
evidence_supported_default
evidence_informed_suggestion
```

and certainty/strength where available.

No generic `6–8 weeks` default is permitted.

---

# 6. Evidence architecture

CU-1 reuses the Clinical Excellence evidence-governance model.

Machine-readable evidence requires two layers:

```text
EvidenceSource = bibliographic/source identity
EvidenceClaim = what that source supports, discourages, or leaves uncertain
```

A `RouteEvidenceProfile` maps a route to its claims across:

```text
diagnostic definition
history
examination
core rehabilitation
adjuncts
timeline
reassessment
safety/differential where relevant
```

No generated statement may be labelled evidence-based unless it resolves to an active claim.

---

# 7. Evidence status visible to clinician

The UI should ultimately distinguish:

```text
SUPPORTED / RECOMMENDED
CONDITIONAL / MAY CONSIDER
CONFLICTING FRAMEWORKS
INSUFFICIENT ROUTE-SPECIFIC EVIDENCE
CLINICIAN-SELECTED — NO EVIDENCE CLAIM ATTACHED
DO NOT OFFER / NOT ROUTINE
```

Evidence status informs but does not auto-select treatment.

---

# 8. Routine-route coverage gate

Before runtime evidence-aware generation is authorized:

```text
EVERY routine route
→ has RouteEvidenceProfile
→ every evidence-labelled rehab direction resolves to >=1 active claim
→ every evidence-labelled timeline resolves to >=1 active timing/reassessment claim
→ evidence gaps are explicit
→ framework conflicts are explicit
```

Rare/advanced routes may carry explicit evidence-gap status, but the system must not invent support.

---

# 9. Deep-gluteal seed fixture

The first seed evidence profile is `deep_gluteal_piriformis_presentation`.

Required behavior based on reviewed sources:

```text
classical DGS definition
→ non-discogenic sciatic nerve disorder/entrapment in deep gluteal space

buttock pain alone
→ does not establish DGS/piriformis syndrome

history + examination
→ central to diagnostic pathway

specific conservative treatment superiority
→ insufficient/low-quality evidence

universal fixed 6–8 week recovery window
→ not evidence-supported
```

The route may still support physiotherapy referral and clinician-selected active rehabilitation, but the evidence panel must represent uncertainty accurately.

---

# 10. Formatter target

Detailed output target:

```text
ΔΙΑΓΝΩΣΗ / ΚΛΙΝΙΚΗ ΕΝΤΥΠΩΣΗ
ΙΣΤΟΡΙΚΟ
ΚΛΙΝΙΚΑ ΕΥΡΗΜΑΤΑ
ΛΕΙΤΟΥΡΓΙΚΗ ΕΠΙΒΑΡΥΝΣΗ
ΑΙΤΗΜΑ
ΣΤΟΧΟΙ + ΧΡΟΝΙΚΟΣ ΟΡΙΖΟΝΤΑΣ
ΚΑΤΕΥΘΥΝΣΕΙΣ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
ΕΠΑΝΕΚΤΙΜΗΣΗ
```

Short output remains compact but must carry meaningful duration/mechanism and timeline when supplied.

Evidence bibliography is clinician-facing by default; an optional bibliography appendix may later be enabled for Detailed output.

---

# 11. Acceptance fixtures

At minimum:

```text
1. chronic deep-gluteal pain with 8-month duration, mechanism and uncertain diagnosis
2. lateral epicondylalgia with history + evidence-linked loading direction
3. knee OA with exercise evidence and a defensible reassessment window
4. postoperative shoulder where protocol overrides generic evidence suggestions
5. fracture where healing/loading restrictions override evidence defaults
6. conflicting-framework adjunct case
7. route with no supported timeline → no invented recovery window
8. clinician-entered goal target → rendered as clinician target, not evidence-derived
9. missing history → omitted, never converted into reassuring negatives
10. evidence-gap route → explicit clinician-facing evidence-gap state
```

---

# 12. REPLAN / BLOCK triggers

STOP and replan if:

```text
route evidence contradicts a frozen clinical recommendation materially
history schema requires new diagnosis inference
reassessment timing cannot be separated from recovery prediction
an evidence claim cannot preserve framework-specific wording/strength
routine-route evidence coverage cannot be completed without uncontrolled scope
```

---

# 13. Exact next action

```text
1. review/freeze typed history and timeline semantics
2. curate route evidence profiles beginning with high-frequency routine routes
3. complete every routine route to coverage gate
4. add evidence/conflict/gap fixtures
5. exact design-completeness review
6. STOP at DESIGN-COMPLETE or BLOCK
```

No runtime evidence-aware generation in this design slice.
