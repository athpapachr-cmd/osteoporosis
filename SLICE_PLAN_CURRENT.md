# SLICE_PLAN_CURRENT.md — CU-1 history + evidence + rehabilitation-sequence design hardening v1

> **STATUS:** ACTIVE PRE-RUNTIME DESIGN HARDENING.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1 history-evidence-rehab-sequence v1.
> **Base:** `ad8045657616cd306b66d3becbda271f00c7fbbc`.
> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **Runtime writer:** NONE.
> **Clinical taxonomy:** frozen and preserved unless a specific evidence conflict requires a narrow reviewed correction.
> **Runtime implementation:** NOT AUTHORIZED in this slice until design-complete review passes.
> **CU-2:** not authorized.
> **PR-1:** remains paused.

---

# 1. Problem

The current CU-1 can produce Greek prose and dynamically relevant fields, but clinician review demonstrates that it is still not a sufficiently complete evidence-grounded referral system.

Three gaps are structural:

```text
A. HISTORY is under-modelled and therefore under-rendered
B. goals are a flat checklist rather than an ordered rehabilitation progression
C. route recommendations are not disease-specific machine-readable literature claims
```

The active objective is therefore not more formatter polish. It is to add the missing clinical-composition and evidence architecture.

---

# 2. Product-owner clarification — what "timeline" means

In CU-1, rehabilitation timing means **ordered therapeutic progression**, not routine prescription of visit frequency or total course duration.

Target model:

```text
initial clinical objective
→ objective/progression criteria met
→ next rehabilitation objective
→ criteria met
→ later functional/load phase
```

Examples of possible objectives include analgesia/symptom control, passive or protected ROM, active-assisted/active ROM, loading, strengthening, endurance, motor control and functional return.

These examples are not a universal protocol. Each selected condition/pathway must define its own evidence-supported sequence.

The physician is not interested in routinely dictating `1–2 sessions/week` or a fixed total number of weeks.

---

# 3. Objective

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
route-specific staged goals
+
criteria for progression from one objective to the next
+
disease-specific evidence-informed rehabilitation directions
+
criteria for reassessment/escalation
+
route-specific bibliography
```

The resulting system must preserve clinician judgment and physiotherapist autonomy while providing a safe and complete direction of care.

---

# 4. New typed objects

Normative candidate design is defined in:

```text
clinic_utilities/contracts/CU1_HISTORY_EVIDENCE_TIMELINE_V1.md
```

Required objects:

```text
ReferralHistoryV2
RouteHistorySelection
RehabilitationSequenceV1
RehabilitationPhaseV1
GoalPlanV2
ReassessmentPlanV2
EvidenceSource
EvidenceClaim
RouteEvidenceProfile
```

No patient identifiers are introduced.

---

# 5. History contract

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

# 6. Criteria-based rehabilitation sequence

Every routine route must have an explicit ordered `RehabilitationSequenceV1`.

Each phase contains:

```text
clinical objective
route-specific intervention directions
progression criteria
precautions / do-not-progress criteria
evidence claim IDs
```

Hard rules:

```text
progression criterion != elapsed time alone
same words across two diseases != same treatment authority
written postoperative/fracture protocol > route evidence sequence
unsupported phase is omitted rather than filled with a generic default
```

---

# 7. Disease-specific evidence architecture

CU-1 reuses the Clinical Excellence evidence-governance model.

Machine-readable evidence requires:

```text
EvidenceSource = bibliographic/source identity
EvidenceClaim = exact clinical claim supported/discouraged/left uncertain
RouteEvidenceProfile = all current claims for one clinical route
RehabilitationSequence = ordered route-specific application of those claims
```

There is no global musculoskeletal treatment template.

Examples:

```text
lateral_elbow_tendinopathy
→ 2022 lateral-elbow CPG + other route-specific evidence
→ elbow-specific intervention/progression claims

achilles_tendinopathy
→ 2024 midportion-Achilles CPG + other route-specific evidence
→ Achilles-specific tendon-loading/progression claims
```

A shared term such as `progressive loading` does not authorize identical referral text or progression logic across the two conditions.

---

# 8. Evidence must be visible in the referral

The literature connection is part of the clinician-to-physiotherapist communication, not merely hidden metadata.

Detailed output target:

```text
ΔΙΑΓΝΩΣΗ / ΚΛΙΝΙΚΗ ΕΝΤΥΠΩΣΗ
ΙΣΤΟΡΙΚΟ
ΚΛΙΝΙΚΑ ΕΥΡΗΜΑΤΑ
ΛΕΙΤΟΥΡΓΙΚΗ ΕΠΙΒΑΡΥΝΣΗ
ΑΙΤΗΜΑ
ΣΤΑΔΙΑΚΟΙ ΣΤΟΧΟΙ ΚΑΙ ΚΡΙΤΗΡΙΑ ΠΡΟΟΔΟΥ
ΠΡΟΤΕΙΝΟΜΕΝΟΣ ΠΡΟΣΑΝΑΤΟΛΙΣΜΟΣ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
ΠΡΟΫΠΟΘΕΣΕΙΣ ΕΠΑΝΕΚΤΙΜΗΣΗΣ / ΚΛΙΜΑΚΩΣΗΣ
ΒΙΒΛΙΟΓΡΑΦΙΚΗ ΒΑΣΗ
```

Short output:

```text
same route-specific evidence authority
+ compressed staged rehabilitation direction
+ compact disease-specific source footer
```

A short referral must not become generic merely because it is shorter.

---

# 9. Evidence wording

The generated referral must preserve evidence strength.

```text
strong/core recommendation
→ direct recommendation wording

conditional / low-certainty
→ "μπορεί να εξεταστεί" / "επικουρικά"

conflicting guidance
→ conflict retained; no silent hybrid

insufficient evidence
→ not presented as routine evidence-based treatment
```

The default bibliography should show 1–3 highest-authority route-specific sources.

---

# 10. Evidence freshness / update lifecycle

Every evidence source/profile includes:

```text
reviewed_on
next_review_due
freshness_state
supersedes / superseded_by
```

New evidence workflow:

```text
new guideline / systematic review detected
→ classify as confirming / no_change / potentially_practice_changing / practice_changing / conflicting
→ clinician/reviewer approval
→ update EvidenceClaim
→ update affected route RehabilitationSequence if warranted
→ regression tests/fixtures
→ version bump + changelog
```

No silent autonomous update of clinical recommendations.

---

# 11. Current seed examples

The evidence registry now includes design seeds for:

```text
deep_gluteal_piriformis_presentation
lateral_elbow_tendinopathy
achilles_tendinopathy
```

The elbow and Achilles seeds exist specifically to prove that two tendinopathies do not collapse into the same generic rehabilitation wording.

The seeds are incomplete and are not runtime authority yet.

---

# 12. Routine-route coverage gate

Before runtime evidence-aware generation is authorized:

```text
EVERY routine route
→ own RouteEvidenceProfile
→ own RehabilitationSequenceV1
→ every rendered phase/intervention/progression criterion resolves to >=1 active route-applicable claim
→ evidence gaps explicit
→ conflicts explicit
→ freshness current or explicitly reviewed
```

Rare/advanced routes may carry explicit evidence-gap status, but the system must not invent support.

---

# 13. Acceptance fixtures

At minimum:

```text
1. chronic deep-gluteal pain with 8-month duration, mechanism and uncertain diagnosis
2. lateral epicondylalgia → elbow-specific staged evidence-linked rehabilitation
3. Achilles tendinopathy → Achilles-specific loading/progression rehabilitation
4. elbow and Achilles outputs materially differ despite both being tendinopathies
5. postoperative shoulder → exact protocol overrides generic route evidence
6. fracture → healing/loading restrictions override evidence defaults
7. conflicting-framework adjunct case
8. route with weak evidence → appropriately broad/cautious sequence, not invented detail
9. missing history → omitted, never converted into reassuring negatives
10. stale/superseded evidence profile → cannot be labelled current guideline-based
```

---

# 14. REPLAN / BLOCK triggers

STOP and replan if:

```text
route evidence contradicts a frozen clinical recommendation materially
history schema requires new diagnosis inference
criteria-based progression cannot be represented without inventing unsupported thresholds
an evidence claim cannot preserve framework-specific wording/strength
routine-route evidence coverage cannot be completed without uncontrolled scope
```

---

# 15. Exact next action

```text
1. freeze typed history + criteria-based progression semantics
2. curate high-frequency route evidence profiles first
3. define complete RehabilitationSequence for each routine route
4. complete every routine route to evidence coverage gate
5. add evidence/conflict/gap/progression fixtures
6. exact design-completeness review
7. STOP at DESIGN-COMPLETE or BLOCK
```

No runtime evidence-aware generation in this design slice.
