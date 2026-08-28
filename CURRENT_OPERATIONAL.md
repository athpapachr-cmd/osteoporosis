# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-28 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified current main at slice start:** `ad8045657616cd306b66d3becbda271f00c7fbbc`.
> **Prior CU-1 dynamic-form fix:** PR #62 squash-merged as `ad8045657616cd306b66d3becbda271f00c7fbbc`.
> **Current major phase:** bounded CU-1 clinician-quality design hardening — history + criteria-based rehabilitation sequence + disease-specific evidence.
> **CU-1 status:** REOPENED / REPLAN REQUIRED BEFORE FURTHER RUNTIME WORK.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **Runtime mutation in this slice:** NOT AUTHORIZED until design/evidence coverage review passes.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Trigger for replan

Product-owner clinician review identified three structural deficits that remain after formatter and dynamic-form maintenance:

```text
1. generated referral lacks a coherent structured HISTORY section
2. goals are flat and do not express a safe criteria-based rehabilitation progression
3. rehabilitation recommendations are not route-specific and are disconnected from explicit current literature provenance
```

Critical correction:

```text
"timeline" in this CU-1 slice
!= physiotherapy session frequency
!= total course duration
!= generic recovery prediction

"timeline" means
ordered rehabilitation phases/goals
→ each phase progresses after functional/clinical criteria are met
→ the next phase becomes appropriate only after the previous phase is sufficiently achieved
```

The physician does not need CU-1 to prescribe routine session frequency or total physiotherapy duration.

---

# 2. Current design authority

Primary design candidate:

```text
clinic_utilities/contracts/CU1_HISTORY_EVIDENCE_TIMELINE_V1.md
```

Machine schema:

```text
clinic_utilities/contracts/cu1_history_timeline_schema_v1.yaml
```

Evidence registry seed:

```text
clinic_utilities/contracts/cu1_evidence_registry_v1.yaml
```

The existing frozen clinical route taxonomy remains preserved unless a specific evidence conflict later demonstrates a narrow correction is required.

---

# 3. Required new objects / semantics

```text
ReferralHistoryV2
RehabilitationSequenceV1
RehabilitationPhaseV1
GoalPlanV2
ReassessmentPlanV2
EvidenceSource
EvidenceClaim
RouteEvidenceProfile
```

Core distinctions:

```text
history fact != diagnosis
rehabilitation phase order != calendar prescription
progression criterion != elapsed time alone
route_A evidence != route_B evidence
clinician-selected intervention != evidence recommendation
low-certainty evidence != strong recommendation
missing route-specific evidence != permission to invent a default
```

---

# 4. Disease-specific evidence requirement

Every routine CU-1 route must have its own versioned `RouteEvidenceProfile`.

Examples:

```text
lateral_elbow_tendinopathy
→ lateral-elbow-specific CPG/review claims
→ elbow-specific rehabilitation sequence

achilles_tendinopathy
→ Achilles-specific CPG/review claims
→ Achilles-specific loading/progression sequence
```

The system must not render the same generic rehabilitation package across unrelated routes merely because both may involve strengthening or loading.

Every generated rehabilitation recommendation must resolve to one or more active route-applicable `EvidenceClaim` objects.

---

# 5. What must appear in the referral

Evidence is not only a hidden clinician-side tooltip.

The generated referral itself must show the disease-specific evidence-informed rehabilitation direction for the selected condition.

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

Short output uses the same route-specific evidence profile in compressed form and includes a compact route-specific source footer.

---

# 6. Rehabilitation progression policy

The core model is criteria-based progression.

```text
phase 1 objective
→ progression criteria met
→ phase 2 objective
→ progression criteria met
→ later functional phase
```

Possible phase vocabulary includes symptom control, ROM restoration, active-assisted/active movement, loading, strengthening, endurance, motor control and return-to-function — but no universal phase sequence is allowed.

Each disease/pathway selects only the phases supported by its own evidence profile.

Written postoperative protocols and fracture/healing restrictions override generic route evidence.

---

# 7. Evidence freshness / renewal

Evidence sources and route profiles carry:

```text
reviewed_on
next_review_due
freshness_state
superseded_by
```

New evidence follows:

```text
new source detected
→ classify impact
→ clinician/reviewer approval
→ update claims
→ update affected route sequence only if warranted
→ regression fixtures
→ version/changelog
```

Evidence surveillance must never silently rewrite clinical recommendations.

---

# 8. Current evidence seeds

The design registry currently includes seed evidence for:

```text
deep_gluteal_piriformis_presentation
nonspecific_low_back_pain
low_back_pain_with_radiating_leg_symptoms
lateral_elbow_tendinopathy
achilles_tendinopathy
```

The elbow and Achilles seeds deliberately demonstrate different route-specific rehabilitation claims rather than a shared generic package.

These are seed profiles only; evidence-aware runtime generation remains blocked until all routine routes reach the coverage gate.

---

# 9. Exact next authorized action

```text
1. freeze ReferralHistoryV2 semantics
2. freeze RehabilitationSequenceV1 / GoalPlanV2 / ReassessmentPlanV2
3. complete EvidenceSource/EvidenceClaim registry schema
4. build RouteEvidenceProfile for every routine route
5. create a disease-specific RehabilitationSequence for every routine route
6. classify evidence gaps/conflicts and evidence strength
7. define route-specific dynamic history prompts
8. create synthetic composition/evidence fixtures
9. exact design-completeness review
10. STOP at DESIGN-COMPLETE or BLOCK
```

---

# 10. Explicitly forbidden until the gate passes

```text
WRITE runtime evidence recommendation logic
USE one generic rehab sequence across conditions
INVENT progression criteria not supported by evidence/protocol
USE elapsed time alone as universal progression criterion
PRESCRIBE routine session frequency or total PT duration as a CU-1 requirement
LABEL clinician preference as guideline recommendation
LABEL stale/superseded evidence as current
OPEN CU-2
RESTART PR-1
CHANGE frozen clinical taxonomy without a specific reviewed evidence conflict
```

---

# 11. Continuity rule

A new conversation can resume by fresh six-canonical bootstrap. It should find this branch/writer lock and continue the history + criteria-based rehabilitation-sequence + disease-specific evidence design hardening only; chat history is not required for continuity.
