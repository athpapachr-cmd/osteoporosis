# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-28 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified current main at slice start:** `ad8045657616cd306b66d3becbda271f00c7fbbc`.
> **Prior CU-1 dynamic-form fix:** PR #62 squash-merged as `ad8045657616cd306b66d3becbda271f00c7fbbc`.
> **Current major phase:** bounded CU-1 clinician-quality design hardening — history + evidence + goal/reassessment timing.
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
2. goals have no explicit evidence-aware time horizon / reassessment semantics
3. rehabilitation/timeline suggestions are disconnected from explicit literature provenance
```

These are not cosmetic defects. They change the referral draft/composition and evidence contracts and therefore trigger a design replan before further runtime mutation.

---

# 2. Current design authority

New design candidate:

```text
clinic_utilities/contracts/CU1_HISTORY_EVIDENCE_TIMELINE_V1.md
```

Seed evidence registry:

```text
clinic_utilities/contracts/cu1_evidence_registry_v1.yaml
```

The existing frozen clinical route taxonomy remains preserved unless a specific evidence conflict later demonstrates a narrow correction is required.

---

# 3. Required new objects / semantics

```text
ReferralHistoryV2
GoalPlanV1
ReassessmentPlanV1
EvidenceSource
EvidenceClaim
RouteEvidenceProfile
```

Core distinctions:

```text
history fact != diagnosis
selected goal != guaranteed outcome
reassessment window != promised recovery
clinician-selected intervention != evidence recommendation
low-certainty evidence != strong recommendation
missing route-specific evidence != permission to invent a default
```

---

# 4. Evidence governance requirement

CU-1 must reuse Clinical Excellence evidence governance.

Every evidence-labelled generated recommendation or timing statement must resolve to one or more active machine-readable `EvidenceClaim` objects with source provenance.

Conflicting frameworks remain separate. The runtime must never manufacture a silent hybrid recommendation.

Evidence status must be visible to the clinician, including explicit `insufficient_evidence` / `evidence_gap` states.

---

# 5. Timeline policy

There is no universal `6–8 week` recovery or goal-achievement default.

The system must distinguish:

```text
reassessment_window
expected_progress_window
goal_achievement_target
safety-triggered earlier reassessment
```

A default time window may be proposed only when an evidence claim supports that exact type of timing statement. Clinician-entered timing remains allowed but must not be relabelled as evidence-derived.

---

# 6. Deep-gluteal seed review

Current reviewed evidence seed demonstrates the intended behavior:

```text
classical DGS definition
→ non-discogenic sciatic nerve disorder/entrapment in deep gluteal space

history/exam
→ central to diagnostic pathway

specific conservative-treatment superiority
→ not established; overall evidence low quality

universal 6–8 week DGS recovery window
→ not supported by the reviewed evidence
```

Therefore the current CU-1 route may describe an uncertain deep-gluteal presentation unless the clinician explicitly asserts DGS/piriformis syndrome, and must not present a fixed treatment duration as evidence-based without supporting evidence.

---

# 7. Exact next authorized action

```text
1. freeze ReferralHistoryV2 semantics
2. freeze GoalPlanV1 / ReassessmentPlanV1 semantics
3. complete CU-1 EvidenceSource/EvidenceClaim registry schema
4. build route evidence map for every routine route
5. classify evidence gaps/conflicts and evidence strength
6. define route-specific dynamic history prompts
7. create synthetic composition/evidence fixtures
8. exact design-completeness review
9. STOP at DESIGN-COMPLETE or BLOCK
```

---

# 8. Explicitly forbidden until the gate passes

```text
WRITE runtime evidence recommendation logic
AUTO-SELECT rehabilitation because evidence says it is supported
INVENT recovery timelines where the literature does not support them
LABEL clinician preference as guideline recommendation
OPEN CU-2
RESTART PR-1
CHANGE frozen clinical taxonomy without a specific reviewed evidence conflict
```

---

# 9. Continuity rule

A new conversation can resume by fresh six-canonical bootstrap. It should find this branch/writer lock and continue the evidence/history/timeline design hardening only; chat history is not required for continuity.
