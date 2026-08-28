# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 DESIGN GATE = `BLOCK`.
> **Updated:** 2026-08-28 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified current remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Current major phase:** CU-1 clinician-quality pre-runtime design hardening — history + criteria-based rehabilitation + route/subtype-specific evidence.
> **CU-1 status:** REOPENED / ACTIVE DESIGN HARDENING / EXACT DESIGN GATE BLOCKED.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft; design-only; reviewed design head before canonical closeout `a73b72beefcaafa901163a77758b795b78101330`.
> **Runtime evidence-aware generation:** NOT AUTHORIZED.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused.

---

# 1. Why CU-1 remains open

Clinician review identified three structural deficits after the prior formatter/dynamic-form work:

```text
1. referral history was not coherent enough
2. goals were flat rather than criteria-based rehabilitation progression
3. rehabilitation directions were not explicitly route/subtype-linked to current evidence
```

The current slice addresses those design/evidence gaps only.

---

# 2. What is now proven at the design-object level

The following semantics are frozen for the current design pass:

```text
ReferralHistoryV2 with explicit provenance
HistoryProvenanceEntryV1
RouteHistoryPromptV1
RehabilitationSequenceV1
RehabilitationPhaseV1
InterventionDirectionV1
RehabilitationCriterionV1
GoalPlanV2
ReassessmentPlanV2
AuthorityReferenceV1
ProtocolConstraintV1
ClinicianModificationV1
EvidenceSourceV1
EvidenceClaimV1
RouteEvidenceProfileV1
```

Proven invariants include:

```text
missing history != negative history
patient statement != objective finding
approximate duration != inferred exact date
progression != elapsed time alone
route A evidence != route B evidence
subtype A authority != subtype B authority
therapist_execution_detail != automatic referral_core
clinician_ui_only != automatic referral_core
clinician instruction != evidence recommendation
patient-specific written protocol != literature recommendation
explicit written protocol/healing restriction > conflicting route default
```

No patient identifiers were added.

---

# 3. Current design/evidence artifacts

Human contract:

```text
clinic_utilities/contracts/CU1_HISTORY_EVIDENCE_TIMELINE_V1.md
```

Machine design schema:

```text
clinic_utilities/contracts/cu1_history_timeline_schema_v1.yaml
```

Evidence manifest:

```text
clinic_utilities/contracts/cu1_evidence_manifest_v1.yaml
```

Active core evidence shard:

```text
clinic_utilities/contracts/cu1_evidence_registry_v1.yaml
```

Staged evidence candidates:

```text
clinic_utilities/contracts/cu1_evidence_tranche2_v1.yaml
clinic_utilities/contracts/cu1_evidence_tranche3_v1.yaml
```

Route work queue:

```text
clinic_utilities/contracts/cu1_evidence_coverage_matrix_v1.yaml
```

Synthetic semantic fixtures:

```text
clinic_utilities/contracts/cu1_history_evidence_fixtures_v1.yaml
```

Exact gate report:

```text
clinic_utilities/contracts/CU1_DESIGN_COMPLETENESS_REVIEW_2026-08-28.md
```

---

# 4. Evidence corpus state

The evidence registry is sharded for maintainability.

Current normative state:

```text
core_seed_registry
→ active design authority

high_frequency_tranche2
→ staged candidate, not schema-frozen

shoulder_hip_meniscus_tranche3
→ staged candidate, not schema-frozen
```

Staged shards may inform the work queue but must not be treated as normative referral authority until they pass the manifest promotion gate.

---

# 5. Current route evidence progress

Active core route-specific profiles include:

```text
deep_gluteal_piriformis_presentation
nonspecific_low_back_pain
low_back_pain_with_radiating_leg_symptoms
lateral_elbow_tendinopathy
achilles_tendinopathy — midportion
achilles_tendinopathy — insertional
```

Current staged research additionally covers or partially covers:

```text
lumbar spinal stenosis / neurogenic claudication
lateral ankle sprain
plantar heel pain / plantar fasciitis
rotator-cuff-related shoulder pain
full-thickness rotator-cuff tear — nonoperative
adhesive capsulitis
knee osteoarthritis
patellofemoral pain
acute isolated meniscal injury — nonoperative
nonarthritic intra-articular hip pain
GTPS / gluteal-tendinopathy scope
carpal tunnel syndrome evidence-gap profile
De Quervain evidence-gap profile
anterior/posterior shoulder-instability evidence seeds
```

Coverage status is not runtime readiness.

---

# 6. Exact design gate result

Formal review result:

```text
ReferralHistoryV2 semantics                 PASS
RehabilitationSequence object semantics     PASS
GoalPlanV2 / ReassessmentPlanV2 semantics   PASS
EvidenceSource / EvidenceClaim schema        PASS
Element-level evidence provenance            PASS
Protocol override model                      PASS
Active core output-scope audit               PASS
Subtype-boundary architecture                PASS
Evidence freshness architecture              PASS

Routine-route evidence coverage              FAIL
Staged-shard promotion/conformance            FAIL
Route-specific history prompt coverage        FAIL
Route-complete fixture corpus                 FAIL
Several route progression/evidence gaps       BLOCKED / EXPLICIT

FINAL RESULT                                  BLOCK
DESIGN-COMPLETE                               NO
RUNTIME AUTHORIZED                            NO
```

---

# 7. Concrete evidence blocks

Fresh evidence review confirms that not every routine route supports the same level of staged precision.

Examples:

```text
DGS
→ low-quality conservative evidence
→ no validated disease-specific progression thresholds

De Quervain
→ current comparative evidence centers on injection/orthosis
→ active progressive rehabilitation sequence not established

carpal tunnel syndrome
→ current management guideline + conditional conservative rehab evidence
→ no validated CU-1-style criteria-based PT sequence established

acute isolated meniscal injury
→ PT may benefit selected non-displaced nonoperative cases by consensus
→ no validated staged progression sequence supplied by the CPG

glenohumeral OA
→ PT may benefit selected patients by AAOS consensus
→ no reliable evidence-derived route progression sequence
```

The correct response is explicit evidence limitation, not a generic invented sequence.

---

# 8. Subtype/applicability boundaries proven by current evidence

Hard boundaries now include:

```text
midportion Achilles != insertional Achilles
rotator-cuff tendinopathy != full-thickness rotator-cuff tear
GTPS/gluteal-tendinopathy evidence != isolated trochanteric bursitis automatically
traumatic anterior instability != posterior/multidirectional instability automatically
acute isolated meniscus != chronic/degenerative meniscal lesion
```

The frozen route taxonomy remains unchanged; evidence applicability conditions handle these distinctions unless a later reviewed conflict proves a taxonomy correction necessary.

---

# 9. Exact next authorized action

Continue on the existing writer only:

```text
design/cu1-history-evidence-timeline-2026-08-28
```

Next order:

```text
1. normalize tranche2 map-key identities and complete required object fields
2. exact reference/applicability/output-scope/freshness review of tranche2
3. promote tranche2 only if the evidence-manifest promotion gate passes
4. repeat for tranche3
5. continue remaining routine routes from cu1_evidence_coverage_matrix_v1.yaml
6. define narrow safe evidence-gap behavior for routes where staged progression is unsupported
7. complete route-specific history prompts and matching fixtures with each route
8. rerun exact design-completeness review
9. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
```

---

# 10. Explicitly forbidden

```text
WRITE runtime evidence recommendation logic
WRITE runtime formatter integration for this evidence corpus
CHANGE persistence/retention behavior
USE a generic MSK rehabilitation fallback
INVENT progression thresholds
USE elapsed time alone as universal progression criterion
LABEL clinician preference as guideline recommendation
LABEL therapist-execution detail as physician prescription by default
USE evidence across a noncovered subtype
PROMOTE staged shard without exact conformance review
MERGE PR #63 merely because several routes are well curated
OPEN CU-2
RESTART PR-1
```

---

# 11. Continuity rule

A fresh session must repeat the six-canonical bootstrap, verify the then-current remote `main` and PR #63 head, inspect this exact `BLOCK` state, and continue only the existing CU-1 design writer unless the canonical lock has changed.
