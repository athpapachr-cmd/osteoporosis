# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 DESIGN GATE = `BLOCK`; TRANCHE2 PROMOTION = `PASS`.
> **Updated:** 2026-08-28 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified current remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Current major phase:** CU-1 clinician-quality pre-runtime design hardening — history + criteria-based rehabilitation + route/subtype-specific evidence.
> **CU-1 status:** REOPENED / ACTIVE DESIGN HARDENING / EXACT OVERALL DESIGN GATE BLOCKED.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft; design-only; tranche2 promotion reviewed at head `ef45e8b683fb67287012cb1ba2327089ef299c18` before this canonical reconciliation commit.
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

# 2. What is proven at the design-object level

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
framework-specific strength != synthetic cross-framework strength
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

Promoted tranche2 staging source + required normalization projection:

```text
clinic_utilities/contracts/cu1_evidence_tranche2_v1.yaml
+
clinic_utilities/contracts/cu1_evidence_tranche2_promotion_v1.yaml
```

Tranche2 promotion review:

```text
clinic_utilities/contracts/CU1_TRANCHE2_PROMOTION_REVIEW_2026-08-28.md
```

Remaining staged evidence candidate:

```text
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

Overall exact gate report:

```text
clinic_utilities/contracts/CU1_DESIGN_COMPLETENESS_REVIEW_2026-08-28.md
```

---

# 4. Evidence corpus state

The evidence registry is sharded for maintainability.

Current normative state:

```text
core_seed_registry
→ ACTIVE DESIGN AUTHORITY
→ schema reviewed

high_frequency_tranche2
→ ACTIVE DESIGN AUTHORITY
→ immutable staging source + reviewed promotion projection
→ promotion gate PASS

shoulder_hip_meniscus_tranche3
→ STAGED CANDIDATE
→ not schema-frozen / not promoted
```

Only active design-authority shards participate in the normative logical design registry.

---

# 5. Tranche2 promotion — proven

Formal tranche2 review result:

```text
identity materialization                      PASS
required non-optional fields                 PASS
source references                            PASS
claim references                             PASS
profile references                           PASS
sequence references                          PASS
route/subtype applicability                  PASS
output-scope compatibility                   PASS
freshness/source identity                    PASS
cross-shard duplicate IDs                    PASS
exact human evidence-scope review            PASS

TRANCHE2 PROMOTION                            PASS
TRANCHE2 STATE                                ACTIVE DESIGN AUTHORITY
RUNTIME AUTHORIZED                            NO
```

Material corrections made before promotion:

```text
plantar heel orthosis
→ removed silent B/C hybrid
→ Grade-B do-not-use-isolated and Grade-C combined-use claims separated

knee OA
→ AAOS 2021 and EULAR 2023-update claims separated
→ AAOS strength is no longer projected onto EULAR
→ no synthetic hybrid phase strength

patellofemoral pain
→ 2025 Dutch multidisciplinary guideline added as a distinct current framework
→ 2019 APTA framework retained separately
→ 6/12-week Dutch items represented as reassessment windows, not automatic phase advancement

lumbar stenosis
→ clinician_ui_only claim removed from rendered objective authority
→ conditional/weak wording preserved in referral-compatible authority

source metadata
→ EULAR OA, LSS and De Quervain 2025 identities hardened
```

The staging file is retained unchanged as audit history; its promotion projection is mandatory for normative reads.

---

# 6. Current route evidence progress

Active core route-specific profiles include:

```text
deep_gluteal_piriformis_presentation
nonspecific_low_back_pain
low_back_pain_with_radiating_leg_symptoms
lateral_elbow_tendinopathy
achilles_tendinopathy — midportion
achilles_tendinopathy — insertional
```

Promoted tranche2 adds authoritative design profiles for:

```text
lumbar spinal stenosis / neurogenic claudication
lateral ankle sprain
plantar heel pain / plantar fasciitis
rotator-cuff-related shoulder pain
knee osteoarthritis
patellofemoral pain
carpal tunnel syndrome — explicit evidence-gap profile
De Quervain — explicit evidence-gap profile
```

Tranche3 research remains staged and includes or partially includes:

```text
full-thickness rotator-cuff tear — nonoperative
adhesive capsulitis
acute isolated meniscal injury — nonoperative
nonarthritic intra-articular hip pain
GTPS / gluteal-tendinopathy scope
anterior/posterior shoulder-instability evidence seeds
```

Coverage status is not runtime readiness.

---

# 7. Exact overall design gate state

The prior overall gate remains correctly blocked:

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
Tranche2 promotion                           PASS

Routine-route evidence coverage              FAIL
Tranche3 promotion/conformance                FAIL
Route-specific history prompt coverage        FAIL
Route-complete fixture corpus                 FAIL
Several route progression/evidence gaps       BLOCKED / EXPLICIT

FINAL RESULT                                  BLOCK
DESIGN-COMPLETE                               NO
RUNTIME AUTHORIZED                            NO
```

---

# 8. Concrete evidence blocks preserved

Promotion does not erase legitimate evidence limitations.

Examples:

```text
DGS
→ low-quality conservative evidence
→ no validated disease-specific progression thresholds

De Quervain
→ current comparative evidence centers on injection/orthosis
→ active progressive rehabilitation sequence not established

carpal tunnel syndrome
→ current management guideline is authoritative for medical management
→ no validated CU-1-style criteria-based PT sequence established by the reviewed source

acute isolated meniscal injury
→ PT may benefit selected non-displaced nonoperative cases by consensus
→ no validated staged progression sequence supplied by the reviewed CPG

glenohumeral OA
→ PT may benefit selected patients by AAOS consensus
→ no reliable evidence-derived route progression sequence
```

The correct response remains explicit evidence limitation, not a generic invented sequence.

---

# 9. Exact next authorized action

Continue on the existing writer only:

```text
design/cu1-history-evidence-timeline-2026-08-28
```

Next order:

```text
1. normalize tranche3 map-key identities and complete required object fields
2. exact reference/applicability/output-scope/freshness review of tranche3
3. promote tranche3 only if the evidence-manifest promotion gate passes
4. continue remaining routine routes from cu1_evidence_coverage_matrix_v1.yaml
5. define narrow safe evidence-gap behavior where staged progression is unsupported
6. complete route-specific history prompts and matching fixtures with each route
7. rerun exact design-completeness review
8. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
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
PROMOTE tranche3 without exact conformance review
MERGE PR #63 merely because tranche2 is promoted
OPEN CU-2
RESTART PR-1
```

---

# 11. Continuity rule

A fresh session must repeat the six-canonical bootstrap, verify the then-current remote `main` and PR #63 head, inspect this exact `BLOCK` state and continue with tranche3 normalization/review/promotion only on the existing CU-1 design writer unless the canonical lock has changed.
