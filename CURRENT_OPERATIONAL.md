# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 DESIGN GATE = `BLOCK`; TRANCHE2 PROMOTION = `PASS`; TRANCHE3 PROMOTION = `PASS`.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified current remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Current major phase:** CU-1 clinician-quality pre-runtime design hardening — history + criteria-based rehabilitation + route/subtype-specific evidence.
> **CU-1 status:** REOPENED / ACTIVE DESIGN HARDENING / EXACT OVERALL DESIGN GATE BLOCKED.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft; design-only; tranche3 promotion/matrix reconciliation reviewed through head `283c2769f88dbc77851e0463b33aa6f9adcd681f` before this canonical reconciliation commit.
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

The design-object and evidence-shard architecture is now hardened, but routine-route coverage and route-complete evidence-bounded composition are not yet complete.

---

# 2. Proven design-object semantics

Frozen design objects:

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
consensus/clinical opinion != low-certainty trial evidence
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

Promoted tranche2:

```text
clinic_utilities/contracts/cu1_evidence_tranche2_v1.yaml
+ clinic_utilities/contracts/cu1_evidence_tranche2_promotion_v1.yaml
+ clinic_utilities/contracts/CU1_TRANCHE2_PROMOTION_REVIEW_2026-08-28.md
```

Promoted tranche3:

```text
clinic_utilities/contracts/cu1_evidence_tranche3_v1.yaml
+ clinic_utilities/contracts/cu1_evidence_tranche3_promotion_v1.yaml
+ clinic_utilities/contracts/cu1_evidence_tranche3_promotion_fix_v1.yaml
+ clinic_utilities/contracts/CU1_TRANCHE3_PROMOTION_REVIEW_2026-08-29.md
```

Route work queue:

```text
clinic_utilities/contracts/cu1_evidence_coverage_matrix_v1.yaml
```

Synthetic semantic fixtures:

```text
clinic_utilities/contracts/cu1_history_evidence_fixtures_v1.yaml
```

Prior overall exact-gate report:

```text
clinic_utilities/contracts/CU1_DESIGN_COMPLETENESS_REVIEW_2026-08-28.md
```

That prior report predates tranche3 promotion. Its overall `BLOCK` remains directionally correct, but its tranche3-integration failure has now been resolved and the overall exact gate must be rerun only after the remaining route work is completed.

---

# 4. Evidence corpus state

Current normative design-time evidence state:

```text
core_seed_registry
→ ACTIVE DESIGN AUTHORITY
→ schema reviewed

high_frequency_tranche2
→ ACTIVE DESIGN AUTHORITY
→ immutable staging source + reviewed promotion projection
→ promotion gate PASS

shoulder_hip_meniscus_tranche3
→ ACTIVE DESIGN AUTHORITY
→ immutable staging source + reviewed promotion projection + mandatory exact-review overlay
→ promotion gate PASS
```

All currently listed shards have passed their native or reviewed promotion/schema gate. Only active design-authority shards participate in the normative logical design registry.

---

# 5. Tranche2 promotion — proven

Formal result:

```text
TRANCHE2 PROMOTION = PASS
STATE               = ACTIVE DESIGN AUTHORITY
RUNTIME AUTHORIZED  = NO
```

Material corrections included plantar-heel orthosis framework separation, AAOS/EULAR knee-OA separation, 2025 Dutch PFP framework addition without silent replacement of APTA 2019, lumbar-stenosis output-scope correction and source-identity hardening.

---

# 6. Tranche3 promotion — proven

Formal review result:

```text
identity materialization                      PASS
required non-optional fields                 PASS
source references                            PASS
claim references                             PASS
profile references                           PASS
sequence references                          PASS
route/subtype applicability                  PASS after mandatory overlay
output-scope compatibility                   PASS after correction
freshness/source identity                    PASS after refresh
cross-shard canonical-ID review              PASS
exact human evidence-scope review            PASS

TRANCHE3 PROMOTION                            PASS
TRANCHE3 STATE                                ACTIVE DESIGN AUTHORITY
ROUTE COMPLETION IMPLIED                      NO
RUNTIME AUTHORIZED                            NO
```

Material tranche3 corrections:

```text
acute isolated meniscus
→ AAOS PT / displaced-ROM-block statements preserved as workgroup clinical opinion / consensus
→ clinician_ui_only safety-authority leak removed
→ routine sequence scoped to acute isolated non-displaced non-repair-candidate nonoperative context

nonarthritic intra-articular hip
→ synthetic C_or_lower strength removed
→ exact Grade B / Grade C distinctions preserved
→ unsupported route-wide progression criterion removed
→ FAIS impingement and instability capsuloligamentous precautions separated
→ other established nonarthritic conditions remain outside automatic sequence applicability

GTPS / gluteal tendinopathy
→ canonical `formal_GTPS_diagnosis` identity restored
→ GTPS and formal gluteal-tendinopathy exercise authorities kept population-specific
→ formal trochanteric bursitis / other greater-trochanteric disorder not silently covered

posterior shoulder instability
→ 2025 Part I + Part II expert consensus preserved as expert consensus
→ no universal initial rehabilitation sequence invented

traumatic anterior shoulder instability
→ 2026 publication of 2024 ESSKA-ESA consensus retained only for traumatic-anterior scope
→ posterior/multidirectional and operative/nonoperative contexts not collapsed

adhesive capsulitis / frozen shoulder
→ stale 'no current CPG' gap removed
→ Korean 2025 CPG + BESS 2025 pathway added with 2023 SR context
→ supervised-PT uncertainty preserved
→ no universal criteria-based phase sequence invented

full-thickness rotator-cuff tear — nonoperative
→ AAOS 2025 PT benefit for patient-reported outcomes preserved
→ structural progression context preserved
→ no universal PT progression protocol invented
```

---

# 7. Current route evidence progress

The normalized logical evidence registry now has reviewed authority for core/tranche2/tranche3 profiles including:

```text
deep-gluteal presentation
nonspecific low-back pain
low-back pain with radiating leg symptoms
lumbar stenosis / neurogenic claudication
rotator-cuff-related shoulder pain
full-thickness rotator-cuff tear — nonoperative
adhesive capsulitis / frozen shoulder
traumatic anterior shoulder-instability evidence profile
posterior shoulder-instability evidence profile
lateral elbow tendinopathy
carpal tunnel evidence-gap profile
De Quervain evidence-gap profile
knee osteoarthritis
patellofemoral pain
acute isolated meniscus — selected nonoperative context
nonarthritic intra-articular hip — reviewed applicable scope
GTPS / formal gluteal tendinopathy scope
lateral ankle sprain
midportion Achilles
insertional Achilles
plantar heel pain / plantar fasciitis
```

Evidence authority does not imply route completeness.

---

# 8. Exact overall design gate state after tranche3 promotion

Current state:

```text
ReferralHistoryV2 semantics                 PASS
RehabilitationSequence object semantics     PASS
GoalPlanV2 / ReassessmentPlanV2 semantics   PASS
EvidenceSource / EvidenceClaim schema        PASS
Element-level evidence provenance            PASS
Protocol override model                      PASS
Active-core output-scope audit               PASS
Subtype-boundary architecture                PASS
Evidence freshness architecture              PASS
Tranche2 promotion                           PASS
Tranche3 promotion                           PASS
Shard integration                            PASS

Routine-route evidence coverage              FAIL
Route-specific history prompt coverage        FAIL
Route-complete fixture corpus                 FAIL
Several route progression/evidence gaps       BLOCKED / EXPLICIT

FINAL RESULT                                  BLOCK
DESIGN-COMPLETE                               NO
RUNTIME AUTHORIZED                            NO
```

The prior tranche3-integration blocker is closed. The remaining block is now route-content completeness, not shard integration.

---

# 9. Concrete evidence blocks preserved

Examples:

```text
DGS
→ low-quality conservative evidence
→ no validated disease-specific progression thresholds

De Quervain
→ comparative evidence centers on injection/orthosis
→ active progressive rehabilitation sequence not established

carpal tunnel syndrome
→ current management guideline is authoritative for medical management
→ no validated CU-1-style criteria-based PT sequence established by reviewed authority

acute isolated meniscal injury
→ selected non-displaced nonoperative PT wording is consensus/clinical opinion
→ no validated staged progression sequence

glenohumeral instability
→ direction, recurrence, age, structural context and operative/nonoperative pathway are material
→ no single generic route-level sequence

adhesive capsulitis
→ current 2025 guidance exists
→ no single validated universal criteria-based progression sequence

glenohumeral OA
→ PT may benefit selected patients by AAOS consensus
→ no reliable evidence-derived route progression sequence
```

The correct response remains explicit evidence limitation, not a generic invented sequence.

---

# 10. Exact next authorized action

Continue on the existing writer only:

```text
design/cu1-history-evidence-timeline-2026-08-28
```

Next route-coverage order from the reconciled matrix:

```text
1. calcific_rotator_cuff_tendinopathy
2. glenohumeral_instability_dislocation_initial_rehabilitation_split
3. glenohumeral_osteoarthritis
4. degenerative_meniscal_lesion_conservative_rehabilitation
5. patellar_tendinopathy
6. thumb_cmc1_osteoarthritis
7. cervical_routes
8. remaining_wrist_hand_and_elbow_routes
9. remaining routine routes in registry order
10. complete route-specific history prompts and matching fixtures alongside each route
11. rerun exact design-completeness review
12. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
```

For every route, preserve explicit evidence-gap behavior when evidence does not support a full staged sequence.

---

# 11. Explicitly forbidden

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
MERGE PR #63 merely because shard promotion gates passed
OPEN CU-2
RESTART PR-1
```

---

# 12. Continuity rule

A fresh session must repeat the six-canonical bootstrap, verify the then-current remote `main` and PR #63 head, inspect this exact `BLOCK` state and continue the remaining route-coverage/history-prompt/fixture work only on the existing CU-1 design writer unless the canonical lock has changed.
