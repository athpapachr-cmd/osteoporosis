# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 DESIGN GATE = `BLOCK`; TRANCHE2 = `PASS`; TRANCHE3 = `PASS`; SHARD INTEGRATION = `PASS`.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft / design-only; tranche3 promotion closeout lineage is complete through `1d69fafeee5a30a8191817748de9d0a2629c981b`; this file update itself may advance the branch by one documentation commit.
> **Runtime evidence-aware generation:** NOT AUTHORIZED.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused.

---

# 1. Current phase

CU-1 remains open because clinician review required:

```text
coherent structured HISTORY
+ criteria-based / evidence-bounded rehabilitation progression
+ route/subtype-specific literature provenance
```

The object/evidence architecture and all currently listed evidence shards are now reviewed. The remaining work is route-content completeness, route-specific history prompts and route-complete regression fixtures.

---

# 2. Proven design invariants

```text
missing history != negative history
patient statement != objective finding
approximate duration != inferred exact date
progression != elapsed time alone
route A evidence != route B evidence
subtype A authority != subtype B authority
clinician_ui_only != rendered referral authority
therapist_execution_detail != automatic referral_core
clinician instruction != evidence recommendation
patient-specific protocol != literature recommendation
explicit written protocol/healing restriction > conflicting route default
framework-specific strength != synthetic cross-framework strength
expert consensus / clinical opinion != low-certainty trial evidence
```

Frozen objects include `ReferralHistoryV2`, `RouteHistoryPromptV1`, `RehabilitationSequenceV1`, `RehabilitationPhaseV1`, `GoalPlanV2`, `ReassessmentPlanV2`, `AuthorityReferenceV1`, `ProtocolConstraintV1`, `ClinicianModificationV1`, `EvidenceSourceV1`, `EvidenceClaimV1` and `RouteEvidenceProfileV1`.

No patient identifiers were added.

---

# 3. Normative evidence corpus

Manifest:

```text
clinic_utilities/contracts/cu1_evidence_manifest_v1.yaml
```

Current logical shards:

```text
core_seed_registry
→ ACTIVE DESIGN AUTHORITY
→ native schema review PASS

high_frequency_tranche2
→ ACTIVE DESIGN AUTHORITY
→ cu1_evidence_tranche2_v1.yaml
→ cu1_evidence_tranche2_promotion_v1.yaml
→ CU1_TRANCHE2_PROMOTION_REVIEW_2026-08-28.md
→ PASS

shoulder_hip_meniscus_tranche3
→ ACTIVE DESIGN AUTHORITY
→ cu1_evidence_tranche3_v1.yaml
→ cu1_evidence_tranche3_promotion_v1.yaml
→ cu1_evidence_tranche3_promotion_fix_v1.yaml
→ CU1_TRANCHE3_PROMOTION_REVIEW_2026-08-29.md
→ PASS
```

All listed shards have passed their native or reviewed promotion/schema gate. No staged shard currently remains in the manifest.

---

# 4. Tranche3 promotion — exact result

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

Material corrections before promotion:

```text
acute isolated meniscus
→ AAOS PT and displaced/ROM-block statements preserved as clinical opinion/consensus
→ prior clinician_ui_only rendered safety leak removed
→ routine sequence limited to acute isolated non-displaced non-repair-candidate nonoperative context

nonarthritic intra-articular hip
→ synthetic C_or_lower grade removed
→ exact Grade B / C distinctions preserved
→ unsupported route-wide progression criterion removed
→ FAIS impingement vs hip-instability precautions separated
→ other established intra-articular conditions not silently covered by the sequence

GTPS / gluteal tendinopathy
→ exact canonical `formal_GTPS_diagnosis` restored
→ GTPS and gluteal-tendinopathy evidence kept population-specific
→ formal trochanteric bursitis / other greater-trochanteric disorders not silently covered

shoulder instability
→ traumatic-anterior and posterior authorities remain direction/context specific
→ posterior literature remains expert consensus
→ no universal initial rehabilitation sequence invented

adhesive capsulitis / frozen shoulder
→ 2025 Korean CPG + BESS pathway added
→ 2023 SR retained as supporting context
→ supervised-PT uncertainty preserved
→ no universal criteria-based progression sequence invented

full-thickness rotator-cuff tear — nonoperative
→ AAOS 2025 PT benefit for patient-reported outcomes preserved
→ structural progression context preserved
→ no universal PT progression protocol invented
```

---

# 5. Current overall gate

```text
object/history/evidence-authority semantics   PASS
element-level evidence provenance             PASS
protocol override model                       PASS
tranche2 promotion                            PASS
tranche3 promotion                            PASS
shard integration                             PASS

routine-route evidence coverage               FAIL
route-specific history prompt coverage         FAIL
route-complete fixture corpus                  FAIL
several route progression/evidence gaps        BLOCKED / EXPLICIT

FINAL RESULT                                   BLOCK
DESIGN-COMPLETE                                NO
RUNTIME AUTHORIZED                             NO
```

The prior tranche3-integration blocker is closed. The remaining block is route-content completeness.

---

# 6. Preserved evidence-gap behavior

Examples:

```text
DGS
→ no validated disease-specific progression thresholds

De Quervain
→ no validated active progressive rehabilitation sequence

carpal tunnel syndrome
→ no validated CU-1-style criteria-based PT sequence from reviewed authority

acute isolated meniscus
→ selected PT wording is consensus/clinical opinion
→ no validated staged progression sequence

glenohumeral instability
→ direction/recurrence/age/structural/operative context material
→ no single generic route sequence

adhesive capsulitis
→ current 2025 guidance exists
→ no single validated universal phase progression

glenohumeral OA
→ PT guidance remains consensus-level without route-specific progression sequence
```

No generic MSK fallback is permitted.

---

# 7. Exact next authorized action

Continue only on the existing writer, route-by-route from the reconciled matrix:

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
10. define reviewed evidence-gap behavior where full staging is unsupported
11. complete route-specific history prompts + matching fixtures alongside route curation
12. rerun exact design-completeness review
13. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
```

---

# 8. Explicitly forbidden

```text
WRITE runtime evidence recommendation logic
WRITE runtime formatter integration
CHANGE persistence/retention behavior
USE generic MSK rehabilitation fallback
INVENT progression thresholds
USE elapsed time alone as universal progression criterion
LABEL clinician preference as guideline recommendation
LABEL therapist execution detail as physician prescription by default
USE evidence across a noncovered subtype
MERGE PR #63 merely because shard integration passed
OPEN CU-2
RESTART PR-1
```

---

# 9. Continuity rule

A fresh session must repeat the six-canonical bootstrap, verify the then-current remote `main` and PR #63 head, inspect this exact `BLOCK`, and continue only the route-coverage/history-prompt/fixture work on the existing CU-1 design writer unless the canonical lock changes.
