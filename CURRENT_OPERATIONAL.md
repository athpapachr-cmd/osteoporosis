# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 DESIGN GATE = `BLOCK`; SHARD INTEGRATION = `PASS`; ROUTE COVERAGE IN PROGRESS.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft / design-only / unmerged; reviewed route coverage is complete through matrix commit `6543e8d06b7492ceb391b22ecf514dbd7c9e3bd2`; this canonical commit may advance branch head.
> **Runtime evidence-aware generation:** NOT AUTHORIZED.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused.

---

# 1. Current phase

CU-1 remains in clinician-quality **pre-runtime design hardening** for:

```text
coherent structured HISTORY
+ criteria-based / evidence-bounded rehabilitation progression
+ route/subtype/management-context-specific literature provenance
```

The object/evidence architecture, tranche2/tranche3 promotion and shard integration are proven. Work is now route-by-route completeness with route-specific history prompts and matching regression fixtures.

---

# 2. Proven design invariants

```text
missing history != negative history
patient statement != objective finding
approximate duration != inferred exact date
progression != elapsed time alone
route A evidence != route B evidence
subtype/context A authority != subtype/context B authority
clinician_ui_only != rendered referral authority
therapist_execution_detail != automatic referral_core
clinician instruction != evidence recommendation
patient-specific protocol != literature recommendation
explicit written protocol/healing restriction > conflicting route default
framework-specific strength != synthetic cross-framework strength
expert consensus / clinical opinion / best-practice opinion != treatment-effect estimate
no eligible comparative trials != low or very-low effect estimate
framework conflict != silent guideline consensus
```

No patient identifiers were added.

---

# 3. Normative evidence corpus

Manifest:

```text
clinic_utilities/contracts/cu1_evidence_manifest_v1.yaml
```

Active logical evidence layers:

```text
core_seed_registry
high_frequency_tranche2 + reviewed promotion projection
shoulder_hip_meniscus_tranche3 + reviewed projection/overlay
route_coverage_extension
shoulder_instability_route_coverage_extension
cu1_evidence_route_coverage_amendments_v1.yaml
```

All listed shards have passed their native or reviewed schema/promotion gate. No staged evidence shard remains.

---

# 4. Newly completed route coverage

## Calcific rotator-cuff tendinopathy — PASS

```text
rep_calcific_rotator_cuff_v1
seq_calcific_rotator_cuff_v1
→ sequence_complete — evidence-bounded
```

Core active rehabilitation is supported. ESWT remains an explicit JOSPT-2025-vs-NICE-HTG645 conflict and is not automatically rendered.

## Glenohumeral instability/dislocation split — PASS

No generic instability sequence exists.

```text
traumatic anterior
posterior nonoperative
atraumatic anterior nonoperative
multidirectional instability nonoperative
```

are distinct evidence contexts. Unresolved direction/cause/management blocks evidence-aware output; postoperative instability belongs to the postoperative owner. Posterior Part-II postoperative RTS evidence is suppressed from nonoperative authority.

## Glenohumeral osteoarthritis — PASS

Management context is explicit:

```text
nonoperative primary GHOA
→ rep_glenohumeral_oa_nonoperative_v1
→ broad APTA best-practice/evidence-gap-aware sequence

preoperative TSA
→ rep_glenohumeral_oa_preop_TSA_v1
→ broad APTA best-practice/evidence-gap-aware sequence

postoperative arthroplasty
→ postoperative_shoulder_rehabilitation
→ procedure/patient-specific protocol owner
```

The 2023 APTA CPG reports no high/moderate-quality comparative evidence for nonoperative PT and no evidence-supported superior PT intervention. A 2026 systematic review confirms no nonsurgical PT RCTs through its June-2025 search. These are represented as `best_practice / insufficient_evidence`, not as low-certainty effect estimates.

Matching history prompts and fixtures exist for all three newly reviewed route groups.

Formal review:

```text
clinic_utilities/contracts/CU1_ROUTE_COVERAGE_REVIEW_2026-08-29.md
```

---

# 5. Current overall gate

```text
object/history/evidence-authority semantics       PASS
element-level evidence provenance                PASS
protocol override model                          PASS
tranche2 promotion                               PASS
tranche3 promotion                               PASS
shard integration                                PASS
native route-coverage gate for reviewed routes   PASS
logical-amendment gate                           PASS

routine-route evidence coverage                  FAIL
route-specific history prompt coverage           FAIL globally
route-complete fixture corpus                    FAIL globally
several route progression/evidence gaps          BLOCKED / EXPLICIT

FINAL RESULT                                     BLOCK
DESIGN-COMPLETE                                  NO
RUNTIME AUTHORIZED                               NO
```

The remaining block is route-content completeness, not shard integration.

---

# 6. Preserved evidence-gap behavior

```text
DGS → no validated disease-specific progression thresholds
De Quervain → no validated active progressive rehabilitation sequence
carpal tunnel → no validated CU-1 criteria-based PT sequence from reviewed authority
acute isolated meniscus → selected PT wording is consensus/clinical opinion; no validated staged sequence
adhesive capsulitis → current guidance exists; no universal validated phase progression
GHOA → PT may benefit by best practice; nonsurgical comparative efficacy remains unestablished
MDI → cautious framework may be described when selected; comparative exercise benefit/harm remains unknown
```

No generic MSK fallback is permitted.

---

# 7. Exact next authorized action

Continue only on the existing writer, route-by-route from the reconciled matrix:

```text
1. degenerative_meniscal_lesion_conservative_rehabilitation
2. patellar_tendinopathy
3. thumb_cmc1_osteoarthritis
4. cervical_routes
5. remaining_wrist_hand_and_elbow_routes
6. remaining routine routes in registry order
7. define reviewed evidence-gap behavior where full staging is unsupported
8. complete route-specific history prompts + matching fixtures alongside each route
9. rerun exact design-completeness review
10. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
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
USE evidence across a noncovered subtype or management context
USE posterior postoperative RTS evidence as nonoperative posterior authority
SILENTLY resolve conflicting ESWT frameworks
REPRESENT best-practice GHOA opinion as comparative treatment efficacy
IMPORT postoperative arthroplasty protocol into nonoperative/preoperative GHOA route
MERGE PR #63 merely because individual routes passed
OPEN CU-2
RESTART PR-1
```

---

# 9. Continuity rule

A fresh session must repeat the six-canonical bootstrap, verify the then-current remote `main` and PR #63 head, inspect this exact `BLOCK`, and continue only the route-coverage/history-prompt/fixture work on the existing CU-1 design writer unless the canonical lock changes.
