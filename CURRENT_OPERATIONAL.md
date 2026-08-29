# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 DESIGN GATE = `BLOCK`; SHARD INTEGRATION = `PASS`; ROUTE COVERAGE IN PROGRESS.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft / design-only / unmerged; reviewed route coverage is complete through matrix commit `795292aca8a526857fa6c24eb3cd2f1668cb91a5`; this canonical commit may advance branch head.
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
population-specific trial superiority != universal protocol superiority
```

No patient identifiers were added.

---

# 3. Normative evidence corpus

Manifest:

```text
clinic_utilities/contracts/cu1_evidence_manifest_v1.yaml
```

Active logical evidence layers now include:

```text
core_seed_registry
high_frequency_tranche2 + reviewed promotion projection
shoulder_hip_meniscus_tranche3 + reviewed projection/overlay
route_coverage_extension
shoulder_instability_route_coverage_extension
degenerative_meniscus_route_coverage_extension
patellar_tendinopathy_route_coverage_extension
cu1_evidence_route_coverage_amendments_v1.yaml
```

All listed shards have passed their native or reviewed schema/promotion gate. No staged evidence shard remains.

---

# 4. Reviewed native route coverage

## Calcific rotator-cuff tendinopathy — PASS

```text
rep_calcific_rotator_cuff_v1
seq_calcific_rotator_cuff_v1
→ sequence_complete — evidence-bounded
```

Core active rehabilitation is supported. ESWT remains an explicit JOSPT-2025-vs-NICE-HTG645 conflict and is not automatically rendered.

## Glenohumeral instability/dislocation split — PASS

No generic instability sequence exists. Traumatic anterior, posterior nonoperative, atraumatic anterior nonoperative and multidirectional-instability nonoperative contexts remain distinct. Unresolved direction/cause/management blocks evidence-aware output; postoperative instability belongs to the postoperative owner. Posterior Part-II postoperative RTS evidence is suppressed from nonoperative authority.

## Glenohumeral osteoarthritis — PASS

Nonoperative and preoperative-TSA contexts use broad APTA best-practice/evidence-gap-aware sequences. Postoperative arthroplasty belongs to `postoperative_shoulder_rehabilitation`. The absence of nonsurgical comparative PT RCTs is not relabelled as a low-certainty treatment effect.

## Degenerative meniscal lesion — PASS

```text
rep_degenerative_meniscus_conservative_v1
seq_degenerative_meniscus_conservative_v1
→ sequence_complete — evidence-bounded
```

The 2025 EU-US meniscus consensus supports nonoperative treatment including PT as first approach for symptomatic degenerative lesions and supports progressive ROM/strength/neuromuscular rehabilitation. ESCAPE 5-year and OMEX 10-year randomized follow-up support exercise-based management compared with arthroscopic partial meniscectomy in common degenerative tears.

Hard boundary:

```text
MRI tear != automatic symptom generator or surgery logic
clicking/catching != true locking
true locking / unresolved structural surgical indication -> block routine sequence + reassess
acute traumatic context -> acute-meniscus owner
postoperative context -> postoperative-knee owner
```

## Patellar tendinopathy — PASS

```text
rep_patellar_tendinopathy_v1
seq_patellar_tendinopathy_v1
→ sequence_complete — single-phase evidence-bounded
```

Current evidence is deliberately reconciled rather than flattened:

```text
Cochrane 2025
→ absolute strengthening effect remains low/very-low certainty by outcome/comparator
→ no universal high-certainty efficacy statement

2026 exercise NMA
→ no clinically meaningful superiority hierarchy among contemporary loading strategies
→ HSR reasonable reference, not mandatory protocol

Breda 2021 PTLE RCT
→ population-specific PTLE signal vs eccentric-only
→ not universal PTLE-superiority authority
```

Therefore progressive tendon/quadriceps loading may be represented as the broad conservative rehabilitation direction, but eccentric/isometric/HSR/PTLE choice and dosing remain therapist execution detail. No universal numeric progression or return-to-sport threshold is rendered. ESWT is not auto-recommended.

Matching history prompts and fixtures exist for all five reviewed native route groups.

Formal route reviews:

```text
clinic_utilities/contracts/CU1_ROUTE_COVERAGE_REVIEW_2026-08-29.md
clinic_utilities/contracts/CU1_DEGENERATIVE_MENISCUS_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/CU1_PATELLAR_TENDINOPATHY_ROUTE_REVIEW_2026-08-29.md
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
patellar tendinopathy → progressive loading direction may be used cautiously; no mandatory loading mode or validated universal numeric RTS threshold
```

No generic MSK fallback is permitted.

---

# 7. Exact next authorized action

Continue only on the existing writer, route-by-route from the reconciled matrix:

```text
1. thumb_cmc1_osteoarthritis
2. cervical_routes
3. remaining_wrist_hand_and_elbow_routes
4. remaining routine routes in registry order
5. define reviewed evidence-gap behavior where full staging is unsupported
6. complete route-specific history prompts + matching fixtures alongside each route
7. rerun exact design-completeness review
8. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
```

A small reconciliation cleanup remains allowed inside the same scope: the internal `review_record` metadata in `cu1_evidence_route_coverage_meniscus_v1.yaml` must be aligned with its already-authoritative dedicated review file without changing clinical content.

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
FREEZE eccentric / isometric / HSR / PTLE as a universal patellar-tendinopathy physician protocol
IMPORT older expert numeric patellar RTS/pain thresholds as current validated clearance rules
MERGE PR #63 merely because individual routes passed
OPEN CU-2
RESTART PR-1
```

---

# 9. Continuity rule

A fresh session must repeat the six-canonical bootstrap, verify the then-current remote `main` and PR #63 head, inspect this exact `BLOCK`, and continue only the route-coverage/history-prompt/fixture work on the existing CU-1 design writer unless the canonical lock changes.
