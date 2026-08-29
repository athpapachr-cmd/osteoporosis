# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 DESIGN GATE = `BLOCK`; SHARD INTEGRATION = `PASS`; ROUTE COVERAGE IN PROGRESS.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft / design-only / unmerged; route-coverage reconciliation is complete through matrix commit `2a51b4a2824e38ed171cda7a90953e41d4ed1dbf`; this canonical commit may advance branch head.
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
expert consensus / clinical opinion != low-certainty trial evidence
no eligible comparative trials != very-low effect estimate
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
→ ACTIVE DESIGN AUTHORITY

high_frequency_tranche2
→ ACTIVE DESIGN AUTHORITY
→ reviewed promotion projection

shoulder_hip_meniscus_tranche3
→ ACTIVE DESIGN AUTHORITY
→ reviewed promotion projection + mandatory overlay

route_coverage_extension
→ ACTIVE DESIGN AUTHORITY
→ native explicit-ID reviewed routes

shoulder_instability_route_coverage_extension
→ ACTIVE DESIGN AUTHORITY
→ native explicit-ID context-scoped route objects

cu1_evidence_route_coverage_amendments_v1.yaml
→ reviewed logical narrowing/suppression layer
→ applied after active-shard merge and before authority resolution
```

All listed shards have passed their native or reviewed schema/promotion gate. No staged evidence shard currently remains.

---

# 4. Newly completed route coverage

## 4.1 Calcific rotator-cuff tendinopathy — PASS

```text
profile: rep_calcific_rotator_cuff_v1
sequence: seq_calcific_rotator_cuff_v1
status: sequence_complete — evidence-bounded
```

Current evidence posture:

```text
active rehabilitation exercise
→ JOSPT 2025 Grade A
→ referral core

individualized education
→ Grade C
→ referral core

lavage
→ Grade B for refractory-to-initial-treatment calcific tendinopathy
→ clinician-facing procedural context, not an initial rehab phase

therapeutic ultrasound
→ Grade C do-not-use/recommend

ESWT
→ JOSPT 2025: Grade C may use/recommend
→ NICE HTG645: efficacy evidence inadequate; research-only
→ explicit framework conflict
→ NOT auto-rendered as unanimous adjunct
```

Matching route-specific history prompts and ESWT-conflict fixtures were added.

## 4.2 Glenohumeral instability/dislocation split — PASS

No generic instability sequence exists.

Reviewed branches:

```text
traumatic anterior — first-time / recurrent context-scoped
posterior — explicit nonoperative-management decision
atraumatic anterior — explicit nonoperative-management decision
multidirectional instability — explicit nonoperative-management decision
```

Behavior:

```text
unresolved direction / cause / management context
→ block evidence-aware sequence until clarified

postoperative instability rehabilitation
→ postoperative_shoulder_rehabilitation
→ patient-specific surgical protocol has precedence
```

Material evidence corrections:

```text
anterior RTS
→ ESSKA-ESA claim narrowed to conservative-treatment context

posterior Part-II Delphi RTS
→ identified as postoperative rehabilitation/RTS authority
→ suppressed from generic nonoperative posterior authority

MDI Cochrane 2026
→ no eligible control/usual-care RCTs
→ efficacy estimate unavailable
→ NOT mislabeled as very-low effect evidence
```

Matching context-leakage fixtures were added.

Formal route review:

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

Examples remain explicit:

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

adhesive capsulitis
→ current 2025 guidance exists
→ no single validated universal phase progression

glenohumeral OA
→ PT guidance remains consensus-level without route-specific progression sequence

MDI
→ cautious nonoperative framework may be described when selected
→ comparative exercise benefit/harm remains unknown from control RCTs
```

No generic MSK fallback is permitted.

---

# 7. Exact next authorized action

Continue only on the existing writer, route-by-route from the reconciled matrix:

```text
1. glenohumeral_osteoarthritis
2. degenerative_meniscal_lesion_conservative_rehabilitation
3. patellar_tendinopathy
4. thumb_cmc1_osteoarthritis
5. cervical_routes
6. remaining_wrist_hand_and_elbow_routes
7. remaining routine routes in registry order
8. define reviewed evidence-gap behavior where full staging is unsupported
9. complete route-specific history prompts + matching fixtures alongside each route
10. rerun exact design-completeness review
11. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
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
MERGE PR #63 merely because individual routes passed
OPEN CU-2
RESTART PR-1
```

---

# 9. Continuity rule

A fresh session must repeat the six-canonical bootstrap, verify the then-current remote `main` and PR #63 head, inspect this exact `BLOCK`, and continue only the route-coverage/history-prompt/fixture work on the existing CU-1 design writer unless the canonical lock changes.
