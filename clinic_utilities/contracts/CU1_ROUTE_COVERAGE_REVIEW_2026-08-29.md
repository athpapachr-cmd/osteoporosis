# CU-1 Route Coverage Review — 2026-08-29

> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`  
> **Runtime mutation:** not authorized and not performed.  
> **Route-coverage shards:** `clinic_utilities/contracts/cu1_evidence_route_coverage_v1.yaml`, `clinic_utilities/contracts/cu1_evidence_route_coverage_instability_v1.yaml`  
> **Logical amendments:** `clinic_utilities/contracts/cu1_evidence_route_coverage_amendments_v1.yaml`  
> **Review mode:** route-by-route exact evidence/applicability/output-scope review.

---

## Route 1 — Calcific rotator-cuff tendinopathy

Canonical route:

```text
calcific_rotator_cuff_tendinopathy
```

Frozen shoulder design requires an imaging/clinician-established calcific diagnosis, keeps ESWT optional rather than mandatory, records prior lavage/barbotage as context, and does not offer therapeutic ultrasound as a standard evidence-supported adjunct.

### Current sources reviewed

1. Desmeules et al. **Rotator Cuff Tendinopathy Diagnosis, Nonsurgical Medical Care, and Rehabilitation: A Clinical Practice Guideline.** J Orthop Sports Phys Ther. 2025;55(4):235-274. DOI `10.2519/jospt.2025.13182`.
2. NICE **Extracorporeal shockwave therapy for calcific tendinopathy in the shoulder**, HealthTech guidance HTG645, originally IPG742, published 2022-11-09 and migrated in January 2026 without recommendation change.

The 2025 CPG explicitly covers rotator-cuff tendinopathy with or without calcification and excludes full-thickness tears from this tendinopathy scope.

### Exact recommendation mapping

```text
patient-centered individualized education
→ Grade C
→ referral_core

active rehabilitation exercise as initial treatment
→ Grade A
→ referral_core

calcific lavage for refractory-to-initial-treatment calcific tendinopathy
→ Grade B
→ clinician_ui_only
→ not initial acute-treatment authority

ESWT for calcific tendinopathy — JOSPT 2025
→ Grade C may use/recommend
→ conflicts with NICE HTG645
→ clinician_ui_only until explicit framework context is resolved

ESWT for calcific tendinopathy — NICE HTG645
→ efficacy evidence inadequate
→ only in context of research
→ conflicts with JOSPT 2025
→ clinician_ui_only

laser for calcific tendinopathy
→ Grade C may use
→ clinician_ui_only because laser is not part of frozen selectable CU-1 adjunct taxonomy

therapeutic ultrasound for calcific tendinopathy
→ Grade C should not use/recommend
→ clinician_ui_only exclusion authority

severe persistent pain/disability despite appropriate nonsurgical care up to 12 weeks
→ Grade F specialist reassessment
→ referral_core reassessment criterion

return to sport capacity/load tolerance + patient-rated/functional measures
→ Grade F
→ optional athlete-specific progression authority
```

### Framework-conflict decision — ESWT

The route must **not** silently choose one framework.

```text
JOSPT 2025 CPG
→ ESWT may be used/recommended (Grade C)

NICE HTG645
→ efficacy evidence inadequate
→ research-only use
```

Therefore:

```text
ESWT != automatic route intervention direction
ESWT != silent guideline-unanimous adjunct
ESWT claims remain separate + mutually conflicting
future rendering requires explicit framework/clinician context
```

### Rehabilitation-sequence decision

A complete evidence-bounded sequence is supportable without inventing numeric thresholds:

```text
required phase
→ individualized education + active rehabilitation
→ no fabricated routine progression threshold
→ specialist reassessment if severe persistent pain/disability despite source-defined adequate nonsurgical-care window

optional athlete phase
→ shoulder/rotator-cuff capacity + load tolerance
→ patient-rated and functional performance measures
→ no elapsed-time-only return rule
```

No separate lavage/ESWT treatment phase is created.

### History prompts added

```text
imaging confirmation / tendon-location context
acute-irritable vs persistent/chronic course
prior initial treatment + response
prior lavage/barbotage
```

### Exact gate

```text
canonical route identity                         PASS
source freshness                                PASS
source/claim reference resolution               PASS
explicit payload IDs                            PASS
required profile fields                         PASS
required sequence fields                        PASS
route applicability                             PASS
full-thickness tear leakage                     PASS
output-scope compatibility                      PASS
framework conflict preservation                 PASS
no generic MSK fallback                         PASS
no invented progression threshold               PASS
route-specific history prompts                  PASS
matching fixtures                               PASS

CALCIFIC ROUTE PROFILE                           PASS
REHABILITATION SEQUENCE                          COMPLETE — EVIDENCE-BOUNDED
RUNTIME AUTHORIZED                              NO
```

---

## Route 2 — Glenohumeral instability / dislocation: initial rehabilitation split

Canonical route container:

```text
glenohumeral_instability_dislocation
```

The frozen shoulder design explicitly requires direction, traumatic/atraumatic context, first-time/recurrent status, current structural/specialist assessment, restrictions and sport/work demands. These variables cannot be collapsed into one generic instability protocol.

### Current sources reviewed

1. Alentorn-Geli et al. / ESSKA-ESA formal consensus. **Age- and time-specific management of traumatic anterior shoulder instability: The 2024 ESSKA-ESA Formal Consensus. Part 2: Treatment and return to sports.** Published 2026. DOI `10.1002/ksa.70497`.
2. Hurley et al. / Posterior Shoulder Instability International Consensus Group. **Posterior Shoulder Instability, Part I — Diagnosis, Nonoperative Management, and Labral Repair.** Arthroscopy. 2025;41(2):166-180.e11. DOI `10.1016/j.arthro.2024.04.035`.
3. Olds M, Uhl TL. **Current Clinical Concepts: Nonoperative Management of Shoulder Instability.** J Athl Train. 2024;59(3):243-254. DOI `10.4085/1062-6050-0468.22`.
4. Karasuyama et al. **Exercise for multidirectional instability of the shoulder.** Cochrane Database Syst Rev. 2026;2:CD015450. DOI `10.1002/14651858.CD015450.pub2`.
5. Housset et al. **Multidirectional instability of the shoulder: a systematic review with a novel classification.** EFORT Open Rev. 2024;9(4):285-296. DOI `10.1530/EOR-23-0029`.
6. Posterior Shoulder Instability Part II and its companion editorial were reviewed specifically to determine whether postoperative rehabilitation/RTS evidence could be applied to nonoperative posterior instability. It cannot.

### Material split

```text
A. traumatic anterior — first-time dislocation
B. traumatic anterior — recurrent instability / preoperative or explicitly selected rehabilitation
C. posterior instability — explicit nonoperative-management decision
D. atraumatic anterior instability — explicit nonoperative-management decision
E. multidirectional instability — explicit nonoperative-management decision
F. direction / cause / management context unresolved
G. postoperative instability rehabilitation
```

Behavior:

```text
A-E → branch-specific evidence profile / sequence
F   → block evidence-aware sequence until clarified
G   → postoperative_shoulder_rehabilitation with protocol precedence
```

### Traumatic anterior

First-time traumatic anterior dislocation is authorized for pain-controlled motion and progressive periscapular/rotator-cuff rehabilitation in the selected rehabilitation pathway, preserving the ESSKA-ESA Grade-D evidence level.

Recurrent traumatic anterior instability is separate. ESSKA-ESA generally recommends surgical management and notes that rehabilitation alone is unlikely to achieve stability in many recurrent cases. Therefore rehabilitation wording is allowed only in an explicit preoperative or selected conservative pathway and is not represented as definitive structural treatment.

Conservative traumatic-anterior RTS is narrowed to the source-specific context and requires pain-free ROM, clinical stability/no apprehension, adequate strength/endurance and sport-specific readiness rather than elapsed time alone.

### Posterior instability — nonoperative

Posterior Part I provides Level-V expert consensus for selection between nonoperative and operative management according to primary/recurrent status, symptoms/functional limitation, pathology and patient preference.

The 2024 Current Clinical Concepts paper supplies the lower-authority direction-specific rehabilitation framework for posterior nonoperative care.

### Posterior Part-II regression correction

The previously promoted Part-II RTS claim is postoperative authority. The companion editorial explicitly notes that conservative rehabilitation/RTS was not addressed.

Therefore:

```text
posterior Part-II RTS claim
!= nonoperative posterior-instability authority
```

It is suppressed from the generic nonoperative route and retained only as potential source material for future postoperative-route curation.

### Atraumatic anterior and MDI

Atraumatic anterior instability uses only a cautious direction-specific nonoperative clinical-concepts framework.

For MDI, the 2026 Cochrane review found no eligible control/usual-care RCTs. Hence:

```text
comparative exercise effect estimate = unavailable
certainty of a nonexistent comparative effect estimate = not_applicable
```

A cautious motor-control/co-contraction/dynamic-stability framework may be described only when nonoperative rehabilitation is explicitly selected; comparative efficacy remains uncertain.

### Exact gate

```text
canonical route identity                            PASS
first-time vs recurrent anterior separation        PASS
traumatic vs atraumatic separation                 PASS
anterior vs posterior vs MDI separation             PASS
management-context gating                          PASS
postoperative route separation                     PASS
anterior conservative RTS scope                    PASS after amendment
posterior Part-II postoperative leak                PASS after suppression
MDI no-RCT certainty semantics                     PASS after amendment
route-specific history prompts                     PASS
matching context-leakage fixtures                  PASS
no generic instability sequence                    PASS
no elapsed-time-only progression                   PASS

GLENOHUMERAL INSTABILITY ROUTE SPLIT                PASS
TRAUMATIC ANTERIOR SEQUENCE                         COMPLETE — EVIDENCE-BOUNDED / CONTEXT-SCOPED
POSTERIOR NONOPERATIVE SEQUENCE                     COMPLETE — EVIDENCE-BOUNDED / LOWER-AUTHORITY
ATRAUMATIC ANTERIOR SEQUENCE                        COMPLETE — EVIDENCE-BOUNDED / LOWER-AUTHORITY
MDI SEQUENCE                                        COMPLETE — CAUTIOUS WITH EXPLICIT EFFICACY UNCERTAINTY
UNRESOLVED DIRECTION/CAUSE/MANAGEMENT                BLOCK OUTPUT UNTIL CLARIFIED
POSTOPERATIVE INSTABILITY                            EXCLUDED FROM THIS ROUTE SPLIT
RUNTIME AUTHORIZED                                  NO
```

---

## Route 3 — Glenohumeral osteoarthritis

Canonical route:

```text
glenohumeral_osteoarthritis
```

Frozen shoulder design uses this route only for clinician-established GHOA with compatible clinical/imaging context. Rehabilitation wording is directed at pain/function, mobility/strength/activity where clinically useful and must not imply structural reversal.

### Current sources reviewed

1. Michener et al. / APTA. **Physical Therapist Management of Glenohumeral Joint Osteoarthritis: A Clinical Practice Guideline from the American Physical Therapy Association.** Phys Ther. 2023;103(6):pzad041. DOI `10.1093/ptj/pzad041`.
2. Muhammad DG, Foster NE, Pelaez M, O'Leary K, Ackerman IN, Quicke JG. **The effectiveness of physiotherapy-led non-surgical and perioperative interventions for glenohumeral osteoarthritis: A systematic review.** Shoulder Elbow. 2026 May 22; online ahead of print. DOI `10.1177/17585732261450961`; PMID 42186453.
3. AAOS continues to list the 2020 Glenohumeral Joint Osteoarthritis CPG as its current GHOA guideline. For PT-specific route design, the 2023 APTA CPG is the more directly applicable source.

### Freshness finding

The 2026 systematic review searched through June 2025 and found four eligible RCTs, all postoperative. It found **no published RCTs evaluating physiotherapist-led interventions in nonsurgical GHOA care**.

This confirms rather than closes the evidence gap documented in the 2023 APTA CPG.

### Material management-context split

```text
A. established primary GHOA — nonoperative management
B. established primary GHOA — scheduled for TSA / preoperative PT context
C. postoperative shoulder arthroplasty
D. management context not stated
```

Behavior:

```text
A
→ rep_glenohumeral_oa_nonoperative_v1
→ seq_glenohumeral_oa_nonoperative_v1

B
→ rep_glenohumeral_oa_preop_TSA_v1
→ seq_glenohumeral_oa_preop_TSA_v1

C
→ postoperative_shoulder_rehabilitation
→ procedure-specific/patient-specific protocol authority

D
→ history prompt requires management-context clarification before selecting A or B
```

### Nonoperative GHOA evidence posture

The APTA CPG states:

```text
high/moderate-quality comparative evidence = absent
aggregate included studies for nonoperative PT comparison = 0
```

Its best-practice position is that PT services **may benefit** patients with GHOA who have not undergone TSA.

It also states that no one specific PT intervention is established as superior. Intervention selection is intentionally individualized according to examination findings, impairments, tissue irritability, patient goals and values.

Therefore the CU-1 sequence is deliberately broad:

```text
one evidence-gap-aware nonoperative phase
→ individualized PT management
→ no evidence-superior named exercise/technique package
→ no invented frequency/duration
→ no universal numeric progression threshold
→ repeated patient-reported function/disability + clinical evaluation for reassessment
```

This is recorded as `best_practice_APTA / certainty=not_applicable`, not as low-certainty treatment-effect evidence.

### Preoperative TSA context

The APTA CPG similarly states as best practice, in the absence of high/moderate-quality evidence, that preoperative PT services may benefit postoperative outcomes.

The route therefore allows one broad preoperative phase without inventing a prehabilitation protocol.

It does **not** borrow postoperative TSA exercises, immobilization timing or strengthening restrictions into the preoperative route.

### Postoperative context boundary

The same APTA CPG contains stronger postoperative TSA evidence, including procedure/timing-sensitive recommendations. Those recommendations are not imported into `glenohumeral_osteoarthritis` because the frozen product has a separate:

```text
postoperative_shoulder_rehabilitation
```

owner.

Thus:

```text
postoperative TSA evidence
!= nonoperative GHOA authority
!= preoperative GHOA authority
```

Patient-specific surgeon/protocol restrictions retain precedence.

### Route-specific history prompts

Added prompts capture:

```text
management context: nonoperative / pre-op TSA / postoperative / not stated
symptom irritability and functional priority
prior nonoperative/PT care and response
imaging / glenoid-deformity / structural context when known
```

### Exact gate

```text
canonical route identity                         PASS
current PT-specific CPG identified               PASS
2026 evidence-gap freshness check                PASS
systematic-review source identity                PASS after amendment
nonoperative vs preoperative split               PASS
postoperative-owner separation                   PASS
best-practice opinion not relabelled as efficacy PASS
no specific PT superiority invented              PASS
no frequency/course duration invented            PASS
no universal progression threshold invented      PASS
route-specific history prompts                   PASS

GHOA NONOPERATIVE PROFILE                        PASS
GHOA NONOPERATIVE SEQUENCE                       COMPLETE — BEST-PRACTICE / EVIDENCE-GAP-AWARE
GHOA PREOPERATIVE TSA PROFILE                    PASS
GHOA PREOPERATIVE SEQUENCE                       COMPLETE — BEST-PRACTICE / EVIDENCE-GAP-AWARE
POSTOPERATIVE ARTHROPLASTY                       SEPARATE ROUTE OWNER
RUNTIME AUTHORIZED                              NO
```

### Remaining route-specific limitations

```text
no nonsurgical PT RCT evidence through June 2025 search
no evidence-supported superior PT intervention
no validated nonoperative staged progression thresholds
preoperative PT recommendation is best-practice opinion
postoperative TSA evidence requires separate procedure/protocol-scoped curation
```

These are explicit evidence limitations, not a reason to fabricate a generic shoulder-OA exercise protocol.

---

## Current route-coverage state after Routes 1–3

```text
calcific_rotator_cuff_tendinopathy
→ PASS / sequence_complete_evidence_bounded

glenohumeral_instability_dislocation
→ PASS as context-gated split

glenohumeral_osteoarthritis
→ PASS as management-context split
→ nonoperative + preoperative TSA broad best-practice sequences
→ postoperative context owned separately
```

## Next route

Per the reconciled work queue:

```text
degenerative_meniscal_lesion_conservative_rehabilitation
```
