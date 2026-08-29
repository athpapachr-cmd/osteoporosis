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

This preserves the frozen shoulder design statement that ESWT may exist as a selectable calcific-specific adjunct without converting that product option into an automatic evidence recommendation.

### Rehabilitation-sequence decision

A complete **evidence-bounded** sequence is supportable without inventing numeric thresholds:

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

No separate lavage/ESWT treatment phase is created. Lavage is a refractory medical/procedural option; ESWT has a live framework conflict.

### History prompts added

```text
imaging confirmation / tendon-location context
acute-irritable vs persistent/chronic course
prior initial treatment + response
prior lavage/barbotage
```

Prompts remain non-inferential and are never auto-selected.

### Exact gate

```text
canonical route identity                         PASS
source freshness                                PASS
source/claim reference resolution               PASS
explicit payload IDs                            PASS
required profile fields                         PASS
required sequence fields                        PASS
route applicability                             PASS
full-thickness tear leakage                     PASS — excluded by source scope
output-scope compatibility                      PASS
framework conflict preservation                 PASS
no generic MSK fallback                         PASS
no invented progression threshold               PASS
route-specific history prompts                  PASS

CALCIFIC ROUTE PROFILE                           PASS
REHABILITATION SEQUENCE                          COMPLETE — EVIDENCE-BOUNDED
RUNTIME AUTHORIZED                              NO
CU-1 DESIGN-COMPLETE                            NO
```

### Remaining route-specific limitations

```text
no universal numeric rehabilitation progression thresholds
lavage efficacy evidence is for refractory/persistent cases rather than initial acute treatment
no evidence that post-lavage ESWT is automatically required or superior
ESWT guideline positions conflict
laser evidence does not itself authorize frozen-taxonomy expansion
```

These limitations are explicit and do not block the calcific route from being counted as a reviewed evidence-bounded route.

---

## Route 2 — Glenohumeral instability / dislocation: initial rehabilitation split

Canonical route container:

```text
glenohumeral_instability_dislocation
```

The frozen shoulder design explicitly requires direction, traumatic/atraumatic context, first-time/recurrent status, current structural/specialist assessment, restrictions and sport/work demands. These are material evidence-applicability variables; they cannot be collapsed into one generic instability protocol.

### Current sources reviewed

1. Alentorn-Geli et al. / ESSKA-ESA formal consensus. **Age- and time-specific management of traumatic anterior shoulder instability: The 2024 ESSKA-ESA Formal Consensus. Part 2: Treatment and return to sports.** Published 2026. DOI `10.1002/ksa.70497`.
2. Hurley et al. / Posterior Shoulder Instability International Consensus Group. **Posterior Shoulder Instability, Part I — Diagnosis, Nonoperative Management, and Labral Repair.** Arthroscopy. 2025;41(2):166-180.e11. DOI `10.1016/j.arthro.2024.04.035`.
3. Olds M, Uhl TL. **Current Clinical Concepts: Nonoperative Management of Shoulder Instability.** J Athl Train. 2024;59(3):243-254. DOI `10.4085/1062-6050-0468.22`.
4. Karasuyama et al. **Exercise for multidirectional instability of the shoulder.** Cochrane Database Syst Rev. 2026;2:CD015450. DOI `10.1002/14651858.CD015450.pub2`.
5. Housset et al. **Multidirectional instability of the shoulder: a systematic review with a novel classification.** EFORT Open Rev. 2024;9(4):285-296. DOI `10.1530/EOR-23-0029`.
6. Posterior Shoulder Instability Part II and its companion editorial were reviewed specifically to determine whether postoperative rehabilitation/RTS evidence could be applied to nonoperative posterior instability. It cannot.

### Material split

The route now resolves only after the clinically material context is known:

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
A-E
→ branch-specific evidence profile / sequence

F
→ block evidence-aware sequence until clarified

G
→ postoperative_shoulder_rehabilitation
→ patient-specific surgical protocol remains higher authority
```

### Traumatic anterior — first-time

ESSKA-ESA supports rehabilitation after first-time traumatic anterior dislocation regardless of whether surgery is planned, while acknowledging lack of high-level evidence for one exact regimen.

The evidence-bounded direction is:

```text
after immobilization / reduction context
→ pain-controlled passive ROM
→ gradual active-assisted exercise
→ when pain allows, periscapular + rotator-cuff strengthening
→ concomitant injury context must be considered
```

This is preserved as Grade D consensus and not upgraded to stronger evidence.

### Traumatic anterior — recurrent

Recurrent instability is materially different.

ESSKA-ESA generally recommends surgical management for recurrent traumatic anterior dislocation and states that rehabilitation alone is unlikely to provide stability in many recurrent cases, particularly contact/collision athletes.

Therefore the generated rehabilitation sequence is authorized only when there is an explicit:

```text
preoperative rehabilitation pathway
OR
selected conservative rehabilitation pathway
```

The evidence-bounded rehabilitation direction is pain-tolerated passive/active-assisted motion followed by proprioceptive and rotator-cuff, deltoid and periscapular strengthening. It must not imply that rehabilitation is definitive treatment for a recurrent structural-instability problem.

### Conservative traumatic-anterior return to sport

The previous tranche3 RTS claim required a scope correction.

ESSKA-ESA Question 23 is specifically about **conservatively treated** traumatic anterior instability. It supports return only after:

```text
full pain-free ROM
+ clinical stability / no apprehension
+ adequate strength and endurance
+ sport-specific readiness
```

The logical claim and optional RTS phase are therefore narrowed to conservative-treatment context. No time-only clearance is permitted.

### Posterior instability — nonoperative

Posterior Part I is Level-V expert consensus for diagnosis and selection between nonoperative and operative management. It supports individualized nonoperative consideration according to:

```text
primary vs recurrent instability
symptoms / functional limitation
underlying pathology
patient preference
```

It does **not** provide a complete comparative evidence-based rehabilitation protocol.

The 2024 Current Clinical Concepts paper supplies a direction- and impairment-specific nonoperative framework:

```text
posterior rotator-cuff / scapular motor control and strength
→ progressive dynamic stability
→ symptom-free functional milestones
→ higher-demand perturbation / sport tasks when ready
```

This is recorded as current clinical-concepts guidance, not as a high-strength CPG recommendation.

### Posterior Part-II regression correction

A material scope leak was identified during exact review.

The previously promoted `posterior_shoulder_instability_RTS_consensus` claim came from the 2025 Part-II Delphi paper. The companion editorial clarifies that Part II addresses **postoperative rehabilitation and return to play**, and that rehabilitation/RTS after conservative treatment was not addressed.

Therefore:

```text
posterior Part-II RTS claim
!= nonoperative posterior-instability authority
```

The logical amendment suppresses this claim from the generic `glenohumeral_instability_dislocation` nonoperative registry. The source remains available for later curation of `postoperative_shoulder_rehabilitation`.

### Atraumatic anterior instability

Atraumatic anterior instability selected for nonoperative care is not treated as traumatic dislocation.

Current clinical-concepts evidence supports a cautious direction-specific framework based on anterior rotator-cuff/scapular motor control, strength and progressive dynamic functional loading according to identified deficits.

No route-specific high-authority CPG or universal numeric progression threshold was identified; this limitation remains explicit.

### Multidirectional instability

MDI is explicitly separated from unidirectional traumatic instability.

The 2024 systematic review confirms substantial diagnostic/classification heterogeneity and supports the principle:

```text
generalized laxity alone != MDI diagnosis
symptomatic instability in >=2 directions + clinical context required
```

The 2026 Cochrane review found **no eligible randomized trials** comparing exercise with placebo, no treatment, waiting-list or usual-care controls through the review search period.

Therefore:

```text
comparative exercise efficacy estimate = unavailable
certainty grade for a nonexistent effect estimate = not_applicable
```

It would be incorrect to label this as a `very_low` effect estimate.

When a clinician has explicitly selected nonoperative rehabilitation, a cautious individualized motor-control, co-contraction/scapular-control and dynamic-stability framework may be described, but its comparative benefit must remain explicitly uncertain. No fabricated progression or RTS threshold is added.

### History prompts added

In addition to the existing direction/recurrence/traumatic-context prompt, the route now requires/supports:

```text
current management strategy
known bone loss / Bankart / Hill-Sachs / other structural context
sport/work/contact-collision/overhead demands
current restrictions after reduction or specialist assessment
```

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
no generic instability sequence                    PASS
no elapsed-time-only progression                   PASS
no invented numeric progression thresholds         PASS

GLENOHUMERAL INSTABILITY ROUTE SPLIT                PASS
TRAUMATIC ANTERIOR SEQUENCE                         COMPLETE — EVIDENCE-BOUNDED / CONTEXT-SCOPED
POSTERIOR NONOPERATIVE SEQUENCE                     COMPLETE — EVIDENCE-BOUNDED / LOWER-AUTHORITY
ATRAUMATIC ANTERIOR SEQUENCE                        COMPLETE — EVIDENCE-BOUNDED / LOWER-AUTHORITY
MDI SEQUENCE                                        COMPLETE — CAUTIOUS EVIDENCE-BOUNDED WITH EXPLICIT EFFICACY UNCERTAINTY
UNRESOLVED DIRECTION/CAUSE/MANAGEMENT                BLOCK OUTPUT UNTIL CLARIFIED
POSTOPERATIVE INSTABILITY                            EXCLUDED FROM THIS ROUTE SPLIT
RUNTIME AUTHORIZED                                  NO
CU-1 DESIGN-COMPLETE                                NO
```

### Remaining route-specific limitations

```text
recurrent traumatic anterior instability often has operative-management implications
posterior nonoperative detailed rehabilitation is not supported by high-level comparative evidence
atraumatic anterior evidence is clinical-concepts level rather than route-specific CPG authority
MDI comparative exercise benefit/harm remains unknown because eligible control RCTs were absent
bone loss / soft-tissue lesion / age / sport context may materially change the management decision
postoperative rehabilitation requires procedure-specific protocol authority
```

These limitations are explicit. They do not justify a generic instability fallback.

---

## Current route-coverage state after Routes 1–2

```text
calcific_rotator_cuff_tendinopathy
→ PASS / sequence_complete_evidence_bounded

glenohumeral_instability_dislocation
→ PASS as context-gated split
→ four evidence-bounded sequence branches
→ unresolved context blocks evidence-aware output
→ postoperative context routed separately
```

## Next route

Per the reconciled work queue:

```text
glenohumeral_osteoarthritis
```
