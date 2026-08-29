# CU-1 Route Coverage Review — 2026-08-29

> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`  
> **Runtime mutation:** not authorized and not performed.  
> **Route-coverage shard:** `clinic_utilities/contracts/cu1_evidence_route_coverage_v1.yaml`  
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

## Next route

Per the reconciled work queue:

```text
glenohumeral_instability_dislocation_initial_rehabilitation_split
```

The next review must preserve direction, first-time/recurrent, age/bone-loss context, and operative-vs-nonoperative boundaries rather than producing a generic instability sequence.
