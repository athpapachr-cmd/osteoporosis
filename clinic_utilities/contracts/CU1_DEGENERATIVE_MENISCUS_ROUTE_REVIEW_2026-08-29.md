# CU-1 Degenerative Meniscus Route Coverage Review — 2026-08-29

> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`  
> **Runtime mutation:** not authorized and not performed.  
> **Route shard:** `clinic_utilities/contracts/cu1_evidence_route_coverage_meniscus_v1.yaml`  
> **Review mode:** exact source / applicability / output-scope / progression / safety review.

## Decision

```text
canonical route identity                         PASS
frozen knee-profile consistency                 PASS
current route-specific consensus                PASS
long-term randomized evidence                   PASS
acute-vs-degenerative separation                PASS
postoperative-owner separation                  PASS
true-locking / structural-exit behavior         PASS
source and claim reference resolution           PASS
explicit payload IDs                            PASS
required profile and sequence fields            PASS
output-scope compatibility                      PASS
criterion-based progression                     PASS
no surgery-derived routine timeline             PASS
no generic knee/MSK fallback                    PASS
route-specific history prompts                  PASS

DEGENERATIVE MENISCUS PROFILE                   PASS
REHABILITATION SEQUENCE                         COMPLETE — EVIDENCE-BOUNDED
RUNTIME AUTHORIZED                              NO
CU-1 DESIGN-COMPLETE                            NO
```

## Route and frozen boundary

Canonical route:

```text
degenerative_meniscal_lesion_conservative_rehabilitation
```

The frozen knee profile requires conservative rehabilitation for common symptomatic degenerative meniscal lesions when no unresolved structural surgical indication is present. It also preserves:

```text
MRI tear != automatically symptomatic pain generator
clicking/catching != true locked knee
true locking or unresolved structural concern -> exit routine route
acute isolated traumatic meniscus != degenerative route
postoperative meniscus rehabilitation != degenerative route
```

## Current sources reviewed

1. Prill et al. / ESSKA-AOSSM-AASPT initiative. **The formal EU-US Meniscus Rehabilitation 2024 Consensus: Part II — Prevention, non-operative treatment and return to sport.** Knee Surg Sports Traumatol Arthrosc. 2025. DOI `10.1002/ksa.12689`.
2. Noorduyn et al. / ESCAPE Research Group. **Effect of Physical Therapy vs Arthroscopic Partial Meniscectomy in People With Degenerative Meniscal Tears: Five-Year Follow-up of the ESCAPE Randomized Clinical Trial.** JAMA Netw Open. 2022;5(7):e2220394. DOI `10.1001/jamanetworkopen.2022.20394`.
3. Berg et al. / OMEX trial investigators. **Arthroscopic partial meniscectomy versus exercise therapy for degenerative meniscal tears: 10-year follow-up of the OMEX randomised controlled trial.** Br J Sports Med. 2025;59(2):91-98. DOI `10.1136/bjsports-2024-108644`.

The 2025 EU-US consensus is the current route-specific rehabilitation framework. ESCAPE and OMEX provide long-term randomized comparative support for exercise-based rehabilitation versus arthroscopic partial meniscectomy in common degenerative tears.

## Exact recommendation mapping

```text
nonoperative treatment including PT as first approach
-> Grade A consensus
-> referral_core

ROM + progressive knee/hip strength + neuromuscular training
-> Grade B
-> referral_core

manual therapy / joint mobilisation when clinically indicated
-> contained within Grade B rehabilitation options
-> not mandatory

supervised rehabilitation + home exercise
-> Grade D
-> therapist_execution_detail
-> no claim of superiority over home-only care

criterion-based monitoring using ROM, effusion, objective strength and functional performance
-> Grade D
-> referral_core progression/reassessment framework

persistent pain, recurrent stiffness/effusion, instability, mechanical symptoms or unexpected neurological symptoms
-> Grade C orthopaedic reassessment
-> referral_core safety/reassessment

failure to reach clinical milestones because of persistent knee symptoms
-> Grade D orthopaedic reassessment
-> referral_core
```

## Long-term randomized evidence

ESCAPE randomized 321 adults aged 45–70 years with degenerative meniscal tears. At 5 years, exercise-based physical therapy remained noninferior to arthroscopic partial meniscectomy for patient-reported knee function, with comparable radiographic OA progression.

OMEX followed 140 participants with degenerative meniscal tears for 10 years. It found no clinically relevant differences in patient-reported outcomes or isokinetic strength between arthroscopic partial meniscectomy and exercise therapy and no meaningful difference in radiographic OA progression.

These trials support the consensus first-line rehabilitation direction. They do not imply that every structural meniscal lesion is suitable for routine conservative management.

## Rehabilitation sequence

The route supports an evidence-bounded two-level sequence without fabricated thresholds.

### Required functional-restoration phase

```text
progressive knee + hip strengthening
+ ROM restoration when restricted
+ neuromuscular control
+ progressive functional loading
```

Progression is based on clinical response and objective knee function, including ROM, effusion, strength and appropriate functional performance. No elapsed-time-only transition is authorized.

### Optional high-demand / sport phase

When high-demand work or sport return is an explicit goal, progression may consider:

```text
subjective + objective knee function
ROM
joint effusion
quadriceps/hamstring strength
coordination/stability performance
psychological readiness
```

The consensus includes surgery-specific healing timelines elsewhere; those timelines are **not** imported into conservative degenerative-meniscus rehabilitation.

## Safety and applicability boundary

The route is not selected merely because MRI shows a degenerative tear.

```text
true locking / unresolved mechanical block
OR other unresolved structural surgical indication
-> block routine evidence-aware sequence
-> clinical / orthopaedic reassessment

acute isolated traumatic tear
-> use acute_isolated_meniscal_injury_nonoperative only if its own applicability conditions are met

post-meniscal surgery
-> postoperative_knee_rehabilitation
-> procedure-specific restrictions / protocol precedence
```

Persistent mechanical symptoms during rehabilitation also trigger reassessment rather than indefinite progression.

## History prompts

The route adds non-inferential prompts for:

```text
true locking distinct from clicking/catching
recurrent/persistent effusion
MRI location/morphology when known
knee-OA overlap
prior conservative rehabilitation and response
work/sport/activity goal
```

## Evidence gaps retained

```text
no reliable prognostic classifier identifies who will fail rehabilitation
higher OA grade, BMI and symptom duration may influence outcome but are not deterministic selectors
no universal numeric progression thresholds
supervised outpatient vs home-only rehabilitation has not been directly compared
MRI degenerative tear alone does not identify the symptom generator
```

## Final route state

```text
rep_degenerative_meniscus_conservative_v1
-> PASS

seq_degenerative_meniscus_conservative_v1
-> COMPLETE — EVIDENCE-BOUNDED

true_locking_or_unresolved_structural_surgical_indication
-> BLOCK ROUTINE SEQUENCE / REASSESS

acute traumatic or postoperative context
-> DIFFERENT ROUTE OWNER
```

The next queue item after fixture + manifest/matrix reconciliation is `patellar_tendinopathy`.
