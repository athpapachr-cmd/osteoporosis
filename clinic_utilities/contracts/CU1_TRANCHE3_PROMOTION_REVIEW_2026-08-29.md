# CU-1 Tranche3 Evidence Promotion Review — 2026-08-29

> **Scope:** exact promotion review of `cu1_evidence_tranche3_v1.yaml` only.
> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **Runtime mutation:** not authorized and not performed.
> **Promotion artifacts:** `cu1_evidence_tranche3_promotion_v1.yaml` followed by mandatory `cu1_evidence_tranche3_promotion_fix_v1.yaml`.

## 1. Decision

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
CU-1 DESIGN-COMPLETE                          NO
RUNTIME AUTHORIZED                            NO
```

Promotion means only that the reviewed logical tranche3 corpus may participate in the design-time evidence registry. It does **not** mean that every represented route has a complete rehabilitation sequence or that evidence-aware runtime generation is authorized.

## 2. Normalization

The immutable staging shard uses YAML map keys as identities and omits fields that the frozen v1.1 object contract expects on normalized logical objects.

The promotion projection therefore deterministically materializes:

```text
source map key  -> EvidenceSource.evidence_id
claim map key   -> EvidenceClaim.claim_id
prompt map key  -> RouteHistoryPrompt.prompt_id
profile map key -> RouteEvidenceProfile.route_evidence_profile_id
sequence map key -> RehabilitationSequence.sequence_id
```

It also supplies the required empty claim collections on route profiles and the required sequence-source / clinician-modification / protocol-override collections on rehabilitation sequences.

The staging source remains unchanged for audit history.

## 3. Material clinical/evidence corrections required before promotion

### 3.1 Acute isolated meniscus — consensus is not low-certainty trial evidence

The AAOS 2024 acute isolated meniscus guideline characterizes the relevant nonoperative PT statement as workgroup clinical opinion and the displaced/displacing tear with ROM restriction statement as consensus/clinical judgment. The staging shard incorrectly represented these as ordinary low-certainty claims.

Promotion therefore replaces them with framework-faithful claims:

```text
acute_meniscus_PT_consensus_nonoperative
acute_meniscus_displaced_ROM_block_consensus_reassessment
```

with:

```text
strength = consensus_AAOS
certainty = not_applicable
```

The routine sequence is condition-scoped to:

```text
acute isolated
+ non-displaced
+ not a repair candidate
+ nonoperative management selected
```

A displaced/displacing tear, true mechanical block/locked knee or repair-candidate context must not receive that routine sequence.

### 3.2 Acute meniscus output-scope leak removed

The staging sequence used a `clinician_ui_only` safety claim as rendered `safety_escalation` authority. That violates the frozen element-level output-scope contract.

Promotion replaces it with an explicitly referral-compatible consensus safety/reassessment claim. No clinician-UI-only claim remains rendered authority.

### 3.3 Nonarthritic hip — synthetic strength removed

The 2023 nonarthritic-hip CPG distinguishes recommendation strength by intervention. The staging claim `C_or_lower_context` was not a valid framework grade.

Promotion preserves the exact relevant levels:

```text
multimodal management, especially FAIS/labral presentations -> Grade B
movement-pattern training when movement dysfunction exists   -> Grade C
therapeutic exercise for identified mobility/flexibility/strength deficits -> Grade C
```

No synthetic combined grade survives the logical merge.

### 3.4 Nonarthritic hip — unsupported progression inference removed

The staging sequence converted repeated objective measures into a route-wide progression criterion. The reviewed 2023 CPG supports examination and impairment/performance measures but does not establish a validated universal progression threshold; published progression criteria are limited.

That rendered progression criterion is therefore removed. The sequence remains an evidence-bounded, single-phase rehabilitation direction with an explicit progression gap.

### 3.5 Nonarthritic hip — FAIS and instability precautions separated

The staging shard combined two materially different cautions into one route-wide rendered statement.

Promotion splits them into condition-scoped claims:

```text
FAIS / impingement presentation
-> avoid repeatedly symptom-provoking impingement ranges

hip-instability context
-> avoid excessive capsuloligamentous stress
```

The rehabilitation sequence is not automatically applied to `other_established_nonarthritic_intraarticular_hip_condition` without a reviewed applicability match.

### 3.6 Greater-trochanteric route — subtype boundary hardened

The frozen hip profile uses exact subtype identity:

```text
formal_GTPS_diagnosis
formal_gluteal_tendinopathy_diagnosis
formal_trochanteric_bursitis_diagnosis
other_established_greater_trochanteric_disorder
```

The staging typo `formal_gtps_diagnosis` is not canonical and is corrected by the mandatory promotion overlay.

The evidence is also separated by actual population:

```text
formal GTPS
-> 2024 GTPS systematic-review exercise authority

formal gluteal tendinopathy
-> 2025 gluteal-tendinopathy exercise + education authority
```

Neither profile nor sequence is evidence authority for isolated formal trochanteric bursitis or another established greater-trochanteric disorder.

### 3.7 Posterior shoulder instability — expert consensus preserved as expert consensus

The reviewed 2025 posterior-instability literature consists of international expert Delphi consensus statements. Promotion adds Part I for nonoperative-management indication context and retains Part II for rehabilitation/return-to-play criteria.

The corpus does not relabel consensus as low/very-low trial certainty and does not manufacture one universal initial rehabilitation sequence.

### 3.8 Traumatic anterior instability — scope remains narrow

The 2024 ESSKA-ESA formal consensus Part 2, published in 2026, supports criteria-based return-to-sport assessment involving pain-free ROM, stability, strength and sport-specific readiness rather than elapsed time alone.

This authority remains limited to traumatic anterior instability and must not be generalized to posterior or multidirectional instability. First-time/recurrent status, age, bone loss, soft-tissue context and operative/nonoperative management remain material.

### 3.9 Adhesive capsulitis — freshness gap corrected

The staging shard stated that no current high-authority route-specific CPG had been identified. That statement is no longer current.

Promotion adds current 2025 frameworks:

```text
Clinical Practice Guidelines for Diagnosis and Non-Surgical Treatment of Primary Frozen Shoulder
British Elbow and Shoulder Society patient care pathway: Frozen shoulder
```

The evidence model now preserves:

```text
primary frozen shoulder manual therapy / ROM exercise -> may be considered
self-stretching -> therapist execution detail / expert consensus
strengthening -> insufficient evidence for routine recommendation
BESS supervised physiotherapy vs natural history -> uncertainty preserved
physiotherapy after injection in BESS pathway -> conditional context
```

These newer sources **do not** justify one universal criteria-based progression sequence. That evidence gap remains explicit.

### 3.10 Full-thickness rotator-cuff tear — benefit does not equal protocol

AAOS 2025 supports physical therapy for patient-reported outcomes in symptomatic full-thickness rotator-cuff tears selected for nonoperative management, while also noting that structural progression may occur over longer follow-up.

The promoted sequence therefore remains deliberately single-phase/evidence-bounded. No universal PT progression protocol is invented from the CPG.

## 4. Source-identity/freshness corrections

Promotion hardens bibliographic identity for:

```text
nonarthritic hip CPG 2023
GTPS systematic review 2024
gluteal tendinopathy systematic review 2025
AAOS acute isolated meniscus CPG 2024
ESSKA-ESA traumatic anterior instability consensus published 2026
posterior instability consensus Parts I and II 2025
adhesive capsulitis systematic review 2023
```

The 2025 frozen-shoulder sources are added as current active evidence sources. No superseded source is presented as current guideline authority.

## 5. Promotion read order

The normative design-time read for tranche3 is:

```text
cu1_evidence_tranche3_v1.yaml
-> cu1_evidence_tranche3_promotion_v1.yaml
-> cu1_evidence_tranche3_promotion_fix_v1.yaml
-> schema/reference/applicability/output-scope/freshness checks
-> logical registry merge
```

The mandatory final overlay exists because the second exact pass found two applicability defects after the first projection: canonical GTPS subtype casing and acute-meniscus sequence-level non-displaced/non-repair-candidate scoping.

## 6. What promotion does not solve

The following remain explicit design limitations rather than silent fallbacks:

```text
full-thickness cuff -> no universal criteria-based PT progression protocol
nonarthritic hip -> no validated route-wide progression thresholds; other-established subtype needs applicability review
GTPS/gluteal tendinopathy -> no exact progression thresholds; bursitis/other subtype not covered
acute isolated meniscus -> PT authority is consensus-only; no validated staged progression sequence
traumatic anterior shoulder instability -> no route-wide initial rehab sequence across management contexts
posterior shoulder instability -> expert consensus; no universal initial nonoperative sequence
adhesive capsulitis -> current guidelines exist, but no single validated criteria-based phase sequence
```

Therefore tranche3 can be promoted as evidence authority while the overall CU-1 design gate remains blocked on route completeness and the broader coverage matrix.

## 7. Final classification

```text
TRANCHE3 NORMALIZATION/REVIEW/PROMOTION = PASS
TRANCHE3 ACTIVE DESIGN AUTHORITY       = YES
ROUTE COVERAGE GATE                    = NOT CLOSED
DESIGN-COMPLETE                        = NO
RUNTIME EVIDENCE GENERATION            = NOT AUTHORIZED
```
