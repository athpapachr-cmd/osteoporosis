# CURRENT.md — Physio Referral commercial product track

> **STATUS:** KNEE-OA V5.1 + POST-USE RECEIVER REFINEMENTS RELEASED / LIVE / AUTHENTICATED PRODUCTION-SMOKE-VERIFIED; `CY_GESY` ACTIVE.
> **Updated:** 2026-09-15 Asia/Nicosia.
> **Diagnosis vertical:** Knee Osteoarthritis only.
> **Current main release commit:** `b962485c741f558121e8daabfcf1d20c84f31f63`.
> **V5.1 release PR:** `#106`; merge commit `8064999ea70e0a90f6073fc6d66a0c8caaba1538`.
> **Post-use Cyprus / clinical-review PR:** `#107`; merge commit `a19f4b9d52c3076f715fd864a5f7664d2b140c81`.
> **Receiver prose / chronicity PR:** `#109`; merge commit `b962485c741f558121e8daabfcf1d20c84f31f63`.
> **Authenticated post-use live smoke:** `34957778592` — SUCCESS.
> **Authenticated receiver-refinement live smoke:** `35019200920` — SUCCESS.
> **Production context:** protected Clinical Excellence Cockpit with explicit `CY_GESY` server-side configuration.
> **Current writer:** none.
> **Root operational authority:** `CURRENT_OPERATIONAL.md` continues to track the parallel Medical Report lifecycle and is intentionally not rewritten by this product closeout.

## Current production state

The Knee-OA physiotherapy-referral product is live inside the authenticated Clinical Excellence Cockpit. The current release chain is:

```text
V5 released / smoke-verified
→ V5.1 examination + evidence-source UX released
→ Cyprus / clinical-review post-use refinement released
→ receiver prose compression + optional symptom chronicity released
```

Current production architecture remains:

```text
real CU-1 validation / safety
+ deterministic Knee-OA projection
+ international evidence core
+ reviewed CY_GESY jurisdiction overlay
+ V5 / V5.1 progressive UI
+ protected Clinical Excellence transport
```

## Current live behavior

The released workflow preserves the V5 first-tap / second-tap interaction and adds the V5.1 examination/detail layer, including directional knee weakness, progressive ROM detail, crepitus, expanded tenderness localization and objective stability findings.

The later post-use refinement keeps atypical observations separate from unresolved safety concerns: observations such as recent trauma, rapid worsening/deformity or a hot/swollen joint are review clues and do not themselves infer fracture, septic arthritis, SIFK/SONK, a second diagnosis or automatic imaging. Blocking behavior remains attached only to explicit unresolved safety concerns.

The current receiver-side refinement additionally:

- compresses duplicated functional-retraining wording to `λειτουργική επανεκπαίδευση με έμφαση στις καταγεγραμμένες λειτουργικές δυσχέρειες`;
- adds optional `Χρονιότητα συμπτωμάτων` under `Περισσότερα → Περιορισμοί & σημείωση` as value `1..99` plus `εβδομάδες / μήνες / έτη`;
- renders chronicity as short context such as `Συμπτωματολογία διάρκειας 8 μηνών.`;
- keeps chronicity context-only: it does not select treatment, change evidence, trigger imaging, create a review clue or alter safety gating;
- adds no structured previous-physiotherapy or response field;
- adds no new ADL / patient-centred boilerplate.

Permanent receiver-history boundary:

```text
ADMINISTRATIVE PHYSIO ACTIVITY
!= KNOWN TREATMENT PROGRAM
!= KNOWN RESPONSE
```

A GeSY `PHYS01` administrative record therefore must not be converted into a structured claim about the programme delivered or its clinical response. Meaningful known history may remain clinician-entered free text.

## Verification

PR #109 exact-head validation passed all seven protected Physio/Cockpit gates before merge. The authenticated live smoke `35019200920` then verified the production bootstrap and project path, `deployment_context == clinical_excellence_cockpit`, explicit `CY_GESY`, the new chronicity qualifier/output, the compressed functional-retraining wording, absence of structured prior-physiotherapy/response UI, and the no-browser-storage boundary.

The smoke used only a generated UUID and non-identifiable synthetic state; the protected credential remained masked.

The preceding PR #107 post-use production state was independently smoke-verified by run `34957778592` — SUCCESS.

## Preserved boundaries

Unchanged:

- Knee Osteoarthritis only;
- no second diagnosis;
- no automatic imaging recommendation;
- no SIFK/SONK inference;
- no international evidence-state reclassification;
- no jurisdiction-driven treatment selection or referral-text rewrite;
- no default-plan change;
- no patient/referral browser persistence;
- no billing/analytics/entitlement expansion;
- no Medical Report mutation;
- frozen CU-1 safety taxonomy remains unchanged.

## Current lifecycle

```text
V5 baseline release                         yes
V5.1 release                                yes
post-use Cyprus / clinical-review release   yes
receiver/chronicity refinement release      yes
authenticated current live smoke            pass — 35019200920
real clinical pilot                         no
commercial validation                       no
second diagnosis                            no
active writer                               none
```

## Next product boundary

The current Knee-OA refinement lifecycle is closed. Product Owner real-use feedback may justify another small bounded refinement, a deliberate pilot/commercial-validation step or a separately authorized next diagnosis. None is automatically authorized by this closeout.

Permanent utility rule:

```text
CLINICALLY INTERESTING != WORKFLOW-USEFUL != RECEIVER-USEFUL != WORTH ADDING
PRODUCT OWNER REQUEST != EVIDENCE != IMPLEMENTATION AUTHORITY
EXTERNAL FEEDBACK != AUTOMATIC IMPLEMENTATION AUTHORITY
```
