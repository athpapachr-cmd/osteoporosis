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
> **Current runtime writer:** PHYSIO coordinator on `feat/physio-r2-ia-correction-2026-10-05`, limited to approved Knee-OA correction seams; root lock in `CURRENT_OPERATIONAL.md`. R2-A pre-code closure PASS and final independent post-code closure PASS at `f5c99fc` are recorded. Draft PR #135 has a bounded CI correction awaiting independent review/new-head CI; merge/deploy HOLD, live release unchanged.
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

**R2 correction candidate, 2026-10-05:** The approved four-section physician UI, visible standalone Οίδημα, Functionality ownership, retirement of routine pain map/tenderness/crepitus/separate effusion, Proposed Plan walking-aid additional option and product-local combination review cue are implemented on the bounded branch. The post-code review chain ended with **PASS / COMPLETE_FOR_DECLARED_SCOPE / no new material finding** at `f5c99fcd34117e8bb1488e7b654db2f2448797d3` after two bounded corrections. Five Product Owner synthetic mobile replays, protected Cockpit/CY_GESY browser smoke, deterministic-output and focused CU-1 regression pass locally. Draft [PR #135](https://github.com/athpapachr-cmd/osteoporosis/pull/135) is open; first CI exposed a bounded PHYSIO decision-pending interaction and isolated prototype dependency mismatch. Their correction passes local 6/6 and 4/4 browser checks but awaits independent delta review and new-head CI. `programme/PHYSIO/P1_R2_PR_CI_TRIAGE.md` records ownership and limits. This is not a production release or new real-use acceptance; merge/deploy remain in HOLD.

The current released Knee-OA refinement lifecycle is closed. Product Owner real-use feedback has authorized a **separate bounded structural correction**, recorded in `programme/PHYSIO/P1_R2A_STRUCTURAL_CORRECTION.md` and coordinated by `programme/PHYSIO/CURRENT.md`. The three R2-A decisions are approved: Functionality owns duplicate functional concepts; routine structured pain map, tenderness, crepitus and separate effusion leave the physician UI while visible standalone Οίδημα remains optional and unselected by default; walking-aid assessment/training sits only in Proposed Plan additional options, default-off. The independent R2-A delta + affected-cumulative pre-code closure returned PASS at frozen design commit `fdb6bff3b852112e8fdfa35a2db5dc8fd8f82eca`; R2-B semantic PASS is reused. Bounded implementation writer scope is claimed; no new release or production behavior is claimed here. Real pilot/commercial validation and any second diagnosis remain separate boundaries.

Permanent utility rule:

```text
CLINICALLY INTERESTING != WORKFLOW-USEFUL != RECEIVER-USEFUL != WORTH ADDING
PRODUCT OWNER REQUEST != EVIDENCE != IMPLEMENTATION AUTHORITY
EXTERNAL FEEDBACK != AUTOMATIC IMPLEMENTATION AUTHORITY
```
