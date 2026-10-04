# PHYSIO CURRENT — Knee-OA Reference Implementation / Physio Core Boundary

> **TASK:** `PHYSIO-P1-KNEE-OA-REFERENCE-VALIDATION-CORE-BOUNDARY-20261001`
> **STATUS:** LOCAL CONTROL PLANE CURRENT / P1 A+C REAL-USE FINDINGS CHECKPOINTED / STRUCTURAL IA CORRECTION REPLANNED / R2 PRE-CODE REVIEW IN PROGRESS / RUNTIME IMPLEMENTATION NOT AUTHORIZED.
> **Date:** 2026-10-04 Asia/Nicosia.
> **Fresh reconciliation main:** `d5da6271567a4141b2708d9fa12e673dfce37131`.
> **Fresh bootstrap main for original P1 design:** `aedfa3e48e8359a449c2c8d8eb2b9f053a64ce47`.
> **Reference implementation:** Knee Osteoarthritis physiotherapy referral.
> **Current PHYSIO runtime writer:** none — the active PHYSIO action is read-only R2 pre-code review.
> **Root operational owner:** unchanged — `CURRENT_OPERATIONAL.md` remains the sole repo-wide writer lock. At this reconciliation it records Cockpit D1.1 in pre-code closure HOLD with no active Cockpit runtime writer; that does not prevent PHYSIO sidecar checkpoints.
> **Product authority:** `commercial_products/physio_referral/*`.
> **Technical authority:** existing CU-1 + Knee-OA owners under `clinic_utilities/*`.

---

## 1. Current state

The existing Physiotherapy Referral product is not greenfield.

Current product authority reports:

```text
Knee OA vertical                         released
V5.1                                    released
post-use Cyprus / clinical review        released
receiver prose / chronicity refinement   released
CY_GESY                                  active
authenticated production smoke           PASS
real clinical pilot                      not proven
commercial validation                    not proven
second diagnosis                         not authorized by default
```

Current release chain owned by the product record:

- PR #106 → `8064999ea70e0a90f6073fc6d66a0c8caaba1538`;
- PR #107 → `a19f4b9d52c3076f715fd864a5f7664d2b140c81`;
- PR #109 → `b962485c741f558121e8daabfcf1d20c84f31f63`;
- authenticated post-use smoke `34957778592` — SUCCESS;
- authenticated receiver-refinement smoke `35019200920` — SUCCESS.

The current repository may be newer than the last Physio runtime release. Product/release truth must therefore be read from its owning artifacts rather than by assuming the Physio release SHA equals current repository `main`.

---

## 2. P0 reuse-before-new-path inventory disposition

P0 established:

### Reusable Physio Core already present

- CU-1 typed state / canonical registry / normalization.
- route requirements, ownership and precedence.
- deterministic validation and safety/disposition.
- deterministic Greek formatter and protected API.
- reusable evidence interaction semantics.
- suggestion-vs-selection boundary.
- manual-text reconciliation boundary.
- progressive disclosure / advanced-capability pattern.
- no-patient-browser-storage boundary.
- jurisdiction overlay architecture.
- exact-head regression/release/smoke discipline.

### Knee-OA-specific

- OA evidence corpus and source positions;
- Knee phenotype/examination/qualifier mappings;
- Knee smart defaults;
- Knee-specific referral composition/presentation;
- Knee-specific clinical review clues;
- OA-specific Cyprus/GeSY local positions.

### Existing product authorities to reuse

Do not recreate:

- a second safety engine;
- a second validation pipeline;
- a second evidence-state model;
- a second jurisdiction engine;
- a second patient-draft store;
- a second Physio product `CURRENT`;
- duplicate global Cockpit navigation.

---

## 3. Why a separate PHYSIO control plane now exists

The prior Physio work correctly lived in the technical/product owners where it was built.

That remains true.

The new `programme/PHYSIO/*` layer exists only because Physiotherapy is now a **parallel programme workstream**, which needs durable cross-conversation navigation and a local NOW without taking authority over the root Osteoporosis/Cockpit lifecycle.

```text
programme/PHYSIO
= workstream coordination

commercial_products/physio_referral
= product/commercial truth

clinic_utilities
= technical/runtime/contracts

root six canonicals
= repo-wide governance
```

No product or runtime content is migrated merely to make the directory self-contained.

---

## 4. Current gaps

The important current gaps are validation/product gaps, not missing feature count:

1. real clinical/workflow pilot not yet established;
2. actual receiving-physiotherapist comparative validation not yet established;
3. actual iPhone Safari / VoiceOver acceptance not yet established;
4. international material claim provenance/locator maintenance should be explicit before paid clinical use;
5. willingness-to-pay / conversion / retention remain hypotheses;
6. actual referral mix/frequency should inform the next diagnosis;
7. diagnosis-agnostic extraction of Knee mechanisms should wait until a second real use demonstrates the abstraction.

---


## 4.1 Product Owner real-use evidence — 2026-10-04

The Product Owner exercised five distinct Knee-OA cases on a real mobile device and reported the findings directly. These cases are **not** silently substituted for the frozen `P1_SYNTHETIC_CASE_SET.md`; they are a separate Product-Owner real-use regression set.

Repeated findings across the five cases:

- the final deterministic referral text was repeatedly judged clinically good;
- the main failure was structural information architecture, not a mobile-only presentation defect;
- the first-tap / hidden-second-tap detail model was not discoverable in routine use;
- clinically related information was split across Clinical Picture, detail bubbles/sheets, Functionality, Examination and generic `Περισσότερα`;
- the generic `Περισσότερα` mixed different semantic worlds, including clinical presentation, examination and rehabilitation choices;
- routine removal/deselection was not immediately discoverable;
- function concepts such as chair rise, stairs and walking limitation were searched for in multiple places because ownership was unclear;
- objective examination concepts such as quadriceps weakness and ROM were not consistently discoverable;
- Case 5 exposed the strongest safety-related usability finding: significant weight-bearing difficulty / acute-joint observations were not easily discoverable even though the clinician would not send a routine physiotherapy referral;
- observed completion time was typically about 2–4 minutes, with a material part of that time spent searching for hidden or duplicated concepts rather than composing the referral.

Interpretation:

```text
GOOD REFERRAL OUTPUT
+ REPEATED NAVIGATION/OWNERSHIP SEARCH
= STRUCTURAL IA CORRECTION REQUIRED
!= MOBILE-ONLY POLISH
!= CLINICAL ENGINE REDESIGN
```

The Product Owner corrected the design direction accordingly:

- ordinary detail should not require a hidden second tap;
- the `Λεπτομέρειες` bubble under Clinical Picture should not remain as a competing navigation route;
- each clinical concept should have one obvious semantic owner;
- Functionality, Examination and Rehabilitation/Proposed Plan must remain distinct;
- the generic miscellaneous `Περισσότερα` target architecture is rejected;
- safety/alternative-pathology support should use evidence-bounded review cues from meaningful observation patterns, not diagnosis checkboxes or automatic imaging.

---

## 4.2 Structural correction / R2 status

A bounded structural design was completed outside the repository working tree used for this checkpoint and the Product Owner proceeded to **R2 PRE-CODE REVIEW**.

The exact reviewed design bytes and the terminal R2 review result are not yet present on `main` at this checkpoint. Therefore:

```text
R2 PRE-CODE REVIEW          IN PROGRESS
RUNTIME IMPLEMENTATION      NOT AUTHORIZED
CU-1 CONTRACT EXTENSION     NOT AUTHORIZED
SECOND DIAGNOSIS            NOT AUTHORIZED
```

Before implementation authority can be claimed, the exact reviewed structural design and terminal R2 result must be durably reconciled into the PHYSIO workstream so that the implementation target is source-stable and reviewable.

The only currently plausible cross-project dependency is whether a pattern-triggered review-cue disposition can be represented by the existing CU-1 safety/disposition contract. R2 must decide `REUSE` versus a separately governed shared-Core change; PHYSIO must not mutate CU-1 speculatively.

---

## 5. Current bounded next slice — P1

`PHYSIO-P1-KNEE-OA-REFERENCE-VALIDATION-CORE-BOUNDARY-20261001`

Purpose:

> Use the existing live Knee-OA product as the reference implementation to validate usefulness and freeze the practical boundary between reusable Physio mechanics and Knee-specific vertical content before opening another diagnosis.

P1 remains **read-only with respect to runtime** unless a concrete observed finding later receives a separate bounded correction authority.

P1 design is now frozen in:

- `programme/PHYSIO/P1_VALIDATION_PROTOCOL.md`;
- `programme/PHYSIO/P1_EVIDENCE_WORKSHEET.md`;
- `programme/PHYSIO/P1_SYNTHETIC_CASE_SET.md`;
- `programme/PHYSIO/P1_CORE_BOUNDARY_LEDGER.md`;
- `programme/PHYSIO/P1_EVIDENCE_PROVENANCE_AUDIT.md` — Lane F completed.

Planned evidence work:

1. real-device / iPhone Safari / accessibility acceptance of the current workflow;
2. small receiving-physiotherapist comparison using synthetic/de-identified referrals:
   - actionability;
   - missing information;
   - autonomy;
   - comprehension;
3. end-to-end time/friction comparison:
   - navigation;
   - structured selection;
   - optional editing;
   - copy/transfer;
4. aggregate referral-frequency / diagnosis-mix discovery with no patient identifiers;
5. bounded willingness-to-pay / product-value discovery;
6. focused material evidence-provenance/locator maintenance assessment — **COMPLETE: NEEDS LOCATOR MAINTENANCE; no sampled clinical-state/runtime correction indicated**;
7. produce a reusable-boundary ledger:

```text
KEEP AS CORE
KEEP VERTICAL-SPECIFIC
CHANGE
REMOVE
EVIDENCE GAP
COMMERCIAL HYPOTHESIS
CROSS-PROJECT DEPENDENCY
```

A runtime correction may be opened only from a concrete observed finding and then requires its own bounded implementation authority/scope.

---

## 6. Cross-project dependencies

Current known dependency classes — **record, do not mutate from PHYSIO without owner decision**:

- global Cockpit Home/navigation;
- shared Clinical Excellence auth/session;
- global patient model / patient-history persistence;
- transcript/capture infrastructure;
- billing / entitlement / account model;
- analytics/telemetry;
- Digital Secretary / GeSY external integration;
- shared Core evidence objects if later generalization requires them.

No current P1 item requires mutation of those owners.

---

## 7. Explicitly deferred / forbidden by default

Until P1 evidence justifies otherwise:

- no second diagnosis implementation;
- no broad CU-1 rewrite;
- no speculative “generic Physio framework” refactor;
- no new patient persistence;
- no analytics/billing/entitlements;
- no new Greece/England profile;
- no autonomous literature-to-live pipeline;
- no duplication of global navigation/auth/safety;
- no root canonical mutation merely to checkpoint PHYSIO progress.

The Product Owner may later authorize one of these, but authorization should follow a bounded problem statement and correct owner determination.

---

## 8. Exact next action

The current bounded action is the **single R2 pre-code review already started by the Product Owner**.

1. Let that R2 review reach a terminal `PASS / BLOCK / PARTIAL` for the declared structural IA + safety-review-cue questions.
2. Do not start runtime implementation while the R2 verdict is pending.
3. When the result returns, reconcile the exact reviewed design identity + R2 result into PHYSIO durable state.
4. If `BLOCK`: make only the bounded design correction, then one delta + affected-cumulative closure review under `PROCEDURES.md` P5/P5.1.
5. If `PASS`: perform the Product Owner implementation checkpoint on the reviewed structural correction and authorize only that bounded Knee-OA slice.
6. Preserve the same Product-Owner real-use cases as the post-change usability regression set; do not rewrite the frozen original P1 synthetic set.
7. Lane B/D/E and formal VoiceOver/Safari acceptance remain later P1 evidence; do not use them to delay the already-identified structural correction.
8. Lane F remains **NEEDS LOCATOR MAINTENANCE**, non-blocking and separate from this IA correction.

Do not start a second diagnosis, generic Physio refactor, speculative CU-1 change or foreign-owner mutation as the next action.

---

## 9. Registry sync

```text
PHYSIO LOCAL CONTROL PLANE     UPDATED — A+C findings + R2 status
ROOT CANONICALS               NOT REQUIRED
GLOBAL PROGRAMME REGISTRY     UPDATE NAVIGATION SUMMARY ONLY
COMMERCIAL PRODUCT CURRENT    CURRENT / NO RUNTIME CHANGE
KNEE-OA TECHNICAL OWNERS      CURRENT / NO RUNTIME CHANGE
CROSS-PROJECT MUTATION        POSSIBLE CU-1 DISPOSITION SEAM — R2 TO DECIDE
```
