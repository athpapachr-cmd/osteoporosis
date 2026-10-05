# PHYSIO CURRENT — Knee-OA Reference Implementation / Physio Core Boundary

> **TASK:** `PHYSIO-P1-KNEE-OA-REFERENCE-VALIDATION-CORE-BOUNDARY-20261001`
> **STATUS:** R2-A APPROVED / PRE-CODE CLOSURE PASS / POST-CODE BLOCK AT `c6eb295` / FIRST CLOSURE BLOCK AT `fbf0701` / SECOND BOUNDED CORRECTION LOCALLY TESTED / FINAL INDEPENDENT DELTA + AFFECTED-CUMULATIVE CLOSURE NEXT / NO PR, MERGE OR DEPLOY.
> **Date:** 2026-10-05 Asia/Nicosia.
> **Fresh coordinator base main:** `b9beca7b1e233245f9ea3429a247a1c80a69ac7c`.
> **Branch:** `feat/physio-r2-ia-correction-2026-10-05`, based on that main; no PR, merge or deploy at this candidate checkpoint.
> **Fresh bootstrap main for original P1 design:** `aedfa3e48e8359a449c2c8d8eb2b9f053a64ce47`.
> **Reference implementation:** Knee Osteoarthritis physiotherapy referral.
> **Current PHYSIO runtime writer:** this PHYSIO coordinator on `feat/physio-r2-ia-correction-2026-10-05`, scoped to Knee-OA product UI/overlay, associated Knee-OA contracts/tests and PHYSIO/product checkpoints; root lock is recorded in `CURRENT_OPERATIONAL.md`.
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

The original R2 pre-code review returned a terminal split verdict; the following is historical and R2-A correction closure has since passed:

```text
R2-A INFORMATION ARCHITECTURE    BLOCK
R2-B REVIEW-CUE / SAFETY DESIGN  PASS
CU-1 SHARED CHANGE REQUIRED      NO
RUNTIME IMPLEMENTATION           NOT AUTHORIZED
```

The review result is checkpointed in `P1_R2_PRECODE_REVIEW_RESULT.md`.

R2-A left exactly three material decisions:
1. duplicate function concepts represented as both findings and functional impairments;
2. pain-location overlap plus the semantic ownership of swelling vs effusion vs hot/swollen review context;
3. hidden `walking_aid_assessment_and_training`: expose or remove.

`P1_R2A_STRUCTURAL_CORRECTION.md` is now the **frozen Product Owner-approved correction design**. The prior coordinator-proposal status was historical, not authority; the three decisions were expressly confirmed by the Product Owner across 2026-10-04/05:

1. **APPROVED:** one Functionality / `functional_impairments` UI owner for duplicate function concepts; legacy finding aliases compatibility-only.
2. **APPROVED:** remove routine structured pain map, tenderness, crepitus and separate effusion. Keep standalone **Οίδημα** visibly in Κλινική εικόνα, default unselected and non-mandatory (`missing != negative`). Preserve explicit hot/swollen pattern observations in the already-PASS R2-B review-cue mechanism; simple swelling alone triggers no alert.
3. **APPROVED:** `walking_aid_assessment_and_training` only in `Προτεινόμενο πλάνο → Πρόσθετες / Περισσότερες επιλογές`, default-off, without auto-suggestion or auto-selection.

The Product Owner also requested bounded implementation after review PASS. One independent R2-A delta + affected-cumulative review returned **PASS / COMPLETE_FOR_DECLARED_SCOPE / no new material finding** at frozen design commit `fdb6bff3b852112e8fdfa35a2db5dc8fd8f82eca`; see `P1_R2A_CLOSURE_REVIEW_RESULT.md`. R2-B semantic PASS is reused, not reopened. The first independent post-code review of `c6eb295` returned **BLOCK** for selected additional-plan control promotion/hierarchy and an invalid intermediate restriction draft; see `P1_R2_POSTCODE_REVIEW_RESULT.md`. The first closure of `fbf0701` closed those two but returned **BLOCK** for stale availability styling on conditional additional evidence; see `P1_R2_POSTCODE_CLOSURE_RESULT.md`. A second bounded correction is locally browser-tested and must receive one independent delta + affected-cumulative closure at the new committed exact head. The five-case matrix and local gate evidence are in `P1_R2_IMPLEMENTATION_CANDIDATE.md`. No shared CU-1 mutation, second diagnosis or persistence was made.

R2-B safety semantics remain PASS and settled. No shared CU-1 mutation is required by R2-B.

---

## 5. Current bounded next slice — P1

`PHYSIO-P1-KNEE-OA-REFERENCE-VALIDATION-CORE-BOUNDARY-20261001`

Purpose:

> Use the existing live Knee-OA product as the reference implementation to validate usefulness and freeze the practical boundary between reusable Physio mechanics and Knee-specific vertical content before opening another diagnosis.

P1 evidence work was originally **read-only with respect to runtime**. The Product Owner's later concrete five-case findings and approved R2 correction supply the separate bounded implementation authority recorded above.

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

Commit the second bounded correction and obtain one independent delta + affected-cumulative post-code closure review at its exact head. On PASS, open a bounded PR with canonical-impact declaration and run exact PR gates. Merge/deploy remain separate authority decisions.

## 9. Registry sync

```text
PHYSIO LOCAL CONTROL PLANE     FIRST CLOSURE BLOCK / second correction browser gates PASS / final closure pending
ROOT CANONICALS               WRITER LOCK checkpoint updated
GLOBAL PROGRAMME REGISTRY     UPDATE NAVIGATION SUMMARY ONLY
COMMERCIAL PRODUCT CURRENT    CANDIDATE checkpoint / live release unchanged
KNEE-OA TECHNICAL OWNERS      BOUNDED candidate / shared CU-1 unchanged
CROSS-PROJECT MUTATION        NONE REQUIRED BY R2-B; CU-1 REUSED UNCHANGED
```
