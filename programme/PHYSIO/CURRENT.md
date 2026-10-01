# PHYSIO CURRENT — Knee-OA Reference Implementation / Physio Core Boundary

> **TASK:** `PHYSIO-P1-KNEE-OA-REFERENCE-VALIDATION-CORE-BOUNDARY-20261001`
> **STATUS:** LOCAL CONTROL PLANE INITIALIZED / P0 INVENTORY COMPLETE / P1 DESIGN-ONLY NEXT.
> **Date:** 2026-10-01 Asia/Nicosia.
> **Fresh bootstrap main:** `860cac7afaa0eefbabd24118ada35430d1820cef`.
> **Reference implementation:** Knee Osteoarthritis physiotherapy referral.
> **Current PHYSIO runtime writer:** none.
> **Root operational owner:** unchanged — `CURRENT_OPERATIONAL.md` remains the sole repo-wide writer lock and currently records the separate PR-1 transcript lifecycle.
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

## 5. Current bounded next slice — P1

`PHYSIO-P1-KNEE-OA-REFERENCE-VALIDATION-CORE-BOUNDARY-20261001`

Purpose:

> Use the existing live Knee-OA product as the reference implementation to validate usefulness and freeze the practical boundary between reusable Physio mechanics and Knee-specific vertical content before opening another diagnosis.

P1 starts **design/read-only with respect to runtime**.

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
6. focused material evidence-provenance/locator maintenance assessment;
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

After this local control-plane checkpoint is present on `main`:

1. fresh-bootstrap PHYSIO from `programme/PHYSIO/PROJECT-INDEX.md` + this file;
2. design the P1 validation protocol/artifact set without changing runtime;
3. identify which evidence can be collected immediately by the Product Owner and which requires a separate reviewer/receiver;
4. checkpoint the approved P1 design here;
5. only then execute the bounded validation.

Do not start a second diagnosis or runtime refactor as the next action.

---

## 9. Registry sync

```text
PHYSIO LOCAL CONTROL PLANE     CURRENT
ROOT CANONICALS               NOT REQUIRED
GLOBAL PROGRAMME REGISTRY     NOT PRESENT / NOT REQUIRED
COMMERCIAL PRODUCT CURRENT    CURRENT
KNEE-OA TECHNICAL OWNERS      CURRENT
CROSS-PROJECT MUTATION        NOT REQUIRED
```
