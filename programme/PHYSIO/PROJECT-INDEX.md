# PHYSIO PROJECT INDEX — Clinical Excellence Cockpit

> **STATUS:** ACTIVE PHYSIO WORKSTREAM LOCAL CONTROL PLANE.
> **ROLE:** durable navigation / workstream boundary for Physiotherapy work.
> **CANONICAL REPOSITORY:** `athpapachr-cmd/osteoporosis`.
> **REFERENCE IMPLEMENTATION:** Knee Osteoarthritis physiotherapy referral.
> **ROOT AUTHORITY:** the six root canonicals remain repo-wide authorities; this file does not replace them.

---

## 1. Authority boundary

PHYSIO is an independent parallel workstream inside the Clinical Excellence Cockpit.

```text
ROOT CANONICALS
= repo-wide architecture / roadmap / writer lock
!= PHYSIO product state

programme/PHYSIO/*
= PHYSIO workstream navigation + local NOW
!= second repo-wide writer lock

commercial_products/physio_referral/*
= current Physio Referral commercial/product truth

clinic_utilities/*
= technical contracts/runtime/source truth
```

Permanent rules:

- `CURRENT_OPERATIONAL.md` remains the sole repo-wide writer lock.
- PHYSIO must not rewrite `CURRENT_OPERATIONAL.md`, `SLICE_PLAN_CURRENT.md`, `TODO.md`, `CLINICAL_EXCELLENCE_PLAN.md` or other root canonicals merely because PHYSIO progresses.
- A change that genuinely requires shared Clinical Excellence Core, the global patient model, global navigation/identity, auth, persistence, analytics/billing/entitlements or another workstream must be recorded in `programme/PHYSIO/CURRENT.md` as a cross-project dependency and returned to the programme/root owner.
- Do not duplicate current product truth into this directory. Link to the owning artifact.

---

## 2. Where Physiotherapy is already built

### Reusable CU-1 technical foundation

Primary technical owners:

```text
clinic_utilities/contracts/CU1_CORE_CONTRACT_V1.md
clinic_utilities/contracts/cu1_contract_manifest_v1.yaml
clinic_utilities/physio_referral_runtime.py
clinic_utilities/physio_referral_api.py
clinic_utilities/physio_referral_formatter_el.py
clinic_utilities/physio_referral_formatter_el_v2.py
clinic_utilities/physio_profiles/*
static/clinic-utilities/physio-referral/*
```

CU-1 already owns reusable typed referral state, canonical IDs/aliases, route ownership and precedence, validation, safety/disposition semantics, deterministic Greek formatting, protected transport and ephemeral first-implementation behavior.

### Knee-OA reference implementation

Product-specific technical/design owners:

```text
clinic_utilities/physio_referral_product/
clinic_utilities/physio_referral_product/contracts/
clinic_utilities/physio_referral_product/knee_oa_projection.py
clinic_utilities/physio_referral_product/knee_oa_presentation_v4.py
clinic_utilities/physio_referral_product/jurisdiction_overlay.py
clinic_utilities/physio_referral_product/jurisdictions/CY_GESY/
static/clinic-utilities/physio-referral/product-*
```

Knee OA is the first reference implementation of the product architecture. It must be reused as evidence for what belongs in Physio Core; it must not be copied wholesale into every future diagnosis.

### Current product / commercial authority

```text
commercial_products/physio_referral/CURRENT.md
commercial_products/physio_referral/PRODUCT_CONTEXT_CURRENT.md
commercial_products/physio_referral/PRODUCT_PLAN.md
commercial_products/physio_referral/PRODUCT_CHANGELOG.md
commercial_products/physio_referral/releases/
commercial_products/physio_referral/reviews/
commercial_products/physio_referral/jurisdictions/
```

The older product-state files under `clinic_utilities/physio_referral_product/` that redirect to `commercial_products/physio_referral/` remain compatibility redirects only.

---

## 3. Reference implementation state

The current Knee-OA release is documented by the commercial product authority as:

```text
Knee OA only
V5.1 + bounded post-use receiver refinements
CY_GESY active
authenticated production smoke verified
real clinical pilot not yet proven
commercial validation not yet proven
second diagnosis not authorized by default
```

Important release evidence includes PRs #106, #107 and #109 and authenticated production-smoke runs `34957778592` and `35019200920`.

Do not reconstruct newer product state from this summary if the owning product `CURRENT.md` has changed.

---

## 4. Reusable Physio architecture already demonstrated

Treat the following as existing reusable mechanisms unless evidence proves a replan is required:

- CU-1 typed draft / registry / normalization / route ownership.
- CU-1 deterministic validation and fail-closed safety/disposition engine.
- Protected Clinical Excellence Physio route/API.
- Deterministic referral projection; routine LLM generation is not required.
- `selection != suggestion != evidence != safety`.
- `symptom != objective finding != diagnosis`.
- `missing != negative`.
- clinician-owned manual text with explicit stale/reconciliation handling.
- progressive disclosure / scan-first advanced UI.
- evidence states with explicit disagreement rather than manufactured consensus.
- evidence availability/provenance kept separate from evidence direction.
- suggestion candidates that require explicit clinician action.
- `JurisdictionOverlayV1`: international evidence != local clinical position != administrative/reimbursement rule != system-lifecycle status.
- explicit configuration for jurisdiction; no silent patient-location inference.
- ephemeral patient/referral draft and no browser patient-draft persistence in the current product.
- exact-head regression/release gates plus authenticated synthetic production smoke.
- clinical / receiving-professional / UX / commercial review lenses.
- product utility gate:

```text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING
```

---

## 5. Knee-OA-specific content that must not be mistaken for Core

Examples include:

- Knee-OA international evidence corpus and item-level positions.
- Knee-OA smart starting rehabilitation plan.
- Knee clinical-picture concepts/qualifiers, examination mappings and Greek referral phrases.
- Knee-specific review clues and presentation refinements.
- Knee-OA `CY_GESY` recommendation/page mappings and local difference matrix.
- Knee-specific deterministic product projection.

The reusable mechanism may be generalized later; diagnosis content remains versioned vertical content.

---

## 6. Workstream sequence

Current PHYSIO progression:

```text
P0 — reuse-before-new-path inventory                 COMPLETE
P1 — Knee-OA reference validation + Core boundary   CURRENT NEXT BOUNDED SLICE
P2 — select one next diagnosis from real evidence   DEFERRED
P3 — generalize only mechanisms proven by ≥2 uses   DEFERRED
```

Do not begin P2 merely to increase feature count.

---

## 7. Fresh-session bootstrap for PHYSIO

A fresh PHYSIO coordinator/session must:

1. fresh-verify `athpapachr-cmd/osteoporosis/main`;
2. read the six root canonicals in the order required by `AGENTS.md`;
3. read this file;
4. read `programme/PHYSIO/CURRENT.md`;
5. read the current Physio commercial authorities:
   - `commercial_products/physio_referral/CURRENT.md`;
   - `PRODUCT_CONTEXT_CURRENT.md`;
   - `PRODUCT_PLAN.md`;
6. inspect only the exact technical/design/release artifacts needed for the current PHYSIO task;
7. respect the root writer lock and any PHYSIO local writer recorded in `CURRENT.md`.

Hard rule:

```text
CHAT MEMORY != PHYSIO CURRENT
KNEE-OA RELEASE SUMMARY != CURRENT PRODUCT AUTHORITY
PHYSIO CURRENT != ROOT CURRENT_OPERATIONAL
```

---

## 8. Cross-project dependency protocol

If PHYSIO discovers a required change outside Physio-owned product/runtime scope:

```text
identify dependency
→ record it in programme/PHYSIO/CURRENT.md
→ name exact owner / seam / requested decision
→ STOP mutation of that foreign owner
→ return dependency to programme/root coordinator
```

Examples:

- shared Clinical Excellence patient model;
- global Cockpit Home/navigation;
- common auth/session model;
- patient persistence/history;
- shared transcript/capture infrastructure;
- billing/entitlements/account model;
- analytics/telemetry;
- global evidence objects;
- Digital Secretary / GeSY integration.

---

## 9. Local state owner

`programme/PHYSIO/CURRENT.md` is the durable PHYSIO workstream NOW.

It records:

- PHYSIO task/slice;
- current source identity;
- current writer/scope;
- evidence already established;
- open gaps;
- cross-project dependencies;
- exact next bounded action;
- explicit forbidden/deferred actions.

It is a sidecar workstream state file, not a second global operational lock.
