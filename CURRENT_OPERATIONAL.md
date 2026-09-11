# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PHYSIO REFERRAL PRODUCTIZATION — KNEE-OA EVIDENCE MODULE / DESIGN ACTIVE
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh base `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Active branch:** `design/physio-referral-knee-oa-evidence-v1-2026-09-11`.
> **Active slice:** `CU1-PRODUCT-KNEE-OA-EVIDENCE-V1-2026-09-11`.
> **ACTIVE DESIGN/CANONICAL WRITER:** this bounded Physio Referral Knee-OA evidence-design slice only.
> **ACTIVE RUNTIME WRITER:** NONE.
> **Production config/secret authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **Deploy/production-smoke authority:** NONE.

---

# 1. Prior closed state

Clinical Learning Hub L-1D is `MERGED / DEPLOYED / PRODUCTION-SMOKE-VERIFIED / CLOSED`. Its frozen/runtime owners are read-only for this slice. No Learning Hub runtime, schema, database, transcript, Practice Review or Signal work is authorized here.

Existing CU-1 Physiotherapy Referral v2 is already implemented/tested/merged/deployed. Its production runtime and frozen broad taxonomy are not being mutated in this design slice.

---

# 2. Product-owner direction

The product owner authorized continuation of the Physio Referral productization work after completion of the separate Learning Hub workstream.

The first commercial/product vertical slice remains deliberately limited to:

```text
Knee Osteoarthritis only
```

The frozen Step-1 UX contract has been carried forward onto the fresh 2026-09-11 `main` ancestry at:

```text
clinic_utilities/physio_referral_product/UX_CONTRACT_CURRENT.md
```

Target product positioning remains an evidence-aware, fast referral assistant suitable for later subscription around the previously selected initial price point of approximately €9.99/month, not a simple text generator.

---

# 3. Current authorized work — Step 2 only

Design and freeze the Knee-OA evidence knowledge module.

Required outputs:

```text
reviewed source registry
per-intervention source positions
explicit evidence-state semantics
cross-guideline conflict semantics
positive suggestion triggers
caution / limited-evidence triggers
source-year vs product-reviewed-on separation
machine-readable Step-2 contract
human-readable design rationale
```

Clinical evidence must remain source-specific. Conflicting frameworks must not be silently averaged into a fake consensus.

---

# 4. Explicit non-authority / hold

Until a later separate product-owner implementation decision:

```text
NO CU-1 runtime code mutation
NO production UI rewrite
NO second diagnosis
NO frozen broad CU-1 taxonomy expansion unless Step-2 review proves a concrete contradiction and triggers REPLAN
NO patient persistence
NO billing/auth/entitlement implementation
NO autonomous literature-to-live-rule updating
NO PR merge
NO deploy
NO production smoke
```

The current branch is design/canonical/evidence-contract work only.

---

# 5. Exact next action

```text
review current authoritative Knee-OA guidance/evidence
→ map only interventions relevant to the existing CU-1 Knee-OA state/options
→ determine whether the frozen five-state UX evidence model is sufficient
→ if cross-guideline disagreement cannot be represented honestly, record explicit REPLAN rather than hide conflict
→ create human + machine Step-2 evidence contracts
→ perform exact design review
→ freeze/release writer lock if clean
```
