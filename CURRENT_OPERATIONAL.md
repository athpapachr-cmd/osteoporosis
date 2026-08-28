# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-28 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified current main after smoke reconciliation:** `62d206dd69e191fe813667280e99498df5438cef`.
> **CU-1 runtime implementation:** PR #56 squash-merged as `c1da07f581cf8ccf1159d18bb63c23b674cbe9bd`.
> **CU-1 control-plane closeout:** PR #57 squash-merged as `53cc6324edbe62243fd887e0073612f669d094cc`; writer-lock release PR #58 squash-merged as `a6a9257bc93693bfdd3d3e37090ebbb157f3634c`.
> **CU-1 production-smoke reconciliation:** PR #59 squash-merged as `62d206dd69e191fe813667280e99498df5438cef`.
> **Focused evidence:** GitHub Actions exact-head run PASS — 29/29 tests at reviewed head `e04004add617afa7222c51d0d669c2134dd8f575`.
> **Production deploy:** Render deploy `dep-da8afeuk1f9s73f5sr6g` = `live`, exact runtime merge commit `c1da07f581cf8ccf1159d18bb63c23b674cbe9bd`.
> **Production browser smoke:** PASS — authenticated product-owner smoke recorded in `clinic_utilities/CU1_PRODUCTION_SMOKE_2026-08-28.md`.
> **Current major phase:** Personal Clinical Excellence foundation / baseline and Clinical Practice Review roadmap.
> **CU-1 status:** CLOSED — IMPLEMENTED + TESTED + MERGED + DEPLOYED + PRODUCTION-SMOKE-VERIFIED.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Proven CU-1 state

```text
pre-code clinical/content design = FROZEN / DESIGN-COMPLETE
machine contract = FROZEN
runtime implementation = COMPLETE
focused automated evidence = PASS (29/29)
independent exact-head review = MERGE-READY / CLEAN
PR #56 = SQUASH-MERGED
control-plane closeout PR #57 = SQUASH-MERGED
writer-lock release PR #58 = SQUASH-MERGED
production-smoke reconciliation PR #59 = SQUASH-MERGED
Render auto-deploy of runtime merge = LIVE
authenticated production browser smoke = PASS
referral persistence = NONE by design
active writer = NONE
```

Implemented protected entrypoints:

```text
GET  /clinical/clinic-utilities/physio-referral
GET  /clinical/clinic-utilities/physio-referral/api/contract
POST /clinical/clinic-utilities/physio-referral/api/validate
POST /clinical/clinic-utilities/physio-referral/api/generate
```

The browser draft and generated text remain ephemeral. No CU-1 referral data are written to PostgreSQL, localStorage or sessionStorage.

---

# 2. Safety / contract evidence

The merged runtime is manifest-driven from:

```text
clinic_utilities/contracts/cu1_contract_manifest_v1.yaml
```

Focused executable evidence covers contract loading/correction precedence, alias normalization, route/context validation, all frozen shared gateways, forged-gateway rejection, safety input semantics, forged acknowledgement/disposition rejection, postoperative/fracture/muscle boundaries, formatter determinism, forbidden reassuring inference and no-persistence behavior.

Important review hardening completed before merge:

```text
arbitrary shared_target_optional → rejected unless exact frozen gateway
unknown acknowledged_rule_ids → rejected
unknown clinician_disposition → rejected
unknown safety input flag → rejected
not_assessed/unselected state → never converted to reassuring negative
```

A duplicate unused router builder remains in `clinic_utilities/physio_referral_runtime.py`; `main.py` imports only the guarded router from `clinic_utilities/physio_referral_api.py`. This is non-blocking maintenance debt, not active behavior and not a reason to reopen CU-1.

---

# 3. Deploy / smoke truth

Render service `osteoporosis` auto-deployed the exact runtime merge commit and reported:

```text
build successful
uvicorn process started
status = live
```

The product owner subsequently executed the authenticated production browser smoke and reported all requested checks as passing:

```text
Clinical Excellence authenticated access = PASS
Clinic Utilities → Physiotherapy Referral load = PASS
representative Knee → Knee OA path = PASS
required-field completion + Validate = PASS
Short referral generation = PASS
Detailed referral generation = PASS
Copy = PASS
Print = PASS
refresh clears prior referral state = PASS
```

Therefore the precise status is:

```text
DEPLOYED = PROVEN
RENDER LIVE = PROVEN
PRODUCTION-SMOKE-VERIFIED = PROVEN
PILOT-VALIDATED = NOT CLAIMED
```

---

# 4. Closed scope / prohibitions

CU-1 closure does not authorize any of the following:

```text
CU-2 implementation
PR-1 runtime resumption
referral persistence or patient-registry linkage
clinical taxonomy reopening
new evidence-sensitive physiotherapy recommendations
```

Frozen CU-1 clinical profiles/contracts remain authoritative unless a future concrete contradiction justifies a separately authorized maintenance slice.

---

# 5. Exact next action

```text
STOP — CU-1 is closed and production-smoke-verified.
Await explicit product-owner selection of the next roadmap slice.
```

No engineering continuation is implied by CU-1 completion.
