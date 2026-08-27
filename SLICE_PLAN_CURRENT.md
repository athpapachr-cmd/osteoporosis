# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 runtime implementation v1

> **STATUS:** CLOSED — IMPLEMENTED / TESTED / MERGED / DEPLOYED.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1 runtime v1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Implementation base:** `7c49c2c6ad5ad9c710a6c02fe1ec4df467b4bab2`.
> **Reviewed implementation head:** `e04004add617afa7222c51d0d669c2134dd8f575`.
> **Merge commit:** `c1da07f581cf8ccf1159d18bb63c23b674cbe9bd` (PR #56 squash merge).
> **Render deploy:** `dep-da8afeuk1f9s73f5sr6g` = live at the same merge commit.
> **Runtime writer:** NONE.
> **Frozen machine contract entrypoint:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`.
> **Prior active slice:** PR-1 remains intentionally paused.

---

# 1. Objective — completed

The first usable CU-1 Physiotherapy Referral v2 was integrated into the protected Clinical Excellence service without reopening the frozen clinical taxonomy or adding referral persistence.

The runtime converts clinician-selected structured inputs into deterministic short/detailed referral text while preserving the frozen semantic and safety invariants.

---

# 2. Delivered scope

```text
A. protected CU-1 utility entrypoint under /clinical/clinic-utilities/physio-referral
B. ephemeral ReferralDraftV1 browser state
C. manifest-driven machine-contract loading/composition
D. canonical ID/alias/context normalization
E. gateway + route + ownership resolution
F. route required/conditional validation
G. declarative safety/consistency rule evaluation
H. deterministic ShortReferralFormatter
I. deterministic DetailedReferralFormatter
J. copy and print actions
K. Clinical Excellence navigation entry to CU-1
L. executable focused contract/safety/gateway tests
```

Explicitly not delivered, by design:

```text
referral persistence in PostgreSQL
localStorage/sessionStorage persistence
patient-registry linkage
saving generated referral text
PDF-specific workflow
CU-2
PR-1 runtime
clinical taxonomy changes
```

---

# 3. Final runtime architecture

```text
clinic_utilities/physio_referral_runtime.py
→ frozen machine-contract loader/composer
→ normalization / validation / declarative rule engine / formatter

clinic_utilities/physio_referral_api.py
→ protected trust boundary
→ exact frozen gateway validation
→ canonical safety-state validation

main.py
→ includes guarded CU-1 router

static/clinic-utilities/physio-referral/*
→ ephemeral browser presentation and copy/print workflow
```

Protected routes:

```text
GET  /clinical/clinic-utilities/physio-referral
GET  /clinical/clinic-utilities/physio-referral/api/contract
POST /clinical/clinic-utilities/physio-referral/api/validate
POST /clinical/clinic-utilities/physio-referral/api/generate
```

No endpoint writes referral data to database or filesystem.

---

# 4. Frozen semantic pipeline implemented

```text
raw structured draft
→ alias normalization
→ context-value normalization
→ frozen registry/gateway trust-boundary validation
→ shared semantic ownership resolution
→ route requirements correction overlay
→ required/conditional/assertion/context validation
→ declarative safety + consistency rule evaluation
→ acknowledgement/disposition gate
→ deterministic short or detailed formatter
```

The runtime does not use clinical-profile Markdown to invent trigger or validation semantics. Profile prose may contribute clinician-facing display labels only.

---

# 5. Final safety / clinical invariants

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != diagnosis
objective deficit != subjective symptom
provocation/test finding != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
unselected safety flag != reassuring negative
nonspecific symptom != autonomous unresolved safety concern unless an explicit frozen rule says so
exact procedure/protocol/restriction state > generic postoperative defaults
```

Additional fail-closed hardening established during implementation review:

```text
forged shared_target_optional → hard block
unknown acknowledged rule id → hard block
unknown clinician disposition → hard block
unknown safety input flag → hard block
```

---

# 6. Persistence/privacy boundary — proven

```text
ReferralDraftV1 = ephemeral
Generated referral text = ephemeral
Server = request/response only
Browser storage = none
Database writes = none
Public fixtures = synthetic only
```

Focused tests explicitly verify absence of CU-1 `localStorage`, `sessionStorage` and SQLAlchemy persistence paths.

---

# 7. Acceptance evidence

Exact reviewed implementation head:

```text
e04004add617afa7222c51d0d669c2134dd8f575
```

Final GitHub Actions evidence:

```text
compile = PASS
focused unittest suite = PASS
29/29 tests = PASS
```

Coverage includes:

```text
manifest and correction overlay
alias normalization
closed route/context/safety namespaces
all frozen gateway mappings
forged gateway rejection
route-required/conditional validation
fracture WB/use boundaries
shared muscle correction boundary
postoperative exclusivity/context
explicit-vs-inferred safety behavior
urgent disposition gating
adjunct/core-rehab consistency
frailty assertion semantics
not-assessed neurological semantics
short/detailed formatter determinism
no browser/server referral persistence
forged acknowledgement/disposition rejection
protected clinical-key dependency
```

Independent exact-head diff review found no frozen clinical profile/contract mutation, no persistence/schema change, no CU-2/PR-1 scope creep and no identifiable patient data.

---

# 8. Merge and deploy evidence

PR #56 was squash-merged with `expected_head_sha=e04004add617afa7222c51d0d669c2134dd8f575`.

Resulting `main` commit:

```text
c1da07f581cf8ccf1159d18bb63c23b674cbe9bd
```

Render auto-deploy:

```text
deploy = dep-da8afeuk1f9s73f5sr6g
commit = c1da07f581cf8ccf1159d18bb63c23b674cbe9bd
build = successful
uvicorn startup = observed
status = live
```

External route-level HTTP smoke from the assistant execution sandbox could not be executed because DNS resolution failed before reaching the Render host. This is recorded as **NOT PROVEN**, not as an application failure.

Therefore:

```text
IMPLEMENTED = PROVEN
TESTED = PROVEN
MERGED = PROVEN
DEPLOYED / RENDER LIVE = PROVEN
EXTERNAL ROUTE-LEVEL HTTP SMOKE FROM THIS SANDBOX = NOT PROVEN
PILOT-VALIDATED = NOT CLAIMED
```

---

# 9. Non-blocking maintenance note

`clinic_utilities/physio_referral_runtime.py` retains an older unused router builder. The active application imports only the guarded router from `clinic_utilities/physio_referral_api.py`. Removal is optional maintenance and does not reopen CU-1.

---

# 10. Stop rule

CU-1 runtime v1 is closed.

```text
NO runtime continuation implied
NO CU-2 authorization implied
NO PR-1 resumption implied
NO persistence expansion implied
NO taxonomy reopening implied
```

After the control-plane closeout is merged and its writer lock released, the next engineering slice requires a new explicit product-owner decision.