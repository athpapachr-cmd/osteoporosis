# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 runtime implementation v1

> **STATUS:** IMPLEMENTATION AUTHORIZED — bounded runtime slice active.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1 runtime v1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Implementation base:** `7c49c2c6ad5ad9c710a6c02fe1ec4df467b4bab2`.
> **Runtime writer:** `feat/cu1-physio-referral-runtime-v1-2026-08-27`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Frozen machine contract entrypoint:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`.
> **Pre-code completeness review:** `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V3.md` = DESIGN-COMPLETE.
> **Prior active slice:** PR-1 remains intentionally paused.

---

# 1. Objective

Deliver the first usable CU-1 Physiotherapy Referral v2 inside the Clinical Excellence service without reopening the frozen clinical taxonomy or adding persistence.

The runtime must turn clinician-selected structured inputs into deterministic short/detailed referral text while preserving all frozen semantic and safety invariants.

---

# 2. In scope

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
L. executable tests derived from frozen semantic fixtures
```

---

# 3. Explicitly out of scope

```text
referral persistence in PostgreSQL
localStorage/sessionStorage persistence
patient-registry linkage
saving generated referral text
PDF-specific generation workflow
CU-2 radiofrequency workflow
PR-1 transcript runtime
clinical taxonomy changes
new evidence-sensitive clinical recommendations
```

If a frozen contract cannot be implemented without changing clinical meaning, STOP and REPLAN.

---

# 4. Runtime architecture

## 4.1 Backend

Add one dedicated runtime module, provisionally:

```text
clinic_utilities/physio_referral_runtime.py
```

Responsibilities:

```text
load cu1_contract_manifest_v1.yaml
resolve listed normative YAML/contract artifacts
apply manifest precedence including route_requirements_correction before validation
expose normalized UI/route metadata needed by the browser
validate ReferralDraftV1 deterministically
evaluate declarative rule DSL deterministically
format short/detailed text deterministically
never interpret clinical profile Markdown for trigger/validation logic
```

The runtime may use a YAML parser dependency because the normative frozen machine artifacts are YAML. The loader must fail closed on missing/unknown normative artifacts rather than silently falling back.

## 4.2 Protected routes

```text
GET  /clinical/clinic-utilities/physio-referral
GET  /clinical/clinic-utilities/physio-referral/api/contract
POST /clinical/clinic-utilities/physio-referral/api/validate
POST /clinical/clinic-utilities/physio-referral/api/generate
```

All `/clinical/*` CU-1 routes use the existing Clinical Excellence browser-session/key protection pattern.

No endpoint writes referral data to database or filesystem.

## 4.3 Browser UI

Presentation assets live under:

```text
static/clinic-utilities/physio-referral/
```

Browser state is in-memory only for the open page lifetime.

Minimum flow:

```text
select body region / gateway
→ select primary route/problem
→ enter/select relevant findings/context/restrictions/goals/directions
→ choose short or detailed output
→ validate
→ display blocking/safety/consistency results
→ generate referral
→ copy or print
```

The UI must not synthesize negative findings from unselected/missing state.

---

# 5. Frozen semantic pipeline

```text
raw structured draft
→ alias normalization
→ context-value normalization
→ registry/gateway validation
→ route ownership/precedence resolution
→ route requirements correction overlay
→ required/conditional/assertion/context validation
→ declarative safety + consistency rule evaluation
→ formatter
```

The implementation must follow the manifest precedence exactly.

---

# 6. Safety and clinical invariants

The runtime must preserve at minimum:

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

Blocking/safety results must come from the frozen machine contract, not ad-hoc UI strings.

---

# 7. Persistence/privacy boundary

```text
ReferralDraftV1 = ephemeral
Generated referral text = ephemeral
Server = request/response only
Browser storage = none
Database writes = none
Public fixtures = synthetic only
```

No name, patient identifier, phone/email/address or real transcript belongs in committed tests/fixtures.

---

# 8. Acceptance tests

Before MERGE-READY, executable evidence must cover:

```text
contract manifest loads and all normative artifacts resolve
correction overlay precedence works
unknown route/context/safety/rule IDs fail closed
aliases normalize before registry validation
closed context enums reject unknown values
all frozen gateway mappings resolve
route ownership/precedence examples resolve deterministically
required/conditional route fields produce canonical validation errors
shared fracture and shared muscle boundary cases
postoperative exclusivity/required restriction cases
safety-input flags trigger only declared rules
no symptom-only invented safety concern
adjunct-without-core-rehab consistency behavior
short formatter deterministic output
long formatter deterministic output
not-assessed does not become normal/negative
browser code contains no localStorage/sessionStorage referral persistence
protected API rejects unauthenticated requests when clinical key is configured
```

Existing semantic fixture YAML should be converted into executable oracles where practical; full runtime tests must construct complete drafts where the manifest requires them.

---

# 9. Implementation evidence / review gate

```text
implementation complete
→ focused automated tests green
→ inspect exact branch-vs-main diff
→ verify no persistence/taxonomy expansion
→ independent exact-head review
→ STOP at MERGE-READY or BLOCK
```

Merge/deploy occurs only after that gate is clean.

---

# 10. REPLAN triggers

STOP and update this slice instead of patching around the design if any of the following is discovered:

```text
frozen route cannot be represented by ReferralDraftV1
manifest precedence is insufficient/contradictory
clinical profile meaning conflicts with normative machine artifact
existing auth/static routing cannot preserve the intended protected boundary
formatter contract is materially ambiguous
semantic fixtures conflict with frozen route requirements
implementation would require persistence to function
```
