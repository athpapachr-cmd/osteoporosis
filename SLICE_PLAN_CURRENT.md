# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** `DESIGN-COMPLETE` — pre-code clinical/content and machine-contract design frozen.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Frozen machine contract entrypoint:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`.
> **Final review:** `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V3.md`.
> **Prior active slice:** PR-1 remains intentionally paused.

CU-1 pre-code design is complete. Runtime implementation remains separately unauthorized.

---

# 1. Frozen design set

Clinical/content authority consists of the 11 frozen regional/shared profiles.

Machine/runtime-design authority is composed through:

```text
clinic_utilities/contracts/cu1_contract_manifest_v1.yaml
```

The manifest normatively composes:

```text
CU1_CORE_CONTRACT_V1.md
cu1_typed_supplement_v1.yaml
cu1_registry_v1.yaml
cu1_route_detail_catalog_v1.yaml
cu1_option_catalog_v1.yaml
cu1_structured_option_scope_v1.yaml
cu1_id_normalization_v1.yaml
cu1_context_value_sets_v1.yaml
cu1_rule_catalog_v1.yaml
cu1_route_requirements_v1.yaml
cu1_route_requirements_correction_v1.yaml
cu1_validation_error_policy_v1.yaml
cu1_design_fixtures_v1.yaml
cu1_r1_r2_design_fixtures_v1.yaml
```

---

# 2. Final completeness result

```text
B1 typed profile-specific state homes = PASS
B2 canonical registry / exact gateways = PASS
B3 route ownership / precedence = PASS
B4 safety result severity/blocking/disposition = PASS
B5 formatter contract = PASS
B6 common enums / ID normalization = PASS
R1 machine-declarative safety/consistency triggers = PASS
R2 machine-declarative route required/conditional validation = PASS
context enum closure = PASS
validation-error behavior = PASS
fixture-scope semantics = PASS
no profile-prose runtime interpretation required for trigger/validation logic = PASS
```

Final classification:

```text
CU-1 PRE-CODE DESIGN = DESIGN-COMPLETE
```

---

# 3. Core invariants preserved

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
```

Additional hard boundaries:

```text
unselected safety flag = no assertion, not reassuring negative
nonspecific symptoms do not autonomously create unresolved safety concerns
postoperative/structural ownership follows frozen precedence
exact procedure/protocol/restriction state overrides generic rehabilitation defaults
missing required structural restriction never becomes an invented permissive restriction
runtime must not read profile Markdown to invent trigger or validation semantics
```

---

# 4. Frozen first implementation direction

```text
ephemeral ReferralDraftV1
→ canonical normalization
→ route/gateway/ownership resolution
→ route requirement validation
→ declarative safety/consistency rule evaluation
→ ShortReferralFormatter / DetailedReferralFormatter
→ generated text
→ copy / print
```

Persistence is deliberately unfrozen and out of first implementation scope.

No runtime implementation has been authorized or started.

---

# 5. Implementation prerequisites if later authorized

A future implementation slice must, before coding:

```text
1. fresh six-canonical bootstrap from current main
2. explicit product-owner authorization for CU-1 runtime implementation
3. fresh inspection of actual Clinical Excellence runtime/navigation/integration seams
4. claim one runtime writer branch in CURRENT_OPERATIONAL.md
5. implement only against cu1_contract_manifest_v1.yaml and frozen clinical sources
6. add executable tests derived from the semantic fixtures and full-draft validation contract
7. preserve ephemeral/no-persistence first-slice boundary
8. STOP for focused evidence + independent exact-head review before merge/deploy
```

---

# 6. Out of scope until separately authorized

```text
production HTML/JS/CSS implementation
FastAPI CU-1 endpoints
referral persistence or patient-data storage
CU-2 Radiofrequency workflow implementation
reopening frozen clinical taxonomy without a proven contradiction
PR-1 Transcript Intake runtime work
```

---

# 7. Stop rule

```text
current design result = DESIGN-COMPLETE
runtime = NOT AUTHORIZED
active runtime writer = NONE
```

The design detour is complete. The next product decision is whether to authorize a dedicated CU-1 implementation slice or leave CU-1 frozen and resume another roadmap item.
