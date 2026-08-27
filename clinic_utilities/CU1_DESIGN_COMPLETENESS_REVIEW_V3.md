# CU-1 Design Completeness Review v3 — 2026-08-27

> **STATUS:** REVIEW COMPLETE — **DESIGN-COMPLETE** for CU-1 pre-code design.
> **Base reviewed:** `213db0c31fab1582c2466b4b42921f5f2b74e299` (`main`).
> **Runtime implementation:** **NOT AUTHORIZED** by this review.
> **Prior reviews:** `CU1_DESIGN_COMPLETENESS_REVIEW.md`, `CU1_DESIGN_COMPLETENESS_REVIEW_V2.md`.

---

# 1. Executive conclusion

CU-1 Physiotherapy Referral v2 now has a sufficiently explicit frozen clinical/content design and machine-declarative contract for a future runtime implementation to be planned without interpreting profile prose to invent route identity, route ownership, structured field requirements, context enums, formal-assertion semantics, safety trigger conditions, validation behavior or formatter semantics.

Final classification:

```text
DESIGN-COMPLETE
```

This classification means only that the **pre-code design contract is complete enough to implement**. It does **not** mean implemented, tested, merged as runtime, deployed, production-smoke-verified or pilot-validated.

Runtime implementation still requires separate explicit product-owner authorization and a fresh implementation slice/branch.

---

# 2. Frozen clinical scope

The 11 regional/shared CU-1 profiles remain frozen and unchanged:

```text
cervical
lumbar
shoulder
elbow
wrist/hand
knee
hip/groin
ankle/foot
shared fracture
shared muscle/myotendinous
shared deconditioning/balance/gait
```

The review found no reason to reopen the broad clinical taxonomy.

---

# 3. B1–B6 disposition

```text
B1 typed homes for profile-specific state = PASS
B2 canonical registry / exact gateways = PASS
B3 route ownership / precedence = PASS
B4 common safety result severity/blocking/disposition = PASS
B5 formatter contract = PASS
B6 common enums / ID normalization = PASS
```

These remain resolved from the prior hardening and repeat review.

---

# 4. R1 — declarative safety/consistency triggers = PASS

`clinic_utilities/contracts/cu1_rule_catalog_v1.yaml` now defines a closed trigger DSL and exact rule applicability over canonical structured state.

Important properties:

```text
explicit clinician safety-input flags are closed and typed
unselected safety flag = no assertion, not a reassuring negative
nonspecific symptoms do not autonomously create unresolved DVT/rupture/infection/etc. concerns
incomplete radicular neurological screen is derived mechanically from typed motor/sensory/reflex states
adjunct-without-core-rehab is mechanical
SIFK missing loading state is mechanical
lower-limb fracture missing WB state is mechanical
severity remains sourced from the canonical safety catalog
runtime may not invent rules or infer unlisted safety flags
```

Classification:

```text
R1 = RESOLVED
```

---

# 5. R2 — route-specific validation = PASS

The composed route-validation contract is:

```text
cu1_route_requirements_v1.yaml
+ cu1_context_value_sets_v1.yaml
+ cu1_route_requirements_correction_v1.yaml
+ cu1_validation_error_policy_v1.yaml
```

It now freezes:

```text
base required/optional draft fields
allowed wording modes
formal-diagnosis assertion behavior
established-structural assertion behavior
route/subtype policies
postoperative required/conditional context
route-specific structural/nonoperative requirements
regional-vs-postoperative exclusivity
shared fracture site applicability and WB/upper-limb-use requirements
shared muscle required context with imaging remaining optional unless explicitly entered
shared deconditioning/frailty context rules
canonical context enum values
validation-error classes and blocking behavior
```

The final R2 correction preserves the frozen shared-muscle profile semantics:

```text
MRI/ultrasound context remains optional
major-avulsion/rupture concern remains a canonical safety-input flag, not an injury-type enum
```

Classification:

```text
R2 = RESOLVED
```

---

# 6. Machine composition / precedence = PASS

`clinic_utilities/contracts/cu1_contract_manifest_v1.yaml` is the single normative machine entrypoint.

It freezes the composition and precedence of:

```text
core contract
typed supplement
route registry
route detail catalog
option/safety catalog
structured-option scope
ID normalization
context value sets
rule catalog
route requirements
exact R2 correction
validation-error policy
semantic fixtures
```

The runtime prohibition is explicit:

```text
runtime_may_read_profile_markdown_for_trigger_or_validation_logic = false
```

Clinical Markdown remains the clinical-content source, but not a hidden runtime rule engine.

---

# 7. Fixture semantics = PASS

The original fixture set remains a partial semantic oracle for routing, precedence, forbidden inference, formatter structure and named safety behavior.

The R1–R2 fixture set is the route/rule validation oracle for the declarative hardening.

Full runtime validation tests must later construct complete `ReferralDraftV1` objects and apply all normative artifacts from the manifest.

---

# 8. Safety/privacy/persistence boundary = PASS FOR PRE-CODE DESIGN

Frozen first implementation direction remains:

```text
ephemeral structured referral draft
→ generated short/detailed text
→ copy / print
```

Persistence is not part of the frozen first implementation scope.

No identifiable patient data belongs in the public repository or synthetic fixtures.

---

# 9. Exit-criteria matrix

```text
[x] frozen clinical profiles cover the approved CU-1 scope
[x] every structured v1 route has a canonical identity or explicit scope disposition
[x] regional→shared gateways resolve to exact targets
[x] primary ownership / postoperative precedence is deterministic
[x] profile-specific structured state has typed homes
[x] common IDs/enums/context values are machine-versioned
[x] safety result behavior is deterministic
[x] safety trigger conditions are machine-declarative
[x] route required/conditional validation is machine-declarative
[x] formal assertion / subtype policies are machine-declarative
[x] formatter semantics are frozen
[x] fixture scope is explicit
[x] runtime need not interpret profile prose to invent trigger/validation semantics
[x] first implementation persistence boundary is explicit
```

No remaining pre-code design blocker was identified.

---

# 10. Final classification and stop rule

```text
CU-1 PRE-CODE DESIGN = DESIGN-COMPLETE
RUNTIME IMPLEMENTATION = NOT AUTHORIZED
ACTIVE RUNTIME WRITER = NONE
```

The next step is **not** automatic coding. A future product-owner authorization must explicitly open a CU-1 implementation slice, claim a runtime writer, inspect the actual integration seams, and implement against the frozen manifest.
