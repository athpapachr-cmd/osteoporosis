# CU-1 Design Completeness Review v2 — 2026-08-27

> **STATUS:** REVIEW COMPLETE — **BLOCK** for runtime implementation authorization.
> **Base reviewed:** `5cd3cdd9cd735b7ac55a1a162bae5a9daee08c1f` (`main`).
> **Prior review:** `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md`.
> **Hardening reviewed:** PR #50 / merged machine contract v1.
> **Runtime changes:** none.

---

# 1. Executive conclusion

The bounded B1–B6 hardening materially improved CU-1 and resolved the prior flat-schema, route-key, precedence, common safety-severity, formatter and enum normalization gaps.

However the repeat review is still **BLOCK**, for two narrower reasons only:

```text
R1 — safety-rule trigger conditions are not machine-declarative
R2 — route-specific required/conditional context and formal-assertion validation are not machine-declarative
```

Clinical/content design remains frozen and should not be broadly reopened.

Runtime implementation remains unauthorized.

---

# 2. Original B1–B6 disposition

## B1 — typed homes for profile-specific state — PASS

`ReferralDraftV1`, `ProblemSelection`, `SharedTarget`, typed restrictions, measurements, shared fracture/muscle/deconditioning contexts and the typed supplement provide explicit homes for the structured state required by the frozen profiles.

Classification:

```text
B1 = RESOLVED
```

## B2 — canonical registry / exact gateways — PASS

The merged contract now contains:

```text
cu1_registry_v1.yaml
cu1_route_detail_catalog_v1.yaml
cu1_option_catalog_v1.yaml
cu1_id_normalization_v1.yaml
cu1_structured_option_scope_v1.yaml
cu1_contract_manifest_v1.yaml
```

Regional→shared mappings are explicit rather than string-derived. Long-tail fracture-site and muscle-group identities are also frozen.

Classification:

```text
B2 = RESOLVED
```

## B3 — route ownership / precedence — PASS

Global precedence plus specific postoperative/structural ownership resolves the previously ambiguous cases.

Examples:

```text
ACL reconstruction → postoperative_knee_rehabilitation
ACL nonoperative/prehab → acl_injury_instability_rehabilitation

flexor/extensor tendon repair
→ digital_tendon_injury_rehabilitation
→ not generic postoperative_wrist_hand_rehabilitation co-primary

thumb UCL/RCL repair
→ thumb_mcp_collateral_ligament_injury_rehabilitation

established fracture rehabilitation
→ shared_fracture semantic owner
```

Classification:

```text
B3 = RESOLVED
```

## B4 — common safety severity/blocking/disposition model — PASS AT RESULT LEVEL

The contract now freezes:

```text
SafetyScreenState
SafetySeverity
ClinicianDisposition
SafetyResult
acknowledged state
highest-severity behavior
formatter blocking
```

The option/safety catalog maps canonical safety rule IDs to severity/disposition behavior.

This closes the prior ambiguity about what a `soft warning`, hard warning, block or urgent reassessment means after a rule has been triggered.

Classification of the original B4:

```text
B4 = RESOLVED AT SAFETY-RESULT LEVEL
```

A narrower trigger-definition gap remains as R1 below.

## B5 — formatter contract — PASS

The contract freezes:

```text
ShortReferralFormatter(draft, registry) -> text
DetailedReferralFormatter(draft, registry) -> text
```

and defines:

- pre-format validation order;
- diagnosis/presentation wording rule;
- omission of `not_assessed` / `not_stated`;
- restriction precedence;
- regional→shared deduplication;
- adjunct placement;
- short-vs-detailed semantic relationship;
- no mutation/inference by formatter;
- safety gate behavior.

Synthetic fixtures establish semantic output expectations.

Classification:

```text
B5 = RESOLVED
```

## B6 — common enums / ID normalization — PASS

Common tri-state/enumeration meanings are frozen and versioned. Mixed-case legacy/profile IDs are handled through explicit aliases and canonical output uses lowercase snake_case.

Classification:

```text
B6 = RESOLVED
```

---

# 3. Remaining blocker R1 — safety triggers are not machine-declarative

The hardening correctly freezes **what happens after a safety rule fires**.

For example, `unresolved_dvt_concern` has a canonical severity and disposition requirement.

But the contract does not yet freeze a declarative rule expressing **which exact structured input state causes that rule to fire**.

The typed supplement currently states conceptually:

```text
evaluate_profile_rules_and_map_to_canonical_safety_rules
```

with trigger sources defined as:

```text
frozen_profile_markdown_and_semantic_fixtures
```

That still leaves an implementation step such as:

> “Read/interpret this profile prose and decide whether this combination of selections activates `unresolved_dvt_concern` versus another safety rule.”

This is especially important for compound conditions such as:

```text
acute trauma + marked weakness + unresolved rupture concern
calf pain/swelling + unresolved DVT concern
fracture context + unknown required loading/WB status
radicular presentation + incomplete neurological assessment
adjunct selected + no core rehabilitation direction
```

The fixtures cover important examples but are not a complete declarative safety-rule engine.

### Required resolution

Freeze a compact declarative `cu1_rule_catalog_v1` (YAML/JSON equivalent) containing, for every v1 safety/consistency rule that runtime may evaluate:

```text
rule_id
applies_to profile/route(s)
trigger expression over canonical fields/IDs
severity_id
acknowledgement/disposition behavior reference
optional suppression/exception condition
```

Runtime should evaluate that catalog mechanically rather than interpret Markdown prose.

Classification:

```text
R1 = BLOCKING
```

---

# 4. Remaining blocker R2 — route-specific validation requirements are not machine-declarative

The core contract correctly says:

```text
validate required typed context for route
formal_assertion_state may only be used where registry permits it
```

But the route registry currently does not fully declare:

```text
required fields per route
conditionally required fields
allowed context keys per route
formal assertion allowed/required and its exact assertion meaning
required subtype state
required procedure/protocol/restriction states for postoperative or structural routes
```

Examples where runtime would still have to infer requirements from prose include:

```text
postoperative shoulder/knee/wrist-hand required procedure/date/protocol/restriction context
shared fracture site-specific applicability of WB vs upper-limb-use fields
formal diagnosis assertion for cervical radiculopathy / CTS / De Quervain / FAIS etc.
confirmed nonoperative structural tear pathways requiring established diagnosis + management decision
```

The existence of typed homes does not itself define which homes are mandatory for each route.

### Required resolution

Freeze a machine-declarative route schema/requirements catalog, either integrated into the route registry or as a companion artifact, containing at minimum:

```text
route_id
required_fields[]
optional_fields[]
conditional_requirements[]
allowed_wording_modes[]
formal_assertion_policy
subtype_policy
shared_context_type if any
restriction/applicability rules
```

Implementation may then validate against the catalog without reading profile prose to invent requirements.

Classification:

```text
R2 = BLOCKING
```

---

# 5. Structured-option scope — PASS

The deliberately curated structured v1 option catalog is acceptable and is not a blocker.

The frozen rule is now explicit:

```text
unlisted profile prose
!= hidden runtime machine control
```

Unlisted clinically valid suggestions may be carried through explicit clinician free text, and promotion to a structured control requires a later design change.

This is preferable to silently inventing machine IDs for every optional sentence in the clinical profiles.

---

# 6. Manifest / multi-artifact composition — PASS

The fact that the contract is split across several artifacts is not itself a blocker because `cu1_contract_manifest_v1.yaml` is the single normative entrypoint and freezes:

```text
artifact set
precedence
normalization order
conflict rule
structured-option boundary
persistence boundary
```

No duplicate authority problem was found that would require collapsing everything into one file.

---

# 7. Repeat-review exit-criteria matrix

```text
[x] every frozen route has a canonical machine identity or explicit structured-v1 scope disposition
[x] every regional→shared gateway resolves to an exact canonical target
[x] no reviewed operative/structural scenario has two unresolved primary owners
[x] profile-specific structured state has typed homes
[x] common tri-state/enums and canonical ID policy are normalized/versioned
[x] safety result severity/blocking/disposition behavior is deterministic after a rule fires
[x] formatter interface/output/omission/precedence rules are frozen
[x] short/detailed fixtures establish semantic equivalence and expected structure
[ ] safety trigger conditions are fully machine-declarative
[ ] route-specific required/conditional validation rules are fully machine-declarative
[ ] no runtime implementation would need to interpret prose to answer a remaining semantic question
```

Because the final three checks fail, CU-1 cannot yet be classified DESIGN-COMPLETE.

---

# 8. Required next hardening scope

Do **not** reopen the clinical profiles.

Only two small design artifacts are required:

```text
1. cu1_rule_catalog_v1
   → declarative trigger expressions for v1 safety/consistency rules

2. cu1_route_requirements_v1
   → route-level required/conditional fields, assertion/subtype policy and context applicability
```

Then repeat the completeness review.

---

# 9. Final classification

```text
BLOCK
```

Reason:

> The B1–B6 hardening successfully froze the data model, namespace, ownership, safety-result behavior and formatter semantics, but runtime would still need to interpret profile prose to determine some safety triggers and route-specific validation requirements.

No runtime implementation is authorized.
