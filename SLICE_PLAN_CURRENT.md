# SLICE_PLAN_CURRENT.md — CU-1 dynamic relevant-field UX maintenance v1

> **STATUS:** ACTIVE MAINTENANCE IMPLEMENTATION.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1 dynamic-form maintenance v1.
> **Maintenance base:** `58f41b5b29ee61280f1557df480989c5830465ba`.
> **Writer:** `fix/cu1-dynamic-relevant-fields-2026-08-28`.
> **Clinical taxonomy:** frozen and unchanged.
> **Machine contract entrypoint:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`.
> **Prior formatter maintenance:** merged in PR #61; final clinician prose acceptance still pending.
> **CU-2:** not authorized.
> **PR-1:** remains paused.

---

# 1. Problem

The current browser UI presents machine-contract catalogs as if they were one universal clinical form. That creates two defects:

```text
A. irrelevant controls are visible for the selected clinical problem
B. routine referral generation can be blocked by interaction friction rather than a true safety requirement
```

For example, an elbow referral should not display walking, stair, lower-limb loading or cervical-only options.

---

# 2. Objective

Convert CU-1 from a global contract inspector into a clinician-facing adaptive referral builder.

```text
region
→ pathway
→ safest allowed wording mode
→ only relevant optional findings/function/goals/rehab
→ only route-triggered structural/safety context
→ generate
```

The browser remains ephemeral and deterministic.

---

# 3. Progressive-disclosure contract

## 3.1 Before region/pathway selection

Do not render downstream clinical catalogs.

## 3.2 After region selection

Show only the routes for that region.

## 3.3 After route selection

Render only the option IDs allowed by a versioned UI relevance scope for the selected profile, with optional route-specific additions/removals.

Relevant domains:

```text
findings
functional impairments
goals
rehab directions
adjuncts
explicit restrictions
safety concerns
```

A global machine option catalog remains validation authority but is not the presentation menu.

## 3.4 Structural/postoperative context

Special context fields appear only when triggered by the selected route/wording/profile.

Examples:

```text
routine elbow tendinopathy → no fracture/loading/protocol fields
postoperative elbow rehabilitation → procedure/protocol/restriction context
shared fracture → fracture-specific status and limb/site-dependent loading/use controls
radicular cervical/lumbar presentation → neurological screen
shared muscle injury → muscle/injury/management context
```

---

# 4. Generation-first interaction

For routine routes:

```text
profile + route
→ auto-select presentation wording when available
→ allow immediate basic referral generation
```

Optional findings, function, goals and rehab selections enrich the referral but do not block it.

If presentation wording is unavailable, choose the least inferential allowed mode that is already asserted by explicit route selection (for example established structural diagnosis). Formal-diagnosis mode that requires a separate clinician assertion must still request that assertion.

No software-generated diagnosis assertion is allowed.

---

# 5. True blocking fields

Retain blocking only when required by the frozen semantic/safety contract, including:

```text
formal diagnostic assertion when required
established diagnosis/nonoperative disposition where explicitly required
postoperative procedure/protocol/restriction state
fracture healing/loading/use state where required
shared muscle structural/management context where required
active safety-rule acknowledgement/disposition
```

The UI must explain missing required fields in plain Greek and visually surface the exact relevant control.

---

# 6. Advanced controls

Global safety and manual canonical-context tools must not dominate routine workflow.

They may remain available under an explicitly collapsed advanced section for unusual clinician-entered context, but they must not expose unrelated routine choices by default.

---

# 7. Acceptance evidence

Tests must prove at minimum:

```text
1. elbow scope excludes walking/stairs/weight-bearing/cervical traction
2. cervical scope excludes lower-limb-only function/loading options
3. knee scope includes walking/stairs/squat and excludes upper-limb gripping/dexterity
4. routine elbow referral can generate with no optional findings/goals/rehab selected
5. presentation wording auto-selects when available
6. structural/postoperative requirements still block when genuinely missing
7. irrelevant context cards stay hidden
8. no hidden stale selection leaks when profile/route changes
9. existing gateway/safety/formatter/no-persistence tests remain green
10. generated prose remains Greek and deterministic
```

---

# 8. REPLAN triggers

STOP and replan if dynamic scoping would require:

```text
new clinical recommendations
new machine IDs
changing route ownership
weakening a genuine safety requirement
inferring a diagnosis or examination finding
```

---

# 9. Stop rule

```text
implementation
→ focused exact-head tests
→ independent exact-head review
→ MERGE-READY or BLOCK
```
