# SLICE_PLAN_CURRENT.md — G-2 Evidence-backed Osteoporosis Guidance Content v1

> **STATUS:** DESIGN-COMPLETE — EVIDENCE/RULE/PROFILE/MILESTONE CONTRACT REVIEWED; RUNTIME NOT IMPLEMENTED.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** `M01-G2-EVIDENCE-GUIDANCE-CONTENT-v1`.
> **Base main:** `5182d250e244b2ed9e086138cb3b2edcdb967e25`.
> **Design branch:** `design/module01-g2-evidence-backed-guidance-2026-08-31`.
> **Runtime writer:** NONE at design closeout.

---

# 1. Objective / result

G-2 defines the minimum evidence-backed osteoporosis guidance content required before the real system-assisted pilot. It converts the generic G-1 capability into a versioned clinical-content contract without turning guidelines into automatic treatment decisions.

The frozen design separates:

```text
product-flow profile
!=
evidence-backed rule
!=
medication safety checklist
!=
exact therapy milestone
!=
clinician treatment decision
```

This design is deliberately narrower than an exhaustive osteoporosis textbook. It prioritizes rules that materially improve the current consultation and can be represented safely from current structured data.

---

# 2. Normative machine artifacts

```text
schemas/osteoporosis_guidance_evidence_registry_v1.yaml
schemas/osteoporosis_guidance_rules_v1.yaml
schemas/osteoporosis_guidance_profiles_v1.yaml
schemas/osteoporosis_therapy_milestones_v1.yaml
schemas/osteoporosis_guidance_contract_manifest_v1.yaml
M01_G2_EVIDENCE_GUIDANCE_REVIEW_V1.md
```

The manifest is the normative machine entrypoint for bounded runtime implementation.

---

# 3. Evidence governance

Primary clinical framework for G-2 v1:

```text
NOGG 2024
```

Supporting evidence is kept explicitly separate:

- current EMA/EU product information for regulatory safety facts;
- Endocrine Society 2020 denosumab recommendations;
- ECTS 2020 denosumab-discontinuation position statement;
- ASBMR/BHOF 2024 as contextual goal-directed-treatment reasoning;
- recent denosumab-discontinuation trials/observational evidence as uncertainty context.

Hard rule:

```text
SOURCE A != SOURCE B
CONFLICT / VARIATION → EXPLICIT PROVENANCE + CLINICIAN JUDGMENT
```

No silent framework hybridization is permitted.

---

# 4. Visit profiles frozen

The design defines profiles for:

1. first assessment — new/uncertain diagnosis;
2. initial-to-service known osteoporosis/osteopenia;
3. routine stable follow-up;
4. treatment start;
5. repeated administration / continuation / due monitoring;
6. treatment change / transition;
7. post-fragility fracture;
8. fracture on treatment;
9. adverse effect / intolerance;
10. treatment completion / consolidation.

A distinct `results_or_workup_review_with_management_decision` profile was evaluated and retained as a **product-flow candidate**, not activated as a new runtime enum in G-2 design. It is not a guideline claim.

---

# 5. Evidence-backed rule set

The G-2 rules registry contains 26 evidence-backed candidate rules covering:

- formal fracture-risk assessment when explicitly indicated;
- VFA/vertebral-imaging structured triggers;
- secondary-cause review;
- falls/function review;
- new-fragility-fracture event override;
- fracture-on-treatment reassessment without automatic failure/switch labeling;
- NOGG very-high-risk review;
- treatment-decision factors and patient preference;
- parenteral vitamin-D preparation;
- denosumab start, due, delay and exit safety;
- medication-specific safety for denosumab, zoledronate, romosozumab, teriparatide and oral bisphosphonates;
- post-anabolic/romosozumab antiresorptive follow-on;
- oral-bisphosphonate 12–16-week and ≥5-year review points;
- zoledronate ≥3-year reassessment;
- targeted lifestyle/falls/bone-health communication.

---

# 6. Final clinical-review corrections

## G2-RV1 — denosumab >7-month rule

The specific NOGG >7-month rebound-risk escalation requires:

```text
reliable actual denosumab administration count >= 2
+ reliable last actual dose date
+ encounter date
+ >7 calendar months since last actual dose
```

It must not be asserted after only one documented dose. General six-month due/time-critical guidance remains separate.

## G2-RV2 — FRAX evidence rule

The initial-visit profile may surface Formal Risk as product flow, but the evidence-specific NOGG FRAX rule activates only when:

```text
NOGG scope eligible
+ formal_risk_indicated == "yes"
```

Initial visit alone does not imply mandatory FRAX.

## G2-RV3 — NOGG very-high-risk criteria

The represented NOGG criteria were verified against Section 4 and include:

- recent vertebral fracture within 2 years;
- ≥2 vertebral fractures;
- BMD T-score ≤ -3.5;
- high-dose glucocorticoids ≥7.5 mg/day prednisolone equivalent over 3 months;
- other/multiple risk factors or FRAX-defined very-high risk.

These surface specialist/parenteral/anabolic consideration, not automatic anabolic selection.

## G2-RV4 — persisted enum predicates

Persisted string enums such as `"yes"` / `"no"` are explicitly quoted in YAML predicates so YAML boolean coercion cannot alter runtime semantics.

---

# 7. Therapy milestone design

Evidence-backed milestone capability includes:

### Denosumab

- every-administration calcium/mineral safety check;
- evidence-derived expected due date at 6 calendar months from reliable actual administration;
- separate >7-month rebound-risk escalation after ≥2 reliable actual doses;
- long-term reassessment at 5 years / sooner with changed risk context;
- explicit denosumab-exit sequential planning at 6 months from last actual dose.

### Oral bisphosphonate

- start safety/use guidance;
- 12–16-week tolerance/adherence/correct-use review;
- ≥5-year fracture-risk reassessment.

### Zoledronate

- administration renal/mineral safety guidance;
- ≥3-year fracture-risk reassessment.

### Romosozumab / teriparatide

- start safety guidance;
- explicit completion/transition milestones with antiresorptive follow-on without delay.

Administration count and elapsed exposure remain separate. Approximate duration does not become an exact milestone date.

---

# 8. Explicit non-rules

G-2 v1 explicitly does **not** encode:

```text
CTX >=280 ng/L → automatic second zoledronate
CTX >=300 ng/L → automatic second zoledronate
CTX at 3 months → mandatory retreatment command
Prolia 4th/8th/10th dose → generic milestone
romosozumab → mandatory cardiology/vascular referral without approved clinic policy
a fracture on treatment → automatic treatment failure/switch
```

CTX-guided post-denosumab management remains clinically relevant, but no validated universal automatic retreatment threshold is asserted.

---

# 9. Runtime activation classification

The design review classifies rules as:

```text
activate_v1
checklist_only
blocked_missing_structured_input
blocked_missing_reliable_linkage
design_only
```

Important first-runtime exclusions:

- `OST_G2_R15_DENOSUMAB_EXIT_CTX_FOLLOWUP` remains blocked until a specific post-exit zoledronate actual event can be linked reliably to the denosumab-exit sequence across visits;
- `OST_G2_R16_DENOSUMAB_EXIT_NO_CTX_OPTION` remains blocked until CTX-monitoring availability is represented explicitly; a blank CTX value is not evidence that monitoring is unavailable.

Medication safety rules may surface **verify/review/confirm** guidance when the current data model cannot prove full safety clearance. Missing/stale data must never produce “safe/cleared”.

---

# 10. G-1 integration boundary

Preferred bounded implementation:

```text
existing G-1 longitudinal projection
+ live current encounter snapshot
→ G2 osteoporosis evidence context
→ pure deterministic evidence evaluator
→ evidence contributions per existing domain/card
→ merge with G-1 Visit Plan using manifest priority
→ existing `Σημερινή ροή` / `Γιατί τώρα:` UI
```

Do not replace the generic G-1 engine with a monolithic osteoporosis engine.

G-1 live-state ownership remains mandatory:

```text
LIVE CONTROL VALUE, INCLUDING BLANK
>
PERSISTED BROWSER CACHE
```

---

# 11. Acceptance evidence

Contract workflow:

```text
G2 evidence guidance contract
run 33358433732
head 6a40a4a87882a4531c69ce9dff5e0ecd46011d84
COMPLETED / SUCCESS
```

The workflow validates YAML syntax, source/claim references, rule/profile/milestone cross-references, domains, active-rule reachability, manifest paths/schemas and explicit forbidden semantics.

Human clinical/runtime review:

```text
M01_G2_EVIDENCE_GUIDANCE_REVIEW_V1.md
```

The review explicitly classifies runtime-safe, checklist-only and blocked rules.

---

# 12. Completion matrix

```text
evidence registry                         DESIGNED / REVIEWED
rules registry                            DESIGNED / REVIEWED
visit profiles                            DESIGNED / REVIEWED
therapy milestones                        DESIGNED / REVIEWED
non-hybridization                         FROZEN
CTX automatic threshold rules             EXPLICITLY FORBIDDEN
generic Prolia ordinal milestones          EXPLICITLY FORBIDDEN
runtime activation classification          FROZEN
machine contract CI                        PASS
human design review                        COMPLETE
runtime implementation                     NO
runtime tests                              NO
merged                                     NO
deployed                                   NO
production-smoke-verified                  NO
pilot-validated                            NO
```

---

# 13. Stop / next action

The evidence/content **design** slice is complete.

The next product-owner-authorized action is a separate bounded runtime implementation branch from the exact design-complete ancestry:

```text
fresh main verification
→ create G-2 runtime implementation branch from design-complete head
→ claim runtime/canonical writer lock
→ implement deterministic evidence context/evaluator + G-1 merge
→ add focused synthetic/runtime regressions + inherited G-1/C1 regressions
→ STOP at implementation/test gate
```

No PR, merge, deploy or production smoke is authorized by this design closeout.
