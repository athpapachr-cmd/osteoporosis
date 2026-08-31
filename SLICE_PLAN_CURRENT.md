# SLICE_PLAN_CURRENT.md — G-2 Evidence-backed Osteoporosis Guidance Content v1

> **STATUS:** ACTIVE DESIGN / EVIDENCE REVIEW.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** `M01-G2-EVIDENCE-GUIDANCE-CONTENT-v1`.
> **Base main:** `5182d250e244b2ed9e086138cb3b2edcdb967e25`.
> **Design branch:** `design/module01-g2-evidence-backed-guidance-2026-08-31`.
> **Runtime writer:** NONE during design/evidence review.

---

# 1. Problem

G-1 proves that dynamic visit planning and `WHY NOW` work in production, but its current base flows are primarily product-flow mechanics. Before the real system-assisted pilot, clinically active guidance must be backed by explicit reviewed osteoporosis evidence or approved clinic policy.

The goal is not an exhaustive osteoporosis textbook. The goal is the **minimum safe and useful content registry** that improves the real encounter and can later be refined from five-case use evidence.

---

# 2. Scope

G-2 v1 will define evidence-backed content for:

1. first assessment — new/uncertain diagnosis;
2. initial-to-service patient with known osteoporosis/osteopenia;
3. results/work-up review with management decision, if justified as a distinct visit intent;
4. routine stable follow-up;
5. treatment start;
6. repeated administration / continuation / due monitoring;
7. post-fragility fracture and fracture-on-treatment event overrides;
8. treatment change / transition / consolidation;
9. adverse-effect/intolerance context;
10. denosumab/time-critical administration and discontinuation/transition safety where exact evidence supports it.

---

# 3. Evidence governance contract

Every clinically active rule must contain at minimum:

```text
rule_id
module/domain
card/domain target
rule_class
trigger/applicability
human guidance objective
WHY NOW text
source_id
source_org/publication
source_version/year
source_locator
recommendation/criterion summary
strength/certainty when available
reviewed_on
status
```

Source classes:

```text
guideline
position_statement
consensus_statement
regulatory_label_or_safety_source
systematic_review_or_key_trial
approved_clinic_policy
product_flow
```

`product_flow` may organize a visit but may not be presented as guideline-backed clinical truth.

---

# 4. Non-hybridization rule

If two frameworks differ materially, the registry preserves separate rule/source records. The runtime may show that guidance varies by framework, but G-2 must not create a silent synthetic threshold.

```text
SOURCE A != SOURCE B
CONFLICT/VARIATION → explicit provenance + clinician judgment
```

---

# 5. Clinical decision boundary

G-2 may surface:

- what should be checked;
- what needs reassessment;
- prerequisites/safety issues;
- timing risk;
- unresolved transition requirements;
- evidence-backed options or decision factors.

G-2 must not silently choose the final drug or substitute for clinician judgment.

---

# 6. Denosumab hard boundary

G-2 may activate exact time-sensitive rules only when the reviewed source supports the exact semantic claim.

Required distinctions:

```text
actual administration date > scheduled label
next-due date may be explicit but must not be invented
administration count != elapsed exposure
late administration risk != discontinuation plan
planned transition != actual antiresorptive administration
BTM monitoring recommendation != automatic retreatment command unless source/policy explicitly defines it
```

No 4th/8th/10th-dose milestone is active merely from ordinal count.

---

# 7. Machine-readable deliverables

Planned design artifacts:

```text
schemas/osteoporosis_guidance_evidence_registry_v1.yaml
schemas/osteoporosis_guidance_rules_v1.yaml
schemas/osteoporosis_guidance_profiles_v1.yaml
schemas/osteoporosis_therapy_milestones_v1.yaml
schemas/osteoporosis_guidance_contract_manifest_v1.yaml
M01_G2_EVIDENCE_GUIDANCE_REVIEW_V1.md
```

The manifest will be the normative machine entrypoint if the design reaches `DESIGN-COMPLETE`.

---

# 8. Runtime mapping constraint

G-2 must map only to real G-1/current form domains such as:

```text
fracture_history
formal_risk
dxa
vfa
secondary_causes
laboratory_monitoring
falls_function
sarcopenia
treatment_history
administrations
treatment_decision
transition_safety
followup_tasks
communication
understanding
```

If useful clinical content has no safe current target, record it as a design/runtime gap rather than forcing it into a wrong domain.

---

# 9. Acceptance fixtures

At minimum design tests/fixtures must cover:

- first assessment with no prior structured history;
- known patient with reusable prior data and pending results;
- routine stable follow-up without new event;
- due/late denosumab administration with exact actual-date evidence;
- new fragility fracture overriding routine continuation;
- fracture on treatment;
- treatment start with agent-specific prerequisites;
- denosumab stopping/transition context;
- consolidation after anabolic/romosozumab where evidence supports sequence guidance;
- conflicting or insufficient source evidence → explicit uncertainty/no automatic rule.

---

# 10. Out of scope

- exhaustive drug monographs;
- unsupported clinic habits presented as guidelines;
- PR-1/PR-2 implementation;
- treatment recommendation automation;
- real-patient data;
- five-case pilot collection;
- KPI scoring/Practice Review changes;
- physiotherapy/RF work.

---

# 11. Current gate

`DESIGN/EVIDENCE REVIEW IN PROGRESS`.

No new clinical runtime rule is active until the evidence registry, rule semantics, profile mapping and conflict handling have been reviewed as one coherent contract.
