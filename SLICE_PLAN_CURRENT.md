# SLICE_PLAN_CURRENT.md — CU-1 global rich referral + evidence panel v1.19

> **STATUS:** ACTIVE RUNTIME IMPLEMENTATION — GLOBAL HORIZONTAL ROLLOUT.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice:** CU-1 Physiotherapy Referral — shared rich rehabilitation document model.
> **Authoritative remote main:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Approved prototype base:** `aebfb5a6ee14a0e44d80dd6183a1877d74567b46`.
> **Writer/runtime writer:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **Product-owner authorization:** YES — global style across all conditions, grounded in route-specific guidelines/literature; clinician-visible references required.
> **Merge/deploy:** NOT AUTHORIZED until exact-head CI and representative output review.

---

# 1. Product job

Generate a concise Greek physiotherapy referral that communicates the rehabilitation destination, clinically appropriate progression and the intervention directions used to pursue each goal, without turning the physician referral into an exercise prescription.

The approved LET prototype is the product-shape reference. It is not the clinical-content template for other diagnoses.

Global implementation must be:

```text
ONE SHARED DOCUMENT MODEL
+
ROUTE / SUBTYPE / CONTEXT-SPECIFIC EVIDENCE AUTHORITY
+
ONE SHORT RENDERER
+
ONE DETAILED RENDERER
+
ONE CLINICIAN EVIDENCE VIEW
```

Custom route-specific formatter functions are not the rollout strategy.

---

# 2. GeSY text contract

Hard output ceiling: **2000 characters**.

Normal Detailed target: **1500–1850 characters**.

Short output: coherent narrative, materially shorter than Detailed, with clinical flow rather than list syntax.

Clinical-context opening should use only actually supplied facts and, when available, compress:

```text
diagnosis/presentation + findings + functional impact + explicit work/sport/activity context
```

Do not add redundant labels such as `Κύριο πρόβλημα:`, `Κλινικά ευρήματα:` or `Λειτουργικός περιορισμός:` when prose can carry the same meaning more efficiently.

---

# 3. Shared Rich Rehabilitation Document Model

The normalized render model remains route-data driven and supports two clinically appropriate detailed organizations:

```text
RichReferralPlan
├── clinical_context
├── staged_layout_optional
│   └── stages[]
│       ├── stage_label
│       ├── goals[]
│       ├── intervention_directions[]
│       └── progress_markers[]
├── section_layout_optional
│   └── detailed_sections[]
│       ├── heading
│       └── sentences[]
├── adjunct_boundary_optional
├── reassessment_optional
├── evidence_state
└── evidence_source_ids[]
```

The default conceptual organization for ordinary musculoskeletal rehabilitation may be staged:

```text
1. Early management / symptom or tissue irritability / mobility / initial activation
2. Capacity restoration / strength / endurance / control / progressive loading
3. Functional reintegration / actual activity demands / self-management / recurrence-risk reduction
```

This is **document organization**, not a universal three-phase evidence claim. Routes may use fewer/different stages, or a section-based referral layout, when their actual authority and product wording are better represented without artificial stages.

A route with no clinically meaningful staged progression must not display `ΣΤΑΔΙΟ 1` merely because the renderer historically required a stage object.

---

# 4. Global content rules

Each rendered goal must have a route-appropriate intervention direction. No orphan goals.

Route-specific examples of possible intervention families include:
- education/self-management and load/activity modification;
- active mobility/ROM when an actual restriction exists;
- progressive tissue loading or strengthening when supported;
- neuromuscular/balance/movement-control work when supported and relevant;
- graded exposure to actual functional/work/sport demand;
- selected manual, orthotic, taping, electrotherapeutic or other adjuncts only within their evidence/applicability boundary;
- postoperative/fracture/healing protection only from explicit route/protocol authority.

Stage 3 should include both functional return and recurrence-risk reduction/self-management when clinically appropriate; this rule does not force a three-stage visual layout.

Evidence complexity may remain high internally while referral prose should remain clinically simple, useful and physician-to-physiotherapist appropriate.

---

# 5. Evidence resolver

The structured CU-1 evidence corpus is the normative source for literature provenance and applicability.

Preserve:
- route/subtype/context boundaries;
- source type and framework identity;
- recommendation direction;
- strength/certainty;
- `referral_core` vs `therapist_execution_detail` vs `clinician_ui_only`;
- conflicts;
- evidence gaps;
- freshness/review status;
- patient-specific protocol precedence.

Automatic referral prose may be supported only by active/applicable `referral_core` authority or explicit patient-specific protocol/clinician instruction. `therapist_execution_detail` must not be promoted into routine physician prescription. `clinician_ui_only` may be displayed in the evidence panel but must not automatically enter the referral.

Evidence gaps must fail transparently. A route without adequate applicable authority must never receive borrowed neighboring-route content or a generic evidence-labelled pathway.

---

# 6. Clinician-only evidence panel

Add an expandable panel labelled:

```text
Τεκμηρίωση / Παραπομπές
```

It is separate from the GeSY referral text and is not copied/printed with it by default.

For the currently selected route/subtype/context show source identity, applicable claims, strength/certainty, freshness and explicit gaps/conflicts. Internal IDs must not be primary user-facing content.

---

# 7. Route-class exceptions

### Postoperative / fracture / tissue-healing
Written protocol, healing, weight-bearing, ROM or loading restrictions override generic active progression. Time may appear only when supplied by authoritative protocol/healing instruction.

### Neurological / nerve routes
Do not invent tendon-style loading stages. Progressive objective neurological deficit or unresolved localization/safety context preserves correct-owner reassessment behavior.

### Evidence-limited routes
If reviewed evidence does not support a route-specific rehabilitation sequence, show the evidence limitation to the clinician and do not fabricate a complete treatment pathway.

### Context-gated routes
A `context_gated` route may generate referral text only when explicit clinician-entered subtype/context resolves to exactly one reviewed rich-referral variant. Missing, unresolved, unsupported or multiply-matched context must **block referral generation**. It must never degrade to the legacy checkbox/list formatter.

```text
context_gated + exact reviewed variant
→ rich referral may render

context_gated + no exact reviewed variant
→ formatter_blocked
→ explicit context/evidence validation state
→ text = null
```

### Pediatric / growth-related routes
Preserve age/skeletal-maturity and condition-specific load-management boundaries; do not apply adult evidence by silent extrapolation.

---

# 8. Short vs Detailed

Both outputs must be generated from the same reviewed route truth.

**Short:** flowing prose with clinical context plus the minimum useful rehabilitation direction. It may use direct referral language such as `Παρακαλώ για ...` when this is the product-owner approved physician-referral register.

**Detailed:** may be either:

```text
staged rehabilitation layout
```

or:

```text
ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ
ΣΤΟΧΟΙ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
ΚΑΤΕΥΘΥΝΣΗ ΦΥΣΙΟΘΕΡΑΠΕΙΑΣ
ΕΠΑΝΕΚΤΙΜΗΣΗ
```

when a disease does not justify artificial stage labels. Both layouts remain data-driven through the shared renderer; no disease-specific Python formatter branch is authorized.

---

# 9. Global acceptance tests

Required gates:

```text
COVERAGE
- every registry route resolves to either an applicable rich plan or an explicit evidence-limited/block state
- no generic cross-route evidence fallback
- every context-gated route with unresolved/unsupported context returns formatter_blocked and no referral text
- direct formatter calls cannot bypass the context gate into legacy prose

CONTENT
- no orphan goals
- goal → intervention linkage
- materially route-specific content
- no artificial stage numbering when section layout is selected
- passive-only care cannot satisfy routine rehabilitation unless route/protocol prevents active progression

PRECISION
- detailed <= 2000 chars
- normal detailed target <= 1850 chars
- no universal sets/reps/kg/hold times
- no unsupported fixed weeks or numeric progression/discharge thresholds

FACTUALITY
- no inferred occupation tasks
- no unselected findings/deficits
- no diagnosis from findings alone
- not assessed != normal
- optional context values marked uncertain/not assessed must not be printed as if clinically established

EVIDENCE GOVERNANCE
- every evidence-derived rendered element resolves to applicable authority
- therapist execution detail does not leak into referral
- evidence gaps are visible, not filled

DUAL OUTPUT
- Short and Detailed reflect the same plan truth

CLINICIAN EVIDENCE UI
- references are human readable
- evidence gaps/freshness visible
- panel content is excluded from referral copy/print by default
```

---

# 10. Exact implementation sequence

1. Add deterministic evidence-corpus resolver and clinician evidence API payload.
2. Add clinician-only Evidence panel to the current UI.
3. Introduce shared `RichReferralPlan` composer/renderer seam.
4. Migrate LET from custom formatter logic into the shared seam with output-equivalence tests.
5. Resolve registry routes horizontally from structured route/evidence data; explicit gap state where unsupported.
6. Add representative cross-route tests and an all-registry coverage gate.
7. Run exact-head CI.
8. Review representative outputs and evidence panel with the product owner before merge/deploy.

---

# 11. Product-acceptance correction — context-gated fallback — 2026-08-30

Product-owner browser review demonstrated that unresolved rich authority must never fall through to the legacy formatter.

```text
SELECTION INPUT != REFERRAL TEXT
UNRESOLVED CONTEXT != PERMISSION FOR LEGACY FALLBACK
CONTEXT-GATED FAILURE MUST BLOCK GENERATION
```

Regression coverage reproduces real clinician behavior, including deliberate omission of a context field, and asserts the final API generation state.

---

# 12. Product-acceptance correction — frozen-shoulder referral wording — 2026-08-30

The primary frozen-shoulder output is the first route explicitly approved for a section-based Detailed layout rather than an artificial single-stage layout.

Approved product boundaries:

- Greek-only referral prose;
- direct physician-referral register (`Παρακαλώ για ...`);
- optional explicitly clinician-entered irritability may appear as `Κλινική ερεθιστικότητα: υψηλή/μέτρια/χαμηλή`;
- uncertain/not-assessed irritability is omitted from referral prose;
- mobility/available ROM/function are the referral core;
- mobilization may be supplementary when appropriate;
- exact exercise/technique choice and dosing remain physiotherapist-owned;
- no routine strengthening recommendation;
- no natural-history evidence commentary in the referral;
- no fixed phases/weeks/numeric transition criteria;
- no `ΣΤΑΔΙΟ 1` when there is no true staged pathway;
- medical reassessment remains compact and explicit.

The shared renderer therefore supports a generic `detailed_sections_el` content structure in addition to staged content. This is a reusable layout capability, not a frozen-shoulder-specific formatter branch.
