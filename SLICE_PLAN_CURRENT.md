# SLICE_PLAN_CURRENT.md — CU-1 global rich referral + evidence panel v1.17

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

The normalized render model is:

```text
RichReferralPlan
├── clinical_context
├── stages[]
│   ├── stage_label
│   ├── goals[]
│   ├── intervention_directions[]
│   └── progress_markers[]
├── adjunct_boundary_optional
├── reassessment_optional
├── evidence_state
└── evidence_source_ids[]
```

The default conceptual organization for ordinary musculoskeletal rehabilitation is:

```text
1. Early management / symptom or tissue irritability / mobility / initial activation
2. Capacity restoration / strength / endurance / control / progressive loading
3. Functional reintegration / actual activity demands / self-management / recurrence-risk reduction
```

This is **document organization**, not a universal three-phase evidence claim. Routes may use fewer/different stages when their actual authority, healing state, neurological context or written protocol requires it.

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

Stage 3 should include both:

```text
functional return
+
recurrence-risk reduction / self-management
```

when clinically appropriate. Typical components are load management, relevant ergonomic/technical adaptation, maintenance of required strength/endurance/control and a self-management strategy. These must not be falsely presented as proven recurrence-prevention interventions when the source only supports broader self-management/load modification.

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

For the currently selected route/subtype/context show:

```text
Core sources
- title
- organization/authors
- year/version
- reference
- DOI/link when available
- freshness / reviewed-on

Relevant claims
- human-readable claim summary
- recommendation direction
- strength/certainty when available
- domain
- whether it supports referral core or is clinician-only/execution detail

Evidence gaps / conflicts
- explicit limitations
- blocked evidence state where applicable
```

The panel must not expose internal IDs as its primary user-facing content.

---

# 7. Route-class exceptions

The global renderer must explicitly support exceptions rather than forcing every route through the LET pattern.

### Postoperative / fracture / tissue-healing
Written protocol, healing, weight-bearing, ROM or loading restrictions override generic active progression. Time may appear only when supplied by authoritative protocol/healing instruction.

### Neurological / nerve routes
Do not invent tendon-style loading stages. Progressive objective neurological deficit or unresolved localization/safety context preserves correct-owner reassessment behavior.

### Evidence-limited routes
If reviewed evidence does not support a route-specific rehabilitation sequence, show the evidence limitation to the clinician and do not fabricate a complete treatment pathway.

### Pediatric / growth-related routes
Preserve age/skeletal-maturity and condition-specific load-management boundaries; do not apply adult evidence by silent extrapolation.

---

# 8. Short vs Detailed

Both outputs must be generated from the same `RichReferralPlan`.

**Short:** flowing prose with beginning → progression → functional/prevention endpoint. It should retain the core intervention method and not collapse to generic goals.

**Detailed:** compact staged format with `Στόχοι`, `Κατευθύνσεις`, `Πρόοδος`; no transition paragraphs; no separate routine monitoring section; concise adjunct/reassessment tail.

---

# 9. Global acceptance tests

Required gates:

```text
COVERAGE
- every registry route resolves to either an applicable rich plan or an explicit evidence-limited/block state
- no generic cross-route evidence fallback

CONTENT
- no orphan goals
- goal → intervention linkage
- materially route-specific content
- Stage 3 return + self-management/recurrence-risk reduction when appropriate
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

EVIDENCE GOVERNANCE
- every evidence-derived rendered element resolves to applicable authority
- source class / recommendation direction / strength / certainty preserved
- conflicting frameworks are not silently hybridized
- therapist execution detail does not leak into referral
- evidence gaps are visible, not filled

DUAL OUTPUT
- Short and Detailed reflect the same plan truth

CLINICIAN EVIDENCE UI
- references are human readable
- DOI/source links available when stored
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
