# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE — CU-1 GLOBAL RICH PHYSIOTHERAPY REFERRAL + CLINICIAN EVIDENCE PANEL.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Approved LET prototype parent:** `feat/cu1-rich-referral-lateral-elbow-2026-08-29` @ `aebfb5a6ee14a0e44d80dd6183a1877d74567b46`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **ACTIVE RUNTIME WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **Runtime authorization:** GLOBAL HORIZONTAL RICH-REFERRAL MODEL + CLINICIAN-ONLY EVIDENCE VIEW + TESTS.
> **Latest product-correction runtime head:** `917f38704745aeec48d8e332bdf5f1d23c82a26d`.
> **Latest focused CI evidence:** workflow run `33303230721` / run #389 — compile PASS, browser JavaScript syntax PASS, focused Python suite PASS.
> **Deploy/merge authorization:** NO — product-owner representative output review first.
> **Preview deployment:** NOT REQUESTED / NOT AUTHORIZED by product owner.
> **Further route-by-route evidence rollout:** HOLD until current frozen-shoulder product text and hierarchical UI behavior are accepted as the quality reference.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused.

---

# 1. Product-owner approval — LOCKED

On 2026-08-29 the product owner approved the lateral-elbow referral product shape and authorized horizontal application across the physiotherapy condition registry using the corresponding guideline/literature authority for each route.

The global model must not be implemented as one custom formatter branch per disease. It must use one shared rehabilitation-document model with route/subtype/context-specific content supplied as structured data/evidence authority.

The approved referral principles are:

```text
COMPACT CLINICAL CONTEXT
+
EARLY MANAGEMENT / INITIAL CAPACITY
+
PROGRESSIVE CAPACITY RESTORATION
+
FUNCTIONAL RETURN
+
RECURRENCE-RISK REDUCTION / SELF-MANAGEMENT
```

Stage structure is optional document organization, not a universal evidence claim. When a condition does not justify staged progression, the shared renderer may use a section-based Detailed layout instead.

---

# 2. GeSY output policy

Hard ceiling: `2000 characters`.

Target for ordinary Detailed output: `~1500–1850 characters`.

Clinical context may include only information actually supplied/selected. Do not invent patient facts, tasks, causality, severity or deficits. Short output is coherent prose; Detailed output is compact, clinically useful physician-to-physiotherapist communication.

---

# 3. Global rehabilitation invariants

1. **Goal without method is insufficient.**
2. **Passive-only care is not a complete routine rehabilitation plan** unless route/protocol context explicitly prevents active progression.
3. **No micromanagement.** No universal sets, repetitions, kilograms, hold times, fixed weeks or unsupported numeric thresholds.
4. **Therapist execution remains therapist-owned.** `therapist_execution_detail` must not automatically enter referral prose.
5. **Evidence applicability is route/subtype/context-specific.** No neighboring-route evidence borrowing.
6. **Evidence gap is explicit.**
7. **Patient-specific protocol wins.**
8. **Short and Detailed are two renderings of the same clinical truth.**
9. **Functional return/self-management remain represented when clinically applicable**, without forcing a visual three-stage layout.
10. **Reassessment remains compact but preserved** where clinically appropriate.
11. **Selection input is not referral prose.**
12. **Context-gated unresolved means generation blocked.**
13. **Clinical condition knowledge is reusable system knowledge.** Condition-specific assessment/evidence learned while building a utility must be reusable by the future condition card, follow-up, referral projection, evidence view and learning/audit surfaces.
14. **Clinical card owns patient-specific facts; referral is a projection.** Future card-to-referral prefill may reuse clinician-reviewed data, but referral editing must not silently mutate the underlying clinical record.
15. **Evidence complexity may be high internally while referral prose remains clinically simple.** Evidence limitations should constrain generated content without automatically becoming explanatory prose in the referral.

---

# 4. Evidence-source policy

The structured CU-1 evidence corpus remains the source of literature identity, applicability, recommendation direction, strength/certainty, freshness and evidence-gap state.

```text
referral_core
therapist_execution_detail
clinician_ui_only
```

Only `referral_core` authority may automatically support referral treatment prose. Execution detail remains physiotherapist-owned. Clinician-only evidence belongs in the evidence view rather than routine GeSY text.

---

# 5. Clinician-only evidence panel — AUTHORIZED

The separate `Τεκμηρίωση / Παραπομπές` panel remains outside referral copy/print and may show source, year/version, DOI/link, recommendation strength/certainty, claim applicability, freshness and evidence gaps/conflicts.

---

# 6. Current implementation boundary

Authorized now:
- shared/global rich-referral document model;
- route/subtype/context evidence resolution;
- Short + Detailed rendering under 2000 characters;
- staged or section-based Detailed layout selected from structured route content;
- explicit evidence-gap behavior instead of generic fallback;
- clinician-only evidence panel/API;
- deterministic cross-route tests;
- presentation-only hierarchical UI relevance resolution from profile → route → subtype → explicit context;
- reusable condition-knowledge architecture for future clinical-card ↔ referral linkage.

Not authorized now:
- direct merge to `main`;
- deployment or preview deployment without explicit product-owner instruction;
- weakening route/subtype safety gates;
- runtime interpretation of profile Markdown for trigger/validation logic;
- inventing evidence to make every route appear complete;
- persistence changes;
- CU-2 or PR-1 restart.

## 6.1 Context-gated legacy fallback — CORRECTED / TESTED

For every context-gated route:

```text
exactly one reviewed rich variant resolves
→ rich referral allowed

missing / unresolved / unsupported context
→ rich_referral_context_required
→ formatter_blocked = true
→ API text = null
→ direct formatter legacy fallback forbidden
```

The earlier wording `fail closed to the non-rich path` is superseded. Non-rich legacy fallback is not fail-closed behavior.

## 6.2 Frozen-shoulder context + hierarchical UI — IMPLEMENTED / TESTED

Primary frozen shoulder requires explicit formal diagnosis + `frozen_shoulder_scope=primary_frozen_shoulder` for rich rendering.

`frozen_shoulder_irritability` is optional and clinician-entered:

```text
high
moderate
low
uncertain_or_not_assessed
```

It is never inferred from pain/ROM selections. High/moderate/low may be projected into the referral; uncertain/not assessed is not printed as an established patient attribute.

The generic UI hierarchy is:

```text
profile
→ route
→ subtype
→ explicit context
```

Reviewed shoulder rich routes use condition-relevant findings/functions rather than the full generic shoulder catalogue. Generic goals/rehab-direction/adjunct checkbox blocks are hidden where the rich evidence plan owns composition.

## 6.3 Frozen-shoulder physician-referral wording — PRODUCT-APPROVED / IMPLEMENTED / TESTED

On 2026-08-30 the product owner reviewed and approved the physician-facing wording direction for primary frozen shoulder.

The approved Short register includes direct referral language:

```text
Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση ...
```

The approved Detailed organization is:

```text
ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ
ΣΤΟΧΟΙ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
ΚΑΤΕΥΘΥΝΣΗ ΦΥΣΙΟΘΕΡΑΠΕΙΑΣ
ΕΠΑΝΕΚΤΙΜΗΣΗ
```

The prior artificial single `ΣΤΑΔΙΟ 1` presentation is removed from rendered frozen-shoulder Detailed output. The shared renderer now supports generic structured `detailed_sections_el`; this is a reusable renderer capability, not disease-specific Python branching.

Frozen-shoulder referral product boundaries now locked:
- Greek-only prose;
- no natural-history superiority commentary;
- no routine strengthening recommendation;
- no fixed disease phases/weeks or numeric transition/discharge thresholds;
- no English internal evidence terminology;
- mobilization may be complementary when appropriate;
- exact exercise/technique selection and dosing remain physiotherapist-owned;
- medical reassessment remains explicit;
- uncertain/not-assessed irritability is omitted from referral text.

Regression coverage now checks the realistic selected case and the above wording boundaries.

Exact runtime/test acceptance:

```text
917f38704745aeec48d8e332bdf5f1d23c82a26d
workflow run 33303230721 / run #389
compile PASS
browser JavaScript syntax PASS
focused Python suite PASS
```

---

# 7. Exact next action

1. Do not create a preview service unless the product owner explicitly asks later.
2. Do not merge/deploy yet.
3. Do not resume broad horizontal route rollout yet.
4. Next product step is to inspect the **actual controlled Short and Detailed frozen-shoulder outputs** from the accepted renderer wording, including the realistic high-irritability case, and decide whether the clinical-context opening itself needs one final language-compression pass.
5. Once frozen shoulder is accepted as a text-quality reference, apply the same product-text review discipline to already promoted rich routes before continuing new disease rollout.
6. Reusable condition knowledge discovered during later route review must be classified for future clinical-card use rather than stored only as referral-specific prose.
