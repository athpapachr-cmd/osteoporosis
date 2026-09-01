# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-2 PRODUCTION-SMOKE-VERIFIED / G-3 GUIDANCE SALIENCE + LONGITUDINAL SUMMARY ACTIVE.
> **Updated:** 2026-09-01 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **G-2 release PR:** `#69` — squash merged.
> **G-2 merge/runtime SHA:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **G-2 Render deploy:** `dep-daaph5vlk1mc73940g60` — `live`, trigger `new_commit`, exact commit `9cfad82d...`.
> **G-3 branch:** `feat/module01-g3-guidance-salience-longitudinal-summary-2026-09-01`.
> **ACTIVE CANONICAL WRITER/LOCK:** this session — G-3 bounded UX/read-only longitudinal scope.
> **ACTIVE RUNTIME WRITER/LOCK:** this session — only the G-3 seams defined below.

---

# 1. G-2 release closeout

The product owner completed the requested authenticated synthetic production smoke on the deployed G-2 ancestry and reported the smokes passed.

Therefore the exact state is:

```text
G-2 DESIGN-COMPLETE = YES
G-2 IMPLEMENTED = YES
G-2 TESTED = YES
G-2 RELEASE REVIEW = PASS
G-2 MERGED = YES
G-2 DEPLOYED = YES
G-2 PRODUCTION-SMOKE-VERIFIED = YES
G-2 PILOT-VALIDATED = NO
```

Do not relabel this product-owner smoke as real-clinic pilot validation.

The G-2 clinical safeguards remain frozen, including inactive R15/R16 and no automatic CTX threshold retreatment command, ordinal Prolia milestones, automatic treatment failure/switch, selected-agent mutation, or automatic named-specialty referral.

---

# 2. Product-owner evidence-from-use after G-2 smoke

Two concrete UX needs were identified during production interaction:

1. When a guidance item becomes newly applicable because current structured data trigger it, it should become visually more salient. Example: VFA/vertebral-imaging guidance when derived height loss reaches at least 4 cm.
2. A concise longitudinal patient summary should always be visible and describe the patient's authoritative course from first completed/amended encounter through the latest reliable state, instead of forcing the clinician to reconstruct the history from separate cards/visits.

These observations are treated as real product evidence, not cosmetic preference only.

---

# 3. Active G-3 slice

Slice ID:

```text
M01-G3-GUIDANCE-SALIENCE-LONGITUDINAL-SUMMARY-v1
```

Bounded objectives:

```text
A. newly surfaced guidance salience
B. always-visible deterministic longitudinal patient summary
```

Hard boundaries:

```text
NO NEW TREATMENT RULES
NO CHANGE TO G-2 EVIDENCE THRESHOLDS
NO AUTOMATIC TREATMENT DECISION
NO NEW AUTHORITATIVE PATIENT WRITE
NO DB MIGRATION
NO AI-GENERATED LONGITUDINAL TRUTH
NO PR-1 / PR-2 TRANSCRIPT WORK
NO PHYSIOTHERAPY / RF MUTATION
```

---

# 4. G-3 design direction

## Guidance salience

`newly surfaced` means an item/domain is absent from the previous stable Visit Plan for the same active patient/case and becomes present after a current-state/history update.

Initial page load does not mark the entire first plan as new.

A newly surfaced item must use both visual emphasis and a textual marker such as `Νέο`, so color is not the only signal. The emphasis is ephemeral/in-memory and does not create new clinical persistence.

## Longitudinal patient summary

Use protected completed/amended encounter payloads plus protected lab snapshots and the existing read-only longitudinal projection. The summary remains derived/read-only and fail-closed.

Minimum sections:

```text
course: first visit → latest visit / encounter count
fractures + current risk context
latest reliable DXA state
therapy timeline / active treatment / actual administrations
latest relevant labs
current unresolved tasks / conflicts
latest reliable management decision when present
```

Missing later values do not erase prior authoritative history. Conflicting history is shown as conflict/uncertain rather than silently resolved.

---

# 5. Existing runtime seams verified

No REPLAN trigger found during initial inspection.

Existing components already support the bounded implementation:

```text
GET /clinical/patient/{patient_id}/encounters
GET /clinical/patient/{patient_id}/labs
schemas/longitudinal_guidance_projection_v1.yaml
static/baseline-audit/progressive-guidance-ui.js
static/baseline-audit/progressive-guidance.css
static/baseline-audit/patient-registry.js
static/baseline-audit/longitudinal.js
```

`progressive-guidance-ui.js` already owns the protected historical encounter fetch and top `Σημερινή ροή` rendering. G-3 must not introduce a second competing guidance renderer or duplicate authoritative history store.

---

# 6. Exact next action

```text
freeze G-3 slice design
→ implement deterministic summary core + minimal UI integration
→ implement newly-surfaced diff/salience state
→ focused regressions
→ inherited G-2/G-1/C1 regressions
→ exact-head review
→ STOP before PR/merge/deploy unless separately authorized
```
