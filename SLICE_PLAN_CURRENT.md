# SLICE_PLAN_CURRENT.md — G-3 Guidance Salience + Longitudinal Patient Summary v1

> **STATUS:** ACTIVE DESIGN / IMPLEMENTATION.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** `M01-G3-GUIDANCE-SALIENCE-LONGITUDINAL-SUMMARY-v1`.
> **Fresh base main:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **Implementation branch:** `feat/module01-g3-guidance-salience-longitudinal-summary-2026-09-01`.
> **Runtime writer:** this session, bounded to G-3 files/seams.

---

# 1. Trigger / evidence from use

Product-owner authenticated G-2 production smoke passed on the deployed exact runtime ancestry.

During that smoke the product owner identified two concrete workflow needs:

1. newly applicable guidance should become visibly more salient at the moment it appears; example: VFA/vertebral-imaging guidance after current height loss reaches the evidence trigger of at least 4 cm;
2. a concise patient-level longitudinal summary should remain visible throughout the encounter and summarize authoritative history from first completed/amended encounter through the latest reliable state.

This is a product-UX/read-only longitudinal slice. It does not reopen G-2 evidence content.

---

# 2. Objectives

## G3-A — newly surfaced guidance salience

When the deterministic Visit Plan gains a card/domain after a current-state or history update, mark it as newly surfaced for the active patient/case.

Requirements:

- initial plan on page load is baseline and is **not** labelled new;
- plan diff is by canonical card/domain identity, not DOM position;
- newly surfaced item gets a visible `Νέο` marker plus stronger border/background emphasis;
- color must not be the only signal;
- critical/event content may keep its stronger semantic styling;
- salience is ephemeral browser state only; no authoritative patient persistence;
- salience disappears when the item stops applying; optional acknowledgement may clear it during the current page session;
- changing unrelated fields must not repeatedly recreate `Νέο` for an already surfaced card.

Primary example fixture:

```text
height loss <4 cm
→ VFA guidance absent
→ current live height/reference values cross to >=4 cm
→ VFA guidance appears
→ VFA card + top summary item visibly marked Νέο
```

## G3-B — always-visible longitudinal patient summary

Render a compact `Σύνοψη ασθενούς` above the dynamic visit flow whenever a protected patient is active.

Authoritative inputs:

```text
completed/amended protected encounters
+ protected lab snapshots
+ existing LongitudinalGuidanceProjectionV1
+ current visit snapshot clearly labelled as current/non-historical
```

Minimum visible domains:

1. **Πορεία** — first encounter date, latest completed/amended encounter date, encounter count.
2. **Κατάγματα / κίνδυνος** — known fracture burden/most recent reliable event and latest explicit formal risk state when available.
3. **DXA** — latest reliable DXA date and key T-scores; no significance claim without comparability/LSC.
4. **Θεραπεία** — active/latest reliable treatment episode, actual administration history, reliable last actual dose and count when available.
5. **Εργαστηριακά** — latest protected lab date and concise clinically useful key values when present.
6. **Τελευταία απόφαση** — latest explicit reliable management decision/selected agent when present, without treating a discussed option as final decision.
7. **Εκκρεμότητες / conflicts** — unresolved tasks, unresolved critical close state and longitudinal conflicts.

The summary must be useful at a glance but must not become a second editable form.

---

# 3. Truth / conflict rules

```text
READ-ONLY SUMMARY != NEW SOURCE OF CLINICAL TRUTH
LATEST BLANK != ERASE PRIOR AUTHORITATIVE FACT
MISSING != NEGATIVE
CONFLICT != CHOOSE LATEST SILENTLY
SCHEDULED DOSE != ACTUAL DOSE
DISCUSSION/OPTION != FINAL DECISION
CURRENT DRAFT != HISTORICAL COMPLETED FACT
```

When history is unavailable, render an explicit unavailable/partial state and never say `0 prior visits` as if proven.

Fracture events, DXA snapshots and treatment/admin history must use stable/exact identities where available and avoid double counting repeated snapshots.

No AI free-text summarization in v1.

---

# 4. Runtime ownership

Existing owners remain:

- protected history fetching and top guidance rendering: `static/baseline-audit/progressive-guidance-ui.js`;
- generic longitudinal treatment/admin/task projection: `static/baseline-audit/progressive-guidance-core.js` + `schemas/longitudinal_guidance_projection_v1.yaml`;
- G-2 evidence evaluation: `static/baseline-audit/osteoporosis-evidence-guidance-core.js`;
- patient registry/authenticated server access: `static/baseline-audit/patient-registry.js`.

G-3 may add one **pure deterministic summary core** but must not create a second network/history owner or second Visit Plan renderer.

Preferred new pure module:

```text
static/baseline-audit/osteoporosis-longitudinal-summary-core.js
```

It receives data; it does not fetch, render, persist, or mutate patient truth.

---

# 5. Data seam / API boundary

Existing protected endpoints are sufficient:

```text
GET /clinical/patient/{patient_id}/encounters
GET /clinical/patient/{patient_id}/labs
```

No DB migration or new API endpoint is authorized for v1.

The guidance UI may extend its current history refresh to fetch encounters + labs once for the active patient and pass them to the pure summary core.

---

# 6. UI contract

Top order:

```text
Patient Registry / active patient
→ Σύνοψη ασθενούς
→ Σημερινή ροή
→ step tabs / clinical cards
```

`Σύνοψη ασθενούς` is always present for an active protected patient and does not depend on encounter archetype.

Recommended presentation:

- compact header with first→latest dates and count;
- responsive grid of concise domain tiles;
- explicit muted `Δεν έχει τεκμηριωθεί` / `Μη διαθέσιμο` / `Ασυμφωνία` states;
- no hidden claim that a missing element is normal;
- no color-only semantics.

New guidance salience:

- summary item class `is-newly-surfaced`;
- destination card/WHY-NOW class `is-newly-surfaced`;
- text badge `Νέο`;
- accessible contrast and no required animation.

---

# 7. Acceptance tests

## Salience

- initial render does not mark all cards new;
- <4 cm → ≥4 cm live height-loss transition newly surfaces VFA and marks it `Νέο`;
- repeat render with unchanged plan does not create a duplicate new transition;
- item removal clears new state;
- switching patient/case resets baseline correctly;
- G-1/G-2 WHY-NOW/provenance remains intact.

## Summary

- chronological first/latest encounter dates and count correct;
- history unavailable != zero visits;
- later blank does not erase prior DXA/treatment/risk fact;
- latest reliable DXA and risk state selected deterministically;
- scheduled-only administration not counted as actual;
- treatment conflict shown as conflict rather than invented current agent;
- latest lab snapshot displayed from protected labs;
- current draft is visually distinct from completed/amended historical facts;
- no authoritative write occurs from rendering.

## Inherited gates

- G-2 evidence guidance contract/runtime regressions pass;
- G-1 progressive guidance regressions pass;
- C1 authoritative Finish/finalization regressions pass.

---

# 8. Out of scope

- new G-2 clinical rules or thresholds;
- AI-generated narrative summary;
- patient-level canonical treatment DB table;
- transcript extraction / PR-1;
- inline transcript population / PR-2;
- Practice Review scoring;
- real 5-case pilot;
- physiotherapy/RF work.

---

# 9. REPLAN triggers

Stop and redesign if inspection shows any of:

- protected historical encounter payloads do not contain enough reliable structured information for a safe summary;
- summary requires silently reconciling materially conflicting values;
- obtaining labs would create duplicate network ownership or bypass protected auth;
- new-guidance detection cannot be based on stable Visit Plan/card identity;
- requested always-visible placement interferes with authoritative Finish/card ownership.

No such trigger has been found in the initial seam inspection.

---

# 10. Stop gate

Authorized work may proceed through:

```text
IMPLEMENTED
→ TESTED
→ exact-head review
```

STOP before PR/merge/deploy unless separately authorized by the product owner.
