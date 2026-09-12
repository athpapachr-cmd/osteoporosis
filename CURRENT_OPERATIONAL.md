# CURRENT_OPERATIONAL.md — Knee-OA usability refinement v2

> **STATUS:** PRODUCT-OWNER USABILITY REFINEMENT — IMPLEMENTATION ACTIVE / SYNTHETIC PROTOTYPE ONLY.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Freshly verified `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Parent closed candidate:** `0f38f4d411146667d854c32c9f5f639f344d7c7f`.
> **Parent tested substantive amendment:** `83a5acd4e5b25413708bbda1715c58b5c78bce08`.
> **Implementation branch:** `feat/physio-knee-oa-usability-refine-v2-2026-09-12`.
> **Active slice:** `CU1-PRODUCT-KNEE-OA-USABILITY-REFINE-V2-20260912`.
> **ACTIVE RUNTIME / DESIGN / CANONICAL WRITER:** this bounded usability-refinement session.
> **PR / merge / deploy / production smoke authority:** NONE.
> **Real-patient use / production integration:** NOT AUTHORIZED.

## 1. Product Owner authority

After visually reviewing the tested post-review amendment, the Product Owner authorized three bounded usability changes:

1. make additional suggestions materially more discoverable without expanding every suggestion inline;
2. expose referral-text editing directly instead of hiding it behind the overflow menu, while preserving manual-text reconciliation safety;
3. add Favorites / Pin-to-top inside `Περισσότερα`, without permanent Hide in this slice.

No clinical/evidence rule change is authorized by this feedback.

## 2. Exact implementation scope

### Additional suggestions

Keep the highest-priority suggestion visible as a compact actionable line. When more suggestions exist, show a visually distinct but restrained `Άλλες {n} προτάσεις` summary panel containing only short item titles. Activating the panel opens the existing suggestions sheet where Add / evidence / dismissal remain explicit.

Do not render all secondary suggestion rationales, citations or controls on the routine surface.

### Direct referral editing

Expose a visible `Επεξεργασία` action in the referral preview rather than requiring the `•••` menu.

The existing safety invariant remains:

```text
manual text != structured state
```

After manual editing, a later structured change must never overwrite the clinician's text silently. It must enter explicit reconciliation state; export remains guarded until the clinician chooses which version is authoritative.

The overflow menu may remain for secondary actions but is no longer the sole route to editing.

### Favorites / Pin to top

Inside `Περισσότερα`, allow optional doctor-facing pin/favorite state for useful advanced controls.

First-slice rules:

- favorite/pin changes ordering/discoverability only;
- favorite/pin never selects a clinical item;
- favorite/pin never changes referral output, evidence state, suggestion eligibility or safety;
- no permanent Hide in this slice;
- no patient data is attached to favorite state;
- the synthetic prototype keeps favorites draft/session-local only: no localStorage/sessionStorage/server persistence;
- future account-level persistence is a separate preference-storage decision.

## 3. Hard exclusions

```text
NO clinical finding/evidence changes
NO permanent Hide
NO patient-state persistence
NO localStorage/sessionStorage
NO account/preferences backend
NO second diagnosis
NO jurisdiction expansion
NO billing/auth/analytics
NO production integration
NO PR/merge/deploy
```

## 4. Acceptance gate

Fresh exact-head tests must prove at minimum:

- one primary suggestion remains actionable inline;
- additional suggestions are conspicuous, correctly counted and title-previewed, and open the full suggestions sheet;
- no secondary suggestion is auto-selected by the summary panel;
- `Επεξεργασία` is directly visible in desktop preview and reachable in mobile preview;
- manual edited text survives subsequent structured changes and enters explicit reconciliation rather than being overwritten;
- Copy/Print remain blocked for stale manual text until reconciliation;
- favorites reorder advanced content without changing structured clinical selection;
- favorites do not survive reset/BFCache in the synthetic prototype and create no storage entries;
- no Hide control exists;
- inherited safety, evidence-conflict, network-failure, mobile-reflow and privacy tests remain green.

## 5. Exact next action

Implement these three bounded UX changes in the synthetic Knee-OA prototype, run focused real-CU1/browser regression, inspect generated screenshots, produce a tested artifact for Product Owner review, then release the writer.
