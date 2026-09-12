# SLICE_PLAN_CURRENT.md — Knee-OA usability refinement v2

> **STATUS:** IMPLEMENTATION ACTIVE / SYNTHETIC PROTOTYPE ONLY.
> **Slice:** `CU1-PRODUCT-KNEE-OA-USABILITY-REFINE-V2-20260912`.
> **Branch:** `feat/physio-knee-oa-usability-refine-v2-2026-09-12`.
> **Parent closeout head:** `0f38f4d411146667d854c32c9f5f639f344d7c7f`.
> **Parent tested substantive head:** `83a5acd4e5b25413708bbda1715c58b5c78bce08`.
> **Fresh main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** ACTIVE, bounded to prototype UX/tests/workflow/canonicals.
> **Release / real clinical use:** NOT AUTHORIZED.

## 1. Design objective

Improve discoverability and personalization without increasing routine clinical complexity.

The slice deliberately changes **interaction presentation**, not clinical semantics.

Preserve:

```text
selection != suggestion != evidence != safety
manual text != structured state
favorite != selection
clinically interesting != worth adding
```

## 2. Additional-suggestion discoverability

Current problem: the highest-priority suggestion is visible, while the remaining count can be visually lost.

Design:

```text
primary suggestion
→ compact actionable line

if additional_count > 0
→ restrained bordered summary panel
→ heading: Άλλες {additional_count} προτάσεις ›
→ preview: short titles only
→ activate anywhere → existing suggestions sheet
```

Rules:

- count means suggestions **in addition to** the primary visible suggestion;
- preview titles are generated from the current eligible candidate list and cannot become stale authority;
- no Add/dismiss/evidence mutation occurs from tapping the summary itself;
- the sheet remains the place for individual Add, evidence and dismissal controls;
- no full rationale/citation stack on the routine surface.

## 3. Direct referral-text editing

Current problem: manual edit is a major clinician action but is hidden in the overflow menu.

Design:

- desktop referral preview exposes a visible `Επεξεργασία` action in the preview heading;
- mobile preview sheet also exposes a visible `Επεξεργασία` action;
- activation uses the existing manual-edit buffer/reconciliation architecture;
- the overflow menu can retain secondary actions but is not required to discover editing.

Safety invariants:

```text
manual edit occurs
→ manual buffer becomes clinician-owned text

structured state later changes
→ manual buffer is preserved
→ export becomes stale/blocked
→ explicit reconciliation required
→ no silent overwrite
```

Do not build bidirectional free-text-to-structured parsing or merge/diff machinery.

## 4. Favorites / pin-to-top inside Περισσότερα

Current problem: advanced content can be long, while different clinicians repeatedly use different subsets.

Design intentionally chooses **pin/favorite**, not Hide.

### Behavior

- advanced controls that represent selectable findings/goals/interventions/adjuncts may expose a small `☆` / `★` pin control where technically appropriate;
- pinned items appear in a compact `★ Συχνά` group at the top of `Περισσότερα`;
- original category remains authoritative and available; pinning must not erase the item from discoverability;
- duplicate interactive selectors must not create conflicting selection state: the pinned surface may either reference the same action semantics safely or use a single rendered control ownership pattern;
- unpin returns the item to normal ordering with no clinical state mutation.

### Persistence boundary

For this synthetic slice:

```text
favorites = ephemeral UI preference only
reset/pagehide/BFCache → cleared
localStorage = none
sessionStorage = none
server persistence = none
```

Later product architecture may persist favorites as **clinician account preference**, explicitly separate from patient/referral state.

### No Hide yet

No permanent hide/archive control is introduced. Hide remains a future hypothesis requiring usage evidence and an always-recoverable design.

## 5. Accessibility and cognitive-load rules

- additional-suggestion panel is a native button with descriptive accessible name/count;
- favorites controls have accessible names such as `Προσθήκη στα Συχνά: {item}` / `Αφαίρεση από τα Συχνά: {item}`;
- star icon is never the only meaning;
- touch targets remain ≥44 CSS px;
- direct edit remains keyboard-reachable;
- mobile reflow and large-text behavior must not regress;
- no colour-only state.

## 6. Out of scope

```text
clinical/evidence rule changes
new clinical fields
permanent Hide
favorites account persistence
preference backend
analytics
second diagnosis
local guideline expansion
PR/merge/deploy
```

## 7. Acceptance

Focused tests must cover:

1. 1 suggestion → no additional panel.
2. 5 suggestions → primary line + `Άλλες 4 προτάσεις` with four current titles.
3. Additional panel → opens full suggestions sheet without selecting anything.
4. Add/dismiss in sheet → count/title preview refreshes deterministically.
5. Direct desktop edit action → manual editor reachable without overflow menu.
6. Mobile preview → edit action directly visible.
7. Manual edit + laterality/finding change → clinician text preserved + stale reconciliation + export blocked.
8. Favorite item → appears in `★ Συχνά` and selection state is unchanged.
9. Selecting through favorite representation → same structured state as normal item, no duplicate/double toggle.
10. Unfavorite → clinical selection preserved.
11. Reset/BFCache → favorites cleared.
12. local/session storage remain empty.
13. No Hide control exists.
14. Inherited safety/evidence/privacy/network/mobile regressions remain green.

## 8. Exact next action

Implement only this bounded usability slice, run the exact prototype gate plus new browser coverage, inspect screenshots and return the tested candidate to the Product Owner for visual/use review.
