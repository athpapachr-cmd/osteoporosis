# SLICE_PLAN_CURRENT.md — Knee-OA `Περισσότερα` visual + IA redesign v3

> **STATUS:** IMPLEMENTATION ACTIVE / SYNTHETIC PROTOTYPE ONLY.
> **Slice:** `CU1-PRODUCT-KNEE-OA-MORE-REDESIGN-V3-20260912`.
> **Branch:** `feat/physio-knee-oa-more-redesign-v3-2026-09-12`.
> **Parent closed head:** `2c19eb7283c96af7868690a892d7e61155674d6b`.
> **Fresh main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** ACTIVE, bounded to prototype UI/tests/workflow/canonicals.
> **Release / real clinical use:** NOT AUTHORIZED.

## 1. Problem

The v2 `Περισσότερα` retains too many simultaneously visible controls. Favorites help frequent access but do not solve the underlying scanning cost because categories/items still compete with similar visual weight.

The redesign must preserve the full capability set while reducing routine visual decisions.

## 2. Design principle

```text
CAPABILITY DEPTH UNDERNEATH
+
CALM SURFACE ABOVE
```

The user should not need to visually parse the entire advanced clinical vocabulary to find one item.

## 3. Routine surface

### `★ Συχνά`

- shown only when one or more items are pinned;
- intended for a small number of clinician shortcuts;
- compact card/row treatment, not full duplicate category rendering;
- pinning never changes clinical selection;
- normal mode does not spray star icons across all controls;
- `Προσαρμογή Συχνών` explicitly enters pin-management mode.

### `Σχετικά τώρα`

- maximum small set (target <=3 visible shortcuts);
- deterministic from already-known structured state only;
- each shortcut maps to an existing advanced option;
- contextual surfacing never auto-selects or infers a finding;
- absent context simply means the section is absent.

### `Όλα`

The complete advanced vocabulary remains available through collapsed category rows. Candidate category structure:

```text
Εξέταση
Λειτουργία
Αποκατάσταση
Συμπληρωματικά
Κλινικός έλεγχος
```

Each row shows:

- category name;
- restrained icon;
- selected count if >0;
- short summary of selected items where useful;
- chevron.

Activating a category opens its full controls in the existing modal/sheet host. Do not show every advanced chip on the routine surface.

## 4. Visual grammar

- section headers are typographic anchors, not bordered cards;
- category rows use larger hit areas and generous vertical spacing;
- selected summary text is muted and truncates gracefully;
- counts are textual/quiet rather than bright badges;
- avoid card-inside-card-inside-card nesting;
- one accent color remains enough for selection/action;
- clinical evidence colours retain their existing semantics and are not reused decoratively;
- iconography is supporting, never the only label;
- desktop and mobile use the same conceptual hierarchy.

## 5. Selected-state visibility

A collapsed category must still answer “what have I already chosen?” without opening it.

Example:

```text
Εξέταση
Effusion · Extension lag
2 ενεργά                                  ›
```

This summary is derived from current structured state. It never becomes a second source of truth.

## 6. Preserve all content

No existing supported advanced clinical option is removed from reachability. The redesign may relocate controls into category detail sheets but not delete/rename clinical IDs or silently suppress selected state.

Safety and readiness remain governed outside browsing convenience. Safety-critical blocking behavior cannot depend on whether a category is open.

## 7. Explicit exclusions

```text
search
permanent Hide
recent-items system
multiple preference tabs
favorite persistence
AI-generated relevance
clinical/evidence rule edits
second diagnosis
PR/merge/deploy
```

## 8. Test acceptance

Focused browser tests must verify:

1. routine advanced surface renders category rows rather than all advanced controls;
2. all existing advanced controls remain reachable through category detail;
3. selected category summary/count updates after selection and survives closing/reopening;
4. favorites are bounded shortcuts and normal mode has no star wall;
5. customization mode can pin/unpin without selecting/deselecting;
6. contextual `Σχετικά τώρα` appears only under defined deterministic conditions and never mutates state merely by appearing;
7. no local/session storage is introduced;
8. inherited manual edit, safety, evidence, qualifier and mobile tests stay green;
9. desktop/mobile screenshots show the intended calmer scanning hierarchy.

## 9. Exact next action

Implement, test and visually inspect this bounded redesign. Stop after a tested synthetic artifact for Product Owner review.