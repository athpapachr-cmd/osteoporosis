# SLICE_PLAN_CURRENT.md — Knee-OA usability refinement v2

> **STATUS:** IMPLEMENTED / FOCUSED TECHNICAL GATE PASS / PRODUCT-OWNER VISUAL REVIEW NEXT.
> **Slice:** `CU1-PRODUCT-KNEE-OA-USABILITY-REFINE-V2-20260912`.
> **Branch:** `feat/physio-knee-oa-usability-refine-v2-2026-09-12`.
> **Parent closeout head:** `0f38f4d411146667d854c32c9f5f639f344d7c7f`.
> **Tested substantive head:** `a87dccc90dab9f50f70505f2565e3405d48de109`.
> **Successful substantive run:** `34675038243`.
> **Fresh main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** NONE.
> **Release / real clinical use:** NOT AUTHORIZED.

## 1. Design objective

Improve discoverability and personalization without increasing routine clinical complexity.

Preserved invariants:

```text
selection != suggestion != evidence != safety
manual text != structured state
favorite != selection
clinically interesting != worth adding
```

## 2. Implemented additional-suggestion design

```text
primary suggestion
→ compact actionable line

additional_count > 0
→ restrained bordered summary
→ Άλλες {additional_count} προτάσεις ›
→ short current titles only
→ activate → full suggestions sheet
```

The count excludes the already visible primary suggestion. Activating the summary does not select/dismiss anything. Individual Add, evidence and dismissal remain in the full sheet.

## 3. Implemented direct-edit design

Desktop live preview exposes a visible `✎ Επεξεργασία` action. Mobile preview sheet exposes the same direct action. The overflow menu remains a secondary route.

Safety invariant:

```text
manual edit
→ clinician-owned manual buffer

later structured change
→ preserve manual text
→ export stale/blocked
→ explicit clinician reconciliation
```

No reverse parsing, bidirectional synchronization or merge engine was added.

## 4. Implemented Favorites / pin-to-top

Advanced selectable content supports `☆` / `★` pinning. Pinned items appear in `★ Συχνά` at the top of `Περισσότερα`.

Rules:

- pinning changes ordering/discoverability only;
- pinning never selects a clinical item;
- favorite representation writes the same underlying structured state only when explicitly selected;
- unpinning preserves clinical selection;
- no Hide control exists;
- favorites are ephemeral in the synthetic prototype;
- reset/pagehide/BFCache clears them;
- localStorage/sessionStorage/server persistence remain absent.

Future persistence is a separate clinician-account preference decision.

## 5. Technical acceptance

First usability-v2 run:

```text
run     34674925907
head    b9889f85a1f4ad6f4da6050b6ebf34eefeb2bcb3
result  FAILURE
reason  inherited generic [data-edit] test locator became ambiguous after adding the direct edit route
```

Correction: the inherited regression now explicitly exercises the legacy menu edit route. The new usability-v2 suite independently exercises the direct route. Manual-text safety expectations were not relaxed.

Successful substantive gate:

```text
run                                       34675038243
head                                      a87dccc90dab9f50f70505f2565e3405d48de109
scope + syntax                            PASS
real CU-1 / HTTP                          15 / 15 PASS
frozen Step-3 exact-output fixtures       15 PASS
post-review clinical/output               11 / 11 PASS
inherited Chromium                        12 / 12 PASS
post-review qualifier Chromium             9 / 9 PASS
usability-v2 Chromium                      5 / 5 PASS
Greek source-summary coverage             54 positions
packaged dependency closure               PASS
```

The five new browser tests cover additional-suggestion count/titles/sheet behavior, single-suggestion absence of extra panel, direct edit/reconciliation on desktop/mobile, favorite semantics/no Hide/no storage, and reset/BFCache clearing.

## 6. Visual inspection

CI screenshots inspected:

- `desktop.png`
- `mobile.png`
- `evidence-conflict.png`
- `suggestions-v2.png`
- `favorites-v2.png`

The additional-suggestions summary is visible without expanding secondary details. The favorites surface stays inside `Περισσότερα`. Direct edit remains visually secondary to Copy. No obvious screenshot clipping was observed.

This inspection does not prove iPhone Safari/VoiceOver or full measured accessibility acceptance.

## 7. Out of scope remains

```text
clinical/evidence rule changes
new clinical fields
permanent Hide
favorite/account persistence
analytics
second diagnosis
jurisdiction expansion
PR/merge/deploy
real-patient use
```

## 8. Exact next action

Product Owner reviews the exact tested synthetic usability-v2 candidate and provides keep/change/remove feedback.

No release or expansion action is inferred from the technical PASS.
