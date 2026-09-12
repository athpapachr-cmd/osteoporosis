# SLICE_PLAN_CURRENT.md — Knee-OA `Περισσότερα` visual + IA redesign v3

> **STATUS:** IMPLEMENTED / FULL INHERITED + V3 TECHNICAL GATE PASS / PRODUCT-OWNER VISUAL REVIEW NEXT.
> **Slice:** `CU1-PRODUCT-KNEE-OA-MORE-REDESIGN-V3-20260912`.
> **Branch:** `feat/physio-knee-oa-more-redesign-v3-2026-09-12`.
> **Parent closed head:** `2c19eb7283c96af7868690a892d7e61155674d6b`.
> **Tested substantive head:** `9deafa2db43d3498c5becf20f77a804b03849d53`.
> **Successful substantive run:** `34677022119`.
> **Fresh main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** NONE; v3 implementation writer released.
> **Release / real clinical use:** NOT AUTHORIZED.
> **Durable product context:** `clinic_utilities/physio_referral_product/PHYSIO_REFERRAL_PRODUCT_CONTEXT_CURRENT.md`.

## 1. Problem addressed

The v2 `Περισσότερα` preserved too many simultaneously visible controls. Favorites improved frequent access but did not solve the underlying scanning cost because categories/items still competed with similar visual weight.

The accepted redesign preserves capability while reducing routine visual decisions.

## 2. Product/design principle

```text
CAPABILITY DEPTH UNDERNEATH
+
CALM SURFACE ABOVE
```

The user should not visually parse the full advanced clinical vocabulary to find one item.

This slice explicitly implements the Product Owner's request for a **visual redesign**, not merely a hierarchy rename.

## 3. Implemented routine surface

### `★ Συχνά`

- personal shortcuts only;
- bounded to six in the synthetic prototype;
- pinning never changes clinical selection;
- normal mode does not display a star on every option;
- `Προσαρμογή Συχνών` deliberately reveals pin controls;
- no Hide;
- no persistence.

### `Σχετικά τώρα`

- at most three deterministic shortcuts;
- derived only from already-declared structured state;
- each maps to an existing field/control;
- appearing/opening never auto-selects or infers a finding/treatment/safety state;
- absent relevant context means the section is absent.

### `Όλα`

The complete advanced vocabulary remains reachable through six scan-friendly category rows:

```text
Εξέταση
Λειτουργία & στόχοι
Αποκατάσταση
Συμπληρωματικά
Περιορισμοί & σημείωση
Κλινικός έλεγχος
```

Each row provides:

- category name;
- restrained supporting icon;
- selected count when >0;
- short selected-item summary when useful;
- chevron/navigation affordance.

Activating a category opens its full controls in the existing sheet/modal host.

## 4. Visual grammar implemented

- stronger section typography and whitespace;
- large calm rows instead of dense chip walls;
- one restrained icon grammar;
- muted secondary summaries;
- quiet textual active counts rather than badge clutter;
- no card-inside-card proliferation;
- evidence colours remain evidence semantics, not decoration;
- icon is never the only label;
- desktop/mobile share the same conceptual hierarchy.

Collapsed rows still answer “what have I selected?” without becoming a second source of truth.

## 5. Capability/safety preservation

No supported advanced clinical option was deleted merely to simplify the UI. Controls were relocated into category sheets while the underlying state and clinical IDs remained authoritative.

Safety/readiness remain independent of browsing convenience. A safety block cannot depend on whether `Περισσότερα` or a category sheet is open.

Core invariants remain:

```text
selection != suggestion != evidence != safety
favorite != selection
contextual shortcut != inference
manual text != structured state
symptom != finding != diagnosis
```

## 6. Implementation defects caught and corrected

### Compound-button rendering defect

The first v3 browser gate exposed that compound v3 buttons were created through a helper that accepted plain text only, causing a blank FFD button. The v3 compound-button primitive was corrected globally rather than patching one control.

### Relevant-Now test race

A later test captured the referral before the previous weakness projection completed, comparing a placeholder against the actual referral. The test was corrected to wait for the true projected text before proving that `Σχετικά τώρα` does not mutate state/text merely by appearing or opening.

No clinical semantics were weakened to make tests pass.

## 7. Technical acceptance

Successful substantive gate:

```text
workflow                                  Physio Knee OA prototype gate
run                                       34677022119
head                                      9deafa2db43d3498c5becf20f77a804b03849d53
scope + syntax                            PASS
real CU-1 / HTTP                          15 / 15 PASS
frozen Step-3 exact-output fixtures       15 PASS
post-review clinical/output               11 / 11 PASS
inherited Chromium                        12 / 12 PASS
post-review qualifier Chromium             9 / 9 PASS
usability-v2 Chromium                      5 / 5 PASS
More-v3 Chromium                           5 / 5 PASS
Greek source-summary coverage             54 positions
packaged dependency closure               PASS
```

Artifact:

```text
id      10292822525
name    knee-oa-prototype-9deafa2db43d3498c5becf20f77a804b03849d53
digest  sha256:1cc6156d5de37e2ef1fcc15de68e6df7811535e10dfec86c20ed1723ba81fb0e
```

Visual evidence inspected:

- `more-v3-overview.png`
- `more-v3-favorites.png`
- `more-v3-mobile.png`
- inherited desktop/mobile/evidence/suggestion screenshots.

No obvious horizontal clipping or reversion to the previous control wall was observed. This does not prove actual iPhone Safari/VoiceOver or measured accessibility acceptance.

## 8. Explicit exclusions remain

```text
search
permanent Hide
recent-items system
multiple preference tabs
favorite/account persistence
AI-generated relevance
clinical/evidence semantic edits
new structured functional-baseline field
routine FFD measurement
second diagnosis
unaudited jurisdiction guidance
PR / merge / deploy
real-patient use
```

## 9. Exact next action

**Product Owner visually/use-tests the exact tested More-v3 synthetic artifact and gives concrete keep/change/remove feedback.**

After Product Owner acceptance, deliberately decide whether Knee-OA is ready to freeze for receiving-physiotherapist/real-user validation or needs one bounded correction. Do not automatically expand to another diagnosis or production release.
