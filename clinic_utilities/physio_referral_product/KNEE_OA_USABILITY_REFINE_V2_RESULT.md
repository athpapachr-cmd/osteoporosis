# Knee OA usability refinement v2 — result

> **Status:** IMPLEMENTED / FOCUSED TECHNICAL GATE PASS / PRODUCT OWNER VISUAL REVIEW NEXT  
> **Branch:** `feat/physio-knee-oa-usability-refine-v2-2026-09-12`  
> **Parent closeout:** `0f38f4d411146667d854c32c9f5f639f344d7c7f`  
> **Tested substantive head:** `a87dccc90dab9f50f70505f2565e3405d48de109`  
> **Successful workflow run:** `34675038243`  
> **Synthetic only:** yes  
> **Release authority:** none

## Implemented Product Owner refinements

### 1. Additional suggestions are now discoverable

The highest-priority suggestion remains a compact actionable line. Additional eligible suggestions now appear in a restrained bordered summary panel:

```text
Άλλες {n} προτάσεις ›
<title 1> · <title 2> · ...
```

The count is the number **in addition to** the primary visible suggestion. Activating the panel opens the existing suggestions sheet. The summary itself does not select, dismiss or mutate a clinical item.

### 2. Referral text editing is directly visible

Desktop live preview now exposes `✎ Επεξεργασία` in the preview heading. The mobile preview sheet exposes the same direct route.

The existing manual-text safety model remains authoritative:

```text
manual text != structured state
structured change after manual edit
→ manual text preserved
→ stale reconciliation state
→ export blocked
→ explicit clinician reconciliation required
```

The legacy overflow edit route remains functional as a secondary route.

### 3. Favorites / pin-to-top added without Hide

Advanced selectable items can be pinned with `☆` / `★`. Pinned items appear in a compact `★ Συχνά` section at the top of `Περισσότερα`.

Hard boundaries:

- favorite != clinical selection;
- pin/unpin does not change referral output, evidence, suggestions or safety;
- selecting from the favorite representation writes the same underlying structured state;
- unpinning preserves any clinical selection;
- no Hide control exists;
- favorites are ephemeral in this synthetic prototype and are cleared on reset/pagehide/BFCache;
- no localStorage/sessionStorage/server persistence was added.

Future persistence, if justified, belongs to clinician-account preferences, not patient/referral state.

## Gate history

Initial usability-v2 run:

```text
run     34674925907
head    b9889f85a1f4ad6f4da6050b6ebf34eefeb2bcb3
result  FAILURE
```

The failure was an inherited browser-test selector ambiguity: the new direct edit and the legacy menu edit both matched a generic `[data-edit]` locator. The inherited test was made explicit about the legacy menu route rather than removing either edit capability or weakening manual-buffer assertions.

Successful substantive run:

```text
run                                       34675038243
head                                      a87dccc90dab9f50f70505f2565e3405d48de109
result                                    SUCCESS
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

The usability-v2 Chromium suite proves:

- five eligible suggestions render as one primary plus `Άλλες 4 προτάσεις` with current titles;
- one eligible suggestion renders no secondary panel;
- the secondary panel opens the complete suggestion sheet without auto-selection;
- direct desktop/mobile edit preserves manual text through a later structured change and keeps export blocked pending reconciliation;
- pinning an advanced item does not select it;
- selection via `★ Συχνά` uses the same underlying structured state;
- unpinning preserves clinical selection;
- reset and simulated BFCache clear favorites;
- localStorage/sessionStorage stay empty;
- no Hide action exists.

## Visual inspection

CI screenshots were inspected for:

- routine desktop surface with visible direct edit;
- mobile reflow;
- additional-suggestion summary and full suggestion sheet;
- `★ Συχνά` pin-to-top section inside `Περισσότερα`;
- inherited mixed-evidence disclosure.

No obvious clipping or routine-surface crowding was observed in these screenshots. This does not substitute for actual iPhone Safari/VoiceOver acceptance.

## Still not proven

```text
Product Owner visual/use acceptance of this exact refinement
actual iPhone Safari / VoiceOver
complete measured accessibility/contrast acceptance
receiving-physiotherapist real-user validation
favorite persistence value in a real account model
production integration / real-patient use
```

## Next action

Product Owner visually/use-tests the exact tested synthetic artifact and decides keep/change/remove. No PR, merge, deploy, second diagnosis or preference-persistence work follows automatically.
