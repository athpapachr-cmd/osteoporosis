# CURRENT_OPERATIONAL.md — Knee-OA usability refinement v2 closeout

> **STATUS:** KNEE-OA USABILITY REFINEMENT V2 — IMPLEMENTED / FOCUSED TECHNICAL GATE PASS / PRODUCT-OWNER VISUAL REVIEW NEXT.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Freshly verified `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Parent closed candidate:** `0f38f4d411146667d854c32c9f5f639f344d7c7f`.
> **Implementation branch:** `feat/physio-knee-oa-usability-refine-v2-2026-09-12`.
> **Closed slice:** `CU1-PRODUCT-KNEE-OA-USABILITY-REFINE-V2-20260912`.
> **Tested substantive head:** `a87dccc90dab9f50f70505f2565e3405d48de109`.
> **Successful substantive run:** `34675038243`.
> **Result record:** `clinic_utilities/physio_referral_product/KNEE_OA_USABILITY_REFINE_V2_RESULT.md`.
> **ACTIVE RUNTIME / DESIGN / CANONICAL WRITER:** NONE; usability-refinement writer released.
> **PR / merge / deploy / production smoke authority:** NONE.
> **Real-patient use / production integration:** NOT AUTHORIZED.

## 1. Implemented Product Owner refinements

### Additional suggestions

The primary suggestion remains directly actionable. When additional candidates exist, the routine surface now shows a restrained bordered `Άλλες {n} προτάσεις ›` summary with short current titles. Activating it opens the full existing suggestions sheet. The summary itself does not select treatment.

### Direct referral editing

The desktop live preview now shows `✎ Επεξεργασία` directly. The mobile preview sheet exposes the same action. The overflow menu remains a secondary route.

Manual-text safety remains unchanged:

```text
manual text != structured state
structured change after manual edit
→ clinician text preserved
→ stale reconciliation
→ export blocked
→ explicit resolution required
```

### Favorites / pin-to-top

`Περισσότερα` now supports doctor-facing `☆` / `★` pinning. Pinned items appear in `★ Συχνά` at the top of the advanced surface.

Hard boundaries:

- favorite/pin does not select anything;
- favorites do not change clinical output, evidence, suggestion eligibility or safety;
- unpinning does not clear clinical selection;
- no Hide control exists;
- synthetic favorites are ephemeral only and cleared on reset/pagehide/BFCache;
- no localStorage/sessionStorage/server preference persistence was introduced.

Future persistence, if justified, belongs to clinician/account preference state, not patient/referral state.

## 2. Gate history

Initial run `34674925907` failed because the new direct edit and the legacy overflow edit both matched the inherited generic `[data-edit]` browser locator. The old regression was made explicit about the legacy menu route while the new suite independently tests the direct route. Manual-buffer behavior itself was not weakened.

Successful substantive gate:

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

CI produced desktop/mobile/mixed-evidence screenshots plus new `suggestions-v2` and `favorites-v2` screenshots. They were visually inspected; no obvious clipping or routine-surface crowding was observed. This is not Safari/VoiceOver acceptance.

## 3. Product boundaries unchanged

The general utility rule still applies:

```text
clinically meaningful
!= workflow-useful
!= receiver-useful
!= worth adding
```

No clinical/evidence semantics, jurisdiction guidance, diagnosis scope, patient persistence or production runtime were expanded in this slice.

## 4. Still unproven

```text
Product Owner visual/use acceptance of this exact candidate
actual iPhone Safari / VoiceOver
complete measured accessibility/contrast acceptance
receiving-physiotherapist real-user validation
whether favorites deserve future cross-device persistence
production integration / real-patient use
commercial willingness-to-pay / retention
```

## 5. Exact next action

**Product Owner visually/use-tests the exact tested synthetic usability-v2 artifact and records concrete keep/change/remove feedback.**

No PR, merge, deploy, second diagnosis, permanent Hide or preference-persistence work follows automatically.
