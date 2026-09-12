# CURRENT.md — Physio Referral commercial product track

> **STATUS:** KNEE-OA V1 RELEASED / CY_GESY JURISDICTION OVERLAY RELEASED / V5 TESTED CANDIDATE IN PRODUCT OWNER REVIEW HOLD.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Diagnosis vertical:** Knee Osteoarthritis only.
> **Canonical main:** `273e22a0ea2bc9b2e5a996fac22a0a88eb54d30e`.
> **Released jurisdiction runtime:** `e52a4851b504476c1e361575d08664c05467ff53`.
> **Final canonical Render deploy:** `dep-dain8hqjnfac73ed65a0` — LIVE.
> **Production jurisdiction profile:** `CY_GESY` via explicit server-side configuration.
> **Final authenticated canonical live smoke:** `34703453615` — SUCCESS.
> **V5 tested candidate:** `fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12` @ `a47357c602120d3678e8f2f23b99775e616c79e1`.
> **V5 gate:** `34693751545` — SUCCESS.
> **Root operational authority:** `CURRENT_OPERATIONAL.md`.

## Product state

The Knee-OA product is released inside the authenticated Clinical Excellence physiotherapy utility.

Current production architecture is:

```text
international evidence core
+
reviewed CY_GESY jurisdiction overlay
+
deterministic Knee-OA referral projection
+
progressive evidence disclosure
```

The routine experience remains intentionally sparse: few meaningful clinician decisions on the main surface, progressive disclosure underneath, direct editing with fail-closed reconciliation, and no routine LLM-generated referral prose.

Commercial/product authority lives under `commercial_products/physio_referral/`; technical contracts/runtime/tests remain under `clinic_utilities/physio_referral_product/`.

## Production lifecycle

Supported current claims:

```text
Knee-OA production release                    yes
CY_GESY jurisdiction overlay                  yes
explicit production jurisdiction config       yes
authenticated protected product smoke         pass
final canonical post-closeout smoke            pass
real clinical pilot                            no
paid/commercial validation                     no
second diagnosis                               no
```

Final authenticated run `34703453615` used the protected GitHub Actions `CLINICAL_DATA_KEY`, never printed the credential and sent only a generated UUID plus non-identifiable smoke state. It verified the final canonical production state after the docs-only closeout.

The local Cyprus/GeSY layer remains display-only context beside the international evidence core. It does not silently rewrite international evidence state, clinician selection, referral prose or safety semantics.

## Product Owner decision — receiver/physiotherapist feedback

On 2026-09-12 the Product Owner explicitly decided that formal physiotherapist/receiver evaluation is **not required as a blocking gate** for current product progress.

The Product Owner may request informal feedback from clinician colleagues later, at a time of their choosing.

Therefore:

```text
physiotherapist / receiver feedback
= potentially useful external feedback
!= prerequisite for V5 release
!= prerequisite for continued Knee-OA refinement
!= automatic prerequisite for a bounded clinical/commercial pilot
```

This does not mean receiver utility is assumed or proven. It means it is not a mandatory immediate gate. Any later colleague feedback should be treated as evidence to consider, not as implementation authority by itself.

## V5 review candidate

A bounded post-use v5 refinement has already been implemented and technically validated on a separate branch. It is **not production yet**.

The candidate changes workflow/presentation only:

- first tap on inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without a popup;
- second tap exposes optional detail;
- `Λειτουργικότητα` keeps its first-tap chooser because a generic unqualified Function state is not useful;
- weakness detail is reduced to clear examination concepts;
- bare `Περιαρθρικά` is removed from routine/advanced UI while historical compatibility remains underneath;
- pain-location duplication is reconciled;
- clinical handoff and physiotherapy plan use separate paragraphs;
- `Επιπλέον στόχος:` becomes connected human prose rather than a mechanical trailing label.

Exact tested candidate head `a47357c602120d3678e8f2f23b99775e616c79e1` passed full gate `34693751545`.

No merge/deploy has occurred for V5 yet. Product Owner disposition remains the next bounded product decision.

## Validation truth still unproven

The following remain unproven and must not be described as completed:

- real clinical pilot value;
- actual paid conversion / willingness-to-pay / retention;
- broader external colleague feedback;
- actual iPhone Safari / VoiceOver acceptance unless separately performed and recorded;
- usefulness beyond Knee Osteoarthritis;
- market need for Greece or England profiles.

Receiver/physiotherapist feedback belongs in this list as **optional later external evidence**, not as a required blocker.

## Next product boundaries

1. Product Owner disposition of the exact tested V5 candidate.
2. If accepted, release V5 through the normal merge → Render deploy → authenticated production smoke lifecycle.
3. Perform Product Owner real-device/use acceptance after V5 release.
4. Select any later pilot/commercial-validation slice explicitly; do not imply one automatically.
5. Colleague/receiver feedback may be collected later when convenient, but it does not block steps 1–4.
6. Do not expand diagnosis count merely to make the product look larger. A second diagnosis requires a fresh bounded decision and authority.

Permanent rule:

```text
CLINICALLY INTERESTING != WORKFLOW-USEFUL != RECEIVER-USEFUL != WORTH ADDING
PRODUCT OWNER REQUEST != EVIDENCE != IMPLEMENTATION AUTHORITY
EXTERNAL FEEDBACK != AUTOMATIC IMPLEMENTATION AUTHORITY
```
