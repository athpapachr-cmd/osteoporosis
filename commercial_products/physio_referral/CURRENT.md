# CURRENT.md — Physio Referral commercial product track

> **STATUS:** KNEE-OA V1 V4 RELEASED / DEPLOYED / AUTHENTICATED LIVE PRODUCT SMOKE PASS; V5 TESTED CANDIDATE IN PRODUCT OWNER REVIEW HOLD.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Diagnosis vertical:** Knee Osteoarthritis only.
> **Current production runtime SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Render deploy:** `dep-daiivp0jo6nc73bl6pug` — LIVE.
> **Authenticated live smoke:** `34681808255`, attempt `3` — SUCCESS.
> **V5 tested candidate:** `fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12` @ `a47357c602120d3678e8f2f23b99775e616c79e1`.
> **V5 gate:** `34693751545` — SUCCESS.
> **Root operational authority:** `CURRENT_OPERATIONAL.md`.

## Product state

The released Knee-OA product remains deployed in the authenticated Clinical Excellence physiotherapy utility. The routine experience remains deterministic, evidence-aware and intentionally sparse: few meaningful clinician decisions on the main surface, progressive disclosure underneath, direct editing with fail-closed reconciliation, and no routine LLM-generated referral prose.

Commercial/product authority lives under `commercial_products/physio_referral/`; technical contracts/runtime/tests remain under `clinic_utilities/physio_referral_product/`.

## Production lifecycle

Supported current claims:

```text
merged production v4                       yes
deployed production v4                     yes
public-asset live smoke                    pass
unauthenticated auth boundary              pass
authenticated protected product smoke      pass
real clinical pilot                        no
receiver field validation                  no
commercial/paid validation                 no
```

Authenticated run `34681808255`, attempt `3`, used the protected GitHub Actions secret `CLINICAL_DATA_KEY`, never printed the credential and sent only synthetic/non-identifiable Knee-OA state. It proved protected page/bootstrap access, deterministic projection and safety fail-closed behavior against live production.

## V5 review candidate

A bounded post-use v5 refinement has already been implemented and technically validated on a separate branch. It is **not production yet**.

The candidate changes only workflow/presentation:

- first tap on inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without a popup;
- second tap exposes optional detail;
- Function keeps its first-tap chooser;
- weakness detail is reduced to clear explicit examination concepts;
- bare `Περιαρθρικά` is removed from routine/advanced UI while historical compatibility remains underneath;
- pain-location duplication is reconciled;
- rich clinical handoff and physiotherapy plan use separate paragraphs;
- `Επιπλέον στόχος:` becomes connected human prose.

Exact tested candidate head `a47357c602120d3678e8f2f23b99775e616c79e1` passed full gate `34693751545`.

No PR/merge/deploy authority exists until Product Owner review accepts that candidate.

## Review history

The four specialist review axes remain:

1. Clinical / Evidence
2. Physiotherapy / receiving-professional utility
3. UX / Product
4. Commercial / Product-Market

The archived fifth upload remains a supplementary combined/multi-axis review, not an additional specialist vote.

## Product truth still unproven

- no receiving-physiotherapist field validation yet;
- no actual iPhone Safari/VoiceOver acceptance yet;
- no paid conversion/retention evidence yet;
- no real clinical pilot validation yet;
- Cyprus/GeSY recommendation-by-recommendation overlay is not activated;
- no second diagnosis has been selected.

## Next product boundaries

1. Product Owner disposition of the exact tested v5 candidate before any release.
2. Primary-source Cyprus/GeSY OA audit and jurisdiction-overlay design may now proceed independently because the authenticated production lifecycle gate is closed.
3. Do not expand diagnosis count merely to make the product look larger.

Permanent rule:

```text
CLINICALLY INTERESTING != WORKFLOW-USEFUL != RECEIVER-USEFUL != WORTH ADDING
PRODUCT OWNER REQUEST != EVIDENCE != IMPLEMENTATION AUTHORITY
```