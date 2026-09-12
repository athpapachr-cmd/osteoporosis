# SLICE_PLAN_CURRENT.md — CYPRUS / GESY OA JURISDICTION OVERLAY V1

> **STATUS:** RELEASE COMPLETE — PRODUCTION-SMOKE-VERIFIED.
> **Closed:** 2026-09-12 Asia/Nicosia.
> **Bootstrap main:** `2eb9c9c21c17537d8827c5ecf9aedb3a802f4193`.
> **Implementation PR:** `#93`.
> **Exact reviewed runtime head:** `6df3fcb8a2d4eb73306945e5fefb6d4375786f96`.
> **Release commit:** `e52a4851b504476c1e361575d08664c05467ff53`.
> **Production profile:** `CY_GESY` via explicit server-side configuration only.
> **Authenticated production smoke:** run `34703101478` — SUCCESS.
> **Writer:** none — slice closed.
> **Diagnosis vertical:** Knee Osteoarthritis only.
> **Accepted design authority:** PR `#92` + `CYPRUS_GESY_OA_OVERLAY_DESIGN_REVIEW_V1.md`.

## 1. Objective — achieved

The reviewed first real jurisdiction capability is now released without contaminating the international Knee-OA evidence core or expanding the routine referral UI.

Released architecture:

```text
InternationalEvidenceItem
  = global reviewed evidence truth

JurisdictionOverlayV1
  = separate local clinical / admin / reimbursement / lifecycle truth

ResolvedEvidenceView
  = international view unchanged
    + optional jurisdiction context for display
```

## 2. Runtime ownership — released

### Generic runtime owner

`clinic_utilities/physio_referral_product/jurisdiction_overlay.py`

Released responsibilities:

- validate profile and local-position machine data;
- resolve only reviewed active profile configured explicitly for the deployment/account;
- map local positions to existing product items when a reviewed mapping exists;
- return display-only local context;
- keep clinical guidance separate from admin/reimbursement/system-lifecycle rules;
- fail closed for malformed, inactive or unknown profile data.

Forbidden responsibilities remain:

- changing international evidence state;
- source voting;
- selecting treatment;
- changing referral prose;
- inferring jurisdiction from patient location;
- persisting patient data.

### Cyprus profile data

`clinic_utilities/physio_referral_product/jurisdictions/CY_GESY/knee_oa_overlay_v1.yaml`

The released machine profile preserves provenance, local/core relationship, policy class, operational status and display policy from the accepted primary-source audit/difference matrix.

## 3. Activation contract — active in production

Profile activation comes only from explicit deployment/account configuration.

Production key:

`PHYSIO_REFERRAL_JURISDICTION_PROFILE=CY_GESY`

Contract remains:

- unset / empty -> no jurisdiction overlay;
- `CY_GESY` -> reviewed Cyprus/GeSY profile;
- any other value -> fail closed to no overlay.

No workstation/IP/geolocation inference is permitted.

## 4. Released data scope

Mapped existing clinical items may receive local context without creating new routine product items.

Released mapped items include:

- therapeutic_exercise;
- progressive_strengthening;
- education_and_self_management;
- manual_therapy;
- soft_tissue_techniques;
- acupuncture;
- dry_needling;
- walking_aid_assessment_and_training;
- orthosis_or_brace_context;
- weight_management.

Local-only additions/differences such as electrotherapy, RF ablation, podiatry, glucosamine/chondroitin, hyaluronan and PRP remain machine-representable only and do not create new routine controls.

## 5. Evidence payload contract — verified

A mapped item may receive a separate `jurisdiction` field while existing international evidence fields remain authoritative.

Verified invariants:

- `evidence_state` unchanged with overlay on/off;
- source-specific international positions unchanged;
- no local row inserted into international source list;
- admin/reimbursement entries excluded from clinical evidence positions;
- bootstrap exposes only non-patient profile metadata required by UI.

## 6. UX behavior — verified

Routine screen remains unchanged:

- no country selector;
- no badge beside every item;
- local agreement silent;
- routine referral text unchanged.

Evidence detail may show restrained local context, including `Κύπρος · διαφέρει`, only for already-existing relevant items.

Operational GeSY information remains outside routine display.

## 7. Test and release evidence

Exact-head runtime verification on `6df3fcb8a2d4eb73306945e5fefb6d4375786f96` passed the focused jurisdiction gate plus inherited Knee-OA/CU-1/browser/integration coverage.

PR `#93` merged as `e52a4851b504476c1e361575d08664c05467ff53`.

Render deploy `dep-dain5cdg1s2s7380u260` completed LIVE with explicit `CY_GESY` configuration.

Authenticated production smoke run `34703101478` completed SUCCESS and verified:

1. protected page/bootstrap active with `CY_GESY` from explicit account configuration;
2. acupuncture international state remains `guideline_conflict_or_mixed` while local direction remains separately `against`;
3. manual therapy international state remains mixed while local direction is separately `conditional_for`;
4. local-only/admin rows do not become clinical evidence items;
5. selection and referral prose remain unchanged by jurisdiction context;
6. safety behavior remains fail-closed;
7. only synthetic/non-identifiable smoke state was sent and no secret value was printed.

## 8. Scope exclusions retained

- no v5 post-use merge or edits;
- no second diagnosis;
- no Greece/England content;
- no electrotherapy selector;
- no billing/session calculator;
- no provider-unit UI;
- no planned-IT enforcement;
- no automated local recommendation activation;
- no evidence-contract reclassification.

## 9. Closure

All implementation and release exit gates are satisfied.

**Jurisdiction Overlay V1 is closed.**

Any further jurisdiction capability, new local control, second diagnosis, or policy automation requires a fresh bounded slice and fresh Product Owner authority.