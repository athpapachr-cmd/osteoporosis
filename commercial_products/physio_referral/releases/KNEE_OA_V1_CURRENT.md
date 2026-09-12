# Knee OA v1 — current commercial release record

> **STATUS:** RELEASE CANDIDATE / EXACT-HEAD TESTED / FINAL RELEASE REVIEW PASS / PR NEXT.
> **Diagnosis:** Knee Osteoarthritis only.
> **Parent closed product candidate:** `b7f2db6fd86f90a32875d9b9cfd356cfa6129c69`.
> **Synthetic substantive candidate:** `9deafa2db43d3498c5becf20f77a804b03849d53`.
> **Cockpit release-candidate head:** `57d879b072fcaf2bef350d5dbb0547ce465764e1`.
> **Integration branch:** `feat/physio-knee-oa-cockpit-integration-v1-2026-09-12`.
> **Release review:** `KNEE_OA_V1_RELEASE_REVIEW.md` — PASS / no open blocker.

## Product surface retained

- explicit OA diagnosis assertion + laterality;
- live deterministic Greek referral, no routine Generate button;
- direct manual editing with stale-text reconciliation;
- reviewed active-rehab defaults;
- evidence-aware suggestions and mixed-guideline disclosure;
- pain/stiffness/weakness qualifiers with symptom/finding/diagnosis separation;
- advanced examination only on demand;
- `★ Συχνά / Σχετικά τώρα / Όλα` scan-first advanced UI;
- Cyprus/GeSY shown as local context only until item-level audit;
- ephemeral patient draft; no analytics/patient persistence.

## Cockpit production integration

The product is integrated into the existing protected Clinical Excellence route:

`/clinical/clinic-utilities/physio-referral`

Authentication and real CU-1 validation/safety remain server-owned. Shared product projection is owned by `clinic_utilities/physio_referral_product/knee_oa_projection.py`; the local prototype is only a loopback test transport.

Production network requests are explicitly non-synthetic. The protected adapter rejects synthetic-marked Cockpit requests before adapting internally to the frozen shared projection compatibility envelope.

## Exact release-candidate gates

At `57d879b072fcaf2bef350d5dbb0547ce465764e1`:

```text
Cockpit integration gate   34679427725   SUCCESS
Inherited product gate     34679427741   SUCCESS
CU-1 focused gate          34679427822   SUCCESS
```

## Release acceptance remaining

```text
bounded PR to current main
PR-head workflows clean
fresh behind=0 / scope check
squash merge
Render exact-merge deploy verification
bounded synthetic/non-identifiable production smoke
final canonical closeout
```

A release PASS does not establish receiving-physiotherapist validation, paid conversion/retention, actual iPhone VoiceOver acceptance or clinical pilot validation.
