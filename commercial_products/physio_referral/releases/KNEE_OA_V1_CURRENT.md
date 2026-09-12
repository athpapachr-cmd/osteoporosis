# Knee OA v1 — current commercial release record

> **STATUS:** COCKPIT PRODUCTION INTEGRATION ACTIVE.
> **Diagnosis:** Knee Osteoarthritis only.
> **Parent closed product candidate:** `b7f2db6fd86f90a32875d9b9cfd356cfa6129c69`.
> **Synthetic substantive candidate:** `9deafa2db43d3498c5becf20f77a804b03849d53`.
> **Synthetic substantive run:** `34677022119` — SUCCESS.
> **Integration branch:** `feat/physio-knee-oa-cockpit-integration-v1-2026-09-12`.

## Product surface retained

- explicit OA diagnosis assertion + laterality;
- live deterministic Greek referral, no routine Generate button;
- direct manual editing with stale-text reconciliation;
- reviewed active-rehab defaults;
- evidence-aware suggestions and mixed-guideline disclosure;
- pain/stiffness/weakness qualifiers with symptom/finding/diagnosis separation;
- advanced exam only on demand;
- `★ Συχνά / Σχετικά τώρα / Όλα` scan-first advanced UI;
- Cyprus/GeSY shown as local context only until item-level audit;
- ephemeral patient draft; no analytics/patient persistence.

## Release acceptance still required on this record

```text
Cockpit protected integration tests
full inherited regression
fresh final release review
PR exact-head gate
squash merge
Render exact-merge deploy verification
bounded synthetic production smoke
canonical closeout
```

A technical PASS does not establish receiving-physiotherapist validation, paid conversion/retention, actual iPhone VoiceOver acceptance or pilot validation.
