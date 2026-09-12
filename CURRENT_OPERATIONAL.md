# CURRENT_OPERATIONAL.md — Cyprus / GeSY OA overlay design complete / review HOLD

> **STATUS:** CYPRUS / GESY OA JURISDICTION OVERLAY V1 — PRIMARY-SOURCE AUDIT + DESIGN COMPLETE / PRODUCT OWNER REVIEW HOLD / NO RUNTIME IMPLEMENTATION.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Bootstrap main:** `a2fa27c7ff26d1dd22cd6f726656ca0532daab75`.
> **Design branch:** `design/physio-cy-gesy-oa-overlay-v1-2026-09-12`.
> **Current production runtime SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Authenticated live smoke:** run `34681808255`, attempt `3` — SUCCESS.
> **ACTIVE DESIGN / RUNTIME WRITER:** NONE.
> **RUNTIME / UI IMPLEMENTATION AUTHORITY:** NONE.
> **V5 tested candidate:** remains separately in Product Owner review HOLD; no merge/deploy authority is implied here.
> **Real-patient data:** NOT USED.

## 1. Completed design deliverables

The bounded Cyprus/GeSY audit/design slice produced:

1. `commercial_products/physio_referral/jurisdictions/CY_GESY/CYPRUS_GESY_OA_SOURCE_AUDIT_V1.md`
2. `commercial_products/physio_referral/jurisdictions/CY_GESY/CYPRUS_GESY_OA_DIFFERENCE_MATRIX_V1.md`
3. `commercial_products/physio_referral/jurisdictions/JURISDICTION_OVERLAY_SCHEMA_V1.yaml`
4. `commercial_products/physio_referral/jurisdictions/CY_GESY/CYPRUS_GESY_OA_OVERLAY_UX_DESIGN_V1.md`
5. `commercial_products/physio_referral/jurisdictions/CY_GESY/CYPRUS_GESY_OA_OVERLAY_DESIGN_REVIEW_V1.md`

Design review disposition:

```text
PRIMARY-SOURCE AUDIT          PASS
DIFFERENCE MATRIX             PASS
MACHINE SCHEMA                PASS AS PROPOSED DESIGN
UX / ROUTINE-EXCLUSION POLICY PASS
RUNTIME CHANGE                NONE
PRODUCT OWNER REVIEW          NEXT
```

## 2. Evidence conclusions

Verified Cyprus/HIO clinical differences/additions exist, including electrotherapy, radiofrequency nerve ablation, podiatry, glucosamine/chondroitin, hyaluronan, PRP-related positions and imaging implementation detail.

The current routine Knee-OA referral core remains compatible with Cyprus on the product's routine defaults:

```text
therapeutic exercise
progressive strengthening
education / self-management
individualized active rehabilitation
```

No current international evidence state requires mutation.

International conflict remains international conflict. A definite Cyprus position on acupuncture/manual therapy is local context only and cannot collapse the global `guideline_conflict_or_mixed` state.

## 3. Operational-policy separation

GeSY physiotherapy access/referral/session/documentation/provider-unit rules were audited separately as administrative/reimbursement policy.

They do not strengthen, weaken or replace clinical evidence.

The HIO May-2026 announcement supports active guideline publication/implementation but describes OA-guideline integration into the GeSY information system as a future action. The currently linked OA PDF still carries draft/December-2025 metadata. Both facts remain explicit.

## 4. Bounded recommendation

`IMPLEMENT JURISDICTION OVERLAY V1`

Meaning only:

- accept a separate machine/provenance layer after Product Owner review;
- keep the international evidence core unchanged;
- keep the current routine Knee-OA main UI unchanged initially;
- surface local differences progressively in evidence detail when relevant;
- keep GeSY administrative/reimbursement rules in a separate operational class;
- use explicit clinician/account jurisdiction configuration, not patient-location inference;
- do not implement Greece/England until real market/workflow need exists.

This recommendation is **not runtime implementation authority**.

## 5. Exact next lifecycle boundary

Product Owner reviews the audit, difference matrix, proposed schema, UX policy and bounded recommendation.

Only if accepted may a later fresh slice define the minimum runtime implementation contract. No UI/runtime/evidence mutation, PR/merge/deploy, second diagnosis or patient persistence is authorized by this design closeout.