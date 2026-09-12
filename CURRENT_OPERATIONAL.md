# CURRENT_OPERATIONAL.md — Cyprus / GeSY OA jurisdiction-overlay audit active

> **STATUS:** DESIGN / EVIDENCE AUDIT ACTIVE — NO UI/RUNTIME MUTATION AUTHORIZED.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Bootstrap main:** `a2fa27c7ff26d1dd22cd6f726656ca0532daab75`.
> **Current production runtime SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Authenticated live smoke:** run `34681808255`, attempt `3` — SUCCESS.
> **ACTIVE DESIGN WRITER:** `design/physio-cy-gesy-oa-overlay-v1-2026-09-12`.
> **Mutation scope:** documentation + machine-design artifacts for Cyprus/GeSY OA source audit, difference matrix, jurisdiction schema and UX policy only.
> **RUNTIME / UI WRITER:** NONE.
> **V5 tested candidate:** remains separately in Product Owner review HOLD; no merge/deploy authority is implied here.
> **Real-patient data:** NOT USED.

## 1. Active bounded task

`CYPRUS / GESY OA JURISDICTION OVERLAY V1`

Purpose:

```text
international evidence core
+
optional jurisdiction / local-system overlay
```

The first target market is Cyprus / GeSY, but the product core must remain internationally reusable.

## 2. Hard boundaries

```text
international evidence position != local policy
Cyprus clinical adaptation != GeSY administrative/reimbursement rule
resource/cost policy != stronger clinical evidence
published guideline != proven active IT enforcement
planned GeSY integration != active system rule
local recommendation != permission to overwrite international state
source count != evidence resolution
Product Owner request != evidence != implementation authority
```

No patient persistence, second diagnosis, autonomous literature update, new clinical default, UI change or runtime activation is authorized in this slice.

## 3. Primary-source audit state

Official HIO/GeSY sources identified include:

- Cyprus adaptation of NICE NG226 for OA;
- HIO adaptation-summary document describing characteristic changes;
- HIO 25-May-2026 implementation announcement;
- current GeSY adult-guideline and OA pages;
- GeSY allied-health access and physiotherapy service/reimbursement rules.

Important source-status nuance already established:

- HIO's May-2026 announcement states that the Cyprus adaptation process is complete and informs providers of guideline implementation;
- the same announcement says HIO **will** integrate the guideline into the GeSY information system, so IT integration is planned rather than proven active;
- the currently linked public OA PDF still carries `ΠΡΟΣΧΕΔΙΟ / Δεκέμβριος 2025` metadata, creating a publication-version inconsistency that must remain visible rather than silently normalized away.

## 4. Exact deliverables before any implementation

1. `CYPRUS_GESY_OA_SOURCE_AUDIT_V1`
2. `CYPRUS_GESY_OA_DIFFERENCE_MATRIX_V1`
3. proposed jurisdiction-overlay machine schema
4. proposed UX behavior for agreement / difference / administrative-resource rule / unknown local status
5. explicit routine-surface exclusions
6. bounded recommendation: `NO CHANGE`, `LOCAL INFO ONLY`, or `IMPLEMENT JURISDICTION OVERLAY V1`

## 5. Exact next action

Complete the recommendation-by-recommendation primary-source audit, compare verified local positions only against the existing reviewed Knee-OA international evidence contract, then freeze the design artifacts for Product Owner review.

Do not implement UI/runtime changes in this slice.