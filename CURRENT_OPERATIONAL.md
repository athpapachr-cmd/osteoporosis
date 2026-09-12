# CURRENT_OPERATIONAL.md — Cyprus / GeSY OA jurisdiction overlay v1 runtime implementation

> **STATUS:** CYPRUS / GESY OA JURISDICTION OVERLAY V1 — PRODUCT OWNER ACCEPTED DESIGN / RUNTIME IMPLEMENTATION ACTIVE.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Bootstrap main:** `2eb9c9c21c17537d8827c5ecf9aedb3a802f4193`.
> **Accepted design PR:** `#92` / squash merge `2eb9c9c21c17537d8827c5ecf9aedb3a802f4193`.
> **Active runtime branch:** `feat/physio-cy-gesy-overlay-v1-runtime-2026-09-12`.
> **ACTIVE DESIGN / RUNTIME WRITER:** `feat/physio-cy-gesy-overlay-v1-runtime-2026-09-12`.
> **Mutation scope:** Knee-OA jurisdiction-overlay runtime/data/tests + exact supporting canonicals only.
> **Current production runtime SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Authenticated v4 live smoke:** run `34681808255`, attempt `3` — SUCCESS.
> **V5 tested candidate:** remains separately in Product Owner review HOLD and is not part of this slice.
> **Real-patient data:** NOT USED.

## 1. Product Owner implementation authority

On 2026-09-12 the Product Owner explicitly accepted the reviewed design with:

`IMPLEMENT JURISDICTION OVERLAY V1`

That authorizes this new bounded runtime slice only. It does not authorize unrelated v5 merge/deploy, second diagnosis, patient persistence, or expansion into Greece/England.

## 2. Runtime target

Implement the smallest useful reviewed architecture:

```text
international evidence core
+
JurisdictionOverlayV1 resolver
+
reviewed CY_GESY Knee-OA local-position data
+
progressive evidence-detail presentation only where relevant
```

Hard invariants:

- international evidence state is never mutated by local overlay;
- local position never auto-selects/deselects an intervention;
- local position never rewrites referral prose;
- local clinical guidance remains separate from GeSY admin/reimbursement/system-lifecycle rules;
- planned GeSY IT integration remains planned, never active by inference;
- no source voting;
- no patient-location inference;
- no patient persistence;
- no autonomous literature-to-live update;
- no second diagnosis.

## 3. Exact implementation scope

1. Add validated generic jurisdiction-overlay runtime loader/resolver.
2. Add reviewed machine-readable `CY_GESY` Knee-OA profile derived only from the accepted audit/matrix.
3. Activate the profile only from explicit deployment/account configuration; no patient-location inference.
4. Attach local-position views to existing evidence-detail payloads without changing international `evidence_state`.
5. Keep local agreement silent on the routine surface.
6. Permit a restrained local-difference cue/detail only for an existing item already being inspected/relevant.
7. Keep administrative/reimbursement rules machine-representable but out of clinical evidence resolution and routine UI.
8. Add exact contract, resolver, integration and browser regressions.
9. Preserve current routine referral prose and no-persistence behavior.

## 4. Explicit exclusions

No new routine controls for electrotherapy, RF ablation, podiatry, glucosamine/chondroitin, hyaluronan, PRP, injections or imaging.

No country selector, badge wall, GeSY reimbursement panel, provider-unit mechanics, or automatic local-rule enforcement.

No mutation of `knee_oa_evidence_contract_v1.yaml` merely because Cyprus differs.

## 5. Release gates

Implementation may proceed to PR only after:

- machine/profile validation PASS;
- proof overlay cannot mutate core evidence state;
- proof local clinical/admin classes remain separate;
- proof inactive/unknown profile fails closed;
- proof routine main UI/referral prose unchanged;
- focused evidence-detail browser acceptance PASS;
- inherited Knee-OA/CU-1 regressions PASS;
- branch remains current with `main` and bounded in scope.

Merge/deploy requires exact-head PR evidence. Post-deploy authenticated production smoke is required before calling overlay production-smoke-verified.
