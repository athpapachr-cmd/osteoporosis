# CURRENT_OPERATIONAL.md — Cyprus / GeSY OA jurisdiction overlay v1

> **STATUS:** RELEASE COMPLETE — PRODUCTION-SMOKE-VERIFIED.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Accepted design PR:** `#92` / squash merge `2eb9c9c21c17537d8827c5ecf9aedb3a802f4193`.
> **Runtime implementation PR:** `#93` / merge commit `e52a4851b504476c1e361575d08664c05467ff53`.
> **Exact reviewed runtime head:** `6df3fcb8a2d4eb73306945e5fefb6d4375786f96`.
> **Current production runtime SHA:** `e52a4851b504476c1e361575d08664c05467ff53`.
> **Production jurisdiction profile:** `CY_GESY`, selected only by explicit server-side configuration `PHYSIO_REFERRAL_JURISDICTION_PROFILE=CY_GESY`.
> **Render deploy:** `dep-dain5cdg1s2s7380u260` — LIVE.
> **Authenticated jurisdiction-overlay live smoke:** run `34703101478` — SUCCESS.
> **Writer:** none — implementation slice closed.
> **V5 tested candidate:** remains separately in Product Owner review HOLD and is not part of this slice.
> **Real-patient data:** NOT USED.

## 1. Product Owner authority and closure

On 2026-09-12 the Product Owner explicitly accepted the reviewed design with:

`IMPLEMENT JURISDICTION OVERLAY V1`

The bounded implementation was completed, reviewed, merged, explicitly configured for Cyprus/GeSY, deployed and authenticated-smoke-verified on the same release commit. The slice is therefore closed.

This closure does not authorize unrelated v5 merge/deploy, second diagnosis, patient persistence, Greece/England expansion, or autonomous local-guideline activation.

## 2. Released architecture

```text
international evidence core
+
JurisdictionOverlayV1 resolver
+
reviewed CY_GESY Knee-OA local-position data
+
progressive evidence-detail presentation only where relevant
```

Hard invariants preserved in implementation and live smoke:

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

## 3. Released implementation scope

1. Validated generic jurisdiction-overlay runtime loader/resolver.
2. Reviewed machine-readable `CY_GESY` Knee-OA profile derived only from the accepted audit/matrix.
3. Explicit deployment/account activation only; no patient-location inference.
4. Local-position views attached beside existing evidence-detail payloads without changing international `evidence_state`.
5. Local agreement silent on the routine surface.
6. Restrained local-difference cue/detail only for an existing item already being inspected/relevant.
7. Administrative/reimbursement rules machine-representable but excluded from clinical evidence resolution and routine UI.
8. Exact contract, resolver, integration and browser regressions.
9. Existing routine referral prose and no-persistence behavior preserved.

## 4. Explicit exclusions retained

No new routine controls for electrotherapy, RF ablation, podiatry, glucosamine/chondroitin, hyaluronan, PRP, injections or imaging.

No country selector, badge wall, GeSY reimbursement panel, provider-unit mechanics, or automatic local-rule enforcement.

No mutation of `knee_oa_evidence_contract_v1.yaml` merely because Cyprus differs.

## 5. Exact-head verification

Runtime exact-head physio-owned gates on `6df3fcb8a2d4eb73306945e5fefb6d4375786f96` completed successfully, including:

- jurisdiction overlay v1 gate;
- Knee-OA Cockpit integration gate;
- clinical-sheet v4 gate;
- prototype gate;
- evidence-design gate;
- CU-1 focused tests.

Clinical Learning red checks were adjacent-owner/scope-guard failures only; their substantive owner tests passed before those guards failed.

## 6. Production verification

Production service `osteoporosis` was configured with:

`PHYSIO_REFERRAL_JURISDICTION_PROFILE=CY_GESY`

Render deploy `dep-dain5cdg1s2s7380u260` completed LIVE on release commit `e52a4851b504476c1e361575d08664c05467ff53`.

Authenticated production smoke run `34703101478` completed SUCCESS and verified with protected `CLINICAL_DATA_KEY` plus synthetic/non-identifiable Knee-OA state only:

- protected page and jurisdiction JS served successfully;
- bootstrap exposed `CY_GESY` from explicit account configuration;
- acupuncture remained internationally `guideline_conflict_or_mixed` while Cyprus remained separately `against`;
- manual therapy international mixed state remained unchanged with separate local `conditional_for` context;
- local-only/admin items did not become clinical evidence items;
- selected rehabilitation directions remained unchanged;
- referral prose did not gain Cyprus/GeSY/source contamination;
- safety fail-closed behavior remained active.

No patient identifiers, patient history, password, session cookie, or secret value were used or printed.

## 7. Current state

Jurisdiction Overlay V1 is **released, explicitly active for `CY_GESY`, and production-smoke-verified**.

Any further jurisdiction work requires a fresh bounded slice and fresh authority.