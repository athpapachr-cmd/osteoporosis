# CURRENT_OPERATIONAL.md — Knee-OA v1 released runtime / archival closeout

> **STATUS:** PHYSIO REFERRAL KNEE-OA V1 — MERGED / DEPLOYED / LIVE PUBLIC-ASSET + AUTH-BOUNDARY SMOKE PASS / FULL AUTHENTICATED LIVE E2E NOT YET PROVEN.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Runtime release PR:** `#87`.
> **Runtime squash-merge SHA:** `eeec7f82b4f5a9054e7df51354803dd75e7dc9eb`.
> **Render deploy:** `dep-daifjt0jo6nc73biqhpg` — `live`, exact source commit `eeec7f82b4f5a9054e7df51354803dd75e7dc9eb`.
> **External live boundary-smoke:** run `34680029691` — SUCCESS.
> **Review archive:** `commercial_products/physio_referral/reviews/archive/2026-09-11-knee-oa-candidate-6539351c/`.
> **ACTIVE RUNTIME / DESIGN WRITER:** NONE.
> **CANONICAL WRITER:** closeout/archive branch only until this bounded documentation PR merges; NONE thereafter.
> **Real-patient data:** NOT USED for release smoke.

## 1. Released production surface

The reviewed Knee-OA product is deployed in the existing authenticated Clinical Excellence physiotherapy utility:

`/clinical/clinic-utilities/physio-referral`

The production architecture remains:

```text
existing Clinical Excellence authentication
→ protected physiotherapy Cockpit route/API
→ shared deterministic Knee-OA projection
→ real CU-1 validation/safety authority
→ reviewed evidence/template/interaction contracts
→ live Greek referral UI
```

No new patient persistence, analytics, billing system, unauthenticated public clinical endpoint or LLM-generated routine referral was introduced.

## 2. Release evidence

The exact product/runtime release passed the dedicated Knee-OA Cockpit, inherited product and CU-1 gates before PR/merge. PR `#87` was then squash-merged.

Render auto-deploy produced `dep-daifjt0jo6nc73biqhpg`, which became `live` at the exact runtime merge SHA.

External live smoke run `34680029691` confirmed:

- deployed Knee-OA product assets return `200`;
- the reviewed scan-first v3 assets are present;
- the protected physiotherapy page returns `401` without auth;
- the protected product bootstrap returns `401` without auth;
- no secret/session/patient data was supplied.

Render request logs independently corroborated those responses.

## 3. Important lifecycle boundary

Do **not** rewrite the current state as full `PRODUCTION-SMOKE-VERIFIED` authenticated end-to-end behavior.

No authorized production credential/session was used during the live smoke. Therefore:

```text
MERGED                                      yes
DEPLOYED                                    yes
public-asset live smoke                     pass
unauthenticated auth-boundary live smoke    pass
authenticated live end-to-end smoke         not yet performed
pilot validated                             no
commercially validated                      no
```

## 4. Review history preserved

The four agreed specialist reviews and one supplementary combined review are now archived as immutable source evidence with exact hashes.

The specialist axes remain:

1. Clinical / Evidence
2. Physiotherapy
3. UX / Product
4. Commercial / Product-Market

`Knee OA Physiotherapy Review.txt` is preserved as the supplementary combined / multi-axis review and must not be counted as a second specialist physiotherapy vote.

## 5. Current hold / next legitimate work

There is no active implementation slice.

Before a second diagnosis or meaningful commercial scale-up, prioritize evidence from:

- authorized authenticated live-session verification when operationally safe;
- actual iPhone Safari / VoiceOver;
- receiving physiotherapists;
- real referral-volume and workflow timing;
- willingness-to-pay / retention discovery;
- item-level Cyprus/GeSY verification if local overlay is activated.

No automatic second-diagnosis expansion is authorized by this technical release.
