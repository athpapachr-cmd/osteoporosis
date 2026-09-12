# Knee OA v1 — current commercial release record

> **STATUS:** RUNTIME RELEASED / MERGED / DEPLOYED / LIVE PUBLIC-ASSET + AUTH-BOUNDARY SMOKE PASS.
> **Diagnosis:** Knee Osteoarthritis only.
> **Runtime release PR:** `#87`.
> **Runtime squash-merge SHA:** `eeec7f82b4f5a9054e7df51354803dd75e7dc9eb`.
> **Render deploy:** `dep-daifjt0jo6nc73biqhpg` — LIVE at exact runtime release SHA.
> **External live boundary-smoke:** `34680029691` — SUCCESS.
> **Release review:** `KNEE_OA_V1_RELEASE_REVIEW.md` — PASS / no open blocker before merge.
> **Release/deploy closeout:** `KNEE_OA_V1_RELEASE_CLOSEOUT.md`.
> **Review archive:** `../reviews/archive/2026-09-11-knee-oa-candidate-6539351c/`.

## Product surface released

- explicit OA diagnosis assertion + laterality;
- live deterministic Greek referral, no routine Generate button;
- direct manual editing with stale-text reconciliation;
- reviewed active-rehab defaults;
- evidence-aware suggestions and mixed-guideline disclosure;
- pain/stiffness/weakness qualifiers with symptom/finding/diagnosis separation;
- advanced examination only on demand;
- `★ Συχνά / Σχετικά τώρα / Όλα` scan-first advanced UI;
- Cyprus/GeSY local-context seam without item-level activation;
- ephemeral patient draft; no analytics/patient persistence.

## Production architecture

The product is deployed at:

`/clinical/clinic-utilities/physio-referral`

Existing authentication and real CU-1 validation/safety remain server-owned. Shared product projection is owned by `clinic_utilities/physio_referral_product/knee_oa_projection.py`; the local prototype remains loopback-only.

## Deployment and smoke

Render auto-deployed the exact runtime merge SHA and reported the deployment live.

External live smoke verified current product assets (`200`) and the unauthenticated protection boundary (`401` for protected page/API). Render logs corroborated those requests. No production credential, session cookie or patient data was used.

Consequently, a full authenticated live end-to-end production smoke remains **not yet performed** and is not silently claimed by this record.

## Historical reviews

Four specialist reviews plus one supplementary combined review are archived byte-for-byte with exact SHA-256 hashes. The supplementary `Knee OA Physiotherapy Review.txt` is not a second specialist PT vote.

## Remaining validation

```text
actual iPhone Safari / VoiceOver                    NOT YET PROVEN
authenticated live production E2E                  NOT YET PERFORMED
receiving-physiotherapist field validation         NOT YET PROVEN
paid conversion / retention                        NOT YET PROVEN
real clinical pilot                                NOT YET PROVEN
Cyprus/GeSY item-level overlay                     NOT ACTIVATED
second diagnosis                                   NOT SELECTED
```
