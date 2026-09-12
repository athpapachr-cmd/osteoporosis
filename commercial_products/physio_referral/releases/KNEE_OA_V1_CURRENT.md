# Knee OA v1 — current commercial release record

> **STATUS:** RUNTIME RELEASED / V4 REFINEMENT MERGED / DEPLOYED / LIVE PUBLIC-ASSET + AUTH-BOUNDARY SMOKE PASS.
> **Diagnosis:** Knee Osteoarthritis only.
> **Original runtime release PR:** `#87`.
> **Original runtime release SHA:** `eeec7f82b4f5a9054e7df51354803dd75e7dc9eb`.
> **Current runtime refinement PR:** `#89`.
> **Exact tested v4 PR head:** `602740d12da0de8ad9a6134f8adf48f4b6f7c691`.
> **Current runtime refinement SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Render deploy:** `dep-daiivp0jo6nc73bl6pug` — LIVE at exact v4 runtime refinement SHA.
> **External live boundary-smoke:** `34689920602` — SUCCESS.
> **V4 release/deploy closeout:** `KNEE_OA_V1_V4_RELEASE_CLOSEOUT.md`.
> **Original v1 release/deploy closeout:** `KNEE_OA_V1_RELEASE_CLOSEOUT.md` — retained as historical evidence.
> **Review archive:** `../reviews/archive/2026-09-11-knee-oa-candidate-6539351c/`.

## Product surface released

- explicit OA diagnosis assertion + laterality;
- live deterministic Greek referral, no routine Generate button;
- compact v4 clinical-picture surface: four equal desktop controls / mobile 2×2 grid;
- `Πόνος`, `Δυσκαμψία`, `Αδυναμία`, `Λειτουργικότητα` open focused contextual sheets;
- direct manual editing with stale-text reconciliation;
- reviewed active-rehab defaults;
- evidence-aware suggestions and mixed-guideline disclosure;
- pain/stiffness/weakness qualifiers with symptom/finding/diagnosis separation;
- advanced examination only on demand;
- `★ Συχνά / Σχετικά τώρα / Όλα` scan-first advanced UI;
- mobile clinical controls retain ≥44 px targets and wrap under large-text enlargement without horizontal overflow;
- Cyprus/GeSY local-context seam without item-level activation;
- ephemeral patient draft; no analytics/patient persistence.

## Production architecture

The product is deployed at:

`/clinical/clinic-utilities/physio-referral`

Existing authentication and real CU-1 validation/safety remain server-owned. Shared product projection remains deterministic; the local prototype remains loopback-only.

The v4 release is a presentation/prose refinement over the released Knee-OA vertical, not a new clinical authority layer.

## Deployment and smoke

PR `#89` was squash-merged only after the exact v4 head passed the dedicated v4, prototype, Cockpit integration, CU-1 and evidence gates.

Render auto-deployed merge SHA `bf527e3836a18491b2758fd293b42f82e0924382` as deployment `dep-daiivp0jo6nc73bl6pug`, which reached `live` at 2026-09-12 10:58:27 UTC.

External live smoke run `34689920602` verified the current product/v4 assets (`200`), the live mobile reflow correction, the production transport boundary, and the unauthenticated protection boundary (`401` for protected page/API). No production credential, session cookie or patient data was used.

Consequently, a full authenticated live end-to-end production smoke remains **not performed by this closeout** and is not silently inferred from exact-head protected FastAPI/Chromium tests.

## Historical reviews

Four specialist reviews plus one supplementary combined review remain archived byte-for-byte with exact SHA-256 hashes. The supplementary `Knee OA Physiotherapy Review.txt` is not a second specialist PT vote.

## Remaining validation

```text
actual iPhone Safari / VoiceOver                    NOT YET PROVEN
authenticated live production E2E                  NOT PERFORMED BY THIS CLOSEOUT
receiving-physiotherapist field validation         NOT YET PROVEN
paid conversion / retention                        NOT YET PROVEN
real clinical pilot                                NOT YET PROVEN
Cyprus/GeSY item-level overlay                     NOT ACTIVATED
second diagnosis                                   NOT SELECTED / NOT AUTHORIZED
```
