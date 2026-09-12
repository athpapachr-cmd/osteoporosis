# CURRENT_OPERATIONAL.md — Knee-OA v1 v4 runtime refinement released / closeout

> **STATUS:** PHYSIO REFERRAL KNEE-OA V1 — V4 REFINEMENT MERGED / DEPLOYED / LIVE PUBLIC-ASSET + AUTH-BOUNDARY SMOKE PASS / FULL AUTHENTICATED LIVE E2E NOT PROVEN BY THIS CLOSEOUT.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Original runtime release PR / SHA:** `#87` / `eeec7f82b4f5a9054e7df51354803dd75e7dc9eb` — historical baseline.
> **Current runtime refinement PR:** `#89`.
> **Exact tested v4 PR head:** `602740d12da0de8ad9a6134f8adf48f4b6f7c691`.
> **Current v4 squash-merge SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Render deploy:** `dep-daiivp0jo6nc73bl6pug` — `live`, exact source commit `bf527e3836a18491b2758fd293b42f82e0924382`, finished 2026-09-12 10:58:27 UTC.
> **External v4 live boundary-smoke:** run `34689920602` — SUCCESS.
> **Exact-head v4 artifact:** `10297161518`, digest `sha256:3ca3d329148752adc2764781e48e6ca189359b6fbd3f69c423b4d51770e06b0b`.
> **V4 release closeout:** `commercial_products/physio_referral/releases/KNEE_OA_V1_V4_RELEASE_CLOSEOUT.md`.
> **Review archive:** `commercial_products/physio_referral/reviews/archive/2026-09-11-knee-oa-candidate-6539351c/`.
> **ACTIVE RUNTIME / DESIGN WRITER:** NONE.
> **CANONICAL WRITER:** bounded v4 release-closeout branch only until closeout PR merges; NONE thereafter.
> **Real-patient data:** NOT USED for release smoke.

## 1. Released production surface

The reviewed Knee-OA product remains deployed in the existing authenticated Clinical Excellence physiotherapy utility:

`/clinical/clinic-utilities/physio-referral`

The production architecture remains:

```text
existing Clinical Excellence authentication
→ protected physiotherapy Cockpit route/API
→ shared deterministic Knee-OA projection
→ real CU-1 validation/safety authority
→ reviewed evidence/template/interaction contracts
→ v4 compact clinical-picture presentation
→ live Greek referral UI
```

V4 changes the presentation/referral-prose layer only. It does not create a new clinical authority layer.

No new patient persistence, analytics, billing system, unauthenticated public clinical endpoint, second diagnosis, jurisdiction rule or LLM-generated routine referral was introduced.

## 2. V4 product refinement now live

The released v4 correction addresses the Product Owner's post-release finding that parent clinical-picture selections and their qualifiers were visually disconnected.

Current production behavior:

- desktop clinical picture: four equal parent controls in one row;
- mobile clinical picture: compact 2×2 grid;
- `Πόνος`, `Δυσκαμψία`, `Αδυναμία`, `Λειτουργικότητα` each open a focused contextual sheet;
- re-tapping an active parent reopens its sheet;
- removal is explicit and clears dependent qualifier meaning;
- mobile controls retain at least 44 px target height;
- large text enlargement may wrap labels rather than causing horizontal overflow;
- deterministic Greek prose avoids unsupported ranking language such as `κυρίως` when no ranking exists;
- explicit quadriceps examination remains an examination finding;
- passive extension deficit remains distinct from stiffness;
- treatment wording preserves physiotherapist autonomy.

The `Περισσότερα` v3, Favorites, contextual suggestions, direct editing, stale-text reconciliation, evidence model and safety behavior remain preserved.

## 3. Exact release evidence

Exact PR head `602740d12da0de8ad9a6134f8adf48f4b6f7c691` completed all relevant gates successfully before merge:

```text
Physio Knee OA clinical-sheet v4 gate    34689630421   SUCCESS
Physio Knee OA prototype gate            34689630422   SUCCESS
Physio Knee OA Cockpit integration gate  34689630419   SUCCESS
CU-1 focused tests                        34689630432   SUCCESS
Physio Knee OA evidence design gate       34689630440   SUCCESS
```

The final gate sequence identified and corrected both inherited test-path drift and real accessibility defects before merge, including 42 px mobile targets and large-text horizontal overflow.

Immediately before squash merge, PR `#89` was mergeable, had zero unresolved review threads and was `0` commits behind `main`.

## 4. Deployment and external live smoke

Render auto-deploy was enabled on `main`; no duplicate manual deploy was triggered.

Deployment `dep-daiivp0jo6nc73bl6pug` became `live` at exact source commit:

`bf527e3836a18491b2758fd293b42f82e0924382`

Temporary no-secret external smoke run `34689920602` then confirmed:

```text
200 index.html
200 production-env.js
200 production-finalize.js
200 product-more-v3.js
200 product-clinical-sheet-v4.js
200 product-clinical-sheet-v4.css
401 protected physiotherapy route without auth
401 protected product bootstrap without auth
```

The smoke also asserted that the deployed production loader references the v4 assets, the v4 surface is present, the live CSS contains the 44 px target + large-text wrapping correction, and the production transport carries `synthetic_only = false`.

No secret/session/patient data was supplied. The temporary smoke workflow was removed after the successful run.

## 5. Lifecycle boundary

Do **not** rewrite the current state as full authenticated `PRODUCTION-SMOKE-VERIFIED` end-to-end behavior.

No authorized production credential/session was used during this v4 live smoke. Therefore:

```text
V4 MERGED                                    yes
V4 DEPLOYED                                  yes
public-asset live smoke                      pass
unauthenticated auth-boundary live smoke     pass
authenticated live end-to-end smoke          not performed by this closeout
pilot validated                              no
receiver validated                           no
commercially validated                       no
```

## 6. Current hold / next legitimate work

There is no active implementation slice and no active writer.

Before a second diagnosis or meaningful commercial scale-up, prioritize evidence from:

- actual iPhone Safari / VoiceOver use;
- authorized authenticated live-session verification when operationally safe and useful;
- receiving physiotherapists;
- real referral-volume and workflow timing;
- willingness-to-pay / retention discovery;
- item-level Cyprus/GeSY verification if local overlay is activated.

No automatic second-diagnosis expansion is authorized by this release.
