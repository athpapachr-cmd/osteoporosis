# SLICE_PLAN_CURRENT.md — Knee-OA v1 post-smoke v4 refinement closed

> **STATUS:** CLOSED — v4 runtime refinement merged and deployed; live public-asset/auth-boundary smoke passed; release closeout recorded.
> **Bounded refinement:** post-release Knee-OA v4 clinical-picture UX + referral-prose correction.
> **Refinement branch:** `fix/physio-knee-oa-clinical-sheet-v4-2026-09-12`.
> **Refinement PR:** `#89`.
> **Exact tested PR head:** `602740d12da0de8ad9a6134f8adf48f4b6f7c691`.
> **Runtime refinement SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Render deploy:** `dep-daiivp0jo6nc73bl6pug` — LIVE at exact runtime refinement SHA.
> **External live boundary-smoke:** `34689920602` — SUCCESS.
> **Writer:** NONE after the bounded release-closeout PR merges.
> **Next implementation slice:** NONE SELECTED.

## 1. Completed

The Knee-OA product lineage now includes:

```text
design/evidence contract
→ deterministic referral contract
→ interaction/traceability contract
→ functional prototype
→ qualifier refinement
→ four specialist reviews + supplementary combined review
→ cross-review amendment
→ usability refinement
→ scan-first advanced redesign
→ protected Cockpit production integration
→ initial release PR #87 / deploy / live boundary smoke
→ Product Owner post-release finding
→ v4 compact clinical-picture redesign + prose correction
→ inherited browser-path migration
→ mobile 44 px target correction
→ large-text reflow correction
→ exact-head full PR gates
→ PR #89 squash merge
→ exact-SHA Render auto-deploy
→ external v4 public-asset + auth-boundary smoke
→ v4 release closeout
```

## 2. V4 scope that is now closed

V4 changed only the bounded presentation/referral-prose layer:

- four equal clinical parent controls on desktop;
- 2×2 clinical grid on mobile;
- focused sheets for pain, stiffness, weakness and function;
- explicit reopen/remove behavior while preserving dependent-state hygiene;
- deterministic Greek prose corrections;
- ≥44 px mobile targets;
- no horizontal overflow under large text enlargement.

Preserved unchanged:

```text
CU-1 taxonomy and safety authority
evidence-state model
suggestion semantics
More-v3 / Favorites behavior
manual-text reconciliation
jurisdiction strategy
single Knee-OA diagnosis vertical
```

No second diagnosis, evidence update, new Cyprus/GeSY rule, patient persistence, analytics or autonomous recommendation behavior was added.

## 3. Exact pre-merge gate evidence

At exact PR head `602740d12da0de8ad9a6134f8adf48f4b6f7c691`:

```text
34689630421  Physio Knee OA clinical-sheet v4 gate    SUCCESS
34689630422  Physio Knee OA prototype gate            SUCCESS
34689630419  Physio Knee OA Cockpit integration gate  SUCCESS
34689630432  CU-1 focused tests                        SUCCESS
34689630440  Physio Knee OA evidence design gate       SUCCESS
```

Exact-head v4 artifact:

`10297161518`

Digest:

`sha256:3ca3d329148752adc2764781e48e6ca189359b6fbd3f69c423b4d51770e06b0b`

## 4. Deployment state

Production route remains:

`/clinical/clinic-utilities/physio-referral`

Render auto-deploy `dep-daiivp0jo6nc73bl6pug` reached `live` at exact runtime refinement SHA `bf527e3836a18491b2758fd293b42f82e0924382` on 2026-09-12 10:58:27 UTC.

Existing authentication and real CU-1 validation/safety remain authoritative. The local loopback prototype is not mounted as a public clinical endpoint.

## 5. Smoke precision

Temporary no-secret run `34689920602` passed against the live Render service.

It proved:

- current Knee-OA and v4 assets are externally visible and return `200`;
- the production loader references the v4 JS/CSS;
- the live v4 CSS contains the 44 px target and large-text wrapping correction;
- production transport remains non-synthetic;
- protected page and product bootstrap reject unauthenticated access with `401`.

It did **not** use an authorized production credential/session. Full authenticated live end-to-end smoke therefore remains a separate future verification step and must not be silently inferred from the exact-head protected FastAPI/Chromium tests.

## 6. Deferred validation, not retroactive release blockers

```text
actual iPhone Safari / VoiceOver
authorized authenticated live E2E smoke
receiving-physiotherapist field validation
real clinical pilot
paid conversion / retention
Cyprus/GeSY item-level activation
second-diagnosis selection
```

## 7. Exact next governance state

No active engineering writer. No second diagnosis is authorized. Future work requires a fresh bounded slice and explicit writer claim, preferably driven by actual clinician/receiver/device/market evidence rather than feature-count pressure.
