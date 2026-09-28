# OST-CLINICAL CURRENT — S1 Fracture / Fragility Semantics

> **TASK:** `S1-FRACTURE-FRAGILITY-P6B-R1-RAW-VALUE-PRESERVATION-CORRECTION`
> **STATUS:** R1 CORRECTION IMPLEMENTED / FIRST EXACT-HEAD CI BLOCKED BY TEST-HARNESS ASSERTION / DRAFT PR #123 OPEN
> **Date:** 2026-09-27 Asia/Nicosia.
> **Fresh base main:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Implementation branch:** `fix/ost-clinical-s1-fracture-fragility-semantics-2026-09-27`.
> **Original S1 tested substantive head:** `bca37aa824e524ac8c2f7783cdf763f19cdcc656`.
> **Blocked P6B predecessor head:** `f4fe36bcfbd0ca475768844387536ec71e2fd38d`.
> **R1 correction implementation commit:** `daea6a9e5b430106b84f8e9c7ef526c65efcb4b0`.
> **Root operational owner:** unchanged — `CURRENT_OPERATIONAL.md` remains OST-CAPTURE / PR-1-owned.

## P6B R1 residual correction checkpoint

Independent P6B reviewed exact head `f4fe36bcfbd0ca475768844387536ec71e2fd38d` and returned **BLOCK** for one residual only: a noncanonical raw `fracture_history.events[].low_trauma` value could render as the blank select option and then be silently rewritten to `""` by an ordinary no-edit save.

The bounded correction at `daea6a9e5b430106b84f8e9c7ef526c65efcb4b0` keeps source preservation separate from clinical interpretation:

- ordinary load → render → save does not write the `low_trauma` DOM value back unless the clinician actually fired an `input/change` on that control;
- exact raw source such as `"unknown"`, `" YES "`, mixed-case/whitespace variants or another legacy token therefore remains byte-for-byte unchanged on no-edit save;
- an explicit clinician edit marks only that rendered control as edited and permits the selected canonical `yes/no/uncertain/""` value to replace the prior raw source;
- the existing normalized G2 semantic interpretation remains unchanged and unknown/unrecognised values remain fail-closed;
- no event, UUID, schema value, migration/backfill, G2 rule or G3 semantic was added or changed.

Correction delta so far:
- `static/baseline-audit/app-core.js`;
- `test_s1_fracture_fragility_app_core.js`;
- this workstream checkpoint.

The focused regression now covers canonical yes/no/uncertain/empty no-edit preservation, exact preservation of `unknown`, `" YES "` and another unrecognised token through the actual load/render/save path, explicit edits to yes/no, semantic interpretation stability before/after no-edit save, and legacy prior=true + zero events through load/render/save.

Repo-local execution is not claimed: the execution container could not resolve `github.com` for a fresh checkout. The executable gate is therefore the PR-triggered GitHub Actions workflow on the exact corrected head.

## First R1 exact-head CI attempt

Checkpoint head `023cbe84335741bc37c289229d77d43fb3ed0e1c` produced:
- Canonical impact guard run `36477246661`: **SUCCESS**;
- G1 progressive guidance run `36477246694`: **SUCCESS**;
- G2 evidence guidance run `36477246759`: **SUCCESS**;
- G3 run `36477246797`: **FAILURE** only at the newly expanded S1 app-core regression.

The failing assertion was test-harness-specific: the zero-event load/render/save case compared the global UUID counter from before `loadCase()`, but `normalizeLoadedCase()` always constructs a base case and therefore calls `createUuid()` twice for non-fracture base identifiers before overlaying the stored case. The stored/loaded/saved fracture-event count remained zero. This does not demonstrate an R1 runtime defect.

Bounded next correction: adjust only the focused S1 regression so load/render is proven by zero structured events and UUID stability is measured from **after load/render to after save**, while retaining the earlier explicit render-no-event-UUID regression. Then rerun the exact-head cumulative gate.

## Governing semantic invariant

```text
FRACTURE EXISTS
!=
FRAGILITY FRACTURE

MISSING / UNKNOWN / UNCERTAIN TRAUMA MECHANISM
!=
POSITIVE FRAGILITY FACT

CONFIRMED EVENT-LEVEL FRAGILITY
⇔ normalized low_trauma === "yes"
```

## Bounded mutation scope / files changed

- `.github/workflows/g3-guidance-summary-tests.yml`
- `programme/OST-CLINICAL/CURRENT.md`
- `static/baseline-audit/app-core.js`
- `static/baseline-audit/osteoporosis-evidence-guidance-core.js`
- `static/baseline-audit/osteoporosis-longitudinal-summary-core.js`
- `static/baseline-audit/progressive-guidance-ui.js`
- `test_g2_evidence_guidance_node.js`
- `test_g3_guidance_summary_node.js`
- `test_g3_guidance_summary_wiring.js`
- `test_s1_fracture_fragility_app_core.js`

`static/baseline-audit/progressive-guidance-core.js` is unchanged from main (blob `ae4ca9422d86887bc5e75cfc1b61d1432bd545b9`).

## Implemented S1 correction

- render/load no longer synthesizes a structured fracture event from legacy `prior_fragility_fracture=true`;
- structured event collection no longer auto-writes `prior_fragility_fracture` or compatibility last-fracture fields;
- G2 current fragility requires interval-fracture context plus confirmed event-level `low_trauma=yes`; encounter archetype cannot manufacture mechanism evidence;
- G2 vertebral fragility count/recency requires explicit `low_trauma=yes`, while generic vertebral/VFA handling remains separate;
- stable-ID snapshots with conflicting explicit low-trauma values fail closed in the derived G2/G3 projection rather than silently preserving a positive interpretation;
- G3 generic fracture count remains generic; confirmed fragility is derived only from structured `low_trauma=yes`;
- stale `event.fragility` is ignored as current fragility authority;
- clinician-facing G3 wording distinguishes generic/legacy fracture history from confirmed low-trauma/fragility events.

No schema, DB, migration or raw-record backfill was introduced.

## Verification completed at substantive head

Exact-head deterministic execution at `bca37aa824e524ac8c2f7783cdf763f19cdcc656`:

- existing G1 progressive-guidance suite: PASS;
- existing G2 suite + full S1 current-fragility / fracture-on-treatment / vertebral / stale-field / stable-ID matrix: PASS;
- existing G3 suite + full S1 generic-vs-confirmed / legacy-zero-event / stale-field / conflict matrix: PASS;
- app-core S1 runtime-equivalent load/render/writer harness: PASS;
- branch merge base = current implementation base, behind = 0 at verification;
- `progressive-guidance-core.js` unchanged.

The existing G3 combined workflow was minimally extended so `app-core.js` and `test_s1_fracture_fragility_app_core.js` are path-triggered, syntax-checked and executed.

## PR-triggered CI evidence

Exact implementation/checkpoint head `442e1e9636ae1e125bda04a5d5123fa124254d17` on draft PR #123:

- G3 guidance salience longitudinal summary — run `36329072458`: **SUCCESS**;
  - step 10 `Run S1 fracture / fragility writer and load-render regressions`: **SUCCESS**;
  - G3 wiring/ownership, G2 contract/core/live/wiring, G1 core/wiring/UI/WHY-NOW, Finish browser and server finalization regressions: **SUCCESS** in the same job;
- G2 evidence guidance runtime — run `36329072454`: **SUCCESS**;
- G1 progressive guidance foundation — run `36329072465`: **SUCCESS**;
- Canonical impact guard — run `36329072455`: **SUCCESS**.

This CURRENT update is documentation-only over that green head; no runtime/test/workflow file changes are introduced by this checkpoint.

## Draft PR

- Draft implementation PR: #123
- URL: https://github.com/athpapachr-cmd/osteoporosis/pull/123
- Base: `main` at `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`
- Opened head: `a6165279a481603bf6b1440a90131137ab3d88cf`
- Canonical Impact Declaration: `release_affecting=yes`, `checkpoint_stage=implementation_tested`, root canonicals `none`, this workstream CURRENT `update`, changelog deferred until completion.
- State: draft / not merged.

## Exact next action

Fresh-verify the PR #123 head containing this R1 checkpoint and require exact-head CI for the focused S1 app-core regression, inherited G1/G2/G3 regression gates, affected syntax/workflow checks and Canonical Impact guard. If green, checkpoint the exact evidence here and hand the resulting head to a **fresh delta+cumulative independent P6B reviewer**. No merge/deploy/smoke.

## Explicitly forbidden

- modifying root `CURRENT_OPERATIONAL.md`;
- PR-1 transcript implementation;
- `clinical_data.py`, database/schema/migrations or patient-registry storage transport;
- new fracture store/event bus/lifecourse engine;
- Product Constitution or OST-LIFECOURSE architecture changes;
- resolving Q1–Q12;
- independent post-code review by this author;
- merge, deploy or production smoke.
