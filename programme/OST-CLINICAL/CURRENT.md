# OST-CLINICAL CURRENT — S1 Fracture / Fragility Semantics

> **TASK:** `S1-FRACTURE-FRAGILITY-P6B-R1-RAW-VALUE-PRESERVATION-CORRECTION`
> **STATUS:** S1 MERGED / P6B PASS / AUTO-DEPLOY LIVE / PRODUCTION SMOKE NOT RUN
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

The focused harness was corrected in test commit `191bc6eed17a9503af2cfc99ae02293dc3ecc118`: load/render is now proved by zero structured events, and UUID stability is measured from **after load/render to after save**, while retaining the earlier explicit render-no-event-UUID regression.

Exact next gate: rerun exact-head CI for the corrected branch and require the full S1 + inherited G1/G2/G3 chain plus Canonical Impact guard to pass before independent review handoff.

## R1 correction green evidence

Exact tested/checkpoint head: `7e6765520b3ce19fba9e87295a329685e0e9b8e0`.

PR-triggered GitHub Actions on that exact head:
- G3 guidance salience longitudinal summary — run `36477440700`: **SUCCESS**;
  - JavaScript syntax: **SUCCESS**;
  - focused `Run S1 fracture / fragility writer and load-render regressions`: **SUCCESS**;
  - G3 salience/summary + wiring/ownership + production-visibility regressions: **SUCCESS**;
  - frozen G2 contract/core/live/wiring regressions: **SUCCESS**;
  - inherited G1 core/wiring/UI/WHY-NOW regressions: **SUCCESS**;
  - authoritative Finish browser + server-finalization regressions: **SUCCESS**;
- G2 evidence guidance runtime — run `36477439960`: **SUCCESS**;
- G1 progressive guidance foundation — run `36477440106`: **SUCCESS**;
- Canonical impact guard — run `36477439970`: **SUCCESS**.

The R1 correction itself remains limited to the existing app-core owner plus its focused regression. G2/G3 semantic files were not changed by the residual correction. The prior independent P6B closures D1–D6 therefore remain exercised by the inherited S1/G2/G3 regression matrix rather than being redesigned.

This final CURRENT mutation is documentation-only over the green corrected head. After it lands, the resulting PR head must be fresh-verified and its PR-triggered checks allowed to complete; no further product/runtime mutation is authorized if those checks remain green.

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

## Independent P6B R1 review disposition

Fresh separate independent delta+cumulative P6B reviewed the exact corrected substantive target:

`ba2f8635372f85cd94409fa66e08c3d8b428bcba`

and returned:

```text
VERDICT: PASS
TARGET DRIFT: NO
D1–D6: PASS / PRESERVED
RAW SOURCE PRESERVATION: PASS
EXPLICIT EDIT PERSISTENCE: PASS
SEMANTIC STABILITY: PASS
PRESERVED G1/G2/G3/VFA/R07/DENOSUMAB BEHAVIOR: PASS
MATERIAL RESIDUAL: NONE
```

The review made no repository mutation and did not authorize merge/deploy/smoke.

Coordinator reconciliation preserves `ba2f863...` as the exact independently reviewed runtime/test target. This CURRENT update is documentation-only and must not be represented as a new independently reviewed runtime head.

Fresh programme check on 2026-09-30 found current `main` at `cbe8d10891d21af2c48da9fabe8bd475f53e584a`, advanced from the original S1 base. The current-main changes inspected from the original base do not directly modify the bounded S1 product/test paths, but release remains a separate lifecycle decision.

## Release reconciliation — 2026-09-30

Fresh programme release reconciliation verified:

```text
fresh main:                 cbe8d10891d21af2c48da9fabe8bd475f53e584a
independently reviewed S1:  ba2f8635372f85cd94409fa66e08c3d8b428bcba
pre-release PR head:        a68ceab003cb58c2ecef36acbbf885b2ee55a157
PR #123:                    OPEN / DRAFT / MERGEABLE
```

Post-review delta:

- exactly one commit;
- only `programme/OST-CLINICAL/CURRENT.md`;
- no runtime/test/workflow semantic delta after the independently reviewed target.

PR-triggered checks on `a68ceab...`:

- Canonical impact guard: SUCCESS;
- G1 progressive guidance foundation: SUCCESS;
- G2 evidence guidance runtime: SUCCESS;
- G3 guidance salience longitudinal summary: SUCCESS.

Fresh-main advancement from the original S1 base is 52 commits. Direct-path overlap with the bounded S1 product/test/workflow surface is **NONE**. The intervening main changes are in Clinical Calendar, Cockpit Home/Surgery Queue, Clinical Learning presentation and associated tests/workflows; they do not modify the S1 fracture/fragility runtime owners or focused regressions.

Release disposition:

```text
INDEPENDENT P6B: PASS
MATERIAL RESIDUAL: NONE
POST-REVIEW RUNTIME DELTA: NONE
FRESH-MAIN DIRECT-PATH OVERLAP: NONE
PR MERGEABLE: YES
RELEASE RECONCILIATION: PASS
MERGE AUTHORITY: GRANTED BY PRODUCT OWNER
DEPLOY/SMOKE AUTHORITY: NOT IMPLIED BY MERGE AUTHORITY
```

The independently reviewed substantive target remains `ba2f863...`. This checkpoint is documentation-only and does not create a new independently reviewed runtime target.

Exact next action:

1. obtain explicit Product Owner merge authority;
2. if granted, squash-merge PR #123;
3. checkpoint the exact merge commit before any further deliberate release action;
4. allow normal Render auto-deploy from `main`;
5. verify deploy identity/status;
6. production-smoke only after the merge/deploy checkpoint and within the bounded S1 semantics.

Do not open another independent S1 review unless runtime/test semantics change.

## Exact next action

1. obtain explicit Product Owner merge authority before merge;
2. if granted, squash-merge PR #123;
3. checkpoint the exact merge commit before deploy/smoke follow-through;
4. allow normal Render auto-deploy from `main`;
5. verify deploy identity/status;
6. production-smoke only after the merge/deploy checkpoint.

Do NOT open another independent S1 review unless the runtime/test delta changes or release reconciliation reveals a material interaction requiring it.

## Product Owner merge authority checkpoint — 2026-09-30

The Product Owner explicitly instructed: `merge`.

PR #123 was then fresh-verified at head `a4d414bb05ffce00ad401ff661133b7c5a6c4458` with:

- Canonical impact guard: SUCCESS;
- G1 progressive guidance foundation: SUCCESS;
- G2 evidence guidance runtime: SUCCESS;
- G3 guidance salience longitudinal summary: SUCCESS;
- post-independent-review delta limited to this workstream CURRENT documentation;
- PR mergeable: YES.

The PR has been moved from DRAFT to READY FOR REVIEW. No code/runtime/test mutation occurred in that transition.

Exact next action after this documentation checkpoint and green checks on the resulting head:

```text
SQUASH MERGE PR #123
→ checkpoint exact merge SHA
→ observe normal Render auto-deploy
→ no production smoke unless separately authorized / procedure clearly permits it
```

## Merge checkpoint — 2026-09-30

Product Owner merge authority was exercised only after exact-head release gates were green.

```text
PR:                         #123
final PR head:              737a0e90899a0cfc991be3fe5192f8249f4c5631
independent reviewed target: ba2f8635372f85cd94409fa66e08c3d8b428bcba
post-review semantic delta:  NONE
merge method:               squash
merge commit:               dd16a505662606f588a2adc8b1ae0ddddd884d19
PR state:                   MERGED
```

Exact final-head checks before merge:

- Canonical impact guard — SUCCESS;
- G1 progressive guidance foundation — SUCCESS;
- G2 evidence guidance runtime — SUCCESS;
- G3 guidance salience longitudinal summary — SUCCESS;
- focused S1 fracture/fragility writer + load/render regression inside G3 — SUCCESS.

Fresh main immediately after merge was verified at the exact merge commit `dd16a505...`.

This checkpoint records the merge. Normal Render auto-deploy followed from `main`. The merge deploy was superseded by the immediate documentation checkpoint deploy, and the resulting deploy `dep-daumvhu417fc73ffcgn0` reached `LIVE` at commit `3c7e502e29bfbc103dc7838a902f3e7adcfca36f`, which contains the same merged S1 runtime plus documentation only. No manual redeploy was triggered. Production smoke has not been run and remains a separate action.

## Explicitly forbidden

- modifying root `CURRENT_OPERATIONAL.md`;
- PR-1 transcript implementation;
- `clinical_data.py`, database/schema/migrations or patient-registry storage transport;
- new fracture store/event bus/lifecourse engine;
- Product Constitution or OST-LIFECOURSE architecture changes;
- resolving Q1–Q12;
- independent post-code review by this author;
- manual redeploy or production smoke without separate authority.
