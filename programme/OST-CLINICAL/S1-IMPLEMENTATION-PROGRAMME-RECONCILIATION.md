# S1 Fracture / Fragility Semantics — Implementation programme reconciliation

> **STATUS:** IMPLEMENTATION CANDIDATE VERIFIED / INDEPENDENT POST-CODE REVIEW READY.
> **Date:** 2026-09-27.
> **Implementation PR:** #123 — OPEN / DRAFT / NOT MERGED.
> **Exact implementation head:** `f4fe36bcfbd0ca475768844387536ec71e2fd38d`.
> **Base main:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Root writer:** unchanged — OST-CAPTURE / PR-1.

## Fresh programme verification

Programme coordinator independently verified:

- `main` remains `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`;
- PR #123 is OPEN / DRAFT / mergeable;
- exact head is `f4fe36bcfbd0ca475768844387536ec71e2fd38d`;
- branch is `behind_by=0` with merge-base equal to exact base main;
- changed-file set is exactly the declared bounded S1 scope;
- no schema/database/persistence/PR-1/LifeCourse architecture spillover is present.

## Exact changed-file set

1. `.github/workflows/g3-guidance-summary-tests.yml`
2. `programme/OST-CLINICAL/CURRENT.md`
3. `static/baseline-audit/app-core.js`
4. `static/baseline-audit/osteoporosis-evidence-guidance-core.js`
5. `static/baseline-audit/osteoporosis-longitudinal-summary-core.js`
6. `static/baseline-audit/progressive-guidance-ui.js`
7. `test_g2_evidence_guidance_node.js`
8. `test_g3_guidance_summary_node.js`
9. `test_g3_guidance_summary_wiring.js`
10. `test_s1_fracture_fragility_app_core.js`

`progressive-guidance-core.js` remains unchanged.

## Exact-head CI verification

All relevant PR-triggered workflows are green on exact head:

- Canonical impact guard — `36329166319` — SUCCESS;
- G1 progressive guidance foundation — `36329166357` — SUCCESS;
- G2 evidence guidance runtime — `36329166353` — SUCCESS;
- G3 guidance salience longitudinal summary — `36329166349` — SUCCESS.

## Accepted implementation intent

The implementation claims to enforce:

```text
FRACTURE EXISTS != FRAGILITY FRACTURE
MISSING / UNKNOWN / UNCERTAIN TRAUMA MECHANISM != POSITIVE FRAGILITY FACT
CONFIRMED EVENT-LEVEL FRAGILITY ⇔ normalized low_trauma === "yes"
```

and to preserve generic fracture, fracture-on-treatment, VFA and unrelated treatment/milestone behavior.

## Backward-compatibility intent

Accepted implementation intent remains:

- no migration/backfill;
- no synthetic event/UUID creation from legacy compatibility flags;
- preserve raw legacy values without upgrading them into canonical event-level truth;
- preserve stable event IDs;
- fail closed on unknown/uncertain/conflicting mechanism state;
- do not silently backfill compatibility summary fields as authoritative facts.

## Programme disposition

```text
S1 PRE-CODE: COMPLETE
S1 IMPLEMENTATION: COMPLETE / TESTED CANDIDATE
S1 INDEPENDENT POST-CODE REVIEW: READY
S1 MERGE: NOT AUTHORIZED
S1 DEPLOY: NOT AUTHORIZED
S1 SMOKE: NOT AUTHORIZED
ROOT WRITER TRANSFER: NO
Q1–Q12: PARKED
```

## Exact next action

Run one fresh independent READ-ONLY exact-head post-code review against PR #123 head `f4fe36bcfbd0ca475768844387536ec71e2fd38d`.

The reviewer must independently inspect code and execute/verify the bounded test evidence, then issue PASS or BLOCK and STOP.

No implementation correction, merge, deploy or smoke is authorized by this reconciliation.