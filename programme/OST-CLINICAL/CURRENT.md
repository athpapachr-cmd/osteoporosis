# OST-CLINICAL CURRENT — S1 Fracture / Fragility Semantics

> **TASK:** `S1-FRACTURE-FRAGILITY-SEMANTICS-CORRECTION`
> **STATUS:** IMPLEMENTATION TESTED / DRAFT PR NEXT
> **Date:** 2026-09-27 Asia/Nicosia.
> **Fresh base main:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Implementation branch:** `fix/ost-clinical-s1-fracture-fragility-semantics-2026-09-27`.
> **Tested substantive head before this checkpoint:** `bca37aa824e524ac8c2f7783cdf763f19cdcc656`.
> **Root operational owner:** unchanged — `CURRENT_OPERATIONAL.md` remains OST-CAPTURE / PR-1-owned.

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

The existing G3 combined workflow was minimally extended so `app-core.js` and `test_s1_fracture_fragility_app_core.js` are path-triggered, syntax-checked and executed. Full PR-triggered CI remains the next gate.

## Draft PR

Not opened yet. Open only after this docs-only checkpoint head is reverified exactly.

## Exact next action

Reverify this checkpoint head, fresh-check main ancestry, then open one draft implementation PR with the required Canonical Impact Declaration. After the PR identity is known, update this workstream CURRENT with that identity and re-run exact-head/PR CI on the final docs head.

## Explicitly forbidden

- modifying root `CURRENT_OPERATIONAL.md`;
- PR-1 transcript implementation;
- `clinical_data.py`, database/schema/migrations or patient-registry storage transport;
- new fracture store/event bus/lifecourse engine;
- Product Constitution or OST-LIFECOURSE architecture changes;
- resolving Q1–Q12;
- independent post-code review by this author;
- merge, deploy or production smoke.
