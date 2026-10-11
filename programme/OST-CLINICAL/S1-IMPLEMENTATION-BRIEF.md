# S1 Fracture / Fragility Semantics — Implementation Author Brief

> **TASK:** `S1-FRACTURE-FRAGILITY-SEMANTICS-CORRECTION`
> **ROLE:** fresh separate implementation author.
> **MODE:** bounded code correction + deterministic tests + draft PR; no merge/deploy/smoke.
> **CLINICAL OWNER:** OST-CLINICAL.
> **ROOT LOCK:** OST-CAPTURE / PR-1 remains unchanged.

## 1. Fresh bootstrap

Fresh-verify `athpapachr-cmd/osteoporosis/main`.

Read the six active canonicals in the order required by `AGENTS.md`.

Then consume fully from draft PR #121 / branch `docs/ost-programme-lifecourse-bootstrap-2026-09-27`:

- `programme/OST-LIFECOURSE/S1-FRACTURE-FRAGILITY-ROUTING-BRIEF.md`
- `programme/OST-CLINICAL/S1-PRECODE-PROGRAMME-RECONCILIATION.md`
- `programme/OST-LIFECOURSE/P1-PROGRAMME-RECONCILIATION.md` for the accepted authoritative-vs-derived fracture ownership boundary only.

Also consume the returned S1 pre-code handback supplied by the Product Owner/coordinator if available in the task context.

Do not assume the prompt-issued SHA remains current. Fresh-inspect all modified runtime owners before editing.

## 2. Role separation

You are NOT:
- the S1 coordinator;
- the OST-LIFECOURSE coordinator;
- the independent post-code reviewer;
- the PR-1 transcript author;
- a merge/deploy/smoke executor.

## 3. Product-owner authority

The Product Owner authorised progression from accepted S1 pre-code reconciliation to one bounded implementation candidate.

You may:
- create one fresh implementation branch from fresh `main`;
- implement only the correction contract below;
- add/extend focused regression tests;
- add the minimum CI path/wiring needed for those tests to run when S1-owned files change;
- create/update `programme/OST-CLINICAL/CURRENT.md` on the implementation branch as the local workstream checkpoint;
- open one draft implementation PR after exact-head verification.

You may NOT merge, deploy or production-smoke.

## 4. Required local workstream checkpoint

On the implementation branch create:

`programme/OST-CLINICAL/CURRENT.md`

with at minimum:
- task ID;
- fresh base-main SHA;
- implementation branch;
- exact current state;
- files in bounded mutation scope;
- tests required/run;
- draft PR identity when opened;
- exact next action;
- explicit forbidden actions;
- statement that root `CURRENT_OPERATIONAL.md` remains PR-1-owned.

Do not modify root `CURRENT_OPERATIONAL.md` for this non-overlapping parallel workstream.

## 5. Governing safety invariant

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

No other value may deterministically become positive fragility.

## 6. Accepted authoritative / derived boundaries

- `fracture_history.events[]` = current structured fracture-fact owner.
- stable event `id` must be preserved.
- `low_trauma` = canonical event-level low-trauma/fragility assertion.
- `risk_context.prior_fragility_fracture` = compatibility/summary, not independent event-level fact authority.
- `risk_context.last_fracture_site/month` = compatibility summary fields, not peer fracture facts.
- `event.fragility` = stale/non-canonical current representation.
- G1 `new_events.fracture` remains generic fracture context.
- `fracture_on_treatment` remains distinct from fragility.
- G2 fragility state/counts are derived interpretations.
- G3 summary is read-only presentation projection.
- encounter archetype is visit intent/context, not trauma-mechanism authority.

## 7. Confirmed defect paths to correct

### D1 — Any fracture event auto-promotes prior fragility
`collectFractureEventsFromDom()` must not set `prior_fragility_fracture=true` simply because events exist.

### D2 — Generic event contaminates last-fragility fields
Generic fractures must not overwrite compatibility fields that are later interpreted as the last fragility fracture. If compatibility synchronization is retained, use only the latest confirmed `low_trauma=yes` event.

### D3 — Render/load synthesizes new factual event
`ensureStep1FractureSeed()` or equivalent load/render behavior must not create a new structured fracture event/UUID merely from a legacy prior-fragility compatibility flag.

### D4 — G3 equates event existence with prior fragility
G3 may count/document generic fracture events, but generic event count must not imply `prior_fragility_fracture=true`, and stale `event.fragility` must not be treated as canonical current evidence.

### D5 — Encounter archetype bypasses event evidence
G2 `current_fragility_fracture` must not become true from `post_fragility_fracture` or `fracture_on_treatment` archetype alone. Existing structured current-event confirmation must require positive event-level evidence.

### D6 — Historical vertebral unknown treated as positive
Historical vertebral fragility counting must use `low_trauma === "yes"`, not `!== "no"`.

## 8. Behaviors that must remain separate/preserved

- generic G1 fracture detection;
- `fracture_on_treatment` relationship semantics;
- R07 reassessment behavior where currently valid independent of fragility;
- VFA/vertebral-fracture handling that is based on vertebral fracture as an imaging/fracture fact rather than fragility classification;
- denosumab and unrelated treatment/milestone rules;
- actual persisted raw historical payloads and existing event IDs.

## 9. Backward-compatibility contract

Do not migrate or rewrite existing protected records as part of S1.

Required handling:
- prior=true + low_trauma=yes → confirmed event-level fragility may be derived;
- prior=true + low_trauma=no → preserve raw inconsistency; event is not fragility;
- prior=true + low_trauma=uncertain → preserve raw record; no positive derived fragility;
- prior=true + missing/empty low_trauma → preserve legacy claim; no invented event-level positive;
- prior=true + zero structured events → preserve legacy raw fields; do not synthesize event/UUID;
- prior=false + low_trauma=yes → structured event controls derived interpretation; do not backfill historical records automatically;
- legacy event.fragility=yes + no low_trauma → preserve raw legacy value/provenance; do not copy to low_trauma;
- same event ID repeated → preserve identity; no duplicate event creation.

Fail closed on conflicting historical snapshots. Do not silently prefer a positive interpretation.

## 10. Expected implementation files

Bounded expected product scope:

- `static/baseline-audit/app-core.js`
- `static/baseline-audit/osteoporosis-evidence-guidance-core.js`
- `static/baseline-audit/osteoporosis-longitudinal-summary-core.js`
- `static/baseline-audit/progressive-guidance-ui.js` only if needed to remove contaminated G3 wording/state
- focused regression tests
- minimum CI workflow/path trigger changes necessary for S1 tests to execute on S1-owned file changes
- `programme/OST-CLINICAL/CURRENT.md`

`static/baseline-audit/progressive-guidance-core.js` should remain unchanged unless a focused failing test demonstrates an actual S1 defect there.

Do not modify for S1:
- `clinical_data.py`
- DB/schema/migrations
- `patient-registry.js` storage transport
- PR-1 transcript files
- Product Constitution / OST-LIFECOURSE architecture.

## 11. Required regression matrix

Implement deterministic coverage for at least:

1. `low_trauma=yes` → may support positive fragility;
2. `low_trauma=no` → generic fracture retained, no positive fragility;
3. `low_trauma=uncertain` → generic fracture retained, no positive fragility;
4. missing/empty `low_trauma` → generic fracture retained, no positive fragility;
5. interval fracture=yes + low_trauma=yes → valid current-fragility G2 R05/R06 remains active;
6. interval fracture=yes + low_trauma=no → generic fracture remains, R05/R06 absent;
7. vertebral + no → fragility vertebral count 0 / recent fragility false; generic VFA handling preserved;
8. vertebral + uncertain/missing → fail-closed fragility; no R08 from that event alone;
9. fracture_on_treatment + no → R07 preserved; R05/R06 absent;
10. fracture_on_treatment + yes → R07 plus valid fragility rules;
11. `post_fragility_fracture` archetype + no confirmed mechanism → archetype alone does not manufacture current fragility;
12. traumatic/uncertain/blank event through app writer → no auto-write prior_fragility=true;
13. legacy prior=true + zero events + load/render/save → zero events remain zero; no new UUID;
14. legacy prior=true + uncertain/missing event → raw state preserved, no downstream upgrade to confirmed fragility;
15. G3 traumatic fracture → generic documented count allowed, but no positive prior-fragility statement from count alone;
16. repeated stable event ID → one event;
17. unrelated denosumab/milestone/generic fracture/VFA behavior remains green.

Test ownership guidance from pre-code reconciliation:
- extend `test_g2_evidence_guidance_node.js`;
- extend `test_g3_guidance_summary_node.js`;
- add focused app-core/browser regression for writer + synthetic-seed behavior;
- preserve `test_progressive_guidance_node.js` generic fracture behavior;
- preserve existing G2/G3 wiring tests.

Also inspect CI path triggers: current S1 evidence found that app-core.js changes may not trigger the relevant G2/G3 gate. Add only the minimum wiring required so the focused S1 regression suite cannot be skipped by path filtering.

## 12. Reuse-before-new-path

Do not create a new fracture helper/store/engine merely to avoid touching the existing owner unless source evidence proves extension is unsafe.

Prefer the smallest coherent correction inside current owners.

## 13. Implementation completion gate

Implementation candidate is ready only when:
- source delta is bounded to the S1 contract;
- all required focused tests pass;
- relevant existing regressions pass;
- CI trigger coverage is adequate for changed S1 paths;
- no raw historical fact is silently rewritten;
- no new structured event is created by render/load compatibility behavior;
- exact head SHA is known;
- `programme/OST-CLINICAL/CURRENT.md` records tested candidate state;
- draft PR is opened with correct Canonical Impact Declaration;
- author returns a concise exact-head handback.

## 14. Draft PR canonical-impact declaration

Use the repository template and declare the actual diff. Expected shape if the implementation is release-affecting:

```text
release_affecting: yes
checkpoint_stage: implementation_tested
root_current: none
slice_plan: none
todo: none
clinical_excellence_plan: none
workstream_current: update
workstream_current_path: programme/OST-CLINICAL/CURRENT.md
changelog: defer_until_completion
reason: Bounded Module-01 S1 fracture/fragility semantics correction using existing owners; root PR-1 lock unchanged.
```

Adjust only if the actual diff/state requires it.

## 15. Required handback

Return:

IMPLEMENTATION SOURCE IDENTITY
→ BRANCH / BASE
→ FILES CHANGED
→ EXACT SEMANTIC CORRECTIONS
→ BACKWARD-COMPATIBILITY BEHAVIOR
→ TESTS / CI EVIDENCE
→ DRAFT PR IDENTITY
→ EXACT HEAD SHA
→ WORKSTREAM CURRENT STATUS
→ RESIDUAL RISKS
→ INDEPENDENT REVIEW READY = YES | NO
→ STOP

Do not perform the independent review yourself.
Do not merge, deploy or smoke.