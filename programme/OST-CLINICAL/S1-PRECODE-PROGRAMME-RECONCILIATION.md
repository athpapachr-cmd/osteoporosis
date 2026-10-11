# Module-01 S1 — Pre-code programme reconciliation

> **STATUS:** PRE-CODE RECONCILIATION ACCEPTED / IMPLEMENTATION CANDIDATE AUTHORIZED.
> **Date:** 2026-09-27.
> **Programme owner:** Clinical Excellence programme coordinator.
> **Clinical workstream:** OST-CLINICAL.
> **Root writer lock:** unchanged — OST-CAPTURE / PR-1.

## Accepted source/result

The returned S1 pre-code handback is accepted as the governing correction contract for the implementation candidate, subject to fresh source verification by the implementation author.

Fresh source identity at coordinator reconciliation:
- main: `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`;
- draft PR #121 head before this checkpoint: `75543d7bcad706b9295c438b00d81563cefd7848`.

## Confirmed defect classes accepted

### D1 — generic fracture auto-promotes prior fragility
`collectFractureEventsFromDom()` may set `risk_context.prior_fragility_fracture=true` merely because an event exists.

### D2 — generic event contaminates last-fragility recency
Generic fracture site/month may be copied into compatibility fields later interpreted as last fragility fracture.

### D3 — load/render can synthesize a new structured event
Legacy `prior_fragility_fracture=true` can cause `ensureStep1FractureSeed()` to create a new UUID/event with unknown `low_trauma`, later persistable on save.

### D4 — G3 summary equates event existence with prior fragility
G3 reads stale `event.fragility` and can report prior fragility from generic fracture existence.

### D5 — encounter archetype bypasses event-level fragility evidence
G2 `currentFragilityFracture()` can return positive from archetype alone.

### D6 — historical vertebral path treats unknown as positive
`low_trauma !== "no"` incorrectly counts uncertain/missing vertebral events as fragility.

## Accepted semantic contract

```text
fracture exists
!=
fracture is low-trauma / fragility

confirmed event-level fragility
⇔ normalized low_trauma === "yes"
```

Everything else — `no`, `uncertain`, missing, empty or unrecognised — is not a deterministic positive fragility assertion.

Authoritative / derived boundaries:
- `fracture_history.events[]` = structured fracture-fact owner;
- event `low_trauma` = canonical event-level low-trauma/fragility assertion;
- `risk_context.prior_fragility_fracture` = compatibility/summary, not independent factual authority;
- `last_fracture_site/month` = compatibility summary fields, not peer fracture facts;
- `event.fragility` = stale/non-canonical current representation;
- G1 generic fracture context remains generic;
- `fracture_on_treatment` remains separate from fragility;
- G2 fragility values are derived interpretations;
- G3 summary is read-only presentation projection;
- encounter archetype is visit intent/context, not trauma-mechanism authority.

## Backward compatibility accepted

Existing protected records are not migrated or rewritten by S1.

Preserve raw legacy values and provenance, but do not manufacture event-level truth from them.

Hard rules:
- unknown/uncertain is not converted to `no`;
- missing is not converted to `yes` or `no`;
- legacy positive flag is not converted into invented `low_trauma=yes`;
- legacy `event.fragility=yes` is not silently copied to canonical `low_trauma`;
- stable event IDs are preserved;
- load/render must not create new factual events.

## Implementation-author authority

The Product Owner instructed the programme to proceed after accepting the pre-code result. This authorizes one **bounded implementation candidate** under OST-CLINICAL.

The implementation author may:
- create one fresh implementation branch from fresh main;
- modify only the accepted existing-owner S1 paths;
- add/extend focused deterministic regression tests;
- make the minimum CI path-trigger/wiring change needed so S1 changes actually execute the focused gate;
- open a draft implementation PR after exact-head tests pass;
- update `programme/OST-CLINICAL/CURRENT.md` on that implementation branch as the local workstream checkpoint.

The implementation author may NOT:
- mutate `CURRENT_OPERATIONAL.md` or steal the PR-1 root lock;
- modify PR-1 transcript implementation;
- introduce schema/database/persistence migrations;
- create a new fracture/event store or projection architecture;
- change LifeCourse/Product Constitution design;
- resolve Q1–Q12;
- merge/deploy/smoke;
- act as the independent post-code reviewer.

## Accepted implementation boundary

Expected product files are limited to:
- `static/baseline-audit/app-core.js`;
- `static/baseline-audit/osteoporosis-evidence-guidance-core.js`;
- `static/baseline-audit/osteoporosis-longitudinal-summary-core.js`;
- `static/baseline-audit/progressive-guidance-ui.js` only where G3 contaminated wording/state requires it;
- focused test files;
- minimum CI workflow/path wiring required to execute those tests.

`progressive-guidance-core.js` should remain unchanged unless a focused test proves a real S1 defect there.

No S1 correction is expected in:
- `clinical_data.py`;
- DB/schema;
- `patient-registry.js`;
- protected encounter persistence architecture.

## Regression contract

At minimum prove:
- `low_trauma=yes` may support positive fragility;
- `low_trauma=no` never becomes positive fragility;
- `low_trauma=uncertain` never becomes positive fragility;
- missing/empty `low_trauma` never becomes positive fragility;
- interval fracture + yes preserves valid G2 fragility behavior;
- interval fracture + no preserves generic fracture but suppresses fragility-specific R05/R06;
- vertebral + no/uncertain/missing does not enter vertebral fragility count/recency/R08 from that event alone;
- fracture-on-treatment + no preserves R07 but not R05/R06;
- fracture-on-treatment + yes preserves R07 and valid fragility guidance;
- fragility-labelled archetype alone cannot manufacture current fragility without confirmed event evidence;
- traumatic/uncertain/blank event does not auto-write compatibility positive;
- legacy prior=true + zero events survives load/render/save with zero structured events and no new UUID;
- G3 may report a generic fracture count without calling it confirmed fragility;
- repeated stable event ID remains one event;
- unrelated denosumab/milestone/VFA/generic-fracture behavior remains green.

## Release boundary

Implementation candidate completion does not authorize merge, deploy or smoke.

After implementation/testing, route the exact head to one fresh independent post-code S1 review.

## Programme disposition

```text
S1 PRE-CODE: COMPLETE / ACCEPTED
S1 IMPLEMENTATION: AUTHORIZED AS BOUNDED CANDIDATE
S1 MERGE: NOT AUTHORIZED
S1 DEPLOY: NOT AUTHORIZED
ROOT WRITER TRANSFER: NO
Q1–Q12: PARKED
```