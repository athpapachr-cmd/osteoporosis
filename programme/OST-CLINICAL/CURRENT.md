# OST-CLINICAL CURRENT

> **STATUS:** S1 FRACTURE/FRAGILITY PRE-CODE COMPLETE / IMPLEMENTATION AUTHOR READY.
> **Workstream:** OST-CLINICAL — Osteoporosis evidence, pathways and clinical-rule semantics.
> **Programme branch:** `docs/ost-programme-lifecourse-bootstrap-2026-09-27`.
> **Fresh main at activation:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Root writer lock:** unchanged — `CURRENT_OPERATIONAL.md` remains owned by OST-CAPTURE / PR-1.

## Active bounded task

`S1-FRACTURE-FRAGILITY-SEMANTICS-CORRECTION`

Pre-code source/contract/test-boundary reconciliation is COMPLETE and accepted.

Durable owners:
- routing: `programme/OST-LIFECOURSE/S1-FRACTURE-FRAGILITY-ROUTING-BRIEF.md`;
- programme reconciliation: `programme/OST-CLINICAL/S1-PRECODE-PROGRAMME-RECONCILIATION.md`;
- implementation contract: `programme/OST-CLINICAL/S1-IMPLEMENTATION-BRIEF.md`.

## Authority

A separate implementation author may create a fresh branch and produce a bounded implementation candidate under the implementation brief.

The S1 workstream is non-overlapping with PR-1 transcript implementation and must not mutate PR-1 scope or the root operational lock.

## Exact next action

Start one fresh separate S1 implementation-author conversation using `programme/OST-CLINICAL/S1-IMPLEMENTATION-BRIEF.md`.

The author must STOP after:
- bounded code correction;
- exact required regression evidence;
- local OST-CLINICAL checkpoint;
- draft implementation PR;
- exact-head handback for independent post-code review.

## Forbidden

- merge/deploy/smoke;
- Product Constitution mutation;
- Q1–Q12 resolution;
- new data stores/schema/database;
- LifeCourse implementation;
- self-approval as independent reviewer.