# RF CURRENT — learned medication dictionary

> **STATUS:** ACTIVE / IMPLEMENTATION AUTHORIZED.
> **Workstream:** native RF v2 Clinic Utility.
> **Branch:** `feat/rf-learned-medication-dictionary-2026-09-26`.
> **Base main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Scope:** medication parsing/classification UX only.
> **Root writer lock:** unchanged; PR-1 remains the repo-wide CURRENT_OPERATIONAL owner.

## Product-owner evidence

The clinician reports that real RF medication paste currently recognizes too few medications in practice and explicitly requested implementation of a mechanism that lets unknown drugs be manually classified and remembered for future use.

This is new material product evidence and reopens only the bounded RF medication-recognition surface.

## Frozen design

```text
paste medication list
→ built-in curated medication dictionary
→ clinician-learned server-side alias dictionary
→ unknown lines remain visible
→ clinician classifies unknown as NSAID / other
→ only normalized drug alias + classification are retained
→ next parse recognizes the learned alias
```

Safety/data rules:

- patient medication paste/source lines are not persisted in the learned dictionary;
- dose/duration are not persisted as dictionary metadata;
- learned aliases are clinician-confirmed, not AI-inferred;
- unknown never defaults silently to NSAID or non-NSAID;
- built-in curated mappings take precedence over learned aliases;
- learned dictionary is server-side, not browser localStorage/sessionStorage;
- RF application medication capacity remains 0..3 NSAIDs + 0..3 other analgesics;
- no RF indication/PDF/imaging/procedure-history semantics change.

## Implementation scope

- persistent learned medication-alias table;
- list/upsert/delete protected RF dictionary endpoints;
- parser support for learned aliases and explicit unrecognized candidates;
- unknown-medication classification UI;
- manual add-to-NSAID / add-to-other rows with explicit "learn" action;
- visible learned provenance;
- focused parser/API/persistence/UI regression coverage.

## Exact next action

Implement the bounded medication-learning slice and run focused RF tests. Do not open a release PR, merge or deploy without a later evidence checkpoint and release decision.
