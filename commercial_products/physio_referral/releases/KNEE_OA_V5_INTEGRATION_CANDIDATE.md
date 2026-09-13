# Knee OA V5 — fresh-main integration candidate

> **STATUS:** SUPERSEDED BY MERGED / DEPLOYED V5 RELEASE.
> **Recorded:** 2026-09-13 Asia/Nicosia.
> **Diagnosis:** Knee Osteoarthritis only.
> **Historical integration base:** `8aeb91ae37b83caaa188054128db98e04b638fd8`.
> **Historical integration branch:** `feat/physio-knee-oa-v5-integration-2026-09-13`.
> **Substantive integrated head:** `9a7f360745710deacb0ff82f03249723bdfe87d6`.
> **Substantive gate:** `34736389860` — SUCCESS.
> **Final reviewed PR head:** `4e4bd2ae40c606562a982b3e38f9f859b49986eb`.
> **Release PR:** `#101` — squash-merged.
> **Release runtime:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE.
> **Post-V5 authenticated smoke:** `34737351354` — QUEUED / NOT YET EXECUTED at this reconciliation.

## Historical role of this record

This file records the fresh-main integration candidate that resolved the independent-review blocker on the older V5 branch.

The original reviewed V5 candidate was:

```text
head  a47357c602120d3678e8f2f23b99775e616c79e1
gate  34693751545 — SUCCESS
```

Independent review returned:

```text
ACCEPT WITH REQUIRED CHANGES
PATCH V5 THEN MERGE
```

The required change was fresh-main integration rather than clinical redesign because `CY_GESY` and later protected Cockpit/shared work had landed after the historical candidate ancestry.

## Product Owner simplification retained in release

The Product Owner explicitly accepted removal of duplicated `Ατροφία τετρακεφάλου` from the weakness second-tap sheet while preserving it as an objective examination finding under:

```text
Περισσότερα → Εξέταση → Ατροφία τετρακεφάλου
```

The deployed weakness refinement contains only:

- `Μυϊκή αδυναμία στην εξέταση`;
- `Αδυναμία τετρακεφάλου στην εξέταση`.

The weakness count excludes separately selected atrophy.

## Candidate verification

The final PR head `4e4bd2ae40c606562a982b3e38f9f859b49986eb` passed all relevant Physio-owned gates and CU-1 focused tests before release.

V5 gate: `34736919952` — SUCCESS.

Final V5 PR-head artifact: `10311660626`, digest `sha256:003311ab6d45297211d9c5e24bfb26768d26d6c435c9b9069e7fff5acf140277`.

## Release supersession

On explicit Product Owner command `RELEASE V5`, PR #101 was squash-merged to:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

Render auto-deploy `dep-daj2398u01pc738ojvkg` reached `live` at that exact commit.

Therefore this candidate is no longer the current release authority. Current lifecycle truth belongs in:

- `CURRENT_OPERATIONAL.md`;
- `SLICE_PLAN_CURRENT.md`;
- `commercial_products/physio_referral/CURRENT.md`;
- `commercial_products/physio_referral/releases/KNEE_OA_V1_CURRENT.md`;
- `commercial_products/physio_referral/releases/KNEE_OA_V5_RELEASE_CLOSEOUT.md`.

The only remaining release-verification gap at this reconciliation is execution of the queued authenticated post-V5 production smoke. A queued smoke is not a PASS and is not evidence of application failure.
