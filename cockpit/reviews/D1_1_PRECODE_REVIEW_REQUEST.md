# D1.1 — Bounded Independent Pre-Code Closure Request

> MODE: fresh independent READ-ONLY R2 **delta + affected cumulative closure** under `PROCEDURES.md` P5/P5.1.
> TARGET BRANCH: `design/cockpit-d1-1-global-context-2026-10-02`.
> TARGET DESIGN: `cockpit/D1_1_GLOBAL_APPOINTMENT_CONTEXT_DESIGN.md`.
> EXPECTED DESIGN BLOB: `261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33`.
> ORIGINAL TARGET: head `9b11f722c485247754fa7219e9b0f0d9167f8734`, design blob `2b438c949e30d00a1d11d9d1431d766954320f15`.
> PRIOR RESULT: BLOCK / COMPLETE_FOR_DECLARED_SCOPE / 0:2:0 / no additional material finding YES / implementation NO.
> STOP BASIS: answer the three closure questions from the named affected seams, then return the verdict. No time-based substitute for evidence closure.
> RUNTIME / TEST IMPLEMENTATION / MERGE / DEPLOY: forbidden.

## Entry and identity

Fresh-verify remote `main`, read the six root canonicals in AGENTS order once, then `PROCEDURES.md`, `cockpit/CURRENT.md`, `cockpit/PRODUCT_CONSTITUTION.md` and this request from the exact target branch. Read the complete design and supplied result in `cockpit/reviews/D1_1_PRECODE_BLOCK_2026-10-03.md`. Verify the expected design blob. The branch may gain checkpoint-only commits; the design blob is the semantic target. If it differs, do not review a guessed/latest candidate: return BLOCK/PARTIAL and identify the mismatch.

The original request is archived at `cockpit/reviews/archive/D1_1_PRECODE_REVIEW_REQUEST_2026-10-02.md`. Reuse the supplied complete original coverage; do not repeat its seven-question full review. Read archived request only if an affected contradiction requires it. The writer's corrections are not independent PASS evidence.

## Finite review questions and permitted evidence

1. **PRE-01 closure:** Does design §5.1 make newly ingested effective-`other` admission snapshot-only through a bounded internal default-off mode? Is only the validated complete snapshot caller allowed to opt in? Does legacy import preserve unrelated discard/removal/counters and relevant behavior without phantom global retention? Are future regressions in §§10/12 explicit?
2. **PRE-02 closure:** Does §5.2 enforce empty phone/null patient link whenever any affected writer persists effective `other`, including clearing a manual override before commit/response? Does it avoid silent restoration of cleared fields? Is the auto-other/manual-relevant/populated-fields → clear-override regression explicit?
3. **Affected cumulative check:** Do those corrections preserve a single normalized store/owner, same-source exact-window missing-row reconciliation, existing manual relevant overrides, protected access, weekly osteoporosis filtering and global projection minimization? Does the design still preserve cross-day Next and lawful Aclasta/other-overlap fail-closed semantics without expanding clinical linkage, Reception, Visit Brief, D2 or GESY scope?

Primary source allow-list after authority/design/result reads:
- `clinical_calendar.py`: `RELEVANT_CATEGORIES`, `list_appointments`, `update_appointment_classification`, `_apply_import_item`, `import_appointments`, `import_appointment_snapshot` and directly used contracts/helpers.
- `test_clinical_calendar_snapshot.py`: the two named old snapshot oracles, manual override test and legacy import test.
- `static/cockpit/app.js` / `test_cockpit_home.py`: only already accepted D1 temporal/minimization consumers if needed to resolve question 3.

Review the design delta against the original blob, then inspect the affected source once. Main/runtime bytes were unchanged at the coordinator's fresh main `5801fbe20cf760eaa9c67e8aa03bc52c5928393b`; check for relevant drift, not unrelated branch history. Follow another file only for a named caller, writer or contradiction that changes one of Q1–Q3; record which question it resolves. If required evidence is unavailable after a targeted retry, mark that question unresolved. No recursive registry/historical review chase, repo-wide audit, web search, dependency installation, full test suite, live patient/portal access, CI/deploy polling or additional reviewers.

This is pre-code closure: source and required future regression oracles are evidence. Do not BLOCK merely because the intended runtime fix/test has not yet been implemented. A reachable design gap or safety/authority contradiction is material; cosmetic alternatives are not.

## Terminal conditions

Maintain a disposition for Q1/Q2/Q3 and PRE-01/PRE-02. Stop immediately when all are disposed; do not fill the budget. If a decisive material BLOCK is established, record it and complete only the bounded affected scan if time permits. Target mismatch, unavailable required evidence or a material scope expansion → BLOCK/PARTIAL/implementation NO with the exact remaining question/evidence; do not investigate a newly expanded programme inside this closure. No automatic restart, extension or “review of the review”.

PASS requires both findings independently CLOSED, complete affected coverage and no new material risk. On PASS, stop this pre-code chain under P5; any later post-code fidelity gate is separate. Return handback and STOP. No implementation or follow-up after output.

The finite evidence map in Q1–Q3 is the stop criterion. Do not search for proof that no other defect exists. `NO ADDITIONAL MATERIAL FINDING=YES` refers only to the declared affected scope.

## Required output

```text
TARGET DESIGN BLOB = <verified blob | unverified>
VERDICT = PASS | BLOCK
COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL
ORIGINAL FINDINGS = D1.1-PRE-01: CLOSED | OPEN | UNVERIFIED; D1.1-PRE-02: CLOSED | OPEN | UNVERIFIED
QUESTION DISPOSITIONS = Q1: <disposition + concise evidence>; Q2: <...>; Q3: <...>
P0:P1:P2 = x:y:z
MATERIAL FINDINGS =
- <id / severity / reachable consequence / violated invariant / evidence / smallest correction> | NONE
NO ADDITIONAL MATERIAL FINDING = YES | NO | NOT_ESTABLISHED
UNREVIEWED / MISSING EVIDENCE = <precise remainder | NONE>
STOP REASON = <Q1–Q3 disposed | decisive material block | unavailable evidence | target mismatch | material scope expansion>
IMPLEMENTATION MAY START = YES | NO
```

For PARTIAL, report known findings only; zero known findings is not PASS. Set NO ADDITIONAL MATERIAL FINDING=NOT_ESTABLISHED when required coverage is incomplete. Use YES for implementation only on complete PASS. STOP.
