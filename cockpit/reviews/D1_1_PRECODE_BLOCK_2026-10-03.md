# D1.1 pre-code BLOCK — supplied result and bounded correction

> DATE: 2026-10-03 Asia/Nicosia.
> SOURCE: Product Owner handback in conversation `6abebc7c-018c-83eb-bd23-273f093e2b44`.
> REQUESTED TARGET: branch `design/cockpit-d1-1-global-context-2026-10-02`, head `9b11f722c485247754fa7219e9b0f0d9167f8734`, design blob `2b438c949e30d00a1d11d9d1431d766954320f15`.
> PROVENANCE LIMIT: reviewer execution trace/exact-head attestation was not supplied. The supplied result is consumed; both cited source paths were checked against the unchanged runtime at fresh main `5801fbe20cf760eaa9c67e8aa03bc52c5928393b`.
> IMPLEMENTATION: NO. INDEPENDENT CLOSURE: PENDING.

## Supplied result

```text
VERDICT = BLOCK
COVERAGE = COMPLETE_FOR_DECLARED_SCOPE
P0:P1:P2 = 0:2:0
MATERIAL FINDINGS =
- D1.1-PRE-01 / P1 / An unrelated row submitted through the still-reachable legacy POST /clinical/calendar/appointments/import can become retained global appointment context without complete-snapshot reconciliation, allowing stale/phantom Previous/Current/Next state / violates the single-owner reuse boundary and the design requirement that global retention is derived from the existing complete Digital Secretary snapshot / current clinical_calendar.py routes both legacy import and complete snapshot through _apply_import_item; the design proposes changing that helper from discard unrelated to retain as other, while the legacy import has no missing-row reconciliation / explicitly make other retention snapshot-only (for example a bounded helper mode/flag), preserve unrelated-row discard for the legacy single-import path, and add a regression proving it
- D1.1-PRE-02 / P1 / Clearing a manual osteoporosis classification can leave an effective other row carrying previously retained phone_e164 and linked_patient_id, violating D1.1 server-side minimization even though the new browser projection excludes those fields / violates the invariant that every effective other row persists phone empty and linked patient null / current update_appointment_classification(... category=None) recomputes row.category but does not clear phone/link; this path does not pass through the proposed snapshot sanitization / explicitly require minimization whenever any mutation path makes the effective category other, including manual-override clearing, and add a focused regression for that transition
NO ADDITIONAL MATERIAL FINDING = YES
IMPLEMENTATION MAY START = NO
```

Formatting/backslash separators were normalized above; finding substance is unchanged. This is the user's independent handback, not a new author review.

## Disposition and source check

| Finding | Source confirmation | Design correction | Status |
|---|---|---|---|
| PRE-01 | `clinical_calendar.py`: shared helper at 389, legacy caller 470, snapshot caller 542; missing-row reconciliation only in snapshot. Existing legacy test at `test_clinical_calendar_snapshot.py:429` covers relevant import only. | Design §§5.1, 10, 12: internal default-off retention mode; complete snapshot caller alone opts in; preserve legacy effective-unrelated skip/removal + relevant imports; future regression required. | CURRENT BLOCKER / in-slice REPLAN; correction drafted, independent closure pending. |
| PRE-02 | `clinical_calendar.py:356–373`: manual-clear path recomputes category and commits without clearing phone/link. Existing manual-clear test starts auto-osteoporosis, not auto-other. | Design §§5.2, 10, 12: every effective-other commit must persist empty/null; clear manual override in the same transaction; no phone/link resurrection on manual promotion; future regression required. | CURRENT BLOCKER / in-slice REPLAN; correction drafted, independent closure pending. |

No runtime/test bytes changed. No test execution or post-code PASS is claimed. Original complete coverage/no-additional-finding statement belongs to the supplied review; the writer does not certify independent closure.

## Review-duration investigation

Observed: Product Owner reports a reviewer continued for over seven hours and returned the result after interruption. Exact original chat prompt is preserved in `D1_1_REVIEW_EXECUTION_NOTES.md`; the original request is archived byte-for-byte in `archive/D1_1_PRECODE_REVIEW_REQUEST_2026-10-02.md`.

Source-proven gap: the original prompt/request requires full bootstrap and seven broad questions, then COMPLETE coverage and NO ADDITIONAL MATERIAL FINDING, without defining sufficient evidence for each or when the reviewer must stop looking for counterexamples. The canonicals and repository references can become recursive reading targets. `PROCEDURES.md` P5 stops a review *chain* but did not define evidence closure within one review.

Inference: this gap can motivate recursive navigation and reassurance searches. The original request already said READ-ONLY and STOP after the result, which does not define when the result is ready. The Work-mode chat exposes no intermediate trace. A separate Codex execution of the same prompt completed in about four minutes, so neither prompt-induced wandering nor a tool/platform stall is established as the specific seven-hour cause.

Correction: `PROCEDURES.md` P5.1 requires a question→evidence map, bootstrap once, source expansion only for a concrete caller/writer/contradiction, explicit meaning of COMPLETE and NO ADDITIONAL MATERIAL FINDING, and immediate exit after the decision is supported. The revised D1.1 request is one delta + affected cumulative closure of PRE-01/PRE-02. It reuses the original review's declared coverage without requesting a fresh search through unrelated architecture.

A timeout may protect an unattended runner from hangs, but it is separate from the causal prompt correction. No timeout configuration is claimed.

## Plan impact and next action

No new task/phase/owner is needed. Root PR-1 slice/phase/roadmap remain unchanged; this is a local D1.1 design correction. Floating Visit Brief follows D1.1; GESY and D2 communication remain deferred. Review method owns the general limits; current files point to it rather than duplicate it.

Next lawful action: one fresh independent READ-ONLY pre-code closure on the corrected design blob/request. No review is launched by this correction session. Only independent PASS with complete affected coverage may release the already authorized D1.1 implementation step. PASS is not release/deploy authority; after closure, STOP this pre-code review chain.
