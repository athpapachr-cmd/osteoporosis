# Web Visit Audio — bounded classification retry correction

> **Implementation checkpoint:** Pre-code R2 passed 0:0:0. First post-code review of `f60c0d12` found unresolved identity plus malformed facts could enter retry; corrected code snapshots first identity state and forbids retry for `conflict|uncertain`. Five focused retry/identity/log regressions and 40 affected tests including Chromium pass. Focused closure is pending; PR #247 remains unmerged.

## Observed production failure

Two Product Owner fictional recordings at `2026-10-28T12:33:24Z` and `12:36:15Z` reached transcription but failed at `stage=classification kind=invalid_output`. Shortening the spoken scenario did not change the result. Render logs contain no clinical content and the ephemeral audio/provider response no longer exists, so the exact deterministic validator code is unavailable.

## Bounded correction

Keep the strict source-fidelity validator and all candidate/identity authority unchanged. For each classification batch:

1. run the existing `visit_audio_analysis_v2` request once;
2. validate it with the existing deterministic `checked_analysis` contract;
3. only when validation raises one of the fixed internal structural codes below, run exactly one corrective request over the same batch and selected-chart identity, with that fixed code appended to the developer instruction;
4. validate the retry through the same unchanged validator; any second failure remains a safe 502.

Retryable fixed codes:

```text
classification_keys
classification_identity
classification_values
classification_excerpt
classification_overlap
classification_accounting
classification_suggestions
```

`source_identity_unresolved`, provider HTTP/network/timeout failures, cancellation and unknown exceptions are never retried. A returned `identity_state=uncertain|conflict` therefore continues to fail immediately. The retry remains inside the existing `core_deadline`; it cannot borrow the five-second finalization reserve.

## Diagnostics and privacy

Failure logging may add only `validation_code=<allowlisted fixed code or none>` beside the existing stage/kind/provider-status metadata. It must never log transcript text, fact excerpts, model output, identity, request/response bodies or exception messages.

The retry sends no new category of data: it reuses the same approved OpenAI project/org headers, `gpt-6-luna`, `store:false`, no tools, standard-retention notice and already-sent batch/selected identity. It increases provider calls only on a rejected structural output and never more than once per affected batch.

## Evidence

- First structurally invalid output followed by a valid retry returns the ordinary exact-source candidate and proves exactly two classification calls.
- Repeated structural failure remains 502 and logs only the allowlisted code.
- Identity conflict/uncertainty, provider failure and cancellation do not retry.
- Existing whole-visit contextual consideration, deadline, candidate/Save separation and browser acceptance tests remain green.

R2 pre-code and post-code review are required because this changes clinical classification failure behavior. No real recording/provider replay is part of author evidence.
