# SLICE_PLAN_CURRENT.md — no-loss web Visit Audio recovery

> **STATUS (2026-10-28): PRE-CODE R2 P1 CORRECTION / CLOSURE HOLD.** Independent review of `94385070` blocked 0:1:0 because client-supplied context/date/transcript could not prove the original recording binding. The corrected design adds a short-lived dedicated-secret HMAC binding over original context/session, patient/version, encounter date and transcript digest; reprocess verifies every claim before provider use and allows only a legitimate same-patient context renewal chain. Identity conflict/uncertain is explicitly non-retryable. Cross-chart/date/transcript/token substitution oracles are required. Focused independent closure is next; PR #247 remains unmerged and runtime unchanged.

## Objective

Prevent loss of a recorded visit when transcription succeeds but structured processing fails, while preserving patient binding, ephemeral retention, clinician review and explicit Save authority.

## Frozen behavior

- Existing protected Start/Stop, informed-recording acknowledgement, OpenAI standard-retention disclosure, project/org headers, `store:false`, exact-source candidate, typed suggestions, display-only contextual considerations, one final patient/content confirmation, Preview and explicit Save remain.
- Audio/transcript remain transient and are never placed in server storage, database, browser persistence or logs.
- Transcript is non-authoritative and never enters Preview or Save.
- Unresolved patient identity still fails closed and cannot be corrected by structural retry.
- No real-patient/provider replay, merge or deploy belongs to implementation evidence.

## New contract

1. Audio route returns `processing_status=complete|transcript_only` after successful transcription.
2. Both states return the protected, unverified transcript; only `complete` returns a validated candidate.
3. Transcript-only failures carry fixed safe stage/code metadata and produce content-free server diagnostics.
4. A protected reprocess route submits the retained transcript to the same classification/candidate pipeline without retranscribing.
5. Browser keeps the original Blob only in memory until successful structuring or explicit discard; failure exposes retry and local audio download.
6. UI has two explicit views: unverified transcript and processed clinical text. Transcript-only state cannot Preview or Save.
7. Every reprocess requires a dedicated-secret signed binding for the original session/patient version/date/transcript digest; client values alone grant no authority.
8. Identity conflict/uncertain remains transcript-recoverable but cannot be reprocessed.

## Evidence questions

- Does every post-transcription failure preserve the exact transcript without leaking content to logs or creating clinical authority?
- Can the same protected context/date/transcript be reprocessed without transcription and transition to a valid candidate?
- Are Blob/transcript lifecycles transient, chart-bound and cleared on discard/success as designed?
- Are transcript and contextual inference absent from Preview/Save, with the existing single final confirmation preserved?
- Do fixed logs distinguish the future validator/provider/timeout class without arbitrary exception text?
- Does every cross-chart/date/transcript/token substitution fail before a provider call while a legitimate same-patient context renewal remains usable?

## Review and release

R2 pre-code review, bounded implementation/tests, exact-code R2, exact-head CI and a separate release decision. This critical safety/data-integrity slice supersedes release of the already-reviewed retry-only head until closed.
