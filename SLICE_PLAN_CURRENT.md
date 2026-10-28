# SLICE_PLAN_CURRENT.md — no-loss web Visit Audio recovery

> **STATUS (2026-10-28): PRODUCTION MIGRATION INSTALLED / PR #248 FINAL CI NEXT.** [PR #248](https://github.com/athpapachr-cmd/osteoporosis/pull/248) publishes independently reviewed code/migration/test `32d451fc` plus release checkpoints; R2 is PASS / COMPLETE / 0:0:0. After its prior exact-head checks passed, the reviewed SQL SHA-256 `5b29d4d1d13ead6c7d0dd376a0a622fad5da1f6c668adeba8292a294d8787863` was installed once on Aiven `osteoporosis` as the existing administrator over verified TLS. Preflight proved the table absent; postcheck proved the exact three columns and three indexes, zero rows, app SELECT/INSERT/DELETE plus only `UPDATE(state)`, and no id/expiry UPDATE, TRUNCATE or schema CREATE. This checkpoint now requires fresh exact-head CI, then guarded squash merge and one native Render auto-deploy. No patient rows were read, and no recording, provider request or clinical write occurred; prior production remains LIVE.

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
6. UI has two compact explicit controls: unverified transcript and processed clinical text. Both content views are collapsed by default and appear only after the clinician selects one; selecting the open control again collapses it. Transcript-only state cannot Preview or Save.
7. Every reprocess requires a dedicated-secret signed binding for the exact original context/patient version/date/transcript digest and an active one-time server capability. Client values alone grant no authority; retry never renews or extends the original origin/expiry.
8. Identity conflict/uncertain remains transcript-recoverable but cannot be reprocessed.
9. A server-side chart/context binding failure never returns transcript/token, even after successful transcription; browser Blob recovery remains available.

## Evidence questions

- Does every post-transcription failure preserve the exact transcript without leaking content to logs or creating clinical authority?
- Can the exact protected context/date/transcript be reprocessed without transcription and transition to a valid candidate?
- Are Blob/transcript lifecycles transient, chart-bound and cleared on discard/success as designed?
- Are both review bodies collapsed after every complete or transcript-only response, opening only on explicit selection and closing on a repeated selection?
- Are transcript and contextual inference absent from Preview/Save, with the existing single final confirmation preserved?
- Do fixed logs distinguish the future validator/provider/timeout class without arbitrary exception text?
- Does every context/date/transcript/token substitution fail before a provider call, including a fresh or renewed same-patient context?
- Does context/patient drift after transcription or classification return no transcript/token to the invalid chart while preserving browser-held audio recovery?

## Review and release

The no-loss backend pre-code and exact-code R2 gates are complete. This bounded presentation correction is R1 because it changes only visibility over unchanged transient data and unchanged clinical authority. Implement/test it, obtain one exact-head R1 review, then require fresh PR exact-head CI. Merge and release remain separate decisions.
