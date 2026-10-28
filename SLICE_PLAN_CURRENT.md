> **2026-10-28 COMPLETE CLINICAL SUMMARY — R1 FINDING CORRECTED / CLOSURE NEXT:** R1 found omitted populated coding; common renderer now shows exact system/code/title and explicit completeness state. Regression covers no coding/incomplete/complete in draft, latest card and reader. 62 affected tests PASS; focused CP2+three complete-case oracles PASS 4/4 after preserving the pre-result disabled chart controls (hide prior chart only when result is ready). Actual provider and saved synthetic chart proofs already confirm all four requested clinical facts. PR #252 remains draft, corrected exact-head closure/CI next, then release; verification branch never merges. No production clinical Save.

> **2026-10-28 DIRECT AUDIO SUMMARY — LIVE / SMOKE PASS / CLOSED:** PR #250 merge `796b11cdb02884deea4a35463c69eff94458cfc2` is LIVE on Render `dep-dbh29tfb9u1s73e71900` at 16:40:12Z. Stop shows validated summary automatically, transcript/editor stay optional, no identity checkbox; explicit Save previews current bytes and writes once. Paste unlocks and failed manual replacement preserves prior transcript recovery or completed summary. Prompt clarifies generic demographics and presenting complaint without overriding conflict verdicts. Independent R2 PASS 0:0:0; 47 affected local tests and all 15 PR checks PASS. Two served assets matched merge SHA-256, three anonymous audio routes returned 401, release-window ERROR/5xx empty. No agent provider/clinical write. Writer released after receipt publication; clinician fictional-quality check remains. Supersedes older checkbox-only instructions below.

# SLICE_PLAN_CURRENT.md — no-loss web Visit Audio recovery

> **STATUS (2026-10-28): LIVE / NO-PHI SMOKE PASS / RELEASE CLOSED.** [PR #247](https://github.com/athpapachr-cmd/osteoporosis/pull/247) merged the reviewed no-loss transcript/audio recovery, contextual clinical considerations and default-collapsed transcript/processed views as `ef4d633a`. Its first deploy failed safely before cutover because `cockpit_app` correctly lacked schema CREATE; prior production stayed LIVE. [PR #248](https://github.com/athpapachr-cmd/osteoporosis/pull/248) moved DDL to the reviewed least-privilege administrator migration and merged as `3500cb49`; Render deploy `dep-dbh0muivcj2c73e00gj0` is LIVE. Three served assets matched exact merge bytes, anonymous capability/draft/reprocess returned 401, and the release window had no app ERROR or HTTP 5xx. No patient rows were read, and no recording, provider request or clinical write occurred. Technical release writer is released; next is one Product Owner fictional recording without Save.

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
