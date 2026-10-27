# SLICE_PLAN_CURRENT.md — web visit audio inside the selected patient chart

> **STATUS (2026-10-27): DESIGN CORRECTION / R2 CLOSURE HOLD.** The Product Owner explicitly replaced the local-only direction with server-side OpenAI processing and chose verified OpenAI Zero Data Retention as the condition for real patient audio. The existing local Priority 4 release remains LIVE and available for rollback. Its approved historical slice is archived at `cockpit/archive/SLICE_PLAN_PRIORITY4_LOCAL_VISIT_AUDIO_2026-10-27.md`.

## Objective

Add direct clinician Start/Stop and automatic transcript-to-editable-draft processing inside the existing selected patient chart. Reuse the authoritative Visit Capture context, strict `VisitCaptureCandidateV1`, protected Preview and one explicit clinician Save. The detailed new design and test/release contract is `cockpit/WEB_VISIT_AUDIO_2026-10-27.md`.

## New boundary

The browser sends a bounded recording to the protected Render service. Server-only providers call OpenAI `gpt-transcribe` and Responses `gpt-6-luna` Structured Outputs. Audio/transcript are temporary and absent from records/logs. `OPENAI_API_KEY` stays in Render only. A purpose-specific provider gate remains disabled for real audio until the exact OpenAI project/key is verified as ZDR.

## Safety invariants

- Recording only after explicit clinician Start and patient-informed acknowledgement; no background capture.
- Selected patient/context remains authoritative; no name-based automatic linkage, no stale-context adoption.
- Model output remains a non-authoritative draft; malformed, incomplete, conflicting or uncertain identity fails closed.
- No automatic clinical write. Existing context-consuming Save provides duplicate-write protection.
- Synthetic data only in tests and release smoke; no provider content or PHI in logs.

## Gates and next action

First pre-code R2 BLOCK 0:3:1; first closure BLOCK 0:1:0 on typed semantic promotion. Corrected design holds these units provisional until explicit clinician type/status/completion selection. One focused closure review is required. After PASS, implement only this bounded seam, run focused synthetic tests, obtain one post-code exact-head R2, then PR/CI/Render checks and no-write production smoke. Real patient first use requires verified ZDR and clinician acceptance.
