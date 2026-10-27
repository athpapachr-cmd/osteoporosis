# SLICE_PLAN_CURRENT.md — web visit audio inside the selected patient chart

> **STATUS (2026-10-27): CODE LIVE / NO-WRITE SMOKE PASS / REAL-USE ZDR GATE PENDING.** The Product Owner explicitly replaced the local-only direction with server-side OpenAI processing and chose verified OpenAI Zero Data Retention as the condition for real patient audio. The existing local Priority 4 path remains available. Its approved historical slice is archived at `cockpit/archive/SLICE_PLAN_PRIORITY4_LOCAL_VISIT_AUDIO_2026-10-27.md`.

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

Final pre-code R2 closure PASS 0:0:0 on `69427a3`; prior findings closed. Runtime and synthetic test head `04b0172` passed 26 affected Python/browser tests, `node --check` and `git diff --check`. Independent post-code R2 on `eefc6c0` returned PASS / complete for affected scope / 0:0:0; the inherited expired-prior CAS behavior was confirmed and the active upload distinction clarified in the design. PR #232 exact final head `1d8a8b0` passed 17/17, merged as `667d76f3d1c3b636d65a869df85d02dff10fe800` and reached LIVE through native Render `dep-dbg4m68473hc73fcj4k0`. No-write asset/auth/log smoke passed; release receipt is in `cockpit/releases/COCKPIT_WEB_VISIT_AUDIO_PRODUCTION_RELEASE_2026-10-27.md`. Next verify exact OpenAI project ZDR proof, then enable the purpose gate and perform clinician first-use acceptance. No real patient processing is authorized until then.
