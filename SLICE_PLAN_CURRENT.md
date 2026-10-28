# SLICE_PLAN_CURRENT.md — web visit audio inside the selected patient chart

> **STATUS (2026-10-28): FICTIONAL PROVIDER PATH WORKED / COMPACT REVIEW CORRECTION ACTIVE.** The Product Owner completed a fictional Start/Stop trial and received a draft, proving authenticated capability and provider routing for that synthetic attempt. The review was not usable: provisional source units felt like nine separate confirmations and the whole internal schema appeared as a very long form. Bounded correction `cockpit/WEB_VISIT_AUDIO_COMPACT_REVIEW_2026-10-28.md` keeps the same candidate, uncertainty, Preview and explicit Save authorities while showing only populated grouped fields and one final confirmation. No real-patient readiness or clinical-write acceptance is established. The local Priority 4 path remains available. Historical slice: `cockpit/archive/SLICE_PLAN_PRIORITY4_LOCAL_VISIT_AUDIO_2026-10-27.md`.

## Objective

Add direct clinician Start/Stop and automatic transcript-to-editable-draft processing inside the existing selected patient chart. Reuse the authoritative Visit Capture context, strict `VisitCaptureCandidateV1`, protected Preview and one explicit clinician Save. The detailed new design and test/release contract is `cockpit/WEB_VISIT_AUDIO_2026-10-27.md`.

## New boundary

The browser sends a bounded recording to the protected Render service. Server-only providers call OpenAI `gpt-transcribe` and Responses `gpt-6-luna` Structured Outputs. Audio/transcript are temporary and absent from records/logs. `OPENAI_API_KEY` stays in Render only. A purpose-specific provider gate remains disabled until the standard-retention notice/acknowledgement change is reviewed, implemented, released and configured. The existing `store:false` stays in place; it does not remove possible Responses abuse-monitoring retention.

## Safety invariants

- Recording only after explicit clinician Start and patient-informed acknowledgement; no background capture.
- Selected patient/context remains authoritative; no name-based automatic linkage, no stale-context adoption.
- Model output remains a non-authoritative draft; malformed, incomplete, conflicting or uncertain identity fails closed.
- No automatic clinical write. Existing context-consuming Save provides duplicate-write protection.
- Synthetic data only in tests and release smoke; no provider content or PHI in logs.

## Gates and next action

Final pre-code R2 closure PASS 0:0:0 on `69427a3`; prior findings closed. Runtime and synthetic test head `04b0172` passed 26 affected Python/browser tests, `node --check` and `git diff --check`. Independent post-code R2 on `eefc6c0` returned PASS / complete for affected scope / 0:0:0; the inherited expired-prior CAS behavior was confirmed and the active upload distinction clarified in the design. PR #232 exact final head `1d8a8b0` passed 17/17, merged as `667d76f3d1c3b636d65a869df85d02dff10fe800` and reached LIVE through native Render `dep-dbg4m68473hc73fcj4k0`. No-write asset/auth/log smoke passed; release receipt is in `cockpit/releases/COCKPIT_WEB_VISIT_AUDIO_PRODUCTION_RELEASE_2026-10-27.md`. The Product Owner subsequently selected standard OpenAI retention instead of waiting for ZDR approval. The bounded addendum `cockpit/WEB_VISIT_AUDIO_STANDARD_RETENTION_2026-10-27.md` passed corrected independent pre/post-code R2 0:0:0; PR #235 final head passed 14/14 and merged as `bb33258ea31a961560c3c34c2d93d7d50fbbcbe3`. One manual code deploy was needed because `[skip render]` in the squash body caused Render to ignore the auto-deploy; `dep-dbgelj0n74is739bjjog` became LIVE. The three dedicated standard-retention settings were then merged into Render without changing the existing masked key, and config deploy `dep-dbgen2gn74is739bon1g` became LIVE at 18:23:03Z. Both releases passed no-PHI static/auth/log smoke. The explicit `OpenAI-Project` header fails closed if the user-supplied project is not authorized by the key. Next: clinician first informed-patient recording, review and explicit Save; authenticated capability, provider/key-project routing and clinical quality have not been independently exercised. No real patient audio/provider call or clinical write was performed by the agent. [Release receipt](cockpit/releases/COCKPIT_WEB_VISIT_AUDIO_STANDARD_RETENTION_RELEASE_2026-10-27.md).
