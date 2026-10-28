# SLICE_PLAN_CURRENT.md — native local audio recorded-patient confirmation guard

> **STATUS (2026-10-28): DESIGN R2 CLOSED / IMPLEMENTATION NEXT.** The Product Owner selected local native processing on the Mac. This bounded first server prerequisite extends the protected Visit Capture patient-confirmation gate from `web_visit_audio` to both audio source roles. The separate native design correction passed independent focused pre-code R2 closure on `51d3a9d0` with P0:P1:P2=0:0:0 after an initial 0:2:0 BLOCK. Real-chart native capture, local model qualification and clinician-voice accuracy remain HOLD. The released web compact-review slice is archived at `cockpit/archive/SLICE_PLAN_WEB_VISIT_AUDIO_2026-10-28.md` and its Product Owner fictional UX recheck remains pending.

## Product Owner step card

- **Problem today:** server Preview/Save requires `recording_patient_confirmed` when a candidate cites `web_visit_audio`, but currently not when it cites the already allowed `local_visit_audio` role. A future native result could bypass a protection that the visible page asks the clinician to perform.
- **After this step:** either audio role requires the same explicit recorded-patient confirmation in protected Preview and Save. A non-audio Visit Capture is unchanged.
- **Example:** a synthetic local-audio candidate with no confirmation receives `can_save=false` in Preview and 409 at Save; with confirmation and the same current context, it may proceed through the existing validation and one-shot Save.
- **Outside this step:** no recording, transcription, model, browser/native UI, context renewal, schema, database, Render configuration, real patient or provider call changes.
- **Why now / next milestone:** close the server authority gap before the native bridge can be implemented. After focused synthetic tests and independent exact-code R2, guarded PR/CI/release follow. Native real-chart recording stays disabled until its separate gates close.

The Product Owner's explicit local-native choice in the active chat is the direction for this prerequisite. It does not waive clinical review or release gates.

## Frozen contract

1. A candidate is recorded audio when **any** source binding has role `web_visit_audio` or `local_visit_audio`. Other source roles retain existing behavior.
2. Protected Preview returns `can_save=false` and the existing identity-confirmation blocking reason when recorded audio is present and `recording_patient_confirmed` is false. The request and candidate are still schema validated under the existing context authority.
3. Protected Save returns 409 for the same false state before consuming context. With true confirmation, existing identity, context, validation and one-shot save rules remain in force. Confirmation alone never overrides an uncertain/conflicting source identity or stale context.
4. A candidate with both audio roles remains guarded. No new endpoint, data field, permission, log content or alternate write is introduced.

## Evidence, review and rollback

Focused synthetic tests cover local false/true Preview and Save, preserved context on false Save, web role regression, non-audio behavior, uncertain identity, stale context and duplicate Save. Reuse existing context/CAS tests where same bytes and question are already covered. No real patient or provider request. R2 because this is a protected clinical-write guard: accepted pre-code design closure applies to the exact guard semantics; one independent exact-code post-code review is still required. Exact-head CI, merge, Render observation and no-PHI smoke are separate release gates. Rollback is a guarded revert of this additive check if a source-proven regression appears; real native capture remains off regardless.

## Next action

Claim the bounded writer for `clinical_data.py`, focused synthetic tests and linked root/Cockpit checkpoints; implement and verify the two-role guard, then independent post-code R2. No native real-chart capture or clinical write.
