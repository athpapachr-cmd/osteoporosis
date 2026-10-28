# SLICE_PLAN_CURRENT.md — native local audio recorded-patient confirmation guard

> **STATUS (2026-10-28): DESIGN R2 CLOSED / IMPLEMENTATION NEXT.** The Product Owner selected local native processing on the Mac. This bounded first server prerequisite extends the protected Visit Capture patient-confirmation gate from `web_visit_audio` to both audio source roles. The separate native design correction passed independent focused pre-code R2 closure on `51d3a9d0` with P0:P1:P2=0:0:0 after an initial 0:2:0 BLOCK. Real-chart native capture, local model qualification and clinician-voice accuracy remain HOLD. The released web compact-review slice is archived at `cockpit/archive/SLICE_PLAN_WEB_VISIT_AUDIO_2026-10-28.md` and its Product Owner fictional UX recheck remains pending.

## Product Owner step card

- **Problem today:** server Preview/Save requires `recording_patient_confirmed` when a candidate cites `web_visit_audio`, but currently not when it cites the already allowed `local_visit_audio` role. The current local Visit Capture page displays an identity checkbox but omits that flag from both protected requests, so a server-only guard would also break its already released local flow.
- **After this step:** either audio role requires the same explicit recorded-patient confirmation in protected Preview and Save. The existing local page sends the flag only while its explicit identity checkbox remains checked, and withdrawal invalidates its visible save readiness. A non-audio Visit Capture is unchanged.
- **Example:** a synthetic local-audio candidate with no confirmation receives `can_save=false` in Preview and 409 at Save; with confirmation and the same current context, it may proceed through the existing validation and one-shot Save.
- **Outside this step:** no recording, transcription, model, native app, context renewal, schema, database, Render configuration, real patient or provider call changes. The existing local page's request payload/withdrawal state change is the only browser delta.
- **Why now / next milestone:** close the server authority gap before the native bridge can be implemented. After focused synthetic tests and independent exact-code R2, guarded PR/CI/release follow. Native real-chart recording stays disabled until its separate gates close.

The Product Owner's explicit local-native choice in the active chat is the direction for this prerequisite. It does not waive clinical review or release gates.

## Frozen contract

1. A candidate is recorded audio when **any** source binding has role `web_visit_audio` or `local_visit_audio`. Other source roles retain existing behavior.
2. Protected Preview returns `can_save=false` and the existing identity-confirmation blocking reason when recorded audio is present and `recording_patient_confirmed` is false. The request and candidate are still schema validated under the existing context authority.
3. Protected Save returns 409 for the same false state before consuming context. With true confirmation, existing identity, context, validation and one-shot save rules remain in force. Confirmation alone never overrides an uncertain/conflicting source identity or stale context.
4. A candidate with both audio roles remains guarded. No new endpoint, data field, permission, log content or alternate write is introduced.
5. The existing local Visit Capture page sends `recording_patient_confirmed=true` only after its `localIdentityConfirmed` control is checked and the exact current local session/result/context remain active. Preview and Save each send the current checkbox value; unchecking it clears visible save readiness so stale confirmation cannot be reused. Ordinary Dia/manual and web chart request paths are not changed.

## Evidence, review and rollback

Focused synthetic tests cover local false/true Preview and Save, preserved context on false Save, web role regression, non-audio behavior, uncertain identity, stale context and duplicate Save. The existing local browser oracle must show identity checkbox false before review; true in both protected Preview and Save bodies after explicit check; and no stale Save after withdrawal. Reuse existing context/CAS tests where same bytes and question are already covered. No real patient or provider request. R2 because this is a protected clinical-write guard: accepted pre-code design closure covers the server semantics; the newly identified local-page binding requires focused pre-code delta review and one independent exact-code post-code review. Exact-head CI, merge, Render observation and no-PHI smoke are separate release gates. Rollback is a guarded revert of this additive check if a source-proven regression appears; real native capture remains off regardless.

## Next action

Close the local-page binding design delta, then implement and verify the two-role server guard and exact local Preview/Save checkbox binding in `static/cockpit/visit-capture/app.js`, followed by one independent post-code R2. No native real-chart capture or clinical write.
