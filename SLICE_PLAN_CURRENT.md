# SLICE_PLAN_CURRENT.md — native local audio recorded-patient confirmation guard

> **STATUS (2026-10-28): PR #242 MERGED / EXACT RENDER DEPLOY OBSERVATION NEXT.** The Product Owner selected local native processing on the Mac. This bounded prerequisite extends protected Visit Capture patient confirmation from `web_visit_audio` to both audio roles and binds the already visible local checkbox to Preview/Save. The native design correction passed focused independent pre-code R2 closure on `51d3a9d0` (0:0:0 after 0:2:0 BLOCK); the discovered existing-page binding delta at `fa31744e` passed focused independent pre-code R2 (0:0:0). The first post-code R2 on `1edf8bda` found one P2: Save could appear enabled after checkbox withdrawal during awaited context renewal. The correction at `70aac6cf` passed independent focused finding/delta/affected-cumulative closure (0:0:0): off-only withdrawal keeps Save disabled, off/on requires a fresh Preview, auth resets remain, and manual catch behavior is unchanged. Four targeted Python tests, the Node UI oracle, JS syntax and diff hygiene PASS. [PR #242](https://github.com/athpapachr-cmd/osteoporosis/pull/242) final head `ac76c106` passed 14/14 CI and squash-merged as `c090846a8258f2b4beb5a77e5c10d97b9ae7bfd3`; Render deploy status and no-PHI smoke remain pending. Real-chart native capture, model qualification and clinician-voice accuracy remain HOLD. The released web compact-review slice is archived at `cockpit/archive/SLICE_PLAN_WEB_VISIT_AUDIO_2026-10-28.md` and its Product Owner fictional UX recheck remains pending.

## Product Owner step card

- **Problem addressed:** the prior live server Preview/Save required `recording_patient_confirmed` for `web_visit_audio` but not the already allowed `local_visit_audio` role. The prior local Visit Capture page displayed an identity checkbox but omitted that flag from both protected requests, so a server-only guard would have broken its released local flow.
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

Publish the bounded canonical-impact PR, verify exact-head CI and fresh release gate, then observe native Render deploy and perform no-PHI smoke if PASS. No native real-chart capture or clinical write.
