# Priority 4 local visit audio — focused POST-CODE R2 P1-01 closure request

**Date:** 2026-10-26 Asia/Nicosia.  
**Mode:** one independent read-only **delta + affected-cumulative** closure review under `PROCEDURES.md` P5; stop after disposition. No edits, installs, patient/data access, PR, merge, deploy or external write.

## Exact targets

- Original reviewed implementation target: `dc5894ec80d29d4a3a6c2f06dba598be931af032`
- Original BLOCK receipt: `cockpit/reviews/LOCAL_VISIT_AUDIO_PRIORITY4_POSTCODE_R2_BLOCK_RECEIPT_2026-10-26.md`
- Corrected code/test/CI target: `b6fd2e3f2d8547bc57a33c917ec7c342bddc614a`
- Correction delta: `35c1d48..b6fd2e3` (the intervening commit is the BLOCK checkpoint; substantive correction is the final commit)
- Frozen design remains `9027d07152729d83972ca725feb533c0e987205d`, blob `644c36a8b557be46e4165a6cce9c63fa4ed54290`
- Fresh remote main at author freeze: `889cc14b0efb607b553db4f2735a3c3bac9dae34`

Exact corrected blobs:

- `local_visit_audio/launcher.js` — `a08c8fd55cc29bf8a215bbaac2088d9b40b19d31`
- `test_local_audio_launcher_lifecycle.js` — `618668bf1e2b2f49a1ddef52dc84e951fb8be8c5`
- `.github/workflows/visit-capture-tests.yml` — `77c3a17a7ddc4a758c907ef56136a7aeab738e96`

No other runtime/test blob changed from the original R2 target.

## Original P1-01 and correction

Original reachable failure: closing/refreshing the launcher, a port message error or an active protocol rejection aborted only the browser fetch. It did not invoke the existing same-session/request `/api/cancel`, allowing decoder/Whisper/Ollama and temporary PHI to remain until completion or the ten-minute timeout.

Correction:

- `closeChannel(message, cancelActive=true)` synchronously initiates the existing keepalive `/api/cancel` while exact active request/session state still exists, then aborts the process fetch and only then clears state;
- pagehide, `port.onmessageerror`, active protocol rejection and child `job_cancel` use this cancellation path;
- a network failure before any server terminal response also asks the loopback service to cancel;
- ordinary validated result/error acknowledgement and post-terminal acknowledgement timeouts pass `cancelActive=false`, avoiding redundant cancellation after backend completion;
- the new executable launcher lifecycle harness proves `cancel` initiation precedes abort/state clear for pagehide, port-message error and active protocol rejection, asserts exact session/request headers and no result, and proves normal result acknowledgement sends no cancel;
- existing `test_loopback_cancel_is_same_session_and_cleans_before_terminal_response` still proves the corresponding backend cancellation event, no result, recovery and temporary-directory removal.

## Focused evidence

- `node --check` launcher and new lifecycle harness PASS.
- `node test_local_audio_launcher_lifecycle.js` PASS.
- Existing no-patient and local reviewed-flow Node smokes PASS.
- `47 passed` affected Python suite, including direct loopback cancellation/cleanup and total processing deadline.
- `37 passed` shared Visit Capture/CP regression suite.
- `git diff --check` PASS.

## Finite closure questions

1. Is original P1-01 closed for pagehide, port error, active protocol failure, child cancel and pre-terminal network failure while preserving exact request/session binding?
2. Is cancel initiated before abort/active-state clearing, so `/api/cancel` retains the required headers and can reach the active server job?
3. Do normal post-result/post-error acknowledgements avoid unnecessary cancellation?
4. Combined with the unchanged blocking-service test, does evidence prove backend cancel, subprocess interruption path, no result and verified temporary cleanup?
5. Does the three-file correction introduce any new material risk in F1/F2/F7/F8 or disturb original F3–F6 PASS?

## Required handback

Return `PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, exact `P0:P1:P2`, each question's disposition, original P1-01 `CLOSED | OPEN`, any new material finding with smallest correction, exact reviewed targets/blobs, and `STOP READ-ONLY`.

A PASS closes implementation fidelity only. It does not authorize model installation, real patient use, PR/merge/deploy or clinical writes.
