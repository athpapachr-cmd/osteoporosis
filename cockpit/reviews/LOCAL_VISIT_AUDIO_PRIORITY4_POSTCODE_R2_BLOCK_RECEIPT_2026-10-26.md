# Priority 4 local visit audio — independent POST-CODE R2 BLOCK receipt

**Date:** 2026-10-26 Asia/Nicosia.
**Reviewer mode:** independent read-only exact-code R2.
**Exact reviewed implementation target:** `dc5894ec80d29d4a3a6c2f06dba598be931af032`.
**Runtime/test commit:** `a66f3b442b68fd98c42e8d796bbcbcc467ea55b6`.
**Request/head:** `7045e757e16b0d0d8e9d1733f1a695a7034c846e`; drift after `dc5894e` was proven documentation-only.
**Fresh remote main:** `889cc14b0efb607b553db4f2735a3c3bac9dae34`.

## Verdict

`BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:1:0`.

F3–F6 passed. F1/F2/F7/F8 are blocked only by P1-01 below. The reviewer found no additional material issue within the declared scope.

## P1-01 — active launcher closure does not cancel the loopback job

Reachable path: close or refresh the local launcher while decoder, Whisper or Ollama is active. `launcher.js` aborts the browser fetch and closes the MessagePort, but does not call the already implemented same-session/request `/api/cancel`. The loopback handler therefore retains temporary audio/transcript/prompt/model work until natural completion or its ten-minute timeout.

This violates the frozen lifecycle/retention contract requiring launcher close/navigation, port message error and active protocol rejection to abort local subprocesses and run verified cleanup with no candidate or late result.

Smallest correction:

1. while an exact request remains active, launcher pagehide, port-message error and active protocol-failure closure call existing `cancelLocalProcessing()` before aborting the process fetch and clearing state;
2. ordinary post-result/post-error acknowledgement closure does not send an unnecessary cancel;
3. a focused launcher lifecycle test proves cancel issuance before state clearing/abort and complements the existing blocking-service test that proves backend cancellation, temporary-directory removal and no result.

## Stop

No files were edited by the reviewer; no model/dependency installed, patient/database data accessed, PR opened, merge/deploy performed or external state changed. Apply only P1-01, run the focused affected evidence and obtain one independent delta plus affected-cumulative closure review under `PROCEDURES.md` P5.
