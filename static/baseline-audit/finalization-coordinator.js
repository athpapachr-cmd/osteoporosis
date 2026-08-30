(() => {
  "use strict";

  let authoritativeFinishInProgress = false;

  function beginAuthoritativeFinish() {
    if (authoritativeFinishInProgress) return false;
    authoritativeFinishInProgress = true;
    return true;
  }

  function endAuthoritativeFinish() {
    authoritativeFinishInProgress = false;
  }

  function isAuthoritativeFinishInProgress() {
    return authoritativeFinishInProgress;
  }

  function shouldSyncDraftOnSave() {
    return !authoritativeFinishInProgress;
  }

  window.BaselineFinalizationCoordinator = Object.freeze({
    beginAuthoritativeFinish,
    endAuthoritativeFinish,
    isAuthoritativeFinishInProgress,
    shouldSyncDraftOnSave
  });
})();
