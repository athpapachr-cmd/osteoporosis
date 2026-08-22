(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const MODULE_KEYS = ["step3", "step4", "longitudinal_review", "step5", "step6", "audit_evaluation_v1"];

  function getCases() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }
    catch { return []; }
  }

  function setCases(cases) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cases));
  }

  function activeId() {
    return localStorage.getItem(ACTIVE_KEY) || "";
  }

  function captureModuleState() {
    const id = activeId();
    const current = getCases().find((c) => c.internal_uuid === id);
    if (!current) return null;
    const preserved = {};
    MODULE_KEYS.forEach((key) => {
      if (Object.prototype.hasOwnProperty.call(current, key)) preserved[key] = current[key];
    });
    return { id, preserved };
  }

  function restoreModuleState(snapshot) {
    if (!snapshot?.id) return;
    const cases = getCases();
    const index = cases.findIndex((c) => c.internal_uuid === snapshot.id);
    if (index < 0) return;
    cases[index] = { ...cases[index], ...snapshot.preserved };
    setCases(cases);
  }

  function preserveAroundLegacySave(button) {
    button.addEventListener("click", () => {
      const snapshot = captureModuleState();
      setTimeout(() => restoreModuleState(snapshot), 0);
    }, true);
  }

  function markPilotComplete() {
    const id = activeId();
    if (!id) return;
    const cases = getCases();
    const index = cases.findIndex((c) => c.internal_uuid === id);
    if (index < 0) return;
    const step6 = cases[index].step6 || {};
    cases[index].pilot_completion = {
      status: "complete",
      completed_at: new Date().toISOString(),
      ready_for_audit_at_completion: step6.capture_quality?.ready_for_audit || "",
      completion_time_minutes: step6.capture_quality?.completion_time_minutes ?? null
    };
    cases[index].implementation_slice = "steps_1_6_pilot_complete";
    setCases(cases);
    const status = document.querySelector("#draftStatus");
    if (status) status.textContent = "Pilot case ολοκληρώθηκε και αποθηκεύτηκε τοπικά";
    const pill = document.querySelector("#pilotPill");
    if (pill) {
      const completed = cases.filter((c) => c.baseline_phase === "pilot" && c.pilot_completion?.status === "complete").length;
      pill.textContent = `PILOT · ${Math.min(completed, 5)}/5 COMPLETE`;
    }
  }

  function isStep6Active() {
    const panel = document.querySelector('[data-step-panel="6"]');
    return panel && !panel.hidden && getComputedStyle(panel).display !== "none";
  }

  const saveTop = document.querySelector("#saveTopBtn");
  const saveDraft = document.querySelector("#saveDraftBtn");
  if (saveTop) preserveAroundLegacySave(saveTop);
  if (saveDraft) preserveAroundLegacySave(saveDraft);

  const finish = document.querySelector("#finishVisitBtn");
  if (finish) {
    finish.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopImmediatePropagation();
      const saver = saveDraft || saveTop;
      if (saver) saver.click();
      setTimeout(() => {
        markPilotComplete();
        window.alert("Το pilot case αποθηκεύτηκε ως complete. Τα KPI παραμένουν κρυφά μέχρι το baseline lock.");
      }, 20);
    }, true);
  }

  const next = document.querySelector("#nextBtn");
  if (next) {
    next.addEventListener("click", (event) => {
      if (!isStep6Active()) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      finish?.click();
    }, true);
  }

  // Refresh completion count without exposing any KPI result.
  const cases = getCases();
  const completed = cases.filter((c) => c.baseline_phase === "pilot" && c.pilot_completion?.status === "complete").length;
  if (completed > 0) {
    const pill = document.querySelector("#pilotPill");
    if (pill) pill.textContent = `PILOT · ${Math.min(completed, 5)}/5 COMPLETE`;
  }
})();
