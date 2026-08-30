(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";

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

  function setDraftStatus(text) {
    const status = document.querySelector("#draftStatus");
    if (status) status.textContent = text;
  }

  function updatePilotPill() {
    const cases = getCases();
    const completed = cases.filter((c) => c.baseline_phase === "pilot" && c.pilot_completion?.status === "complete").length;
    const pill = document.querySelector("#pilotPill");
    if (pill && completed > 0) pill.textContent = `PILOT · ${Math.min(completed, 5)}/5 COMPLETE`;
  }

  function markPilotComplete() {
    const id = activeId();
    if (!id) throw new Error("Δεν υπάρχει ενεργό pilot case.");
    const cases = getCases();
    const index = cases.findIndex((c) => c.internal_uuid === id);
    if (index < 0) throw new Error("Το ενεργό pilot case δεν βρέθηκε στο local store.");
    const step6 = cases[index].step6 || {};
    cases[index].pilot_completion = {
      status: "complete",
      completed_at: new Date().toISOString(),
      ready_for_audit_at_completion: step6.capture_quality?.ready_for_audit || "",
      completion_time_minutes: step6.capture_quality?.completion_time_minutes ?? null
    };
    cases[index].implementation_slice = "steps_1_6_pilot_complete";
    setCases(cases);
    updatePilotPill();
    return cases[index];
  }

  function isStep6Active() {
    const panel = document.querySelector('[data-step-panel="6"]');
    return panel && !panel.hidden && getComputedStyle(panel).display !== "none";
  }

  function flushLocalPersistence() {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  const saveTop = document.querySelector("#saveTopBtn");
  const saveDraft = document.querySelector("#saveDraftBtn");
  const finish = document.querySelector("#finishVisitBtn");

  async function authoritativeFinish(event) {
    event?.preventDefault?.();
    event?.stopImmediatePropagation?.();

    const coordinator = window.BaselineFinalizationCoordinator;
    const registry = window.ClinicalRegistry;
    if (!coordinator || typeof registry?.finalizeActiveEncounter !== "function") {
      setDraftStatus("Δεν έγινε completion: ο protected server finalization μηχανισμός δεν είναι διαθέσιμος.");
      return;
    }
    if (!coordinator.beginAuthoritativeFinish()) return;

    if (finish) finish.disabled = true;
    try {
      const saver = saveDraft || saveTop;
      if (!saver) throw new Error("Δεν βρέθηκε local Save control.");

      saver.click();
      await flushLocalPersistence();
      markPilotComplete();
      setDraftStatus("Pilot case αποθηκεύτηκε τοπικά · επιβεβαιώνεται το protected completion…");

      const row = await registry.finalizeActiveEncounter();
      if (!row || !["completed", "amended"].includes(row.status)) {
        throw new Error(`Μη αναμενόμενο server status: ${row?.status || "κενό"}`);
      }

      if (row.status === "completed") {
        setDraftStatus("Pilot case ολοκληρώθηκε και συγχρονίστηκε στον protected server ως completed.");
      } else {
        setDraftStatus("Η ολοκλήρωση συγχρονίστηκε στον protected server ως amended.");
      }
      window.alert("Το pilot case αποθηκεύτηκε και ο protected server επιβεβαίωσε το τελικό status. Τα KPI παραμένουν κρυφά μέχρι το baseline lock.");
    } catch (error) {
      const message = error?.message || "Άγνωστο σφάλμα συγχρονισμού";
      setDraftStatus(`Τα δεδομένα διατηρήθηκαν τοπικά, αλλά δεν επιβεβαιώθηκε protected completion: ${message}`);
      window.alert("Τα δεδομένα διατηρήθηκαν τοπικά, αλλά το protected completion δεν επιβεβαιώθηκε. Διόρθωσε τη σύνδεση και ξαναπάτησε Finish.");
    } finally {
      coordinator.endAuthoritativeFinish();
      if (finish) finish.disabled = false;
    }
  }

  if (finish) finish.addEventListener("click", authoritativeFinish, true);

  const next = document.querySelector("#nextBtn");
  if (next) {
    next.addEventListener("click", (event) => {
      if (!isStep6Active()) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      finish?.click();
    }, true);
  }

  updatePilotPill();
})();
