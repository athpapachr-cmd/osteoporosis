(() => {
  "use strict";

  // Compatibility filename retained so the production-proven single Finish
  // ownership/load-order seam does not change during C2.
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

  function markEncounterComplete() {
    const id = activeId();
    if (!id) throw new Error("Δεν υπάρχει ενεργή επίσκεψη.");
    const cases = getCases();
    const index = cases.findIndex((c) => c.internal_uuid === id);
    if (index < 0) throw new Error("Η ενεργή επίσκεψη δεν βρέθηκε στο προσωρινό browser cache.");
    const step6 = cases[index].step6 || {};
    cases[index].encounter_completion = {
      status: "complete",
      completed_at: new Date().toISOString(),
      ready_for_audit_at_completion: step6.capture_quality?.ready_for_audit || "",
      completion_time_minutes: step6.capture_quality?.completion_time_minutes ?? null
    };
    cases[index].workflow_mode = "clinical";
    if (!cases[index].baseline_phase || cases[index].baseline_phase === "pilot") cases[index].baseline_phase = "clinical";
    cases[index].implementation_slice = "clinical_encounter_complete";
    setCases(cases);
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
      if (!saver) throw new Error("Δεν βρέθηκε Save control.");

      saver.click();
      await flushLocalPersistence();
      markEncounterComplete();
      setDraftStatus("Η επίσκεψη διατηρήθηκε στο προσωρινό browser cache · επιβεβαιώνεται το protected completion…");

      const row = await registry.finalizeActiveEncounter();
      if (!row || !["completed", "amended"].includes(row.status)) {
        throw new Error(`Μη αναμενόμενο server status: ${row?.status || "κενό"}`);
      }

      if (row.status === "completed") {
        setDraftStatus("Η επίσκεψη ολοκληρώθηκε και συγχρονίστηκε στον protected server ως completed.");
      } else {
        setDraftStatus("Η ολοκλήρωση συγχρονίστηκε στον protected server ως amended.");
      }
      window.alert("Η επίσκεψη αποθηκεύτηκε και ο protected server επιβεβαίωσε το τελικό status.");
    } catch (error) {
      const message = error?.message || "Άγνωστο σφάλμα συγχρονισμού";
      setDraftStatus(`Οι τοπικές αλλαγές διατηρήθηκαν, αλλά δεν επιβεβαιώθηκε protected completion: ${message}`);
      window.alert("Οι τοπικές αλλαγές διατηρήθηκαν, αλλά το protected completion δεν επιβεβαιώθηκε. Έλεγξε τη σύνδεση ή πιθανό cross-device conflict και ξαναπάτησε Τέλος επίσκεψης.");
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
})();
