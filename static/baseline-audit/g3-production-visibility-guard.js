(() => {
  "use strict";

  const ACTIVE_PATIENT_KEY = "osteoporosis.clinical.activePatient.v1";

  function activePatientId() {
    return sessionStorage.getItem(ACTIVE_PATIENT_KEY) || "";
  }

  function ensureSummaryRoot() {
    let root = document.querySelector("#patientLongitudinalSummary");
    if (root) return root;
    root = document.createElement("section");
    root.id = "patientLongitudinalSummary";
    root.className = "patient-longitudinal-summary";
    const flow = document.querySelector("#progressiveGuidanceSummary") || document.querySelector(".step-tabs");
    if (flow?.parentNode) flow.parentNode.insertBefore(root, flow);
    return root;
  }

  function renderNoPatientPlaceholder() {
    if (activePatientId()) return;
    const root = ensureSummaryRoot();
    const hasMeaningfulContent = root.querySelector(".patient-summary-head") && root.textContent.includes("Σύνοψη ασθενούς");
    if (hasMeaningfulContent && !root.hidden) return;

    root.hidden = false;
    root.innerHTML = "";

    const head = document.createElement("div");
    head.className = "patient-summary-head";
    const title = document.createElement("h2");
    title.textContent = "Σύνοψη ασθενούς";
    const badge = document.createElement("span");
    badge.className = "patient-summary-readonly";
    badge.textContent = "Read-only longitudinal";
    head.append(title, badge);

    const status = document.createElement("div");
    status.className = "progressive-guidance-meta";
    status.textContent = "Δεν έχει επιλεγεί protected patient. Άνοιξε ασθενή από το Patient Registry για να εμφανιστεί η longitudinal σύνοψη από τις προηγούμενες ολοκληρωμένες/τροποποιημένες επισκέψεις.";

    root.append(head, status);
  }

  const root = ensureSummaryRoot();
  if (typeof MutationObserver !== "undefined") {
    const observer = new MutationObserver(() => {
      if (!activePatientId() && (root.hidden || !root.textContent.includes("Σύνοψη ασθενούς"))) {
        renderNoPatientPlaceholder();
      }
    });
    observer.observe(root, { attributes: true, childList: true, subtree: true, attributeFilter: ["hidden"] });
  }

  renderNoPatientPlaceholder();
  setTimeout(renderNoPatientPlaceholder, 250);
})();
