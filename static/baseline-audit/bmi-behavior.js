(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";

  const weight = () => $("#weightKg");
  const height = () => $("#currentHeightCm");
  const bmi = () => $("#bmi");

  function numberOrNull(value) {
    if (value === "" || value === null || value === undefined) return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function hasMeasuredPair() {
    const w = numberOrNull(weight()?.value);
    const h = numberOrNull(height()?.value);
    return w !== null && h !== null && w > 0 && h > 0;
  }

  function activeStoredCase() {
    const uuid = localStorage.getItem(ACTIVE_KEY) || "";
    if (!uuid) return null;
    try {
      const cases = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(cases) ? cases.find((item) => item?.internal_uuid === uuid) || null : null;
    } catch {
      return null;
    }
  }

  function ensureModeNote() {
    const input = bmi();
    if (!input) return null;
    let note = $("#bmiModeNote");
    if (note) return note;
    note = document.createElement("small");
    note.id = "bmiModeNote";
    note.className = "bmi-mode-note";
    input.insertAdjacentElement("afterend", note);
    return note;
  }

  function setManualMode({ clearDerived = false } = {}) {
    const input = bmi();
    if (!input) return;
    const note = ensureModeNote();
    const wasDerived = input.dataset.bmiMode === "derived";

    input.readOnly = false;
    input.removeAttribute("aria-readonly");
    input.dataset.bmiMode = "manual";
    input.title = "Χειροκίνητο / external BMI όταν δεν υπάρχουν και βάρος και ύψος.";
    input.placeholder = "χειροκίνητο / external BMI";
    if (note) note.textContent = "Χειροκίνητο / external BMI";

    if (clearDerived && wasDerived && input.value !== "") {
      input.value = "";
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  function setDerivedMode() {
    const input = bmi();
    if (!input) return;
    const note = ensureModeNote();
    input.readOnly = true;
    input.setAttribute("aria-readonly", "true");
    input.dataset.bmiMode = "derived";
    input.title = "Υπολογίζεται αυτόματα από το τρέχον βάρος και ύψος.";
    input.placeholder = "αυτόματο από βάρος + ύψος";
    if (note) note.textContent = "Αυτόματο από βάρος + ύψος";
  }

  function sync({ allowClear = true } = {}) {
    const input = bmi();
    if (!input) return;

    if (hasMeasuredPair()) {
      setDerivedMode();
      return;
    }

    const storedSource = activeStoredCase()?.anthropometrics?.bmi_source || "";
    const derivedBefore = input.dataset.bmiMode === "derived" || storedSource === "calculated_weight_height";
    setManualMode({ clearDerived: allowClear && derivedBefore });
  }

  function deferredSync(options) {
    setTimeout(() => sync(options), 0);
  }

  ["input", "change"].forEach((type) => {
    document.addEventListener(type, (event) => {
      if (event.target === weight() || event.target === height()) deferredSync({ allowClear: true });
    });
  });

  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-load-case], [data-nav-action=\"new-case\"]")) deferredSync({ allowClear: false });
  });

  // Run after app-core has hydrated/recalculated the active case.
  deferredSync({ allowClear: false });
})();
