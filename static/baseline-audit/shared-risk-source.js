(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const $ = (s, r = document) => r.querySelector(s);

  const SHARED_CONTROLS = [
    "#s3FallsCount",
    "#s3Cfs",
    "#s3Cognitive",
    "#s3Immobility",
    "#s3SarcApplicable",
    "#s3SarcMethod",
    "#s3SarcF"
  ];

  function getCases() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function setCases(cases) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cases));
  }

  function activeId() {
    return localStorage.getItem(ACTIVE_KEY) || "";
  }

  function activeCase() {
    const id = activeId();
    return getCases().find((c) => c.internal_uuid === id) || null;
  }

  function canonicalValues() {
    const risk = activeCase()?.risk_context || {};
    const method = risk.sarcopenia_screen_method === "sarc_f"
      ? "SARC_F"
      : risk.sarcopenia_screen_method || "";
    return {
      falls: risk.falls_last_12_months ?? "",
      cfs: risk.cfs_score ?? "",
      cognitive: Boolean(risk.cognitive_impairment),
      immobility: Boolean(risk.significant_immobility),
      sarcApplicable: risk.sarcopenia_case_finding_relevant ? "yes" : "no",
      sarcMethod: method,
      sarcF: risk.sarc_f_score ?? ""
    };
  }

  function ensureProjectionNote() {
    const panel = $('[data-step-panel="3"]');
    if (!panel || $("#s3CanonicalRiskNote")) return;
    const functionCard = $("#s3FallsCount")?.closest("article");
    if (!functionCard) return;
    const note = document.createElement("div");
    note.id = "s3CanonicalRiskNote";
    note.className = "derived-note";
    note.innerHTML = "Falls count, CFS, cognitive impairment, immobility και βασικό sarcopenia screening είναι <strong>canonical στο Step 1</strong>. Εδώ εμφανίζονται μόνο ως read-only projection· τα επιπλέον functional/sarcopenia tests παραμένουν editable.";
    functionCard.prepend(note);
  }

  function markReadOnly() {
    SHARED_CONTROLS.forEach((selector) => {
      const node = $(selector);
      if (!node) return;
      node.disabled = true;
      node.setAttribute("aria-disabled", "true");
      node.dataset.canonicalSource = "step1";
      const label = node.closest("label");
      if (label) {
        label.classList.add("canonical-projection");
        label.title = "Canonical source: Step 1";
      }
    });
  }

  function syncProjectionFromStep1() {
    const values = canonicalValues();
    const setValue = (selector, value) => { const n = $(selector); if (n) n.value = value ?? ""; };
    const setCheck = (selector, value) => { const n = $(selector); if (n) n.checked = Boolean(value); };

    setValue("#s3FallsCount", values.falls);
    setValue("#s3Cfs", values.cfs);
    setCheck("#s3Cognitive", values.cognitive);
    setCheck("#s3Immobility", values.immobility);
    setValue("#s3SarcApplicable", values.sarcApplicable);
    setValue("#s3SarcMethod", values.sarcMethod);
    setValue("#s3SarcF", values.sarcF);

    markReadOnly();
    ensureProjectionNote();
  }

  function stripDuplicateStep3Storage() {
    const id = activeId();
    if (!id) return;
    const cases = getCases();
    const index = cases.findIndex((c) => c.internal_uuid === id);
    if (index < 0 || !cases[index].step3) return;

    const step3 = { ...cases[index].step3 };
    if (step3.function) {
      step3.function = { ...step3.function };
      delete step3.function.falls_count_12m;
      delete step3.function.cfs;
      delete step3.function.cognitive_impairment;
      delete step3.function.significant_immobility;
    }
    if (step3.sarcopenia) {
      step3.sarcopenia = { ...step3.sarcopenia };
      delete step3.sarcopenia.applicable;
      delete step3.sarcopenia.method;
      delete step3.sarcopenia.sarc_f;
      delete step3.sarcopenia.derived;
    }

    cases[index] = { ...cases[index], step3 };
    setCases(cases);
  }

  function scheduleSync() {
    setTimeout(() => {
      syncProjectionFromStep1();
      stripDuplicateStep3Storage();
    }, 0);
  }

  const panel = $('[data-step-panel="3"]');
  if (panel) {
    panel.addEventListener("input", () => setTimeout(stripDuplicateStep3Storage, 0));
    panel.addEventListener("change", () => setTimeout(stripDuplicateStep3Storage, 0));
  }

  document.querySelectorAll(".step-tab").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.step === "3") scheduleSync();
    });
  });

  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-load-case]") || event.target.closest('[data-nav-action="new-case"]')) {
      scheduleSync();
    }
  });

  ["#saveTopBtn", "#saveDraftBtn", "#finishVisitBtn"].forEach((selector) => {
    const node = $(selector);
    if (node) node.addEventListener("click", () => setTimeout(stripDuplicateStep3Storage, 25));
  });

  const style = document.createElement("style");
  style.textContent = `
    .canonical-projection { opacity: .78; }
    .canonical-projection [data-canonical-source="step1"] { cursor: not-allowed; background: #f8fafc; }
  `;
  document.head.appendChild(style);

  scheduleSync();
})();
