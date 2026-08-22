(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const $ = (s, r=document) => r.querySelector(s);

  const MACHINE_OPTIONS = [
    ["", "—"],
    ["hologic_horizon", "Hologic Horizon"],
    ["hologic_discovery", "Hologic Discovery"],
    ["ge_lunar_idxa", "GE Lunar iDXA"],
    ["ge_lunar_prodigy", "GE Lunar Prodigy"],
    ["norland", "Norland"],
    ["other_unknown", "Other / unknown"]
  ];

  const LABEL_TO_KEY = new Map(MACHINE_OPTIONS.map(([k,l]) => [l.toLowerCase(), k]));
  const KEYS = new Set(MACHINE_OPTIONS.map(([k]) => k));

  function getCases(){
    try {
      const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(data) ? data : [];
    } catch { return []; }
  }

  function setCases(cases){ localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
  function activeUuid(){ return localStorage.getItem(ACTIVE_KEY) || ""; }
  function activeCase(){ const id=activeUuid(); return getCases().find(c => c.internal_uuid === id) || null; }
  function optionHtml(){ return MACHINE_OPTIONS.map(([v,l]) => `<option value="${v}">${l}</option>`).join(""); }

  function normalizeStoredMachine(rawMachine, rawLabel){
    const machine = String(rawMachine || "").trim();
    const label = String(rawLabel || "").trim();
    if (!machine) return { machine:"", machine_label:label };
    if (KEYS.has(machine)) return { machine, machine_label:label };
    const mapped = LABEL_TO_KEY.get(machine.toLowerCase());
    if (mapped) return { machine:mapped, machine_label:label };
    return { machine:"other_unknown", machine_label:label || machine };
  }

  function persistSelection(){
    const id = activeUuid();
    const select = $("#s3DxaMachine");
    const label = $("#s3MachineLocalLabel");
    if (!id || !select) return;
    const cases = getCases();
    const i = cases.findIndex(c => c.internal_uuid === id);
    if (i < 0) return;
    const existingStep3 = cases[i].step3 || {};
    const existingDxa = existingStep3.dxa || {};
    cases[i] = {
      ...cases[i],
      step3: {
        ...existingStep3,
        dxa: {
          ...existingDxa,
          machine: select.value || "",
          machine_label: (label?.value || "").trim()
        }
      }
    };
    setCases(cases);
  }

  function ensureNativeSelect(){
    const old = $("#s3DxaMachine");
    if (!old) return;

    const c = activeCase();
    const stored = normalizeStoredMachine(c?.step3?.dxa?.machine, c?.step3?.dxa?.machine_label);
    const domValue = String(old.value || "").trim();
    const domNormalized = normalizeStoredMachine(domValue, "");
    const effective = domValue ? domNormalized : stored;

    let select = old;
    if (old.tagName !== "SELECT") {
      select = document.createElement("select");
      select.id = "s3DxaMachine";
      select.innerHTML = optionHtml();
      old.replaceWith(select);
    } else if (!select.options.length) {
      select.innerHTML = optionHtml();
    }

    select.value = effective.machine || "";

    const parent = select.closest("label");
    let labelInput = $("#s3MachineLocalLabel");
    if (!labelInput && parent) {
      labelInput = document.createElement("input");
      labelInput.id = "s3MachineLocalLabel";
      labelInput.type = "text";
      labelInput.maxLength = 80;
      labelInput.placeholder = "Local machine label / ID (optional)";
      parent.appendChild(labelInput);
    }
    if (labelInput) labelInput.value = effective.machine_label || "";

    persistSelection();
  }

  function bind(){
    ensureNativeSelect();

    document.addEventListener("change", (event) => {
      if (event.target?.id === "s3DxaMachine" || event.target?.id === "s3MachineLocalLabel") {
        setTimeout(persistSelection, 0);
      }
    });
    document.addEventListener("input", (event) => {
      if (event.target?.id === "s3MachineLocalLabel") setTimeout(persistSelection, 0);
    });

    document.addEventListener("click", (event) => {
      if (event.target.closest('[data-step="3"]') || event.target.closest("[data-load-case]") || event.target.closest('[data-nav-action="new-case"]')) {
        setTimeout(ensureNativeSelect, 0);
      }
    });

    ["#saveTopBtn", "#saveDraftBtn", "#finishVisitBtn"].forEach((selector) => {
      $(selector)?.addEventListener("click", () => setTimeout(persistSelection, 0));
    });
  }

  bind();
})();
