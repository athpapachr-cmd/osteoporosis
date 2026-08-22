(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase";
  const PRIVACY_KEY = "osteoporosis.baselineAuditPilot.privacyDismissed";
  const PILOT_TARGET = 5;

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const elements = {
    pilotPill: $("#pilotPill"),
    caseIdDisplay: $("#caseIdDisplay"),
    encounterDate: $("#encounterDate"),
    encounterType: $("#encounterType"),
    ageYears: $("#ageYears"),
    ageMirror: $("#ageMirror"),
    bmi: $("#bmi"),
    menopauseBlock: $("#menopauseBlock"),
    heidiOutput: $("#heidiOutput"),
    heidiReviewed: $("#heidiReviewed"),
    heidiCorrection: $("#heidiCorrection"),
    heidiComment: $("#heidiComment"),
    quickNotes: $("#quickNotes"),
    progressFill: $("#progressFill"),
    progressText: $("#progressText"),
    draftStatus: $("#draftStatus"),
    saveTopBtn: $("#saveTopBtn"),
    saveDraftBtn: $("#saveDraftBtn"),
    nextBtn: $("#nextBtn"),
    finishVisitBtn: $("#finishVisitBtn"),
    cancelCaseBtn: $("#cancelCaseBtn"),
    casesDialog: $("#casesDialog"),
    privacyDialog: $("#privacyDialog"),
    caseList: $("#caseList"),
    privacyStrip: $("#privacyStrip"),
    dismissPrivacyBtn: $("#dismissPrivacyBtn"),
    signalChecks: $("#signalChecks")
  };

  let currentCase = createEmptyCase(1);
  let dirty = false;
  let activeStep = 1;

  function isoToday() {
    const now = new Date();
    const offset = now.getTimezoneOffset();
    const local = new Date(now.getTime() - offset * 60_000);
    return local.toISOString().slice(0, 10);
  }

  function createUuid() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `case-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function getStore() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function setStore(cases) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cases));
  }

  function nextSequence() {
    const used = getStore()
      .map((item) => Number(item.case_sequence_number || 0))
      .filter(Number.isFinite);
    const next = used.length ? Math.max(...used) + 1 : 1;
    return Math.min(Math.max(next, 1), 999);
  }

  function createEmptyCase(sequence) {
    const seq = Number(sequence || 1);
    return {
      schema: "baseline_osteoporosis_case_form_v1",
      schema_version: 1,
      baseline_phase: "pilot",
      internal_uuid: createUuid(),
      case_sequence_number: seq,
      case_id: `PILOT-${String(seq).padStart(3, "0")}`,
      local_patient_token: createUuid(),
      encounter_date: isoToday(),
      encounter_type: "",
      first_core_baseline_encounter_for_patient: "",
      age_years: "",
      sex: "",
      menopause_status: "",
      bmi: "",
      visit_reason: "",
      osteoporosis_status: "",
      signals: [],
      heidi: {
        used: "",
        output_available: "",
        reviewed_by_clinician: "",
        material_correction_required: "",
        comment: ""
      },
      quick_notes: "",
      created_at: new Date().toISOString(),
      updated_at: null,
      implementation_slice: "step_1_only"
    };
  }

  function safeText(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function markDirty() {
    dirty = true;
    elements.draftStatus.textContent = "Υπάρχουν μη αποθηκευμένες αλλαγές";
    updateProgress();
  }

  function setSelected(field, value, mark = true) {
    if (field === "heidi_used") {
      currentCase.heidi.used = value;
    } else {
      currentCase[field] = value;
    }

    $$(`[data-field="${field}"]`).forEach((button) => {
      button.classList.toggle("selected", button.dataset.value === value);
      button.setAttribute("aria-pressed", button.dataset.value === value ? "true" : "false");
    });

    if (field === "sex") {
      syncSexDependentUi();
    }
    if (field === "heidi_used") {
      syncHeidiUi();
    }
    if (mark) markDirty();
  }

  function syncSexDependentUi() {
    const female = currentCase.sex === "female";
    elements.menopauseBlock.hidden = !female;
    if (!female) {
      currentCase.menopause_status = "";
      $$('[data-field="menopause_status"]').forEach((button) => button.classList.remove("selected"));
    }
  }

  function syncHeidiUi() {
    const used = currentCase.heidi.used === "yes";
    $$('[data-heidi-dependent]').forEach((field) => {
      field.disabled = !used;
    });

    if (!used) {
      currentCase.heidi.output_available = "";
      currentCase.heidi.reviewed_by_clinician = "";
      currentCase.heidi.material_correction_required = "";
      currentCase.heidi.comment = "";
      elements.heidiOutput.value = "";
      elements.heidiReviewed.value = "";
      elements.heidiCorrection.value = "";
      elements.heidiComment.value = "";
    }
  }

  function syncSimpleInputsFromState() {
    elements.caseIdDisplay.textContent = currentCase.case_id;
    elements.encounterDate.value = currentCase.encounter_date || isoToday();
    elements.encounterType.value = currentCase.encounter_type || "";
    elements.ageYears.value = currentCase.age_years ?? "";
    elements.ageMirror.value = currentCase.age_years ?? "";
    elements.bmi.value = currentCase.bmi ?? "";
    elements.heidiOutput.value = currentCase.heidi.output_available || "";
    elements.heidiReviewed.value = currentCase.heidi.reviewed_by_clinician || "";
    elements.heidiCorrection.value = currentCase.heidi.material_correction_required || "";
    elements.heidiComment.value = currentCase.heidi.comment || "";
    elements.quickNotes.value = currentCase.quick_notes || "";

    [
      ["sex", currentCase.sex],
      ["first_core_baseline_encounter_for_patient", currentCase.first_core_baseline_encounter_for_patient],
      ["visit_reason", currentCase.visit_reason],
      ["osteoporosis_status", currentCase.osteoporosis_status],
      ["menopause_status", currentCase.menopause_status],
      ["heidi_used", currentCase.heidi.used]
    ].forEach(([field, value]) => setSelected(field, value || "", false));

    $$("#signalChecks input[type='checkbox']").forEach((checkbox) => {
      checkbox.checked = currentCase.signals.includes(checkbox.value);
    });

    syncSexDependentUi();
    syncHeidiUi();
    updatePilotPill();
    updateProgress();
  }

  function updatePilotPill() {
    const seq = Math.max(1, Number(currentCase.case_sequence_number || 1));
    elements.pilotPill.textContent = `PILOT CASE ${Math.min(seq, PILOT_TARGET)}/${PILOT_TARGET}`;
  }

  function calculateProgress() {
    const checks = [
      Boolean(currentCase.encounter_date),
      Boolean(currentCase.encounter_type),
      Boolean(currentCase.age_years),
      Boolean(currentCase.sex),
      Boolean(currentCase.first_core_baseline_encounter_for_patient),
      Boolean(currentCase.visit_reason),
      Boolean(currentCase.osteoporosis_status),
      Boolean(currentCase.heidi.used),
      currentCase.signals.length > 0
    ];

    if (currentCase.sex === "female") {
      checks.push(Boolean(currentCase.menopause_status));
    }

    if (currentCase.heidi.used === "yes") {
      checks.push(Boolean(currentCase.heidi.output_available));
    }

    const done = checks.filter(Boolean).length;
    return Math.round((done / checks.length) * 100);
  }

  function updateProgress() {
    const progress = calculateProgress();
    elements.progressFill.style.width = `${progress}%`;
    elements.progressText.textContent = `${progress}%`;
  }

  function snapshotInputs() {
    currentCase.encounter_date = elements.encounterDate.value;
    currentCase.encounter_type = elements.encounterType.value;
    currentCase.age_years = elements.ageYears.value;
    currentCase.bmi = elements.bmi.value;
    currentCase.heidi.output_available = elements.heidiOutput.value;
    currentCase.heidi.reviewed_by_clinician = elements.heidiReviewed.value;
    currentCase.heidi.material_correction_required = elements.heidiCorrection.value;
    currentCase.heidi.comment = elements.heidiComment.value.trim();
    currentCase.quick_notes = elements.quickNotes.value.trim();
    currentCase.signals = $$("#signalChecks input[type='checkbox']:checked").map((item) => item.value);
  }

  function saveDraft(showStatus = true) {
    snapshotInputs();
    currentCase.updated_at = new Date().toISOString();

    const cases = getStore();
    const index = cases.findIndex((item) => item.internal_uuid === currentCase.internal_uuid);
    if (index >= 0) cases[index] = currentCase;
    else cases.push(currentCase);

    setStore(cases);
    localStorage.setItem(ACTIVE_KEY, currentCase.internal_uuid);
    dirty = false;

    if (showStatus) {
      const time = new Intl.DateTimeFormat("el-GR", { hour: "2-digit", minute: "2-digit" }).format(new Date());
      elements.draftStatus.textContent = `Draft αποθηκεύτηκε τοπικά στις ${time}`;
    }
    renderCaseList();
    return currentCase;
  }

  function loadCase(uuid) {
    const found = getStore().find((item) => item.internal_uuid === uuid);
    if (!found) return;
    currentCase = found;
    currentCase.heidi ||= { used: "", output_available: "", reviewed_by_clinician: "", material_correction_required: "", comment: "" };
    currentCase.signals ||= [];
    dirty = false;
    localStorage.setItem(ACTIVE_KEY, uuid);
    syncSimpleInputsFromState();
    elements.draftStatus.textContent = "Φορτώθηκε τοπικό draft";
    if (elements.casesDialog.open) elements.casesDialog.close();
    switchStep(1);
  }

  function deleteCase(uuid) {
    const cases = getStore();
    const target = cases.find((item) => item.internal_uuid === uuid);
    const label = target?.case_id || "το case";
    if (!window.confirm(`Να διαγραφεί οριστικά το τοπικό draft ${label};`)) return;
    const nextCases = cases.filter((item) => item.internal_uuid !== uuid);
    setStore(nextCases);
    if (uuid === currentCase.internal_uuid) {
      newCase(false);
    }
    renderCaseList();
  }

  function renderCaseList() {
    const cases = getStore().sort((a, b) => Number(a.case_sequence_number) - Number(b.case_sequence_number));
    if (!cases.length) {
      elements.caseList.innerHTML = '<div class="placeholder-card"><p>Δεν υπάρχουν αποθηκευμένα drafts σε αυτόν τον browser.</p></div>';
      return;
    }

    elements.caseList.innerHTML = cases.map((item) => {
      const updated = item.updated_at ? new Date(item.updated_at).toLocaleString("el-GR") : "—";
      return `
        <div class="case-list-item">
          <div>
            <strong>${safeText(item.case_id)}</strong>
            <span>${safeText(item.encounter_date || "χωρίς ημερομηνία")} · ενημέρωση ${safeText(updated)}</span>
          </div>
          <div class="case-list-actions">
            <button type="button" data-load-case="${safeText(item.internal_uuid)}">Άνοιγμα</button>
            <button type="button" data-delete-case="${safeText(item.internal_uuid)}">Διαγραφή</button>
          </div>
        </div>`;
    }).join("");
  }

  function newCase(confirmUnsaved = true) {
    if (confirmUnsaved && dirty) {
      const proceed = window.confirm("Υπάρχουν μη αποθηκευμένες αλλαγές. Να δημιουργηθεί νέο case χωρίς αποθήκευση;");
      if (!proceed) return;
    }
    currentCase = createEmptyCase(nextSequence());
    localStorage.setItem(ACTIVE_KEY, currentCase.internal_uuid);
    dirty = false;
    syncSimpleInputsFromState();
    elements.draftStatus.textContent = "Νέο pilot case — δεν έχει αποθηκευτεί";
    switchStep(1);
  }

  function switchStep(step) {
    const numeric = Number(step);
    activeStep = numeric;
    $$(".step-tab").forEach((button) => button.classList.toggle("active", Number(button.dataset.step) === numeric));
    $$(".step-panel").forEach((panel) => panel.classList.toggle("active", Number(panel.dataset.stepPanel) === numeric));
    elements.nextBtn.textContent = numeric >= 6 ? "Τέλος →" : "Επόμενο →";
  }

  function setupChoiceButtons() {
    $$('[data-field][data-value]').forEach((button) => {
      button.addEventListener("click", () => {
        setSelected(button.dataset.field, button.dataset.value);
      });
    });
  }

  function setupInputs() {
    elements.encounterDate.addEventListener("input", () => {
      currentCase.encounter_date = elements.encounterDate.value;
      markDirty();
    });
    elements.encounterType.addEventListener("change", () => {
      currentCase.encounter_type = elements.encounterType.value;
      markDirty();
    });

    const syncAge = (value) => {
      currentCase.age_years = value;
      elements.ageYears.value = value;
      elements.ageMirror.value = value;
      markDirty();
    };
    elements.ageYears.addEventListener("input", () => syncAge(elements.ageYears.value));
    elements.ageMirror.addEventListener("input", () => syncAge(elements.ageMirror.value));

    elements.bmi.addEventListener("input", () => {
      currentCase.bmi = elements.bmi.value;
      markDirty();
    });

    elements.heidiOutput.addEventListener("change", () => {
      currentCase.heidi.output_available = elements.heidiOutput.value;
      markDirty();
    });
    elements.heidiReviewed.addEventListener("change", () => {
      currentCase.heidi.reviewed_by_clinician = elements.heidiReviewed.value;
      markDirty();
    });
    elements.heidiCorrection.addEventListener("change", () => {
      currentCase.heidi.material_correction_required = elements.heidiCorrection.value;
      markDirty();
    });
    elements.heidiComment.addEventListener("input", markDirty);
    elements.quickNotes.addEventListener("input", markDirty);

    $$("#signalChecks input[type='checkbox']").forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        const none = $("#signalChecks input[value='none_known']");
        if (checkbox.value === "none_known" && checkbox.checked) {
          $$("#signalChecks input[type='checkbox']").forEach((item) => {
            if (item !== checkbox) item.checked = false;
          });
        } else if (checkbox.checked && none) {
          none.checked = false;
        }
        currentCase.signals = $$("#signalChecks input[type='checkbox']:checked").map((item) => item.value);
        markDirty();
      });
    });
  }

  function setupNavigation() {
    $$(".step-tab").forEach((button) => button.addEventListener("click", () => switchStep(button.dataset.step)));

    elements.nextBtn.addEventListener("click", () => {
      if (activeStep < 6) {
        switchStep(activeStep + 1);
      } else {
        window.alert("Η τελική υποβολή θα ενεργοποιηθεί όταν υλοποιηθούν και τα 6 βήματα. Το Step 1 λειτουργεί ήδη ως pilot draft.");
      }
    });

    elements.saveTopBtn.addEventListener("click", () => saveDraft());
    elements.saveDraftBtn.addEventListener("click", () => saveDraft());

    elements.finishVisitBtn.addEventListener("click", () => {
      saveDraft();
      window.alert("Το Step 1 αποθηκεύτηκε ως pilot draft. Δεν χαρακτηρίζεται ακόμη ως ολοκληρωμένο baseline case μέχρι να υλοποιηθούν τα υπόλοιπα βήματα.");
    });

    elements.cancelCaseBtn.addEventListener("click", () => {
      if (!window.confirm("Να καθαριστεί το τρέχον case από την οθόνη; Το αποθηκευμένο draft, αν υπάρχει, θα παραμείνει στα Cases.")) return;
      newCase(false);
    });

    $$('[data-nav-action="new-case"]').forEach((button) => button.addEventListener("click", () => newCase(true)));
    $$('[data-nav-action="cases"]').forEach((button) => button.addEventListener("click", () => {
      renderCaseList();
      elements.casesDialog.showModal();
    }));
    $$('[data-nav-action="privacy"]').forEach((button) => button.addEventListener("click", () => elements.privacyDialog.showModal()));
    $$('[data-nav-action="heidi"]').forEach((button) => button.addEventListener("click", () => {
      switchStep(1);
      elements.heidiComment.focus();
      elements.heidiComment.scrollIntoView({ behavior: "smooth", block: "center" });
    }));

    elements.caseList.addEventListener("click", (event) => {
      const loadButton = event.target.closest("[data-load-case]");
      const deleteButton = event.target.closest("[data-delete-case]");
      if (loadButton) loadCase(loadButton.dataset.loadCase);
      if (deleteButton) deleteCase(deleteButton.dataset.deleteCase);
    });
  }

  function setupPrivacy() {
    const dismissed = localStorage.getItem(PRIVACY_KEY) === "yes";
    elements.privacyStrip.hidden = dismissed;
    elements.dismissPrivacyBtn.addEventListener("click", () => {
      localStorage.setItem(PRIVACY_KEY, "yes");
      elements.privacyStrip.hidden = true;
    });
  }

  function restoreActiveCase() {
    const activeUuid = localStorage.getItem(ACTIVE_KEY);
    const stored = getStore();
    const found = stored.find((item) => item.internal_uuid === activeUuid);
    if (found) {
      currentCase = found;
      currentCase.heidi ||= { used: "", output_available: "", reviewed_by_clinician: "", material_correction_required: "", comment: "" };
      currentCase.signals ||= [];
      elements.draftStatus.textContent = "Φορτώθηκε το τελευταίο τοπικό draft";
    } else {
      currentCase = createEmptyCase(nextSequence());
      localStorage.setItem(ACTIVE_KEY, currentCase.internal_uuid);
    }
  }

  window.addEventListener("beforeunload", (event) => {
    if (!dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });

  setupChoiceButtons();
  setupInputs();
  setupNavigation();
  setupPrivacy();
  restoreActiveCase();
  syncSimpleInputsFromState();
  renderCaseList();
})();
