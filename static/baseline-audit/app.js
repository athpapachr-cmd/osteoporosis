(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const PRIVACY_KEY = "osteoporosis.baselineAuditPilot.privacyDismissed";
  const PILOT_TARGET = 5;
  const LOCAL_LOW_BMI_THRESHOLD = 20; // workflow flag only; not a FRAX/NOGG threshold.

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const valueOrNull = (value) => value === "" || value === null || value === undefined ? null : value;
  const numberOrNull = (value) => value === "" || value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value);

  const el = {
    pilotPill: $("#pilotPill"),
    caseIdDisplay: $("#caseIdDisplay"),
    encounterDate: $("#encounterDate"),
    ageYears: $("#ageYears"),
    encounterArchetype: $("#encounterArchetype"),
    weightKg: $("#weightKg"),
    currentHeightCm: $("#currentHeightCm"),
    heightSource: $("#heightSource"),
    referenceHeightCm: $("#referenceHeightCm"),
    bmi: $("#bmi"),
    heightLossDisplay: $("#heightLossDisplay"),
    menopauseBlock: $("#menopauseBlock"),
    priorFragilityFracture: $("#priorFragilityFracture"),
    fractureDetails: $("#fractureDetails"),
    lastFractureSite: $("#lastFractureSite"),
    lastFractureMonth: $("#lastFractureMonth"),
    parentalHipFracture: $("#parentalHipFracture"),
    frailtyImmobility: $("#frailtyImmobility"),
    frailtyDetails: $("#frailtyDetails"),
    cfsScore: $("#cfsScore"),
    cognitiveImpairment: $("#cognitiveImpairment"),
    significantImmobility: $("#significantImmobility"),
    glucocorticoids: $("#glucocorticoids"),
    gcDetails: $("#gcDetails"),
    gcDoseMg: $("#gcDoseMg"),
    gcDurationMonths: $("#gcDurationMonths"),
    fallsLast12m: $("#fallsLast12m"),
    secondaryContext: $("#secondaryContext"),
    secondaryDetails: $("#secondaryDetails"),
    sarcopeniaRelevant: $("#sarcopeniaRelevant"),
    sarcopeniaDetails: $("#sarcopeniaDetails"),
    sarcopeniaMethod: $("#sarcopeniaMethod"),
    sarcFScoreWrap: $("#sarcFScoreWrap"),
    sarcFScore: $("#sarcFScore"),
    heidiOutput: $("#heidiOutput"),
    heidiReviewed: $("#heidiReviewed"),
    heidiCorrection: $("#heidiCorrection"),
    heidiCorrectionCategories: $("#heidiCorrectionCategories"),
    heidiCategoryChecks: $("#heidiCategoryChecks"),
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
    dismissPrivacyBtn: $("#dismissPrivacyBtn")
  };

  let currentCase = createEmptyCase(1);
  let dirty = false;
  let activeStep = 1;
  let bmiAutocalculated = false;

  function isoToday() {
    const now = new Date();
    const offset = now.getTimezoneOffset();
    return new Date(now.getTime() - offset * 60_000).toISOString().slice(0, 10);
  }

  function createUuid() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
    return `case-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function createEmptyCase(sequence) {
    const seq = Number(sequence || 1);
    return {
      schema: "baseline_osteoporosis_case_form_v1",
      schema_version: "1.1-step1-refined",
      baseline_phase: "pilot",
      internal_uuid: createUuid(),
      case_sequence_number: seq,
      case_id: `PILOT-${String(seq).padStart(3, "0")}`,
      local_patient_token: createUuid(),
      encounter_date: isoToday(),
      age_years: null,
      sex: "",
      menopause_status: "",
      patient_relationship_status: "",
      encounter_archetype: "",
      first_core_baseline_encounter_for_patient: "",
      osteoporosis_status: "",
      anthropometrics: {
        weight_kg: null,
        current_height_cm: null,
        height_source: "",
        reference_height_cm: null,
        bmi: null,
        bmi_source: "",
        derived_height_loss_cm: null
      },
      risk_context: {
        prior_fragility_fracture: false,
        last_fracture_site: "",
        last_fracture_month: "",
        parental_hip_fracture: false,
        frailty_or_immobility: false,
        cfs_score: null,
        cognitive_impairment: false,
        significant_immobility: false,
        glucocorticoids: false,
        glucocorticoid_prednisolone_mg_day: null,
        glucocorticoid_duration_months: null,
        falls_last_12_months: null,
        secondary_context: false,
        secondary_conditions: [],
        sarcopenia_case_finding_relevant: false,
        sarcopenia_screen_method: "",
        sarc_f_score: null,
        derived: {}
      },
      heidi: {
        used: "",
        output_available: "",
        reviewed_by_clinician: "",
        material_correction_required: "",
        correction_categories: []
      },
      quick_notes: "",
      created_at: new Date().toISOString(),
      updated_at: null,
      implementation_slice: "step_1_refined"
    };
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
    const used = getStore().map((item) => Number(item.case_sequence_number || 0)).filter(Number.isFinite);
    return used.length ? Math.max(...used) + 1 : 1;
  }

  function safeText(value) {
    return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  }

  function markDirty() {
    dirty = true;
    el.draftStatus.textContent = "Υπάρχουν μη αποθηκευμένες αλλαγές";
    recalculateDerived();
    updateProgress();
  }

  function setSelected(field, value, mark = true) {
    if (field === "heidi_used") currentCase.heidi.used = value;
    else currentCase[field] = value;

    $$(`[data-field="${field}"][data-value]`).forEach((button) => {
      const selected = button.dataset.value === value;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", selected ? "true" : "false");
    });

    if (field === "sex") syncSexDependentUi();
    if (field === "heidi_used") syncHeidiUi();
    if (mark) markDirty();
  }

  function syncSexDependentUi() {
    const female = currentCase.sex === "female";
    el.menopauseBlock.hidden = !female;
    if (!female) {
      currentCase.menopause_status = "";
      $$('[data-field="menopause_status"]').forEach((button) => button.classList.remove("selected"));
    }
  }

  function syncHeidiUi() {
    const used = currentCase.heidi.used === "yes";
    $$('[data-heidi-dependent]').forEach((field) => field.disabled = !used);
    if (!used) {
      currentCase.heidi.output_available = "";
      currentCase.heidi.reviewed_by_clinician = "";
      currentCase.heidi.material_correction_required = "";
      currentCase.heidi.correction_categories = [];
      el.heidiOutput.value = "";
      el.heidiReviewed.value = "";
      el.heidiCorrection.value = "";
    }
    syncHeidiCorrectionUi();
  }

  function syncHeidiCorrectionUi() {
    const show = currentCase.heidi.used === "yes" && currentCase.heidi.material_correction_required === "yes";
    el.heidiCorrectionCategories.hidden = !show;
    if (!show) {
      currentCase.heidi.correction_categories = [];
      $$('#heidiCategoryChecks input[type="checkbox"]').forEach((checkbox) => checkbox.checked = false);
    }
  }

  function syncRiskDetailUi() {
    el.fractureDetails.hidden = !currentCase.risk_context.prior_fragility_fracture;
    el.frailtyDetails.hidden = !currentCase.risk_context.frailty_or_immobility;
    el.gcDetails.hidden = !currentCase.risk_context.glucocorticoids;
    el.secondaryDetails.hidden = !currentCase.risk_context.secondary_context;
    el.sarcopeniaDetails.hidden = !currentCase.risk_context.sarcopenia_case_finding_relevant;
    const sarcF = currentCase.risk_context.sarcopenia_case_finding_relevant && currentCase.risk_context.sarcopenia_screen_method === "sarc_f";
    el.sarcFScoreWrap.hidden = !sarcF;
  }

  function clearDependentRiskFields(key) {
    const risk = currentCase.risk_context;
    if (key === "fracture") {
      risk.last_fracture_site = "";
      risk.last_fracture_month = "";
      el.lastFractureSite.value = "";
      el.lastFractureMonth.value = "";
    } else if (key === "frailty") {
      risk.cfs_score = null;
      risk.cognitive_impairment = false;
      risk.significant_immobility = false;
      el.cfsScore.value = "";
      el.cognitiveImpairment.checked = false;
      el.significantImmobility.checked = false;
    } else if (key === "gc") {
      risk.glucocorticoid_prednisolone_mg_day = null;
      risk.glucocorticoid_duration_months = null;
      el.gcDoseMg.value = "";
      el.gcDurationMonths.value = "";
    } else if (key === "secondary") {
      risk.secondary_conditions = [];
      $$('input[name="secondaryCondition"]').forEach((checkbox) => checkbox.checked = false);
    } else if (key === "sarcopenia") {
      risk.sarcopenia_screen_method = "";
      risk.sarc_f_score = null;
      el.sarcopeniaMethod.value = "";
      el.sarcFScore.value = "";
    }
  }

  function monthsBetween(yyyyMm, encounterDate) {
    if (!yyyyMm || !encounterDate) return null;
    const [y, m] = yyyyMm.split("-").map(Number);
    const d = new Date(`${encounterDate}T00:00:00`);
    if (!y || !m || Number.isNaN(d.getTime())) return null;
    return (d.getFullYear() - y) * 12 + (d.getMonth() - (m - 1));
  }

  function recalculateDerived() {
    const a = currentCase.anthropometrics;
    const risk = currentCase.risk_context;

    const weight = numberOrNull(el.weightKg.value);
    const height = numberOrNull(el.currentHeightCm.value);
    if (weight !== null && height !== null && height > 0) {
      const bmi = weight / Math.pow(height / 100, 2);
      a.bmi = Math.round(bmi * 10) / 10;
      a.bmi_source = "calculated_weight_height";
      bmiAutocalculated = true;
      el.bmi.value = a.bmi.toFixed(1);
    } else if (!bmiAutocalculated) {
      a.bmi = numberOrNull(el.bmi.value);
      a.bmi_source = a.bmi === null ? "" : "manual_or_external";
    } else if (weight === null || height === null) {
      bmiAutocalculated = false;
      a.bmi = numberOrNull(el.bmi.value);
      a.bmi_source = a.bmi === null ? "" : "manual_or_external";
    }

    const referenceHeight = numberOrNull(el.referenceHeightCm.value);
    const currentHeight = numberOrNull(el.currentHeightCm.value);
    a.derived_height_loss_cm = referenceHeight !== null && currentHeight !== null ? Math.max(0, Math.round((referenceHeight - currentHeight) * 10) / 10) : null;
    el.heightLossDisplay.textContent = a.derived_height_loss_cm === null ? "—" : `${a.derived_height_loss_cm.toFixed(1)} cm`;

    const fractureAgeMonths = monthsBetween(risk.last_fracture_month, currentCase.encounter_date);
    const gcDose = numberOrNull(risk.glucocorticoid_prednisolone_mg_day);
    const gcDuration = numberOrNull(risk.glucocorticoid_duration_months);
    const falls = numberOrNull(risk.falls_last_12_months);
    const sarcF = numberOrNull(risk.sarc_f_score);

    let gcDoseBand = null;
    if (gcDose !== null) {
      if (gcDose < 2.5) gcDoseBand = "lt_2_5";
      else if (gcDose < 7.5) gcDoseBand = "2_5_to_lt_7_5";
      else gcDoseBand = "gte_7_5";
    }

    risk.derived = {
      low_bmi_workflow_flag: a.bmi !== null ? a.bmi < LOCAL_LOW_BMI_THRESHOLD : null,
      low_bmi_workflow_threshold: LOCAL_LOW_BMI_THRESHOLD,
      height_loss_ge_4_cm: a.derived_height_loss_cm !== null ? a.derived_height_loss_cm >= 4 : null,
      months_since_last_fragility_fracture: fractureAgeMonths,
      fracture_within_24_months: fractureAgeMonths !== null ? fractureAgeMonths >= 0 && fractureAgeMonths <= 24 : null,
      recent_vertebral_fracture_within_24_months: fractureAgeMonths !== null && risk.last_fracture_site === "vertebral" ? fractureAgeMonths >= 0 && fractureAgeMonths <= 24 : null,
      recurrent_falls_2_or_more_last_12m: falls !== null ? falls >= 2 : null,
      glucocorticoid_dose_band: gcDoseBand,
      glucocorticoid_exposure_3m_or_more: gcDuration !== null ? gcDuration >= 3 : null,
      glucocorticoid_gt_20_mg_day: gcDose !== null ? gcDose > 20 : null,
      sarc_f_positive_ge_4: sarcF !== null ? sarcF >= 4 : null
    };
  }

  function snapshotInputs() {
    currentCase.encounter_date = el.encounterDate.value || isoToday();
    currentCase.age_years = numberOrNull(el.ageYears.value);
    currentCase.encounter_archetype = el.encounterArchetype.value;

    const a = currentCase.anthropometrics;
    a.weight_kg = numberOrNull(el.weightKg.value);
    a.current_height_cm = numberOrNull(el.currentHeightCm.value);
    a.height_source = el.heightSource.value;
    a.reference_height_cm = numberOrNull(el.referenceHeightCm.value);
    if (!bmiAutocalculated) {
      a.bmi = numberOrNull(el.bmi.value);
      a.bmi_source = a.bmi === null ? "" : "manual_or_external";
    }

    const risk = currentCase.risk_context;
    risk.prior_fragility_fracture = el.priorFragilityFracture.checked;
    risk.last_fracture_site = el.lastFractureSite.value;
    risk.last_fracture_month = el.lastFractureMonth.value;
    risk.parental_hip_fracture = el.parentalHipFracture.checked;
    risk.frailty_or_immobility = el.frailtyImmobility.checked;
    risk.cfs_score = numberOrNull(el.cfsScore.value);
    risk.cognitive_impairment = el.cognitiveImpairment.checked;
    risk.significant_immobility = el.significantImmobility.checked;
    risk.glucocorticoids = el.glucocorticoids.checked;
    risk.glucocorticoid_prednisolone_mg_day = numberOrNull(el.gcDoseMg.value);
    risk.glucocorticoid_duration_months = numberOrNull(el.gcDurationMonths.value);
    risk.falls_last_12_months = numberOrNull(el.fallsLast12m.value);
    risk.secondary_context = el.secondaryContext.checked;
    risk.secondary_conditions = $$('input[name="secondaryCondition"]:checked').map((item) => item.value);
    risk.sarcopenia_case_finding_relevant = el.sarcopeniaRelevant.checked;
    risk.sarcopenia_screen_method = el.sarcopeniaMethod.value;
    risk.sarc_f_score = numberOrNull(el.sarcFScore.value);

    currentCase.heidi.output_available = el.heidiOutput.value;
    currentCase.heidi.reviewed_by_clinician = el.heidiReviewed.value;
    currentCase.heidi.material_correction_required = el.heidiCorrection.value;
    currentCase.heidi.correction_categories = $$('#heidiCategoryChecks input[type="checkbox"]:checked').map((item) => item.value);
    currentCase.quick_notes = el.quickNotes.value.trim();

    recalculateDerived();
  }

  function calculateProgress() {
    snapshotInputs();
    const checks = [
      Boolean(currentCase.encounter_date),
      currentCase.age_years !== null,
      Boolean(currentCase.sex),
      Boolean(currentCase.patient_relationship_status),
      Boolean(currentCase.encounter_archetype),
      Boolean(currentCase.first_core_baseline_encounter_for_patient),
      Boolean(currentCase.osteoporosis_status),
      Boolean(currentCase.heidi.used)
    ];
    if (currentCase.sex === "female") checks.push(Boolean(currentCase.menopause_status));
    if (currentCase.heidi.used === "yes") checks.push(Boolean(currentCase.heidi.output_available), Boolean(currentCase.heidi.reviewed_by_clinician));
    if (currentCase.risk_context.prior_fragility_fracture) checks.push(Boolean(currentCase.risk_context.last_fracture_site), Boolean(currentCase.risk_context.last_fracture_month));
    if (currentCase.risk_context.glucocorticoids) checks.push(currentCase.risk_context.glucocorticoid_prednisolone_mg_day !== null, currentCase.risk_context.glucocorticoid_duration_months !== null);
    const completed = checks.filter(Boolean).length;
    return Math.round((completed / checks.length) * 100);
  }

  function updateProgress() {
    const progress = calculateProgress();
    el.progressFill.style.width = `${progress}%`;
    el.progressText.textContent = `${progress}%`;
  }

  function saveDraft(showStatus = true) {
    snapshotInputs();
    currentCase.updated_at = new Date().toISOString();
    const cases = getStore();
    const index = cases.findIndex((item) => item.internal_uuid === currentCase.internal_uuid);
    if (index >= 0) cases[index] = currentCase; else cases.push(currentCase);
    setStore(cases);
    localStorage.setItem(ACTIVE_KEY, currentCase.internal_uuid);
    dirty = false;
    if (showStatus) {
      const time = new Intl.DateTimeFormat("el-GR", { hour: "2-digit", minute: "2-digit" }).format(new Date());
      el.draftStatus.textContent = `Draft αποθηκεύτηκε τοπικά στις ${time}`;
    }
    renderCaseList();
  }

  function normalizeLoadedCase(found) {
    const base = createEmptyCase(found.case_sequence_number || 1);
    return {
      ...base,
      ...found,
      anthropometrics: { ...base.anthropometrics, ...(found.anthropometrics || {}) },
      risk_context: { ...base.risk_context, ...(found.risk_context || {}), derived: { ...(found.risk_context?.derived || {}) } },
      heidi: { ...base.heidi, ...(found.heidi || {}) }
    };
  }

  function loadCase(uuid) {
    const found = getStore().find((item) => item.internal_uuid === uuid);
    if (!found) return;
    currentCase = normalizeLoadedCase(found);
    dirty = false;
    localStorage.setItem(ACTIVE_KEY, uuid);
    syncUiFromState();
    el.draftStatus.textContent = "Φορτώθηκε τοπικό draft";
    if (el.casesDialog.open) el.casesDialog.close();
    switchStep(1);
  }

  function renderCaseList() {
    const cases = getStore().sort((a, b) => Number(a.case_sequence_number) - Number(b.case_sequence_number));
    if (!cases.length) {
      el.caseList.innerHTML = '<div class="placeholder-card"><p>Δεν υπάρχουν αποθηκευμένα drafts σε αυτόν τον browser.</p></div>';
      return;
    }
    el.caseList.innerHTML = cases.map((item) => {
      const updated = item.updated_at ? new Date(item.updated_at).toLocaleString("el-GR") : "—";
      const relationship = item.patient_relationship_status === "new_to_service" ? "Νέα" : item.patient_relationship_status === "established_patient" ? "Υφιστάμενη" : "—";
      return `<div class="case-list-item"><div><strong>${safeText(item.case_id)}</strong><span>${safeText(item.encounter_date || "χωρίς ημερομηνία")} · ${relationship} · ενημέρωση ${safeText(updated)}</span></div><div class="case-list-actions"><button type="button" data-load-case="${safeText(item.internal_uuid)}">Άνοιγμα</button><button type="button" data-delete-case="${safeText(item.internal_uuid)}">Διαγραφή</button></div></div>`;
    }).join("");
  }

  function deleteCase(uuid) {
    const target = getStore().find((item) => item.internal_uuid === uuid);
    if (!window.confirm(`Να διαγραφεί οριστικά το τοπικό draft ${target?.case_id || ""};`)) return;
    setStore(getStore().filter((item) => item.internal_uuid !== uuid));
    if (currentCase.internal_uuid === uuid) newCase(false);
    renderCaseList();
  }

  function newCase(confirmUnsaved = true) {
    if (confirmUnsaved && dirty && !window.confirm("Υπάρχουν μη αποθηκευμένες αλλαγές. Να δημιουργηθεί νέο case;")) return;
    currentCase = createEmptyCase(nextSequence());
    bmiAutocalculated = false;
    localStorage.setItem(ACTIVE_KEY, currentCase.internal_uuid);
    dirty = false;
    syncUiFromState();
    el.draftStatus.textContent = "Νέο pilot case — δεν έχει αποθηκευτεί";
    switchStep(1);
  }

  function syncUiFromState() {
    el.caseIdDisplay.textContent = currentCase.case_id;
    el.encounterDate.value = currentCase.encounter_date || isoToday();
    el.ageYears.value = currentCase.age_years ?? "";
    el.encounterArchetype.value = currentCase.encounter_archetype || "";

    const a = currentCase.anthropometrics;
    el.weightKg.value = a.weight_kg ?? "";
    el.currentHeightCm.value = a.current_height_cm ?? "";
    el.heightSource.value = a.height_source || "";
    el.referenceHeightCm.value = a.reference_height_cm ?? "";
    el.bmi.value = a.bmi ?? "";
    bmiAutocalculated = a.bmi_source === "calculated_weight_height";

    [
      ["sex", currentCase.sex],
      ["patient_relationship_status", currentCase.patient_relationship_status],
      ["first_core_baseline_encounter_for_patient", currentCase.first_core_baseline_encounter_for_patient],
      ["osteoporosis_status", currentCase.osteoporosis_status],
      ["menopause_status", currentCase.menopause_status],
      ["heidi_used", currentCase.heidi.used]
    ].forEach(([field, value]) => setSelected(field, value || "", false));

    const risk = currentCase.risk_context;
    el.priorFragilityFracture.checked = risk.prior_fragility_fracture;
    el.lastFractureSite.value = risk.last_fracture_site || "";
    el.lastFractureMonth.value = risk.last_fracture_month || "";
    el.parentalHipFracture.checked = risk.parental_hip_fracture;
    el.frailtyImmobility.checked = risk.frailty_or_immobility;
    el.cfsScore.value = risk.cfs_score ?? "";
    el.cognitiveImpairment.checked = risk.cognitive_impairment;
    el.significantImmobility.checked = risk.significant_immobility;
    el.glucocorticoids.checked = risk.glucocorticoids;
    el.gcDoseMg.value = risk.glucocorticoid_prednisolone_mg_day ?? "";
    el.gcDurationMonths.value = risk.glucocorticoid_duration_months ?? "";
    el.fallsLast12m.value = risk.falls_last_12_months ?? "";
    el.secondaryContext.checked = risk.secondary_context;
    $$('input[name="secondaryCondition"]').forEach((checkbox) => checkbox.checked = risk.secondary_conditions.includes(checkbox.value));
    el.sarcopeniaRelevant.checked = risk.sarcopenia_case_finding_relevant;
    el.sarcopeniaMethod.value = risk.sarcopenia_screen_method || "";
    el.sarcFScore.value = risk.sarc_f_score ?? "";

    el.heidiOutput.value = currentCase.heidi.output_available || "";
    el.heidiReviewed.value = currentCase.heidi.reviewed_by_clinician || "";
    el.heidiCorrection.value = currentCase.heidi.material_correction_required || "";
    $$('#heidiCategoryChecks input[type="checkbox"]').forEach((checkbox) => checkbox.checked = currentCase.heidi.correction_categories.includes(checkbox.value));
    el.quickNotes.value = currentCase.quick_notes || "";

    syncSexDependentUi();
    syncRiskDetailUi();
    syncHeidiUi();
    recalculateDerived();
    updatePilotPill();
    updateProgress();
  }

  function updatePilotPill() {
    const seq = Math.max(1, Number(currentCase.case_sequence_number || 1));
    el.pilotPill.textContent = `PILOT CASE ${seq}/${PILOT_TARGET}`;
  }

  function switchStep(step) {
    activeStep = Number(step);
    $$(".step-tab").forEach((button) => button.classList.toggle("active", Number(button.dataset.step) === activeStep));
    $$(".step-panel").forEach((panel) => panel.classList.toggle("active", Number(panel.dataset.stepPanel) === activeStep));
    el.nextBtn.textContent = activeStep >= 6 ? "Τέλος →" : "Επόμενο →";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function bindChoiceButtons() {
    $$('[data-field][data-value]').forEach((button) => button.addEventListener("click", () => setSelected(button.dataset.field, button.dataset.value)));
  }

  function bindInputs() {
    [el.encounterDate, el.ageYears, el.encounterArchetype, el.weightKg, el.currentHeightCm, el.heightSource, el.referenceHeightCm, el.bmi, el.lastFractureSite, el.lastFractureMonth, el.cfsScore, el.gcDoseMg, el.gcDurationMonths, el.fallsLast12m, el.sarcFScore, el.quickNotes].forEach((input) => {
      input.addEventListener("input", () => {
        if (input === el.bmi) bmiAutocalculated = false;
        snapshotInputs();
        markDirty();
      });
      input.addEventListener("change", () => {
        snapshotInputs();
        markDirty();
      });
    });

    el.heidiOutput.addEventListener("change", () => { currentCase.heidi.output_available = el.heidiOutput.value; markDirty(); });
    el.heidiReviewed.addEventListener("change", () => { currentCase.heidi.reviewed_by_clinician = el.heidiReviewed.value; markDirty(); });
    el.heidiCorrection.addEventListener("change", () => { currentCase.heidi.material_correction_required = el.heidiCorrection.value; syncHeidiCorrectionUi(); markDirty(); });
    el.sarcopeniaMethod.addEventListener("change", () => { currentCase.risk_context.sarcopenia_screen_method = el.sarcopeniaMethod.value; syncRiskDetailUi(); markDirty(); });

    const toggles = [
      [el.priorFragilityFracture, "prior_fragility_fracture", "fracture"],
      [el.parentalHipFracture, "parental_hip_fracture", null],
      [el.frailtyImmobility, "frailty_or_immobility", "frailty"],
      [el.glucocorticoids, "glucocorticoids", "gc"],
      [el.secondaryContext, "secondary_context", "secondary"],
      [el.sarcopeniaRelevant, "sarcopenia_case_finding_relevant", "sarcopenia"]
    ];
    toggles.forEach(([input, field, clearKey]) => input.addEventListener("change", () => {
      currentCase.risk_context[field] = input.checked;
      if (!input.checked && clearKey) clearDependentRiskFields(clearKey);
      syncRiskDetailUi();
      markDirty();
    }));

    [el.cognitiveImpairment, el.significantImmobility].forEach((input) => input.addEventListener("change", () => { snapshotInputs(); markDirty(); }));
    $$('input[name="secondaryCondition"]').forEach((checkbox) => checkbox.addEventListener("change", () => { snapshotInputs(); markDirty(); }));
    $$('#heidiCategoryChecks input[type="checkbox"]').forEach((checkbox) => checkbox.addEventListener("change", () => { snapshotInputs(); markDirty(); }));
  }

  function bindNavigation() {
    $$(".step-tab").forEach((button) => button.addEventListener("click", () => switchStep(button.dataset.step)));
    el.nextBtn.addEventListener("click", () => {
      if (activeStep < 6) switchStep(activeStep + 1);
      else window.alert("Η τελική ολοκλήρωση θα ενεργοποιηθεί όταν υλοποιηθούν όλα τα Steps. Το Step 1 αποθηκεύεται ήδη ως pilot draft.");
    });
    el.saveTopBtn.addEventListener("click", () => saveDraft());
    el.saveDraftBtn.addEventListener("click", () => saveDraft());
    el.finishVisitBtn.addEventListener("click", () => {
      saveDraft();
      window.alert("Το Step 1 αποθηκεύτηκε. Το case δεν χαρακτηρίζεται ακόμη complete μέχρι να υλοποιηθούν Steps 2–6.");
    });
    el.cancelCaseBtn.addEventListener("click", () => {
      if (!window.confirm("Να καθαριστεί το τρέχον case από την οθόνη; Το αποθηκευμένο draft θα παραμείνει στα Cases.")) return;
      newCase(false);
    });
    $$('[data-nav-action="new-case"]').forEach((button) => button.addEventListener("click", () => newCase(true)));
    $$('[data-nav-action="cases"]').forEach((button) => button.addEventListener("click", () => { renderCaseList(); el.casesDialog.showModal(); }));
    $$('[data-nav-action="privacy"]').forEach((button) => button.addEventListener("click", () => el.privacyDialog.showModal()));
    $$('[data-nav-action="heidi"]').forEach((button) => button.addEventListener("click", () => { switchStep(1); document.querySelector(".heidi-card")?.scrollIntoView({ behavior: "smooth", block: "center" }); }));
    el.caseList.addEventListener("click", (event) => {
      const loadButton = event.target.closest("[data-load-case]");
      const deleteButton = event.target.closest("[data-delete-case]");
      if (loadButton) loadCase(loadButton.dataset.loadCase);
      if (deleteButton) deleteCase(deleteButton.dataset.deleteCase);
    });
  }

  function setupPrivacy() {
    el.privacyStrip.hidden = localStorage.getItem(PRIVACY_KEY) === "yes";
    el.dismissPrivacyBtn.addEventListener("click", () => { localStorage.setItem(PRIVACY_KEY, "yes"); el.privacyStrip.hidden = true; });
  }

  function restoreActiveCase() {
    const activeUuid = localStorage.getItem(ACTIVE_KEY);
    const found = getStore().find((item) => item.internal_uuid === activeUuid);
    if (found) {
      currentCase = normalizeLoadedCase(found);
      el.draftStatus.textContent = "Φορτώθηκε το τελευταίο τοπικό draft";
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

  bindChoiceButtons();
  bindInputs();
  bindNavigation();
  setupPrivacy();
  restoreActiveCase();
  syncUiFromState();
  renderCaseList();
})();
