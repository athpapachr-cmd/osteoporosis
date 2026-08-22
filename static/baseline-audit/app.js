(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const PRIVACY_KEY = "osteoporosis.baselineAuditPilot.privacyDismissed";
  const PILOT_TARGET = 5;
  const LOCAL_LOW_BMI_THRESHOLD = 20;

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const numberOrNull = (value) => value === "" || value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value);

  function injectStep2Assets() {
    if (!document.querySelector('link[data-step2-style]')) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = "./step2.css";
      link.dataset.step2Style = "true";
      document.head.appendChild(link);
    }

    const panel = $('[data-step-panel="2"]');
    if (!panel) return;
    panel.classList.remove("placeholder-panel");
    panel.innerHTML = `
      <div class="context-note" id="step2ContextNote">
        <strong>Step 2 — Ιστορικό & Κίνδυνος:</strong>
        καταγράφουμε τι ελέγχθηκε και ποιο risk framework χρησιμοποιήθηκε, χωρίς να παράγεται live score ή θεραπευτική καθοδήγηση.
      </div>

      <div class="step2-grid">
        <article class="card step2-card span-2">
          <div class="card-heading"><div><h2>Ιστορικό καταγμάτων</h2><p>Full history, interval update ή focused review ανάλογα με το encounter.</p></div></div>
          <div class="step2-top-grid">
            <label><span>Έγινε review ιστορικού καταγμάτων;</span><select id="fractureHistoryReviewed"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="unknown">Δεν είμαι βέβαιος</option></select></label>
            <label><span>Scope review</span><select id="fractureReviewScope"><option value="">—</option><option value="full_history">Πλήρες ιστορικό</option><option value="interval_update">Update από προηγούμενη επίσκεψη</option><option value="focused_current_fracture">Focused — τρέχον/πρόσφατο κάταγμα</option><option value="not_reviewed">Δεν έγινε review</option></select></label>
            <label><span>Νέο κάταγμα από τελευταίο review;</span><select id="intervalFractureStatus"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="unknown">Άγνωστο</option><option value="not_applicable">Δεν εφαρμόζεται</option></select></label>
          </div>
          <div class="fracture-events-head"><div><strong>Structured fracture events</strong><span>Πρόσθεσε όσα γεγονότα ήταν κλινικά σχετικά με τη σημερινή αξιολόγηση.</span></div><button class="btn secondary" type="button" id="addFractureEventBtn">＋ Κάταγμα</button></div>
          <div id="fractureEvents" class="fracture-events"></div>
          <div class="empty-events" id="fractureEventsEmpty">Δεν έχουν προστεθεί structured fracture events.</div>
        </article>

        <article class="card step2-card">
          <div class="card-heading"><div><h2>FRAX / formal risk assessment</h2><p>Καταγράφεται η πραγματική χρήση του εργαλείου, όχι internal surrogate score.</p></div></div>
          <div class="field-stack">
            <label><span>Formal risk assessment ενδείκνυτο σήμερα;</span><select id="formalRiskIndicated"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option></select></label>
            <label><span>Έγινε formal assessment;</span><select id="formalRiskDone"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">Δεν εφαρμόζεται</option></select></label>
          </div>
          <div id="fraxDetails" class="frax-details" hidden>
            <div class="mini-grid">
              <label><span>Εργαλείο</span><select id="riskToolName"><option value="frax">FRAX</option><option value="fraxplus">FRAXplus</option><option value="other">Άλλο</option></select></label>
              <label><span>Country / surrogate model</span><input id="fraxCountryModel" type="text" maxlength="80" placeholder="π.χ. UK / Greece / άλλο" /></label>
              <label><span>FN BMD χρησιμοποιήθηκε;</span><select id="femoralNeckBmdUsed"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="unknown">Άγνωστο</option><option value="not_applicable">N/A</option></select></label>
              <label><span>MOF 10-year %</span><input id="fraxMof" type="number" step="0.1" min="0" max="100" /></label>
              <label><span>Hip 10-year %</span><input id="fraxHip" type="number" step="0.1" min="0" max="100" /></label>
            </div>
          </div>
        </article>

        <article class="card step2-card">
          <div class="card-heading"><div><h2>FRAX inputs / context που ελέγχθηκαν</h2><p>Μόνο όσα χρειάζονται για αναπαραγωγιμότητα του formal assessment.</p></div></div>
          <div class="step2-top-grid compact-grid">
            <label><span>Κάπνισμα τώρα</span><select id="currentSmoking"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="unknown">Άγνωστο</option></select></label>
            <label><span>Alcohol ≥3 units/day</span><select id="highAlcohol"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="unknown">Άγνωστο</option></select></label>
            <label><span>Ρευματοειδής αρθρίτιδα</span><select id="rheumatoidArthritis"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="unknown">Άγνωστο</option></select></label>
          </div>
          <div class="source-summary" id="step1RiskSummary"></div>
        </article>

        <article class="card step2-card span-2">
          <div class="card-heading"><div><h2>Framework, κατηγορία κινδύνου & contextual adjustment</h2><p>Το framework δηλώνεται ρητά. Δεν γίνεται silent hybridization διαφορετικών thresholds.</p></div></div>
          <div class="step2-top-grid four-cols">
            <label><span>Declared framework</span><select id="declaredRiskFramework"><option value="">—</option><option value="nogg_2024">NOGG 2024</option><option value="aace_2020">AACE 2020</option><option value="iof_esceo">IOF / ESCEO</option><option value="local_cy">Local / Cyprus protocol</option><option value="other">Άλλο</option><option value="none_declared">Δεν δηλώθηκε</option></select></label>
            <label><span>Resulting risk category</span><select id="resultingRiskCategory"><option value="">—</option><option value="low">Low</option><option value="intermediate">Intermediate</option><option value="high">High</option><option value="very_high">Very high</option><option value="uncertain">Uncertain</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Contextual adjustment / override;</span><select id="contextualAdjustment"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option></select></label>
          </div>
          <div id="adjustmentDetails" class="adjustment-details" hidden>
            <span class="field-label">Reason(s)</span>
            <div class="chip-checks" id="adjustmentReasonChecks">
              <label><input type="checkbox" value="recent_fracture" />Recent fracture</label>
              <label><input type="checkbox" value="recurrent_falls" />Recurrent falls</label>
              <label><input type="checkbox" value="glucocorticoid_dose" />GC dose</label>
              <label><input type="checkbox" value="type2_diabetes" />ΣΔ2</label>
              <label><input type="checkbox" value="spine_hip_discordance" />Spine–hip discordance</label>
              <label><input type="checkbox" value="tbs_or_other_bone_quality" />TBS / bone quality</label>
              <label><input type="checkbox" value="frailty_or_clinical_context" />Frailty / clinical context</label>
              <label><input type="checkbox" value="other" />Άλλο</label>
            </div>
            <label class="override-note"><span>Σύντομος λόγος override / adjustment <small>(προαιρετικό)</small></span><textarea id="riskOverrideReason" rows="2" maxlength="600" placeholder="Χωρίς αναγνωριστικά στοιχεία"></textarea></label>
          </div>
        </article>
      </div>`;
  }

  injectStep2Assets();

  const el = {
    pilotPill: $("#pilotPill"), caseIdDisplay: $("#caseIdDisplay"), encounterDate: $("#encounterDate"), ageYears: $("#ageYears"), encounterArchetype: $("#encounterArchetype"),
    weightKg: $("#weightKg"), currentHeightCm: $("#currentHeightCm"), heightSource: $("#heightSource"), referenceHeightCm: $("#referenceHeightCm"), bmi: $("#bmi"), heightLossDisplay: $("#heightLossDisplay"), menopauseBlock: $("#menopauseBlock"),
    priorFragilityFracture: $("#priorFragilityFracture"), fractureDetails: $("#fractureDetails"), lastFractureSite: $("#lastFractureSite"), lastFractureMonth: $("#lastFractureMonth"), parentalHipFracture: $("#parentalHipFracture"),
    frailtyImmobility: $("#frailtyImmobility"), frailtyDetails: $("#frailtyDetails"), cfsScore: $("#cfsScore"), cognitiveImpairment: $("#cognitiveImpairment"), significantImmobility: $("#significantImmobility"),
    glucocorticoids: $("#glucocorticoids"), gcDetails: $("#gcDetails"), gcDoseMg: $("#gcDoseMg"), gcDurationMonths: $("#gcDurationMonths"), fallsLast12m: $("#fallsLast12m"),
    secondaryContext: $("#secondaryContext"), secondaryDetails: $("#secondaryDetails"), sarcopeniaRelevant: $("#sarcopeniaRelevant"), sarcopeniaDetails: $("#sarcopeniaDetails"), sarcopeniaMethod: $("#sarcopeniaMethod"), sarcFScoreWrap: $("#sarcFScoreWrap"), sarcFScore: $("#sarcFScore"),
    heidiOutput: $("#heidiOutput"), heidiReviewed: $("#heidiReviewed"), heidiCorrection: $("#heidiCorrection"), heidiCorrectionCategories: $("#heidiCorrectionCategories"), quickNotes: $("#quickNotes"),
    progressFill: $("#progressFill"), progressText: $("#progressText"), draftStatus: $("#draftStatus"), saveTopBtn: $("#saveTopBtn"), saveDraftBtn: $("#saveDraftBtn"), nextBtn: $("#nextBtn"), finishVisitBtn: $("#finishVisitBtn"), cancelCaseBtn: $("#cancelCaseBtn"),
    casesDialog: $("#casesDialog"), privacyDialog: $("#privacyDialog"), caseList: $("#caseList"), privacyStrip: $("#privacyStrip"), dismissPrivacyBtn: $("#dismissPrivacyBtn"),
    step2ContextNote: $("#step2ContextNote"), fractureHistoryReviewed: $("#fractureHistoryReviewed"), fractureReviewScope: $("#fractureReviewScope"), intervalFractureStatus: $("#intervalFractureStatus"), fractureEvents: $("#fractureEvents"), fractureEventsEmpty: $("#fractureEventsEmpty"), addFractureEventBtn: $("#addFractureEventBtn"),
    formalRiskIndicated: $("#formalRiskIndicated"), formalRiskDone: $("#formalRiskDone"), fraxDetails: $("#fraxDetails"), riskToolName: $("#riskToolName"), fraxCountryModel: $("#fraxCountryModel"), femoralNeckBmdUsed: $("#femoralNeckBmdUsed"), fraxMof: $("#fraxMof"), fraxHip: $("#fraxHip"),
    currentSmoking: $("#currentSmoking"), highAlcohol: $("#highAlcohol"), rheumatoidArthritis: $("#rheumatoidArthritis"), step1RiskSummary: $("#step1RiskSummary"), declaredRiskFramework: $("#declaredRiskFramework"), resultingRiskCategory: $("#resultingRiskCategory"), contextualAdjustment: $("#contextualAdjustment"), adjustmentDetails: $("#adjustmentDetails"), riskOverrideReason: $("#riskOverrideReason")
  };

  let currentCase = createEmptyCase(1);
  let dirty = false;
  let activeStep = 1;
  let bmiAutocalculated = false;

  function isoToday() { const now = new Date(); const offset = now.getTimezoneOffset(); return new Date(now.getTime() - offset * 60_000).toISOString().slice(0, 10); }
  function createUuid() { return window.crypto?.randomUUID ? window.crypto.randomUUID() : `case-${Date.now()}-${Math.random().toString(16).slice(2)}`; }
  function getStore() { try { const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); return Array.isArray(parsed) ? parsed : []; } catch { return []; } }
  function setStore(cases) { localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
  function nextSequence() { const used = getStore().map(i => Number(i.case_sequence_number || 0)).filter(Number.isFinite); return used.length ? Math.max(...used) + 1 : 1; }
  function safeText(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }

  function createEmptyCase(sequence) {
    const seq = Number(sequence || 1);
    return {
      schema: "baseline_osteoporosis_case_form_v1", schema_version: "1.2-steps1-2", baseline_phase: "pilot", internal_uuid: createUuid(), case_sequence_number: seq, case_id: `PILOT-${String(seq).padStart(3, "0")}`, local_patient_token: createUuid(), encounter_date: isoToday(), age_years: null, sex: "", menopause_status: "", patient_relationship_status: "", encounter_archetype: "", first_core_baseline_encounter_for_patient: "", osteoporosis_status: "",
      anthropometrics: { weight_kg: null, current_height_cm: null, height_source: "", reference_height_cm: null, bmi: null, bmi_source: "", derived_height_loss_cm: null },
      risk_context: { prior_fragility_fracture: false, last_fracture_site: "", last_fracture_month: "", parental_hip_fracture: false, frailty_or_immobility: false, cfs_score: null, cognitive_impairment: false, significant_immobility: false, glucocorticoids: false, glucocorticoid_prednisolone_mg_day: null, glucocorticoid_duration_months: null, falls_last_12_months: null, secondary_context: false, secondary_conditions: [], sarcopenia_case_finding_relevant: false, sarcopenia_screen_method: "", sarc_f_score: null, derived: {} },
      heidi: { used: "", output_available: "", reviewed_by_clinician: "", material_correction_required: "", correction_categories: [] },
      fracture_history: { reviewed: "", review_scope: "", interval_fracture_status: "", events: [] },
      risk_assessment: { formal_indicated: "", formal_done: "", tool_name: "frax", country_or_surrogate_model: "", frax_mof_percent: null, frax_hip_percent: null, femoral_neck_bmd_used: "", current_smoking: "", high_alcohol_3_units_day: "", rheumatoid_arthritis: "", declared_framework: "", resulting_risk_category: "", contextual_adjustment: "", adjustment_reasons: [], override_reason: "" },
      quick_notes: "", created_at: new Date().toISOString(), updated_at: null, implementation_slice: "steps_1_2"
    };
  }

  function normalizeLoadedCase(found) {
    const base = createEmptyCase(found.case_sequence_number || 1);
    return { ...base, ...found, anthropometrics: { ...base.anthropometrics, ...(found.anthropometrics || {}) }, risk_context: { ...base.risk_context, ...(found.risk_context || {}), derived: { ...(found.risk_context?.derived || {}) } }, heidi: { ...base.heidi, ...(found.heidi || {}) }, fracture_history: { ...base.fracture_history, ...(found.fracture_history || {}), events: [...(found.fracture_history?.events || [])] }, risk_assessment: { ...base.risk_assessment, ...(found.risk_assessment || {}), adjustment_reasons: [...(found.risk_assessment?.adjustment_reasons || [])] } };
  }

  function markDirty() { dirty = true; el.draftStatus.textContent = "Υπάρχουν μη αποθηκευμένες αλλαγές"; snapshotAll(); updateProgress(); }
  function setSelected(field, value, mark = true) {
    if (field === "heidi_used") currentCase.heidi.used = value; else currentCase[field] = value;
    $$(`[data-field="${field}"][data-value]`).forEach(button => { const selected = button.dataset.value === value; button.classList.toggle("selected", selected); button.setAttribute("aria-pressed", selected ? "true" : "false"); });
    if (field === "sex") syncSexUi(); if (field === "heidi_used") syncHeidiUi(); if (mark) markDirty();
  }

  function syncSexUi() { const female = currentCase.sex === "female"; el.menopauseBlock.hidden = !female; if (!female) { currentCase.menopause_status = ""; $$('[data-field="menopause_status"]').forEach(b => b.classList.remove("selected")); } }
  function syncHeidiUi() { const used = currentCase.heidi.used === "yes"; $$('[data-heidi-dependent]').forEach(f => f.disabled = !used); if (!used) { currentCase.heidi.output_available = ""; currentCase.heidi.reviewed_by_clinician = ""; currentCase.heidi.material_correction_required = ""; currentCase.heidi.correction_categories = []; el.heidiOutput.value = ""; el.heidiReviewed.value = ""; el.heidiCorrection.value = ""; } syncHeidiCorrectionUi(); }
  function syncHeidiCorrectionUi() { const show = currentCase.heidi.used === "yes" && currentCase.heidi.material_correction_required === "yes"; el.heidiCorrectionCategories.hidden = !show; if (!show) { currentCase.heidi.correction_categories = []; $$('#heidiCategoryChecks input').forEach(c => c.checked = false); } }
  function syncRiskDetailUi() { const r = currentCase.risk_context; el.fractureDetails.hidden = !r.prior_fragility_fracture; el.frailtyDetails.hidden = !r.frailty_or_immobility; el.gcDetails.hidden = !r.glucocorticoids; el.secondaryDetails.hidden = !r.secondary_context; el.sarcopeniaDetails.hidden = !r.sarcopenia_case_finding_relevant; el.sarcFScoreWrap.hidden = !(r.sarcopenia_case_finding_relevant && r.sarcopenia_screen_method === "sarc_f"); }
  function syncStep2Ui() { const ra = currentCase.risk_assessment; el.fraxDetails.hidden = ra.formal_done !== "yes"; el.adjustmentDetails.hidden = ra.contextual_adjustment !== "yes"; updateStep2ContextNote(); renderStep1RiskSummary(); }

  function clearDependentRiskFields(key) {
    const r = currentCase.risk_context;
    if (key === "fracture") { r.last_fracture_site = ""; r.last_fracture_month = ""; el.lastFractureSite.value = ""; el.lastFractureMonth.value = ""; }
    if (key === "frailty") { r.cfs_score = null; r.cognitive_impairment = false; r.significant_immobility = false; el.cfsScore.value = ""; el.cognitiveImpairment.checked = false; el.significantImmobility.checked = false; }
    if (key === "gc") { r.glucocorticoid_prednisolone_mg_day = null; r.glucocorticoid_duration_months = null; el.gcDoseMg.value = ""; el.gcDurationMonths.value = ""; }
    if (key === "secondary") { r.secondary_conditions = []; $$('input[name="secondaryCondition"]').forEach(c => c.checked = false); }
    if (key === "sarcopenia") { r.sarcopenia_screen_method = ""; r.sarc_f_score = null; el.sarcopeniaMethod.value = ""; el.sarcFScore.value = ""; }
  }

  function monthsBetween(yyyyMm, encounterDate) { if (!yyyyMm || !encounterDate) return null; const [y, m] = yyyyMm.split("-").map(Number); const d = new Date(`${encounterDate}T00:00:00`); if (!y || !m || Number.isNaN(d.getTime())) return null; return (d.getFullYear() - y) * 12 + (d.getMonth() - (m - 1)); }
  function recalculateDerived() {
    const a = currentCase.anthropometrics, r = currentCase.risk_context;
    const weight = numberOrNull(el.weightKg.value), height = numberOrNull(el.currentHeightCm.value);
    if (weight !== null && height !== null && height > 0) { a.bmi = Math.round((weight / Math.pow(height / 100, 2)) * 10) / 10; a.bmi_source = "calculated_weight_height"; bmiAutocalculated = true; el.bmi.value = a.bmi.toFixed(1); }
    else if (!bmiAutocalculated) { a.bmi = numberOrNull(el.bmi.value); a.bmi_source = a.bmi === null ? "" : "manual_or_external"; }
    else { bmiAutocalculated = false; a.bmi = numberOrNull(el.bmi.value); a.bmi_source = a.bmi === null ? "" : "manual_or_external"; }
    const refH = numberOrNull(el.referenceHeightCm.value), curH = numberOrNull(el.currentHeightCm.value); a.derived_height_loss_cm = refH !== null && curH !== null ? Math.max(0, Math.round((refH - curH) * 10) / 10) : null; el.heightLossDisplay.textContent = a.derived_height_loss_cm === null ? "—" : `${a.derived_height_loss_cm.toFixed(1)} cm`;
    const fractureAge = monthsBetween(r.last_fracture_month, currentCase.encounter_date), dose = numberOrNull(r.glucocorticoid_prednisolone_mg_day), duration = numberOrNull(r.glucocorticoid_duration_months), falls = numberOrNull(r.falls_last_12_months), sarcF = numberOrNull(r.sarc_f_score);
    let band = null; if (dose !== null) band = dose < 2.5 ? "lt_2_5" : dose < 7.5 ? "2_5_to_lt_7_5" : "gte_7_5";
    r.derived = { low_bmi_workflow_flag: a.bmi !== null ? a.bmi < LOCAL_LOW_BMI_THRESHOLD : null, low_bmi_workflow_threshold: LOCAL_LOW_BMI_THRESHOLD, height_loss_ge_4_cm: a.derived_height_loss_cm !== null ? a.derived_height_loss_cm >= 4 : null, months_since_last_fragility_fracture: fractureAge, fracture_within_24_months: fractureAge !== null ? fractureAge >= 0 && fractureAge <= 24 : null, recent_vertebral_fracture_within_24_months: fractureAge !== null && r.last_fracture_site === "vertebral" ? fractureAge >= 0 && fractureAge <= 24 : null, recurrent_falls_2_or_more_last_12m: falls !== null ? falls >= 2 : null, glucocorticoid_dose_band: band, glucocorticoid_exposure_3m_or_more: duration !== null ? duration >= 3 : null, glucocorticoid_gt_20_mg_day: dose !== null ? dose > 20 : null, sarc_f_positive_ge_4: sarcF !== null ? sarcF >= 4 : null };
  }

  function makeFractureEvent(seed = {}) { return { id: seed.id || createUuid(), site: seed.site || "", month: seed.month || "", low_trauma: seed.low_trauma || "", occurred_on_treatment: seed.occurred_on_treatment || "", vertebral_level: seed.vertebral_level || "" }; }
  function ensureStep1FractureSeed() { if (!currentCase.fracture_history.events.length && currentCase.risk_context.prior_fragility_fracture) currentCase.fracture_history.events.push(makeFractureEvent({ site: currentCase.risk_context.last_fracture_site, month: currentCase.risk_context.last_fracture_month })); }
  function renderFractureEvents() {
    ensureStep1FractureSeed();
    const events = currentCase.fracture_history.events;
    el.fractureEventsEmpty.hidden = events.length > 0;
    el.fractureEvents.innerHTML = events.map((event, idx) => `
      <div class="fracture-event" data-event-id="${safeText(event.id)}">
        <div class="event-number">${idx + 1}</div>
        <label><span>Site</span><select data-event-field="site"><option value="">—</option><option value="vertebral" ${event.site === "vertebral" ? "selected" : ""}>Σπονδυλικό</option><option value="hip" ${event.site === "hip" ? "selected" : ""}>Ισχίο</option><option value="distal_radius" ${event.site === "distal_radius" ? "selected" : ""}>Περιφερική κερκίδα</option><option value="proximal_humerus" ${event.site === "proximal_humerus" ? "selected" : ""}>Εγγύς βραχιόνιο</option><option value="pelvis" ${event.site === "pelvis" ? "selected" : ""}>Πύελος</option><option value="other" ${event.site === "other" ? "selected" : ""}>Άλλο</option></select></label>
        <label><span>Μήνας / έτος</span><input data-event-field="month" type="month" value="${safeText(event.month)}" /></label>
        <label><span>Low trauma / fragility;</span><select data-event-field="low_trauma"><option value="">—</option><option value="yes" ${event.low_trauma === "yes" ? "selected" : ""}>Ναι</option><option value="no" ${event.low_trauma === "no" ? "selected" : ""}>Όχι</option><option value="uncertain" ${event.low_trauma === "uncertain" ? "selected" : ""}>Αβέβαιο</option></select></label>
        <label><span>Υπό θεραπεία;</span><select data-event-field="occurred_on_treatment"><option value="">—</option><option value="yes" ${event.occurred_on_treatment === "yes" ? "selected" : ""}>Ναι</option><option value="no" ${event.occurred_on_treatment === "no" ? "selected" : ""}>Όχι</option><option value="unknown" ${event.occurred_on_treatment === "unknown" ? "selected" : ""}>Άγνωστο</option><option value="not_applicable" ${event.occurred_on_treatment === "not_applicable" ? "selected" : ""}>N/A</option></select></label>
        <label class="vertebral-level ${event.site === "vertebral" ? "" : "is-hidden"}"><span>Vertebral level/type</span><input data-event-field="vertebral_level" type="text" maxlength="40" value="${safeText(event.vertebral_level)}" placeholder="π.χ. L1 / morphometric" /></label>
        <button type="button" class="remove-event" data-remove-event="${safeText(event.id)}" aria-label="Διαγραφή κατάγματος">×</button>
      </div>`).join("");
  }

  function collectFractureEventsFromDom() {
    $$(".fracture-event", el.fractureEvents).forEach(row => {
      const event = currentCase.fracture_history.events.find(e => e.id === row.dataset.eventId); if (!event) return;
      $$('[data-event-field]', row).forEach(field => event[field.dataset.eventField] = field.value);
    });
    if (currentCase.fracture_history.events.length) {
      currentCase.risk_context.prior_fragility_fracture = true;
      const dated = currentCase.fracture_history.events.filter(e => e.month).sort((a, b) => b.month.localeCompare(a.month));
      const latest = dated[0] || currentCase.fracture_history.events[0];
      currentCase.risk_context.last_fracture_site = latest.site || currentCase.risk_context.last_fracture_site;
      currentCase.risk_context.last_fracture_month = latest.month || currentCase.risk_context.last_fracture_month;
    }
  }

  function updateStep2ContextNote() {
    const archetype = currentCase.encounter_archetype;
    const map = {
      initial_assessment_new_or_uncertain_diagnosis: "Αρχική αξιολόγηση: το Step 2 επιτρέπει full fracture-history reconstruction και formal risk capture.",
      initial_assessment_known_osteoporosis_or_osteopenia: "Αρχική αξιολόγηση γνωστής νόσου: καταγράφεται το προϋπάρχον history και η τρέχουσα risk interpretation.",
      routine_followup_stable: "Stable follow-up: μπορείς να δηλώσεις interval update και formal reassessment ως not applicable όταν πράγματι δεν έγινε/δεν χρειαζόταν.",
      treatment_start: "Έναρξη θεραπείας: καταγράφεται το risk framework που στήριξε την απόφαση.",
      treatment_continuation_or_due_monitoring: "Continuation/monitoring: καταγράφεται targeted risk update χωρίς υποχρεωτική επανάληψη ολόκληρου FRAX αν δεν έγινε.",
      treatment_change_or_transition: "Treatment transition: καταγράφεται το τρέχον risk context και το declared framework πριν από sequencing decisions.",
      post_fragility_fracture: "Post-fragility fracture: δίνεται έμφαση σε ακριβή fracture event/recency και risk reassessment.",
      fracture_on_treatment: "Fracture on treatment: το event συνδέεται με treatment exposure και επαναξιολόγηση κινδύνου.",
      adverse_effect_or_intolerance: "Adverse effect visit: formal risk reassessment μπορεί να είναι targeted ή not applicable· η φόρμα καταγράφει τι έγινε.",
      treatment_completion_or_consolidation: "Completion/consolidation: καταγράφεται το risk context που χρησιμοποιήθηκε για exit/consolidation planning."
    };
    const text = map[archetype] || "Καταγράφουμε τι ελέγχθηκε και ποιο risk framework χρησιμοποιήθηκε, χωρίς live scoring.";
    el.step2ContextNote.innerHTML = `<strong>Step 2 — Ιστορικό & Κίνδυνος:</strong> ${text}`;
  }

  function renderStep1RiskSummary() {
    const r = currentCase.risk_context, a = currentCase.anthropometrics;
    const items = [];
    if (a.bmi !== null) items.push(`BMI ${a.bmi}`);
    if (r.parental_hip_fracture) items.push("γονεϊκό κάταγμα ισχίου");
    if (r.glucocorticoids) items.push("γλυκοκορτικοειδή");
    if (r.falls_last_12_months !== null) items.push(`${r.falls_last_12_months} πτώσεις/12μηνο`);
    if (r.secondary_conditions.length) items.push(`${r.secondary_conditions.length} secondary/context factor(s)`);
    if (r.frailty_or_immobility) items.push("frailty/immobility context");
    el.step1RiskSummary.innerHTML = items.length ? `<strong>Από Step 1:</strong> ${items.map(safeText).join(" · ")}` : `<span>Δεν έχουν καταγραφεί πρόσθετα Step 1 risk-context στοιχεία.</span>`;
  }

  function snapshotStep1() {
    currentCase.encounter_date = el.encounterDate.value || isoToday(); currentCase.age_years = numberOrNull(el.ageYears.value); currentCase.encounter_archetype = el.encounterArchetype.value;
    const a = currentCase.anthropometrics; a.weight_kg = numberOrNull(el.weightKg.value); a.current_height_cm = numberOrNull(el.currentHeightCm.value); a.height_source = el.heightSource.value; a.reference_height_cm = numberOrNull(el.referenceHeightCm.value); if (!bmiAutocalculated) { a.bmi = numberOrNull(el.bmi.value); a.bmi_source = a.bmi === null ? "" : "manual_or_external"; }
    const r = currentCase.risk_context; r.prior_fragility_fracture = el.priorFragilityFracture.checked; r.last_fracture_site = el.lastFractureSite.value; r.last_fracture_month = el.lastFractureMonth.value; r.parental_hip_fracture = el.parentalHipFracture.checked; r.frailty_or_immobility = el.frailtyImmobility.checked; r.cfs_score = numberOrNull(el.cfsScore.value); r.cognitive_impairment = el.cognitiveImpairment.checked; r.significant_immobility = el.significantImmobility.checked; r.glucocorticoids = el.glucocorticoids.checked; r.glucocorticoid_prednisolone_mg_day = numberOrNull(el.gcDoseMg.value); r.glucocorticoid_duration_months = numberOrNull(el.gcDurationMonths.value); r.falls_last_12_months = numberOrNull(el.fallsLast12m.value); r.secondary_context = el.secondaryContext.checked; r.secondary_conditions = $$('input[name="secondaryCondition"]:checked').map(i => i.value); r.sarcopenia_case_finding_relevant = el.sarcopeniaRelevant.checked; r.sarcopenia_screen_method = el.sarcopeniaMethod.value; r.sarc_f_score = numberOrNull(el.sarcFScore.value);
    currentCase.heidi.output_available = el.heidiOutput.value; currentCase.heidi.reviewed_by_clinician = el.heidiReviewed.value; currentCase.heidi.material_correction_required = el.heidiCorrection.value; currentCase.heidi.correction_categories = $$('#heidiCategoryChecks input:checked').map(i => i.value); currentCase.quick_notes = el.quickNotes.value.trim(); recalculateDerived();
  }

  function snapshotStep2() {
    collectFractureEventsFromDom();
    const fh = currentCase.fracture_history; fh.reviewed = el.fractureHistoryReviewed.value; fh.review_scope = el.fractureReviewScope.value; fh.interval_fracture_status = el.intervalFractureStatus.value;
    const ra = currentCase.risk_assessment; ra.formal_indicated = el.formalRiskIndicated.value; ra.formal_done = el.formalRiskDone.value; ra.tool_name = el.riskToolName.value; ra.country_or_surrogate_model = el.fraxCountryModel.value.trim(); ra.femoral_neck_bmd_used = el.femoralNeckBmdUsed.value; ra.frax_mof_percent = numberOrNull(el.fraxMof.value); ra.frax_hip_percent = numberOrNull(el.fraxHip.value); ra.current_smoking = el.currentSmoking.value; ra.high_alcohol_3_units_day = el.highAlcohol.value; ra.rheumatoid_arthritis = el.rheumatoidArthritis.value; ra.declared_framework = el.declaredRiskFramework.value; ra.resulting_risk_category = el.resultingRiskCategory.value; ra.contextual_adjustment = el.contextualAdjustment.value; ra.adjustment_reasons = $$('#adjustmentReasonChecks input:checked').map(i => i.value); ra.override_reason = el.riskOverrideReason.value.trim();
  }

  function snapshotAll() { snapshotStep1(); snapshotStep2(); }

  function calculateProgress() {
    const checks = [Boolean(currentCase.encounter_date), currentCase.age_years !== null, Boolean(currentCase.sex), Boolean(currentCase.patient_relationship_status), Boolean(currentCase.encounter_archetype), Boolean(currentCase.first_core_baseline_encounter_for_patient), Boolean(currentCase.osteoporosis_status), Boolean(currentCase.heidi.used), Boolean(currentCase.fracture_history.reviewed), Boolean(currentCase.fracture_history.review_scope), Boolean(currentCase.risk_assessment.formal_indicated)];
    if (currentCase.sex === "female") checks.push(Boolean(currentCase.menopause_status));
    if (currentCase.heidi.used === "yes") checks.push(Boolean(currentCase.heidi.output_available), Boolean(currentCase.heidi.reviewed_by_clinician));
    if (currentCase.risk_context.prior_fragility_fracture) checks.push(Boolean(currentCase.risk_context.last_fracture_site), Boolean(currentCase.risk_context.last_fracture_month));
    if (currentCase.risk_context.glucocorticoids) checks.push(currentCase.risk_context.glucocorticoid_prednisolone_mg_day !== null, currentCase.risk_context.glucocorticoid_duration_months !== null);
    if (currentCase.risk_assessment.formal_done === "yes") checks.push(Boolean(currentCase.risk_assessment.tool_name), Boolean(currentCase.risk_assessment.country_or_surrogate_model), currentCase.risk_assessment.frax_mof_percent !== null, currentCase.risk_assessment.frax_hip_percent !== null, Boolean(currentCase.risk_assessment.declared_framework));
    return Math.round((checks.filter(Boolean).length / checks.length) * 100);
  }
  function updateProgress() { const p = calculateProgress(); el.progressFill.style.width = `${p}%`; el.progressText.textContent = `${p}%`; }

  function saveDraft(showStatus = true) { snapshotAll(); currentCase.updated_at = new Date().toISOString(); const cases = getStore(); const idx = cases.findIndex(i => i.internal_uuid === currentCase.internal_uuid); if (idx >= 0) cases[idx] = currentCase; else cases.push(currentCase); setStore(cases); localStorage.setItem(ACTIVE_KEY, currentCase.internal_uuid); dirty = false; if (showStatus) { const time = new Intl.DateTimeFormat("el-GR", { hour: "2-digit", minute: "2-digit" }).format(new Date()); el.draftStatus.textContent = `Draft αποθηκεύτηκε τοπικά στις ${time}`; } renderCaseList(); }

  function syncUiFromState() {
    el.caseIdDisplay.textContent = currentCase.case_id; el.encounterDate.value = currentCase.encounter_date || isoToday(); el.ageYears.value = currentCase.age_years ?? ""; el.encounterArchetype.value = currentCase.encounter_archetype || "";
    const a = currentCase.anthropometrics; el.weightKg.value = a.weight_kg ?? ""; el.currentHeightCm.value = a.current_height_cm ?? ""; el.heightSource.value = a.height_source || ""; el.referenceHeightCm.value = a.reference_height_cm ?? ""; el.bmi.value = a.bmi ?? ""; bmiAutocalculated = a.bmi_source === "calculated_weight_height";
    [["sex", currentCase.sex], ["patient_relationship_status", currentCase.patient_relationship_status], ["first_core_baseline_encounter_for_patient", currentCase.first_core_baseline_encounter_for_patient], ["osteoporosis_status", currentCase.osteoporosis_status], ["menopause_status", currentCase.menopause_status], ["heidi_used", currentCase.heidi.used]].forEach(([field, value]) => setSelected(field, value || "", false));
    const r = currentCase.risk_context; el.priorFragilityFracture.checked = r.prior_fragility_fracture; el.lastFractureSite.value = r.last_fracture_site || ""; el.lastFractureMonth.value = r.last_fracture_month || ""; el.parentalHipFracture.checked = r.parental_hip_fracture; el.frailtyImmobility.checked = r.frailty_or_immobility; el.cfsScore.value = r.cfs_score ?? ""; el.cognitiveImpairment.checked = r.cognitive_impairment; el.significantImmobility.checked = r.significant_immobility; el.glucocorticoids.checked = r.glucocorticoids; el.gcDoseMg.value = r.glucocorticoid_prednisolone_mg_day ?? ""; el.gcDurationMonths.value = r.glucocorticoid_duration_months ?? ""; el.fallsLast12m.value = r.falls_last_12_months ?? ""; el.secondaryContext.checked = r.secondary_context; $$('input[name="secondaryCondition"]').forEach(c => c.checked = r.secondary_conditions.includes(c.value)); el.sarcopeniaRelevant.checked = r.sarcopenia_case_finding_relevant; el.sarcopeniaMethod.value = r.sarcopenia_screen_method || ""; el.sarcFScore.value = r.sarc_f_score ?? "";
    el.heidiOutput.value = currentCase.heidi.output_available || ""; el.heidiReviewed.value = currentCase.heidi.reviewed_by_clinician || ""; el.heidiCorrection.value = currentCase.heidi.material_correction_required || ""; $$('#heidiCategoryChecks input').forEach(c => c.checked = currentCase.heidi.correction_categories.includes(c.value)); el.quickNotes.value = currentCase.quick_notes || "";
    const fh = currentCase.fracture_history; el.fractureHistoryReviewed.value = fh.reviewed || ""; el.fractureReviewScope.value = fh.review_scope || ""; el.intervalFractureStatus.value = fh.interval_fracture_status || ""; renderFractureEvents();
    const ra = currentCase.risk_assessment; el.formalRiskIndicated.value = ra.formal_indicated || ""; el.formalRiskDone.value = ra.formal_done || ""; el.riskToolName.value = ra.tool_name || "frax"; el.fraxCountryModel.value = ra.country_or_surrogate_model || ""; el.femoralNeckBmdUsed.value = ra.femoral_neck_bmd_used || ""; el.fraxMof.value = ra.frax_mof_percent ?? ""; el.fraxHip.value = ra.frax_hip_percent ?? ""; el.currentSmoking.value = ra.current_smoking || ""; el.highAlcohol.value = ra.high_alcohol_3_units_day || ""; el.rheumatoidArthritis.value = ra.rheumatoid_arthritis || ""; el.declaredRiskFramework.value = ra.declared_framework || ""; el.resultingRiskCategory.value = ra.resulting_risk_category || ""; el.contextualAdjustment.value = ra.contextual_adjustment || ""; $$('#adjustmentReasonChecks input').forEach(c => c.checked = ra.adjustment_reasons.includes(c.value)); el.riskOverrideReason.value = ra.override_reason || "";
    syncSexUi(); syncRiskDetailUi(); syncHeidiUi(); syncStep2Ui(); recalculateDerived(); updatePilotPill(); updateProgress();
  }

  function updatePilotPill() { el.pilotPill.textContent = `PILOT CASE ${Math.max(1, Number(currentCase.case_sequence_number || 1))}/${PILOT_TARGET}`; }
  function loadCase(uuid) { const found = getStore().find(i => i.internal_uuid === uuid); if (!found) return; currentCase = normalizeLoadedCase(found); dirty = false; localStorage.setItem(ACTIVE_KEY, uuid); syncUiFromState(); el.draftStatus.textContent = "Φορτώθηκε τοπικό draft"; if (el.casesDialog.open) el.casesDialog.close(); switchStep(1); }
  function deleteCase(uuid) { const target = getStore().find(i => i.internal_uuid === uuid); if (!window.confirm(`Να διαγραφεί οριστικά το τοπικό draft ${target?.case_id || ""};`)) return; setStore(getStore().filter(i => i.internal_uuid !== uuid)); if (currentCase.internal_uuid === uuid) newCase(false); renderCaseList(); }
  function renderCaseList() { const cases = getStore().sort((a, b) => Number(a.case_sequence_number) - Number(b.case_sequence_number)); if (!cases.length) { el.caseList.innerHTML = '<div class="placeholder-card"><p>Δεν υπάρχουν αποθηκευμένα drafts σε αυτόν τον browser.</p></div>'; return; } el.caseList.innerHTML = cases.map(item => { const updated = item.updated_at ? new Date(item.updated_at).toLocaleString("el-GR") : "—"; const relationship = item.patient_relationship_status === "new_to_service" ? "Νέος/α" : item.patient_relationship_status === "established_patient" ? "Υφιστάμενος/η" : "—"; return `<div class="case-list-item"><div><strong>${safeText(item.case_id)}</strong><span>${safeText(item.encounter_date || "χωρίς ημερομηνία")} · ${relationship} · ${safeText(item.encounter_archetype || "—")} · ${safeText(updated)}</span></div><div class="case-list-actions"><button type="button" data-load-case="${safeText(item.internal_uuid)}">Άνοιγμα</button><button type="button" data-delete-case="${safeText(item.internal_uuid)}">Διαγραφή</button></div></div>`; }).join(""); }
  function newCase(confirmUnsaved = true) { if (confirmUnsaved && dirty && !window.confirm("Υπάρχουν μη αποθηκευμένες αλλαγές. Να δημιουργηθεί νέο case;")) return; currentCase = createEmptyCase(nextSequence()); bmiAutocalculated = false; localStorage.setItem(ACTIVE_KEY, currentCase.internal_uuid); dirty = false; syncUiFromState(); el.draftStatus.textContent = "Νέο pilot case — δεν έχει αποθηκευτεί"; switchStep(1); }

  function switchStep(step) { activeStep = Number(step); $$(".step-tab").forEach(b => b.classList.toggle("active", Number(b.dataset.step) === activeStep)); $$(".step-panel").forEach(p => p.classList.toggle("active", Number(p.dataset.stepPanel) === activeStep)); el.nextBtn.textContent = activeStep >= 6 ? "Τέλος →" : "Επόμενο →"; if (activeStep === 2) { snapshotStep1(); renderFractureEvents(); syncStep2Ui(); } window.scrollTo({ top: 0, behavior: "smooth" }); }

  function bindChoiceButtons() { $$('[data-field][data-value]').forEach(b => b.addEventListener("click", () => setSelected(b.dataset.field, b.dataset.value))); }
  function bindStep1Inputs() {
    [el.encounterDate, el.ageYears, el.encounterArchetype, el.weightKg, el.currentHeightCm, el.heightSource, el.referenceHeightCm, el.bmi, el.lastFractureSite, el.lastFractureMonth, el.cfsScore, el.gcDoseMg, el.gcDurationMonths, el.fallsLast12m, el.sarcFScore, el.quickNotes].forEach(input => { input.addEventListener("input", () => { if (input === el.bmi) bmiAutocalculated = false; markDirty(); }); input.addEventListener("change", markDirty); });
    el.heidiOutput.addEventListener("change", () => { currentCase.heidi.output_available = el.heidiOutput.value; markDirty(); }); el.heidiReviewed.addEventListener("change", () => { currentCase.heidi.reviewed_by_clinician = el.heidiReviewed.value; markDirty(); }); el.heidiCorrection.addEventListener("change", () => { currentCase.heidi.material_correction_required = el.heidiCorrection.value; syncHeidiCorrectionUi(); markDirty(); }); el.sarcopeniaMethod.addEventListener("change", () => { currentCase.risk_context.sarcopenia_screen_method = el.sarcopeniaMethod.value; syncRiskDetailUi(); markDirty(); });
    [[el.priorFragilityFracture, "prior_fragility_fracture", "fracture"], [el.parentalHipFracture, "parental_hip_fracture", null], [el.frailtyImmobility, "frailty_or_immobility", "frailty"], [el.glucocorticoids, "glucocorticoids", "gc"], [el.secondaryContext, "secondary_context", "secondary"], [el.sarcopeniaRelevant, "sarcopenia_case_finding_relevant", "sarcopenia"]].forEach(([input, field, clearKey]) => input.addEventListener("change", () => { currentCase.risk_context[field] = input.checked; if (!input.checked && clearKey) clearDependentRiskFields(clearKey); syncRiskDetailUi(); markDirty(); }));
    [el.cognitiveImpairment, el.significantImmobility].forEach(input => input.addEventListener("change", markDirty)); $$('input[name="secondaryCondition"]').forEach(c => c.addEventListener("change", markDirty)); $$('#heidiCategoryChecks input').forEach(c => c.addEventListener("change", markDirty));
  }

  function bindStep2Inputs() {
    [el.fractureHistoryReviewed, el.fractureReviewScope, el.intervalFractureStatus, el.formalRiskIndicated, el.riskToolName, el.fraxCountryModel, el.femoralNeckBmdUsed, el.fraxMof, el.fraxHip, el.currentSmoking, el.highAlcohol, el.rheumatoidArthritis, el.declaredRiskFramework, el.resultingRiskCategory, el.riskOverrideReason].forEach(input => { input.addEventListener("input", markDirty); input.addEventListener("change", markDirty); });
    el.formalRiskDone.addEventListener("change", () => { currentCase.risk_assessment.formal_done = el.formalRiskDone.value; el.fraxDetails.hidden = el.formalRiskDone.value !== "yes"; markDirty(); });
    el.contextualAdjustment.addEventListener("change", () => { currentCase.risk_assessment.contextual_adjustment = el.contextualAdjustment.value; el.adjustmentDetails.hidden = el.contextualAdjustment.value !== "yes"; markDirty(); });
    $$('#adjustmentReasonChecks input').forEach(c => c.addEventListener("change", markDirty));
    el.addFractureEventBtn.addEventListener("click", () => { snapshotStep1(); collectFractureEventsFromDom(); currentCase.fracture_history.events.push(makeFractureEvent()); renderFractureEvents(); markDirty(); });
    el.fractureEvents.addEventListener("input", event => { const row = event.target.closest(".fracture-event"); if (!row) return; collectFractureEventsFromDom(); if (event.target.dataset.eventField === "site") renderFractureEvents(); markDirty(); });
    el.fractureEvents.addEventListener("change", event => { const row = event.target.closest(".fracture-event"); if (!row) return; collectFractureEventsFromDom(); if (event.target.dataset.eventField === "site") renderFractureEvents(); markDirty(); });
    el.fractureEvents.addEventListener("click", event => { const btn = event.target.closest("[data-remove-event]"); if (!btn) return; collectFractureEventsFromDom(); currentCase.fracture_history.events = currentCase.fracture_history.events.filter(e => e.id !== btn.dataset.removeEvent); renderFractureEvents(); markDirty(); });
  }

  function bindNavigation() {
    $$(".step-tab").forEach(b => b.addEventListener("click", () => switchStep(b.dataset.step)));
    el.nextBtn.addEventListener("click", () => { if (activeStep < 6) switchStep(activeStep + 1); else window.alert("Η τελική ολοκλήρωση θα ενεργοποιηθεί όταν υλοποιηθούν όλα τα Steps."); });
    el.saveTopBtn.addEventListener("click", () => saveDraft()); el.saveDraftBtn.addEventListener("click", () => saveDraft()); el.finishVisitBtn.addEventListener("click", () => { saveDraft(); window.alert("Steps 1–2 αποθηκεύτηκαν. Το case δεν χαρακτηρίζεται ακόμη complete μέχρι να υλοποιηθούν Steps 3–6."); });
    el.cancelCaseBtn.addEventListener("click", () => { if (window.confirm("Να καθαριστεί το τρέχον case από την οθόνη; Το αποθηκευμένο draft θα παραμείνει στα Cases.")) newCase(false); });
    $$('[data-nav-action="new-case"]').forEach(b => b.addEventListener("click", () => newCase(true))); $$('[data-nav-action="cases"]').forEach(b => b.addEventListener("click", () => { renderCaseList(); el.casesDialog.showModal(); })); $$('[data-nav-action="privacy"]').forEach(b => b.addEventListener("click", () => el.privacyDialog.showModal())); $$('[data-nav-action="heidi"]').forEach(b => b.addEventListener("click", () => { switchStep(1); document.querySelector(".heidi-card")?.scrollIntoView({ behavior: "smooth", block: "center" }); }));
    el.caseList.addEventListener("click", event => { const load = event.target.closest("[data-load-case]"); const del = event.target.closest("[data-delete-case]"); if (load) loadCase(load.dataset.loadCase); if (del) deleteCase(del.dataset.deleteCase); });
  }

  function setupPrivacy() { el.privacyStrip.hidden = localStorage.getItem(PRIVACY_KEY) === "yes"; el.dismissPrivacyBtn.addEventListener("click", () => { localStorage.setItem(PRIVACY_KEY, "yes"); el.privacyStrip.hidden = true; }); }
  function restoreActiveCase() { const activeUuid = localStorage.getItem(ACTIVE_KEY), found = getStore().find(i => i.internal_uuid === activeUuid); if (found) { currentCase = normalizeLoadedCase(found); el.draftStatus.textContent = "Φορτώθηκε το τελευταίο τοπικό draft"; } else { currentCase = createEmptyCase(nextSequence()); localStorage.setItem(ACTIVE_KEY, currentCase.internal_uuid); } }

  window.addEventListener("beforeunload", event => { if (!dirty) return; event.preventDefault(); event.returnValue = ""; });

  bindChoiceButtons(); bindStep1Inputs(); bindStep2Inputs(); bindNavigation(); setupPrivacy(); restoreActiveCase(); syncUiFromState(); renderCaseList();
})();
