(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const ACTIVE_PATIENT_KEY = "osteoporosis.clinical.activePatient.v1";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const DOMAIN_LABELS = Object.freeze({
    fracture_history: "Ιστορικό καταγμάτων",
    formal_risk: "Formal fracture risk",
    dxa: "DXA / longitudinal DXA",
    vfa: "VFA / vertebral imaging",
    secondary_causes: "Secondary causes",
    laboratory_monitoring: "Εργαστηριακά / monitoring",
    falls_function: "Πτώσεις / frailty / function",
    sarcopenia: "Σαρκοπενία",
    treatment_history: "Θεραπευτικό ιστορικό",
    administrations: "Administrations / due dates",
    treatment_decision: "Κλινική απόφαση",
    transition_safety: "Transition / sequencing safety",
    followup_tasks: "Follow-up / care tasks",
    communication: "Επικοινωνία",
    understanding: "Κατανόηση / teach-back",
    reflection: "Post-visit reflection",
    documentation_capture: "Documentation / capture"
  });

  const CARD_ANCHORS = Object.freeze({
    fracture_history: ["#fractureHistoryReviewed"],
    formal_risk: ["#formalRiskIndicated", "#declaredRiskFramework", "#longitudinalRiskCard"],
    dxa: ["#s3DxaUsed", "#longitudinalDxaCard"],
    vfa: ["#s3VfaIndicated"],
    secondary_causes: ["#s3SecondaryIndicated"],
    laboratory_monitoring: ["#s3Ca"],
    falls_function: ["#s3FallsReviewed"],
    sarcopenia: ["#s3SarcApplicable"],
    treatment_history: ["#s4AddEpisode"],
    administrations: ["#s4AddAdministration"],
    treatment_decision: ["#s4DecisionType"],
    transition_safety: ["#s4TransitionRelevant"],
    followup_tasks: ["#s4AddTask", "#s4PlanComplete"],
    communication: ["#s5ConditionRisk"],
    understanding: ["#s5UnderstandCondition"],
    reflection: ["#s5WentWell"],
    documentation_capture: ["#s6Sources", "#s6GesyAvailable", "#s6HeidiFinalNote", "#s6Reliability"]
  });

  let historyPatientId = "";
  let historicalEncounters = [];
  let historyLoadState = "not_loaded";
  let historyLoadError = "";
  let lastPlan = null;
  let lastEvidenceContext = null;
  let lastEvidenceContributions = [];
  let refreshTimer = null;

  function getCases() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function activeUuid() {
    return localStorage.getItem(ACTIVE_KEY) || "";
  }

  function activeCase() {
    const id = activeUuid();
    return getCases().find(item => item.internal_uuid === id) || null;
  }

  function activePatientId() {
    return sessionStorage.getItem(ACTIVE_PATIENT_KEY) || "";
  }

  function numberOrNull(value) {
    return value === "" || value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value);
  }

  function liveValue(selector, persistedValue = "") {
    const node = $(selector);
    return node ? String(node.value ?? "") : persistedValue;
  }

  function liveTrimmedValue(selector, persistedValue = "") {
    const node = $(selector);
    return node ? String(node.value ?? "").trim() : persistedValue;
  }

  function liveNumber(selector, persistedValue = null) {
    const node = $(selector);
    return node ? numberOrNull(node.value) : persistedValue;
  }

  function liveChecked(selector, persistedValue = false) {
    const node = $(selector);
    return node ? Boolean(node.checked) : Boolean(persistedValue);
  }

  function liveChoiceValue(field, persistedValue = "") {
    const controls = $$(`[data-field="${field}"][data-value]`);
    if (!controls.length) return persistedValue;
    const selected = controls.find(node => node.classList?.contains?.("selected") || node.getAttribute?.("aria-pressed") === "true");
    return selected?.dataset?.value || "";
  }

  function fractureEventsFromDom(root = document) {
    return $$(".fracture-event", root).map(row => {
      const event = { id: row.dataset.eventId || "" };
      $$('[data-event-field]', row).forEach(field => {
        event[field.dataset.eventField] = field.value;
      });
      return event;
    });
  }

  function repeatRowsFromDom(rootSelector, rowSelector, idAttribute, persistedRows = []) {
    const root = $(rootSelector);
    if (!root) return Array.isArray(persistedRows) ? persistedRows : [];
    return $$(rowSelector, root).map(row => {
      const out = { id: row.getAttribute?.(idAttribute) || row.dataset?.[idAttribute.replace(/^data-/, "").replace(/-([a-z])/g, (_, c) => c.toUpperCase())] || "" };
      $$('[data-k]', row).forEach(node => {
        let value = node.value;
        if (node.type === "number") value = value === "" ? null : Number(value);
        out[node.dataset.k] = value;
      });
      return out;
    });
  }

  function currentCaseSnapshot() {
    const base = activeCase() || {};
    const anthropometrics = { ...(base.anthropometrics || {}) };
    const riskContext = { ...(base.risk_context || {}) };
    const riskAssessment = { ...(base.risk_assessment || {}) };
    const fractureHistory = { ...(base.fracture_history || {}) };
    const step3Base = base.step3 || {};
    const step4Base = base.step4 || {};

    const currentHeightNode = $("#currentHeightCm");
    const referenceHeightNode = $("#referenceHeightCm");
    anthropometrics.current_height_cm = currentHeightNode ? numberOrNull(currentHeightNode.value) : anthropometrics.current_height_cm ?? null;
    anthropometrics.reference_height_cm = referenceHeightNode ? numberOrNull(referenceHeightNode.value) : anthropometrics.reference_height_cm ?? null;
    if (currentHeightNode || referenceHeightNode) {
      anthropometrics.derived_height_loss_cm = anthropometrics.current_height_cm !== null && anthropometrics.reference_height_cm !== null
        ? Math.max(0, Math.round((anthropometrics.reference_height_cm - anthropometrics.current_height_cm) * 10) / 10)
        : null;
    }

    riskContext.glucocorticoids = liveChecked("#glucocorticoids", riskContext.glucocorticoids);
    riskContext.glucocorticoid_prednisolone_mg_day = liveNumber("#gcDoseMg", riskContext.glucocorticoid_prednisolone_mg_day ?? null);
    riskContext.glucocorticoid_duration_months = liveNumber("#gcDurationMonths", riskContext.glucocorticoid_duration_months ?? null);
    riskContext.falls_last_12_months = liveNumber("#fallsLast12m", riskContext.falls_last_12_months ?? null);

    riskAssessment.formal_indicated = liveValue("#formalRiskIndicated", riskAssessment.formal_indicated || "");
    riskAssessment.declared_framework = liveValue("#declaredRiskFramework", riskAssessment.declared_framework || "");
    riskAssessment.resulting_risk_category = liveValue("#resultingRiskCategory", riskAssessment.resulting_risk_category || "");

    const intervalNode = $("#intervalFractureStatus");
    if (intervalNode) fractureHistory.interval_fracture_status = String(intervalNode.value ?? "");
    const fractureRoot = $("#fractureEvents");
    if (fractureRoot) fractureHistory.events = fractureEventsFromDom(fractureRoot);

    const dxa = { ...(step3Base.dxa || {}) };
    dxa.used = liveValue("#s3DxaUsed", dxa.used || "");
    dxa.spine_t = liveNumber("#s3SpineT", dxa.spine_t ?? null);
    dxa.total_hip_t = liveNumber("#s3TotalHipT", dxa.total_hip_t ?? null);
    dxa.femoral_neck_t = liveNumber("#s3FnT", dxa.femoral_neck_t ?? null);
    const secondary = { ...(step3Base.secondary || {}) };
    secondary.prior_workup_adequate = liveValue("#s3PriorWorkupAdequate", secondary.prior_workup_adequate || "");
    const step3 = { ...step3Base, dxa, secondary };

    const treatmentEpisodes = repeatRowsFromDom("#s4Episodes", "[data-episode-id]", "data-episode-id", step4Base.treatment_episodes);
    const administrations = repeatRowsFromDom("#s4Administrations", "[data-admin-id]", "data-admin-id", step4Base.administrations);
    const decision = {
      ...(step4Base.decision || {}),
      type: liveValue("#s4DecisionType", step4Base.decision?.type || ""),
      selected_agent: liveValue("#s4SelectedAgent", step4Base.decision?.selected_agent || "")
    };
    const transition = {
      ...(step4Base.transition || {}),
      relevant: liveValue("#s4TransitionRelevant", step4Base.transition?.relevant || ""),
      type: liveValue("#s4TransitionType", step4Base.transition?.type || ""),
      prior_end_date: liveValue("#s4PriorAgentEnd", step4Base.transition?.prior_end_date || ""),
      next_agent: liveValue("#s4NextAgent", step4Base.transition?.next_agent || ""),
      next_agent_date: liveValue("#s4NextAgentDate", step4Base.transition?.next_agent_date || "")
    };
    const step4 = { ...step4Base, treatment_episodes: treatmentEpisodes, administrations, decision, transition };

    return {
      ...base,
      internal_uuid: base.internal_uuid || activeUuid(),
      encounter_archetype: liveValue("#encounterArchetype", base.encounter_archetype || ""),
      encounter_date: liveValue("#encounterDate", base.encounter_date || ""),
      age_years: liveNumber("#ageYears", base.age_years ?? null),
      sex: liveChoiceValue("sex", base.sex || ""),
      menopause_status: liveChoiceValue("menopause_status", base.menopause_status || ""),
      patient_relationship_status: liveChoiceValue("patient_relationship_status", base.patient_relationship_status || ""),
      osteoporosis_status: liveChoiceValue("osteoporosis_status", base.osteoporosis_status || ""),
      quick_notes: liveTrimmedValue("#quickNotes", base.quick_notes || ""),
      anthropometrics,
      risk_context: riskContext,
      risk_assessment: riskAssessment,
      fracture_history: fractureHistory,
      step3,
      step4
    };
  }

  async function fetchHistoricalEncounters(patientId) {
    if (!patientId) return [];
    const res = await fetch(`/clinical/patient/${encodeURIComponent(patientId)}/encounters`, {
      method: "GET",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) throw new Error(`Historical encounters unavailable (${res.status})`);
    const body = await res.json();
    return Array.isArray(body) ? body : [];
  }

  function historyStateSnapshot() {
    return {
      status: historyLoadState,
      patient_id: historyPatientId || null,
      encounter_count: historicalEncounters.length,
      error_present: Boolean(historyLoadError)
    };
  }

  function longitudinalMetaText(context = {}) {
    const patientId = activePatientId();
    if (!patientId) {
      return "Δεν έχει φορτωθεί protected patient — η ροή βασίζεται μόνο στο τρέχον local encounter.";
    }
    if (historyLoadState === "loading") {
      return "Φορτώνεται το προηγούμενο protected ιστορικό. Η σημερινή local ροή παραμένει διαθέσιμη, αλλά το longitudinal context δεν είναι ακόμη πλήρες.";
    }
    if (historyLoadState === "unavailable") {
      return "Δεν ήταν δυνατή η φόρτωση του προηγούμενου protected ιστορικού. Μην θεωρήσεις ότι δεν υπάρχουν προηγούμενες επισκέψεις ή εκκρεμότητες.";
    }
    if (historyLoadState !== "loaded") {
      return "Το προηγούμενο protected ιστορικό δεν έχει φορτωθεί ακόμη. Δεν εξάγεται συμπέρασμα απουσίας ιστορικού.";
    }
    return `${context.prior_encounter_count || 0} προηγούμενες ολοκληρωμένες/τροποποιημένες επισκέψεις${context.latest_prior_encounter_date ? ` · τελευταία ${context.latest_prior_encounter_date}` : ""}`;
  }

  function cardsForDomain(domain) {
    const cards = [];
    (CARD_ANCHORS[domain] || []).forEach(selector => {
      const node = $(selector);
      const card = node?.matches?.("article.card") ? node : node?.closest?.("article.card");
      if (card && !cards.includes(card)) cards.push(card);
    });
    return cards;
  }

  function conciseSourceLabel(ref) {
    const value = String(ref || "");
    if (value.startsWith("NOGG_2024")) return "NOGG 2024";
    if (value.startsWith("ENDOCRINE_SOCIETY_2020")) return "Endocrine Society 2020";
    if (value.startsWith("ECTS_2020")) return "ECTS 2020";
    if (value.startsWith("EMA_PROLIA")) return "EMA Prolia";
    if (value.startsWith("EMA_ACLASTA")) return "EMA Aclasta";
    if (value.startsWith("EMA_EVENITY")) return "EMA Evenity";
    if (value.startsWith("EMA_FORSTEO")) return "EMA Forsteo";
    return value.split("#")[0].replaceAll("_", " ");
  }

  function evidenceMetadata(item) {
    const rules = Array.isArray(item?.evidence_rules) ? item.evidence_rules : [];
    const sources = [];
    rules.forEach(rule => (rule.source_refs || []).forEach(ref => {
      const label = conciseSourceLabel(ref);
      if (label && !sources.includes(label)) sources.push(label);
    }));
    return {
      rules,
      sources,
      checklistOnly: rules.some(rule => rule.activation_mode === "checklist_only")
    };
  }

  function appendEvidenceMeta(root, item) {
    const meta = evidenceMetadata(item);
    if (!meta.rules.length) return;
    const wrap = document.createElement("div");
    wrap.className = "progressive-guidance-evidence";
    const ruleIds = meta.rules.map(rule => rule.rule_id).filter(Boolean);
    if (ruleIds.length) wrap.dataset.ruleIds = ruleIds.join(",");
    if (meta.checklistOnly) {
      const safety = document.createElement("span");
      safety.className = "progressive-guidance-checklist-note";
      safety.textContent = "Safety checklist — απαιτεί κλινική επιβεβαίωση, όχι automatic clearance.";
      wrap.appendChild(safety);
    }
    if (meta.sources.length) {
      const provenance = document.createElement("span");
      provenance.className = "progressive-guidance-provenance";
      provenance.textContent = `Τεκμηρίωση: ${meta.sources.join(" · ")}`;
      wrap.appendChild(provenance);
    }
    root.appendChild(wrap);
  }

  function clearGuidanceFromCards() {
    $$('article.card.guidance-surfaced').forEach(card => {
      card.classList.remove("guidance-surfaced");
      card.style.removeProperty("order");
      card.removeAttribute("data-guidance-priority");
      card.removeAttribute("data-guidance-domain");
      $(":scope > .progressive-why-now", card)?.remove();
    });
  }

  function applyPlanToCards(plan) {
    clearGuidanceFromCards();
    (plan?.ordered_cards || []).forEach(item => {
      cardsForDomain(item.card_id).forEach(card => {
        card.classList.add("guidance-surfaced");
        card.dataset.guidancePriority = String(item.priority);
        card.dataset.guidanceDomain = item.card_id;
        card.style.order = String(-1000 + Number(item.priority || 999));

        const reason = document.createElement("div");
        reason.className = "progressive-why-now";
        if ((item.reason_codes || []).includes("NEW_EVENT") || (item.evidence_rules || []).some(rule => ["event_triggered", "critical_safety"].includes(rule.rule_class))) reason.classList.add("is-event");
        const strong = document.createElement("strong");
        strong.textContent = "Γιατί τώρα: ";
        reason.appendChild(strong);
        reason.appendChild(document.createTextNode(item.why_now || "Σχετικό με τη σημερινή επίσκεψη."));
        if ((item.reason_codes || []).length) {
          const badge = document.createElement("span");
          badge.className = "progressive-guidance-live-badge";
          badge.textContent = "Σήμερα";
          reason.appendChild(badge);
        }
        appendEvidenceMeta(reason, item);
        const heading = $(":scope > .card-heading", card) || $(".card-heading", card);
        if (heading) heading.insertAdjacentElement("afterend", reason);
        else card.prepend(reason);
      });
    });
  }

  function ensureSummary() {
    let root = $("#progressiveGuidanceSummary");
    if (root) return root;
    root = document.createElement("section");
    root.id = "progressiveGuidanceSummary";
    root.className = "progressive-guidance-summary";
    const tabs = $(".step-tabs");
    if (tabs?.parentNode) tabs.parentNode.insertBefore(root, tabs);
    return root;
  }

  function addTextRow(root, className, label, value) {
    const row = document.createElement("div");
    row.className = className;
    const strong = document.createElement("strong");
    strong.textContent = label;
    row.appendChild(strong);
    row.appendChild(document.createTextNode(value));
    root.appendChild(row);
  }

  function renderSummary(context, plan, projection) {
    const core = window.BaselineProgressiveGuidanceCore;
    const root = ensureSummary();
    root.innerHTML = "";

    const head = document.createElement("div");
    head.className = "progressive-guidance-head";
    const title = document.createElement("h2");
    title.textContent = "Σημερινή ροή";
    const intent = document.createElement("span");
    intent.className = "progressive-guidance-intent";
    intent.textContent = core.archetypeLabel(context.encounter_archetype);
    head.append(title, intent);
    root.appendChild(head);

    if (context.visit_context_text) {
      addTextRow(root, "progressive-guidance-context", "Σύντομο context: ", context.visit_context_text);
    } else if (context.encounter_archetype === "other") {
      const empty = document.createElement("div");
      empty.className = "progressive-guidance-empty";
      empty.textContent = "Το visit type είναι Other. Γράψε προαιρετικά στο σύντομο context τι έγινε· δεν γίνεται αυτόματη κλινική ταξινόμηση από free text στο G-1/G-2.";
      root.appendChild(empty);
    }

    addTextRow(
      root,
      historyLoadState === "unavailable" ? "progressive-guidance-conflict" : "progressive-guidance-meta",
      "Longitudinal context: ",
      longitudinalMetaText(context)
    );

    const cards = plan?.ordered_cards || [];
    if (cards.length) {
      const list = document.createElement("div");
      list.className = "progressive-guidance-list";
      cards.slice(0, 12).forEach(item => {
        const box = document.createElement("div");
        box.className = "progressive-guidance-item";
        const name = document.createElement("strong");
        name.textContent = DOMAIN_LABELS[item.card_id] || item.card_id;
        const why = document.createElement("span");
        why.textContent = `Γιατί τώρα: ${item.why_now || "Σχετικό με τη σημερινή επίσκεψη."}`;
        box.append(name, why);
        appendEvidenceMeta(box, item);
        list.appendChild(box);
      });
      root.appendChild(list);
    } else {
      const empty = document.createElement("div");
      empty.className = "progressive-guidance-empty";
      empty.textContent = context.encounter_archetype
        ? "Δεν υπάρχει ειδικό priority rule για αυτή τη ροή. Τα υπάρχοντα cards παραμένουν διαθέσιμα case-by-case."
        : "Επίλεξε τύπο επίσκεψης για να οργανωθεί η βασική ροή.";
      root.appendChild(empty);
    }

    const conflicts = projection?.conflict_records || [];
    if (conflicts.length) {
      const warning = document.createElement("div");
      warning.className = "progressive-guidance-conflict";
      warning.textContent = `Longitudinal context: ${conflicts.length} ασυμφωνία/ες δεν επιλύθηκαν αυτόματα. Έλεγξε Treatment history / Administrations πριν βασιστείς στη χρονογραμμή.`;
      root.appendChild(warning);
    }
  }

  function computeAndRender() {
    const core = window.BaselineProgressiveGuidanceCore;
    if (!core) return;
    const current = currentCaseSnapshot();
    const projection = core.buildLongitudinalProjection(historicalEncounters, { currentInternalUuid: current.internal_uuid || "" });
    const context = core.buildEncounterContext(current, projection, {
      encounter_archetype: liveValue("#encounterArchetype", ""),
      encounter_date: liveValue("#encounterDate", ""),
      visit_context_text: liveTrimmedValue("#quickNotes", ""),
      interval_fracture_status: liveValue("#intervalFractureStatus", "")
    });
    const basePlan = core.buildVisitPlan(context);
    const g2 = window.BaselineOsteoporosisEvidenceGuidance;
    let plan = basePlan;
    lastEvidenceContext = null;
    lastEvidenceContributions = [];
    if (g2) {
      lastEvidenceContext = g2.buildEvidenceContext(current, projection, context, { historicalEncounters });
      lastEvidenceContributions = g2.evaluateEvidenceGuidance(lastEvidenceContext);
      plan = g2.mergeEvidenceContributions(basePlan, lastEvidenceContributions);
    }
    lastPlan = plan;
    applyPlanToCards(plan);
    renderSummary(context, plan, projection);
  }

  function scheduleRender(delay = 0) {
    if (refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => {
      refreshTimer = null;
      computeAndRender();
    }, delay);
  }

  async function refreshHistory({ force = false } = {}) {
    const patientId = activePatientId();
    if (!patientId) {
      historyPatientId = "";
      historicalEncounters = [];
      historyLoadState = "not_loaded";
      historyLoadError = "";
      scheduleRender(0);
      return;
    }

    if (!force && patientId === historyPatientId && historyLoadState === "loaded") {
      scheduleRender(0);
      return;
    }

    const requestedPatientId = patientId;
    historyPatientId = requestedPatientId;
    historicalEncounters = [];
    historyLoadState = "loading";
    historyLoadError = "";
    scheduleRender(0);

    try {
      const rows = await fetchHistoricalEncounters(requestedPatientId);
      if (activePatientId() !== requestedPatientId || historyPatientId !== requestedPatientId) return;
      historicalEncounters = rows;
      historyLoadState = "loaded";
      historyLoadError = "";
    } catch (error) {
      if (activePatientId() !== requestedPatientId || historyPatientId !== requestedPatientId) return;
      historicalEncounters = [];
      historyLoadState = "unavailable";
      historyLoadError = error?.message || "history unavailable";
    }
    scheduleRender(0);
  }

  function bind() {
    $("#encounterArchetype")?.addEventListener("change", () => scheduleRender(0));
    $("#encounterDate")?.addEventListener("change", () => scheduleRender(0));
    $("#ageYears")?.addEventListener("input", () => scheduleRender(80));
    $("#quickNotes")?.addEventListener("input", () => scheduleRender(80));

    ["1", "2", "3", "4"].forEach(step => {
      const panel = $(`[data-step-panel='${step}']`);
      panel?.addEventListener("change", () => scheduleRender(0));
      panel?.addEventListener("input", () => scheduleRender(100));
    });

    $$(".step-tab").forEach(button => button.addEventListener("click", () => scheduleRender(0)));
    document.addEventListener("click", event => {
      if (event.target.closest?.("[data-field][data-value]")) scheduleRender(0);
      if (event.target.closest?.("#s4AddEpisode, #s4AddAdministration, [data-remove-episode], [data-remove-admin], #addFractureEventBtn, [data-remove-event]")) scheduleRender(20);
      if (event.target.closest?.("[data-load-case]") || event.target.closest?.('[data-nav-action="new-case"]')) scheduleRender(60);
      if (event.target.closest?.("#clinicalSearchResults .btn") || event.target.closest?.("#clinicalCreatePatientBtn")) setTimeout(() => refreshHistory({ force: true }), 180);
    });

    const currentPatient = $("#clinicalCurrentPatient");
    if (currentPatient && typeof MutationObserver !== "undefined") {
      const observer = new MutationObserver(() => refreshHistory({ force: true }));
      observer.observe(currentPatient, { childList: true, subtree: true, characterData: true });
    }
  }

  window.ProgressiveGuidanceUI = Object.freeze({
    refresh: () => refreshHistory({ force: true }),
    getLastPlan: () => lastPlan,
    getLastEvidenceContext: () => lastEvidenceContext,
    getLastEvidenceContributions: () => lastEvidenceContributions.slice(),
    getHistoryLoadState: () => historyStateSnapshot(),
    getLongitudinalMetaText: context => longitudinalMetaText(context),
    getCurrentCaseSnapshot: () => currentCaseSnapshot()
  });

  if (!document.querySelector('link[data-progressive-guidance-style]')) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "./progressive-guidance.css";
    link.dataset.progressiveGuidanceStyle = "true";
    document.head.appendChild(link);
  }

  bind();
  scheduleRender(0);
  setTimeout(() => refreshHistory({ force: true }), 120);
})();
