(() => {
  "use strict";

  const asArray = value => Array.isArray(value) ? value : [];
  const clean = value => typeof value === "string" ? value.trim() : "";
  const isObject = value => value && typeof value === "object" && !Array.isArray(value);
  const isCompleted = status => status === "completed" || status === "amended";
  const isIsoDate = value => /^\d{4}-\d{2}-\d{2}$/.test(clean(value));
  const isYearMonth = value => /^\d{4}-\d{2}$/.test(clean(value));
  const hasValue = value => value !== null && value !== undefined && value !== "";

  const LAB_FIELDS = Object.freeze([
    ["ca", "Ca"],
    ["vitamin_d", "25-OH Vit D"],
    ["egfr", "eGFR"],
    ["pth", "PTH"],
    ["ctx", "CTX"],
    ["p1np", "P1NP"]
  ]);

  function payloadOf(row) {
    if (isObject(row?.payload)) return row.payload;
    if (isObject(row?.payload_json)) return row.payload_json;
    return {};
  }

  function compareEncounter(a, b) {
    const ad = clean(a?.encounter_date);
    const bd = clean(b?.encounter_date);
    if (ad !== bd) return ad.localeCompare(bd);
    return clean(a?.updated_at).localeCompare(clean(b?.updated_at));
  }

  function historicalRows(encounters, currentInternalUuid = "") {
    return asArray(encounters)
      .filter(row => isCompleted(row?.status))
      .filter(row => !currentInternalUuid || clean(payloadOf(row)?.internal_uuid) !== currentInternalUuid)
      .slice()
      .sort(compareEncounter);
  }

  function latestMatching(rows, getter, predicate = hasValue) {
    for (let i = rows.length - 1; i >= 0; i -= 1) {
      const value = getter(payloadOf(rows[i]), rows[i]);
      if (predicate(value)) return { value, row: rows[i] };
    }
    return null;
  }

  function numberOrNull(value) {
    if (value === "" || value === null || value === undefined || Number.isNaN(Number(value))) return null;
    return Number(value);
  }

  function dxaFromPayload(payload) {
    const dxa = isObject(payload?.step3?.dxa) ? payload.step3.dxa : {};
    if (clean(dxa.used) !== "yes") return null;
    const date = clean(dxa.date);
    const point = {
      date: isIsoDate(date) ? date : null,
      spine_t: numberOrNull(dxa.spine_t),
      total_hip_t: numberOrNull(dxa.total_hip_t),
      femoral_neck_t: numberOrNull(dxa.femoral_neck_t),
      machine: clean(dxa.machine) || null,
      machine_label: clean(dxa.machine_label) || null
    };
    const hasMeasurement = [point.spine_t, point.total_hip_t, point.femoral_neck_t].some(value => value !== null);
    return point.date || hasMeasurement ? point : null;
  }

  function buildDxa(rows) {
    const latest = latestMatching(rows, payload => dxaFromPayload(payload), value => isObject(value));
    if (!latest) return { state: "not_documented" };
    return {
      state: "documented",
      ...latest.value,
      source_encounter_id: latest.row?.encounter_id || null,
      source_encounter_date: latest.row?.encounter_date || null,
      interpretation_note: "Περιγραφική τελευταία DXA· δεν δηλώνεται σημαντική μεταβολή χωρίς κατάλληλη συγκρισιμότητα/LSC."
    };
  }

  function buildRisk(rows) {
    const latest = latestMatching(rows, payload => {
      const risk = isObject(payload?.risk_assessment) ? payload.risk_assessment : {};
      const category = clean(risk.resulting_risk_category);
      const framework = clean(risk.declared_framework);
      const mof = numberOrNull(risk.frax_mof_percent);
      const hip = numberOrNull(risk.frax_hip_percent);
      if (!category && !framework && mof === null && hip === null) return null;
      return { category: category || null, framework: framework || null, mof, hip };
    }, value => isObject(value));
    if (!latest) return { state: "not_documented" };
    return {
      state: "documented",
      ...latest.value,
      source_encounter_id: latest.row?.encounter_id || null,
      source_encounter_date: latest.row?.encounter_date || null
    };
  }

  function fractureKey(event) {
    const stableId = clean(event?.id);
    if (stableId) return `id:${stableId}`;
    const site = clean(event?.site);
    const month = clean(event?.month);
    if (site && isYearMonth(month)) return `site-month:${site}|${month}`;
    return "";
  }

  function buildFractures(rows) {
    const events = new Map();
    let priorFragilitySeen = false;
    let fallbackSite = "";
    let fallbackMonth = "";
    let unidentifiedPresent = false;

    rows.forEach(row => {
      const payload = payloadOf(row);
      const risk = isObject(payload?.risk_context) ? payload.risk_context : {};
      if (risk.prior_fragility_fracture === true) priorFragilitySeen = true;
      if (clean(risk.last_fracture_site)) fallbackSite = clean(risk.last_fracture_site);
      if (isYearMonth(risk.last_fracture_month)) fallbackMonth = clean(risk.last_fracture_month);

      asArray(payload?.fracture_history?.events).forEach(event => {
        const key = fractureKey(event);
        if (!key) {
          unidentifiedPresent = true;
          return;
        }
        if (!events.has(key)) {
          events.set(key, {
            id: clean(event?.id) || null,
            site: clean(event?.site) || null,
            month: isYearMonth(event?.month) ? clean(event.month) : null,
            fragility: clean(event?.fragility) || null,
            occurred_on_treatment: clean(event?.occurred_on_treatment) || null,
            source_encounter_ids: []
          });
        }
        const stored = events.get(key);
        if (row?.encounter_id && !stored.source_encounter_ids.includes(row.encounter_id)) stored.source_encounter_ids.push(row.encounter_id);
      });
    });

    const unique = Array.from(events.values());
    const dated = unique.filter(event => event.month).sort((a, b) => a.month.localeCompare(b.month));
    const latest = dated.length ? dated[dated.length - 1] : null;
    const anyKnown = unique.length || priorFragilitySeen || fallbackSite || fallbackMonth;
    if (!anyKnown) return { state: "not_documented", documented_count: 0, count_reliability: "absent" };

    return {
      state: "documented",
      documented_count: unique.length,
      count_reliability: unidentifiedPresent ? "partial" : "reliable",
      prior_fragility_fracture: priorFragilitySeen || unique.length > 0,
      most_recent: latest || (fallbackSite || fallbackMonth ? { site: fallbackSite || null, month: fallbackMonth || null } : null)
    };
  }

  function buildTreatment(projection = {}) {
    const conflicts = asArray(projection?.conflict_records).filter(item => item?.domain === "treatment_history" || item?.domain === "administrations");
    const treatment = isObject(projection?.treatment_projection) ? projection.treatment_projection : {};
    const admin = isObject(projection?.administration_projection) ? projection.administration_projection : {};
    const active = isObject(treatment.active_episode) ? treatment.active_episode : null;
    const events = asArray(admin.unique_actual_events);
    const latestActual = events.slice().filter(item => isIsoDate(item?.actual_date)).sort((a, b) => a.actual_date.localeCompare(b.actual_date)).pop() || null;
    const counts = isObject(admin.administration_count_by_agent) ? { ...admin.administration_count_by_agent } : {};
    const reliability = isObject(admin.count_reliability_by_agent) ? { ...admin.count_reliability_by_agent } : {};

    if (!active && !events.length && !asArray(treatment.historical_episodes).length && !conflicts.length) {
      return { state: "not_documented", conflicts: [] };
    }

    return {
      state: conflicts.length ? "conflicting" : "documented",
      active_episode: active ? {
        agent: clean(active.agent) || null,
        start_date: isIsoDate(active.start_date) ? clean(active.start_date) : null,
        status: clean(active.status) || null,
        source_encounter_id: active.source_encounter_id || null
      } : null,
      actual_event_count: events.length,
      latest_actual: latestActual ? { agent: clean(latestActual.agent) || null, actual_date: latestActual.actual_date } : null,
      administration_count_by_agent: counts,
      count_reliability_by_agent: reliability,
      conflicts: conflicts.map(item => ({ domain: item.domain, summary_code: item.summary_code }))
    };
  }

  function buildDecision(rows) {
    const latest = latestMatching(rows, payload => {
      const decision = isObject(payload?.step4?.decision) ? payload.step4.decision : {};
      const type = clean(decision.type);
      const selectedAgent = clean(decision.selected_agent);
      if (!type && !selectedAgent) return null;
      return {
        type: type || null,
        selected_agent: selectedAgent || null,
        patient_accepted: clean(decision.patient_accepted) || null
      };
    }, value => isObject(value));
    if (!latest) return { state: "not_documented" };
    return {
      state: "documented",
      ...latest.value,
      source_encounter_id: latest.row?.encounter_id || null,
      source_encounter_date: latest.row?.encounter_date || null
    };
  }

  function buildLabs(labs) {
    const rows = asArray(labs).filter(row => isIsoDate(row?.lab_date)).slice().sort((a, b) => clean(a.lab_date).localeCompare(clean(b.lab_date)));
    if (!rows.length) return { state: "not_documented", date: null, values: [] };
    const latest = rows[rows.length - 1];
    const values = LAB_FIELDS
      .filter(([key]) => hasValue(latest?.values?.[key]))
      .map(([key, label]) => ({ key, label, value: latest.values[key] }));
    return { state: "documented", date: latest.lab_date, values };
  }

  function buildUnresolved(projection = {}) {
    const tasks = asArray(projection?.unresolved_task_projection).map(task => ({
      task_type: clean(task?.task_type) || "other",
      due_date: isIsoDate(task?.due_date) ? task.due_date : null,
      timeframe_text: clean(task?.timeframe_text) || null,
      source_encounter_id: task?.source_encounter_id || null
    }));
    const conflicts = asArray(projection?.conflict_records).map(item => ({ domain: clean(item?.domain) || "unknown", summary_code: clean(item?.summary_code) || "CONFLICT" }));
    const closeState = clean(projection?.prior_close_projection?.latest_unresolved_critical) || "absent";
    return {
      state: conflicts.length ? "conflicting" : tasks.length || closeState === "yes" ? "unresolved" : "clear_or_absent",
      tasks,
      unresolved_critical: closeState,
      conflicts
    };
  }

  function buildCourse(rows) {
    if (!rows.length) return { state: "no_completed_history", encounter_count: 0, first_date: null, latest_date: null };
    return {
      state: "documented",
      encounter_count: rows.length,
      first_date: clean(rows[0].encounter_date) || null,
      latest_date: clean(rows[rows.length - 1].encounter_date) || null
    };
  }

  function buildSummary({ encounters = [], labs = [], projection = {}, currentCase = {}, historyStatus = "loaded" } = {}) {
    if (historyStatus === "unavailable") {
      return {
        schema_version: "osteoporosis_longitudinal_summary_v1",
        state: "unavailable",
        message: "Το protected ιστορικό δεν είναι διαθέσιμο· δεν συμπεραίνεται απουσία προηγούμενων επισκέψεων ή ευρημάτων."
      };
    }
    if (historyStatus === "loading" || historyStatus === "not_loaded") {
      return {
        schema_version: "osteoporosis_longitudinal_summary_v1",
        state: historyStatus,
        message: "Φορτώνεται το protected longitudinal ιστορικό."
      };
    }

    const rows = historicalRows(encounters, clean(currentCase?.internal_uuid));
    return {
      schema_version: "osteoporosis_longitudinal_summary_v1",
      state: "ready",
      course: buildCourse(rows),
      fractures: buildFractures(rows),
      risk: buildRisk(rows),
      dxa: buildDxa(rows),
      treatment: buildTreatment(projection),
      labs: buildLabs(labs),
      decision: buildDecision(rows),
      unresolved: buildUnresolved(projection),
      current_visit: {
        state: clean(currentCase?.internal_uuid) ? "current_non_historical" : "absent",
        encounter_date: isIsoDate(currentCase?.encounter_date) ? clean(currentCase.encounter_date) : null,
        archetype: clean(currentCase?.encounter_archetype) || null
      }
    };
  }

  window.BaselineOsteoporosisLongitudinalSummary = Object.freeze({
    buildSummary
  });
})();
