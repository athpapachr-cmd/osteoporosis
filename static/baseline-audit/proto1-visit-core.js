(() => {
  "use strict";
  const rows = value => Array.isArray(value) ? value : [];
  const clean = value => typeof value === "string" ? value.trim() : "";
  const iso = value => /^\d{4}-\d{2}-\d{2}$/.test(clean(value));
  const completed = row => ["completed", "amended"].includes(row?.status);
  const source = (row, editor) => ({ encounter_id: row?.encounter_id || null, encounter_date: row?.encounter_date || null, editor });
  const point = (date, type, label, row, editor, uncertainty = "") => ({
    date: clean(date) || null,
    precision: iso(date) ? "day" : /^\d{4}-\d{2}$/.test(clean(date)) ? "month" : /^\d{4}$/.test(clean(date)) ? "year" : "unknown",
    type, label, source: source(row, editor), uncertainty
  });

  function build({ patientId = "", encounters = [], projection = {}, summary = {}, plan = {}, current = {}, historyStatus = "not_loaded" } = {}) {
    const history = rows(encounters).filter(completed).filter(row => !current?.internal_uuid || row?.payload?.internal_uuid !== current.internal_uuid);
    const active = projection?.treatment_projection?.active_episode;
    const actual = rows(projection?.administration_projection?.unique_actual_events).filter(item => item.agent === "denosumab" && iso(item.actual_date));
    const conflict = rows(projection?.conflict_records).some(item => ["administrations", "treatment_history"].includes(item.domain));
    const reliable = historyStatus === "loaded" && !conflict && projection?.administration_projection?.count_reliability_by_agent?.denosumab !== "conflicting";
    const lastActual = reliable ? actual.slice().sort((a, b) => a.actual_date.localeCompare(b.actual_date)).pop() || null : null;
    const previousDecision = summary?.decision?.state === "documented" ? summary.decision : null;
    const stopLike = ["stop", "switch", "complete_course", "consolidate"].includes(clean(previousDecision?.type));
    const decisionRow = history.find(row => row.encounter_id === previousDecision?.source_encounter_id);
    const explicitExit = clean(decisionRow?.payload?.step4?.transition?.type) === "denosumab_exit";
    const stop = stopLike && (previousDecision?.selected_agent === "denosumab" || explicitExit);
    const ambiguousStop = stopLike && !stop;
    const successor = stop && clean(decisionRow?.payload?.step4?.transition?.next_agent)
      ? { agent: clean(decisionRow.payload.step4.transition.next_agent), planned_date: clean(decisionRow.payload.step4.transition.next_agent_date) || null, source: source(decisionRow, "4") }
      : null;
    const hasDenosumabEpisode = history.some(row => rows(row?.payload?.step4?.treatment_episodes).some(ep => ep.agent === "denosumab"));
    const relevant = active?.agent === "denosumab" || actual.length > 0 || hasDenosumabEpisode;
    const seenRules = new Set();
    const guidance = rows(plan?.evidence_contributions).length ? rows(plan.evidence_contributions) : rows(plan?.ordered_cards).flatMap(item => rows(item.evidence_rules));
    const uniqueGuidance = guidance.filter(rule => { if (!rule?.rule_id || seenRules.has(rule.rule_id)) return false; seenRules.add(rule.rule_id); return true; });
    const timing = !stop && !ambiguousStop && reliable && lastActual ? uniqueGuidance.find(rule => rule.rule_id === "OST_G2_R12_DENOSUMAB_EVIDENCE_DUE" && iso(rule.evidence_expected_due_date)) : null;
    const delayed = !stop && !ambiguousStop && reliable && lastActual ? uniqueGuidance.find(rule => rule.rule_id === "OST_G2_R24_DENOSUMAB_GT7M_REBOUND_ESCALATION") : null;
    const milestones = [];
    const plans = [];
    const obligations = [];
    history.forEach(row => {
      milestones.push(point(row.encounter_date, "visit", "Ολοκληρωμένη επίσκεψη", row, "1"));
      rows(row?.payload?.step4?.treatment_episodes).filter(ep => ep.agent === "denosumab" && ["active", "stopped", "completed"].includes(ep.status)).forEach(ep => {
        milestones.push(point(ep.start_date || ep.end_date, "treatment", `Denosumab · ${ep.status}`, row, "4", "Snapshot θεραπείας· συνέχεια μη επιβεβαιωμένη"));
      });
      rows(row?.payload?.step4?.administrations).filter(a => a.agent === "denosumab" && !iso(a.actual_date) && (iso(a.scheduled_date) || ["missed", "planned", "due", "overdue"].includes(a.status))).forEach(a => {
        const item = point(a.scheduled_date, "plan", `Denosumab · ${a.status || "planned"}`, row, "4", "Δεν τεκμηριώνει χορήγηση");
        plans.push(item); milestones.push(item);
      });
      const decision = row?.payload?.step4?.decision || {};
      if (clean(decision.type)) milestones.push(point(row.encounter_date, "decision", `Απόφαση · ${decision.type}${decision.selected_agent ? ` · ${decision.selected_agent}` : ""}`, row, "4"));
      rows(row?.payload?.step4?.tasks).forEach(task => {
        if (!clean(task.type) && !clean(task.due_date) && !clean(task.timeframe_text)) return;
        const item = point(task.due_date, "obligation", `Εκκρεμότητα · ${task.type || "other"} · ${task.status || "planned"}`, row, "4", "Η συνέχεια με άλλες εγγραφές δεν έχει επιβεβαιωθεί");
        item.timeframe = clean(task.timeframe_text) || null;
        item.status = clean(task.status) || "planned";
        obligations.push(item);
        if (item.status === "planned") milestones.push(item);
      });
    });
    actual.forEach(event => {
      const row = history.find(item => rows(event.source_encounter_ids).includes(item.encounter_id));
      milestones.push(point(event.actual_date, "actual", "Καταγεγραμμένη πραγματική χορήγηση denosumab", row, "4", conflict ? "Ιστορική ασυμφωνία" : ""));
    });
    if (summary?.dxa?.state === "documented" && summary.dxa.source_encounter_id) {
      const row = history.find(item => item.encounter_id === summary.dxa.source_encounter_id);
      if (row) milestones.push(point(summary.dxa.date, "investigation", "Καταγεγραμμένη DXA · χωρίς αυτόματο συμπέρασμα μεταβολής", row, "3"));
    }
    if (successor) milestones.push(point(successor.planned_date, "plan", `Σχεδιασμένη επόμενη θεραπεία · ${successor.agent}`, decisionRow, "4", "Δεν τεκμηριώνει πραγματική έναρξη"));
    milestones.sort((a, b) => (a.date || "").localeCompare(b.date || ""));
    let focus = "verification";
    if (stop) focus = "transition";
    else if (ambiguousStop || (historyStatus === "loaded" && conflict)) focus = "verification";
    else if (delayed || (timing && iso(current.encounter_date) && current.encounter_date > timing.evidence_expected_due_date)) focus = "delay";
    else if (timing) focus = "due";
    else if (plans.length || obligations.some(item => item.status === "planned")) focus = "continuity";
    else if (relevant) focus = "followup";
    const reconciliation = [];
    for (const planned of plans) {
      const candidate = actual.find(a => a.actual_date >= (planned.date || "") && planned.date && a.actual_date.slice(0, 7) === planned.date.slice(0, 7));
      if (candidate) reconciliation.push({ kind: "planned_actual", first: planned, second: milestones.find(p => p.type === "actual" && p.date === candidate.actual_date), reason: "Ίδιος παράγοντας και κοντινή ημερομηνία· η σχέση δεν έχει επιβεβαιωθεί." });
    }
    for (let i = 0; i < obligations.length; i += 1) {
      for (let j = i + 1; j < obligations.length; j += 1) {
        const first = obligations[i], second = obligations[j];
        if (first.label === second.label && first.source.encounter_id !== second.source.encounter_id && first.date !== second.date) {
          reconciliation.push({ kind: "obligation", first, second, reason: "Ίδιος τύπος εργασίας σε διαφορετικές επισκέψεις με διαφορετικό χρόνο· η ταυτότητα δεν αποδεικνύεται από τον τύπο ή την ημερομηνία." });
        }
      }
    }
    const snapshots = milestones.filter(item => item.type === "treatment");
    for (let i = 1; i < snapshots.length; i += 1) {
      const first = snapshots[i - 1], second = snapshots[i];
      if (first.source.encounter_id !== second.source.encounter_id && first.label === second.label) {
        reconciliation.push({ kind: "episode", first, second, reason: "Ο ίδιος παράγοντας εμφανίζεται σε διαδοχικά θεραπευτικά snapshots· η συνέχεια δεν έχει επιβεβαιωθεί." });
      }
    }
    return {
      schema_version: "ost_proto1_visit_projection_v1", patient_id: patientId || null, history_status: historyStatus,
      relevant, focus, suggested_focus: focus, asserted_treatment: active?.agent === "denosumab" && !stop && !ambiguousStop ? "denosumab" : null,
      last_recorded_actual: lastActual, actual_history_state: !reliable ? "conflicting_or_unavailable" : lastActual ? "recorded" : "unknown",
      prior_final_decision: previousDecision, successor_plan: successor, derived_due: timing ? { date: timing.evidence_expected_due_date, rule_id: timing.rule_id, source_refs: timing.source_refs, why_now: timing.why_now } : null,
      delay_guidance: delayed || null, plans, obligations, milestones, reconciliation,
      uncertainty: historyStatus !== "loaded" ? "Protected history unavailable or loading" : conflict ? "Conflicting administration/treatment history" : ambiguousStop ? "Prior stop/transition decision has no reliable denosumab attribution" : !lastActual ? "No reliable recorded actual denosumab date" : null,
      current_guidance: uniqueGuidance.filter(rule => /DENOSUMAB/.test(rule.rule_id || "")),
      current_decision: current?.step4?.decision || null
    };
  }

  function previsit(projection) {
    if (!projection || !projection.patient_id || projection.history_status !== "loaded") return null;
    return {
      patient_id: projection.patient_id, focus: projection.focus, asserted_treatment: projection.asserted_treatment,
      last_recorded_actual: projection.last_recorded_actual,
      prior_final_decision: projection.prior_final_decision,
      open_obligations: projection.obligations.filter(item => item.status === "planned").slice(0, 3),
      uncertainty: projection.uncertainty,
      milestones: projection.milestones.slice(-5)
    };
  }
  window.OstProto1VisitCore = Object.freeze({ build, previsit });
})();
