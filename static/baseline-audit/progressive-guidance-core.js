(() => {
  "use strict";

  const DOMAIN_ORDER = [
    "fracture_history", "formal_risk", "dxa", "vfa", "secondary_causes", "laboratory_monitoring",
    "falls_function", "sarcopenia", "treatment_history", "administrations", "treatment_decision",
    "transition_safety", "followup_tasks", "communication", "understanding", "reflection", "documentation_capture"
  ];

  const REASON_PRIORITY = Object.freeze({
    NEW_EVENT: 10,
    UNRESOLVED_PRIOR: 20,
    EXPLICIT_DUE_STATE: 30,
    TREATMENT_CONTEXT: 40,
    VISIT_TYPE_CORE: 50,
    CONTEXTUAL: 60
  });

  const BASE_FLOW = Object.freeze({
    initial_assessment_new_or_uncertain_diagnosis: [
      "fracture_history", "formal_risk", "dxa", "vfa", "secondary_causes", "laboratory_monitoring",
      "falls_function", "treatment_decision", "followup_tasks", "communication", "documentation_capture"
    ],
    initial_assessment_known_osteoporosis_or_osteopenia: [
      "fracture_history", "dxa", "secondary_causes", "treatment_history", "treatment_decision",
      "followup_tasks", "communication", "documentation_capture"
    ],
    routine_followup_stable: [
      "fracture_history", "treatment_history", "followup_tasks", "documentation_capture"
    ],
    treatment_start: [
      "treatment_history", "treatment_decision", "administrations", "followup_tasks", "communication", "documentation_capture"
    ],
    treatment_continuation_or_due_monitoring: [
      "fracture_history", "treatment_history", "administrations", "laboratory_monitoring", "treatment_decision",
      "followup_tasks", "communication", "documentation_capture"
    ],
    treatment_change_or_transition: [
      "fracture_history", "treatment_history", "administrations", "treatment_decision", "transition_safety",
      "followup_tasks", "communication", "documentation_capture"
    ],
    post_fragility_fracture: [
      "fracture_history", "formal_risk", "dxa", "vfa", "secondary_causes", "laboratory_monitoring",
      "falls_function", "treatment_history", "treatment_decision", "followup_tasks", "communication", "documentation_capture"
    ],
    fracture_on_treatment: [
      "fracture_history", "formal_risk", "dxa", "vfa", "secondary_causes", "laboratory_monitoring",
      "falls_function", "treatment_history", "administrations", "treatment_decision", "transition_safety",
      "followup_tasks", "communication", "documentation_capture"
    ],
    adverse_effect_or_intolerance: [
      "treatment_history", "administrations", "treatment_decision", "transition_safety", "followup_tasks",
      "communication", "documentation_capture"
    ],
    treatment_completion_or_consolidation: [
      "treatment_history", "dxa", "laboratory_monitoring", "administrations", "treatment_decision", "transition_safety",
      "followup_tasks", "communication", "documentation_capture"
    ],
    other: []
  });

  const ARCHETYPE_LABELS = Object.freeze({
    initial_assessment_new_or_uncertain_diagnosis: "Αρχική αξιολόγηση — νέα/αβέβαιη διάγνωση",
    initial_assessment_known_osteoporosis_or_osteopenia: "Αρχική αξιολόγηση — γνωστή οστεοπόρωση/οστεοπενία",
    routine_followup_stable: "Routine follow-up — σταθερή",
    treatment_start: "Έναρξη θεραπείας",
    treatment_continuation_or_due_monitoring: "Συνέχιση / χορήγηση / due monitoring",
    treatment_change_or_transition: "Αλλαγή / transition θεραπείας",
    post_fragility_fracture: "Μετά από κάταγμα ευθραυστότητας",
    fracture_on_treatment: "Κάταγμα υπό θεραπεία",
    adverse_effect_or_intolerance: "Ανεπιθύμητη ενέργεια / δυσανεξία",
    treatment_completion_or_consolidation: "Ολοκλήρωση / consolidation",
    other: "Άλλο / περιγράφεται στο σύντομο context"
  });

  const asArray = value => Array.isArray(value) ? value : [];
  const clean = value => typeof value === "string" ? value.trim() : "";
  const isIsoDate = value => /^\d{4}-\d{2}-\d{2}$/.test(clean(value));
  const isCompletedStatus = status => status === "completed" || status === "amended";

  function compareEncounter(a, b) {
    const ad = clean(a?.encounter_date);
    const bd = clean(b?.encounter_date);
    if (ad !== bd) return ad.localeCompare(bd);
    return clean(a?.updated_at).localeCompare(clean(b?.updated_at));
  }

  function normalizeAgent(value) {
    return clean(value).toLowerCase();
  }

  function makeConflict(domain, summaryCode, sourceEncounterIds, candidateValues = []) {
    return {
      conflict_id: `${domain}:${summaryCode}:${sourceEncounterIds.join("|")}`,
      domain,
      summary_code: summaryCode,
      source_encounter_ids: Array.from(new Set(sourceEncounterIds.filter(Boolean))),
      candidate_values: candidateValues,
      resolution_state: "unresolved"
    };
  }

  function buildLongitudinalProjection(encounters, { currentInternalUuid = "" } = {}) {
    const prior = asArray(encounters)
      .filter(row => isCompletedStatus(row?.status))
      .filter(row => !currentInternalUuid || row?.payload?.internal_uuid !== currentInternalUuid)
      .slice()
      .sort(compareEncounter);

    const conflicts = [];
    const actualEvents = new Map();
    const stableIdLocations = new Map();

    prior.forEach(row => {
      asArray(row?.payload?.step4?.administrations).forEach(admin => {
        const agent = normalizeAgent(admin?.agent);
        const actualDate = clean(admin?.actual_date);
        if (!agent || !isIsoDate(actualDate)) return;

        const stableId = clean(admin?.id);
        if (stableId) {
          const priorLocation = stableIdLocations.get(stableId);
          const currentLocation = `${agent}|${actualDate}`;
          if (priorLocation && priorLocation !== currentLocation) {
            conflicts.push(makeConflict(
              "administrations",
              "SAME_EVENT_ID_DIFFERENT_ACTUAL_FACT",
              [row.encounter_id],
              [{ id: stableId, prior: priorLocation, current: currentLocation }]
            ));
          } else if (!priorLocation) {
            stableIdLocations.set(stableId, currentLocation);
          }
        }

        const key = `${agent}|${actualDate}`;
        let event = actualEvents.get(key);
        if (!event) {
          event = {
            agent,
            actual_date: actualDate,
            scheduled_date: isIsoDate(admin?.scheduled_date) ? admin.scheduled_date : null,
            next_due_date: isIsoDate(admin?.next_due_date) ? admin.next_due_date : null,
            source_encounter_ids: [],
            stable_ids: new Set(),
            next_due_candidates: new Set()
          };
          actualEvents.set(key, event);
        }
        if (row.encounter_id) event.source_encounter_ids.push(row.encounter_id);
        if (stableId) event.stable_ids.add(stableId);
        if (isIsoDate(admin?.next_due_date)) event.next_due_candidates.add(admin.next_due_date);
        if (!event.scheduled_date && isIsoDate(admin?.scheduled_date)) event.scheduled_date = admin.scheduled_date;
      });
    });

    const uniqueActualEvents = Array.from(actualEvents.values()).map(event => {
      const dueValues = Array.from(event.next_due_candidates);
      if (dueValues.length > 1) {
        conflicts.push(makeConflict(
          "administrations",
          "CONFLICTING_NEXT_DUE_FOR_SAME_ACTUAL_EVENT",
          event.source_encounter_ids,
          dueValues
        ));
      }
      return {
        agent: event.agent,
        actual_date: event.actual_date,
        scheduled_date: event.scheduled_date,
        next_due_date: dueValues.length === 1 ? dueValues[0] : null,
        source_encounter_ids: Array.from(new Set(event.source_encounter_ids)),
        identity_basis: event.stable_ids.size === 1 ? "stable_event_id" : "agent_plus_exact_actual_date"
      };
    }).sort((a, b) => a.actual_date.localeCompare(b.actual_date) || a.agent.localeCompare(b.agent));

    const countByAgent = {};
    const lastActualByAgent = {};
    const latestNextDueByAgent = {};
    uniqueActualEvents.forEach(event => {
      countByAgent[event.agent] = (countByAgent[event.agent] || 0) + 1;
      if (!lastActualByAgent[event.agent] || event.actual_date > lastActualByAgent[event.agent]) {
        lastActualByAgent[event.agent] = event.actual_date;
        latestNextDueByAgent[event.agent] = event.next_due_date || null;
      }
    });

    const conflictingAgents = new Set();
    conflicts.forEach(conflict => {
      if (conflict.domain !== "administrations") return;
      asArray(conflict.candidate_values).forEach(value => {
        if (value && typeof value === "object" && value.current) conflictingAgents.add(clean(value.current).split("|")[0]);
      });
      if (conflict.summary_code === "CONFLICTING_NEXT_DUE_FOR_SAME_ACTUAL_EVENT") {
        const sourceIds = new Set(conflict.source_encounter_ids);
        uniqueActualEvents.filter(event => event.source_encounter_ids.some(id => sourceIds.has(id))).forEach(event => conflictingAgents.add(event.agent));
      }
    });

    const countReliabilityByAgent = {};
    Object.keys(countByAgent).forEach(agent => {
      countReliabilityByAgent[agent] = conflictingAgents.has(agent) ? "conflicting" : "reliable";
    });

    let latestTreatmentSnapshot = null;
    for (let i = prior.length - 1; i >= 0; i -= 1) {
      const episodes = asArray(prior[i]?.payload?.step4?.treatment_episodes).filter(ep => ep && typeof ep === "object");
      if (episodes.length) {
        latestTreatmentSnapshot = { encounter_id: prior[i].encounter_id, encounter_date: prior[i].encounter_date, episodes };
        break;
      }
    }

    const activeEpisodes = latestTreatmentSnapshot
      ? latestTreatmentSnapshot.episodes.filter(ep => clean(ep?.status) === "active")
      : [];
    if (activeEpisodes.length > 1) {
      conflicts.push(makeConflict(
        "treatment_history",
        "MULTIPLE_ACTIVE_TREATMENT_EPISODES",
        [latestTreatmentSnapshot.encounter_id],
        activeEpisodes.map(ep => ({ id: clean(ep?.id), agent: normalizeAgent(ep?.agent) }))
      ));
    }

    const taskState = new Map();
    prior.forEach(row => {
      asArray(row?.payload?.step4?.tasks).forEach(task => {
        const type = clean(task?.type);
        const dueDate = isIsoDate(task?.due_date) ? task.due_date : "";
        const timeframe = clean(task?.timeframe_text);
        if (!type && !dueDate && !timeframe) return;
        const semanticKey = `${type}|${dueDate}|${timeframe}`;
        taskState.set(semanticKey, {
          task_type: type || "other",
          due_date: dueDate || null,
          timeframe_text: timeframe || null,
          status: clean(task?.status) || "planned",
          source_encounter_id: row.encounter_id
        });
      });
    });
    const unresolvedTasks = Array.from(taskState.values()).filter(task => task.status === "planned");

    const latest = prior.length ? prior[prior.length - 1] : null;
    const latestClose = latest?.payload?.step4?.close || null;
    const priorCloseProjection = {
      latest_unresolved_critical: latestClose?.unresolved_critical === "yes"
        ? "yes"
        : latestClose?.unresolved_critical === "no" ? "no" : latestClose ? "uncertain" : "absent",
      source_encounter_id: latest?.encounter_id || null,
      note_present: Boolean(clean(latestClose?.note))
    };

    return {
      schema_version: "longitudinal_guidance_projection_v1",
      module: "osteoporosis",
      source_encounter_ids: prior.map(row => row.encounter_id).filter(Boolean),
      prior_encounter_count: prior.length,
      latest_completed_or_amended_encounter_id: latest?.encounter_id || null,
      latest_completed_or_amended_encounter_date: latest?.encounter_date || null,
      latest_coarse_visit_intent: clean(latest?.payload?.encounter_archetype) || null,
      treatment_projection: {
        active_episode: activeEpisodes.length === 1 ? { ...activeEpisodes[0], source_encounter_id: latestTreatmentSnapshot?.encounter_id || null } : null,
        historical_episodes: latestTreatmentSnapshot ? latestTreatmentSnapshot.episodes.map(ep => ({ ...ep, source_encounter_id: latestTreatmentSnapshot.encounter_id })) : [],
        source_strategy: "latest_reliable_snapshot_plus_conflict_detection"
      },
      administration_projection: {
        unique_actual_events: uniqueActualEvents,
        administration_count_by_agent: countByAgent,
        last_actual_administration_by_agent: lastActualByAgent,
        latest_next_due_by_agent: latestNextDueByAgent,
        count_reliability_by_agent: countReliabilityByAgent
      },
      unresolved_task_projection: unresolvedTasks,
      prior_close_projection: priorCloseProjection,
      conflict_records: conflicts,
      quality: {
        treatment_timeline: conflicts.some(x => x.domain === "treatment_history") ? "conflicting" : latestTreatmentSnapshot ? "partial" : "absent",
        administration_history: conflicts.some(x => x.domain === "administrations") ? "conflicting" : uniqueActualEvents.length ? "reliable" : "absent",
        task_state: unresolvedTasks.length ? "partial" : prior.length ? "reliable" : "absent",
        can_compute_reliable_administration_count: Object.values(countReliabilityByAgent).every(value => value === "reliable")
      }
    };
  }

  function classifyExplicitDue(nextDueDate, encounterDate) {
    if (!isIsoDate(nextDueDate) || !isIsoDate(encounterDate)) return null;
    if (nextDueDate < encounterDate) return "overdue";
    if (nextDueDate === encounterDate) return "due";
    return "not_due";
  }

  function buildEncounterContext(currentCase, projection = {}, overrides = {}) {
    const c = currentCase && typeof currentCase === "object" ? currentCase : {};
    const encounterArchetype = clean(overrides.encounter_archetype) || clean(c.encounter_archetype) || "";
    const encounterDate = clean(overrides.encounter_date) || clean(c.encounter_date) || "";
    const visitContextText = clean(overrides.visit_context_text) || clean(c.quick_notes);
    const intervalFractureStatus = clean(overrides.interval_fracture_status) || clean(c?.fracture_history?.interval_fracture_status);
    const fractureEvents = asArray(c?.fracture_history?.events);
    const newFracture = encounterArchetype === "post_fragility_fracture" || encounterArchetype === "fracture_on_treatment" || intervalFractureStatus === "yes";
    const fractureOnTreatment = encounterArchetype === "fracture_on_treatment" || (newFracture && fractureEvents.some(event => clean(event?.occurred_on_treatment) === "yes"));

    const treatmentAgents = new Set();
    asArray(c?.step4?.treatment_episodes).forEach(ep => {
      if (clean(ep?.status) === "active" && normalizeAgent(ep?.agent)) treatmentAgents.add(normalizeAgent(ep.agent));
    });
    const projectedActiveAgent = normalizeAgent(projection?.treatment_projection?.active_episode?.agent);
    if (projectedActiveAgent) treatmentAgents.add(projectedActiveAgent);

    const explicitDue = [];
    const explicitOverdue = [];
    const currentAdmins = asArray(c?.step4?.administrations);
    currentAdmins.forEach(admin => {
      const agent = normalizeAgent(admin?.agent) || "treatment";
      const status = clean(admin?.status);
      if (status === "due") explicitDue.push(agent);
      if (status === "overdue" || status === "missed") explicitOverdue.push(agent);
      const dateState = classifyExplicitDue(admin?.next_due_date, encounterDate);
      if (dateState === "due") explicitDue.push(agent);
      if (dateState === "overdue") explicitOverdue.push(agent);
    });
    Object.entries(projection?.administration_projection?.latest_next_due_by_agent || {}).forEach(([agent, nextDue]) => {
      const dateState = classifyExplicitDue(nextDue, encounterDate);
      if (dateState === "due") explicitDue.push(agent);
      if (dateState === "overdue") explicitOverdue.push(agent);
    });

    const unresolvedTasks = asArray(projection?.unresolved_task_projection);
    const priorCritical = projection?.prior_close_projection?.latest_unresolved_critical === "yes";

    return {
      schema_version: "encounter_context_v1",
      module: "osteoporosis",
      encounter_archetype: encounterArchetype,
      patient_relationship: clean(c.patient_relationship_status) || "unknown",
      encounter_date: encounterDate,
      visit_context_text: visitContextText,
      prior_encounter_count: Number(projection?.prior_encounter_count || 0),
      latest_prior_encounter_date: projection?.latest_completed_or_amended_encounter_date || null,
      active_treatment_agents: Array.from(treatmentAgents).sort(),
      administration_count_by_agent: projection?.administration_projection?.administration_count_by_agent || {},
      last_actual_administration_by_agent: projection?.administration_projection?.last_actual_administration_by_agent || {},
      explicit_due_agents: Array.from(new Set(explicitDue)).sort(),
      explicit_overdue_agents: Array.from(new Set(explicitOverdue)).sort(),
      new_events: {
        fracture: newFracture ? "yes" : "no",
        fracture_on_treatment: fractureOnTreatment ? "yes" : "no"
      },
      unresolved_prior_items: unresolvedTasks,
      prior_unresolved_critical: priorCritical,
      projection_conflicts: asArray(projection?.conflict_records)
    };
  }

  function buildVisitPlan(context) {
    const ctx = context && typeof context === "object" ? context : {};
    const states = new Map();

    function add(domain, reasonCode, humanReason, priority) {
      if (!DOMAIN_ORDER.includes(domain)) return;
      const existing = states.get(domain) || {
        card_id: domain,
        visibility: "surfaced",
        priority,
        reason_codes: [],
        why_now_reasons: [],
        prior_data_state: "uncertain",
        capture_state: "unresolved",
        provisional_candidate_count: 0,
        conflict_count: 0,
        critical_unresolved: false
      };
      existing.priority = Math.min(existing.priority, priority);
      if (!existing.reason_codes.includes(reasonCode)) existing.reason_codes.push(reasonCode);
      if (humanReason && !existing.why_now_reasons.includes(humanReason)) existing.why_now_reasons.push(humanReason);
      states.set(domain, existing);
    }

    const archetype = clean(ctx.encounter_archetype);
    asArray(BASE_FLOW[archetype]).forEach(domain => add(
      domain,
      "VISIT_TYPE_CORE",
      `Βασικό για τον δηλωμένο τύπο επίσκεψης: ${ARCHETYPE_LABELS[archetype] || archetype}`,
      REASON_PRIORITY.VISIT_TYPE_CORE
    ));

    if (ctx?.new_events?.fracture === "yes") {
      ["fracture_history", "formal_risk", "dxa", "vfa", "secondary_causes", "laboratory_monitoring", "falls_function", "treatment_history", "treatment_decision", "followup_tasks"].forEach(domain => add(
        domain,
        "NEW_EVENT",
        "Νέο/τρέχον κάταγμα αλλάζει τη σημερινή ροή.",
        REASON_PRIORITY.NEW_EVENT
      ));
    }
    if (ctx?.new_events?.fracture_on_treatment === "yes") {
      ["administrations", "transition_safety"].forEach(domain => add(
        domain,
        "NEW_EVENT",
        "Το κάταγμα δηλώθηκε ότι συνέβη υπό θεραπεία.",
        REASON_PRIORITY.NEW_EVENT
      ));
    }

    const unresolvedCount = asArray(ctx.unresolved_prior_items).length;
    if (unresolvedCount || ctx.prior_unresolved_critical) {
      add(
        "followup_tasks",
        "UNRESOLVED_PRIOR",
        ctx.prior_unresolved_critical
          ? "Υπήρχε κρίσιμη εκκρεμότητα στην προηγούμενη ολοκληρωμένη επίσκεψη."
          : `Υπάρχουν ${unresolvedCount} προηγούμενα planned task(s) χωρίς τεκμηριωμένη ολοκλήρωση.`,
        REASON_PRIORITY.UNRESOLVED_PRIOR
      );
      const item = states.get("followup_tasks");
      if (item && ctx.prior_unresolved_critical) item.critical_unresolved = true;
    }

    const overdueAgents = asArray(ctx.explicit_overdue_agents);
    const dueAgents = asArray(ctx.explicit_due_agents);
    if (overdueAgents.length || dueAgents.length) {
      const text = overdueAgents.length
        ? `Υπάρχει ρητά καταγεγραμμένο overdue/missed treatment timing για: ${overdueAgents.join(", ")}.`
        : `Υπάρχει ρητά καταγεγραμμένη due ημερομηνία σήμερα για: ${dueAgents.join(", ")}.`;
      ["administrations", "followup_tasks"].forEach(domain => add(domain, "EXPLICIT_DUE_STATE", text, REASON_PRIORITY.EXPLICIT_DUE_STATE));
    }

    const treatmentContext = asArray(ctx.active_treatment_agents);
    if (treatmentContext.length || ["treatment_start", "treatment_continuation_or_due_monitoring", "treatment_change_or_transition", "treatment_completion_or_consolidation", "fracture_on_treatment", "adverse_effect_or_intolerance"].includes(archetype)) {
      ["treatment_history", "administrations"].forEach(domain => add(
        domain,
        "TREATMENT_CONTEXT",
        treatmentContext.length ? `Υπάρχει ενεργό/πρόσφατο treatment context: ${treatmentContext.join(", ")}.` : "Ο δηλωμένος τύπος επίσκεψης αφορά θεραπεία/χορήγηση.",
        REASON_PRIORITY.TREATMENT_CONTEXT
      ));
    }

    if (asArray(ctx.projection_conflicts).length) {
      ["treatment_history", "administrations"].forEach(domain => add(
        domain,
        "CONTEXTUAL",
        "Υπάρχει ασυμφωνία στο longitudinal treatment context που δεν επιλύθηκε αυτόματα.",
        REASON_PRIORITY.CONTEXTUAL
      ));
      ["treatment_history", "administrations"].forEach(domain => {
        const item = states.get(domain);
        if (item) item.conflict_count = asArray(ctx.projection_conflicts).length;
      });
    }

    const orderedCards = Array.from(states.values())
      .map(item => ({ ...item, why_now: item.why_now_reasons.join(" · ") }))
      .sort((a, b) => a.priority - b.priority || DOMAIN_ORDER.indexOf(a.card_id) - DOMAIN_ORDER.indexOf(b.card_id));

    return {
      schema_version: "visit_plan_v1",
      module: "osteoporosis",
      encounter_archetype: archetype,
      visit_context_text: clean(ctx.visit_context_text),
      ordered_cards: orderedCards,
      critical_unresolved: orderedCards.filter(item => item.critical_unresolved).map(item => item.card_id),
      close_requirements: [],
      rule_trace: orderedCards.flatMap(item => item.reason_codes.map(code => ({ matched: true, reason_code: code, card_id: item.card_id })))
    };
  }

  function archetypeLabel(value) {
    return ARCHETYPE_LABELS[clean(value)] || (clean(value) ? clean(value) : "Δεν έχει επιλεγεί τύπος επίσκεψης");
  }

  window.BaselineProgressiveGuidanceCore = Object.freeze({
    DOMAIN_ORDER: Object.freeze(DOMAIN_ORDER.slice()),
    REASON_PRIORITY,
    BASE_FLOW,
    archetypeLabel,
    buildLongitudinalProjection,
    buildEncounterContext,
    buildVisitPlan,
    classifyExplicitDue
  });
})();
