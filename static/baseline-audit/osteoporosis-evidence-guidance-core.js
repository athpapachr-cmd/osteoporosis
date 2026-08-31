(() => {
  "use strict";

  const DOMAIN_ORDER = [
    "fracture_history", "formal_risk", "dxa", "vfa", "secondary_causes", "laboratory_monitoring",
    "falls_function", "sarcopenia", "treatment_history", "administrations", "treatment_decision",
    "transition_safety", "followup_tasks", "communication", "understanding", "reflection", "documentation_capture"
  ];

  const ORAL_BISPHOSPHONATES = new Set(["alendronate", "risedronate", "ibandronate_oral"]);
  const PARENTERAL_AGENTS = new Set(["zoledronate", "denosumab", "teriparatide", "romosozumab", "ibandronate_iv"]);
  const BLOCKED_RULE_IDS = Object.freeze([
    "OST_G2_R15_DENOSUMAB_EXIT_CTX_FOLLOWUP",
    "OST_G2_R16_DENOSUMAB_EXIT_NO_CTX_OPTION"
  ]);

  const asArray = value => Array.isArray(value) ? value : [];
  const clean = value => typeof value === "string" ? value.trim() : "";
  const numberOrNull = value => value === "" || value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value);
  const isIsoDate = value => /^\d{4}-\d{2}-\d{2}$/.test(clean(value));
  const normalizeAgent = value => clean(value).toLowerCase();
  const isCompletedStatus = status => status === "completed" || status === "amended";

  function parseIsoDate(value) {
    if (!isIsoDate(value)) return null;
    const [year, month, day] = value.split("-").map(Number);
    const date = new Date(Date.UTC(year, month - 1, day));
    if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) return null;
    return date;
  }

  function formatIsoDate(date) {
    if (!(date instanceof Date) || Number.isNaN(date.getTime())) return null;
    return date.toISOString().slice(0, 10);
  }

  function addCalendarMonths(value, months) {
    const parsed = parseIsoDate(value);
    if (!parsed || !Number.isInteger(months)) return null;
    const year = parsed.getUTCFullYear();
    const month = parsed.getUTCMonth();
    const day = parsed.getUTCDate();
    const targetFirst = new Date(Date.UTC(year, month + months, 1));
    const lastDay = new Date(Date.UTC(targetFirst.getUTCFullYear(), targetFirst.getUTCMonth() + 1, 0)).getUTCDate();
    targetFirst.setUTCDate(Math.min(day, lastDay));
    return formatIsoDate(targetFirst);
  }

  function addCalendarYears(value, years) {
    const parsed = parseIsoDate(value);
    if (!parsed || !Number.isInteger(years)) return null;
    const month = parsed.getUTCMonth();
    const day = parsed.getUTCDate();
    const target = new Date(Date.UTC(parsed.getUTCFullYear() + years, month, 1));
    const lastDay = new Date(Date.UTC(target.getUTCFullYear(), month + 1, 0)).getUTCDate();
    target.setUTCDate(Math.min(day, lastDay));
    return formatIsoDate(target);
  }

  function daysBetween(start, end) {
    const a = parseIsoDate(start);
    const b = parseIsoDate(end);
    if (!a || !b) return null;
    return Math.floor((b.getTime() - a.getTime()) / 86_400_000);
  }

  function monthsSinceYearMonth(value, encounterDate) {
    if (!/^\d{4}-\d{2}$/.test(clean(value)) || !isIsoDate(encounterDate)) return null;
    const [year, month] = value.split("-").map(Number);
    if (!year || month < 1 || month > 12) return null;
    const [ey, em] = encounterDate.split("-").map(Number);
    return (ey - year) * 12 + (em - month);
  }

  function deduplicateFractureEvents(events) {
    const seen = new Set();
    const result = [];
    asArray(events).forEach(event => {
      if (!event || typeof event !== "object") return;
      const id = clean(event.id);
      const key = id
        ? `id:${id}`
        : `fact:${clean(event.site)}|${clean(event.month)}|${clean(event.vertebral_level || event.vertebral_level_or_type)}|${clean(event.low_trauma)}`;
      if (seen.has(key)) return;
      seen.add(key);
      result.push({ ...event });
    });
    return result;
  }

  function collectHistoricalFractures(historicalEncounters, currentInternalUuid) {
    const events = [];
    asArray(historicalEncounters)
      .filter(row => isCompletedStatus(clean(row?.status)))
      .filter(row => !currentInternalUuid || clean(row?.payload?.internal_uuid) !== currentInternalUuid)
      .forEach(row => {
        asArray(row?.payload?.fracture_history?.events).forEach(event => events.push(event));
      });
    return events;
  }

  function buildAdministrationTimeline(currentAdministrations, projection = {}) {
    const byFact = new Map();
    let currentConflict = false;
    const currentStableIds = new Map();

    asArray(projection?.administration_projection?.unique_actual_events).forEach(event => {
      const agent = normalizeAgent(event?.agent);
      const actualDate = clean(event?.actual_date);
      if (!agent || !isIsoDate(actualDate)) return;
      byFact.set(`${agent}|${actualDate}`, { agent, actual_date: actualDate, source: "historical" });
    });

    asArray(currentAdministrations).forEach(event => {
      const agent = normalizeAgent(event?.agent);
      const actualDate = clean(event?.actual_date);
      if (!agent || !isIsoDate(actualDate)) return;
      const stableId = clean(event?.id);
      if (stableId) {
        const fact = `${agent}|${actualDate}`;
        const priorFact = currentStableIds.get(stableId);
        if (priorFact && priorFact !== fact) currentConflict = true;
        else currentStableIds.set(stableId, fact);
      }
      byFact.set(`${agent}|${actualDate}`, { agent, actual_date: actualDate, source: "current_or_historical" });
    });

    const events = Array.from(byFact.values()).sort((a, b) => a.actual_date.localeCompare(b.actual_date) || a.agent.localeCompare(b.agent));
    const countByAgent = {};
    const lastByAgent = {};
    events.forEach(event => {
      countByAgent[event.agent] = (countByAgent[event.agent] || 0) + 1;
      if (!lastByAgent[event.agent] || event.actual_date > lastByAgent[event.agent]) lastByAgent[event.agent] = event.actual_date;
    });

    const reliabilityByAgent = { ...(projection?.administration_projection?.count_reliability_by_agent || {}) };
    if (currentConflict) {
      Object.keys(countByAgent).forEach(agent => { reliabilityByAgent[agent] = "conflicting"; });
    }

    return {
      events,
      count_by_agent: countByAgent,
      last_actual_by_agent: lastByAgent,
      reliability_by_agent: reliabilityByAgent,
      current_conflict: currentConflict
    };
  }

  function treatmentSnapshot(currentCase, projection = {}) {
    const currentEpisodes = asArray(currentCase?.step4?.treatment_episodes).filter(item => item && typeof item === "object");
    const projectionConflict = asArray(projection?.conflict_records).some(record => record?.domain === "treatment_history");
    const sourceEpisodes = currentEpisodes.length
      ? currentEpisodes
      : projection?.treatment_projection?.active_episode
        ? [projection.treatment_projection.active_episode]
        : [];
    const active = sourceEpisodes.filter(item => clean(item?.status) === "active");
    return {
      source: currentEpisodes.length ? "current" : "projection",
      episodes: sourceEpisodes,
      active_episodes: active,
      reliable: !projectionConflict && active.length <= 1
    };
  }

  function minimumTScore(dxa = {}) {
    if (clean(dxa?.used) !== "yes") return null;
    const values = [dxa?.spine_t, dxa?.total_hip_t, dxa?.femoral_neck_t].map(numberOrNull).filter(value => value !== null);
    return values.length ? Math.min(...values) : null;
  }

  function noggScopeEligible(currentCase) {
    const age = numberOrNull(currentCase?.age_years);
    const sex = clean(currentCase?.sex);
    if (age === null || age < 50) return false;
    if (sex === "male") return true;
    return sex === "female" && clean(currentCase?.menopause_status) === "postmenopausal";
  }

  function frameworkAllowsNogg(currentCase) {
    const declared = clean(currentCase?.risk_assessment?.declared_framework);
    return !declared || declared === "nogg_2024";
  }

  function currentFragilityFracture(currentCase, currentFractures) {
    const archetype = clean(currentCase?.encounter_archetype);
    if (archetype === "post_fragility_fracture" || archetype === "fracture_on_treatment") return true;
    if (clean(currentCase?.fracture_history?.interval_fracture_status) !== "yes") return false;
    return asArray(currentFractures).some(event => clean(event?.low_trauma) === "yes");
  }

  function activeAgentSet(snapshot) {
    return new Set(snapshot.active_episodes.map(item => normalizeAgent(item?.agent)).filter(Boolean));
  }

  function exactActiveEpisodeForAgent(snapshot, agent) {
    if (!snapshot.reliable) return null;
    const matches = snapshot.active_episodes.filter(item => normalizeAgent(item?.agent) === agent);
    if (matches.length !== 1 || !isIsoDate(matches[0]?.start_date)) return null;
    return matches[0];
  }

  function hasCurrentAdministration(currentCase, agent) {
    return asArray(currentCase?.step4?.administrations).some(item => normalizeAgent(item?.agent) === agent && clean(item?.status) !== "not_applicable");
  }

  function buildEvidenceContext(currentCase, projection = {}, baseContext = {}, options = {}) {
    const current = currentCase && typeof currentCase === "object" ? currentCase : {};
    const encounterDate = clean(current.encounter_date || baseContext.encounter_date);
    const currentFractures = deduplicateFractureEvents(current?.fracture_history?.events);
    const allFractures = deduplicateFractureEvents([
      ...collectHistoricalFractures(options.historicalEncounters, clean(current.internal_uuid)),
      ...currentFractures
    ]);
    const fragilityNow = currentFragilityFracture(current, currentFractures);
    const fractureOnTreatment = clean(baseContext?.new_events?.fracture_on_treatment) === "yes" || clean(current.encounter_archetype) === "fracture_on_treatment";
    const dxa = current?.step3?.dxa || {};
    const minT = minimumTScore(dxa);
    const heightLoss = numberOrNull(current?.anthropometrics?.derived_height_loss_cm);
    const gcDose = numberOrNull(current?.risk_context?.glucocorticoid_prednisolone_mg_day);
    const gcMonths = numberOrNull(current?.risk_context?.glucocorticoid_duration_months);
    const falls = numberOrNull(current?.risk_context?.falls_last_12_months);
    const timeline = buildAdministrationTimeline(current?.step4?.administrations, projection);
    const treatment = treatmentSnapshot(current, projection);
    const activeAgents = activeAgentSet(treatment);
    const projectedAgent = normalizeAgent(projection?.treatment_projection?.active_episode?.agent);
    if (!current?.step4?.treatment_episodes?.length && projectedAgent) activeAgents.add(projectedAgent);

    const vertebralFragility = allFractures.filter(event => clean(event?.site) === "vertebral" && clean(event?.low_trauma) !== "no");
    const recentVertebral = vertebralFragility.some(event => {
      const months = monthsSinceYearMonth(event?.month, encounterDate);
      return months !== null && months >= 0 && months <= 24;
    });

    const decision = current?.step4?.decision || {};
    const transition = current?.step4?.transition || {};
    const selectedAgent = normalizeAgent(decision?.selected_agent);
    const nextAgent = normalizeAgent(transition?.next_agent);
    const plannedAgent = selectedAgent || nextAgent;

    return {
      schema_version: "osteoporosis_evidence_context_v1",
      encounter_archetype: clean(current.encounter_archetype),
      encounter_date: encounterDate,
      age_years: numberOrNull(current.age_years),
      sex: clean(current.sex),
      menopause_status: clean(current.menopause_status),
      osteoporosis_status: clean(current.osteoporosis_status),
      nogg_scope_eligible: noggScopeEligible(current),
      nogg_framework_allowed: frameworkAllowsNogg(current),
      declared_framework: clean(current?.risk_assessment?.declared_framework),
      formal_risk_indicated: clean(current?.risk_assessment?.formal_indicated),
      resulting_risk_category: clean(current?.risk_assessment?.resulting_risk_category),
      current_fragility_fracture: fragilityNow,
      fracture_on_treatment: fractureOnTreatment,
      fracture_events: allFractures,
      vertebral_fracture_count: vertebralFragility.length,
      recent_vertebral_fracture_within_24_months: recentVertebral,
      derived_height_loss_cm: heightLoss,
      minimum_t_score: minT,
      glucocorticoids: Boolean(current?.risk_context?.glucocorticoids),
      glucocorticoid_dose_mg_day: gcDose,
      glucocorticoid_duration_months: gcMonths,
      falls_last_12_months: falls,
      secondary_prior_workup_adequate: clean(current?.step3?.secondary?.prior_workup_adequate),
      administration_timeline: timeline,
      treatment_snapshot: treatment,
      active_treatment_agents: Array.from(activeAgents).sort(),
      decision: {
        type: clean(decision?.type),
        selected_agent: selectedAgent
      },
      transition: {
        relevant: clean(transition?.relevant),
        type: clean(transition?.type),
        next_agent: nextAgent
      },
      planned_agent: plannedAgent,
      current_administrations: asArray(current?.step4?.administrations).map(item => ({ ...item })),
      current_treatment_episodes: asArray(current?.step4?.treatment_episodes).map(item => ({ ...item })),
      projection_conflicts: asArray(projection?.conflict_records),
      base_new_events: baseContext?.new_events || {}
    };
  }

  function makeContribution(ruleId, domains, ruleClass, priority, whyNow, objective, sourceRefs, strength, activationMode, extra = {}) {
    return {
      rule_id: ruleId,
      rule_class: ruleClass,
      priority,
      domains: asArray(domains),
      why_now: whyNow,
      guidance_objective: objective,
      source_refs: asArray(sourceRefs),
      strength: strength || "",
      activation_mode: activationMode,
      ...extra
    };
  }

  function evaluateEvidenceGuidance(context) {
    const ctx = context && typeof context === "object" ? context : {};
    const out = [];
    const push = contribution => out.push(contribution);
    const archetype = clean(ctx.encounter_archetype);
    const frameworkAllowed = Boolean(ctx.nogg_framework_allowed);
    const scope = Boolean(ctx.nogg_scope_eligible);
    const minT = numberOrNull(ctx.minimum_t_score);
    const gcDose = numberOrNull(ctx.glucocorticoid_dose_mg_day);
    const gcMonths = numberOrNull(ctx.glucocorticoid_duration_months);
    const falls = numberOrNull(ctx.falls_last_12_months);
    const activeAgents = new Set(asArray(ctx.active_treatment_agents));
    const timeline = ctx.administration_timeline || {};
    const denosumabReliable = clean(timeline?.reliability_by_agent?.denosumab) !== "conflicting" && !timeline.current_conflict;
    const denosumabCount = Number(timeline?.count_by_agent?.denosumab || 0);
    const lastDenosumab = clean(timeline?.last_actual_by_agent?.denosumab);
    const decisionType = clean(ctx?.decision?.type);
    const selectedAgent = normalizeAgent(ctx?.decision?.selected_agent);
    const transitionType = clean(ctx?.transition?.type);

    if (scope && frameworkAllowed && clean(ctx.formal_risk_indicated) === "yes") {
      push(makeContribution(
        "OST_G2_R01_INITIAL_FORMAL_RISK", ["formal_risk"], "contextual", 45,
        "Έχει δηλωθεί ότι formal fracture-risk assessment ενδείκνυται σήμερα· το NOGG χρησιμοποιεί FRAX όταν υπάρχει clinical risk factor σε άτομα εντός του πεδίου εφαρμογής του.",
        "Perform/record formal fracture-risk assessment when explicitly indicated, preserving the declared framework/country model.",
        ["NOGG_2024_SECTION3#frax_with_clinical_risk_factor"], "strong_when_nogg_scope_and_clinical_risk_factor_apply", "activate_v1"
      ));
    }

    const vfaTrigger = (numberOrNull(ctx.derived_height_loss_cm) !== null && Number(ctx.derived_height_loss_cm) >= 4)
      || (ctx.glucocorticoids && gcMonths !== null && gcMonths >= 3)
      || (minT !== null && minT <= -2.5)
      || asArray(ctx.fracture_events).some(event => clean(event?.site) === "vertebral");
    if (vfaTrigger) {
      push(makeContribution(
        "OST_G2_R02_VFA_STRUCTURED_TRIGGER", ["vfa"], "contextual", 25,
        "Υπάρχει δομημένη ένδειξη για έλεγχο σπονδυλικού κατάγματος/VFA.",
        "Review/perform/arrange vertebral imaging or VFA when an evidence-backed indication is present.",
        ["NOGG_2024_SECTION3#vfa_indications"], "strong", "activate_v1"
      ));
    }

    const osteoporosisConfirmed = ["osteoporosis", "fragility_fracture"].includes(clean(ctx.osteoporosis_status));
    if ((osteoporosisConfirmed || ctx.current_fragility_fracture || ctx.fracture_on_treatment)
      && !(clean(ctx.secondary_prior_workup_adequate) === "yes" && !ctx.current_fragility_fracture && !ctx.fracture_on_treatment)) {
      push(makeContribution(
        "OST_G2_R03_SECONDARY_CAUSE_REVIEW", ["secondary_causes"], "contextual", 35,
        "Οστεοπόρωση ή κάταγμα ευθραυστότητας απαιτεί έλεγχο για υποκείμενα/δευτεροπαθή αίτια ανάλογα με το κλινικό πλαίσιο.",
        "Confirm that underlying/secondary causes have been considered and relevant investigations reviewed or arranged.",
        ["NOGG_2024_SECTION3#investigate_underlying_causes"], "strong", "activate_v1"
      ));
    }

    if (ctx.current_fragility_fracture || ["initial_assessment_new_or_uncertain_diagnosis", "initial_assessment_known_osteoporosis_or_osteopenia"].includes(archetype) || (falls !== null && falls >= 1)) {
      push(makeContribution(
        "OST_G2_R04_FALLS_RISK_AFTER_FRACTURE_OR_INITIAL", ["falls_function"], "contextual", 45,
        "Η πτώση/λειτουργικότητα επηρεάζει άμεσα τον κίνδυνο νέου κατάγματος και την πρόληψη επόμενου συμβάντος.",
        "Assess current falls risk/function and act when material risk is identified.",
        ["NOGG_2024_SUMMARY#falls_assessment"], "guideline_main_recommendation", "activate_v1"
      ));
    }

    if (ctx.current_fragility_fracture) {
      push(makeContribution(
        "OST_G2_R05_NEW_FRAGILITY_FRACTURE_PROMPT_REASSESSMENT", ["formal_risk"], "event_triggered", 10,
        "Νέο κάταγμα ευθραυστότητας αυξάνει τον άμεσο κίνδυνο επανακατάγματος και απαιτεί άμεση επανεκτίμηση.",
        "Reassess fracture risk promptly after a new fragility fracture.",
        ["NOGG_2024_SUMMARY#prompt_after_fragility_fracture", "NOGG_2024_SUMMARY#repeat_risk_after_new_fracture"], "guideline_main_recommendation", "activate_v1"
      ));
      push(makeContribution(
        "OST_G2_R06_NEW_FRAGILITY_FRACTURE_TREATMENT_PLAN", ["treatment_decision"], "event_triggered", 10,
        "Μετά από νέο κάταγμα η θεραπευτική απόφαση δεν πρέπει να μένει χωρίς σαφές πλάνο ή αιτιολογημένη αναβολή.",
        "Address treatment need/options promptly without auto-selecting the drug.",
        ["NOGG_2024_SUMMARY#prompt_after_fragility_fracture"], "guideline_main_recommendation", "activate_v1"
      ));
    }

    if (ctx.fracture_on_treatment) {
      push(makeContribution(
        "OST_G2_R07_FRACTURE_ON_TREATMENT_REVIEW", ["treatment_history", "secondary_causes", "formal_risk", "dxa"], "event_triggered", 10,
        "Κάταγμα υπό θεραπεία απαιτεί έλεγχο πραγματικής έκθεσης/adherence και επανεκτίμηση πριν θεωρηθεί αποτυχία ή αποφασιστεί αλλαγή.",
        "Review adherence/actual exposure, secondary causes and fracture risk with BMD context; do not auto-label failure or switch.",
        ["NOGG_2024_SECTION7#fracture_on_treatment_review", "NOGG_2024_SECTION7#fracture_not_automatic_failure"], "strong", "activate_v1",
        { forbidden_output: "automatic_treatment_failure_or_switch" }
      ));
    }

    const noggVeryHigh = scope && clean(ctx.declared_framework) === "nogg_2024" && (
      clean(ctx.resulting_risk_category) === "very_high"
      || Boolean(ctx.recent_vertebral_fracture_within_24_months)
      || Number(ctx.vertebral_fracture_count || 0) >= 2
      || (minT !== null && minT <= -3.5)
      || (gcDose !== null && gcDose >= 7.5 && gcMonths !== null && gcMonths >= 3)
    );
    if (noggVeryHigh) {
      push(makeContribution(
        "OST_G2_R08_EXPLICIT_VERY_HIGH_RISK_REVIEW", ["treatment_decision", "followup_tasks"], "contextual", 20,
        "Υπάρχει ρητός NOGG δείκτης πολύ υψηλού καταγματικού κινδύνου που αλλάζει το βάθος της θεραπευτικής συζήτησης και μπορεί να απαιτεί specialist strategy.",
        "Surface specialist/parenteral/anabolic consideration without automatically choosing an agent.",
        ["NOGG_2024_SUMMARY#very_high_risk_referral", "NOGG_2024_SECTION6#very_high_risk_anabolic"], "conditional_referral_plus_conditional_anabolic_consideration", "activate_v1"
      ));
    }

    if (["treatment_start", "treatment_change_or_transition", "post_fragility_fracture", "fracture_on_treatment", "treatment_completion_or_consolidation"].includes(archetype)) {
      push(makeContribution(
        "OST_G2_R09_SHARED_TREATMENT_DECISION_FACTORS", ["treatment_decision", "communication"], "contextual", 45,
        "Η σημερινή επίσκεψη περιλαμβάνει θεραπευτική απόφαση· κίνδυνος, καταλληλότητα και προτίμηση ασθενούς πρέπει να είναι ορατά στο rationale.",
        "Make fracture risk, suitability, alternatives and patient preference explicit in the treatment decision.",
        ["NOGG_2024_SECTION6#treatment_choice_factors"], "strong", "activate_v1"
      ));
    }

    if (PARENTERAL_AGENTS.has(clean(ctx.planned_agent))) {
      push(makeContribution(
        "OST_G2_R10_PARENTERAL_VITAMIN_D_REPLETION", ["laboratory_monitoring"], "agent_specific", 20,
        "Η προγραμματισμένη παρεντερική θεραπεία απαιτεί έλεγχο/διόρθωση vitamin-D deficiency πριν από την έναρξη.",
        "Verify/address vitamin-D deficiency or insufficiency before parenteral osteoporosis therapy.",
        ["NOGG_2024_FAQ#parenteral_vitamin_d_before_start"], "nogg_practical_guidance", "checklist_only"
      ));
    }

    if (decisionType === "start" && selectedAgent === "denosumab") {
      push(makeContribution(
        "OST_G2_R11_DENOSUMAB_START_LONG_TERM_PLAN", ["transition_safety", "communication", "followup_tasks"], "agent_specific", 15,
        "Πριν από την έναρξη denosumab πρέπει να είναι σαφές ότι η θεραπεία είναι time-critical και πώς θα αντιμετωπιστεί πιθανή μελλοντική διακοπή.",
        "Establish a long-term denosumab plan, six-month scheduling and an exit strategy before the first dose.",
        ["NOGG_2024_SECTION6#denosumab_long_term_plan", "ENDOCRINE_SOCIETY_2020_DENOSUMAB#denosumab_no_stop_without_followon"], "strong_plus_good_practice", "activate_v1"
      ));
    }

    if (denosumabReliable && activeAgents.has("denosumab") && isIsoDate(lastDenosumab) && isIsoDate(ctx.encounter_date)) {
      const expectedDue = addCalendarMonths(lastDenosumab, 6);
      if (expectedDue && ctx.encounter_date >= expectedDue) {
        push(makeContribution(
          "OST_G2_R12_DENOSUMAB_EVIDENCE_DUE", ["administrations", "followup_tasks"], "milestone_due", 12,
          `Η denosumab είναι θεραπεία ανά 6 μήνες· από την τελευταία πραγματική δόση (${lastDenosumab}) η evidence-derived αναμενόμενη ημερομηνία είναι ${expectedDue}.`,
          "Surface denosumab timing as due/late from the exact actual prior administration date; keep the derived due date ephemeral.",
          ["NOGG_2024_SECTION6#denosumab_q6m", "ENDOCRINE_SOCIETY_2020_DENOSUMAB#denosumab_q6m_no_holiday", "EMA_PROLIA_2026_08_19#prolia_q6m"], "regulatory_plus_guideline", "activate_v1",
          { evidence_expected_due_date: expectedDue, persistence: "ephemeral_only" }
        ));
      }
    }

    if ((decisionType === "start" && selectedAgent === "denosumab") || hasCurrentAdministration({ step4: { administrations: ctx.current_administrations } }, "denosumab")) {
      push(makeContribution(
        "OST_G2_R13_DENOSUMAB_PRE_DOSE_CALCIUM", ["laboratory_monitoring"], "agent_specific", 15,
        "Πριν από τη σημερινή denosumab δόση χρειάζεται επιβεβαίωση calcium/mineral safety.",
        "Verify calcium and correct hypocalcaemia/vitamin-D deficiency before denosumab; do not infer clearance from missing/stale data.",
        ["NOGG_2024_SECTION6#denosumab_calcium_each_dose", "EMA_PROLIA_2026_08_19#prolia_hypocalcaemia"], "regulatory_safety", "checklist_only"
      ));
    }

    const denosumabExit = transitionType === "denosumab_exit" || (decisionType === "stop" && (activeAgents.has("denosumab") || lastDenosumab));
    if (denosumabExit) {
      const sequentialDate = denosumabReliable && isIsoDate(lastDenosumab) ? addCalendarMonths(lastDenosumab, 6) : null;
      push(makeContribution(
        "OST_G2_R14_DENOSUMAB_EXIT_6M_SEQUENTIAL", ["transition_safety", "administrations", "laboratory_monitoring", "followup_tasks"], "critical_safety", 5,
        sequentialDate
          ? `Η διακοπή denosumab έχει rebound/σπονδυλικό κίνδυνο· το reviewed NOGG sequential timing είναι 6 μήνες από την τελευταία πραγματική δόση (${sequentialDate}), εφόσον η επιλογή είναι κλινικά κατάλληλη.`
          : "Η διακοπή denosumab έχει rebound/σπονδυλικό κίνδυνο· χρειάζεται ακριβής τελευταία πραγματική δόση και time-locked sequential antiresorptive plan.",
        "Make the denosumab exit plan explicit; NOGG recommends IV zoledronate at six months when stopping, subject to clinical suitability, without silently selecting the agent.",
        ["NOGG_2024_SECTION6#denosumab_stop_zoledronate_6m", "ECTS_2020_DENOSUMAB_DISCONTINUATION#antiresorptive_6m_after_final_denosumab"], "nogg_strong_with_supporting_ects_position", "activate_v1",
        { evidence_sequential_date: sequentialDate, forbidden_output: "automatic_selected_agent" }
      ));
    }

    const currentZoledronate = hasCurrentAdministration({ step4: { administrations: ctx.current_administrations } }, "zoledronate");
    if ((decisionType === "start" && selectedAgent === "zoledronate") || currentZoledronate) {
      push(makeContribution(
        "OST_G2_R17_ZOLEDRONATE_START_SAFETY", ["laboratory_monitoring"], "critical_safety", 5,
        "Η σημερινή zoledronate χορήγηση έχει συγκεκριμένες renal/mineral prerequisites που πρέπει να επιβεβαιωθούν πριν από την έγχυση.",
        "Verify renal function, calcium and vitamin-D safety before zoledronate; do not infer clearance from missing/stale data.",
        ["NOGG_2024_SECTION6#zoledronate_renal_hypocalcaemia", "EMA_ACLASTA_2026_04_20#aclasta_severe_renal", "EMA_ACLASTA_2026_04_20#aclasta_hypocalcaemia"], "regulatory_safety", "checklist_only"
      ));
    }

    if (decisionType === "start" && selectedAgent === "romosozumab") {
      push(makeContribution(
        "OST_G2_R18_ROMOSOZUMAB_START_SAFETY", ["laboratory_monitoring", "treatment_decision", "communication"], "critical_safety", 5,
        "Η έναρξη romosozumab απαιτεί CV και calcium safety review πριν από την πρώτη δόση.",
        "Verify prior MI/stroke, cardiovascular risk and hypocalcaemia/calcium-vitamin-D context; this is a checklist, not automated clearance.",
        ["NOGG_2024_SECTION6#romosozumab_cv_hypocalcaemia", "EMA_EVENITY_CURRENT#evenity_mi_stroke_hypocalcaemia"], "regulatory_safety", "checklist_only"
      ));
    }

    if (decisionType === "start" && selectedAgent === "teriparatide") {
      push(makeContribution(
        "OST_G2_R19_TERIPARATIDE_START_SAFETY", ["laboratory_monitoring", "treatment_decision"], "critical_safety", 5,
        "Η έναρξη teriparatide απαιτεί συγκεκριμένο metabolic/safety screen και σαφές course plan.",
        "Verify calcium/PTH/renal and major contraindication context before teriparatide; this is a checklist, not automated clearance.",
        ["NOGG_2024_SECTION6#teriparatide_safety", "EMA_FORSTEO_2026_01_28#forsteo_course_24m", "EMA_FORSTEO_2026_01_28#forsteo_major_restrictions"], "regulatory_safety", "checklist_only"
      ));
    }

    const activeRomo = exactActiveEpisodeForAgent(ctx.treatment_snapshot, "romosozumab");
    const activeTeri = exactActiveEpisodeForAgent(ctx.treatment_snapshot, "teriparatide");
    const romoCourseReached = activeRomo && isIsoDate(ctx.encounter_date) && ctx.encounter_date >= addCalendarMonths(activeRomo.start_date, 12);
    const teriCourseReached = activeTeri && isIsoDate(ctx.encounter_date) && ctx.encounter_date >= addCalendarMonths(activeTeri.start_date, 24);
    if (["post_romosozumab", "post_teriparatide"].includes(transitionType) || romoCourseReached || teriCourseReached) {
      push(makeContribution(
        "OST_G2_R20_POST_ANABOLIC_CONSOLIDATION", ["transition_safety", "treatment_decision", "followup_tasks"], "critical_safety", 8,
        "Η ολοκλήρωση anabolic/romosozumab course απαιτεί άμεσο antiresorptive follow-on plan για διατήρηση του οφέλους.",
        "Ensure an antiresorptive follow-on plan is explicit without delay after teriparatide/romosozumab completion.",
        ["NOGG_2024_SECTION6#post_anabolic_antiresorptive"], "strong", "activate_v1"
      ));
    }

    for (const agent of ORAL_BISPHOSPHONATES) {
      const episode = exactActiveEpisodeForAgent(ctx.treatment_snapshot, agent);
      if (!episode || !isIsoDate(ctx.encounter_date)) continue;
      const review5y = addCalendarYears(episode.start_date, 5);
      if (review5y && ctx.encounter_date >= review5y) {
        push(makeContribution(
          "OST_G2_R21_ORAL_BISPHOSPHONATE_5Y_REASSESS", ["formal_risk", "dxa", "treatment_decision"], "milestone_due", 25,
          "Έχει συμπληρωθεί evidence-backed oral-bisphosphonate review point· χρειάζεται επανεκτίμηση αντί για αυτόματη συνέχιση ή αυτόματο holiday.",
          "Reassess fracture risk/adherence/secondary context and ongoing strategy after at least five years of oral bisphosphonate therapy.",
          ["NOGG_2024_SECTION7#oral_bp_review_5y"], "strong", "activate_v1"
        ));
      }
    }

    const zolEpisode = exactActiveEpisodeForAgent(ctx.treatment_snapshot, "zoledronate");
    if (zolEpisode && isIsoDate(ctx.encounter_date)) {
      const review3y = addCalendarYears(zolEpisode.start_date, 3);
      if (review3y && ctx.encounter_date >= review3y) {
        push(makeContribution(
          "OST_G2_R22_ZOLEDRONATE_3Y_REASSESS", ["formal_risk", "dxa", "treatment_decision"], "milestone_due", 25,
          "Έχει συμπληρωθεί evidence-backed IV-zoledronate review point· η συνέχιση ή παύση πρέπει να βασιστεί σε επανεκτίμηση κινδύνου.",
          "Reassess fracture risk and ongoing strategy after at least three years of IV zoledronate therapy.",
          ["NOGG_2024_SECTION7#iv_zoledronate_review_3y"], "strong", "activate_v1"
        ));
      }
    }

    if (["initial_assessment_new_or_uncertain_diagnosis", "initial_assessment_known_osteoporosis_or_osteopenia", "post_fragility_fracture"].includes(archetype)) {
      push(makeContribution(
        "OST_G2_R23_TARGETED_LIFESTYLE_COMMUNICATION", ["communication"], "contextual", 60,
        "Η αρχική/μετά-κάταγμα επίσκεψη είναι κατάλληλο σημείο για στοχευμένη πρόληψη πτώσεων, άσκηση και bone-health nutrition.",
        "Address exercise, falls prevention and adequate calcium/vitamin-D intake as relevant rather than repeating generic counselling at every visit.",
        ["NOGG_2024_SECTION5#calcium_vitamin_d_repletion", "NOGG_2024_SECTION5#exercise"], "guideline_recommendation", "activate_v1"
      ));
    }

    if (denosumabReliable && activeAgents.has("denosumab") && denosumabCount >= 2 && isIsoDate(lastDenosumab) && isIsoDate(ctx.encounter_date)) {
      const sevenMonthPoint = addCalendarMonths(lastDenosumab, 7);
      if (sevenMonthPoint && ctx.encounter_date > sevenMonthPoint) {
        push(makeContribution(
          "OST_G2_R24_DENOSUMAB_GT7M_REBOUND_ESCALATION", ["administrations", "transition_safety", "followup_tasks"], "critical_safety", 4,
          "Έχουν προηγηθεί ≥2 αξιόπιστα τεκμηριωμένες denosumab δόσεις και έχουν περάσει >7 μήνες από την τελευταία πραγματική δόση· το NOGG επισημαίνει ουσιαστικό rebound high-turnover/vertebral-fracture κίνδυνο.",
          "Escalate a >7-month interval after at least two documented doses as a rebound-risk safety problem, without inferring a missing dose or auto-selecting rescue therapy.",
          ["NOGG_2024_FAQ#denosumab_gt7m_rebound_risk"], "nogg_practical_guidance", "activate_v1"
        ));
      }
    }

    if (decisionType === "start" && ORAL_BISPHOSPHONATES.has(selectedAgent)) {
      push(makeContribution(
        "OST_G2_R25_ORAL_BISPHOSPHONATE_START_SAFETY_USE", ["treatment_decision", "communication"], "agent_specific", 15,
        "Η έναρξη oral bisphosphonate απαιτεί έλεγχο oesophageal/upright/renal-mineral suitability και σαφείς οδηγίες σωστής λήψης.",
        "Verify oral-bisphosphonate suitability and correct administration instructions; this is a checklist, not automated clearance.",
        ["NOGG_2024_SECTION6#oral_bisphosphonate_use_and_contraindications"], "product_use_and_safety_guidance", "checklist_only"
      ));
    }

    for (const agent of ORAL_BISPHOSPHONATES) {
      const episode = exactActiveEpisodeForAgent(ctx.treatment_snapshot, agent);
      if (!episode || !isIsoDate(ctx.encounter_date)) continue;
      const exposureDays = daysBetween(episode.start_date, ctx.encounter_date);
      if (exposureDays !== null && exposureDays >= 84 && exposureDays <= 112) {
        push(makeContribution(
          "OST_G2_R26_ORAL_BISPHOSPHONATE_EARLY_REVIEW", ["treatment_history", "communication", "followup_tasks"], "milestone_due", 30,
          "Είναι το NOGG 12–16 εβδομάδων early review window για tolerance/adherence και σωστή λήψη oral bisphosphonate.",
          "Review early tolerance, adherence and correct administration after oral-bisphosphonate initiation.",
          ["NOGG_2024_FAQ#oral_bp_early_tolerance_adherence_review"], "nogg_practical_guidance", "activate_v1"
        ));
      }
    }

    return out;
  }

  function mergeEvidenceContributions(basePlan, contributions) {
    const base = basePlan && typeof basePlan === "object" ? basePlan : { ordered_cards: [], rule_trace: [] };
    const states = new Map();

    asArray(base.ordered_cards).forEach(item => {
      states.set(item.card_id, {
        ...item,
        reason_codes: asArray(item.reason_codes).slice(),
        why_now_reasons: asArray(item.why_now_reasons).length ? asArray(item.why_now_reasons).slice() : (clean(item.why_now) ? [clean(item.why_now)] : []),
        evidence_rules: asArray(item.evidence_rules).slice()
      });
    });

    asArray(contributions).forEach(contribution => {
      asArray(contribution?.domains).forEach(domain => {
        if (!DOMAIN_ORDER.includes(domain)) return;
        const existing = states.get(domain) || {
          card_id: domain,
          visibility: "surfaced",
          priority: Number(contribution.priority || 999),
          reason_codes: [],
          why_now_reasons: [],
          prior_data_state: "uncertain",
          capture_state: "unresolved",
          provisional_candidate_count: 0,
          conflict_count: 0,
          critical_unresolved: false,
          evidence_rules: []
        };
        existing.priority = Math.min(Number(existing.priority || 999), Number(contribution.priority || 999));
        const reasonCode = `G2:${contribution.rule_id}`;
        if (!existing.reason_codes.includes(reasonCode)) existing.reason_codes.push(reasonCode);
        if (clean(contribution.why_now) && !existing.why_now_reasons.includes(clean(contribution.why_now))) existing.why_now_reasons.push(clean(contribution.why_now));
        if (!existing.evidence_rules.some(rule => rule.rule_id === contribution.rule_id)) existing.evidence_rules.push({ ...contribution });
        states.set(domain, existing);
      });
    });

    const orderedCards = Array.from(states.values())
      .map(item => ({ ...item, why_now: item.why_now_reasons.join(" · ") }))
      .sort((a, b) => Number(a.priority) - Number(b.priority) || DOMAIN_ORDER.indexOf(a.card_id) - DOMAIN_ORDER.indexOf(b.card_id));

    const evidenceTrace = asArray(contributions).flatMap(contribution => asArray(contribution.domains).map(domain => ({
      matched: true,
      reason_code: `G2:${contribution.rule_id}`,
      rule_id: contribution.rule_id,
      rule_class: contribution.rule_class,
      card_id: domain,
      source_refs: asArray(contribution.source_refs),
      activation_mode: contribution.activation_mode
    })));

    return {
      ...base,
      ordered_cards: orderedCards,
      critical_unresolved: orderedCards.filter(item => item.critical_unresolved).map(item => item.card_id),
      rule_trace: [...asArray(base.rule_trace), ...evidenceTrace],
      evidence_contributions: asArray(contributions).map(item => ({ ...item }))
    };
  }

  window.BaselineOsteoporosisEvidenceGuidance = Object.freeze({
    BLOCKED_RULE_IDS,
    ORAL_BISPHOSPHONATES: Object.freeze(Array.from(ORAL_BISPHOSPHONATES)),
    PARENTERAL_AGENTS: Object.freeze(Array.from(PARENTERAL_AGENTS)),
    addCalendarMonths,
    addCalendarYears,
    daysBetween,
    buildEvidenceContext,
    evaluateEvidenceGuidance,
    mergeEvidenceContributions
  });
})();
