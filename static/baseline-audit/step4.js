(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const AGENTS = [
    ["none", "Καμία"],
    ["alendronate", "Alendronate"],
    ["risedronate", "Risedronate"],
    ["ibandronate_oral", "Ibandronate oral"],
    ["zoledronate", "Zoledronate"],
    ["ibandronate_iv", "Ibandronate IV"],
    ["denosumab", "Denosumab"],
    ["teriparatide", "Teriparatide"],
    ["romosozumab", "Romosozumab"],
    ["raloxifene", "Raloxifene"],
    ["hormone_therapy", "Hormone therapy"],
    ["other", "Άλλο"]
  ];

  function options(list, selected = "") {
    return list.map(([value, label]) => `<option value="${value}" ${value === selected ? "selected" : ""}>${label}</option>`).join("");
  }

  function injectAssets() {
    if (!document.querySelector('link[data-step4-style]')) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = "./step4.css";
      link.dataset.step4Style = "true";
      document.head.appendChild(link);
    }

    const panel = $('[data-step-panel="4"]');
    if (!panel) return;
    panel.classList.remove("placeholder-panel");
    panel.innerHTML = `
      <div class="context-note step4-context" id="step4ContextNote">
        <strong>Step 4 — Απόφαση & Πλάνο:</strong> καταγράφουμε τι πραγματικά αποφασίστηκε, γιατί, και τι πρέπει να γίνει μετά — χωρίς live treatment coaching στο baseline.
      </div>

      <div class="step4-grid">
        <article class="card step4-card span-2">
          <div class="card-heading"><div><h2>Θεραπευτικό ιστορικό</h2><p>Exact dates όταν είναι γνωστές. Αν όχι, κράτησε approximate duration χωρίς να εφευρίσκεις ημερομηνίες.</p></div><button class="btn secondary" id="s4AddEpisode" type="button">＋ Episode</button></div>
          <div id="s4Episodes" class="repeat-stack"></div>
          <div id="s4EpisodesEmpty" class="empty-repeat">Δεν έχει προστεθεί θεραπευτικό episode.</div>
        </article>

        <article class="card step4-card span-2">
          <div class="card-heading"><div><h2>Administrations / due dates</h2><p>Για denosumab, IV bisphosphonate, romosozumab ή άλλο time-critical administration.</p></div><button class="btn secondary" id="s4AddAdministration" type="button">＋ Administration</button></div>
          <div id="s4Administrations" class="repeat-stack"></div>
          <div id="s4AdministrationsEmpty" class="empty-repeat">Δεν υπάρχουν administration events.</div>
        </article>

        <article class="card step4-card span-2">
          <div class="card-heading"><div><h2>Σημερινή κλινική απόφαση</h2><p>Η απόφαση του κλινικού, όχι recommendation του συστήματος.</p></div></div>
          <div class="s4-grid four">
            <label><span>Decision</span><select id="s4DecisionType"><option value="">—</option><option value="start">Έναρξη</option><option value="continue">Συνέχιση</option><option value="stop">Διακοπή</option><option value="switch">Αλλαγή</option><option value="defer">Αναβολή</option><option value="no_drug_treatment">Χωρίς φαρμακευτική θεραπεία</option><option value="complete_course">Ολοκλήρωση course</option><option value="consolidate">Consolidation</option><option value="refer">Παραπομπή</option><option value="uncertain">Αβέβαιο</option></select></label>
            <label><span>Selected agent</span><select id="s4SelectedAgent"><option value="">—</option>${options(AGENTS)}</select></label>
            <label><span>Safety/contraindication review</span><select id="s4SafetyReview"><option value="">—</option><option value="done">Έγινε</option><option value="not_done">Δεν έγινε</option><option value="not_applicable">N/A</option><option value="uncertain">Αβέβαιο</option></select></label>
            <label><span>Sequencing review</span><select id="s4SequencingReview"><option value="">—</option><option value="done">Έγινε</option><option value="not_done">Δεν έγινε</option><option value="not_applicable">N/A</option><option value="uncertain">Αβέβαιο</option></select></label>
          </div>

          <div class="s4-section-label">Reason(s) for decision</div>
          <div class="chip-checks compact" id="s4DecisionReasons">
            <label><input type="checkbox" value="fracture_risk" />Fracture risk</label>
            <label><input type="checkbox" value="very_high_risk" />Very-high risk</label>
            <label><input type="checkbox" value="new_fragility_fracture" />Νέο κάταγμα</label>
            <label><input type="checkbox" value="fracture_on_treatment" />Κάταγμα υπό θεραπεία</label>
            <label><input type="checkbox" value="inadequate_response" />Suboptimal response</label>
            <label><input type="checkbox" value="adherence_problem" />Adherence</label>
            <label><input type="checkbox" value="adverse_effect_or_intolerance" />Ανεπιθύμητη ενέργεια</label>
            <label><input type="checkbox" value="contraindication" />Contraindication</label>
            <label><input type="checkbox" value="treatment_duration_or_review_point" />Duration/review point</label>
            <label><input type="checkbox" value="denosumab_exit_or_delay_risk" />Denosumab exit/delay</label>
            <label><input type="checkbox" value="anabolic_or_romosozumab_completion" />Post-anabolic/romosozumab</label>
            <label><input type="checkbox" value="patient_preference" />Patient preference</label>
            <label><input type="checkbox" value="cost_or_access" />Access/cost</label>
            <label><input type="checkbox" value="other" />Άλλο</label>
          </div>

          <div class="s4-grid three">
            <label><span>Patient preference documented</span><select id="s4PreferenceDocumented"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option><option value="uncertain">Αβέβαιο</option></select></label>
            <label><span>Patient accepted plan</span><select id="s4PatientAccepted"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="undecided">Δεν αποφάσισε</option><option value="not_applicable">N/A</option><option value="unknown">Άγνωστο</option></select></label>
            <label><span>Confidence % <small>(προαιρετικό)</small></span><input id="s4Confidence" type="number" min="0" max="100" step="5" /></label>
          </div>

          <label class="full-field"><span>Κλινικό rationale <small>(σύντομο, χωρίς αναγνωριστικά)</small></span><textarea id="s4Rationale" rows="3" maxlength="900"></textarea></label>
        </article>

        <article class="card step4-card">
          <div class="card-heading"><div><h2>Transition / sequencing safety</h2><p>Ενεργοποιείται όταν υπάρχει transition, exit ή consolidation.</p></div></div>
          <div class="field-stack">
            <label><span>Transition relevant;</span><select id="s4TransitionRelevant"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option></select></label>
            <div id="s4TransitionDetails" hidden>
              <label><span>Type</span><select id="s4TransitionType"><option value="">—</option><option value="denosumab_exit">Denosumab exit</option><option value="post_teriparatide">Post-teriparatide</option><option value="post_romosozumab">Post-romosozumab</option><option value="bisphosphonate_holiday_or_restart">Bisphosphonate holiday/restart</option><option value="other">Άλλο</option></select></label>
              <label><span>Prior agent — last dose/end</span><input id="s4PriorAgentEnd" type="date" /></label>
              <label><span>Next agent</span><select id="s4NextAgent"><option value="">—</option>${options(AGENTS)}</select></label>
              <label><span>Planned next-agent date</span><input id="s4NextAgentDate" type="date" /></label>
              <label><span>Explicit transition plan;</span><select id="s4TransitionExplicit"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
              <label><span>Unresolved safety issue;</span><select id="s4SafetyUnresolved"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option></select></label>
              <label><span>Safety note <small>(προαιρετικό)</small></span><textarea id="s4SafetyNote" rows="2" maxlength="500"></textarea></label>
            </div>
          </div>
        </article>

        <article class="card step4-card">
          <div class="card-heading"><div><h2>Follow-up / Care Tasks</h2><p>Αργότερα αυτά θα γίνουν reusable CareTask objects.</p></div><button class="btn secondary" id="s4AddTask" type="button">＋ Task</button></div>
          <div id="s4Tasks" class="repeat-stack"></div>
          <div id="s4TasksEmpty" class="empty-repeat">Δεν έχουν προστεθεί tasks.</div>
        </article>

        <article class="card step4-card span-2 close-card">
          <div class="card-heading"><div><h2>Κλείσιμο Step 4</h2><p>Το baseline καταγράφει αν το πλάνο έκλεισε και αν έμεινε critical εκκρεμότητα.</p></div></div>
          <div class="s4-grid three">
            <label><span>Plan complete;</span><select id="s4PlanComplete"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option></select></label>
            <label><span>Unresolved critical item;</span><select id="s4CriticalUnresolved"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option></select></label>
            <label><span>Σύντομη σημείωση <small>(μόνο αν χρειάζεται)</small></span><input id="s4CriticalNote" type="text" maxlength="500" /></label>
          </div>
        </article>
      </div>`;
  }

  function getCases() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch { return []; } }
  function setCases(cases) { localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
  function activeUuid() { return localStorage.getItem(ACTIVE_KEY) || ""; }
  function getActiveCase() { const id = activeUuid(); return getCases().find((item) => item.internal_uuid === id) || null; }

  function defaultState() {
    return {
      treatment_episodes: [], administrations: [],
      decision: { type: "", selected_agent: "", safety_review: "", sequencing_review: "", reasons: [], preference_documented: "", patient_accepted: "", confidence_percent: null, rationale: "" },
      transition: { relevant: "", type: "", prior_end_date: "", next_agent: "", next_agent_date: "", explicit_plan: "", unresolved_safety: "", note: "" },
      tasks: [], close: { plan_complete: "", unresolved_critical: "", note: "" }, updated_at: null
    };
  }

  function normalize(raw) {
    const base = defaultState();
    if (!raw || typeof raw !== "object") return base;
    return {
      ...base, ...raw,
      treatment_episodes: Array.isArray(raw.treatment_episodes) ? raw.treatment_episodes : [],
      administrations: Array.isArray(raw.administrations) ? raw.administrations : [],
      tasks: Array.isArray(raw.tasks) ? raw.tasks : [],
      decision: { ...base.decision, ...(raw.decision || {}) },
      transition: { ...base.transition, ...(raw.transition || {}) },
      close: { ...base.close, ...(raw.close || {}) }
    };
  }

  let state = defaultState();
  let loadedUuid = "";

  function archetypeLabel(value) {
    const map = {
      initial_assessment_new_or_uncertain_diagnosis: "Αρχική αξιολόγηση — νέα/αβέβαιη διάγνωση",
      initial_assessment_known_osteoporosis_or_osteopenia: "Αρχική αξιολόγηση — γνωστή οστεοπόρωση/οστεοπενία",
      routine_followup_stable: "Routine follow-up — stable",
      treatment_start: "Έναρξη θεραπείας",
      treatment_continuation_or_due_monitoring: "Συνέχιση / due monitoring",
      treatment_change_or_transition: "Αλλαγή / transition",
      post_fragility_fracture: "Μετά από κάταγμα ευθραυστότητας",
      fracture_on_treatment: "Κάταγμα υπό θεραπεία",
      adverse_effect_or_intolerance: "Ανεπιθύμητη ενέργεια / δυσανεξία",
      treatment_completion_or_consolidation: "Ολοκλήρωση / consolidation"
    };
    return map[value] || "Το applicability προσαρμόζεται στον τύπο της σημερινής επίσκεψης.";
  }

  function episodeTemplate(item = {}) {
    const id = item.id || crypto.randomUUID();
    return `<div class="repeat-item" data-episode-id="${id}">
      <button class="repeat-remove" type="button" data-remove-episode="${id}" aria-label="Αφαίρεση">×</button>
      <div class="s4-grid six">
        <label><span>Agent</span><select data-k="agent"><option value="">—</option>${options(AGENTS, item.agent)}</select></label>
        <label><span>Status</span><select data-k="status"><option value="">—</option><option value="planned" ${item.status === "planned" ? "selected" : ""}>Planned</option><option value="active" ${item.status === "active" ? "selected" : ""}>Active</option><option value="completed" ${item.status === "completed" ? "selected" : ""}>Completed</option><option value="stopped" ${item.status === "stopped" ? "selected" : ""}>Stopped</option><option value="holiday" ${item.status === "holiday" ? "selected" : ""}>Holiday</option><option value="unknown" ${item.status === "unknown" ? "selected" : ""}>Unknown</option></select></label>
        <label><span>Start</span><input data-k="start_date" type="date" value="${item.start_date || ""}" /></label>
        <label><span>End</span><input data-k="end_date" type="date" value="${item.end_date || ""}" /></label>
        <label><span>Duration y <small>αν dates unknown</small></span><input data-k="duration_years" type="number" min="0" max="50" step="0.1" value="${item.duration_years ?? ""}" /></label>
        <label><span>Adherence</span><select data-k="adherence"><option value="">—</option><option value="good" ${item.adherence === "good" ? "selected" : ""}>Good</option><option value="partial" ${item.adherence === "partial" ? "selected" : ""}>Partial</option><option value="poor" ${item.adherence === "poor" ? "selected" : ""}>Poor</option><option value="unknown" ${item.adherence === "unknown" ? "selected" : ""}>Unknown</option><option value="not_applicable" ${item.adherence === "not_applicable" ? "selected" : ""}>N/A</option></select></label>
        <label><span>Tolerance</span><select data-k="tolerance"><option value="">—</option><option value="good" ${item.tolerance === "good" ? "selected" : ""}>Good</option><option value="minor_adverse_effects" ${item.tolerance === "minor_adverse_effects" ? "selected" : ""}>Minor AE</option><option value="significant_adverse_effects" ${item.tolerance === "significant_adverse_effects" ? "selected" : ""}>Significant AE</option><option value="unknown" ${item.tolerance === "unknown" ? "selected" : ""}>Unknown</option><option value="not_applicable" ${item.tolerance === "not_applicable" ? "selected" : ""}>N/A</option></select></label>
        <label><span>Fracture on episode</span><select data-k="fractures_on_episode"><option value="">—</option><option value="yes" ${item.fractures_on_episode === "yes" ? "selected" : ""}>Ναι</option><option value="no" ${item.fractures_on_episode === "no" ? "selected" : ""}>Όχι</option><option value="unknown" ${item.fractures_on_episode === "unknown" ? "selected" : ""}>Άγνωστο</option></select></label>
        <label><span>Response</span><select data-k="response_context"><option value="">—</option><option value="appropriate" ${item.response_context === "appropriate" ? "selected" : ""}>Appropriate</option><option value="suboptimal" ${item.response_context === "suboptimal" ? "selected" : ""}>Suboptimal</option><option value="uncertain" ${item.response_context === "uncertain" ? "selected" : ""}>Uncertain</option><option value="not_assessed" ${item.response_context === "not_assessed" ? "selected" : ""}>Not assessed</option></select></label>
      </div>
      <div class="s4-grid two"><label><span>Reason started <small>(optional)</small></span><input data-k="reason_started" type="text" maxlength="300" value="${escapeHtml(item.reason_started || "")}" /></label><label><span>Reason stopped/switched <small>(optional)</small></span><input data-k="reason_stopped" type="text" maxlength="300" value="${escapeHtml(item.reason_stopped || "")}" /></label></div>
    </div>`;
  }

  function administrationTemplate(item = {}) {
    const id = item.id || crypto.randomUUID();
    return `<div class="repeat-item" data-admin-id="${id}"><button class="repeat-remove" type="button" data-remove-admin="${id}">×</button><div class="s4-grid five">
      <label><span>Agent</span><select data-k="agent"><option value="">—</option>${options(AGENTS, item.agent)}</select></label>
      <label><span>Scheduled</span><input data-k="scheduled_date" type="date" value="${item.scheduled_date || ""}" /></label>
      <label><span>Actual</span><input data-k="actual_date" type="date" value="${item.actual_date || ""}" /></label>
      <label><span>Status</span><select data-k="status"><option value="">—</option><option value="done" ${item.status === "done" ? "selected" : ""}>Done</option><option value="due" ${item.status === "due" ? "selected" : ""}>Due</option><option value="overdue" ${item.status === "overdue" ? "selected" : ""}>Overdue</option><option value="missed" ${item.status === "missed" ? "selected" : ""}>Missed</option><option value="planned" ${item.status === "planned" ? "selected" : ""}>Planned</option><option value="not_applicable" ${item.status === "not_applicable" ? "selected" : ""}>N/A</option></select></label>
      <label><span>Next due</span><input data-k="next_due_date" type="date" value="${item.next_due_date || ""}" /></label>
    </div></div>`;
  }

  function taskTemplate(item = {}) {
    const id = item.id || crypto.randomUUID();
    const types = [["lab","Labs"],["DXA","DXA"],["administration","Administration"],["followup_visit","Follow-up visit"],["referral","Referral"],["VFA_or_imaging","VFA/imaging"],["adherence_check","Adherence check"],["exercise_or_falls","Exercise/falls"],["nutrition","Nutrition"],["other","Other"]];
    return `<div class="repeat-item compact" data-task-id="${id}"><button class="repeat-remove" type="button" data-remove-task="${id}">×</button><div class="s4-grid four">
      <label><span>Task</span><select data-k="type"><option value="">—</option>${options(types, item.type)}</select></label>
      <label><span>Due date</span><input data-k="due_date" type="date" value="${item.due_date || ""}" /></label>
      <label><span>Timeframe text</span><input data-k="timeframe_text" type="text" maxlength="100" value="${escapeHtml(item.timeframe_text || "")}" placeholder="π.χ. σε 6 μήνες" /></label>
      <label><span>Status</span><select data-k="status"><option value="planned" ${!item.status || item.status === "planned" ? "selected" : ""}>Planned</option><option value="already_done" ${item.status === "already_done" ? "selected" : ""}>Already done</option><option value="not_applicable" ${item.status === "not_applicable" ? "selected" : ""}>N/A</option></select></label>
    </div></div>`;
  }

  function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c])); }

  function renderRepeats() {
    $("#s4Episodes").innerHTML = state.treatment_episodes.map(episodeTemplate).join("");
    $("#s4EpisodesEmpty").hidden = state.treatment_episodes.length > 0;
    $("#s4Administrations").innerHTML = state.administrations.map(administrationTemplate).join("");
    $("#s4AdministrationsEmpty").hidden = state.administrations.length > 0;
    $("#s4Tasks").innerHTML = state.tasks.map(taskTemplate).join("");
    $("#s4TasksEmpty").hidden = state.tasks.length > 0;
  }

  function collectRepeat(container, attr) {
    return $$(`[${attr}]`, container).map((row) => {
      const out = { id: row.getAttribute(attr) };
      $$('[data-k]', row).forEach((node) => {
        let value = node.value;
        if (node.type === "number") value = value === "" ? null : Number(value);
        out[node.dataset.k] = value;
      });
      return out;
    });
  }

  function collect() {
    state.treatment_episodes = collectRepeat($("#s4Episodes"), "data-episode-id");
    state.administrations = collectRepeat($("#s4Administrations"), "data-admin-id");
    state.tasks = collectRepeat($("#s4Tasks"), "data-task-id");
    state.decision = {
      type: $("#s4DecisionType").value, selected_agent: $("#s4SelectedAgent").value,
      safety_review: $("#s4SafetyReview").value, sequencing_review: $("#s4SequencingReview").value,
      reasons: $$('#s4DecisionReasons input:checked').map((x) => x.value),
      preference_documented: $("#s4PreferenceDocumented").value, patient_accepted: $("#s4PatientAccepted").value,
      confidence_percent: $("#s4Confidence").value === "" ? null : Number($("#s4Confidence").value), rationale: $("#s4Rationale").value.trim()
    };
    state.transition = {
      relevant: $("#s4TransitionRelevant").value, type: $("#s4TransitionType").value, prior_end_date: $("#s4PriorAgentEnd").value,
      next_agent: $("#s4NextAgent").value, next_agent_date: $("#s4NextAgentDate").value, explicit_plan: $("#s4TransitionExplicit").value,
      unresolved_safety: $("#s4SafetyUnresolved").value, note: $("#s4SafetyNote").value.trim()
    };
    state.close = { plan_complete: $("#s4PlanComplete").value, unresolved_critical: $("#s4CriticalUnresolved").value, note: $("#s4CriticalNote").value.trim() };
  }

  function setValue(id, value) { const node = $(id); if (node) node.value = value ?? ""; }

  function hydrate() {
    const active = getActiveCase();
    const context = $("#step4ContextNote");
    if (context) context.innerHTML = `<strong>Step 4 — Απόφαση & Πλάνο:</strong> ${archetypeLabel(active?.encounter_archetype || "")}`;
    renderRepeats();
    const d = state.decision;
    setValue("#s4DecisionType", d.type); setValue("#s4SelectedAgent", d.selected_agent); setValue("#s4SafetyReview", d.safety_review); setValue("#s4SequencingReview", d.sequencing_review);
    $$('#s4DecisionReasons input').forEach((x) => x.checked = d.reasons.includes(x.value));
    setValue("#s4PreferenceDocumented", d.preference_documented); setValue("#s4PatientAccepted", d.patient_accepted); setValue("#s4Confidence", d.confidence_percent); setValue("#s4Rationale", d.rationale);
    const t = state.transition;
    setValue("#s4TransitionRelevant", t.relevant); setValue("#s4TransitionType", t.type); setValue("#s4PriorAgentEnd", t.prior_end_date); setValue("#s4NextAgent", t.next_agent); setValue("#s4NextAgentDate", t.next_agent_date); setValue("#s4TransitionExplicit", t.explicit_plan); setValue("#s4SafetyUnresolved", t.unresolved_safety); setValue("#s4SafetyNote", t.note);
    setValue("#s4PlanComplete", state.close.plan_complete); setValue("#s4CriticalUnresolved", state.close.unresolved_critical); setValue("#s4CriticalNote", state.close.note);
    syncVisibility();
  }

  function syncVisibility() { const box = $("#s4TransitionDetails"); if (box) box.hidden = $("#s4TransitionRelevant").value !== "yes"; }

  function loadState() {
    const active = getActiveCase();
    loadedUuid = active?.internal_uuid || activeUuid();
    state = normalize(active?.step4);
    hydrate();
  }

  function persist() {
    if (!loadedUuid) loadedUuid = activeUuid();
    if (!loadedUuid) return;
    collect();
    state.updated_at = new Date().toISOString();
    const cases = getCases();
    const idx = cases.findIndex((item) => item.internal_uuid === loadedUuid);
    if (idx < 0) return;
    cases[idx] = { ...cases[idx], step4: state };
    setCases(cases);
  }

  function addEpisode() { collect(); state.treatment_episodes.push({ id: crypto.randomUUID() }); renderRepeats(); persist(); }
  function addAdmin() { collect(); state.administrations.push({ id: crypto.randomUUID() }); renderRepeats(); persist(); }
  function addTask() { collect(); state.tasks.push({ id: crypto.randomUUID(), status: "planned" }); renderRepeats(); persist(); }

  function bind() {
    const panel = $('[data-step-panel="4"]');
    if (!panel) return;
    panel.addEventListener("input", () => { syncVisibility(); persist(); });
    panel.addEventListener("change", () => { syncVisibility(); persist(); });
    $("#s4AddEpisode").addEventListener("click", addEpisode);
    $("#s4AddAdministration").addEventListener("click", addAdmin);
    $("#s4AddTask").addEventListener("click", addTask);
    panel.addEventListener("click", (event) => {
      const ep = event.target.closest("[data-remove-episode]");
      const ad = event.target.closest("[data-remove-admin]");
      const task = event.target.closest("[data-remove-task]");
      if (ep) { collect(); state.treatment_episodes = state.treatment_episodes.filter((x) => x.id !== ep.dataset.removeEpisode); renderRepeats(); persist(); }
      if (ad) { collect(); state.administrations = state.administrations.filter((x) => x.id !== ad.dataset.removeAdmin); renderRepeats(); persist(); }
      if (task) { collect(); state.tasks = state.tasks.filter((x) => x.id !== task.dataset.removeTask); renderRepeats(); persist(); }
    });
    $$(".step-tab").forEach((button) => button.addEventListener("click", () => { if (button.dataset.step === "4") setTimeout(loadState, 0); }));
    document.addEventListener("click", (event) => { if (event.target.closest("[data-load-case]") || event.target.closest('[data-nav-action="new-case"]')) setTimeout(loadState, 0); });
    ["#saveTopBtn", "#saveDraftBtn", "#finishVisitBtn"].forEach((selector) => $(selector)?.addEventListener("click", () => setTimeout(persist, 0)));
  }

  injectAssets();
  bind();
  loadState();
})();