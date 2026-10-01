(() => {
  "use strict";
  const $ = selector => document.querySelector(selector);
  const row = (id, date, step4) => ({ encounter_id: id, encounter_date: date, status: "completed", payload: { internal_uuid: id, step4 } });
  const episode = { id: "ep1", agent: "denosumab", status: "active", start_date: "2025-01-01" };
  const admin = (id, actual, scheduled = "") => ({ id, agent: "denosumab", actual_date: actual, scheduled_date: scheduled, status: actual ? "done" : "planned" });
  const base = () => [
    row("synthetic-e1", "2025-07-01", { treatment_episodes: [episode], administrations: [admin("a1", "2025-07-01")], decision: { type: "continue", selected_agent: "denosumab" }, tasks: [] }),
    row("synthetic-e2", "2026-01-01", { treatment_episodes: [episode], administrations: [admin("a2", "2026-01-01")], decision: { type: "continue", selected_agent: "denosumab" }, tasks: [] })
  ];
  let history = [];
  let today = "2026-07-01";
  let sourceId = "";
  const labels = { due: "στην ώρα της", delayed: "καθυστέρηση", transition: "μετάβαση", conflict: "αντικρουόμενο ιστορικό", continuity: "αβέβαιη συνέχεια" };

  function scenario(name) {
    history = base();
    today = name === "delayed" ? "2026-08-15" : "2026-07-01";
    if (name === "transition") history.push(row("synthetic-e3", "2026-02-01", { treatment_episodes: [{ ...episode, status: "stopped", end_date: "2026-02-01" }], administrations: [], decision: { type: "switch", selected_agent: "zoledronate" }, transition: { type: "denosumab_exit", next_agent: "zoledronate", next_agent_date: "2026-07-01" }, tasks: [] }));
    if (name === "conflict") history.push(row("synthetic-e4", "2026-01-10", { treatment_episodes: [episode], administrations: [admin("a2", "2026-01-10")], tasks: [] }));
    if (name === "continuity") history.push(
      row("synthetic-e5", "2026-03-01", { treatment_episodes: [episode], administrations: [admin("planned", "", "2026-03-01")], tasks: [{ id: "t1", type: "dxa", due_date: "2026-07-01", status: "planned" }] }),
      row("synthetic-e6", "2026-04-01", { treatment_episodes: [episode], administrations: [admin("later", "2026-03-08")], tasks: [{ id: "t2", type: "dxa", due_date: "2026-06-01", status: "planned" }] })
    );
    sourceId = "";
    document.body.classList.remove("proto1-editor-open");
    render();
  }

  function renderSource() {
    const selected = history.find(item => item.encounter_id === sourceId) || history.at(-1);
    if (!selected) return;
    $("#sourceVisit").textContent = `${selected.encounter_date} · ${selected.encounter_id} · ολοκληρωμένη synthetic επίσκεψη`;
    $("#sourceEpisodes").textContent = (selected.payload.step4.treatment_episodes || []).map(item => `${item.agent} · ${item.status}`).join(" · ") || "Δεν υπάρχει εγγραφή";
    $("#sourceDecision").textContent = selected.payload.step4.decision ? `${selected.payload.step4.decision.type} · ${selected.payload.step4.decision.selected_agent || "—"}` : "Δεν έχει καταγραφεί";
    $("#sourceTasks").textContent = (selected.payload.step4.tasks || []).map(item => `${item.type} · ${item.due_date} · ${item.status}`).join(" · ") || "Δεν υπάρχουν εργασίες";
    const holder = $("#sourceAdministrations");
    holder.replaceChildren();
    for (const item of selected.payload.step4.administrations || []) {
      const line = document.createElement("label");
      line.className = "source-row";
      const name = document.createElement("span");
      name.textContent = `${item.agent} · ${item.status} · πραγματική ημερομηνία`;
      const input = document.createElement("input");
      input.type = "date";
      input.value = item.actual_date || "";
      input.setAttribute("aria-label", `Πραγματική ημερομηνία ${item.id}`);
      input.addEventListener("change", () => { item.actual_date = input.value; render(); });
      line.append(name, input);
      holder.append(line);
    }
    if (!holder.children.length) holder.textContent = "Δεν υπάρχει καταγεγραμμένη χορήγηση";
  }

  function render() {
    const g1 = window.BaselineProgressiveGuidanceCore;
    const g2 = window.BaselineOsteoporosisEvidenceGuidance;
    const g3 = window.BaselineOsteoporosisLongitudinalSummary;
    const current = { internal_uuid: "synthetic-today", encounter_date: today, encounter_archetype: "treatment_continuation_or_due_monitoring", step4: {} };
    const projection = g1.buildLongitudinalProjection(history, { currentInternalUuid: current.internal_uuid });
    const context = g1.buildEncounterContext(current, projection);
    const evidence = g2.buildEvidenceContext(current, projection, context, { historicalEncounters: history });
    const plan = g2.mergeEvidenceContributions(g1.buildVisitPlan(context), g2.evaluateEvidenceGuidance(evidence));
    const summary = g3.buildSummary({ encounters: history, projection, currentCase: current, historyStatus: "loaded" });
    renderSource();
    window.dispatchEvent(new CustomEvent("ost:guidance-rendered", { detail: { patientId: "SYNTHETIC-1", historicalEncounters: history, projection, summary, plan, current, historyStatus: "loaded" } }));
    $("#demoStatus").textContent = `Synthetic σενάριο: ${labels[$("#scenario").value]}. Η πηγή μπορεί να ανοίξει από την προβολή. Η αλλαγή ημερομηνίας επανυπολογίζει το αποτέλεσμα μόνο σε αυτή τη σελίδα.`;
  }

  window.ClinicalRegistry = {
    activeEncounterId: () => "synthetic-today",
    openEncounter: (id, step) => { sourceId = id; renderSource(); document.body.classList.add("proto1-editor-open"); $(`.step-tab[data-step="${step || "4"}"]`)?.click(); }
  };
  document.querySelectorAll(".step-tab").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll(".step-tab").forEach(node => node.classList.toggle("active", node === button));
    document.querySelectorAll(".step-panel").forEach(node => node.classList.toggle("active", node.dataset.stepPanel === button.dataset.step));
  }));
  $("#scenario").addEventListener("change", event => scenario(event.target.value));
  scenario("due");
})();
