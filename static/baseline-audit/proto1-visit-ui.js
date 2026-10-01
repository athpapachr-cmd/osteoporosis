(() => {
  "use strict";
  const $ = selector => document.querySelector(selector);
  const FOCUS = Object.freeze({ due: "Χορήγηση / παρακολούθηση σήμερα", delay: "Καθυστέρηση θεραπείας", transition: "Μετάβαση θεραπείας", verification: "Επιβεβαίωση ιστορικού", continuity: "Εκκρεμής συνέχεια", followup: "Επανεκτίμηση" });
  const EDITORS = Object.freeze({ actual: ["4", "#s4Administrations"], plan: ["4", "#s4Administrations"], treatment: ["4", "#s4Episodes"], decision: ["4", "#s4DecisionType"], obligation: ["4", "#s4Tasks"], investigation: ["3", "#s3DxaUsed"], visit: ["1", "#encounterArchetype"], labs: ["3", "#s3LabsDate"] });
  let latest = null;
  let chosenFocus = "";
  let lastSuggested = "";
  let focusPatientId = "";
  let dispositions = new Map();

  function el(tag, className = "", value = "") {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value) node.textContent = value;
    return node;
  }
  function action(label, fn, className = "") {
    const button = el("button", `proto1-action ${className}`, label);
    button.type = "button";
    button.addEventListener("click", fn);
    return button;
  }
  function line(parent, label, value, className = "") {
    const node = el("div", `proto1-line ${className}`);
    node.append(el("strong", "", label), el("span", "", value || "Άγνωστο"));
    parent.append(node);
    return node;
  }
  function openEditor(step, target, encounterId = null) {
    if (encounterId && window.ClinicalRegistry?.openEncounter) {
      const currentId = window.ClinicalRegistry.activeEncounterId?.();
      if (currentId && currentId !== encounterId) sessionStorage.setItem("ost.proto1.returnEncounter", currentId);
      else if (!currentId) {
        const currentUuid = localStorage.getItem("osteoporosis.baselineAuditPilot.activeCase.v1_1");
        if (currentUuid) sessionStorage.setItem("ost.proto1.returnUuid", currentUuid);
      }
      window.ClinicalRegistry.openEncounter(encounterId, step);
      return;
    }
    document.body.classList.add("proto1-editor-open");
    window.location.hash = `proto1-editor-${step}`;
    $(`.step-tab[data-step="${step}"]`)?.click();
    setTimeout(() => (target ? $(target) : $(`[data-step-panel="${step}"]`))?.scrollIntoView({ block: "center", behavior: "smooth" }), 30);
  }
  function openSource(item) {
    const [step, target] = EDITORS[item.type] || [item.source?.editor || "4", null];
    openEditor(step, target, item.source?.encounter_id || null);
  }
  function renderTrajectory(root, projection) {
    const card = el("section", "proto1-card proto1-trajectory");
    card.append(el("h3", "", "Πορεία επισκέψεων και ορόσημων"));
    const points = projection.milestones.slice(-12);
    if (!points.length) card.append(el("p", "proto1-muted", "Δεν υπάρχουν τεκμηριωμένα ορόσημα. Το ιστορικό παραμένει άγνωστο."));
    const list = el("ol", "proto1-milestones");
    points.forEach(item => {
      const li = el("li", `proto1-milestone type-${item.type}`);
      const date = item.date || "Ημερομηνία άγνωστη";
      const src = item.source?.encounter_id ? `Επίσκεψη ${item.source.encounter_date || "—"}` : "Πηγή χωρίς encounter ID";
      li.append(el("strong", "", item.label), el("span", "", `${date} · ${item.precision} · ${src}`));
      if (item.uncertainty) li.append(el("small", "proto1-uncertain", item.uncertainty));
      li.append(action("Άνοιγμα πηγής", () => openSource(item)));
      list.append(li);
    });
    card.append(list);
    root.append(card);
  }
  function renderReconciliation(root, projection) {
    if (!projection.reconciliation.length) return;
    const card = el("section", "proto1-card proto1-reconciliation");
    card.append(el("h3", "", "Πιθανή σχέση εγγραφών"));
    projection.reconciliation.slice(0, 3).forEach(candidate => {
      const key = `${candidate.first.source.encounter_id || ""}|${candidate.first.date}|${candidate.second?.date}`;
      const state = dispositions.get(key) || "unresolved";
      const row = el("div", "proto1-proposal");
      line(row, "Αρχική εγγραφή", `${candidate.first.label} · ${candidate.first.date} · επίσκεψη ${candidate.first.source.encounter_date || "—"}`);
      line(row, "Μεταγενέστερη", `${candidate.second?.label || "Πραγματική χορήγηση"} · ${candidate.second?.date || "—"} · επίσκεψη ${candidate.second?.source?.encounter_date || "—"}`);
      line(row, "Γιατί προτείνεται", candidate.reason);
      line(row, "Επίδραση", "Αν επιβεβαιωνόταν, θα μπορούσαν να φαίνονται ως σχετιζόμενα σημεία. Η πορεία δεν τα συνδέει τώρα.");
      line(row, "Κατάσταση", state === "separate" ? "Χωριστές για αυτή την προβολή" : "Μη επιβεβαιωμένη σχέση");
      const controls = el("div", "proto1-actions");
      const confirm = action("Επιβεβαίωση σύνδεσης", () => {});
      confirm.disabled = true;
      confirm.title = "Δεν υπάρχει ακόμη εγκεκριμένη διαδρομή επίμονης σύνδεσης στον ιδιοκτήτη της εγγραφής.";
      controls.append(confirm, action("Διόρθωση πηγής", () => openSource(candidate.first)), action("Διατήρηση χωριστά", () => { dispositions.set(key, "separate"); render(latest); }), action("Ανεπίλυτο", () => { dispositions.delete(key); render(latest); }));
      row.append(controls, el("small", "proto1-muted", "Η επιβεβαίωση απαιτεί εγκεκριμένη διαδρομή στον ιδιοκτήτη των εγγραφών. Καμία εγγραφή δεν συγχωνεύεται εδώ."));
      card.append(row);
    });
    root.append(card);
  }
  function render(detail) {
    latest = detail;
    const core = window.OstProto1VisitCore;
    if (!core || !detail) return;
    const projection = core.build({ patientId: detail.patientId, encounters: detail.historicalEncounters, projection: detail.projection, summary: detail.summary, plan: detail.plan, current: detail.current, historyStatus: detail.historyStatus });
    window.OstProto1VisitUI.lastProjection = projection;
    const enabled = Boolean(detail.patientId && detail.historyStatus === "loaded" && projection.relevant);
    document.body.classList.toggle("proto1-active", enabled);
    const root = $("#proto1VisitWorkspace");
    if (!root) return;
    root.hidden = !enabled;
    if (!enabled) { root.replaceChildren(); return; }
    const title = $(".title-block h1");
    const subtitle = $(".title-block p");
    if (title) title.textContent = "Οστεοπόρωση · σημερινή επίσκεψη";
    if (subtitle) subtitle.textContent = "Τρέχον κλινικό θέμα και τεκμηριωμένη πορεία";
    if (projection.patient_id !== focusPatientId) { chosenFocus = ""; focusPatientId = projection.patient_id; }
    if (projection.suggested_focus !== lastSuggested) {
      if (!chosenFocus || chosenFocus === lastSuggested) chosenFocus = projection.suggested_focus;
      lastSuggested = projection.suggested_focus;
    }
    if (!chosenFocus) chosenFocus = projection.suggested_focus;
    root.replaceChildren();
    const header = el("div", "proto1-header");
    header.append(el("div", "proto1-eyebrow", "Σημερινή επίσκεψη · προστατευμένο ιστορικό"), el("h2", "", `Ασθενής ${projection.patient_id}`));
    const selectWrap = el("label", "proto1-focus-label", "Εστίαση σήμερα");
    const select = el("select", "proto1-focus");
    Object.entries(FOCUS).forEach(([key, label]) => { const option = el("option", "", label); option.value = key; select.append(option); });
    select.value = chosenFocus;
    select.addEventListener("change", () => { chosenFocus = select.value; render(latest); });
    selectWrap.append(select, el("small", "", "Πρόταση οργάνωσης · μπορείς να την αλλάξεις"));
    header.append(selectWrap, action("Μητρώο ασθενών", () => { document.body.classList.add("proto1-editor-open"); $("#clinicalRegistry")?.scrollIntoView({ block: "start" }); }), action("Επεξεργασία πηγών", () => openEditor("4", "#s4Administrations")));
    root.append(header);
    const main = el("div", "proto1-columns");
    const current = el("section", "proto1-card proto1-current");
    current.append(el("h3", "", "Τι έχει σημασία τώρα"));
    line(current, "Καταγεγραμμένη κατάσταση", projection.asserted_treatment ? "Ενεργό denosumab κατά την τελευταία καταγραφή" : "Δεν επιβεβαιώνεται ενεργό denosumab");
    line(current, "Τελευταία πραγματική χορήγηση", projection.last_recorded_actual?.actual_date || "Άγνωστη / μη αξιόπιστη", projection.actual_history_state !== "recorded" ? "proto1-uncertain" : "");
    if (projection.last_recorded_actual) {
      const sourceId = projection.last_recorded_actual.source_encounter_ids?.[0];
      current.append(action("Πηγή πραγματικής χορήγησης", () => openEditor("4", "#s4Administrations", sourceId)));
    }
    if (projection.prior_final_decision) line(current, "Προηγούμενη τελική απόφαση", `${projection.prior_final_decision.type || "—"} · ${projection.prior_final_decision.selected_agent || "—"} · ${projection.prior_final_decision.source_encounter_date || "ημερομηνία άγνωστη"}`);
    else line(current, "Προηγούμενη τελική απόφαση", "Δεν έχει καταγραφεί");
    if (projection.prior_final_decision?.source_encounter_id) current.append(action("Πηγή προηγούμενης απόφασης", () => openEditor("4", "#s4DecisionType", projection.prior_final_decision.source_encounter_id)));
    if (projection.successor_plan) line(current, "Σχεδιασμένη επόμενη θεραπεία", `${projection.successor_plan.agent} · ${projection.successor_plan.planned_date || "χρόνος άγνωστος"} · όχι πραγματική χορήγηση`);
    const todayDecision = projection.current_decision || {};
    line(current, "Σημερινή απόφαση", todayDecision.type ? `${todayDecision.type} · ${todayDecision.selected_agent || "χωρίς καταγεγραμμένο παράγοντα"} · τρέχουσα επεξεργασία` : "Δεν έχει καταγραφεί ακόμη");
    if (projection.uncertainty) line(current, "Αβεβαιότητα", projection.uncertainty, "proto1-uncertain");
    if (projection.derived_due) line(current, "Παράγωγη αναμενόμενη ημερομηνία", `${projection.derived_due.date} · ${projection.derived_due.rule_id} · ${projection.derived_due.source_refs?.join(", ") || ""}`);
    else if (chosenFocus === "due" || chosenFocus === "delay" || chosenFocus === "verification") line(current, "Χρονισμός", "Δεν προκύπτει αξιόπιστο συμπέρασμα από τα διαθέσιμα στοιχεία");
    if (projection.plans.length) line(current, "Προγραμματισμένο / χαμένο", `${projection.plans.length} εγγραφή/ές · δεν τεκμηριώνουν πραγματική χορήγηση`);
    const labs = detail.summary?.labs;
    if (["due", "delay"].includes(chosenFocus)) {
      const fractureStatus = detail.current?.fracture_history?.interval_fracture_status;
      line(current, "Νέο κάταγμα από την τελευταία επίσκεψη", fractureStatus === "yes" ? "Καταγράφηκε σήμερα — άνοιξε το συμβάν" : fractureStatus === "no" ? "Ρητά αρνητικό στη σημερινή εγγραφή" : "Δεν έχει καταγραφεί σήμερα");
      if (labs?.state !== "documented") line(current, "Σχετικά εργαστηριακά", detail.historyStatus === "loaded" ? "Δεν υπάρχει καταγεγραμμένο πρόσφατο αποτέλεσμα στο προστατευμένο ιστορικό" : "Μη διαθέσιμα");
    }
    if (labs?.state === "documented") {
      line(current, "Τελευταία καταγεγραμμένα εργαστηριακά", `${labs.date || "ημερομηνία άγνωστη"} · ${(labs.values || []).slice(0, 3).map(item => `${item.label} ${item.value}`).join(" · ") || "τιμές μη διαθέσιμες"}`);
      const labSource = (detail.historicalLabs || []).find(item => item.lab_date === labs.date)?.source_encounter_id;
      current.append(action("Πηγή εργαστηριακών", () => openEditor("3", "#s3LabsDate", labSource)));
    }
    const relevantGuidance = projection.current_guidance.filter(rule => chosenFocus === "transition" ? /EXIT/.test(rule.rule_id) : chosenFocus === "delay" ? /R24|R12/.test(rule.rule_id) : /R12|R13|R24/.test(rule.rule_id));
    relevantGuidance.slice(0, 3).forEach(rule => {
      const block = el("details", "proto1-guidance");
      block.append(el("summary", "", `Τρέχουσα καθοδήγηση · ${rule.rule_id}`), el("p", "", rule.why_now || ""), el("small", "", `Πηγή: ${rule.source_refs?.join(" · ") || "—"}`));
      current.append(block);
    });
    (detail.plan?.ordered_cards || []).filter(item => (item.reason_codes || []).includes("NEW_EVENT")).slice(0, 2).forEach(item => {
      const block = el("details", "proto1-guidance proto1-event");
      block.append(el("summary", "", "Νέο συμβάν · σημερινό θέμα"), el("p", "", item.why_now || ""));
      block.append(action("Άνοιγμα σχετικής πηγής", () => openEditor(item.card_id === "fracture_history" ? "2" : "4", item.card_id === "fracture_history" ? "#fractureEvents" : "#s4DecisionType")));
      current.append(block);
    });
    current.append(el("p", "proto1-muted", "Η καθοδήγηση δεν καταγράφει σημερινή απόφαση ή χορήγηση."));
    const actions = el("div", "proto1-actions");
    actions.append(action("Χορηγήσεις", () => openEditor("4", "#s4Administrations")), action("Απόφαση / μετάβαση", () => openEditor("4", "#s4DecisionType")), action("Εργαστηριακά", () => openEditor("3", "#s3LabsDate")), action("Κάταγμα / νέο συμβάν", () => openEditor("2", "#fractureEvents")));
    current.append(actions);
    const obligations = el("section", "proto1-card proto1-obligations");
    obligations.append(el("h3", "", "Εκκρεμότητες και κλείσιμο"));
    const open = projection.obligations.filter(item => item.status === "planned");
    if (!open.length) obligations.append(el("p", "proto1-muted", "Καμία καταγεγραμμένη ανοικτή εκκρεμότητα· η απουσία εγγραφής δεν σημαίνει ολοκλήρωση."));
    open.slice(-3).forEach(item => {
      line(obligations, item.label, `${item.date || item.timeframe || "χρόνος άγνωστος"} · πηγή ${item.source.encounter_date || "—"}`);
      obligations.append(action("Άνοιγμα εργασίας", () => openSource(item)));
    });
    obligations.append(action("Σημερινή απόφαση / κλείσιμο", () => openEditor("4", "#s4DecisionType")));
    main.append(current, obligations);
    root.append(main);
    renderTrajectory(root, projection);
    renderReconciliation(root, projection);
  }
  window.addEventListener("ost:guidance-rendered", event => render(event.detail));
  document.addEventListener("input", event => {
    if (!document.body.classList.contains("proto1-active") || !event.target.closest?.("[data-step-panel], .case-meta")) return;
    $("#proto1VisitWorkspace")?.replaceChildren(el("p", "proto1-muted", "Επανυπολογισμός τρέχουσας καθοδήγησης…"));
  }, true);
  window.OstProto1VisitUI = { lastProjection: null, getPrevisitProjection: () => window.OstProto1VisitCore?.previsit(window.OstProto1VisitUI.lastProjection) || null, openEditor };
  const workspace = el("section", "proto1-workspace");
  workspace.id = "proto1VisitWorkspace";
  workspace.hidden = true;
  $(".step-tabs")?.parentNode?.insertBefore(workspace, $(".step-tabs"));
  const back = action("← Επιστροφή στη σημερινή επίσκεψη", () => {
    const returnId = sessionStorage.getItem("ost.proto1.returnEncounter");
    const returnUuid = sessionStorage.getItem("ost.proto1.returnUuid");
    sessionStorage.removeItem("ost.proto1.returnEncounter");
    sessionStorage.removeItem("ost.proto1.returnUuid");
    document.body.classList.remove("proto1-source-mode");
    window.location.hash = "proto1-visit";
    if (returnId && returnId !== window.ClinicalRegistry?.activeEncounterId?.()) { window.ClinicalRegistry?.openEncounter?.(returnId); return; }
    if (returnUuid) { localStorage.setItem("osteoporosis.baselineAuditPilot.activeCase.v1_1", returnUuid); window.location.reload(); return; }
    document.body.classList.remove("proto1-editor-open");
    window.ProgressiveGuidanceUI?.refresh?.();
    workspace.scrollIntoView({ block: "start" });
  }, "proto1-return");
  $(".step-tabs")?.parentNode?.insertBefore(back, $(".step-tabs"));
  if (sessionStorage.getItem("ost.proto1.returnEncounter") || sessionStorage.getItem("ost.proto1.returnUuid")) document.body.classList.add("proto1-source-mode");
  if (/^#proto1-editor-[1-6]$/.test(window.location.hash)) {
    const step = window.location.hash.slice(-1);
    setTimeout(() => openEditor(step, null), 100);
  }
  window.ProgressiveGuidanceUI?.refresh?.();
})();
