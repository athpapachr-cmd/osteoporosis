(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const num = (v) => v === "" || v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v);

  const YNU = [
    ["", "—"], ["yes", "Ναι"], ["no", "Όχι"], ["not_applicable", "N/A"], ["uncertain", "Αβέβαιο"]
  ];
  const UNDERSTANDING = [
    ["", "—"], ["yes", "Ναι"], ["partly", "Μερικώς"], ["no", "Όχι"], ["uncertain", "Αβέβαιο"], ["not_applicable", "N/A"]
  ];
  const YESNO = [["", "—"], ["yes", "Ναι"], ["no", "Όχι"]];
  const SIGNAL_DOMAINS = [
    ["knowledge", "Γνώση"], ["reasoning", "Κλινικός συλλογισμός"], ["execution", "Εκτέλεση/workflow"],
    ["communication_system", "Επικοινωνία / σύστημα"], ["documentation", "Τεκμηρίωση"], ["safety", "Ασφάλεια"], ["other", "Άλλο"]
  ];
  const INFO_TYPES = [
    ["condition", "Πάθηση / κίνδυνος"], ["exercise", "Άσκηση"], ["nutrition", "Διατροφή"], ["supplements", "Συμπληρώματα"],
    ["medication", "Φάρμακα"], ["administration_timing", "Χρονοδιάγραμμα χορήγησης"], ["other", "Άλλο"]
  ];

  const optionHtml = (items) => items.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  const fieldRow = (id, label) => `<div class="s5-domain-row"><span>${label}</span><select id="${id}">${optionHtml(YNU)}</select></div>`;

  function getCases() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch { return []; } }
  function setCases(cases) { localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
  function activeUuid() { return localStorage.getItem(ACTIVE_KEY) || ""; }
  function getActiveCase() { const id = activeUuid(); return getCases().find((x) => x.internal_uuid === id) || null; }

  function defaultState() {
    return {
      communication: {
        condition_risk_explained: "", results_status_explained: "", exercise_discussed: "", nutrition_discussed: "",
        calcium_discussed: "", vitamin_d_discussed: "", other_supplements_discussed: "", medication_plan_explained: "",
        treatment_reason_explained: "", alternatives_tradeoffs_discussed: "", duration_timing_review_explained: "",
        safety_points_explained: "", missed_dose_timing_message: "", sequencing_transition_message: "",
        questions_addressed: "", preferences_elicited: "", preferences_influenced_plan: ""
      },
      understanding: {
        condition: "", plan: "", rationale: "", teach_back: "", misunderstanding_detected: "", misunderstanding_corrected: "",
        information_given: "", information_types: []
      },
      reflection: {
        what_went_well: "", missed_uncertain: "", missed_domain: "", short_reflection: "", case_review_signal: "no",
        learning_signal: "no", communication_signal: "no", safety_signal: "no", confidence_percent: null
      },
      updated_at: null
    };
  }

  function normalize(raw) {
    const b = defaultState();
    if (!raw || typeof raw !== "object") return b;
    return {
      ...b, ...raw,
      communication: { ...b.communication, ...(raw.communication || {}) },
      understanding: { ...b.understanding, ...(raw.understanding || {}), information_types: Array.isArray(raw?.understanding?.information_types) ? raw.understanding.information_types : [] },
      reflection: { ...b.reflection, ...(raw.reflection || {}) }
    };
  }

  let state = defaultState();
  let loadedUuid = "";

  function archetypeLabel(value) {
    const map = {
      initial_assessment_new_or_uncertain_diagnosis: "Αρχική αξιολόγηση — νέα/αβέβαιη διάγνωση",
      initial_assessment_known_osteoporosis_or_osteopenia: "Αρχική αξιολόγηση — γνωστή οστεοπόρωση/οστεοπενία",
      routine_followup_stable: "Routine follow-up — σταθερή",
      treatment_start: "Έναρξη θεραπείας",
      treatment_continuation_or_due_monitoring: "Συνέχιση / due monitoring",
      treatment_change_or_transition: "Αλλαγή / transition θεραπείας",
      post_fragility_fracture: "Μετά από κάταγμα ευθραυστότητας",
      fracture_on_treatment: "Κάταγμα υπό θεραπεία",
      adverse_effect_or_intolerance: "Ανεπιθύμητη ενέργεια / δυσανεξία",
      treatment_completion_or_consolidation: "Ολοκλήρωση / consolidation",
      other: "Άλλο"
    };
    return map[value] || "Δεν έχει οριστεί archetype";
  }

  function injectAssets() {
    if (!document.querySelector('link[data-step5-style]')) {
      const link = document.createElement("link"); link.rel = "stylesheet"; link.href = "./step5.css"; link.dataset.step5Style = "true"; document.head.appendChild(link);
    }

    const panel = $('[data-step-panel="5"]');
    if (!panel) return;
    panel.classList.remove("placeholder-panel");
    panel.innerHTML = `
      <div class="context-note" id="step5ContextNote"><strong>Step 5 — Επικοινωνία:</strong> post-visit capture χωρίς live performance feedback.</div>
      <div class="step5-grid">
        <article class="card step5-card span-2">
          <div class="card-heading"><div><h2>Τι συζητήθηκε σήμερα</h2><p>Καταγράφεται μόνο ό,τι ήταν σχετικό με τη συγκεκριμένη επίσκεψη. Το N/A παραμένει διαφορετικό από το «δεν έγινε».</p></div></div>
          <div class="s5-context-chip" id="s5ArchetypeChip"></div>

          <div class="s5-section-title">Πάθηση / risk / lifestyle</div>
          <div class="s5-domain-grid">
            ${fieldRow("s5ConditionRisk", "Εξηγήθηκε η πάθηση / ο κίνδυνος κατάγματος;")}
            ${fieldRow("s5ResultsStatus", "Εξηγήθηκαν τα αποτελέσματα / το σημερινό status;")}
            ${fieldRow("s5Exercise", "Συζητήθηκε άσκηση / φυσική δραστηριότητα;")}
            ${fieldRow("s5Nutrition", "Συζητήθηκε διατροφή;")}
            ${fieldRow("s5Calcium", "Συζητήθηκε ασβέστιο;")}
            ${fieldRow("s5VitD", "Συζητήθηκε βιταμίνη D;")}
            ${fieldRow("s5Supplements", "Συζητήθηκαν άλλα συμπληρώματα;")}
          </div>

          <div class="s5-section-title">Φαρμακευτικό πλάνο / shared decision</div>
          <div class="s5-domain-grid">
            ${fieldRow("s5MedicationPlan", "Εξηγήθηκε το φαρμακευτικό ή no-drug πλάνο;")}
            ${fieldRow("s5TreatmentReason", "Εξηγήθηκε γιατί προτείνεται η συγκεκριμένη αντιμετώπιση;")}
            ${fieldRow("s5Alternatives", "Συζητήθηκαν εναλλακτικές / trade-offs όταν relevant;")}
            ${fieldRow("s5DurationTiming", "Εξηγήθηκε διάρκεια / timing / review point;")}
            ${fieldRow("s5SafetyPoints", "Εξηγήθηκαν τα ουσιώδη safety points;")}
            ${fieldRow("s5MissedDose", "Συζητήθηκε missed-dose / timing message όταν relevant;")}
            ${fieldRow("s5Sequencing", "Εξηγήθηκε sequencing / transition όταν relevant;")}
            ${fieldRow("s5Questions", "Απαντήθηκαν οι ερωτήσεις / ανησυχίες;")}
            ${fieldRow("s5Preferences", "Διερευνήθηκαν οι προτιμήσεις της ασθενούς;")}
            ${fieldRow("s5PreferencesPlan", "Οι προτιμήσεις επηρέασαν το τελικό πλάνο όπου relevant;")}
          </div>
          <div class="s5-baseline-note">Δεν εμφανίζεται score ή μήνυμα «έλειψε κάτι». Η αξιολόγηση θα γίνει μετά το baseline lock.</div>
        </article>

        <article class="card step5-card span-2">
          <div class="card-heading"><div><h2>Κατανόηση & teach-back</h2><p>Η κλινική σου εντύπωση καταγράφεται χωριστά από το μελλοντικό Patient Voice.</p></div></div>
          <div class="s5-understanding-grid">
            <label><span>Κατανόησε την πάθηση / risk;</span><select id="s5UnderstandCondition">${optionHtml(UNDERSTANDING)}</select></label>
            <label><span>Κατανόησε το πλάνο;</span><select id="s5UnderstandPlan">${optionHtml(UNDERSTANDING)}</select></label>
            <label><span>Κατανόησε το rationale;</span><select id="s5UnderstandRationale">${optionHtml(UNDERSTANDING)}</select></label>
            <label><span>Χρησιμοποιήθηκε teach-back;</span><select id="s5TeachBack"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Εντοπίστηκε παρανόηση;</span><select id="s5Misunderstanding">${optionHtml([...YESNO,["uncertain","Αβέβαιο"]])}</select></label>
            <label><span>Διορθώθηκε η παρανόηση;</span><select id="s5Corrected">${optionHtml(UNDERSTANDING)}</select></label>
          </div>
          <div class="s5-grid three" style="margin-top:12px">
            <label><span>Δόθηκε γραπτή/ψηφιακή πληροφορία;</span><select id="s5InformationGiven"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
          </div>
          <div class="chip-checks compact s5-info-types" id="s5InfoTypes">${INFO_TYPES.map(([v,l])=>`<label><input type="checkbox" value="${v}" />${l}</label>`).join("")}</div>
          <div class="s5-note">Το Patient Voice θα παραμείνει ξεχωριστό instrument, ώστε η δική σου εκτίμηση ότι «κατάλαβε» να μην υποκαθιστά το τι αναφέρει η ίδια η ασθενής.</div>
        </article>

        <article class="card step5-card span-2">
          <div class="card-heading"><div><h2>Immediate post-visit reflection</h2><p>Σύντομο. Στόχος είναι να εντοπίζονται πιθανά Signals, όχι να γράφεται δεύτερη κλινική σημείωση.</p></div></div>
          <div class="s5-reflection">
            <div>
              <label><span>Τι πήγε καλά; <small>(προαιρετικό)</small></span><textarea id="s5WentWell" rows="3" maxlength="500"></textarea></label>
              <label><span>Σύντομο reflection <small>(προαιρετικό)</small></span><textarea id="s5Reflection" rows="3" maxlength="700" placeholder="Κάτι που θα ήθελες να ξαναδείς ή να βελτιώσεις…"></textarea></label>
            </div>
            <div class="s5-signal-box">
              <label><span>Έμεινε κάτι που ίσως χάθηκε / είναι αβέβαιο;</span><select id="s5MissedUncertain"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option></select></label>
              <label><span>Κύριο domain αν υπάρχει</span><select id="s5MissedDomain"><option value="">—</option>${optionHtml(SIGNAL_DOMAINS)}</select></label>
              <div class="s5-section-title">Potential signals</div>
              <div class="chip-checks compact">
                <label><input id="s5CaseReview" type="checkbox" />Case review</label>
                <label><input id="s5LearningSignal" type="checkbox" />Learning</label>
                <label><input id="s5CommunicationSignal" type="checkbox" />Communication/system</label>
                <label><input id="s5SafetySignal" type="checkbox" />Safety</label>
              </div>
              <label style="margin-top:10px"><span>Confidence για τη συνολική επίσκεψη % <small>(optional)</small></span><input id="s5Confidence" type="number" min="0" max="100" step="5" /></label>
            </div>
          </div>
        </article>
      </div>`;
  }

  function v(id) { return $(id)?.value ?? ""; }
  function checked(id) { return Boolean($(id)?.checked); }
  function setValue(id, value) { const n = $(id); if (n) n.value = value ?? ""; }
  function setCheck(id, value) { const n = $(id); if (n) n.checked = Boolean(value); }

  function collect() {
    state.communication = {
      condition_risk_explained: v("#s5ConditionRisk"), results_status_explained: v("#s5ResultsStatus"), exercise_discussed: v("#s5Exercise"),
      nutrition_discussed: v("#s5Nutrition"), calcium_discussed: v("#s5Calcium"), vitamin_d_discussed: v("#s5VitD"), other_supplements_discussed: v("#s5Supplements"),
      medication_plan_explained: v("#s5MedicationPlan"), treatment_reason_explained: v("#s5TreatmentReason"), alternatives_tradeoffs_discussed: v("#s5Alternatives"),
      duration_timing_review_explained: v("#s5DurationTiming"), safety_points_explained: v("#s5SafetyPoints"), missed_dose_timing_message: v("#s5MissedDose"),
      sequencing_transition_message: v("#s5Sequencing"), questions_addressed: v("#s5Questions"), preferences_elicited: v("#s5Preferences"), preferences_influenced_plan: v("#s5PreferencesPlan")
    };
    state.understanding = {
      condition: v("#s5UnderstandCondition"), plan: v("#s5UnderstandPlan"), rationale: v("#s5UnderstandRationale"), teach_back: v("#s5TeachBack"),
      misunderstanding_detected: v("#s5Misunderstanding"), misunderstanding_corrected: v("#s5Corrected"), information_given: v("#s5InformationGiven"),
      information_types: $$('#s5InfoTypes input[type="checkbox"]:checked').map((x)=>x.value)
    };
    state.reflection = {
      what_went_well: v("#s5WentWell").trim(), missed_uncertain: v("#s5MissedUncertain"), missed_domain: v("#s5MissedDomain"), short_reflection: v("#s5Reflection").trim(),
      case_review_signal: checked("#s5CaseReview") ? "yes" : "no", learning_signal: checked("#s5LearningSignal") ? "yes" : "no",
      communication_signal: checked("#s5CommunicationSignal") ? "yes" : "no", safety_signal: checked("#s5SafetySignal") ? "yes" : "no", confidence_percent: num(v("#s5Confidence"))
    };
  }

  function persist() {
    if (!loadedUuid) loadedUuid = activeUuid();
    if (!loadedUuid) return;
    collect(); state.updated_at = new Date().toISOString();
    const cases = getCases(); const i = cases.findIndex((x) => x.internal_uuid === loadedUuid); if (i < 0) return;
    cases[i] = { ...cases[i], step5: state }; setCases(cases);
  }

  function hydrate() {
    const c = state.communication;
    const map = {
      "#s5ConditionRisk":c.condition_risk_explained,"#s5ResultsStatus":c.results_status_explained,"#s5Exercise":c.exercise_discussed,"#s5Nutrition":c.nutrition_discussed,
      "#s5Calcium":c.calcium_discussed,"#s5VitD":c.vitamin_d_discussed,"#s5Supplements":c.other_supplements_discussed,"#s5MedicationPlan":c.medication_plan_explained,
      "#s5TreatmentReason":c.treatment_reason_explained,"#s5Alternatives":c.alternatives_tradeoffs_discussed,"#s5DurationTiming":c.duration_timing_review_explained,
      "#s5SafetyPoints":c.safety_points_explained,"#s5MissedDose":c.missed_dose_timing_message,"#s5Sequencing":c.sequencing_transition_message,"#s5Questions":c.questions_addressed,
      "#s5Preferences":c.preferences_elicited,"#s5PreferencesPlan":c.preferences_influenced_plan
    };
    Object.entries(map).forEach(([id,val])=>setValue(id,val));
    const u = state.understanding;
    setValue("#s5UnderstandCondition",u.condition); setValue("#s5UnderstandPlan",u.plan); setValue("#s5UnderstandRationale",u.rationale); setValue("#s5TeachBack",u.teach_back);
    setValue("#s5Misunderstanding",u.misunderstanding_detected); setValue("#s5Corrected",u.misunderstanding_corrected); setValue("#s5InformationGiven",u.information_given);
    $$('#s5InfoTypes input[type="checkbox"]').forEach((x)=>x.checked=u.information_types.includes(x.value));
    const r = state.reflection;
    setValue("#s5WentWell",r.what_went_well); setValue("#s5MissedUncertain",r.missed_uncertain); setValue("#s5MissedDomain",r.missed_domain); setValue("#s5Reflection",r.short_reflection); setValue("#s5Confidence",r.confidence_percent);
    setCheck("#s5CaseReview",r.case_review_signal === "yes"); setCheck("#s5LearningSignal",r.learning_signal === "yes"); setCheck("#s5CommunicationSignal",r.communication_signal === "yes"); setCheck("#s5SafetySignal",r.safety_signal === "yes");
    const active = getActiveCase();
    const label = archetypeLabel(active?.encounter_archetype || "");
    if ($("#s5ArchetypeChip")) $("#s5ArchetypeChip").textContent = label;
    if ($("#step5ContextNote")) $("#step5ContextNote").innerHTML = `<strong>Step 5 — Επικοινωνία:</strong> ${label}. Post-visit capture χωρίς live performance feedback.`;
    syncVisibility();
  }

  function syncVisibility() {
    const info = $("#s5InfoTypes"); if (info) info.hidden = v("#s5InformationGiven") !== "yes";
    const corrected = $("#s5Corrected")?.closest("label"); if (corrected) corrected.hidden = v("#s5Misunderstanding") !== "yes";
    const domain = $("#s5MissedDomain")?.closest("label"); if (domain) domain.hidden = v("#s5MissedUncertain") !== "yes" && v("#s5MissedUncertain") !== "uncertain";
  }

  function loadState() {
    const active = getActiveCase(); const id = active?.internal_uuid || activeUuid();
    if (!id) { state = defaultState(); loadedUuid = ""; hydrate(); return; }
    state = normalize(active?.step5); loadedUuid = id; hydrate();
  }

  function bind() {
    const panel = $('[data-step-panel="5"]'); if (!panel) return;
    panel.addEventListener("input", () => { syncVisibility(); persist(); });
    panel.addEventListener("change", () => { syncVisibility(); persist(); });
    $$(".step-tab").forEach((button) => button.addEventListener("click", () => { if (button.dataset.step === "5") setTimeout(loadState, 0); }));
    document.addEventListener("click", (event) => {
      if (event.target.closest("[data-load-case]") || event.target.closest('[data-nav-action="new-case"]')) setTimeout(loadState, 0);
    });
    ["#saveTopBtn", "#saveDraftBtn", "#finishVisitBtn"].forEach((selector) => { const node = $(selector); if (node) node.addEventListener("click", () => setTimeout(persist, 0)); });
  }

  injectAssets(); bind(); loadState();
})();
