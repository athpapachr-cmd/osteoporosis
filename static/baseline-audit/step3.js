(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const num = (value) => value === "" || value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value);

  function injectAssets() {
    if (!document.querySelector('link[data-step3-style]')) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = "./step3.css";
      link.dataset.step3Style = "true";
      document.head.appendChild(link);
    }

    const panel = $('[data-step-panel="3"]');
    if (!panel) return;
    panel.classList.remove("placeholder-panel");
    panel.innerHTML = `
      <div class="context-note step3-context" id="step3ContextNote">
        <strong>Step 3 — Εξετάσεις & Αποτελέσματα:</strong>
        μεταφέρουμε τα χρήσιμα στοιχεία του παλιού Cockpit, αλλά τα οργανώνουμε ώστε να ξεχωρίζει τι εξετάστηκε, τι ήταν applicable και τι πραγματικά χρησιμοποιήθηκε στη σημερινή απόφαση.
      </div>

      <div class="step3-grid">
        <article class="card step3-card span-2">
          <div class="card-heading"><div><h2>DXA — σημερινό κλινικό context</h2><p>T-score + BMD g/cm² + ποιότητα/συγκρισιμότητα. Τα numeric fields είναι προαιρετικά αν το report είναι διαθέσιμο αλλού.</p></div></div>
          <div class="step3-top-grid">
            <label><span>DXA χρησιμοποιήθηκε στη σημερινή διαχείριση;</span><select id="s3DxaUsed"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option></select></label>
            <label><span>Ημερομηνία τελευταίου DXA</span><input id="s3DxaDate" type="date" /></label>
            <label><span>Κέντρο / facility <small>(προαιρετικό)</small></span><input id="s3DxaFacility" type="text" maxlength="100" /></label>
            <label><span>Μηχάνημα / model <small>(προαιρετικό)</small></span><input id="s3DxaMachine" type="text" maxlength="100" /></label>
          </div>

          <div id="s3DxaDetails" class="s3-collapsible" hidden>
            <div class="dxa-table-wrap">
              <table class="dxa-entry-table">
                <thead><tr><th>Site</th><th>BMD g/cm²</th><th>T-score</th></tr></thead>
                <tbody>
                  <tr><td>Οσφ. μοίρα</td><td><input id="s3SpineBmd" type="number" step="0.001" min="0.1" max="3" /></td><td><input id="s3SpineT" type="number" step="0.1" min="-8" max="5" /></td></tr>
                  <tr><td>Total hip</td><td><input id="s3TotalHipBmd" type="number" step="0.001" min="0.1" max="3" /></td><td><input id="s3TotalHipT" type="number" step="0.1" min="-8" max="5" /></td></tr>
                  <tr><td>Femoral neck</td><td><input id="s3FnBmd" type="number" step="0.001" min="0.1" max="3" /></td><td><input id="s3FnT" type="number" step="0.1" min="-8" max="5" /></td></tr>
                </tbody>
              </table>
            </div>

            <div class="step3-top-grid four-cols">
              <label><span>ROI / excluded vertebrae issue</span><select id="s3RoiIssue"><option value="">—</option><option value="none_known">Κανένα γνωστό</option><option value="present">Υπάρχει</option><option value="uncertain">Αβέβαιο</option><option value="not_reviewed">Δεν ελέγχθηκε</option></select></label>
              <label><span>Artifact / technical limitation</span><select id="s3Artifact"><option value="">—</option><option value="none_known">Κανένα γνωστό</option><option value="present">Υπάρχει</option><option value="uncertain">Αβέβαιο</option><option value="not_reviewed">Δεν ελέγχθηκε</option></select></label>
              <label><span>Z-score χρησιμοποιήθηκε όταν relevant;</span><select id="s3ZScoreUsed"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
              <label><span>Έγινε longitudinal comparison;</span><select id="s3Longitudinal"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
            </div>

            <div id="s3LongitudinalDetails" class="longitudinal-box" hidden>
              <div class="step3-top-grid four-cols">
                <label><span>Comparison scan</span><input id="s3ComparisonDate" type="date" /></label>
                <label><span>Ίδιο μηχάνημα / cross-calibrated;</span><select id="s3ComparableMachine"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="unknown">Άγνωστο</option><option value="not_applicable">N/A</option></select></label>
                <label><span>Facility LSC γνωστό;</span><select id="s3LscKnown"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
                <label><span>BMD/LSC ή δηλωμένη μη-συγκρισιμότητα;</span><select id="s3ChangeValid"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
              </div>
              <div class="step3-top-grid compact-three">
                <label><span>Spine LSC % <small>(προαιρετικό)</small></span><input id="s3SpineLsc" type="number" step="0.1" min="0" max="30" /></label>
                <label><span>Total hip LSC % <small>(προαιρετικό)</small></span><input id="s3HipLsc" type="number" step="0.1" min="0" max="30" /></label>
              </div>
            </div>
          </div>
        </article>

        <article class="card step3-card">
          <div class="card-heading"><div><h2>VFA / Vertebral imaging</h2><p>Ξεχωριστά η ένδειξη από το τι τελικά έγινε.</p></div></div>
          <div class="field-stack">
            <label><span>Υπήρχε ένδειξη για VFA / vertebral imaging;</span><select id="s3VfaIndicated"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option></select></label>
          </div>
          <div class="chip-checks compact s3-reasons" id="s3VfaReasons">
            <label><input type="checkbox" value="height_loss_ge_4_cm" />Απώλεια ύψους ≥4 cm</label>
            <label><input type="checkbox" value="kyphosis" />Κύφωση</label>
            <label><input type="checkbox" value="acute_back_pain_with_risk" />Οξύ άλγος ράχης + risk</label>
            <label><input type="checkbox" value="long_term_glucocorticoids" />Μακροχρόνια GC</label>
            <label><input type="checkbox" value="t_score_le_minus_2_5" />T-score ≤ −2.5</label>
            <label><input type="checkbox" value="prior_suspected_vertebral" />Προηγ./ύποπτο σπονδυλικό</label>
            <label><input type="checkbox" value="other" />Άλλο</label>
          </div>
          <div class="field-stack">
            <label><span>Action</span><select id="s3VfaAction"><option value="">—</option><option value="performed">Έγινε σήμερα/πρόσφατα</option><option value="already_available_reviewed">Υπήρχε και ελέγχθηκε</option><option value="arranged">Ζητήθηκε / κανονίστηκε</option><option value="reasoned_not_done">Δεν έγινε — τεκμηριωμένα</option><option value="missed">Δεν έγινε</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Modality <small>(προαιρετικό)</small></span><select id="s3VfaModality"><option value="">—</option><option value="VFA">VFA</option><option value="spine_xray">Ακτινογραφία ΣΣ</option><option value="CT">CT</option><option value="MRI">MRI</option><option value="other">Άλλο</option></select></label>
            <label><span>Βρέθηκε σπονδυλικό κάταγμα;</span><select id="s3VertebralFound"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Genant/grade καταγράφηκε αν διαθέσιμο;</span><select id="s3GenantRecorded"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
          </div>
        </article>

        <article class="card step3-card">
          <div class="card-heading"><div><h2>Secondary causes — process</h2><p>Το audit μετρά αν έγινε/κανονίστηκε ο έλεγχος, όχι αν αντέγραψες κάθε εργαστηριακή τιμή.</p></div></div>
          <div class="field-stack">
            <label><span>Secondary-cause review ενδείκνυτο;</span><select id="s3SecondaryIndicated"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option></select></label>
            <label><span>Ιστορικό secondary causes ελέγχθηκε;</span><select id="s3SecondaryHistory"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Προηγούμενος έλεγχος θεωρήθηκε ακόμη επαρκής;</span><select id="s3PriorWorkupAdequate"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Relevant labs ελέγχθηκαν ή κανονίστηκαν;</span><select id="s3LabsReviewed"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Υπάρχει unresolved secondary-cause question;</span><select id="s3SecondaryUnresolved"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option></select></label>
          </div>
        </article>

        <article class="card step3-card span-2">
          <div class="card-heading"><div><h2>Εργαστηριακά</h2><p>Προαιρετική καταχώρηση τιμών. Μεταφέρουμε τα χρήσιμα fields του παλιού Cockpit χωρίς να κάνουμε τη φόρμα υποχρεωτικό δεύτερο εργαστηριακό αρχείο.</p></div></div>
          <details class="lab-section" open>
            <summary>Core mineral / renal labs</summary>
            <div class="lab-grid">
              <label><span>Ca mg/dL</span><input id="s3Ca" type="number" step="0.01" /></label>
              <label><span>Phosphate mg/dL</span><input id="s3Phos" type="number" step="0.01" /></label>
              <label><span>25-OH Vit D ng/mL</span><input id="s3VitD" type="number" step="0.1" /></label>
              <label><span>PTH pg/mL</span><input id="s3Pth" type="number" step="0.1" /></label>
              <label><span>Creatinine mg/dL</span><input id="s3Creat" type="number" step="0.01" /></label>
              <label><span>eGFR mL/min/1.73m²</span><input id="s3Egfr" type="number" step="1" /></label>
              <label><span>Urea mg/dL</span><input id="s3Urea" type="number" step="0.1" /></label>
              <label><span>Total ALP U/L</span><input id="s3Alp" type="number" step="0.1" /></label>
              <label><span>Mg mg/dL</span><input id="s3Mg" type="number" step="0.01" /></label>
            </div>
          </details>

          <details class="lab-section">
            <summary>Bone turnover markers</summary>
            <div class="lab-grid">
              <label><span>CTX ng/mL</span><input id="s3Ctx" type="number" step="0.001" /></label>
              <label><span>P1NP ng/mL</span><input id="s3P1np" type="number" step="0.1" /></label>
              <label><span>Bone ALP U/L</span><input id="s3BoneAlp" type="number" step="0.1" /></label>
              <label><span>Osteocalcin ng/mL</span><input id="s3Osteocalcin" type="number" step="0.1" /></label>
              <label><span>Context</span><select id="s3BtmContext"><option value="">—</option><option value="baseline">Baseline</option><option value="adherence_response">Adherence / response</option><option value="treatment_failure">Treatment failure</option><option value="transition">Transition</option><option value="other">Άλλο</option><option value="not_applicable">N/A</option></select></label>
            </div>
          </details>

          <details class="lab-section">
            <summary>Conditional / secondary-cause labs</summary>
            <div class="lab-grid">
              <label><span>Glucose mg/dL</span><input id="s3Glucose" type="number" step="0.1" /></label>
              <label><span>HbA1c %</span><input id="s3Hba1c" type="number" step="0.1" /></label>
              <label><span>TSH</span><input id="s3Tsh" type="number" step="0.01" /></label>
              <label><span>FT4</span><input id="s3Ft4" type="number" step="0.01" /></label>
              <label><span>ESR mm/h</span><input id="s3Esr" type="number" step="1" /></label>
              <label><span>CRP mg/L</span><input id="s3Crp" type="number" step="0.1" /></label>
              <label><span>Testosterone ng/dL</span><input id="s3Testosterone" type="number" step="0.1" /></label>
              <label><span>FSH IU/L</span><input id="s3Fsh" type="number" step="0.1" /></label>
              <label><span>Estradiol pg/mL</span><input id="s3Estradiol" type="number" step="0.1" /></label>
              <label><span>Morning cortisol µg/dL</span><input id="s3Cortisol" type="number" step="0.1" /></label>
              <label><span>24h urine Ca mg</span><input id="s3UrineCa" type="number" step="1" /></label>
            </div>
            <div class="chip-checks compact s3-status-checks">
              <label><input id="s3Cbc" type="checkbox" /> CBC reviewed/arranged</label>
              <label><input id="s3Liver" type="checkbox" /> Liver profile</label>
              <label><input id="s3Celiac" type="checkbox" /> Celiac screen</label>
              <label><input id="s3Spep" type="checkbox" /> SPEP / light chains</label>
            </div>
          </details>
        </article>

        <article class="card step3-card">
          <div class="card-heading"><div><h2>Πτώσεις, ευπάθεια & λειτουργικότητα</h2><p>Outpatient osteoporosis context — όχι αντιγραφή όλων των hospital-specific Morse items.</p></div></div>
          <div class="field-stack">
            <label><span>Falls history ελέγχθηκε;</span><select id="s3FallsReviewed"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Πτώσεις τελευταίων 12 μηνών</span><input id="s3FallsCount" type="number" min="0" max="50" /></label>
            <label><span>Fall injury / fracture related;</span><select id="s3FallInjury"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="unknown">Άγνωστο</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Frailty / function review;</span><select id="s3FunctionReviewed"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
          </div>
          <div class="step3-top-grid compact-three">
            <label><span>CFS 1–9</span><input id="s3Cfs" type="number" min="1" max="9" /></label>
            <label><span>Ambulatory aid</span><select id="s3Aid"><option value="">—</option><option value="none">Κανένα</option><option value="cane">Μπαστούνι</option><option value="crutches">Πατερίτσες</option><option value="walker">Περιπατητήρας</option><option value="wheelchair">Αμαξίδιο</option><option value="other">Άλλο</option><option value="unknown">Άγνωστο</option></select></label>
            <label><span>Gait / balance concern</span><select id="s3GaitConcern"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option><option value="not_assessed">Δεν αξιολογήθηκε</option></select></label>
            <label><span>TUG seconds <small>(προαιρετικό)</small></span><input id="s3Tug" type="number" step="0.1" min="0" max="300" /></label>
          </div>
          <div class="chip-checks compact">
            <label><input id="s3Cognitive" type="checkbox" /> Γνωστική διαταραχή</label>
            <label><input id="s3Immobility" type="checkbox" /> Σημαντική ακινησία</label>
          </div>
          <label class="field-row"><span class="field-label">Αν βρέθηκε ουσιαστικός κίνδυνος, υπήρξε action;</span><select id="s3FallsAction"><option value="">—</option><option value="addressed">Αντιμετωπίστηκε στη σημερινή επίσκεψη</option><option value="referral_or_plan">Referral / συγκεκριμένο plan</option><option value="no_action">Όχι action</option><option value="not_applicable">N/A</option></select></label>
        </article>

        <article class="card step3-card">
          <div class="card-heading"><div><h2>Σαρκοπενία</h2><p>Conditional case-finding· δεν είναι υποχρεωτικό full workup σε κάθε ασθενή.</p></div></div>
          <div class="field-stack">
            <label><span>Case-finding applicable σήμερα;</span><select id="s3SarcApplicable"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option></select></label>
            <label><span>Screen method</span><select id="s3SarcMethod"><option value="">—</option><option value="SARC_F">SARC-F</option><option value="clinical_suspicion">Κλινική υποψία</option><option value="not_done">Δεν έγινε</option><option value="not_applicable">N/A</option></select></label>
          </div>
          <div class="step3-top-grid compact-three">
            <label><span>SARC-F score</span><input id="s3SarcF" type="number" min="0" max="10" /></label>
            <label><span>5-chair stand sec</span><input id="s3ChairStand" type="number" step="0.1" min="0" max="300" /></label>
            <label><span>Grip strength kg</span><input id="s3Grip" type="number" step="0.1" min="0" max="100" /></label>
            <label><span>Gait speed m/s</span><input id="s3GaitSpeed" type="number" step="0.01" min="0" max="5" /></label>
            <label><span>SPPB 0–12</span><input id="s3Sppb" type="number" step="1" min="0" max="12" /></label>
            <label><span>TUG sec</span><input id="s3SarcTug" type="number" step="0.1" min="0" max="300" /></label>
          </div>
          <div class="field-stack">
            <label><span>Probable sarcopenia signal</span><select id="s3ProbableSarc"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option><option value="not_assessed">Δεν αξιολογήθηκε</option></select></label>
            <label><span>Action αν θετικό <small>(προαιρετικό)</small></span><input id="s3SarcAction" type="text" maxlength="250" placeholder="π.χ. further assessment / exercise / referral" /></label>
          </div>
          <div class="derived-note">SARC-F ≥4 μπορεί να αποθηκεύεται ως derived signal, αλλά δεν εμφανίζεται ως performance feedback κατά το baseline.</div>
        </article>
      </div>`;
  }

  function defaultState() {
    return {
      version: "step3-v1",
      dxa: {
        used: "", date: "", facility: "", machine: "", spine_bmd: null, spine_t: null,
        total_hip_bmd: null, total_hip_t: null, femoral_neck_bmd: null, femoral_neck_t: null,
        z_score_used: "", roi_issue: "", artifact: "", longitudinal: "", comparison_date: "",
        comparable_machine: "", lsc_known: "", spine_lsc_percent: null, hip_lsc_percent: null, change_valid: ""
      },
      vfa: { indicated: "", reasons: [], action: "", modality: "", vertebral_found: "", genant_recorded: "" },
      secondary: { indicated: "", history_reviewed: "", prior_workup_adequate: "", labs_reviewed: "", unresolved: "" },
      labs: {
        ca: null, phosphate: null, vitamin_d: null, pth: null, creatinine: null, egfr: null, urea: null, total_alp: null, magnesium: null,
        ctx: null, p1np: null, bone_alp: null, osteocalcin: null, btm_context: "", glucose: null, hba1c: null,
        tsh: null, ft4: null, esr: null, crp: null, testosterone: null, fsh: null, estradiol: null, cortisol: null, urine_ca_24h: null,
        cbc_reviewed: false, liver_profile_reviewed: false, celiac_screen_reviewed: false, spep_light_chains_reviewed: false
      },
      function: {
        falls_reviewed: "", falls_count_12m: null, fall_injury_related: "", function_reviewed: "", cfs: null,
        cognitive_impairment: false, significant_immobility: false, ambulatory_aid: "", gait_balance_concern: "", tug_seconds: null, action: ""
      },
      sarcopenia: {
        applicable: "", method: "", sarc_f: null, chair_stand_seconds: null, grip_strength_kg: null,
        gait_speed_m_s: null, sppb: null, tug_seconds: null, probable_signal: "", action: "", derived: {}
      },
      updated_at: null
    };
  }

  function getCases() {
    try {
      const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(data) ? data : [];
    } catch { return []; }
  }

  function setCases(cases) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cases));
  }

  function activeUuid() {
    return localStorage.getItem(ACTIVE_KEY) || "";
  }

  function getActiveCase() {
    const id = activeUuid();
    return getCases().find((item) => item.internal_uuid === id) || null;
  }

  function archetypeLabel(value) {
    const map = {
      initial_assessment_new_or_uncertain_diagnosis: "Αρχική αξιολόγηση: πλήρης diagnostic/risk/function εικόνα όπου applicable.",
      initial_assessment_known_osteoporosis_or_osteopenia: "Αρχική αξιολόγηση γνωστής νόσου: έλεγχος ποιότητας DXA, προηγούμενου workup και τρέχου λειτουργικού κινδύνου.",
      routine_followup_stable: "Stable follow-up: μόνο ό,τι είναι due ή έχει αλλάξει· όχι μηχανική επανάληψη πλήρους workup.",
      treatment_start: "Έναρξη θεραπείας: επάρκεια source data και medication-specific safety labs όπου χρειάζονται.",
      treatment_continuation_or_due_monitoring: "Συνέχιση/monitoring: due labs, DXA και interval function/falls όταν relevant.",
      treatment_change_or_transition: "Αλλαγή/transition: response context, safety labs και BTM όταν χρησιμοποιούνται κλινικά.",
      post_fragility_fracture: "Μετά από κάταγμα: vertebral burden, falls/function και secondary causes όπου indicated.",
      fracture_on_treatment: "Κάταγμα υπό θεραπεία: response/secondary-cause/function reassessment όπου relevant.",
      adverse_effect_or_intolerance: "Adverse effect: targeted investigations ανάλογα με το πρόβλημα.",
      treatment_completion_or_consolidation: "Completion/consolidation: response context και safety/transition monitoring.",
      other: "Άλλο encounter: καταγράφονται μόνο τα κλινικά applicable στοιχεία."
    };
    return map[value] || "Επίλεξε encounter type στο Step 1 ώστε το audit να γνωρίζει ποια domains είναι applicable.";
  }

  let state = defaultState();
  let loadedUuid = "";

  function mergeDefaults(saved) {
    const base = defaultState();
    const s = saved || {};
    return {
      ...base, ...s,
      dxa: { ...base.dxa, ...(s.dxa || {}) },
      vfa: { ...base.vfa, ...(s.vfa || {}), reasons: Array.isArray(s.vfa?.reasons) ? s.vfa.reasons : [] },
      secondary: { ...base.secondary, ...(s.secondary || {}) },
      labs: { ...base.labs, ...(s.labs || {}) },
      function: { ...base.function, ...(s.function || {}) },
      sarcopenia: { ...base.sarcopenia, ...(s.sarcopenia || {}), derived: { ...(s.sarcopenia?.derived || {}) } }
    };
  }

  function seedFromStep1(activeCase) {
    const risk = activeCase?.risk_context || {};
    if (state.function.falls_count_12m === null) state.function.falls_count_12m = num($("#fallsLast12m")?.value ?? risk.falls_last_12_months);
    if (state.function.cfs === null) state.function.cfs = num($("#cfsScore")?.value ?? risk.cfs_score);
    if (!state.function.cognitive_impairment) state.function.cognitive_impairment = Boolean($("#cognitiveImpairment")?.checked || risk.cognitive_impairment);
    if (!state.function.significant_immobility) state.function.significant_immobility = Boolean($("#significantImmobility")?.checked || risk.significant_immobility);

    const sarcRelevant = Boolean($("#sarcopeniaRelevant")?.checked || risk.sarcopenia_case_finding_relevant);
    if (!state.sarcopenia.applicable && sarcRelevant) state.sarcopenia.applicable = "yes";
    if (!state.sarcopenia.method) {
      const method = $("#sarcopeniaMethod")?.value || risk.sarcopenia_screen_method || "";
      if (method === "sarc_f") state.sarcopenia.method = "SARC_F";
      else if (method === "clinical_suspicion") state.sarcopenia.method = "clinical_suspicion";
      else if (method === "not_done") state.sarcopenia.method = "not_done";
    }
    if (state.sarcopenia.sarc_f === null) state.sarcopenia.sarc_f = num($("#sarcFScore")?.value ?? risk.sarc_f_score);
  }

  function loadState() {
    const active = getActiveCase();
    const id = active?.internal_uuid || activeUuid();
    if (!id) {
      state = defaultState();
      loadedUuid = "";
      hydrate();
      return;
    }
    state = mergeDefaults(active?.step3);
    loadedUuid = id;
    seedFromStep1(active);
    hydrate();
  }

  function persist() {
    if (!loadedUuid) loadedUuid = activeUuid();
    if (!loadedUuid) return;
    collect();
    state.updated_at = new Date().toISOString();
    const cases = getCases();
    const index = cases.findIndex((item) => item.internal_uuid === loadedUuid);
    if (index < 0) return;
    cases[index] = { ...cases[index], step3: state };
    setCases(cases);
  }

  function v(id) { return $(id)?.value ?? ""; }
  function checked(id) { return Boolean($(id)?.checked); }

  function collect() {
    state.dxa = {
      ...state.dxa,
      used: v("#s3DxaUsed"), date: v("#s3DxaDate"), facility: v("#s3DxaFacility").trim(), machine: v("#s3DxaMachine").trim(),
      spine_bmd: num(v("#s3SpineBmd")), spine_t: num(v("#s3SpineT")), total_hip_bmd: num(v("#s3TotalHipBmd")), total_hip_t: num(v("#s3TotalHipT")),
      femoral_neck_bmd: num(v("#s3FnBmd")), femoral_neck_t: num(v("#s3FnT")), z_score_used: v("#s3ZScoreUsed"), roi_issue: v("#s3RoiIssue"), artifact: v("#s3Artifact"),
      longitudinal: v("#s3Longitudinal"), comparison_date: v("#s3ComparisonDate"), comparable_machine: v("#s3ComparableMachine"), lsc_known: v("#s3LscKnown"),
      spine_lsc_percent: num(v("#s3SpineLsc")), hip_lsc_percent: num(v("#s3HipLsc")), change_valid: v("#s3ChangeValid")
    };
    state.vfa = {
      indicated: v("#s3VfaIndicated"),
      reasons: $$('#s3VfaReasons input[type="checkbox"]:checked').map((x) => x.value),
      action: v("#s3VfaAction"), modality: v("#s3VfaModality"), vertebral_found: v("#s3VertebralFound"), genant_recorded: v("#s3GenantRecorded")
    };
    state.secondary = {
      indicated: v("#s3SecondaryIndicated"), history_reviewed: v("#s3SecondaryHistory"), prior_workup_adequate: v("#s3PriorWorkupAdequate"),
      labs_reviewed: v("#s3LabsReviewed"), unresolved: v("#s3SecondaryUnresolved")
    };
    state.labs = {
      ca: num(v("#s3Ca")), phosphate: num(v("#s3Phos")), vitamin_d: num(v("#s3VitD")), pth: num(v("#s3Pth")), creatinine: num(v("#s3Creat")), egfr: num(v("#s3Egfr")),
      urea: num(v("#s3Urea")), total_alp: num(v("#s3Alp")), magnesium: num(v("#s3Mg")), ctx: num(v("#s3Ctx")), p1np: num(v("#s3P1np")), bone_alp: num(v("#s3BoneAlp")), osteocalcin: num(v("#s3Osteocalcin")),
      btm_context: v("#s3BtmContext"), glucose: num(v("#s3Glucose")), hba1c: num(v("#s3Hba1c")), tsh: num(v("#s3Tsh")), ft4: num(v("#s3Ft4")), esr: num(v("#s3Esr")), crp: num(v("#s3Crp")),
      testosterone: num(v("#s3Testosterone")), fsh: num(v("#s3Fsh")), estradiol: num(v("#s3Estradiol")), cortisol: num(v("#s3Cortisol")), urine_ca_24h: num(v("#s3UrineCa")),
      cbc_reviewed: checked("#s3Cbc"), liver_profile_reviewed: checked("#s3Liver"), celiac_screen_reviewed: checked("#s3Celiac"), spep_light_chains_reviewed: checked("#s3Spep")
    };
    state.function = {
      falls_reviewed: v("#s3FallsReviewed"), falls_count_12m: num(v("#s3FallsCount")), fall_injury_related: v("#s3FallInjury"), function_reviewed: v("#s3FunctionReviewed"), cfs: num(v("#s3Cfs")),
      cognitive_impairment: checked("#s3Cognitive"), significant_immobility: checked("#s3Immobility"), ambulatory_aid: v("#s3Aid"), gait_balance_concern: v("#s3GaitConcern"), tug_seconds: num(v("#s3Tug")), action: v("#s3FallsAction")
    };
    const sarcF = num(v("#s3SarcF"));
    state.sarcopenia = {
      applicable: v("#s3SarcApplicable"), method: v("#s3SarcMethod"), sarc_f: sarcF, chair_stand_seconds: num(v("#s3ChairStand")), grip_strength_kg: num(v("#s3Grip")),
      gait_speed_m_s: num(v("#s3GaitSpeed")), sppb: num(v("#s3Sppb")), tug_seconds: num(v("#s3SarcTug")), probable_signal: v("#s3ProbableSarc"), action: v("#s3SarcAction").trim(),
      derived: { sarc_f_positive_ge_4: sarcF === null ? null : sarcF >= 4 }
    };
  }

  function setValue(id, value) { const node = $(id); if (node) node.value = value ?? ""; }
  function setCheck(id, value) { const node = $(id); if (node) node.checked = Boolean(value); }

  function hydrate() {
    const active = getActiveCase();
    const context = $("#step3ContextNote");
    if (context) context.innerHTML = `<strong>Step 3 — Εξετάσεις & Αποτελέσματα:</strong> ${archetypeLabel(active?.encounter_archetype || "")}`;

    const d = state.dxa;
    setValue("#s3DxaUsed", d.used); setValue("#s3DxaDate", d.date); setValue("#s3DxaFacility", d.facility); setValue("#s3DxaMachine", d.machine);
    setValue("#s3SpineBmd", d.spine_bmd); setValue("#s3SpineT", d.spine_t); setValue("#s3TotalHipBmd", d.total_hip_bmd); setValue("#s3TotalHipT", d.total_hip_t); setValue("#s3FnBmd", d.femoral_neck_bmd); setValue("#s3FnT", d.femoral_neck_t);
    setValue("#s3ZScoreUsed", d.z_score_used); setValue("#s3RoiIssue", d.roi_issue); setValue("#s3Artifact", d.artifact); setValue("#s3Longitudinal", d.longitudinal); setValue("#s3ComparisonDate", d.comparison_date); setValue("#s3ComparableMachine", d.comparable_machine); setValue("#s3LscKnown", d.lsc_known); setValue("#s3SpineLsc", d.spine_lsc_percent); setValue("#s3HipLsc", d.hip_lsc_percent); setValue("#s3ChangeValid", d.change_valid);

    setValue("#s3VfaIndicated", state.vfa.indicated); setValue("#s3VfaAction", state.vfa.action); setValue("#s3VfaModality", state.vfa.modality); setValue("#s3VertebralFound", state.vfa.vertebral_found); setValue("#s3GenantRecorded", state.vfa.genant_recorded);
    $$('#s3VfaReasons input[type="checkbox"]').forEach((x) => x.checked = state.vfa.reasons.includes(x.value));

    setValue("#s3SecondaryIndicated", state.secondary.indicated); setValue("#s3SecondaryHistory", state.secondary.history_reviewed); setValue("#s3PriorWorkupAdequate", state.secondary.prior_workup_adequate); setValue("#s3LabsReviewed", state.secondary.labs_reviewed); setValue("#s3SecondaryUnresolved", state.secondary.unresolved);

    const l = state.labs;
    [["#s3Ca",l.ca],["#s3Phos",l.phosphate],["#s3VitD",l.vitamin_d],["#s3Pth",l.pth],["#s3Creat",l.creatinine],["#s3Egfr",l.egfr],["#s3Urea",l.urea],["#s3Alp",l.total_alp],["#s3Mg",l.magnesium],["#s3Ctx",l.ctx],["#s3P1np",l.p1np],["#s3BoneAlp",l.bone_alp],["#s3Osteocalcin",l.osteocalcin],["#s3Glucose",l.glucose],["#s3Hba1c",l.hba1c],["#s3Tsh",l.tsh],["#s3Ft4",l.ft4],["#s3Esr",l.esr],["#s3Crp",l.crp],["#s3Testosterone",l.testosterone],["#s3Fsh",l.fsh],["#s3Estradiol",l.estradiol],["#s3Cortisol",l.cortisol],["#s3UrineCa",l.urine_ca_24h]].forEach(([id,val]) => setValue(id,val));
    setValue("#s3BtmContext", l.btm_context); setCheck("#s3Cbc", l.cbc_reviewed); setCheck("#s3Liver", l.liver_profile_reviewed); setCheck("#s3Celiac", l.celiac_screen_reviewed); setCheck("#s3Spep", l.spep_light_chains_reviewed);

    const f = state.function;
    setValue("#s3FallsReviewed", f.falls_reviewed); setValue("#s3FallsCount", f.falls_count_12m); setValue("#s3FallInjury", f.fall_injury_related); setValue("#s3FunctionReviewed", f.function_reviewed); setValue("#s3Cfs", f.cfs); setCheck("#s3Cognitive", f.cognitive_impairment); setCheck("#s3Immobility", f.significant_immobility); setValue("#s3Aid", f.ambulatory_aid); setValue("#s3GaitConcern", f.gait_balance_concern); setValue("#s3Tug", f.tug_seconds); setValue("#s3FallsAction", f.action);

    const s = state.sarcopenia;
    setValue("#s3SarcApplicable", s.applicable); setValue("#s3SarcMethod", s.method); setValue("#s3SarcF", s.sarc_f); setValue("#s3ChairStand", s.chair_stand_seconds); setValue("#s3Grip", s.grip_strength_kg); setValue("#s3GaitSpeed", s.gait_speed_m_s); setValue("#s3Sppb", s.sppb); setValue("#s3SarcTug", s.tug_seconds); setValue("#s3ProbableSarc", s.probable_signal); setValue("#s3SarcAction", s.action);

    syncVisibility();
  }

  function syncVisibility() {
    const dxaDetails = $("#s3DxaDetails");
    if (dxaDetails) dxaDetails.hidden = v("#s3DxaUsed") !== "yes";
    const longitudinal = $("#s3LongitudinalDetails");
    if (longitudinal) longitudinal.hidden = v("#s3Longitudinal") !== "yes";
  }

  function bind() {
    const panel = $('[data-step-panel="3"]');
    if (!panel) return;

    panel.addEventListener("input", () => { syncVisibility(); persist(); });
    panel.addEventListener("change", () => { syncVisibility(); persist(); });

    $$(".step-tab").forEach((button) => button.addEventListener("click", () => {
      if (button.dataset.step === "3") setTimeout(loadState, 0);
    }));

    document.addEventListener("click", (event) => {
      if (event.target.closest("[data-load-case]") || event.target.closest('[data-nav-action="new-case"]')) {
        setTimeout(loadState, 0);
      }
    });

    ["#saveTopBtn", "#saveDraftBtn", "#finishVisitBtn"].forEach((selector) => {
      const node = $(selector);
      if (node) node.addEventListener("click", () => setTimeout(persist, 0));
    });
  }

  injectAssets();
  bind();
  loadState();
})();
