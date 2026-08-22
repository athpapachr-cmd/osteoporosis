(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const num = (v) => v === "" || v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v);

  const SOURCES = [
    ["clinician_postvisit_recall", "Clinician post-visit recall"],
    ["formal_GeSY_note", "Formal GeSY note"],
    ["clinician_reviewed_Heidi_output", "Clinician-reviewed Heidi output"],
    ["DXA_report", "DXA report"],
    ["laboratory_results", "Laboratory results"],
    ["imaging_report", "Imaging report"],
    ["medication_or_administration_record", "Medication / administration record"],
    ["patient_provided_information", "Patient-provided information"],
    ["referral_or_external_note", "Referral / external note"],
    ["other", "Other"]
  ];

  const DOMAINS = [
    ["fracture_history", "Ιστορικό καταγμάτων"],
    ["formal_risk_assessment", "FRAX / formal risk"],
    ["DXA_VFA_imaging", "DXA / VFA / imaging"],
    ["secondary_causes_and_labs", "Secondary causes / labs"],
    ["falls_frailty_function_sarcopenia", "Falls / frailty / function / sarcopenia"],
    ["treatment_history_adherence_tolerance", "Treatment history / adherence / tolerance"],
    ["treatment_decision_and_rationale", "Decision / rationale"],
    ["sequencing_and_safety", "Sequencing / safety"],
    ["monitoring_and_followup_plan", "Monitoring / follow-up"],
    ["communication_and_patient_preference", "Communication / patient preference"]
  ];

  const TRACE = [["", "—"], ["complete", "Πλήρες"], ["partial", "Μερικό"], ["absent", "Απόν"], ["not_applicable", "N/A"], ["unknown", "Άγνωστο"]];
  const HEIDI_TRACE = [["", "—"], ["complete", "Πλήρες"], ["partial", "Μερικό"], ["absent", "Απόν"], ["not_used", "Δεν χρησιμοποιήθηκε"], ["not_applicable", "N/A"], ["unknown", "Άγνωστο"]];
  const DISCREPANCY = [["", "—"], ["yes", "Ναι"], ["no", "Όχι"], ["uncertain", "Αβέβαιο"], ["not_applicable", "N/A"]];
  const YESNO = [["", "—"], ["yes", "Ναι"], ["no", "Όχι"], ["uncertain", "Αβέβαιο"]];
  const YNNA = [["", "—"], ["yes", "Ναι"], ["no", "Όχι"], ["not_applicable", "N/A"], ["unknown", "Άγνωστο"]];
  const optionHtml = (items) => items.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");

  function getCases() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch { return []; } }
  function setCases(cases) { localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
  function activeUuid() { return localStorage.getItem(ACTIVE_KEY) || ""; }
  function activeCase() { const id = activeUuid(); return getCases().find((x) => x.internal_uuid === id) || null; }

  function domainDefaults() {
    const x = {};
    DOMAINS.forEach(([key]) => x[key] = { formal_record_trace: "", heidi_trace: "", material_discrepancy: "" });
    return x;
  }

  function defaultState() {
    return {
      sources: { used: [], primary: "", conflict_present: "", conflict_resolution: "", conflict_note: "" },
      trace: domainDefaults(),
      formal_record: { gesy_note_available: "", gesy_note_status: "", note_finalized: "", important_content_missing: "", missing_domains: [], comment: "" },
      heidi_final: { used: "", output_available: "", reviewed: "", correction_required: "", correction_categories: [], final_approved_note_exists: "", used_in_formal_record: "", material_info_not_in_formal: "" },
      capture_quality: { reliability: "", major_gap: "", gap_domains: [], limitation_reasons: [], completion_time_minutes: null, ready_for_audit: "", comment: "" },
      updated_at: null
    };
  }

  function normalize(raw) {
    const b = defaultState();
    if (!raw || typeof raw !== "object") return b;
    const trace = domainDefaults();
    DOMAINS.forEach(([k]) => trace[k] = { ...trace[k], ...(raw.trace?.[k] || {}) });
    return {
      ...b, ...raw,
      sources: { ...b.sources, ...(raw.sources || {}), used: Array.isArray(raw?.sources?.used) ? raw.sources.used : [] },
      trace,
      formal_record: { ...b.formal_record, ...(raw.formal_record || {}), missing_domains: Array.isArray(raw?.formal_record?.missing_domains) ? raw.formal_record.missing_domains : [] },
      heidi_final: { ...b.heidi_final, ...(raw.heidi_final || {}), correction_categories: Array.isArray(raw?.heidi_final?.correction_categories) ? raw.heidi_final.correction_categories : [] },
      capture_quality: { ...b.capture_quality, ...(raw.capture_quality || {}), gap_domains: Array.isArray(raw?.capture_quality?.gap_domains) ? raw.capture_quality.gap_domains : [], limitation_reasons: Array.isArray(raw?.capture_quality?.limitation_reasons) ? raw.capture_quality.limitation_reasons : [] }
    };
  }

  let state = defaultState();
  let loaded = "";

  function injectAssets() {
    if (!document.querySelector('link[data-step6-style]')) {
      const link = document.createElement("link"); link.rel = "stylesheet"; link.href = "./step6.css"; link.dataset.step6Style = "true"; document.head.appendChild(link);
    }
    const panel = $('[data-step-panel="6"]');
    if (!panel) return;
    panel.classList.remove("placeholder-panel");
    panel.innerHTML = `
      <div class="context-note"><strong>Step 6 — Τεκμηρίωση & Capture Sources:</strong> ξεχωρίζουμε τι έγινε κλινικά από το τι είναι traceable στο GeSY ή στο Heidi.</div>
      <div class="step6-grid">
        <article class="card step6-card span-2">
          <div class="card-heading"><div><h2>Πηγές που χρησιμοποιήθηκαν για το post-visit capture</h2><p>Η πηγή είναι provenance, όχι βαθμός ποιότητας.</p></div></div>
          <div class="s6-source-list" id="s6Sources">${SOURCES.map(([v,l])=>`<label><input type="checkbox" value="${v}"/>${l}</label>`).join("")}</div>
          <div class="s6-grid three" style="margin-top:12px">
            <label><span>Primary source</span><select id="s6PrimarySource"><option value="">—</option>${optionHtml(SOURCES)}</select></label>
            <label><span>Υπήρχε conflict μεταξύ πηγών;</span><select id="s6SourceConflict">${optionHtml(YESNO)}</select></label>
            <label data-s6-conflict-dependent hidden><span>Resolution</span><select id="s6ConflictResolution"><option value="">—</option><option value="resolved_by_clinician_review">Resolved by clinician review</option><option value="unresolved">Unresolved</option><option value="not_applicable">N/A</option></select></label>
          </div>
          <label data-s6-conflict-dependent hidden><span>Conflict note <small>(προαιρετικό, χωρίς αναγνωριστικά)</small></span><input id="s6ConflictNote" maxlength="400" /></label>
        </article>

        <article class="card step6-card span-2">
          <div class="card-heading"><div><h2>Documentation trace matrix</h2><p>Το “Απόν” στο GeSY δεν σημαίνει ότι το κλινικό process δεν έγινε.</p></div></div>
          <div class="s6-trace-wrap"><table class="s6-trace-table"><thead><tr><th>Domain</th><th>Formal GeSY trace</th><th>Heidi trace</th><th>Material discrepancy</th></tr></thead><tbody>
            ${DOMAINS.map(([k,l])=>`<tr data-s6-domain="${k}"><td><strong>${l}</strong></td><td><select data-field="formal_record_trace">${optionHtml(TRACE)}</select></td><td><select data-field="heidi_trace">${optionHtml(HEIDI_TRACE)}</select></td><td><select data-field="material_discrepancy">${optionHtml(DISCREPANCY)}</select></td></tr>`).join("")}
          </tbody></table></div>
          <div class="s6-note">Clinical process = Steps 1–5. Documentation trace = ξεχωριστός άξονας. Δεν μετατρέπουμε documentation gap σε clinical omission.</div>
        </article>

        <article class="card step6-card">
          <div class="card-heading"><div><h2>Formal GeSY record</h2></div></div>
          <div class="s6-grid two">
            <label><span>GeSY note available;</span><select id="s6GesyAvailable">${optionHtml(YESNO)}</select></label>
            <label><span>Overall note status</span><select id="s6GesyStatus"><option value="">—</option><option value="absent">Absent</option><option value="minimal">Minimal</option><option value="partial">Partial</option><option value="clinically_substantial">Clinically substantial</option><option value="not_assessed">Not assessed</option></select></label>
            <label><span>Note finalized/saved;</span><select id="s6GesyFinalized"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="uncertain">Αβέβαιο</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Important clinical content missing;</span><select id="s6ImportantMissing">${optionHtml(YESNO)}</select></label>
          </div>
          <div class="s6-domain-list" id="s6MissingDomains" style="margin-top:10px">${DOMAINS.map(([v,l])=>`<label><input type="checkbox" value="${v}"/>${l}</label>`).join("")}</div>
          <label><span>Formal record comment <small>(προαιρετικό)</small></span><textarea id="s6FormalComment" rows="2" maxlength="500"></textarea></label>
        </article>

        <article class="card step6-card">
          <div class="card-heading"><div><h2>Heidi — final review</h2><p>Δεν επικολλάμε transcript ούτε manual diff.</p></div></div>
          <div class="s6-heidi-seed" id="s6HeidiSeed"></div>
          <div class="s6-grid two">
            <label><span>Final clinician-approved note exists;</span><select id="s6HeidiFinalNote">${optionHtml(YNNA)}</select></label>
            <label><span>Heidi content used in formal record;</span><select id="s6HeidiInFormal"><option value="">—</option><option value="yes">Ναι</option><option value="partial">Μερικώς</option><option value="no">Όχι</option><option value="unknown">Άγνωστο</option><option value="not_applicable">N/A</option></select></label>
            <label><span>Material info in Heidi but not formal record;</span><select id="s6HeidiMaterialMissing">${optionHtml(DISCREPANCY)}</select></label>
          </div>
          <div class="s6-warning-note">Η χρήση Heidi δεν είναι quality metric. Το output μετρά μόνο ως πηγή αφού έχει ελεγχθεί από κλινικό.</div>
        </article>

        <article class="card step6-card span-2 s6-close">
          <div class="card-heading"><div><h2>Capture quality & case close</h2><p>Αποτιμά την αξιοπιστία των δεδομένων του audit, όχι την ποιότητα της κλινικής πράξης.</p></div></div>
          <div class="s6-grid three">
            <label><span>Overall capture reliability</span><select id="s6Reliability"><option value="">—</option><option value="strong">Strong</option><option value="adequate">Adequate</option><option value="limited">Limited</option><option value="unreliable">Unreliable</option><option value="uncertain">Uncertain</option></select></label>
            <label><span>Major information gap remaining;</span><select id="s6MajorGap">${optionHtml(YESNO)}</select></label>
            <label><span>Ready for later audit calculation;</span><select id="s6ReadyForAudit">${optionHtml(YESNO)}</select></label>
            <label><span>Completion time <small>(min, optional)</small></span><input id="s6CompletionTime" type="number" min="0" max="60" step="0.5" /></label>
          </div>
          <div class="s6-section-title"><strong>Gap domains</strong></div>
          <div class="s6-domain-list" id="s6GapDomains">${DOMAINS.map(([v,l])=>`<label><input type="checkbox" value="${v}"/>${l}</label>`).join("")}</div>
          <div class="s6-section-title" style="margin-top:10px"><strong>Why capture may be limited</strong></div>
          <div class="s6-source-list" id="s6LimitationReasons">
            <label><input type="checkbox" value="rushed_postvisit_capture"/>Rushed post-visit capture</label>
            <label><input type="checkbox" value="incomplete_formal_record"/>Incomplete formal record</label>
            <label><input type="checkbox" value="unavailable_external_report"/>Unavailable external report</label>
            <label><input type="checkbox" value="uncertain_patient_history"/>Uncertain patient history</label>
            <label><input type="checkbox" value="heidi_not_reviewed"/>Heidi not reviewed</label>
            <label><input type="checkbox" value="conflicting_sources"/>Conflicting sources</label>
            <label><input type="checkbox" value="other"/>Other</label>
          </div>
          <label><span>Short capture comment <small>(προαιρετικό, χωρίς αναγνωριστικά)</small></span><textarea id="s6CaptureComment" rows="2" maxlength="500"></textarea></label>
          <div class="s6-note">Δεν εμφανίζεται KPI score ή documentation verdict στο baseline. Το field→KPI contract θα υπολογιστεί σε επόμενο βήμα.</div>
        </article>
      </div>`;
  }

  function seedHeidiFromStep1(c) {
    const h = c?.heidi || {};
    if (!state.heidi_final.used) state.heidi_final.used = h.used || "";
    if (!state.heidi_final.output_available) state.heidi_final.output_available = h.output_available || "";
    if (!state.heidi_final.reviewed) state.heidi_final.reviewed = h.reviewed_by_clinician || "";
    if (!state.heidi_final.correction_required) state.heidi_final.correction_required = h.material_correction_required || "";
    if (!state.heidi_final.correction_categories.length && Array.isArray(h.correction_categories)) state.heidi_final.correction_categories = [...h.correction_categories];
  }

  function loadState() {
    const c = activeCase();
    const id = c?.internal_uuid || activeUuid();
    if (!id) { state = defaultState(); loaded = ""; hydrate(); return; }
    state = normalize(c?.step6);
    loaded = id;
    seedHeidiFromStep1(c);
    hydrate();
  }

  function syncConflictVisibility(clearWhenHidden = false) {
    const hasConflict = $('#s6SourceConflict')?.value === "yes";
    $$('[data-s6-conflict-dependent]').forEach(node => node.hidden = !hasConflict);
    if (!hasConflict && clearWhenHidden) {
      setValue('#s6ConflictResolution', '');
      setValue('#s6ConflictNote', '');
      state.sources.conflict_resolution = "";
      state.sources.conflict_note = "";
    }
  }

  function collect() {
    const conflictPresent = $('#s6SourceConflict')?.value || "";
    state.sources = {
      used: $$('#s6Sources input:checked').map(x=>x.value),
      primary: $('#s6PrimarySource')?.value || "",
      conflict_present: conflictPresent,
      conflict_resolution: conflictPresent === "yes" ? ($('#s6ConflictResolution')?.value || "") : "",
      conflict_note: conflictPresent === "yes" ? ($('#s6ConflictNote')?.value.trim() || "") : ""
    };
    DOMAINS.forEach(([key]) => {
      const row = $(`[data-s6-domain="${key}"]`);
      if (!row) return;
      state.trace[key] = {
        formal_record_trace: $('[data-field="formal_record_trace"]', row)?.value || "",
        heidi_trace: $('[data-field="heidi_trace"]', row)?.value || "",
        material_discrepancy: $('[data-field="material_discrepancy"]', row)?.value || ""
      };
    });
    state.formal_record = {
      gesy_note_available: $('#s6GesyAvailable')?.value || "",
      gesy_note_status: $('#s6GesyStatus')?.value || "",
      note_finalized: $('#s6GesyFinalized')?.value || "",
      important_content_missing: $('#s6ImportantMissing')?.value || "",
      missing_domains: $$('#s6MissingDomains input:checked').map(x=>x.value),
      comment: $('#s6FormalComment')?.value.trim() || ""
    };
    state.heidi_final = {
      ...state.heidi_final,
      final_approved_note_exists: $('#s6HeidiFinalNote')?.value || "",
      used_in_formal_record: $('#s6HeidiInFormal')?.value || "",
      material_info_not_in_formal: $('#s6HeidiMaterialMissing')?.value || ""
    };
    state.capture_quality = {
      reliability: $('#s6Reliability')?.value || "",
      major_gap: $('#s6MajorGap')?.value || "",
      gap_domains: $$('#s6GapDomains input:checked').map(x=>x.value),
      limitation_reasons: $$('#s6LimitationReasons input:checked').map(x=>x.value),
      completion_time_minutes: num($('#s6CompletionTime')?.value),
      ready_for_audit: $('#s6ReadyForAudit')?.value || "",
      comment: $('#s6CaptureComment')?.value.trim() || ""
    };
  }

  function persist() {
    if (!loaded) loaded = activeUuid();
    if (!loaded) return;
    collect(); state.updated_at = new Date().toISOString();
    const cases = getCases(); const i = cases.findIndex(x=>x.internal_uuid===loaded); if (i < 0) return;
    cases[i] = { ...cases[i], step6: state }; setCases(cases);
  }

  function setValue(id, value) { const n = $(id); if (n) n.value = value ?? ""; }
  function setChecks(selector, values) { $$(selector).forEach(x=>x.checked=(values||[]).includes(x.value)); }
  function label(v) { return ({yes:"Ναι",no:"Όχι",unknown:"Άγνωστο",not_applicable:"N/A",partial:"Μερικώς"})[v] || v || "—"; }

  function hydrate() {
    setChecks('#s6Sources input', state.sources.used); setValue('#s6PrimarySource', state.sources.primary); setValue('#s6SourceConflict', state.sources.conflict_present); setValue('#s6ConflictResolution', state.sources.conflict_resolution); setValue('#s6ConflictNote', state.sources.conflict_note);
    syncConflictVisibility(true);
    DOMAINS.forEach(([key])=>{
      const row=$(`[data-s6-domain="${key}"]`); if(!row)return; const t=state.trace[key]||{};
      const a=$('[data-field="formal_record_trace"]',row), b=$('[data-field="heidi_trace"]',row), c=$('[data-field="material_discrepancy"]',row);
      if(a)a.value=t.formal_record_trace||""; if(b)b.value=t.heidi_trace||""; if(c)c.value=t.material_discrepancy||"";
    });
    setValue('#s6GesyAvailable', state.formal_record.gesy_note_available); setValue('#s6GesyStatus', state.formal_record.gesy_note_status); setValue('#s6GesyFinalized', state.formal_record.note_finalized); setValue('#s6ImportantMissing', state.formal_record.important_content_missing); setChecks('#s6MissingDomains input', state.formal_record.missing_domains); setValue('#s6FormalComment', state.formal_record.comment);
    setValue('#s6HeidiFinalNote', state.heidi_final.final_approved_note_exists); setValue('#s6HeidiInFormal', state.heidi_final.used_in_formal_record); setValue('#s6HeidiMaterialMissing', state.heidi_final.material_info_not_in_formal);
    const seed=$('#s6HeidiSeed'); if(seed) seed.innerHTML=`<div class="s6-seed-box"><small>Used</small><strong>${label(state.heidi_final.used)}</strong></div><div class="s6-seed-box"><small>Output</small><strong>${label(state.heidi_final.output_available)}</strong></div><div class="s6-seed-box"><small>Clinician reviewed</small><strong>${label(state.heidi_final.reviewed)}</strong></div><div class="s6-seed-box"><small>Material correction</small><strong>${label(state.heidi_final.correction_required)}</strong></div>`;
    setValue('#s6Reliability', state.capture_quality.reliability); setValue('#s6MajorGap', state.capture_quality.major_gap); setChecks('#s6GapDomains input', state.capture_quality.gap_domains); setChecks('#s6LimitationReasons input', state.capture_quality.limitation_reasons); setValue('#s6CompletionTime', state.capture_quality.completion_time_minutes); setValue('#s6ReadyForAudit', state.capture_quality.ready_for_audit); setValue('#s6CaptureComment', state.capture_quality.comment);
  }

  function bind() {
    const panel = $('[data-step-panel="6"]'); if (!panel) return;
    panel.addEventListener('input', persist);
    panel.addEventListener('change', (event) => {
      if (event.target?.id === 's6SourceConflict') syncConflictVisibility(true);
      persist();
    });
    $$('.step-tab').forEach(btn=>btn.addEventListener('click',()=>{if(btn.dataset.step==='6') setTimeout(loadState,0);}));
    document.addEventListener('click',(e)=>{if(e.target.closest('[data-load-case]')||e.target.closest('[data-nav-action="new-case"]')) setTimeout(loadState,0);});
    ['#saveTopBtn','#saveDraftBtn','#finishVisitBtn'].forEach(s=>{const n=$(s); if(n)n.addEventListener('click',()=>setTimeout(persist,0));});
  }

  injectAssets(); bind(); loadState();
})();
