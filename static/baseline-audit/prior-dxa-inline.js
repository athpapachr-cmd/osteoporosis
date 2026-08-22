(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const MACHINE_VALUES = new Set(["", "hologic_horizon", "hologic_discovery", "ge_lunar_idxa", "ge_lunar_prodigy", "norland", "other_unknown"]);
  const num = (v) => v === "" || v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v);
  const uid = () => window.crypto?.randomUUID?.() || `dxa-${Date.now()}-${Math.random().toString(16).slice(2)}`;

  function getCases() { try { const x = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); return Array.isArray(x) ? x : []; } catch { return []; } }
  function setCases(cases) { localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
  function activeUuid() { return localStorage.getItem(ACTIVE_KEY) || ""; }

  function validDate(value) {
    const s = String(value || "").trim();
    return /^\d{4}-\d{2}-\d{2}$/.test(s) ? s : "";
  }

  function normalizeHistory() {
    const id = activeUuid();
    if (!id) return false;
    const cases = getCases();
    const i = cases.findIndex(c => c.internal_uuid === id);
    if (i < 0) return false;
    const review = cases[i].longitudinal_review;
    if (!review || !Array.isArray(review.dxa_history)) return false;
    let changed = false;
    review.dxa_history = review.dxa_history.map(row => {
      const next = { ...row };
      if (!next._id) { next._id = uid(); changed = true; }
      const safeDate = validDate(next.date);
      if (safeDate !== (next.date || "")) { next.date = safeDate; changed = true; }
      if (!MACHINE_VALUES.has(next.machine || "")) {
        const legacy = String(next.machine || "").trim();
        next.machine = "other_unknown";
        if (!next.machine_label && legacy) next.machine_label = legacy.slice(0, 80);
        changed = true;
      }
      return next;
    });
    if (changed) {
      cases[i] = { ...cases[i], longitudinal_review: { ...review, dxa_history: review.dxa_history, updated_at: new Date().toISOString() } };
      setCases(cases);
    }
    return changed;
  }

  function forceLongitudinalReload() {
    const tab = $('.step-tab[data-step="3"]');
    if (tab) setTimeout(() => tab.click(), 0);
  }

  function ensureEditor() {
    const card = $("#longitudinalDxaCard");
    if (!card) return null;
    let editor = $("#lrDxaInlineEditor", card);
    if (editor) return editor;
    editor = document.createElement("div");
    editor.id = "lrDxaInlineEditor";
    editor.className = "lr-dxa-inline-editor";
    editor.hidden = true;
    editor.innerHTML = `
      <div class="lr-inline-head"><strong>Prior DXA</strong><span>Inline καταχώρηση — χωρίς browser prompts.</span></div>
      <div class="lr-inline-grid">
        <label><span>Ημερομηνία</span><input id="lrPrevDxaDate" type="date" /></label>
        <label><span>Machine</span><select id="lrPrevDxaMachine">
          <option value="">—</option><option value="hologic_horizon">Hologic Horizon</option><option value="hologic_discovery">Hologic Discovery</option><option value="ge_lunar_idxa">GE Lunar iDXA</option><option value="ge_lunar_prodigy">GE Lunar Prodigy</option><option value="norland">Norland</option><option value="other_unknown">Other / unknown</option>
        </select></label>
        <label><span>Local machine label</span><input id="lrPrevDxaMachineLabel" maxlength="80" /></label>
        <label><span>Spine BMD</span><input id="lrPrevSpineBmd" type="number" step="0.001" /></label>
        <label><span>Spine T-score</span><input id="lrPrevSpineT" type="number" step="0.1" /></label>
        <label><span>Total hip BMD</span><input id="lrPrevHipBmd" type="number" step="0.001" /></label>
        <label><span>Total hip T-score</span><input id="lrPrevHipT" type="number" step="0.1" /></label>
        <label><span>Femoral neck BMD</span><input id="lrPrevFnBmd" type="number" step="0.001" /></label>
        <label><span>Femoral neck T-score</span><input id="lrPrevFnT" type="number" step="0.1" /></label>
      </div>
      <div class="lr-inline-error" id="lrPrevDxaError" aria-live="polite"></div>
      <div class="lr-inline-actions"><button type="button" class="btn secondary" id="lrCancelDxaInline">Ακύρωση</button><button type="button" class="btn primary" id="lrSaveDxaInline">Προσθήκη DXA</button></div>`;
    const table = $("#lrDxaTable", card);
    if (table) card.insertBefore(editor, table); else card.appendChild(editor);
    return editor;
  }

  function openEditor() {
    const editor = ensureEditor();
    if (!editor) return;
    editor.hidden = false;
    ["#lrPrevDxaDate", "#lrPrevDxaMachine", "#lrPrevDxaMachineLabel", "#lrPrevSpineBmd", "#lrPrevSpineT", "#lrPrevHipBmd", "#lrPrevHipT", "#lrPrevFnBmd", "#lrPrevFnT"].forEach(s => { const n = $(s); if (n) n.value = ""; });
    const err = $("#lrPrevDxaError"); if (err) err.textContent = "";
    $("#lrPrevDxaDate")?.focus();
  }

  function closeEditor() { const editor = $("#lrDxaInlineEditor"); if (editor) editor.hidden = true; }

  function saveInline() {
    const date = validDate($("#lrPrevDxaDate")?.value);
    const err = $("#lrPrevDxaError");
    if (!date) { if (err) err.textContent = "Η ημερομηνία είναι υποχρεωτική."; return; }
    const id = activeUuid();
    const cases = getCases();
    const i = cases.findIndex(c => c.internal_uuid === id);
    if (i < 0) { if (err) err.textContent = "Αποθήκευσε πρώτα το case."; return; }
    const machine = MACHINE_VALUES.has($("#lrPrevDxaMachine")?.value || "") ? ($("#lrPrevDxaMachine")?.value || "") : "other_unknown";
    const row = {
      _id: uid(), date, machine,
      machine_label: String($("#lrPrevDxaMachineLabel")?.value || "").trim().slice(0, 80),
      spine_bmd: num($("#lrPrevSpineBmd")?.value), spine_t: num($("#lrPrevSpineT")?.value),
      total_hip_bmd: num($("#lrPrevHipBmd")?.value), total_hip_t: num($("#lrPrevHipT")?.value),
      fn_bmd: num($("#lrPrevFnBmd")?.value), fn_t: num($("#lrPrevFnT")?.value)
    };
    const review = cases[i].longitudinal_review || { risk_categories:{mof:"",hip:"",overall:""}, fraxplus:{used:"",adjusted_mof:null,adjusted_hip:null,dominant_adjustment:"",adjustments:[],note:""}, frax_history:[], dxa_history:[] };
    const history = Array.isArray(review.dxa_history) ? [...review.dxa_history, row] : [row];
    cases[i] = { ...cases[i], longitudinal_review: { ...review, dxa_history: history, updated_at: new Date().toISOString() } };
    setCases(cases);
    closeEditor();
    forceLongitudinalReload();
  }

  function annotateRows() {
    normalizeHistory();
    const id = activeUuid();
    const c = getCases().find(x => x.internal_uuid === id);
    const history = Array.isArray(c?.longitudinal_review?.dxa_history) ? c.longitudinal_review.dxa_history : [];
    const sorted = [...history].sort((a,b) => String(a.date || "").localeCompare(String(b.date || "")));
    const rows = $$("#lrDxaTable tbody tr").filter(tr => !tr.textContent.includes("Current"));
    rows.forEach((tr, index) => {
      const item = sorted[index];
      if (!item?._id) return;
      tr.dataset.dxaHistoryId = item._id;
      const btn = $("[data-remove-dxa]", tr);
      if (btn) btn.dataset.historyId = item._id;
    });
  }

  function removeById(historyId) {
    if (!historyId) return;
    const id = activeUuid();
    const cases = getCases();
    const i = cases.findIndex(c => c.internal_uuid === id);
    if (i < 0) return;
    const review = cases[i].longitudinal_review;
    if (!review || !Array.isArray(review.dxa_history)) return;
    cases[i] = { ...cases[i], longitudinal_review: { ...review, dxa_history: review.dxa_history.filter(r => r._id !== historyId), updated_at: new Date().toISOString() } };
    setCases(cases);
    forceLongitudinalReload();
  }

  document.addEventListener("click", event => {
    const add = event.target.closest("#lrAddDxa");
    if (add) { event.preventDefault(); event.stopImmediatePropagation(); openEditor(); return; }
    const save = event.target.closest("#lrSaveDxaInline");
    if (save) { event.preventDefault(); event.stopImmediatePropagation(); saveInline(); return; }
    const cancel = event.target.closest("#lrCancelDxaInline");
    if (cancel) { event.preventDefault(); event.stopImmediatePropagation(); closeEditor(); return; }
    const remove = event.target.closest("[data-remove-dxa]");
    if (remove?.dataset.historyId) { event.preventDefault(); event.stopImmediatePropagation(); removeById(remove.dataset.historyId); }
  }, true);

  const observer = new MutationObserver(() => { ensureEditor(); annotateRows(); });
  observer.observe(document.documentElement, { childList: true, subtree: true });

  if (!document.querySelector('style[data-prior-dxa-inline-style]')) {
    const style = document.createElement("style");
    style.dataset.priorDxaInlineStyle = "true";
    style.textContent = `.lr-dxa-inline-editor{margin:12px 0;padding:14px;border:1px solid #d8dee8;border-radius:12px;background:#f8fafc}.lr-inline-head{display:flex;gap:10px;align-items:baseline;margin-bottom:10px}.lr-inline-head span{font-size:12px;color:#64748b}.lr-inline-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.lr-inline-grid label{display:flex;flex-direction:column;gap:5px}.lr-inline-grid label>span{font-size:12px;color:#475569}.lr-inline-error{min-height:18px;margin-top:8px;font-size:12px;color:#b91c1c}.lr-inline-actions{display:flex;justify-content:flex-end;gap:8px}@media(max-width:900px){.lr-inline-grid{grid-template-columns:1fr}}`;
    document.head.appendChild(style);
  }

  normalizeHistory();
  setTimeout(() => { ensureEditor(); annotateRows(); }, 0);
})();