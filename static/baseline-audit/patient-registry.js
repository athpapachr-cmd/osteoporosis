(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const ACTIVE_PATIENT_KEY = "osteoporosis.clinical.activePatient.v1";
  const LINKS_KEY = "osteoporosis.clinical.encounterLinks.v1";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  const LAB_LABELS = [
    ["ca", "Ca"], ["phosphate", "Phosphate"], ["vitamin_d", "25-OH Vit D"], ["pth", "PTH"],
    ["creatinine", "Creatinine"], ["egfr", "eGFR"], ["urea", "Urea"], ["total_alp", "Total ALP"],
    ["magnesium", "Mg"], ["ctx", "CTX"], ["p1np", "P1NP"], ["bone_alp", "Bone ALP"],
    ["osteocalcin", "Osteocalcin"], ["glucose", "Glucose"], ["hba1c", "HbA1c"], ["tsh", "TSH"],
    ["ft4", "FT4"], ["esr", "ESR"], ["crp", "CRP"], ["testosterone", "Testosterone"],
    ["fsh", "FSH"], ["estradiol", "Estradiol"], ["cortisol", "Morning cortisol"], ["urine_ca_24h", "24h urine Ca"]
  ];

  function getCases() {
    try { const x = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); return Array.isArray(x) ? x : []; }
    catch { return []; }
  }
  function setCases(cases) { localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
  function activeUuid() { return localStorage.getItem(ACTIVE_KEY) || ""; }
  function activeCase() { const id = activeUuid(); return getCases().find(x => x.internal_uuid === id) || null; }
  function getLinks() { try { return JSON.parse(localStorage.getItem(LINKS_KEY) || "{}"); } catch { return {}; } }
  function setLinks(x) { localStorage.setItem(LINKS_KEY, JSON.stringify(x)); }
  function activePatientId() { return sessionStorage.getItem(ACTIVE_PATIENT_KEY) || ""; }
  function setActivePatientId(id) { if (id) sessionStorage.setItem(ACTIVE_PATIENT_KEY, id); else sessionStorage.removeItem(ACTIVE_PATIENT_KEY); }

  async function api(path, options = {}) {
    const res = await fetch(path, { credentials: "same-origin", ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
    let body = null;
    try { body = await res.json(); } catch { body = null; }
    if (!res.ok) {
      const err = new Error(body?.detail || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return body;
  }

  function injectUi() {
    if (!document.querySelector("style[data-clinical-registry-style]")) {
      const style = document.createElement("style");
      style.dataset.clinicalRegistryStyle = "true";
      style.textContent = `
        .clinical-registry{margin:0 0 14px;padding:14px 16px;border:1px solid #d9e2ec;border-radius:14px;background:#f8fafc}
        .clinical-registry-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}
        .clinical-registry-head h2{margin:0;font-size:16px}.clinical-status{font-size:12px;color:#64748b}
        .clinical-status.ok{color:#166534}.clinical-status.err{color:#b91c1c}
        .clinical-row{display:grid;grid-template-columns:minmax(180px,1fr) auto auto;gap:8px;align-items:end}
        .clinical-row label{display:flex;flex-direction:column;gap:5px}.clinical-row input{min-width:0}
        .clinical-auth{display:flex;gap:8px;align-items:end;margin-bottom:10px}.clinical-auth label{flex:1;display:flex;flex-direction:column;gap:5px}
        .clinical-results,.clinical-encounters{margin-top:10px;display:grid;gap:6px}.clinical-result,.clinical-encounter{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:8px 10px;border:1px solid #e2e8f0;border-radius:10px;background:#fff}
        .clinical-result small,.clinical-encounter small{color:#64748b}.clinical-current{margin-top:10px;padding:8px 10px;border-radius:10px;background:#eef6ff;font-size:13px}
        .clinical-labs{margin-top:14px;overflow:auto}.clinical-labs table{border-collapse:collapse;width:100%;font-size:12px}.clinical-labs th,.clinical-labs td{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:right;white-space:nowrap}.clinical-labs th:first-child,.clinical-labs td:first-child{text-align:left;position:sticky;left:0;background:#f8fafc}
        @media(max-width:800px){.clinical-row{grid-template-columns:1fr}.clinical-auth{flex-direction:column;align-items:stretch}}
      `;
      document.head.appendChild(style);
    }

    if ($("#clinicalRegistry")) return;
    const section = document.createElement("section");
    section.id = "clinicalRegistry";
    section.className = "clinical-registry";
    section.innerHTML = `
      <div class="clinical-registry-head"><h2>Patient Registry</h2><span id="clinicalStatus" class="clinical-status">Έλεγχος σύνδεσης…</span></div>
      <div id="clinicalAuth" class="clinical-auth" hidden>
        <label><span>Clinical access key</span><input id="clinicalKeyInput" type="password" autocomplete="current-password" /></label>
        <button class="btn dark" type="button" id="clinicalLoginBtn">Σύνδεση</button>
      </div>
      <div id="clinicalRegistryBody" hidden>
        <div class="clinical-row">
          <label><span>Patient ID</span><input id="clinicalPatientSearch" type="text" maxlength="120" placeholder="Αναζήτηση με ID" /></label>
          <button class="btn secondary" type="button" id="clinicalSearchBtn">Αναζήτηση</button>
          <button class="btn dark" type="button" id="clinicalCreatePatientBtn">＋ Νέος ασθενής</button>
        </div>
        <div id="clinicalCurrentPatient" class="clinical-current" hidden></div>
        <div id="clinicalSearchResults" class="clinical-results"></div>
        <div id="clinicalEncounterList" class="clinical-encounters"></div>
        <div id="clinicalLabHistory" class="clinical-labs"></div>
      </div>`;
    const anchor = $(".privacy-strip") || $(".case-meta");
    anchor?.parentNode?.insertBefore(section, anchor);

    const privacy = $("#privacyStrip");
    if (privacy) privacy.firstChild.textContent = "Clinical mode: patient data sync to the protected server database after authentication; localStorage is a working cache only. ";
  }

  function setStatus(text, kind = "") {
    const n = $("#clinicalStatus"); if (!n) return;
    n.textContent = text; n.className = `clinical-status ${kind}`;
  }

  async function checkAuth() {
    try {
      const status = await api("/clinical/status", { method: "GET", headers: {} });
      $("#clinicalAuth").hidden = true;
      $("#clinicalRegistryBody").hidden = false;
      setStatus(`Protected DB · ${status.database_dialect}`, "ok");
      const pid = activePatientId(); if (pid) await openPatient(pid);
      return true;
    } catch (err) {
      $("#clinicalAuth").hidden = false;
      $("#clinicalRegistryBody").hidden = true;
      setStatus(err.status === 503 ? "CLINICAL_DATA_KEY δεν έχει ρυθμιστεί" : "Απαιτείται σύνδεση", "err");
      return false;
    }
  }

  async function login() {
    const key = $("#clinicalKeyInput")?.value || "";
    if (!key) return;
    try {
      await api("/clinical/login", { method: "POST", body: JSON.stringify({ key }) });
      $("#clinicalKeyInput").value = "";
      await checkAuth();
    } catch (err) { setStatus(err.message, "err"); }
  }

  async function searchPatients() {
    const q = $("#clinicalPatientSearch")?.value.trim() || "";
    try {
      const rows = await api(`/clinical/patients?query=${encodeURIComponent(q)}&limit=20`, { method: "GET", headers: {} });
      renderSearchResults(rows || []);
    } catch (err) { setStatus(err.message, "err"); }
  }

  function renderSearchResults(rows) {
    const root = $("#clinicalSearchResults"); if (!root) return;
    root.innerHTML = "";
    rows.forEach(row => {
      const item = document.createElement("div"); item.className = "clinical-result";
      const left = document.createElement("div");
      const strong = document.createElement("strong"); strong.textContent = row.patient_id;
      const small = document.createElement("small"); small.textContent = ` · ${row.encounter_count} visits · ${row.lab_snapshot_count} lab dates`;
      left.append(strong, small);
      const btn = document.createElement("button"); btn.className = "btn secondary"; btn.type = "button"; btn.textContent = "Άνοιγμα";
      btn.addEventListener("click", () => openPatient(row.patient_id));
      item.append(left, btn); root.appendChild(item);
    });
  }

  async function createPatient() {
    const patientId = $("#clinicalPatientSearch")?.value.trim() || "";
    if (!patientId) { setStatus("Βάλε Patient ID πρώτα", "err"); return; }
    try {
      await api("/clinical/patients", { method: "POST", body: JSON.stringify({ patient_id: patientId, demographics: {} }) });
      await openPatient(patientId);
    } catch (err) { setStatus(err.message, "err"); }
  }

  async function openPatient(patientId) {
    try {
      const [patient, encounters, labs] = await Promise.all([
        api(`/clinical/patient/${encodeURIComponent(patientId)}`, { method: "GET", headers: {} }),
        api(`/clinical/patient/${encodeURIComponent(patientId)}/encounters`, { method: "GET", headers: {} }),
        api(`/clinical/patient/${encodeURIComponent(patientId)}/labs`, { method: "GET", headers: {} })
      ]);
      setActivePatientId(patientId);
      $("#clinicalPatientSearch").value = patientId;
      const current = $("#clinicalCurrentPatient");
      current.hidden = false;
      current.innerHTML = "";
      const text = document.createElement("span"); text.textContent = `Ενεργός ασθενής: ${patient.patient_id} · ${encounters.length} επισκέψεις · ${labs.length} ημερομηνίες εργαστηριακών`;
      const newBtn = document.createElement("button"); newBtn.className = "btn secondary"; newBtn.type = "button"; newBtn.textContent = "＋ Νέα επίσκεψη"; newBtn.style.marginLeft = "10px";
      newBtn.addEventListener("click", () => $("[data-nav-action='new-case']")?.click());
      current.append(text, newBtn);
      renderEncounters(encounters);
      renderLabs(labs);
      setStatus("Patient record loaded", "ok");
    } catch (err) { setStatus(err.message, "err"); }
  }

  function renderEncounters(encounters) {
    const root = $("#clinicalEncounterList"); if (!root) return;
    root.innerHTML = "";
    encounters.forEach(row => {
      const item = document.createElement("div"); item.className = "clinical-encounter";
      const left = document.createElement("div");
      const strong = document.createElement("strong"); strong.textContent = row.encounter_date;
      const small = document.createElement("small"); small.textContent = ` · ${row.status}`;
      left.append(strong, small);
      const btn = document.createElement("button"); btn.className = "btn secondary"; btn.type = "button"; btn.textContent = "Φόρτωση";
      btn.addEventListener("click", () => loadEncounter(row.encounter_id));
      item.append(left, btn); root.appendChild(item);
    });
  }

  function renderLabs(labs) {
    const roots = [$("#clinicalLabHistory")].filter(Boolean);
    const step3 = $("#s3LabsDate")?.closest("article");
    if (step3) {
      let inline = $("#clinicalLabHistoryInline", step3);
      if (!inline) { inline = document.createElement("div"); inline.id = "clinicalLabHistoryInline"; inline.className = "clinical-labs"; step3.appendChild(inline); }
      roots.push(inline);
    }
    const html = buildLabTable(labs);
    roots.forEach(root => { root.innerHTML = html; });
  }

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[ch]);
  }

  function buildLabTable(labs) {
    if (!labs?.length) return "<small>Δεν υπάρχουν αποθηκευμένα laboratory snapshots.</small>";
    const dates = labs.map(x => x.lab_date);
    const rows = LAB_LABELS.filter(([key]) => labs.some(x => x.values?.[key] !== null && x.values?.[key] !== undefined && x.values?.[key] !== ""));
    if (!rows.length) return "<small>Υπάρχουν ημερομηνίες εργαστηριακών χωρίς numeric values.</small>";
    return `<table><thead><tr><th>Εξέταση</th>${dates.map(d => `<th>${esc(d)}</th>`).join("")}</tr></thead><tbody>${rows.map(([key,label]) => `<tr><td><strong>${esc(label)}</strong></td>${labs.map(x => `<td>${esc(x.values?.[key] ?? "—")}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }

  async function loadEncounter(encounterId) {
    try {
      const row = await api(`/clinical/encounter/${encodeURIComponent(encounterId)}`, { method: "GET", headers: {} });
      const payload = row.payload && typeof row.payload === "object" ? { ...row.payload } : {};
      if (!payload.internal_uuid) payload.internal_uuid = crypto?.randomUUID?.() || `case-${Date.now()}`;
      const cases = getCases(); const i = cases.findIndex(x => x.internal_uuid === payload.internal_uuid);
      if (i >= 0) cases[i] = payload; else cases.push(payload);
      setCases(cases); localStorage.setItem(ACTIVE_KEY, payload.internal_uuid);
      const links = getLinks(); links[payload.internal_uuid] = { patient_id: row.patient_id, encounter_id: row.encounter_id }; setLinks(links);
      setActivePatientId(row.patient_id);
      location.reload();
    } catch (err) { setStatus(err.message, "err"); }
  }

  function hasLabValues(labs) {
    if (!labs || !labs.labs_date) return false;
    return LAB_LABELS.some(([key]) => labs[key] !== null && labs[key] !== undefined && labs[key] !== "");
  }

  async function syncLabs(patientId, encounterId, labs) {
    if (!hasLabValues(labs)) return;
    const values = {};
    LAB_LABELS.forEach(([key]) => { if (labs[key] !== undefined) values[key] = labs[key]; });
    values.btm_context = labs.btm_context || "";
    const existing = await api(`/clinical/patient/${encodeURIComponent(patientId)}/labs`, { method: "GET", headers: {} });
    const match = (existing || []).find(x => x.source_encounter_id === encounterId && x.lab_date === labs.labs_date);
    if (match) {
      await api(`/clinical/lab/${encodeURIComponent(match.lab_snapshot_id)}`, { method: "PUT", body: JSON.stringify({ lab_date: labs.labs_date, source_encounter_id: encounterId, values }) });
    } else {
      await api(`/clinical/patient/${encodeURIComponent(patientId)}/labs`, { method: "POST", body: JSON.stringify({ lab_date: labs.labs_date, source_encounter_id: encounterId, values }) });
    }
  }

  async function syncActiveEncounter(statusOverride = null) {
    const patientId = activePatientId(); const c = activeCase();
    if (!patientId || !c) return;
    const date = c.encounter_date || $("#encounterDate")?.value || "";
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return;
    try {
      const links = getLinks(); const link = links[c.internal_uuid];
      let row;
      if (link?.encounter_id && link.patient_id === patientId) {
        row = await api(`/clinical/encounter/${encodeURIComponent(link.encounter_id)}`, { method: "PUT", body: JSON.stringify({ encounter_date: date, status: statusOverride || "draft", payload: c }) });
      } else {
        row = await api(`/clinical/patient/${encodeURIComponent(patientId)}/encounters`, { method: "POST", body: JSON.stringify({ encounter_date: date, status: statusOverride || "draft", payload: c }) });
        links[c.internal_uuid] = { patient_id: patientId, encounter_id: row.encounter_id }; setLinks(links);
      }
      await syncLabs(patientId, row.encounter_id, c.step3?.labs);
      setStatus(`Synced ${date}`, "ok");
    } catch (err) { setStatus(`Sync failed: ${err.message}`, "err"); }
  }

  function bind() {
    $("#clinicalLoginBtn")?.addEventListener("click", login);
    $("#clinicalKeyInput")?.addEventListener("keydown", e => { if (e.key === "Enter") login(); });
    $("#clinicalSearchBtn")?.addEventListener("click", searchPatients);
    $("#clinicalPatientSearch")?.addEventListener("keydown", e => { if (e.key === "Enter") searchPatients(); });
    $("#clinicalCreatePatientBtn")?.addEventListener("click", createPatient);
    ["#saveTopBtn", "#saveDraftBtn"].forEach(s => $(s)?.addEventListener("click", () => setTimeout(() => syncActiveEncounter("draft"), 120)));
    $("#finishVisitBtn")?.addEventListener("click", () => setTimeout(() => syncActiveEncounter("completed"), 160));
    $$(".step-tab").forEach(btn => btn.addEventListener("click", () => { if (btn.dataset.step === "3") setTimeout(async () => { const pid = activePatientId(); if (!pid) return; try { renderLabs(await api(`/clinical/patient/${encodeURIComponent(pid)}/labs`, { method: "GET", headers: {} })); } catch {} }, 80); }));
  }

  injectUi(); bind(); checkAuth();
})();
