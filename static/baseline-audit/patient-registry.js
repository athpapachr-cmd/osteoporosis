(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const ACTIVE_PATIENT_KEY = "osteoporosis.clinical.activePatient.v1";
  const LINKS_KEY = "osteoporosis.clinical.encounterLinks.v1";
  const AUTOSAVE_DELAY_MS = 900;

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

  let autosaveTimer = null;
  let syncQueue = Promise.resolve();
  const conflictedUuids = new Set();

  function getCases() {
    try { const x = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); return Array.isArray(x) ? x : []; }
    catch { return []; }
  }
  function setCases(cases) { localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
  function activeUuid() { return localStorage.getItem(ACTIVE_KEY) || ""; }
  function activeCase() { const id = activeUuid(); return getCases().find(x => x.internal_uuid === id) || null; }
  function getLinks() { try { const value = JSON.parse(localStorage.getItem(LINKS_KEY) || "{}"); return value && typeof value === "object" ? value : {}; } catch { return {}; } }
  function setLinks(x) { localStorage.setItem(LINKS_KEY, JSON.stringify(x)); }
  function activePatientId() { return sessionStorage.getItem(ACTIVE_PATIENT_KEY) || ""; }
  function setActivePatientId(id) { if (id) sessionStorage.setItem(ACTIVE_PATIENT_KEY, id); else sessionStorage.removeItem(ACTIVE_PATIENT_KEY); }

  function deepEqual(a, b) {
    try { return JSON.stringify(a ?? null) === JSON.stringify(b ?? null); }
    catch { return false; }
  }

  function serverPayload(caseData) {
    const payload = JSON.parse(JSON.stringify(caseData || {}));
    payload.workflow_mode = "clinical";
    if (!payload.baseline_phase || payload.baseline_phase === "pilot") payload.baseline_phase = "clinical";
    return payload;
  }

  async function api(path, options = {}) {
    const res = await fetch(path, { credentials: "same-origin", ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
    let body = null;
    try { body = await res.json(); } catch { body = null; }
    if (!res.ok) {
      const err = new Error(body?.detail || `HTTP ${res.status}`);
      err.status = res.status;
      err.body = body;
      throw err;
    }
    return body;
  }

  function lastTextSpan(button) {
    const spans = $$('span', button);
    return spans.length ? spans[spans.length - 1] : null;
  }

  function replaceTextNodeAfter(node, text) {
    if (!node) return;
    let sibling = node.nextSibling;
    while (sibling && sibling.nodeType !== 3 && sibling !== node.parentNode?.lastChild) sibling = sibling.nextSibling;
    if (sibling?.nodeType === 3) sibling.textContent = ` ${text} `;
  }

  function sanitizeLegacyShellText() {
    const pill = $("#pilotPill");
    if (pill && /pilot/i.test(pill.textContent || "")) pill.textContent = "ΕΠΙΣΚΕΨΗ · DRAFT";
    const identity = $("#caseIdDisplay");
    if (identity && /^PILOT-/i.test(identity.textContent || "")) identity.textContent = "Νέα επίσκεψη";
    const draft = $("#draftStatus");
    if (draft && /pilot case/i.test(draft.textContent || "")) draft.textContent = draft.textContent.replace(/pilot case/ig, "επίσκεψη");
  }

  function applyClinicalShell() {
    document.title = "Clinical Excellence — Osteoporosis";
    const description = $('meta[name="description"]');
    if (description) description.content = "Protected patient-centric osteoporosis clinical workspace.";

    const title = $(".title-block h1");
    if (title) title.textContent = "Clinical Excellence — Osteoporosis";
    const subtitle = $(".title-block > p");
    if (subtitle) subtitle.textContent = "Κλινική επίσκεψη · protected server-backed workspace";

    $$('[data-nav-action="new-case"]').forEach(button => {
      const text = lastTextSpan(button); if (text) text.textContent = "Νέα επίσκεψη";
    });
    $$('[data-nav-action="cases"]').forEach(button => {
      const text = lastTextSpan(button); if (text) text.textContent = "Επισκέψεις";
    });

    const banner = $(".baseline-banner");
    if (banner) {
      const strong = $("strong", banner); if (strong) strong.textContent = "Clinical Guidance ενεργή";
      const span = $("div > span", banner); if (span) span.textContent = "Η επίσκεψη καταγράφεται στον protected patient record. Routine performance feedback παραμένει κρυφό μέχρι το κατάλληλο measurement phase.";
    }

    const privacy = $("#privacyStrip");
    if (privacy) {
      const strong = $("strong", privacy);
      if (strong) {
        strong.textContent = "Protected clinical mode:";
        replaceTextNodeAfter(strong, "Οι επισκέψεις συγχρονίζονται στον protected server μετά από authentication. Απόφυγε περιττά αναγνωριστικά σε free-text πεδία.");
      }
    }

    const caseCode = $("#caseIdDisplay")?.closest(".case-code");
    const caseLabel = $(".meta-label", caseCode || document);
    if (caseCode && caseLabel) caseLabel.textContent = "Επίσκεψη";

    const sampleBox = $(".sampling-box");
    if (sampleBox) sampleBox.hidden = true;

    const casesDialog = $("#casesDialog");
    const casesHeading = casesDialog ? $("h2, h3", casesDialog) : null;
    if (casesHeading) casesHeading.textContent = "Επισκέψεις";

    sanitizeLegacyShellText();
    [$("#pilotPill"), $("#caseIdDisplay"), $("#draftStatus")].filter(Boolean).forEach(node => {
      const observer = new MutationObserver(sanitizeLegacyShellText);
      observer.observe(node, { childList: true, characterData: true, subtree: true });
    });
  }

  function injectUi() {
    applyClinicalShell();
    if (!document.querySelector("style[data-clinical-registry-style]")) {
      const style = document.createElement("style");
      style.dataset.clinicalRegistryStyle = "true";
      style.textContent = `
        .clinical-registry{margin:0 0 14px;padding:14px 16px;border:1px solid #d9e2ec;border-radius:14px;background:#f8fafc}
        .clinical-registry-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}
        .clinical-registry-head h2{margin:0;font-size:16px}.clinical-status{font-size:12px;color:#64748b}
        .clinical-status.ok{color:#166534}.clinical-status.err{color:#b91c1c}.clinical-status.saving{color:#92400e}
        .clinical-row{display:grid;grid-template-columns:minmax(180px,1fr) auto auto;gap:8px;align-items:end}
        .clinical-row label{display:flex;flex-direction:column;gap:5px}.clinical-row input{min-width:0}
        .clinical-auth{display:flex;gap:8px;align-items:end;margin-bottom:10px}.clinical-auth label{flex:1;display:flex;flex-direction:column;gap:5px}
        .clinical-results,.clinical-encounters{margin-top:10px;display:grid;gap:6px}.clinical-result,.clinical-encounter{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:8px 10px;border:1px solid #e2e8f0;border-radius:10px;background:#fff}
        .clinical-result small,.clinical-encounter small{color:#64748b}.clinical-current{margin-top:10px;padding:8px 10px;border-radius:10px;background:#eef6ff;font-size:13px}
        .clinical-conflict{margin:10px 0 0;padding:10px 12px;border:2px solid #b45309;border-radius:10px;background:#fffbeb;display:flex;gap:10px;align-items:center;justify-content:space-between}
        .clinical-conflict strong{display:block}.clinical-conflict span{font-size:12px;color:#78350f}
        .clinical-labs{margin-top:14px;overflow:auto}.clinical-labs table{border-collapse:collapse;width:100%;font-size:12px}.clinical-labs th,.clinical-labs td{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:right;white-space:nowrap}.clinical-labs th:first-child,.clinical-labs td:first-child{text-align:left;position:sticky;left:0;background:#f8fafc}
        @media(max-width:800px){.clinical-row{grid-template-columns:1fr}.clinical-auth{flex-direction:column;align-items:stretch}.clinical-conflict{align-items:stretch;flex-direction:column}}
      `;
      document.head.appendChild(style);
    }

    if ($("#clinicalRegistry")) return;
    const section = document.createElement("section");
    section.id = "clinicalRegistry";
    section.className = "clinical-registry";
    section.innerHTML = `
      <div class="clinical-registry-head"><h2>Ασθενείς & Επισκέψεις</h2><span id="clinicalStatus" class="clinical-status">Έλεγχος σύνδεσης…</span></div>
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
        <div id="clinicalConflict" class="clinical-conflict" hidden>
          <div><strong>Σύγκρουση έκδοσης</strong><span id="clinicalConflictText">Η επίσκεψη άλλαξε σε άλλη συσκευή. Η αυτόματη αντικατάσταση μπλοκαρίστηκε.</span></div>
          <button class="btn secondary" type="button" id="clinicalReloadServerBtn">Φόρτωση server έκδοσης</button>
        </div>
        <div id="clinicalSearchResults" class="clinical-results"></div>
        <div id="clinicalEncounterList" class="clinical-encounters"></div>
        <div id="clinicalLabHistory" class="clinical-labs"></div>
      </div>`;
    const anchor = $(".privacy-strip") || $(".case-meta");
    anchor?.parentNode?.insertBefore(section, anchor);
  }

  function setStatus(text, kind = "") {
    const n = $("#clinicalStatus"); if (!n) return;
    n.textContent = text; n.className = `clinical-status ${kind}`;
  }

  function visitStatusLabel(status) {
    return status === "completed" ? "ΟΛΟΚΛΗΡΩΜΕΝΗ" : status === "amended" ? "ΤΡΟΠΟΠΟΙΗΜΕΝΗ" : "DRAFT";
  }

  function updateVisitIdentity(row = null) {
    const pill = $("#pilotPill");
    const display = $("#caseIdDisplay");
    if (!row) {
      if (pill) pill.textContent = "ΕΠΙΣΚΕΨΗ · DRAFT";
      if (display) display.textContent = "Νέα επίσκεψη";
      return;
    }
    const label = visitStatusLabel(row.status);
    if (pill) pill.textContent = `ΕΠΙΣΚΕΨΗ · ${label}`;
    if (display) display.textContent = `${row.encounter_date} · ${label.toLowerCase()}`;
  }

  function updateLink(uuid, row) {
    if (!uuid || !row?.encounter_id) return;
    const links = getLinks();
    links[uuid] = {
      patient_id: row.patient_id,
      encounter_id: row.encounter_id,
      updated_at: row.updated_at,
      status: row.status
    };
    setLinks(links);
  }

  function activeLink() {
    const uuid = activeUuid();
    return uuid ? getLinks()[uuid] || null : null;
  }

  function hideConflict(uuid = activeUuid()) {
    if (uuid) conflictedUuids.delete(uuid);
    const banner = $("#clinicalConflict"); if (banner) banner.hidden = true;
  }

  function showConflict(message) {
    const uuid = activeUuid();
    if (uuid) conflictedUuids.add(uuid);
    const banner = $("#clinicalConflict");
    const text = $("#clinicalConflictText");
    if (text) text.textContent = `${message || "Η επίσκεψη άλλαξε σε άλλη συσκευή."} Η αυτόματη αντικατάσταση μπλοκαρίστηκε. Οι τοπικές αλλαγές παραμένουν σε αυτόν τον browser μέχρι να επιλέξεις φόρτωση server έκδοσης.`;
    if (banner) banner.hidden = false;
    setStatus("Conflict — απαιτείται έλεγχος", "err");
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
      const small = document.createElement("small"); small.textContent = ` · ${row.encounter_count} επισκέψεις · ${row.lab_snapshot_count} ημερομηνίες εργαστηριακών`;
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

  async function fetchPatientBundle(patientId) {
    return Promise.all([
      api(`/clinical/patient/${encodeURIComponent(patientId)}`, { method: "GET", headers: {} }),
      api(`/clinical/patient/${encodeURIComponent(patientId)}/encounters`, { method: "GET", headers: {} }),
      api(`/clinical/patient/${encodeURIComponent(patientId)}/labs`, { method: "GET", headers: {} })
    ]);
  }

  async function openPatient(patientId) {
    try {
      const [patient, encounters, labs] = await fetchPatientBundle(patientId);
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
    const currentEncounterId = activeLink()?.encounter_id || "";
    encounters.forEach(row => {
      const item = document.createElement("div"); item.className = "clinical-encounter";
      if (row.encounter_id === currentEncounterId) item.dataset.active = "true";
      const left = document.createElement("div");
      const strong = document.createElement("strong"); strong.textContent = row.encounter_date;
      const small = document.createElement("small"); small.textContent = ` · ${row.status}${row.encounter_id === currentEncounterId ? " · ενεργή" : ""}`;
      left.append(strong, small);
      const btn = document.createElement("button"); btn.className = "btn secondary"; btn.type = "button"; btn.textContent = "Άνοιγμα";
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

  async function loadEncounter(encounterId, { confirmDiscard = false } = {}) {
    if (confirmDiscard && !window.confirm("Η φόρτωση της server έκδοσης θα αντικαταστήσει τις μη συγχρονισμένες τοπικές αλλαγές σε αυτή τη συσκευή. Συνέχεια;")) return;
    try {
      const row = await api(`/clinical/encounter/${encodeURIComponent(encounterId)}`, { method: "GET", headers: {} });
      const payload = row.payload && typeof row.payload === "object" ? { ...row.payload } : {};
      if (!payload.internal_uuid) payload.internal_uuid = crypto?.randomUUID?.() || `case-${Date.now()}`;
      const cases = getCases(); const i = cases.findIndex(x => x.internal_uuid === payload.internal_uuid);
      if (i >= 0) cases[i] = payload; else cases.push(payload);
      setCases(cases); localStorage.setItem(ACTIVE_KEY, payload.internal_uuid);
      updateLink(payload.internal_uuid, row);
      setActivePatientId(row.patient_id);
      hideConflict(payload.internal_uuid);
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
      if (deepEqual(match.values || {}, values)) return;
      await api(`/clinical/lab/${encodeURIComponent(match.lab_snapshot_id)}`, { method: "PUT", body: JSON.stringify({ lab_date: labs.labs_date, source_encounter_id: encounterId, values }) });
    } else {
      await api(`/clinical/patient/${encodeURIComponent(patientId)}/labs`, { method: "POST", body: JSON.stringify({ lab_date: labs.labs_date, source_encounter_id: encounterId, values }) });
    }
  }

  async function recoverExistingEncounter(patientId, current) {
    if (!current?.internal_uuid) return null;
    const encounters = await api(`/clinical/patient/${encodeURIComponent(patientId)}/encounters`, { method: "GET", headers: {} });
    return (encounters || []).find(row => row.payload?.internal_uuid === current.internal_uuid) || null;
  }

  async function performSync(statusOverride = null, { strict = false } = {}) {
    const patientId = activePatientId();
    const c = activeCase();
    if (!patientId || !c) {
      const err = new Error(!patientId ? "Δεν υπάρχει ενεργός protected patient." : "Δεν υπάρχει ενεργό encounter payload.");
      setStatus(err.message, "err");
      if (strict) throw err;
      return null;
    }
    if (conflictedUuids.has(c.internal_uuid)) {
      const err = new Error("Η επίσκεψη έχει conflict με νεότερη server έκδοση. Φόρτωσε πρώτα τη server έκδοση.");
      setStatus(err.message, "err");
      if (strict) throw err;
      return null;
    }

    const date = c.encounter_date || $("#encounterDate")?.value || "";
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      const err = new Error("Το encounter date δεν είναι έγκυρο για server sync.");
      setStatus(err.message, "err");
      if (strict) throw err;
      return null;
    }

    setStatus("Αποθήκευση στον protected server…", "saving");
    try {
      const payload = serverPayload(c);
      const links = getLinks();
      let link = links[c.internal_uuid] || null;
      let row;

      if (!link?.encounter_id) {
        const recovered = await recoverExistingEncounter(patientId, c);
        if (recovered) {
          updateLink(c.internal_uuid, recovered);
          link = getLinks()[c.internal_uuid];
        }
      }

      if (link?.encounter_id) {
        if (link.patient_id !== patientId) {
          const err = new Error("Το local encounter link ανήκει σε διαφορετικό patient. Η εγγραφή μπλοκαρίστηκε.");
          showConflict(err.message);
          if (strict) throw err;
          return null;
        }
        if (!link.updated_at) {
          const err = new Error("Η παλιά browser cache δεν έχει server version token. Φόρτωσε την επίσκεψη από τον server πριν την αλλάξεις.");
          showConflict(err.message);
          if (strict) throw err;
          return null;
        }
        row = await api(`/clinical/encounter/${encodeURIComponent(link.encounter_id)}`, {
          method: "PUT",
          body: JSON.stringify({
            encounter_date: date,
            status: statusOverride || "draft",
            payload,
            expected_updated_at: link.updated_at
          })
        });
      } else {
        row = await api(`/clinical/patient/${encodeURIComponent(patientId)}/encounters`, {
          method: "POST",
          body: JSON.stringify({ encounter_date: date, status: statusOverride || "draft", payload })
        });
      }

      updateLink(c.internal_uuid, row);
      hideConflict(c.internal_uuid);
      updateVisitIdentity(row);
      await syncLabs(patientId, row.encounter_id, c.step3?.labs);
      setStatus(`Synced ${date} · ${row.status}`, "ok");
      return row;
    } catch (err) {
      if (err.status === 409) showConflict(err.message);
      else setStatus(`Sync failed: ${err.message}`, "err");
      if (strict) throw err;
      return null;
    }
  }

  function enqueueSync(statusOverride = null, options = {}) {
    const task = syncQueue.then(() => performSync(statusOverride, options));
    syncQueue = task.catch(() => null);
    return task;
  }

  function scheduleDraftSyncFromSave() {
    const coordinator = window.BaselineFinalizationCoordinator;
    if (coordinator && !coordinator.shouldSyncDraftOnSave()) return;
    const uuid = activeUuid();
    if (uuid && conflictedUuids.has(uuid)) return;
    setTimeout(() => enqueueSync("draft"), 120);
  }

  async function finalizeActiveEncounter() {
    return enqueueSync("completed", { strict: true });
  }

  function encounterMutationTarget(target) {
    if (!target?.closest) return false;
    if (target.closest("#clinicalRegistry, #privacyDialog, #casesDialog")) return false;
    if (target.closest("#saveTopBtn, #saveDraftBtn, #finishVisitBtn, .step-tab")) return false;
    return Boolean(target.closest(".step-panel, .case-meta"));
  }

  function scheduleAutosave(event) {
    if (!activePatientId() || !encounterMutationTarget(event.target)) return;
    const uuid = activeUuid();
    if (uuid && conflictedUuids.has(uuid)) return;
    const coordinator = window.BaselineFinalizationCoordinator;
    if (coordinator?.isAuthoritativeFinishInProgress?.()) return;
    clearTimeout(autosaveTimer);
    autosaveTimer = setTimeout(() => {
      const save = $("#saveDraftBtn") || $("#saveTopBtn");
      save?.click();
    }, AUTOSAVE_DELAY_MS);
  }

  function bindWorkspaceNavigation() {
    document.addEventListener("click", event => {
      const casesNav = event.target.closest?.('[data-nav-action="cases"]');
      if (casesNav) {
        event.preventDefault();
        event.stopImmediatePropagation();
        $("#clinicalRegistry")?.scrollIntoView?.({ behavior: "smooth", block: "start" });
        return;
      }

      const newVisit = event.target.closest?.('[data-nav-action="new-case"]');
      if (!newVisit) return;
      if (!activePatientId()) {
        event.preventDefault();
        event.stopImmediatePropagation();
        setStatus("Επίλεξε ή δημιούργησε protected patient πριν από Νέα επίσκεψη.", "err");
        return;
      }
      hideConflict();
      updateVisitIdentity(null);
      setTimeout(() => {
        const save = $("#saveDraftBtn") || $("#saveTopBtn");
        save?.click();
      }, 0);
    }, true);
  }

  async function reloadServerVersion() {
    const link = activeLink();
    if (!link?.encounter_id) {
      setStatus("Δεν υπάρχει server encounter για επαναφόρτωση.", "err");
      return;
    }
    await loadEncounter(link.encounter_id, { confirmDiscard: true });
  }

  function bind() {
    $("#clinicalLoginBtn")?.addEventListener("click", login);
    $("#clinicalKeyInput")?.addEventListener("keydown", e => { if (e.key === "Enter") login(); });
    $("#clinicalSearchBtn")?.addEventListener("click", searchPatients);
    $("#clinicalPatientSearch")?.addEventListener("keydown", e => { if (e.key === "Enter") searchPatients(); });
    $("#clinicalCreatePatientBtn")?.addEventListener("click", createPatient);
    $("#clinicalReloadServerBtn")?.addEventListener("click", reloadServerVersion);
    ["#saveTopBtn", "#saveDraftBtn"].forEach(s => $(s)?.addEventListener("click", scheduleDraftSyncFromSave));
    $$(".step-tab").forEach(btn => btn.addEventListener("click", () => { if (btn.dataset.step === "3") setTimeout(async () => { const pid = activePatientId(); if (!pid) return; try { renderLabs(await api(`/clinical/patient/${encodeURIComponent(pid)}/labs`, { method: "GET", headers: {} })); } catch {} }, 80); }));
    document.addEventListener("input", scheduleAutosave);
    document.addEventListener("change", scheduleAutosave);
    document.addEventListener("click", event => {
      if (event.target.closest?.("[data-field][data-value], #addFractureEventBtn, [data-remove-event], #s4AddEpisode, #s4AddAdministration, [data-remove-episode], [data-remove-administration], .adaptive-use-domain, .adaptive-reset-domain")) scheduleAutosave(event);
    });
    bindWorkspaceNavigation();
  }

  window.ClinicalRegistry = Object.freeze({
    finalizeActiveEncounter,
    syncActiveEncounter: (status = "draft") => enqueueSync(status),
    reloadServerVersion
  });

  injectUi();
  bind();
  checkAuth();
})();
