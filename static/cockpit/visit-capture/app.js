(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const CLIENT_SESSION_KEY = "clinical.visitCapture.clientSession.v1";
  const LOCAL_PROTOCOL = "local_visit_audio_v1";
  const LOCAL_ORIGIN = "http://127.0.0.1:8765";
  const LOCAL_MIME = "audio/webm;codecs=opus";
  const LOCAL_DISPOSITIONS = [
    "reason", "patient_history", "objective_result", "clinician_interpretation",
    "option_discussed", "clinician_recommendation", "final_decision", "completed_action",
    "patient_accepted", "patient_declined", "patient_undecided", "follow_up",
    "pending_obligation", "safety_net", "uncertainty", "administrative", "unclassified",
  ];
  const LOCAL_LABELS = {
    reason: "Λόγος επίσκεψης", patient_history: "Ιστορικό", objective_result: "Αντικειμενικό εύρημα",
    clinician_interpretation: "Κλινική εκτίμηση", option_discussed: "Επιλογή που συζητήθηκε",
    clinician_recommendation: "Κλινική σύσταση", final_decision: "Τελική απόφαση",
    completed_action: "Ολοκληρωμένη πράξη", patient_accepted: "Ο ασθενής αποδέχθηκε",
    patient_declined: "Ο ασθενής αρνήθηκε", patient_undecided: "Ο ασθενής δεν αποφάσισε",
    follow_up: "Παρακολούθηση", pending_obligation: "Πιθανή εκκρεμότητα",
    safety_net: "Safety-net", uncertainty: "Αβεβαιότητα", administrative: "Διοικητικό",
    unclassified: "Χωρίς ασφαλή ταξινόμηση",
  };
  const LOCAL_RISK_FLAGS = new Set([
    "digit_or_measurement", "medication", "laterality", "negation",
    "uncertainty_or_conflict", "unclassified", "semantic_status_conflict",
  ]);
  const LOCAL_WHISPER_FLAGS = new Set(["low_confidence", "no_speech_anomaly"]);
  const LOCAL_ERROR_CODES = new Set([
    "filevault_unavailable", "invalid_request", "audio_limit", "decode_failed",
    "transcription_failed", "classification_failed", "validation_failed", "timeout",
    "cancelled", "cleanup_failed", "internal_failure",
  ]);
  const state = {
    mode: "",
    authenticated: false,
    patients: [],
    suggestedPatientId: "",
    patientSearchTimer: null,
    patientSearchRevision: 0,
    contextId: "",
    patientId: "",
    candidate: null,
    preview: null,
    level: "snapshot",
    previewTimer: null,
    contextExpiresAt: "",
    localConnected: false,
    localPort: null,
    localNonce: "",
    localSessionId: "",
    localRequestId: "",
    localResult: null,
    localCandidateActive: false,
    localConfirmationVersion: 0,
    mediaRecorder: null,
    mediaStream: null,
    audioChunks: [],
    recordingStartedAt: 0,
    recordingTimer: null,
  };

  function clientSessionId() {
    let value = sessionStorage.getItem(CLIENT_SESSION_KEY);
    if (!value) {
      value = crypto.randomUUID ? crypto.randomUUID() : `vc-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      sessionStorage.setItem(CLIENT_SESSION_KEY, value);
    }
    return value;
  }

  async function apiJson(url, options = {}) {
    const response = await fetch(url, {
      credentials: "same-origin",
      headers: {"Content-Type": "application/json", ...(options.headers || {})},
      ...options,
    });
    let body = null;
    try { body = await response.json(); } catch (_) { body = null; }
    if (!response.ok) {
      const error = new Error(body?.detail || `HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return body;
  }

  function setChip(id, text, kind = "") {
    const node = $(id);
    node.textContent = text;
    node.className = `state-chip ${kind}`.trim();
  }

  function clearCandidateState(message) {
    clearTimeout(state.previewTimer);
    state.localConfirmationVersion++;
    state.candidate = null;
    state.preview = null;
    state.localResult = null;
    state.localCandidateActive = false;
    $("candidateInput").value = "";
    $("candidateInput").disabled = state.mode === "record" && !state.contextId;
    $("candidateMessage").textContent = message || (
      state.mode === "demo" ? "Μπορείς να ξεκινήσεις αμέσως, χωρίς ασθενή." : "Επιβεβαίωσε πρώτα τον ασθενή."
    );
    $("previewText").textContent = "Η προεπισκόπηση θα εμφανιστεί μόλις επικολλήσεις μία καταγραφή.";
    $("previewText").className = "preview-text empty";
    clearFormattedPreview();
    $("blockingNotice").hidden = true;
    $("saveBtn").disabled = true;
    $("saveMessage").textContent = "";
    $("localReviewItems").replaceChildren();
    $("localReviewPanel").hidden = true;
    $("localIdentityConfirmed").checked = false;
    $("localRisksConfirmed").checked = false;
    $("patientInformed").checked = false;
    $("gesyCopyRow").hidden = true;
    setChip("candidateState", state.mode === "demo" ? "Έτοιμο για δοκιμή" : (state.contextId ? "Έτοιμο για insert" : "Αναμονή"));
    setChip("previewState", "Δεν υπάρχει preview");
  }

  function resetPatientContext() {
    disconnectLocalCompanion("Η αλλαγή ασθενούς έκλεισε την τοπική συνεδρία.");
    state.contextId = "";
    state.contextExpiresAt = "";
    state.patientId = "";
    $("confirmedPatient").hidden = true;
    $("patientBox").hidden = false;
    clearCandidateState();
    updateLocalAudioControls();
  }

  function patientLabel(patient) {
    const d = patient.demographics || {};
    const first = d.first_name || d.firstName || d.given_name || d.firstname || d["όνομα"] || d["ονομα"];
    const last = d.last_name || d.lastName || d.family_name || d.surname || d.lastname || d["επώνυμο"] || d["επωνυμο"];
    const firstLast = [first, last].filter(Boolean).join(" ");
    const name = String(d.full_name || d.fullName || d.name || d["ονοματεπώνυμο"] || d["ονοματεπωνυμο"] || firstLast || "").trim();
    return (name ? name + " · " : "Ασθενής · ") + patient.patient_id;
  }

  function renderPatients() {
    const selected = $("patientSelect");
    const previous = selected.value;
    selected.replaceChildren();
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = state.patients.length ? "Επίλεξε ασθενή…" : "Δεν υπάρχουν αποτελέσματα";
    selected.appendChild(empty);
    state.patients.forEach((patient) => {
      const option = document.createElement("option");
      option.value = patient.patient_id;
      option.textContent = patientLabel(patient);
      selected.appendChild(option);
    });
    if (state.patients.some((patient) => patient.patient_id === previous)) selected.value = previous;
    $("confirmPatientBtn").disabled = !state.authenticated || state.patients.length === 0;
  }

  function clearPatientLookup() {
    clearTimeout(state.patientSearchTimer);
    state.patientSearchRevision++;
    state.patients = [];
    renderPatients();
    $("morePatientsBtn").hidden = true;
    $("morePatientsBtn").disabled = false;
  }

  function loadPatients() {
    // No default "latest 100" registry listing. Every search goes to the
    // protected backend against ALL registered patients and is paginated.
    if (state.suggestedPatientId && !$("patientSearch").value.trim()) {
      $("patientSearch").value = state.suggestedPatientId;
      state.suggestedPatientId = "";
    }
    schedulePatientSearch();
  }

  async function searchPatientPage(append = false) {
    const term = $("patientSearch").value.trim();
    if (!term || !state.authenticated || state.mode !== "record") return;
    const revision = ++state.patientSearchRevision;
    const offset = append ? state.patients.length : 0;
    $("morePatientsBtn").hidden = true;
    $("patientMessage").textContent = "Αναζήτηση στο πλήρες μητρώο…";
    try {
      const url = "/clinical/patients?query=" + encodeURIComponent(term)
        + "&limit=20&offset=" + offset;
      const patients = await apiJson(url, {method: "GET"});
      if (revision !== state.patientSearchRevision || state.mode !== "record"
          || !$("patientSearch").value.trim() || $("patientSearch").value.trim() !== term
          || !state.authenticated) return;
      state.patients = append ? state.patients.concat(patients) : patients;
      renderPatients();
      $("morePatientsBtn").hidden = patients.length < 20;
      if (state.patients.length === 0) {
        $("patientMessage").textContent = "Δεν βρέθηκε ασθενής με αυτά τα στοιχεία. Δοκίμασε άλλο όνομα ή αναγνωριστικό.";
      } else {
        $("patientMessage").textContent = "Βρέθηκαν " + state.patients.length
          + " αποτελέσματα μέχρι τώρα σε ολόκληρο το μητρώο."
          + (patients.length === 20 ? " Πάτησε «Περισσότερα» ή γράψε πιο συγκεκριμένα." : "");
      }
    } catch (error) {
      if (revision !== state.patientSearchRevision) return;
      if (!append) state.patients = [];
      renderPatients();
      $("morePatientsBtn").hidden = true;
      $("patientMessage").textContent = "Δεν ήταν δυνατή η αναζήτηση: " + error.message;
    }
  }

  function schedulePatientSearch() {
    clearPatientLookup();
    const term = $("patientSearch").value.trim();
    if (!state.authenticated) {
      $("patientMessage").textContent = "Συνδέσου πρώτα για να αναζητήσεις το μητρώο. Η δοκιμή δεν χρειάζεται σύνδεση.";
    } else if (state.mode !== "record" || !term) {
      $("patientMessage").textContent = "Πληκτρολόγησε όνομα ή αναγνωριστικό. Η αναζήτηση καλύπτει όλους τους καταχωρισμένους ασθενείς.";
    } else {
      $("patientMessage").textContent = "Αναζήτηση στο πλήρες μητρώο…";
      state.patientSearchTimer = setTimeout(() => searchPatientPage(false), 250);
    }
  }

  function selectMode(mode) {
    if (mode !== "demo" && mode !== "record") return;
    if (state.mode === mode) return;
    if (state.mode === "record" && mode !== "record" && state.contextId) {
      disconnectLocalCompanion("Η αλλαγή λειτουργίας έκλεισε την τοπική συνεδρία.");
    }
    state.mode = mode;
    state.contextId = "";
    state.contextExpiresAt = "";
    state.patientId = "";
    clearPatientLookup();
    $("demoModeBtn").classList.toggle("active", mode === "demo");
    $("recordModeBtn").classList.toggle("active", mode === "record");
    $("demoModeBtn").setAttribute("aria-pressed", String(mode === "demo"));
    $("recordModeBtn").setAttribute("aria-pressed", String(mode === "record"));
    $("patientPanel").hidden = mode !== "record";
    $("localAudioPanel").hidden = mode !== "record";
    $("saveRow").hidden = mode !== "record";
    $("saveMessage").hidden = mode !== "record";
    $("copyDiaPromptBtn").hidden = mode !== "demo";
    $("exampleBtn").hidden = mode !== "demo";
    $("copyStatus").textContent = "";
    $("confirmedPatient").hidden = true;
    $("patientBox").hidden = false;
    $("modeNotice").textContent = mode === "demo"
      ? "Δοκιμή μόνο στην οθόνη, χωρίς patient ID ή αποθήκευση. Επικόλλησε σύνοψη Dia ή χρησιμοποίησε το συνθετικό παράδειγμα."
      : "Για τελική αποθήκευση επιβεβαίωσε υπάρχοντα ασθενή και επικόλλησε το δομημένο VisitCaptureCandidateV1 JSON. Η ελεύθερη σύνοψη Dia είναι μόνο για δοκιμή.";
    $("candidateHint").textContent = mode === "demo"
      ? "Επικόλλησε τη σύνοψη του Dia. Τρεις προβολές χωρίς εγγραφή σε ασθενή."
      : "Χρησιμοποίησε δομημένο VisitCaptureCandidateV1 JSON. Δεν αποθηκεύεται πριν πατήσεις «Αποθήκευση».";
    $("candidateLabel").textContent = mode === "demo" ? "Σύνοψη Dia (απλό κείμενο)" : "Δομημένο VisitCaptureCandidateV1 JSON";
    $("candidateInput").placeholder = mode === "demo"
      ? "Επικόλλησε το κείμενο με τις ενότητες SNAPSHOT, VISIT BRIEF, ENCOUNTER DETAIL."
      : '{"schema_version":"visit_capture_candidate_v1", ...}';
    $("previewHint").textContent = mode === "demo"
      ? "Προεπισκόπηση μόνο στην οθόνη, χωρίς αποθήκευση ή ασθενή."
      : "Οι τρεις προβολές παράγονται από το ίδιο δομημένο encounter στον προστατευμένο server.";
    clearCandidateState();
    updateLocalAudioControls();
    if (mode === "record") loadPatients();
  }

  async function checkAuth(initialLoad = false) {
    try {
      await apiJson("/clinical/status", {method: "GET"});
      state.authenticated = true;
      $("authToggle").hidden = false;
      $("authToggle").setAttribute("aria-expanded", "false");
      $("authCard").hidden = true;
      $("loginBox").hidden = true;
      setChip("authState", "Authenticated", "ok");
      const fromQuery = new URLSearchParams(location.search).get("patient_id") || "";
      if (fromQuery && state.mode === "demo") {
        state.suggestedPatientId = fromQuery;
        // Keep unsaved Dia text untouched on Clinical Key refresh.
        if (initialLoad && !$("candidateInput").value && !state.preview && !state.candidate) {
          selectMode("record");
        }
      } else if (state.mode === "record") {
        loadPatients();
      }
    } catch (error) {
      state.authenticated = false;
      $("authToggle").hidden = true;
      $("authCard").hidden = false;
      $("loginBox").hidden = false;
      setChip("authState", error.status === 503 ? "Clinical access disabled" : "Απαιτείται σύνδεση", "err");
      if (state.mode === "record") {
        resetPatientContext();
        loadPatients();
      }
    }
  }

  async function login() {
    const key = $("clinicalKey").value;
    if (!key) return;
    try {
      await apiJson("/clinical/login", {method: "POST", body: JSON.stringify({key})});
      $("clinicalKey").value = "";
      await checkAuth();
    } catch (error) {
      setChip("authState", error.message, "err");
    }
  }

  async function confirmPatient() {
    if (state.mode !== "record" || !state.authenticated) return;
    const patientId = $("patientSelect").value.trim();
    if (!patientId) {
      $("patientMessage").textContent = "Επίλεξε ασθενή από τη λίστα.";
      return;
    }
    try {
      const context = await apiJson("/clinical/visit-capture/context", {
        method: "POST",
        body: JSON.stringify({
          patient_id: patientId,
          client_session_id: clientSessionId(),
        }),
      });
      state.contextId = context.context_id;
      state.contextExpiresAt = context.expires_at || "";
      state.patientId = context.patient_id;
      const patient = state.patients.find((item) => item.patient_id === context.patient_id);
      $("confirmedPatientId").textContent = patient ? patientLabel(patient) : context.patient_id;
      $("confirmedPatient").hidden = false;
      $("patientBox").hidden = true;
      clearCandidateState("Η καταγραφή παραμένει προσωρινή μέχρι το ρητό Save.");
      $("patientMessage").textContent = "";
      $("candidateInput").focus();
      updateLocalAudioControls();
    } catch (error) {
      resetPatientContext();
      $("patientMessage").textContent = error.message;
    }
  }

  // Formatting is entirely presentational: source text is preserved and
  // nothing here infers a clinical fact, validates a diagnosis, or calls Save.
  function foldHeading(value) {
    return String(value || "").trim()
      .replace(/^#{1,6}\s*/, "")
      .replace(/^\*+|\*+$/g, "")
      .replace(/:$/, "").trim()
      .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
      .toLocaleLowerCase("el");
  }

  function isReviewHeading(value) {
    return /^σημεια προς (?:ελεγχο|επιβεβαιωση)(?:$|\s|\()/.test(foldHeading(value));
  }

  function readExplicitReviewNotes(text) {
    const notes = [];
    const seen = new Set();
    let collecting = false;
    let pending = "";
    function emit() {
      const note = pending.trim();
      if (note && !seen.has(note)) { seen.add(note); notes.push(note); }
      pending = "";
    }
    String(text || "").replace(/\r\n?/g, "\n").split("\n").forEach((line) => {
      const heading = foldHeading(line);
      if (isReviewHeading(line)) { emit(); collecting = true; return; }
      // Reuse the same explicit section grammar as the structured renderer:
      // any subsequent recognized clinical heading ends review-note collection.
      if (/^(snapshot|visit brief|encounter detail)$/.test(heading) ||
          Object.keys(sectionNames).some((level) => sourceSectionHeading(line, level))) {
        if (collecting) emit();
        collecting = false;
        return;
      }
      if (!collecting) return;
      const item = line.trim().match(/^(?:\d{1,3}[.)]|[-•])\s+(.+)/);
      if (item) {
        emit();
        pending = item[1];
      } else if (!line.trim()) {
        emit();
      } else if (pending) {
        pending += " " + line.trim();
      }
    });
    emit();
    return notes;
  }

  // Split only on explicit, source-supplied section headings. Other text stays
  // intact; unrecognized content is never silently converted into a diagnosis.
  const sectionNames = {
    snapshot: ["κλινικη εικονα", "μεταβολη", "αποφαση", "εκκρεμοτητες"],
    brief: ["λογος επισκεψης", "ιστορικο και μεταβολες", "ιστορικο",
      "μεταβολες", "ευρηματα", "κλινικη εκτιμηση", "αποφασεις",
      "φαρμακα", "εκκρεμοτητες", "εκκρεμοτητες και παρακολουθηση",
      "παρακολουθηση", "πλανο"],
    detail: ["πηγες", "πηγη", "ιστορικο", "ευρηματα", "κλινικη εκτιμηση",
      "εξετασεις και κατασταση", "εξετασεις", "φαρμακευτικη αγωγη",
      "φαρμακα", "αποφασεις και παρακολουθηση", "αποφασεις",
      "παρακολουθηση", "κωδικοποιηση / διοικητικα", "κωδικοποιηση",
      "διοικητικα", "εκκρεμοτητες", "διαγνωσεις", "σημεια προς επιβεβαιωση",
      "σημεια προς ελεγχο"]
  };

  function sourceSectionHeading(line, level) {
    const value = line.trim().replace(/^#{1,6}\s*/, "").replace(/^\*+|\*+$/g, "");
    const match = value.match(/^([^:：]{2,100})\s*[:：]\s*(.*)$/);
    if (match) {
      const name = foldHeading(match[1]);
      if (sectionNames[level].includes(name) || (level === "detail" && isReviewHeading(name))) {
        return {title: match[1].trim(), after: match[2].trim()};
      }
    }
    if (sectionNames[level].includes(foldHeading(value)) ||
        (level === "detail" && isReviewHeading(value))) {
      return {title: value.replace(/:$/, "").trim(), after: ""};
    }
    if (level === "detail") {
      const source = value.match(/^(Πηγή\s+[Α-ΩA-Za-z0-9]+)\s*[—–-]\s*(.*)$/i);
      if (source) return {title: source[1].trim(), after: source[2].trim()};
    }
    return null;
  }

  function divideExplicitSections(text, level) {
    const sections = [];
    let current = {title: "", lines: []};
    String(text || "").replace(/\r\n?/g, "\n").split("\n").forEach((line) => {
      const heading = sourceSectionHeading(line, level);
      if (heading) {
        if (current.lines.join("\n").trim() || current.title) sections.push(current);
        current = {title: heading.title, lines: heading.after ? [heading.after] : []};
      } else {
        current.lines.push(line);
      }
    });
    if (current.lines.join("\n").trim() || current.title) sections.push(current);
    return sections.map((part) => ({
      title: part.title,
      body: part.lines.join("\n").trim(),
    }));
  }

  function clearFormattedPreview() {
    $("structuredPreview").replaceChildren();
    $("structuredPreview").hidden = true;
    $("previewText").hidden = false;
    $("diaReviewList").replaceChildren();
    $("diaReviewPanel").hidden = true;
    $("diaReviewDetails").open = false;
  }

  function renderExplicitReviewNotes(preview) {
    const items = state.mode === "demo" && Array.isArray(preview.reviewItems) ? preview.reviewItems : [];
    $("diaReviewPanel").hidden = items.length === 0;
    const list = $("diaReviewList");
    const wasOpen = $("diaReviewDetails").open;
    list.replaceChildren();
    if (!items.length) { $("diaReviewDetails").open = false; return; }
    $("diaReviewCount").textContent = items.length + (items.length === 1 ?
      " σημείο προς επιβεβαίωση" : " σημεία προς επιβεβαίωση");
    items.forEach((item) => {
      const line = document.createElement("li");
      line.textContent = item;
      list.appendChild(line);
    });
    $("diaReviewDetails").open = wasOpen;
  }

  function renderStructuredPreview(preview) {
    const host = $("structuredPreview");
    host.replaceChildren();
    host.hidden = true;
    $("previewText").hidden = false;
    if (state.mode !== "demo") return;
    const sections = divideExplicitSections(preview[state.level] || "", state.level);
    const headers = sections.filter((section) => Boolean(section.title));
    if (headers.length < 2) return;  // Old freeform notes stay readable verbatim.
    sections.forEach((part, index) => {
      const article = document.createElement(state.level === "detail" ? "details" : "section");
      article.className = state.level === "detail" ? "detail-section" : "brief-section";
      if (state.level === "detail") {
        const heading = document.createElement("summary");
        heading.textContent = part.title || "Εισαγωγή";
        article.open = index === 0;
        article.appendChild(heading);
      } else {
        const heading = document.createElement("h3");
        heading.textContent = part.title || "Εισαγωγή";
        article.appendChild(heading);
      }
      const body = document.createElement("div");
      body.className = "clinical-section-body";
      body.textContent = part.body || "Δεν έχει δοθεί περιεχόμενο για αυτή την ενότητα.";
      article.appendChild(body);
      host.appendChild(article);
    });
    host.hidden = false;
    $("previewText").hidden = true;
  }

  function parseDiaSummary(input) {
    const parts = {snapshot: [], brief: [], detail: []};
    const keys = {SNAPSHOT: "snapshot", "VISIT BRIEF": "brief", "ENCOUNTER DETAIL": "detail"};
    let current = null;
    const preface = [];
    const seen = new Set();
    input.replace(/\r\n?/g, "\n").split("\n").forEach((line) => {
      const label = line.trim()
        .replace(/^#{1,6}\s*/, "")
        .replace(/^\*+|\*+$/g, "")
        .replace(/^=+\s*|\s*=+$/g, "")
        .replace(/:$/, "").trim().toUpperCase();
      if (Object.prototype.hasOwnProperty.call(keys, label)) {
        current = keys[label];
        seen.add(current);
      } else if (current) {
        parts[current].push(line);
      } else {
        preface.push(line);
      }
    });
    if (seen.size === 0) {
      return {
        snapshot: "Δεν δόθηκε ξεχωριστό Snapshot από το Dia.",
        brief: input.trim(),
        detail: "Δεν δόθηκε ξεχωριστό Encounter Detail από το Dia.",
        reviewItems: readExplicitReviewNotes(input),
        complete: false,
      };
    }
    if (preface.join("\n").trim()) parts.brief.unshift(preface.join("\n").trim());
    const missing = "Δεν δόθηκε αυτή η ενότητα από το Dia. Έλεγξε το prompt και την απάντηση.";
    return {
      snapshot: parts.snapshot.join("\n").trim() || missing,
      brief: parts.brief.join("\n").trim() || missing,
      detail: parts.detail.join("\n").trim() || missing,
      reviewItems: readExplicitReviewNotes(input),
      complete: ["snapshot", "brief", "detail"].every((key) => parts[key].join("\n").trim()),
    };
  }

  async function copyDiaPrompt() {
    const prompt = $("diaPromptTemplate").content.textContent.trim();
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(prompt);
      } else {
        const temporary = document.createElement("textarea");
        temporary.value = prompt;
        document.body.appendChild(temporary);
        temporary.select();
        const copied = document.execCommand("copy");
        temporary.remove();
        if (!copied) throw new Error("Η αντιγραφή δεν υποστηρίζεται.");
      }
      $("copyStatus").textContent = "Το prompt αντιγράφηκε. Επικόλλησέ το στο Dia.";
    } catch (_) {
      $("copyStatus").textContent = "Δεν επιτράπηκε η αντιγραφή. Δοκίμασε από ασφαλή σύνδεση HTTPS.";
    }
  }

  function loadExample() {
    if (state.mode !== "demo") return;
    $("candidateInput").value = [
      "SNAPSHOT",
      "Συνθετικό περιστατικό: αυχεναλγία μετά από άσκηση. Ήπιος περιορισμός κίνησης. Συμφωνήθηκε επανεκτίμηση.",
      "",
      "VISIT BRIEF",
      "Συνθετικό παράδειγμα: ενήλικος με αυχεναλγία μετά από άσκηση. Περιγράφεται ενόχληση στη στροφή της κεφαλής. Δεν παρέχονται στοιχεία προηγούμενης επίσκεψης για σύγκριση. Προτάθηκε σταδιακή κινητοποίηση σύμφωνα με την ανοχή και επανεκτίμηση.",
      "",
      "ENCOUNTER DETAIL",
      "Λόγος επίσκεψης: αυχεναλγία μετά από άσκηση. Ευρήματα: ήπιος περιορισμός ενεργητικής στροφής όπως αναφέρεται στο συνθετικό υλικό. Απόφαση: σταδιακή κινητοποίηση, παρακολούθηση και επανεκτίμηση. Φάρμακα/κωδικοποίηση: δεν δόθηκαν στοιχεία. Εκκρεμεί: επανεκτίμηση.",
    ].join("\n");
    schedulePreview();
  }

  function exactKeys(value, keys) {
    return value && typeof value === "object" && !Array.isArray(value)
      && Object.keys(value).sort().join("|") === [...keys].sort().join("|");
  }

  function randomBase64Url(bytes) {
    const data = new Uint8Array(bytes);
    crypto.getRandomValues(data);
    let binary = "";
    data.forEach((value) => { binary += String.fromCharCode(value); });
    return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function localEnvelope(type, requestId, payload) {
    const message = {protocol: LOCAL_PROTOCOL, type, nonce: state.localNonce,
      local_session_id: state.localSessionId, payload};
    if (requestId) message.request_id = requestId;
    return message;
  }

  function updateLocalAudioControls() {
    const available = state.mode === "record" && state.authenticated && Boolean(state.contextId)
      && state.localConnected && !state.mediaRecorder;
    $("startRecordingBtn").disabled = !available || !$("patientInformed").checked;
    $("stopRecordingBtn").disabled = !state.mediaRecorder;
    if (!state.localConnected) {
      setChip("localAudioState", "Δεν συνδέθηκε", "err");
    } else if (state.mediaRecorder) {
      setChip("localAudioState", "Ηχογράφηση", "err");
    } else if (!state.contextId) {
      setChip("localAudioState", "Αναμονή ασθενούς");
    } else {
      setChip("localAudioState", "Τοπικά έτοιμο", "ok");
    }
  }

  function releaseMedia() {
    clearInterval(state.recordingTimer);
    state.recordingTimer = null;
    if (state.mediaStream) state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null;
    state.mediaRecorder = null;
    state.audioChunks = [];
    state.recordingStartedAt = 0;
    $("localAudioPanel").classList.remove("recording");
    $("recordingClock").textContent = "00:00";
    updateLocalAudioControls();
  }

  function cancelLocalRecording(message) {
    if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
      state.mediaRecorder.onstop = null;
      state.mediaRecorder.stop();
    }
    if (state.localPort && state.localRequestId) {
      state.localPort.postMessage(localEnvelope("job_cancel", state.localRequestId, {}));
    }
    state.localRequestId = "";
    releaseMedia();
    if (message) $("localAudioMessage").textContent = message;
  }

  function validLocalResult(result) {
    if (!exactKeys(result, ["schema_version", "protocol", "local_session_id", "request_id", "source_ref", "captured_at", "duration_ms", "items"])) return false;
    if (result.schema_version !== "local_visit_audio_result_v1" || result.protocol !== LOCAL_PROTOCOL
        || result.local_session_id !== state.localSessionId || result.request_id !== state.localRequestId
        || result.source_ref !== "local-audio:" + state.localRequestId
        || typeof result.captured_at !== "string" || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z$/.test(result.captured_at)
        || Number.isNaN(Date.parse(result.captured_at))
        || !Number.isInteger(result.duration_ms) || result.duration_ms < 1 || result.duration_ms > 1800000
        || !Array.isArray(result.items) || result.items.length < 1 || result.items.length > 512
        || new TextEncoder().encode(JSON.stringify(result)).byteLength > 1048576) return false;
    return result.items.every((item, index) => exactKeys(item, ["id", "text", "proposed_dispositions", "risk_flags", "whisper_flags"])
      && item.id === index && typeof item.text === "string" && item.text.length >= 1 && item.text.length <= 4000
      && new TextEncoder().encode(item.text).byteLength <= 16384
      && Array.isArray(item.proposed_dispositions) && item.proposed_dispositions.length > 0
      && item.proposed_dispositions.every((value) => LOCAL_DISPOSITIONS.includes(value))
      && new Set(item.proposed_dispositions).size === item.proposed_dispositions.length
      && Array.isArray(item.risk_flags) && item.risk_flags.every((value) => LOCAL_RISK_FLAGS.has(value))
      && new Set(item.risk_flags).size === item.risk_flags.length
      && Array.isArray(item.whisper_flags) && item.whisper_flags.every((value) => LOCAL_WHISPER_FLAGS.has(value))
      && new Set(item.whisper_flags).size === item.whisper_flags.length);
  }

  function disconnectLocalCompanion(message) {
    cancelLocalRecording();
    if (state.localPort) state.localPort.close();
    state.localPort = null;
    state.localConnected = false;
    state.localNonce = "";
    state.localSessionId = "";
    if (message) $("localAudioMessage").textContent = message;
    updateLocalAudioControls();
  }

  function localPortMessage(event) {
    const message = event.data;
    if (!message || message.protocol !== LOCAL_PROTOCOL || message.nonce !== state.localNonce
        || message.local_session_id !== state.localSessionId
        || (event.ports && event.ports.length !== 0)) {
      disconnectLocalCompanion("Απορρίφθηκε μη έγκυρο τοπικό μήνυμα."); return;
    }
    if (message.type === "companion_ready" && !state.localConnected) {
      if (!exactKeys(message, ["protocol", "type", "nonce", "local_session_id", "payload"])
          || !exactKeys(message.payload, [])) {
        disconnectLocalCompanion("Απορρίφθηκε μη έγκυρη σύνδεση."); return;
      }
      state.localConnected = true;
      $("localAudioMessage").textContent = "Το Visit Capture συνδέθηκε με τον τοπικό companion.";
      updateLocalAudioControls();
      return;
    }
    if (message.request_id !== state.localRequestId) {
      disconnectLocalCompanion("Απορρίφθηκε παλιό τοπικό αποτέλεσμα."); return;
    }
    if (message.type === "job_progress") {
      const labels = {transferring: "Μεταφορά…", decoding: "Αποκωδικοποίηση…", transcribing: "Τοπική μεταγραφή…", classifying: "Τοπική ταξινόμηση…", validating: "Έλεγχος…", cleaning: "Ασφαλής διαγραφή…"};
      if (!exactKeys(message, ["protocol", "type", "nonce", "local_session_id", "request_id", "payload"])
          || !exactKeys(message.payload, ["stage"]) || !labels[message.payload.stage]) {
        disconnectLocalCompanion("Απορρίφθηκε μη έγκυρη πρόοδος."); return;
      }
      $("localAudioMessage").textContent = labels[message.payload.stage];
    } else if (message.type === "job_error") {
      if (!exactKeys(message, ["protocol", "type", "nonce", "local_session_id", "request_id", "payload"])
          || !exactKeys(message.payload, ["code", "request_id"])
          || message.payload.request_id !== state.localRequestId
          || !LOCAL_ERROR_CODES.has(message.payload.code)) {
        disconnectLocalCompanion("Απορρίφθηκε μη έγκυρο τοπικό σφάλμα."); return;
      }
      $("localAudioMessage").textContent = "Η τοπική επεξεργασία απέτυχε: " + message.payload.code;
      state.localPort.postMessage(localEnvelope("job_ack", state.localRequestId, {}));
      state.localRequestId = "";
      state.localPort.close();
      state.localPort = null;
      state.localConnected = false;
      updateLocalAudioControls();
    } else if (message.type === "job_result") {
      if (!exactKeys(message, ["protocol", "type", "nonce", "local_session_id", "request_id", "payload"])
          || !exactKeys(message.payload, ["result"]) || !validLocalResult(message.payload.result)) {
        disconnectLocalCompanion("Το τοπικό αποτέλεσμα απέτυχε στον έλεγχο."); return;
      }
      state.localResult = message.payload.result;
      renderLocalReview();
      state.localPort.postMessage(localEnvelope("job_ack", state.localRequestId, {}));
      state.localRequestId = "";
      state.localPort.close();
      state.localPort = null;
      state.localConnected = false;
      $("localAudioMessage").textContent = "Ολοκληρώθηκε τοπικά. Έλεγξε το προσχέδιο και επιβεβαίωσε τον ασθενή.";
      updateLocalAudioControls();
      setChip("localAudioState", "Τοπικά ολοκληρώθηκε", "ok");
    } else {
      disconnectLocalCompanion("Απορρίφθηκε μήνυμα εκτός σειράς.");
    }
  }

  function initializeLocalCompanion() {
    if (typeof location.hash !== "string" || !location.hash.startsWith("#local_visit_audio_v1=")) return;
    const nonce = location.hash.slice("#local_visit_audio_v1=".length);
    history.replaceState(null, "", location.pathname + location.search);
    if (!/^[A-Za-z0-9_-]{43}$/.test(nonce) || !window.opener) return;
    state.localNonce = nonce;
    const opener = window.opener;
    const listener = (event) => {
      const message = event.data;
      if (event.origin !== LOCAL_ORIGIN || event.source !== opener
          || !exactKeys(message, ["protocol", "type", "nonce", "local_session_id"])
          || message.protocol !== LOCAL_PROTOCOL || message.type !== "bootstrap_port"
          || message.nonce !== nonce || !/^[A-Za-z0-9_-]{22}$/.test(message.local_session_id)
          || event.ports.length !== 1) {
        (event.ports || []).forEach((candidate) => candidate.close());
        window.removeEventListener("message", listener);
        window.opener = null;
        disconnectLocalCompanion("Απορρίφθηκε μη έγκυρη τοπική σύνδεση.");
        return;
      }
      window.removeEventListener("message", listener);
      state.localSessionId = message.local_session_id;
      state.localPort = event.ports[0];
      state.localPort.onmessage = localPortMessage;
      state.localPort.onmessageerror = () => disconnectLocalCompanion("Το ασφαλές τοπικό κανάλι έκλεισε.");
      state.localPort.start();
      window.opener = null;
    };
    window.addEventListener("message", listener);
    opener.postMessage({protocol: LOCAL_PROTOCOL, type: "bootstrap_ready", nonce}, LOCAL_ORIGIN);
    setTimeout(() => {
      window.removeEventListener("message", listener);
      if (!state.localPort) disconnectLocalCompanion("Η τοπική σύνδεση έληξε.");
    }, 10000);
  }

  async function startLocalRecording() {
    if ($("startRecordingBtn").disabled || !state.contextId || !state.patientId) return;
    if (!window.MediaRecorder || !MediaRecorder.isTypeSupported(LOCAL_MIME)) {
      $("localAudioMessage").textContent = "Ο browser δεν υποστηρίζει το ασφαλές format ήχου."; return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true});
      if (!state.contextId || !$("patientInformed").checked) {
        stream.getTracks().forEach((track) => track.stop()); return;
      }
      state.mediaStream = stream;
      state.audioChunks = [];
      const recorder = new MediaRecorder(stream, {mimeType: LOCAL_MIME});
      state.mediaRecorder = recorder;
      recorder.ondataavailable = (event) => { if (event.data?.size) state.audioChunks.push(event.data); };
      recorder.onerror = () => cancelLocalRecording("Σφάλμα μικροφώνου· η καταγραφή απορρίφθηκε.");
      recorder.onstop = finishLocalRecording;
      state.recordingStartedAt = Date.now();
      recorder.start(1000);
      $("localAudioPanel").classList.add("recording");
      $("localAudioMessage").textContent = "Ηχογράφηση μόνο σε αυτόν τον Mac.";
      state.recordingTimer = setInterval(() => {
        const elapsed = Math.min(1800, Math.floor((Date.now() - state.recordingStartedAt) / 1000));
        $("recordingClock").textContent = String(Math.floor(elapsed / 60)).padStart(2, "0") + ":" + String(elapsed % 60).padStart(2, "0");
        if (elapsed >= 1800) stopLocalRecording();
      }, 1000);
      updateLocalAudioControls();
    } catch (_) {
      releaseMedia();
      $("localAudioMessage").textContent = "Δεν δόθηκε πρόσβαση στο μικρόφωνο.";
    }
  }

  function stopLocalRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") state.mediaRecorder.stop();
  }

  async function finishLocalRecording() {
    const durationMs = Math.min(1800000, Math.max(1, Date.now() - state.recordingStartedAt));
    const blob = new Blob(state.audioChunks, {type: LOCAL_MIME});
    const patientId = state.patientId;
    const contextId = state.contextId;
    const localSessionId = state.localSessionId;
    releaseMedia();
    if (!state.localPort || !state.localConnected || patientId !== state.patientId || blob.size < 1 || blob.size > 67108864) {
      $("localAudioMessage").textContent = "Η καταγραφή απορρίφθηκε πριν από την τοπική επεξεργασία."; return;
    }
    const audio = await blob.arrayBuffer();
    if (!state.localPort || !state.localConnected || patientId !== state.patientId
        || contextId !== state.contextId || localSessionId !== state.localSessionId) {
      $("localAudioMessage").textContent = "Η ταυτότητα άλλαξε· η καταγραφή απορρίφθηκε."; return;
    }
    state.localRequestId = randomBase64Url(16);
    $("localAudioMessage").textContent = "Μεταφορά μόνο στον τοπικό companion…";
    state.localPort.postMessage(localEnvelope("job_start", state.localRequestId, {
      mime_type: LOCAL_MIME, duration_ms: durationMs, byte_length: audio.byteLength, audio,
    }), [audio]);
  }

  function createOption(value, label, selected) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    option.selected = selected;
    return option;
  }

  function renderLocalReview() {
    const host = $("localReviewItems");
    host.replaceChildren();
    const risky = state.localResult.items.filter((item) => item.risk_flags.length || item.whisper_flags.length);
    $("localRiskPanel").hidden = risky.length === 0;
    $("localRiskMessage").textContent = risky.length ? risky.length + " στοιχεία έχουν αριθμό, φάρμακο, πλευρά, άρνηση, αβεβαιότητα ή σύγκρουση." : "";
    state.localResult.items.forEach((item) => {
      const row = document.createElement("article");
      row.className = "local-review-item" + ((item.risk_flags.length || item.whisper_flags.length) ? " risk" : "");
      row.dataset.itemId = String(item.id);
      const text = document.createElement("textarea");
      text.className = "local-item-text";
      text.value = item.text;
      text.dataset.originalText = item.text;
      text.maxLength = 4000;
      const controls = document.createElement("div");
      controls.className = "local-review-controls";
      const includeLabel = document.createElement("label");
      const include = document.createElement("input");
      include.type = "checkbox";
      include.checked = true;
      include.className = "local-include-check";
      includeLabel.append(include, document.createTextNode(" Συμπερίληψη"));
      controls.appendChild(includeLabel);
      const disposition = document.createElement("select");
      disposition.className = "local-disposition";
      const primaryProposals = item.proposed_dispositions.filter((value) => value !== "pending_obligation");
      const conflict = primaryProposals.length !== 1
        || item.risk_flags.includes("semantic_status_conflict");
      disposition.appendChild(createOption("", "Επίλεξε κατηγορία…", conflict));
      const proposed = primaryProposals[0] || "";
      LOCAL_DISPOSITIONS.filter((value) => value !== "pending_obligation")
        .forEach((value) => disposition.appendChild(createOption(value, LOCAL_LABELS[value], !conflict && value === proposed)));
      controls.appendChild(disposition);
      const type = document.createElement("select");
      type.className = "local-decision-type";
      type.appendChild(createOption("", "Επίλεξε κλινικό τύπο…", true));
      [["therapy", "Θεραπεία"], ["referral", "Παραπομπή"], ["medication", "Φάρμακο"], ["investigation", "Εξέταση"], ["advice", "Συμβουλή"], ["sick_leave", "Άδεια"], ["admin", "Διοικητικό"]].forEach(([value, label]) => type.appendChild(createOption(value, label, false)));
      type.hidden = !["clinician_recommendation", "final_decision"].includes(disposition.value);
      controls.appendChild(type);
      disposition.addEventListener("change", () => { type.hidden = !["clinician_recommendation", "final_decision"].includes(disposition.value); });
      const pendingLabel = document.createElement("label");
      const pending = document.createElement("input");
      pending.type = "checkbox";
      pending.className = "local-pending-check";
      pendingLabel.append(pending, document.createTextNode(" Δημιουργία εκκρεμότητας"));
      controls.appendChild(pendingLabel);
      const pendingType = document.createElement("select");
      pendingType.className = "local-pending-type";
      pendingType.hidden = true;
      pendingType.appendChild(createOption("", "Τύπος εκκρεμότητας…", true));
      [["test_or_investigation", "Εξέταση"], ["referral", "Παραπομπή"], ["follow_up", "Παρακολούθηση"], ["administrative", "Διοικητικό"], ["other", "Άλλο"]].forEach(([value, label]) => pendingType.appendChild(createOption(value, label, false)));
      pending.addEventListener("change", () => { pendingType.hidden = !pending.checked; });
      controls.appendChild(pendingType);
      const badges = document.createElement("div");
      badges.className = "local-badges";
      [...item.risk_flags, ...item.whisper_flags].forEach((flag) => {
        const badge = document.createElement("span"); badge.className = "local-badge"; badge.textContent = flag; badges.appendChild(badge);
      });
      if (item.proposed_dispositions.includes("pending_obligation")) {
        const badge = document.createElement("span");
        badge.className = "local-badge";
        badge.textContent = "πρόταση: πιθανή εκκρεμότητα";
        badges.appendChild(badge);
      }
      controls.appendChild(badges);
      row.append(text, controls);
      host.appendChild(row);
    });
    $("localReviewPanel").hidden = false;
    $("localIdentityConfirmed").checked = false;
    $("localRisksConfirmed").checked = risky.length === 0;
    $("prepareLocalCandidateBtn").disabled = true;
  }

  function fact(text, provenance) {
    return {text, provenance: [provenance], verification: provenance === "clinician-edit" ? "clinician_confirmed" : "source_stated", certainty: "certain"};
  }

  function cyprusDate() {
    const parts = new Intl.DateTimeFormat("en-CA", {timeZone: "Asia/Nicosia", year: "numeric", month: "2-digit", day: "2-digit"}).formatToParts(new Date());
    const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
    return values.year + "-" + values.month + "-" + values.day;
  }

  function buildLocalCandidate() {
    if (!state.localResult || !$("localIdentityConfirmed").checked) throw new Error("Επιβεβαίωσε ότι η καταγραφή ανήκει στον επιλεγμένο ασθενή.");
    const sourceRef = state.localResult.source_ref;
    const groups = {reason: [], history: [], findings: [], clinical_impression: [], options_discussed: [], recommendations: [], completed_actions: [], patient_preferences: [], decisions: [], follow_up: [], safety_net: [], uncertainties: [], administrative_notes: [], pending: []};
    let edited = false;
    Array.from($("localReviewItems").children).forEach((row) => {
      if (!row.querySelector(".local-include-check").checked) return;
      const textNode = row.querySelector(".local-item-text");
      const text = textNode.value.trim();
      const disposition = row.querySelector(".local-disposition").value;
      const decisionType = row.querySelector(".local-decision-type").value;
      if (!text || !disposition) throw new Error("Κάθε στοιχείο χρειάζεται κείμενο και κατηγορία.");
      if (["clinician_recommendation", "final_decision"].includes(disposition) && !decisionType) throw new Error("Επίλεξε κλινικό τύπο για κάθε σύσταση ή τελική απόφαση.");
      const changed = text !== textNode.dataset.originalText;
      edited = edited || changed;
      const provenance = changed ? "clinician-edit" : sourceRef;
      const base = fact(text, provenance);
      if (disposition === "reason") groups.reason.push(base);
      else if (disposition === "patient_history") groups.history.push(base);
      else if (disposition === "objective_result") groups.findings.push({...base, region: null});
      else if (disposition === "clinician_interpretation") groups.clinical_impression.push(base);
      else if (disposition === "option_discussed") groups.options_discussed.push(base);
      else if (disposition === "clinician_recommendation") groups.recommendations.push({...base, type: decisionType});
      else if (disposition === "final_decision") groups.decisions.push({...base, type: decisionType, responsible_role: null});
      else if (disposition === "completed_action") groups.completed_actions.push(base);
      else if (disposition.startsWith("patient_")) groups.patient_preferences.push({...base, status: disposition.slice("patient_".length)});
      else if (disposition === "follow_up") groups.follow_up.push(base);
      else if (disposition === "safety_net") groups.safety_net.push(base);
      else if (["uncertainty", "unclassified"].includes(disposition)) groups.uncertainties.push({...base, certainty: "uncertain"});
      else if (disposition === "administrative") groups.administrative_notes.push(base);
      const pending = row.querySelector(".local-pending-check");
      if (pending.checked) {
        const itemType = row.querySelector(".local-pending-type").value;
        if (!itemType) throw new Error("Επίλεξε τύπο για κάθε εκκρεμότητα.");
        groups.pending.push({source_excerpt: text, item_type: itemType, description: text, trigger: null, status: "open", responsible_role: "unspecified", external_dependency: null, review_date: null, provenance: [provenance]});
      }
    });
    if (!groups.reason.length) throw new Error("Επίλεξε τουλάχιστον έναν λόγο επίσκεψης.");
    const bindings = [{source_ref: sourceRef, role: "local_visit_audio", captured_at: state.localResult.captured_at, source_date: cyprusDate()}];
    if (edited) bindings.push({source_ref: "clinician-edit", role: "clinician_edit", captured_at: new Date().toISOString(), source_date: cyprusDate()});
    return {schema_version: "visit_capture_candidate_v1", source_identity_state: "consistent",
      encounter_date: cyprusDate(), visit_type: null, specialty: "Ορθοπαιδική",
      reason_for_visit: fact(groups.reason.map((item) => item.text).join("\n"), groups.reason.some((item) => item.provenance[0] === "clinician-edit") ? "clinician-edit" : sourceRef),
      source_bindings: bindings, comparison_basis: {type: "unavailable", source_ref: null}, what_changed: [],
      history: groups.history, findings: groups.findings, clinical_impression: groups.clinical_impression,
      options_discussed: groups.options_discussed, recommendations: groups.recommendations,
      completed_actions: groups.completed_actions, patient_preferences: groups.patient_preferences,
      coding: {gesy_coded: [], coding_complete: false}, decisions: groups.decisions, medications: [],
      pending: groups.pending, follow_up: groups.follow_up, next_contact: null,
      safety_net: groups.safety_net, administrative_notes: groups.administrative_notes,
      uncertainties: groups.uncertainties};
  }

  async function renewLocalContext() {
    const identity = {patientId: state.patientId, contextId: state.contextId, localSessionId: state.localSessionId, requestId: state.localResult?.request_id || ""};
    if (!identity.patientId || !identity.contextId || !identity.localSessionId || !identity.requestId || !$("localIdentityConfirmed").checked) throw new Error("Η ταυτότητα ή η τοπική συνεδρία άλλαξε.");
    const renewed = await apiJson("/clinical/visit-capture/context/renew", {method: "POST", body: JSON.stringify({prior_context_id: identity.contextId, patient_id: identity.patientId, client_session_id: clientSessionId()})});
    if (state.patientId !== identity.patientId || state.contextId !== identity.contextId || state.localSessionId !== identity.localSessionId || state.localResult?.request_id !== identity.requestId || !$("localIdentityConfirmed").checked || renewed.patient_id !== identity.patientId) throw new Error("Η ταυτότητα άλλαξε κατά την ανανέωση.");
    state.contextId = renewed.context_id;
    state.contextExpiresAt = renewed.expires_at || "";
  }

  async function prepareLocalCandidate() {
    try {
      const candidate = buildLocalCandidate();
      await renewLocalContext();
      state.localCandidateActive = true;
      $("candidateInput").value = JSON.stringify(candidate, null, 2);
      await requestPreview();
    } catch (error) {
      $("candidateMessage").textContent = error.message;
      setChip("previewState", "Απαιτείται διόρθωση", "err");
      if (error.status === 401 || error.status === 409) resetPatientContext();
    }
  }

  async function copyGesy() {
    if (!state.preview?.gesy_copy) return;
    try {
      await navigator.clipboard.writeText(state.preview.gesy_copy);
      $("gesyCopyStatus").textContent = "Αντιγράφηκε. Δεν έγινε αποστολή ή εγγραφή στο ΓεΣΥ.";
    } catch (_) {
      $("gesyCopyStatus").textContent = "Δεν επιτράπηκε η αντιγραφή.";
    }
  }

  function parseCandidate() {
    const text = $("candidateInput").value.trim();
    if (!text) {
      state.candidate = null;
      state.preview = null;
      $("saveBtn").disabled = true;
      setChip("candidateState", "Αναμονή");
      return null;
    }
    try {
      const parsed = JSON.parse(text);
      state.candidate = parsed;
      setChip("candidateState", "Έγκυρο JSON", "ok");
      return parsed;
    } catch (_) {
      state.candidate = null;
      state.preview = null;
      clearFormattedPreview();
      $("saveBtn").disabled = true;
      setChip("candidateState", "Μη έγκυρο JSON", "err");
      $("candidateMessage").textContent = "Το Dia πρέπει να εισαγάγει ένα έγκυρο VisitCaptureCandidateV1 JSON.";
      return null;
    }
  }

  function renderPreview() {
    const preview = state.preview;
    if (!preview) return;
    $("previewText").textContent = preview[state.level] || "";
    $("previewText").className = "preview-text";
    renderExplicitReviewNotes(preview);
    renderStructuredPreview(preview);
    document.querySelectorAll(".segment").forEach((button) => {
      button.classList.toggle("active", button.dataset.level === state.level);
    });
  }

  function invalidateLocalPreview(message) {
    clearTimeout(state.previewTimer);
    state.preview = null;
    $("saveBtn").disabled = true;
    $("gesyCopyRow").hidden = true;
    $("previewText").textContent = "Η τοπική προεπισκόπηση χρειάζεται νέα επιβεβαίωση ασθενούς.";
    $("previewText").className = "preview-text empty";
    clearFormattedPreview();
    $("blockingNotice").hidden = false;
    $("blockingNotice").textContent = message;
    setChip("previewState", "Απαιτείται επιβεβαίωση", "err");
  }

  async function requestPreview() {
    if (state.mode === "demo") {
      const text = $("candidateInput").value.trim();
      state.candidate = null;
      $("saveBtn").disabled = true;
      if (!text) {
        state.preview = null;
        $("previewText").textContent = "Επικόλλησε μια σύνοψη ή πάτησε «Συνθετικό παράδειγμα».";
        $("previewText").className = "preview-text empty";
        clearFormattedPreview();
        setChip("candidateState", "Αναμονή");
        setChip("previewState", "Δεν υπάρχει preview");
        return;
      }
      state.preview = parseDiaSummary(text);
      setChip("candidateState", state.preview.complete ? "3 ενότητες" : "Ελλιπής σύνοψη", state.preview.complete ? "ok" : "err");
      setChip("previewState", "Τοπική προεπισκόπηση", "ok");
      $("candidateMessage").textContent = state.preview.complete
        ? "Η σύνοψη εμφανίζεται μόνο στον browser. Δεν έγινε εγγραφή."
        : "Η προεπισκόπηση είναι διαθέσιμη, αλλά λείπουν μία ή περισσότερες ενότητες. Δεν έγινε εγγραφή.";
      $("blockingNotice").hidden = true;
      renderPreview();
      return;
    }
    const candidate = parseCandidate();
    if (!candidate || !state.contextId) return;
    const local = state.localCandidateActive ? {
      version: state.localConfirmationVersion,
      contextId: state.contextId,
      patientId: state.patientId,
      localSessionId: state.localSessionId,
      requestId: state.localResult?.request_id,
      candidateText: $("candidateInput").value,
      confirmed: $("localIdentityConfirmed").checked,
    } : null;
    if (local && (!local.confirmed || !local.patientId || !local.localSessionId || !local.requestId)) {
      invalidateLocalPreview("Επιβεβαίωσε ότι η καταγραφή ανήκει στον επιλεγμένο ασθενή.");
      return;
    }
    try {
      const payload = {context_id: state.contextId, candidate};
      if (local) payload.recording_patient_confirmed = local.confirmed;
      const preview = await apiJson("/clinical/visit-capture/preview", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (local && (
        !state.localCandidateActive || state.localConfirmationVersion !== local.version ||
        state.contextId !== local.contextId || state.patientId !== local.patientId ||
        state.localSessionId !== local.localSessionId || state.localResult?.request_id !== local.requestId ||
        $("candidateInput").value !== local.candidateText || !$("localIdentityConfirmed").checked
      )) return;
      if (preview.patient_id !== state.patientId) throw new Error("Patient context mismatch");
      state.preview = preview;
      $("candidateMessage").textContent = "Το structured candidate πέρασε τον server-side έλεγχο.";
      setChip("previewState", preview.can_save ? "Έτοιμο για έλεγχο" : "Απαιτείται διόρθωση", preview.can_save ? "ok" : "err");
      $("blockingNotice").hidden = preview.can_save;
      $("blockingNotice").textContent = preview.blocking_reason || "";
      $("saveBtn").disabled = !preview.can_save;
      $("gesyCopyRow").hidden = !preview.gesy_copy;
      renderPreview();
    } catch (error) {
      state.preview = null;
      clearFormattedPreview();
      $("saveBtn").disabled = true;
      $("gesyCopyRow").hidden = true;
      $("blockingNotice").hidden = false;
      $("blockingNotice").textContent = error.message;
      setChip("previewState", "Αποτυχία validation", "err");
      if (error.status === 409) {
        setChip("authState", "Επιβεβαίωσε ξανά τον ασθενή", "err");
        resetPatientContext();
      }
    }
  }

  function schedulePreview() {
    clearTimeout(state.previewTimer);
    state.previewTimer = setTimeout(requestPreview, 250);
  }

  async function save() {
    if (state.mode !== "record" || !state.preview?.can_save || !state.candidate || !state.contextId) return;
    const local = state.localCandidateActive ? {
      version: state.localConfirmationVersion,
      patientId: state.patientId,
      localSessionId: state.localSessionId,
      requestId: state.localResult?.request_id,
      candidate: state.candidate,
      candidateText: $("candidateInput").value,
      preview: state.preview,
      clientSessionId: clientSessionId(),
    } : null;
    if (local && (!$("localIdentityConfirmed").checked || !local.requestId)) {
      invalidateLocalPreview("Επιβεβαίωσε ότι η καταγραφή ανήκει στον επιλεγμένο ασθενή.");
      return;
    }
    $("saveBtn").disabled = true;
    $("saveMessage").textContent = "Αποθήκευση…";
    try {
      if (local) await renewLocalContext();
      if (local && (
        !state.localCandidateActive || state.localConfirmationVersion !== local.version ||
        state.patientId !== local.patientId || state.localSessionId !== local.localSessionId ||
        state.localResult?.request_id !== local.requestId || state.candidate !== local.candidate ||
        state.preview !== local.preview || $("candidateInput").value !== local.candidateText ||
        clientSessionId() !== local.clientSessionId || !$("localIdentityConfirmed").checked
      )) throw new Error("Η επιβεβαίωση ασθενούς ή το προσχέδιο άλλαξε. Κάνε νέα προεπισκόπηση.");
      const payload = {context_id: state.contextId, candidate: state.candidate};
      if (local) payload.recording_patient_confirmed = true;
      const result = await apiJson("/clinical/visit-capture/save", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      $("saveMessage").textContent = `Αποθηκεύτηκε: ${result.encounter.encounter_date} · ${result.pending.length} εκκρεμότητες.`;
      setChip("previewState", "Αποθηκεύτηκε", "ok");
      state.contextId = "";
      $("candidateInput").disabled = true;
      $("saveBtn").disabled = true;
      $("changePatientBtn").textContent = "Νέα καταγραφή";
    } catch (error) {
      $("saveMessage").textContent = error.message;
      setChip("previewState", "Δεν αποθηκεύτηκε", "err");
      if (error.status === 401 || error.status === 409) resetPatientContext();
      else $("saveBtn").disabled = false;
    }
  }

  $("loginBtn").addEventListener("click", login);
  $("clinicalKey").addEventListener("keydown", (event) => { if (event.key === "Enter") login(); });
  $("authToggle").addEventListener("click", () => {
    const open = $("authCard").hidden;
    $("authCard").hidden = !open;
    $("loginBox").hidden = !open;
    $("authToggle").setAttribute("aria-expanded", String(open));
  });
  $("demoModeBtn").addEventListener("click", () => selectMode("demo"));
  $("recordModeBtn").addEventListener("click", () => selectMode("record"));
  $("patientSearch").addEventListener("input", schedulePatientSearch);
  $("morePatientsBtn").addEventListener("click", () => searchPatientPage(true));
  $("confirmPatientBtn").addEventListener("click", confirmPatient);
  $("changePatientBtn").addEventListener("click", () => {
    resetPatientContext();
    $("patientSearch").focus();
  });
  $("copyDiaPromptBtn").addEventListener("click", copyDiaPrompt);
  $("exampleBtn").addEventListener("click", loadExample);
  $("patientInformed").addEventListener("change", () => {
    if (!$("patientInformed").checked && state.mediaRecorder) {
      cancelLocalRecording("Η ενημέρωση ασθενούς αποεπιλέχθηκε· η καταγραφή απορρίφθηκε.");
    } else {
      updateLocalAudioControls();
    }
  });
  $("startRecordingBtn").addEventListener("click", startLocalRecording);
  $("stopRecordingBtn").addEventListener("click", stopLocalRecording);
  $("localIdentityConfirmed").addEventListener("change", () => {
    state.localConfirmationVersion++;
    $("prepareLocalCandidateBtn").disabled = !$("localIdentityConfirmed").checked || !$("localRisksConfirmed").checked;
    if (state.localCandidateActive) {
      invalidateLocalPreview("Επιβεβαίωσε ξανά ότι η καταγραφή ανήκει στον επιλεγμένο ασθενή.");
      if ($("localIdentityConfirmed").checked) schedulePreview();
    }
  });
  $("localRisksConfirmed").addEventListener("change", () => {
    $("prepareLocalCandidateBtn").disabled = !$("localIdentityConfirmed").checked || !$("localRisksConfirmed").checked;
  });
  $("prepareLocalCandidateBtn").addEventListener("click", prepareLocalCandidate);
  $("copyGesyBtn").addEventListener("click", copyGesy);
  $("candidateInput").addEventListener("input", schedulePreview);
  $("saveBtn").addEventListener("click", save);
  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      state.level = button.dataset.level;
      renderPreview();
    });
  });

  if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
    window.addEventListener("clinical:authenticated", () => checkAuth());
    window.addEventListener("pagehide", () => cancelLocalRecording("Η σελίδα έκλεισε· η καταγραφή απορρίφθηκε."));
  }
  initializeLocalCompanion();
  selectMode("demo");
  checkAuth(true);
})();
