(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const CLIENT_SESSION_KEY = "clinical.visitCapture.clientSession.v1";
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
    state.candidate = null;
    state.preview = null;
    $("candidateInput").value = "";
    $("candidateInput").disabled = state.mode === "record" && !state.contextId;
    $("candidateMessage").textContent = message || (
      state.mode === "demo" ? "Μπορείς να ξεκινήσεις αμέσως, χωρίς ασθενή." : "Επιβεβαίωσε πρώτα τον ασθενή."
    );
    $("previewText").textContent = "Η προεπισκόπηση θα εμφανιστεί μόλις επικολλήσεις μία καταγραφή.";
    $("previewText").className = "preview-text empty";
    $("blockingNotice").hidden = true;
    $("saveBtn").disabled = true;
    $("saveMessage").textContent = "";
    setChip("candidateState", state.mode === "demo" ? "Έτοιμο για δοκιμή" : (state.contextId ? "Έτοιμο για insert" : "Αναμονή"));
    setChip("previewState", "Δεν υπάρχει preview");
  }

  function resetPatientContext() {
    state.contextId = "";
    state.patientId = "";
    $("confirmedPatient").hidden = true;
    $("patientBox").hidden = false;
    clearCandidateState();
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
    state.mode = mode;
    state.contextId = "";
    state.patientId = "";
    clearPatientLookup();
    $("demoModeBtn").classList.toggle("active", mode === "demo");
    $("recordModeBtn").classList.toggle("active", mode === "record");
    $("demoModeBtn").setAttribute("aria-pressed", String(mode === "demo"));
    $("recordModeBtn").setAttribute("aria-pressed", String(mode === "record"));
    $("patientPanel").hidden = mode !== "record";
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
    if (mode === "record") loadPatients();
  }

  async function checkAuth() {
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
        selectMode("record");
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
      state.patientId = context.patient_id;
      const patient = state.patients.find((item) => item.patient_id === context.patient_id);
      $("confirmedPatientId").textContent = patient ? patientLabel(patient) : context.patient_id;
      $("confirmedPatient").hidden = false;
      $("patientBox").hidden = true;
      clearCandidateState("Η καταγραφή παραμένει προσωρινή μέχρι το ρητό Save.");
      $("patientMessage").textContent = "";
      $("candidateInput").focus();
    } catch (error) {
      resetPatientContext();
      $("patientMessage").textContent = error.message;
    }
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
        complete: false,
      };
    }
    if (preface.join("\n").trim()) parts.brief.unshift(preface.join("\n").trim());
    const missing = "Δεν δόθηκε αυτή η ενότητα από το Dia. Έλεγξε το prompt και την απάντηση.";
    return {
      snapshot: parts.snapshot.join("\n").trim() || missing,
      brief: parts.brief.join("\n").trim() || missing,
      detail: parts.detail.join("\n").trim() || missing,
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
    document.querySelectorAll(".segment").forEach((button) => {
      button.classList.toggle("active", button.dataset.level === state.level);
    });
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
    try {
      const preview = await apiJson("/clinical/visit-capture/preview", {
        method: "POST",
        body: JSON.stringify({context_id: state.contextId, candidate}),
      });
      if (preview.patient_id !== state.patientId) throw new Error("Patient context mismatch");
      state.preview = preview;
      $("candidateMessage").textContent = "Το structured candidate πέρασε τον server-side έλεγχο.";
      setChip("previewState", preview.can_save ? "Έτοιμο για έλεγχο" : "Απαιτείται διόρθωση", preview.can_save ? "ok" : "err");
      $("blockingNotice").hidden = preview.can_save;
      $("blockingNotice").textContent = preview.blocking_reason || "";
      $("saveBtn").disabled = !preview.can_save;
      renderPreview();
    } catch (error) {
      state.preview = null;
      $("saveBtn").disabled = true;
      $("blockingNotice").hidden = false;
      $("blockingNotice").textContent = error.message;
      setChip("previewState", "Αποτυχία validation", "err");
      if (error.status === 409) {
        state.contextId = "";
        $("candidateInput").disabled = true;
        $("confirmedPatient").hidden = true;
        $("patientBox").hidden = false;
        setChip("authState", "Επιβεβαίωσε ξανά τον ασθενή", "err");
      }
    }
  }

  function schedulePreview() {
    clearTimeout(state.previewTimer);
    state.previewTimer = setTimeout(requestPreview, 250);
  }

  async function save() {
    if (state.mode !== "record" || !state.preview?.can_save || !state.candidate || !state.contextId) return;
    $("saveBtn").disabled = true;
    $("saveMessage").textContent = "Αποθήκευση…";
    try {
      const result = await apiJson("/clinical/visit-capture/save", {
        method: "POST",
        body: JSON.stringify({context_id: state.contextId, candidate: state.candidate}),
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
      $("saveBtn").disabled = false;
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
  $("candidateInput").addEventListener("input", schedulePreview);
  $("saveBtn").addEventListener("click", save);
  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      state.level = button.dataset.level;
      renderPreview();
    });
  });

  selectMode("demo");
  checkAuth();
})();
