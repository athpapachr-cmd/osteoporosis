(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const CLIENT_SESSION_KEY = "clinical.visitCapture.clientSession.v1";
  const state = {
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

  function clearCandidateState(message = "Επιβεβαίωσε πρώτα τον ασθενή.") {
    state.candidate = null;
    state.preview = null;
    $("candidateInput").value = "";
    $("candidateInput").disabled = !state.contextId;
    $("candidateMessage").textContent = message;
    $("previewText").textContent = "Η προεπισκόπηση θα εμφανιστεί αυτόματα όταν το candidate είναι έγκυρο.";
    $("previewText").className = "preview-text empty";
    $("blockingNotice").hidden = true;
    $("saveBtn").disabled = true;
    $("saveMessage").textContent = "";
    setChip("candidateState", state.contextId ? "Έτοιμο για insert" : "Αναμονή");
    setChip("previewState", "Δεν υπάρχει preview");
  }

  function resetPatientContext() {
    state.contextId = "";
    state.patientId = "";
    $("confirmedPatient").hidden = true;
    $("patientBox").hidden = false;
    clearCandidateState();
  }

  async function checkAuth() {
    try {
      await apiJson("/clinical/status", {method: "GET"});
      $("loginBox").hidden = true;
      $("patientBox").hidden = false;
      setChip("authState", "Authenticated", "ok");
      const fromQuery = new URLSearchParams(location.search).get("patient_id") || "";
      if (fromQuery) $("patientId").value = fromQuery;
    } catch (error) {
      $("loginBox").hidden = false;
      $("patientBox").hidden = true;
      setChip("authState", error.status === 503 ? "Clinical access disabled" : "Απαιτείται σύνδεση", "err");
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
    const patientId = $("patientId").value.trim();
    if (!patientId) {
      $("candidateMessage").textContent = "Χρειάζεται internal patient ID.";
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
      $("confirmedPatientId").textContent = context.patient_id;
      $("confirmedPatient").hidden = false;
      $("patientBox").hidden = true;
      $("candidateInput").disabled = false;
      clearCandidateState("Το candidate παραμένει προσωρινό μέχρι το Save.");
      setChip("authState", "Patient confirmed", "ok");
      $("candidateInput").focus();
    } catch (error) {
      resetPatientContext();
      setChip("authState", error.message, "err");
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
    if (!state.preview?.can_save || !state.candidate || !state.contextId) return;
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
  $("confirmPatientBtn").addEventListener("click", confirmPatient);
  $("patientId").addEventListener("keydown", (event) => { if (event.key === "Enter") confirmPatient(); });
  $("changePatientBtn").addEventListener("click", () => {
    resetPatientContext();
    $("patientId").focus();
  });
  $("candidateInput").addEventListener("input", schedulePreview);
  $("saveBtn").addEventListener("click", save);
  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      state.level = button.dataset.level;
      renderPreview();
    });
  });

  checkAuth();
})();
