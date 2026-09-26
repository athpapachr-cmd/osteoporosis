(() => {
  "use strict";

  const API_BASE = "/clinical/clinic-utilities/sick-leave";
  const state = {
    contract: null,
    signatureFile: null,
    previous: null,
    derivedFromDocumentId: "",
    relation: null,
    templateId: "classic",
    colorTheme: "navy",
  };

  const $ = (id) => document.getElementById(id);

  function localToday() {
    const now = new Date();
    const yyyy = now.getFullYear();
    const mm = String(now.getMonth() + 1).padStart(2, "0");
    const dd = String(now.getDate()).padStart(2, "0");
    return `${yyyy}-${mm}-${dd}`;
  }

  function formatGreekDate(value) {
    if (!value) return "—";
    const parts = value.split("-");
    if (parts.length !== 3) return value;
    return `${parts[2]}/${parts[1]}/${parts[0]}`;
  }

  function escapeSummary(value) {
    return String(value || "").trim() || "—";
  }

  async function api(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, { credentials: "same-origin", ...options });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (Array.isArray(body.detail)) {
          detail = body.detail.map((item) => item.msg || JSON.stringify(item)).join(" · ");
        } else {
          detail = body.detail || detail;
        }
      } catch (_) {}
      throw new Error(detail);
    }
    return response;
  }

  function showError(message) {
    const box = $("errorBox");
    box.textContent = message;
    box.hidden = false;
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function clearError() {
    $("errorBox").hidden = true;
    $("errorBox").textContent = "";
  }

  function setWorking(working) {
    $("workingBox").hidden = !working;
    $("previewButton").disabled = working || !state.contract?.clinician_configured;
    $("downloadButton").disabled = working || !state.contract?.clinician_configured;
  }

  function updateIdLabel() {
    $("idNumberLabel").textContent = $("idType").value === "ARC" ? "Αριθμός ARC" : "Αριθμός ΑΔΤ";
    updateSummary();
  }

  function inclusiveDays() {
    const start = $("leaveFrom").value;
    const end = $("leaveTo").value;
    if (!start || !end) return null;
    const a = new Date(`${start}T00:00:00`);
    const b = new Date(`${end}T00:00:00`);
    if (Number.isNaN(a.valueOf()) || Number.isNaN(b.valueOf()) || b < a) return null;
    return Math.round((b - a) / 86400000) + 1;
  }

  function updateSummary() {
    const days = inclusiveDays();
    $("durationPill").textContent = days ? `Διάρκεια: ${days} ημέρες` : "Διάρκεια: —";
    $("summaryPatient").textContent = escapeSummary($("patientName").value);
    const idNumber = $("idNumber").value.trim();
    $("summaryId").textContent = idNumber ? `${$("idType").value}: ${idNumber}` : "—";
    const start = $("leaveFrom").value;
    const end = $("leaveTo").value;
    $("summaryDates").textContent = start && end ? `${formatGreekDate(start)} → ${formatGreekDate(end)}${days ? ` · ${days} ημέρες` : ""}` : "—";
    $("summaryIssued").textContent = formatGreekDate($("issuedOn").value);
    $("summarySignature").textContent = state.signatureFile ? "Ψηφιακή εικόνα · τρέχουσα συνεδρία" : "Χειρόγραφη";
    if ($("summaryAppearance")) $("summaryAppearance").textContent = appearanceLabel();
  }

  function gatherDraft() {
    const draft = {
      patient_name: $("patientName").value.trim(),
      id_type: $("idType").value,
      id_number: $("idNumber").value.trim(),
      diagnosis: $("diagnosis").value.trim(),
      leave_from: $("leaveFrom").value,
      leave_to: $("leaveTo").value,
      issued_on: $("issuedOn").value,
      derived_from_document_id: state.derivedFromDocumentId,
      relation: state.relation,
    };
    const missing = [];
    if (!draft.patient_name) missing.push("όνομα ασθενούς");
    if (!draft.id_number) missing.push(draft.id_type === "ARC" ? "ARC" : "ΑΔΤ");
    if (!draft.diagnosis) missing.push("διάγνωση");
    if (!draft.leave_from) missing.push("ημερομηνία έναρξης");
    if (!draft.leave_to) missing.push("ημερομηνία λήξης");
    if (!draft.issued_on) missing.push("ημερομηνία έκδοσης");
    if (missing.length) throw new Error(`Συμπληρώστε: ${missing.join(", ")}.`);
    if (draft.leave_to < draft.leave_from) throw new Error("Η λήξη της άδειας δεν μπορεί να προηγείται της έναρξης.");
    return draft;
  }

  function formDataForDraft() {
    const form = new FormData();
    form.append("draft_json", JSON.stringify(gatherDraft()));
    form.append("template_id", state.templateId);
    form.append("color_theme", state.colorTheme);
    if (state.signatureFile) form.append("signature", state.signatureFile, state.signatureFile.name);
    return form;
  }

  function appearanceLabel() {
    const template = state.contract?.appearance?.templates?.find((item) => item.id === state.templateId);
    const theme = state.contract?.appearance?.color_themes?.find((item) => item.id === state.colorTheme);
    return `${template?.label || state.templateId} · ${theme?.label || state.colorTheme}`;
  }

  function selectTemplate(templateId) {
    const allowed = state.contract?.appearance?.templates?.map((item) => item.id) || ["classic", "modern", "minimal", "compact", "formal"];
    if (!allowed.includes(templateId)) return;
    state.templateId = templateId;
    document.querySelectorAll("[data-template]").forEach((button) => {
      const selected = button.dataset.template === templateId;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", selected ? "true" : "false");
    });
    updateSummary();
  }

  function selectColorTheme(colorTheme) {
    const allowed = state.contract?.appearance?.color_themes?.map((item) => item.id) || ["navy", "teal", "graphite", "burgundy", "forest", "monochrome"];
    if (!allowed.includes(colorTheme)) return;
    state.colorTheme = colorTheme;
    document.querySelectorAll("[data-theme]").forEach((button) => {
      const selected = button.dataset.theme === colorTheme;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", selected ? "true" : "false");
    });
    updateSummary();
  }

  function filenameFromDisposition(header) {
    const match = String(header || "").match(/filename\*=UTF-8''([^;]+)/i);
    if (!match) return "Αναρρωτική.pdf";
    try { return decodeURIComponent(match[1]); } catch (_) { return "Αναρρωτική.pdf"; }
  }

  async function previewPdf() {
    clearError();
    let previewWindow = null;
    try {
      // Open synchronously from the clinician click so browsers do not classify
      // the eventual PDF navigation as an unsolicited pop-up.
      previewWindow = window.open("", "_blank");
      if (!previewWindow) throw new Error("Ο browser εμπόδισε την προεπισκόπηση. Επιτρέψτε pop-ups για αυτή τη σελίδα.");
      previewWindow.opener = null;
      setWorking(true);
      const response = await api("/api/preview", { method: "POST", body: formDataForDraft() });
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      previewWindow.location.replace(url);
      setTimeout(() => URL.revokeObjectURL(url), 60000);
    } catch (error) {
      if (previewWindow && !previewWindow.closed) previewWindow.close();
      showError(error.message);
    } finally {
      setWorking(false);
    }
  }

  async function downloadPdf() {
    clearError();
    setWorking(true);
    try {
      const response = await api("/api/pdf", { method: "POST", body: formDataForDraft() });
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filenameFromDisposition(response.headers.get("Content-Disposition"));
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      showError(error.message);
    } finally {
      setWorking(false);
    }
  }

  function clearPatientForm() {
    $("patientName").value = "";
    $("idType").value = "ADT";
    $("idNumber").value = "";
    $("diagnosis").value = "";
    $("leaveFrom").value = "";
    $("leaveTo").value = "";
    $("issuedOn").value = localToday();
    state.previous = null;
    state.derivedFromDocumentId = "";
    state.relation = null;
    $("previousCard").hidden = true;
    $("previousPdfInput").value = "";
    clearError();
    updateIdLabel();
    updateSummary();
    $("patientName").focus();
  }

  async function importPrevious(file) {
    if (!file) return;
    clearError();
    setWorking(true);
    try {
      const form = new FormData();
      form.append("previous_pdf", file, file.name);
      const response = await api("/api/import-previous", { method: "POST", body: form });
      state.previous = await response.json();
      const p = state.previous.previous;
      $("previousSummary").textContent = `${p.patient_name} · ${p.id_type} ${p.id_number} · ${formatGreekDate(p.leave_from)}–${formatGreekDate(p.leave_to)} · ${p.diagnosis}`;
      $("previousCard").hidden = false;
    } catch (error) {
      state.previous = null;
      $("previousCard").hidden = true;
      showError(error.message);
    } finally {
      setWorking(false);
      $("previousPdfInput").value = "";
    }
  }

  function applyReuse(kind) {
    const source = state.previous?.[kind];
    if (!source) return;
    $("patientName").value = source.patient_name || "";
    $("idType").value = source.id_type || "ADT";
    $("idNumber").value = source.id_number || "";
    $("diagnosis").value = source.diagnosis || "";
    $("leaveFrom").value = source.leave_from || "";
    $("leaveTo").value = "";
    $("issuedOn").value = localToday();
    state.derivedFromDocumentId = source.derived_from_document_id || "";
    state.relation = source.relation || null;
    updateIdLabel();
    updateSummary();
    (kind === "extension" ? $("leaveTo") : $("diagnosis")).focus();
  }

  function chooseSignature(file) {
    if (!file) return;
    const allowed = ["image/png", "image/jpeg"];
    if (!allowed.includes(file.type) && !/\.(png|jpe?g)$/i.test(file.name)) {
      showError("Η υπογραφή πρέπει να είναι PNG ή JPEG.");
      return;
    }
    if (file.size > (state.contract?.limits?.signature_bytes || 2 * 1024 * 1024)) {
      showError("Η υπογραφή υπερβαίνει το επιτρεπτό μέγεθος.");
      return;
    }
    state.signatureFile = file;
    $("signatureStatus").textContent = `✓ ${file.name} · έτοιμη για την τρέχουσα συνεδρία`;
    $("removeSignatureButton").hidden = false;
    clearError();
    updateSummary();
  }

  function removeSignature() {
    state.signatureFile = null;
    $("signatureInput").value = "";
    $("signatureStatus").textContent = "Δεν έχει φορτωθεί υπογραφή. Το PDF μπορεί να εκδοθεί και για χειρόγραφη υπογραφή.";
    $("removeSignatureButton").hidden = true;
    updateSummary();
  }

  async function loadContract() {
    try {
      const response = await api("/api/contract");
      state.contract = await response.json();
      $("configWarning").hidden = Boolean(state.contract.clinician_configured);
      $("downloadButton").disabled = !state.contract.clinician_configured;
      $("previewButton").disabled = !state.contract.clinician_configured;
      const appearance = state.contract.appearance || {};
      selectTemplate(appearance.default_template || "classic");
      selectColorTheme(appearance.default_color_theme || "navy");
    } catch (error) {
      showError(error.message);
      $("downloadButton").disabled = true;
      $("previewButton").disabled = true;
    }
  }

  function bind() {
    ["patientName", "idNumber", "diagnosis", "leaveFrom", "leaveTo", "issuedOn"].forEach((id) => {
      $(id).addEventListener("input", updateSummary);
      $(id).addEventListener("change", updateSummary);
    });
    $("idType").addEventListener("change", updateIdLabel);
    $("previewButton").addEventListener("click", previewPdf);
    $("downloadButton").addEventListener("click", downloadPdf);
    $("clearButton").addEventListener("click", clearPatientForm);
    $("signatureButton").addEventListener("click", () => $("signatureInput").click());
    $("signatureInput").addEventListener("change", (event) => chooseSignature(event.target.files?.[0]));
    $("removeSignatureButton").addEventListener("click", removeSignature);
    $("importPreviousButton").addEventListener("click", () => $("previousPdfInput").click());
    $("previousPdfInput").addEventListener("change", (event) => importPrevious(event.target.files?.[0]));
    $("extendButton").addEventListener("click", () => applyReuse("extension"));
    $("samePatientButton").addEventListener("click", () => applyReuse("new_leave_same_patient"));
    document.querySelectorAll("[data-template]").forEach((button) => {
      button.addEventListener("click", () => selectTemplate(button.dataset.template));
    });
    document.querySelectorAll("[data-theme]").forEach((button) => {
      button.addEventListener("click", () => selectColorTheme(button.dataset.theme));
    });
  }

  $("issuedOn").value = localToday();
  bind();
  updateIdLabel();
  updateSummary();
  loadContract();
})();
