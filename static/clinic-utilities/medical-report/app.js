(() => {
  "use strict";

  const API_BASE = "/clinical/clinic-utilities/medical-report";
  const SOURCE_TYPES = [
    ["other", "Άλλο / αυτόματη ταξινόμηση"],
    ["gesy_visit", "Ιστορικό / επίσκεψη ΓεΣΥ"],
    ["clinician_note_self", "Δική μου ιατρική σημείωση"],
    ["clinician_note_other", "Σημείωση άλλου ιατρού"],
    ["specialist_report", "Γνωμάτευση ειδικού"],
    ["hospital_record", "Νοσοκομειακό αρχείο"],
    ["emergency_record", "ΤΑΕΠ / επείγοντα"],
    ["admission_note", "Σημείωμα εισαγωγής"],
    ["discharge_summary", "Εξιτήριο"],
    ["procedure_note", "Επέμβαση / procedure note"],
    ["imaging_report", "Απεικονιστική γνωμάτευση"],
    ["lab_report", "Εργαστηριακή εξέταση"],
    ["physiotherapy_report", "Φυσιοθεραπευτική έκθεση"],
    ["prior_medical_report", "Προηγούμενη ιατρική έκθεση"],
    ["sick_leave_certificate", "Αναρρωτική άδεια"],
  ];
  const state = { contract: null, files: [], fileTypes: [], draft: null, research: null, signatureFile: null };
  const $ = (id) => document.getElementById(id);
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function localToday() {
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  }

  function greekDate(value) {
    if (!value) return "—";
    const p = String(value).split("-");
    return p.length === 3 ? `${p[2]}/${p[1]}/${p[0]}` : value;
  }

  async function api(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, { credentials: "same-origin", ...options });
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        message = Array.isArray(body.detail) ? body.detail.map((x) => x.msg || JSON.stringify(x)).join(" · ") : body.detail || message;
      } catch (_) {}
      throw new Error(message);
    }
    return response;
  }

  function setError(message = "") {
    $("errorBox").hidden = !message;
    $("errorBox").textContent = message;
    if (message) $("errorBox").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function setWorking(active, text = "Επεξεργασία ιατρικών πηγών…") {
    $("workingBox").hidden = !active;
    $("workingBox").textContent = text;
    $("generateButton").disabled = active;
    $("researchButton").disabled = active || !state.draft;
    $("previewButton").disabled = active || !state.draft;
    $("downloadButton").disabled = active || !state.draft;
  }

  function currentCase() {
    const patientName = $("patientName").value.trim();
    if (!patientName) throw new Error("Συμπληρώστε το όνομα ασθενούς.");
    const reportDate = $("reportDate").value;
    if (!reportDate) throw new Error("Συμπληρώστε την ημερομηνία έκθεσης.");
    return {
      report_type: $("reportType").value,
      patient_name: patientName,
      id_type: $("idType").value,
      id_number: $("idNumber").value.trim(),
      birth_date: $("birthDate").value || null,
      occupation: $("occupation").value.trim(),
      incident_date: $("incidentDate").value || null,
      report_date: reportDate,
      instructing_party: $("instructingParty").value.trim(),
      instructing_reference: $("instructingReference").value.trim(),
      purpose_and_questions: $("purposeQuestions").value.trim(),
      clinician_context: $("clinicianContext").value.trim(),
    };
  }

  function updateSummary() {
    $("summaryPatient").textContent = $("patientName").value.trim() || "—";
    $("summaryIncident").textContent = greekDate($("incidentDate").value);
    $("summarySources").textContent = String(state.files.length + ($("clinicianContext").value.trim() ? 1 : 0));
    $("fileCounter").textContent = `${state.files.length} αρχεία`;
  }

  function invalidateDraftForSourceChange() {
    if (!state.draft) return;
    state.draft = null;
    state.research = null;
    $("analysisWorkspace").hidden = true;
    $("summaryAI").textContent = "Απαιτεί νέα δημιουργία";
    $("summaryResearch").textContent = "Δεν έγινε";
    invalidateConfirmation();
  }

  function updateFiles() {
    const list = $("fileList");
    list.replaceChildren();
    state.files.forEach((file, index) => {
      const li = el("li");
      const info = el("div");
      info.append(el("span", "", file.name), el("span", "meta", ` · ${(file.size / 1024).toFixed(0)} KB`));
      const select = el("select", "source-type-select");
      select.setAttribute("aria-label", `Τύπος πηγής για ${file.name}`);
      SOURCE_TYPES.forEach(([value, label]) => {
        const option = el("option", "", label);
        option.value = value;
        option.selected = (state.fileTypes[index] || "other") === value;
        select.append(option);
      });
      select.addEventListener("change", () => {
        state.fileTypes[index] = select.value;
        invalidateDraftForSourceChange();
      });
      li.append(info, select);
      list.append(li);
    });
    updateSummary();
  }

  function invalidateConfirmation() {
    $("clinicianConfirmed").checked = false;
  }

  function renderWarnings(items = []) {
    const root = $("warningList");
    root.replaceChildren();
    items.forEach((text) => root.append(el("div", "warning-chip", text)));
  }

  function renderSources() {
    const root = $("sourcesPanel");
    root.replaceChildren(el("h2", "", "Πηγές & σύνοψη"));
    const summaries = new Map((state.draft.analysis.source_summaries || []).map((x) => [x.source_id, x]));
    (state.draft.sources || []).forEach((source) => {
      const summary = summaries.get(source.source_id);
      const box = el("div", "source-item");
      box.append(el("strong", "", source.filename));
      const typeText = summary?.proposed_source_type && summary.proposed_source_type !== source.source_type
        ? `${source.source_type} · AI πρόταση: ${summary.proposed_source_type}`
        : source.source_type;
      box.append(el("div", "meta", `${source.status} · ${source.page_count} σελ. · ${source.character_count} χαρακτήρες · ${typeText}`));
      if (summary?.summary) box.append(el("p", "", summary.summary));
      if (summary?.author || summary?.specialty || summary?.institution) box.append(el("div", "meta", [summary.author, summary.specialty, summary.institution].filter(Boolean).join(" · ")));
      root.append(box);
    });
  }

  function renderEvidence() {
    const root = $("evidencePanel");
    root.replaceChildren(el("h2", "", "Evidence Ledger"));
    (state.draft.analysis.evidence_items || []).forEach((item) => {
      const box = el("div", `evidence-item${item.conflict_key ? " conflict" : ""}`);
      box.append(el("span", "evidence-type", item.evidence_type));
      box.append(el("strong", "", item.statement));
      const pages = (item.page_numbers || []).length ? ` · σελ. ${(item.page_numbers || []).join(", ")}` : "";
      box.append(el("div", "meta", `${item.source_id}${pages}${item.date_text ? ` · ${item.date_text}` : item.event_date ? ` · ${greekDate(item.event_date)}` : ""} · ${item.certainty}`));
      root.append(box);
    });
  }

  function renderTimeline() {
    const root = $("timelinePanel");
    root.replaceChildren(el("h2", "", "Χρονολογική πορεία"));
    (state.draft.analysis.timeline || []).forEach((item) => {
      const box = el("div", `timeline-item${(item.conflict_flags || []).length ? " conflict" : ""}`);
      box.append(el("strong", "", `${item.event_date ? greekDate(item.event_date) : item.date_text || "Χωρίς ακριβή ημερομηνία"} · ${item.title}`));
      box.append(el("p", "", item.summary));
      box.append(el("div", "meta", (item.source_ids || []).join(" · ")));
      root.append(box);
    });
  }

  function renderDiagnoses() {
    const root = $("diagnosesPanel");
    root.replaceChildren(el("h2", "", "Διαγνώσεις / αιτιώδης συνάφεια"));
    (state.draft.analysis.diagnosis_analyses || []).forEach((item) => {
      const box = el("div", "diagnosis-item");
      box.append(el("strong", "", item.diagnosis));
      if (item.causation_draft) box.append(el("p", "", item.causation_draft));
      if (item.pre_existing_discussion) box.append(el("p", "meta", `Προϋπάρχον: ${item.pre_existing_discussion}`));
      if ((item.alternative_causes || []).length) box.append(el("p", "meta", `Εναλλακτικές αιτίες: ${item.alternative_causes.join(" · ")}`));
      if (item.uncertainty) box.append(el("p", "meta", `Αβεβαιότητα: ${item.uncertainty}`));
      root.append(box);
    });
  }

  function renderReportSections() {
    const root = $("reportSections");
    root.replaceChildren();
    (state.draft.analysis.report_sections || []).forEach((section, index) => {
      const box = el("div", "report-section");
      const label = el("label");
      label.append(el("span", "", section.title));
      const textarea = el("textarea");
      textarea.value = section.draft_text || "";
      textarea.dataset.sectionIndex = String(index);
      textarea.addEventListener("input", invalidateConfirmation);
      label.append(textarea);
      box.append(label);
      root.append(box);
    });
  }

  function renderPrognosisQuestions() {
    const root = $("prognosisQuestions");
    root.replaceChildren();
    (state.draft.analysis.prognosis_questions || []).forEach((item) => {
      root.append(el("div", "question", `${item.diagnosis_or_problem}: ${item.question}${item.rationale ? ` — ${item.rationale}` : ""}`));
    });
  }

  function renderDraft() {
    $("analysisWorkspace").hidden = false;
    renderSources(); renderEvidence(); renderTimeline(); renderDiagnoses(); renderReportSections(); renderPrognosisQuestions();
    renderWarnings([...(state.draft.deterministic_warnings || []), ...(state.draft.analysis.warnings || [])]);
    $("summaryAI").textContent = `Έτοιμο · ${state.draft.usage?.model || "AI"}`;
    $("summaryResearch").textContent = "Δεν έγινε";
    state.research = null;
    $("researchResult").hidden = true;
    $("researchText").value = "";
    $("citationList").replaceChildren();
    invalidateConfirmation();
    $("analysisWorkspace").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function generateDraft() {
    setError();
    try {
      const caseData = currentCase();
      if (!caseData.clinician_context && !state.files.length) throw new Error("Δώστε το ιστορικό ή/και τουλάχιστον ένα σχετικό ιατρικό αρχείο.");
      setWorking(true, "Ανάγνωση πηγών και δημιουργία προσχεδίου…");
      const form = new FormData();
      form.append("case_json", JSON.stringify(caseData));
      form.append("file_source_types_json", JSON.stringify(state.fileTypes));
      state.files.forEach((file) => form.append("files", file, file.name));
      const response = await api("/api/analyze", { method: "POST", body: form });
      state.draft = await response.json();
      renderDraft();
    } catch (error) {
      setError(error.message);
      $("summaryAI").textContent = "Αποτυχία";
    } finally { setWorking(false); }
  }

  async function researchLiterature() {
    if (!state.draft) return;
    setError();
    setWorking(true, "Στοχευμένη αναζήτηση βιβλιογραφίας…");
    try {
      const response = await api("/api/research", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case: state.draft.case, analysis: state.draft.analysis }),
      });
      state.research = await response.json();
      $("researchText").value = state.research.research_text || "";
      $("researchText").addEventListener("input", invalidateConfirmation, { once: false });
      const list = $("citationList"); list.replaceChildren();
      (state.research.citations || []).forEach((citation) => {
        const li = el("li"); const a = el("a", "", citation.title || citation.url);
        a.href = citation.url; a.target = "_blank"; a.rel = "noopener noreferrer"; li.append(a); list.append(li);
      });
      $("researchResult").hidden = false;
      $("summaryResearch").textContent = `${(state.research.citations || []).length} πηγές`;
      invalidateConfirmation();
    } catch (error) { setError(error.message); }
    finally { setWorking(false); }
  }

  function editedSections() {
    if (!state.draft) return [];
    return (state.draft.analysis.report_sections || []).map((section, index) => {
      const textarea = document.querySelector(`textarea[data-section-index="${index}"]`);
      return { ...section, draft_text: textarea ? textarea.value.trim() : section.draft_text };
    });
  }

  function finalPayload() {
    if (!state.draft) throw new Error("Δεν υπάρχει προσχέδιο έκθεσης.");
    if (!$("clinicianConfirmed").checked) throw new Error("Απαιτείται η τελική ιατρική επιβεβαίωση πριν το PDF.");
    return {
      case: state.draft.case,
      sections: editedSections(),
      research_text: $("researchText").value.trim(),
      citations: state.research?.citations || [],
      declaration_text: $("declarationText").value.trim(),
      clinician_confirmed: true,
    };
  }

  function pdfForm() {
    const form = new FormData();
    form.append("report_json", JSON.stringify(finalPayload()));
    if (state.signatureFile) form.append("signature", state.signatureFile, state.signatureFile.name);
    return form;
  }

  async function previewPdf() {
    setError();
    let win = window.open("", "_blank");
    if (!win) { setError("Ο browser εμπόδισε την προεπισκόπηση PDF."); return; }
    win.opener = null;
    try {
      setWorking(true, "Δημιουργία τελικής έκθεσης…");
      const response = await api("/api/preview", { method: "POST", body: pdfForm() });
      const url = URL.createObjectURL(await response.blob());
      win.location.replace(url); setTimeout(() => URL.revokeObjectURL(url), 60000);
    } catch (error) { win.close(); setError(error.message); }
    finally { setWorking(false); }
  }

  function filenameFromDisposition(header) {
    const match = String(header || "").match(/filename\*=UTF-8''([^;]+)/i);
    if (!match) return "Ιατρική_Έκθεση.pdf";
    try { return decodeURIComponent(match[1]); } catch (_) { return "Ιατρική_Έκθεση.pdf"; }
  }

  async function downloadPdf() {
    setError();
    try {
      setWorking(true, "Δημιουργία τελικής έκθεσης…");
      const response = await api("/api/pdf", { method: "POST", body: pdfForm() });
      const url = URL.createObjectURL(await response.blob());
      const a = document.createElement("a"); a.href = url; a.download = filenameFromDisposition(response.headers.get("Content-Disposition"));
      document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { setError(error.message); }
    finally { setWorking(false); }
  }

  function chooseSignature(file) {
    if (!file) return;
    const limit = state.contract?.limits?.max_signature_bytes || 2 * 1024 * 1024;
    if (file.size > limit || (!/image\/(png|jpeg)/.test(file.type) && !/\.(png|jpe?g)$/i.test(file.name))) {
      setError("Η υπογραφή πρέπει να είναι PNG/JPEG εντός του επιτρεπτού μεγέθους."); return;
    }
    state.signatureFile = file;
    $("signatureStatus").textContent = `✓ ${file.name} · μόνο για αυτή τη συνεδρία`;
    $("removeSignatureButton").hidden = false;
  }

  function removeSignature() {
    state.signatureFile = null; $("signatureInput").value = "";
    $("signatureStatus").textContent = "Χειρόγραφη υπογραφή ή φόρτωση εικόνας μόνο για αυτή τη σελίδα.";
    $("removeSignatureButton").hidden = true;
  }

  function bindTabs() {
    document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === button));
      document.querySelectorAll(".tab-panel").forEach((panel) => { panel.hidden = panel.id !== button.dataset.tab; });
    }));
  }

  async function loadContract() {
    try {
      const response = await api("/api/contract"); state.contract = await response.json();
      const ai = state.contract.ai || {};
      if (!state.contract.clinician_configured) {
        $("providerWarning").textContent = "Δεν υπάρχει server-side clinician profile. Η τελική PDF έκδοση δεν είναι διαθέσιμη."; $("providerWarning").hidden = false;
      } else if (!ai.enabled || !ai.api_key_configured) {
        $("providerWarning").textContent = "Το AI provider δεν έχει ενεργοποιηθεί ακόμη στον server. Το UI είναι διαθέσιμο, αλλά η δημιουργία προσχεδίου παραμένει κλειδωμένη."; $("providerWarning").hidden = false;
      } else if (!ai.phi_provider_approved) {
        $("providerWarning").textContent = "Το AI είναι ρυθμισμένο, αλλά η χρήση αναγνωρίσιμων ιατρικών δεδομένων δεν έχει εγκριθεί ακόμη από το provider/privacy gate."; $("providerWarning").hidden = false;
      }
    } catch (error) { setError(error.message); }
  }

  function bind() {
    $("sourceFiles").addEventListener("change", (event) => {
      state.files = Array.from(event.target.files || []);
      state.fileTypes = state.files.map(() => "other");
      invalidateDraftForSourceChange();
      updateFiles();
    });
    $("generateButton").addEventListener("click", generateDraft);
    $("researchButton").addEventListener("click", researchLiterature);
    $("previewButton").addEventListener("click", previewPdf);
    $("downloadButton").addEventListener("click", downloadPdf);
    $("signatureButton").addEventListener("click", () => $("signatureInput").click());
    $("signatureInput").addEventListener("change", (event) => chooseSignature(event.target.files?.[0]));
    $("removeSignatureButton").addEventListener("click", removeSignature);
    $("declarationText").addEventListener("input", invalidateConfirmation);
    ["patientName", "incidentDate", "clinicianContext"].forEach((id) => $(id).addEventListener("input", updateSummary));
    ["reportType", "idType", "idNumber", "birthDate", "occupation", "reportDate", "instructingParty", "instructingReference", "purposeQuestions"].forEach((id) => $(id).addEventListener("input", invalidateConfirmation));
    bindTabs();
  }

  $("reportDate").value = localToday();
  bind(); updateFiles(); updateSummary(); loadContract(); setWorking(false);
})();
