(() => {
  "use strict";

  const ENDPOINT = "/clinical/transcript/extract";
  let candidates = [];
  let dialog = null;
  let textarea = null;
  let statusNode = null;
  let resultsNode = null;
  let submitButton = null;

  function addStyles() {
    if (document.querySelector("style[data-transcript-capture-style]")) return;
    const style = document.createElement("style");
    style.dataset.transcriptCaptureStyle = "true";
    style.textContent = `
      .transcript-capture-card{width:min(900px,calc(100vw - 32px));max-height:90vh;overflow:auto}
      .transcript-capture-card textarea{width:100%;min-height:220px;resize:vertical;font:inherit;padding:12px}
      .transcript-capture-warning{padding:10px 12px;border:1px solid currentColor;border-radius:8px;margin:12px 0;font-size:.92rem}
      .transcript-capture-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:12px;flex-wrap:wrap}
      .transcript-candidate{border:1px solid #d9d9d9;border-radius:10px;padding:12px;margin-top:10px}
      .transcript-candidate h3{margin:0 0 6px;font-size:1rem}
      .transcript-candidate dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 10px;margin:8px 0}
      .transcript-candidate dt{font-weight:700}.transcript-candidate dd{margin:0;overflow-wrap:anywhere}
      .transcript-target{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82rem}
      .transcript-status{min-height:1.4em;margin-top:8px}.transcript-status[data-state="error"]{font-weight:700}
    `;
    document.head.appendChild(style);
  }

  function makeDialog() {
    if (dialog) return dialog;
    dialog = document.createElement("dialog");
    dialog.id = "transcriptCaptureDialog";
    dialog.className = "modal";
    dialog.innerHTML = `
      <div class="modal-card transcript-capture-card" role="document">
        <div class="modal-head">
          <div><h2>Heidi transcript → υποψήφια κλινικά δεδομένα</h2><p>Προσωρινή AI εξαγωγή για ιατρικό έλεγχο.</p></div>
          <button class="icon-btn" type="button" data-transcript-close aria-label="Κλείσιμο">×</button>
        </div>
        <div class="transcript-capture-warning"><strong>AI-extracted candidate ≠ επιβεβαιωμένο κλινικό δεδομένο.</strong><br>Το PR-1 δεν γράφει τίποτε στον φάκελο. Μέχρι να κλείσει ξεχωριστά το transcript privacy/provider gate, χρησιμοποιούνται μόνο συνθετικά ή απο-αναγνωρισμένα transcripts.</div>
        <label><span>Heidi transcript</span><textarea data-transcript-input maxlength="120000" autocomplete="off" spellcheck="false" placeholder="Επικόλλησε εδώ transcript για προσωρινή εξαγωγή…"></textarea></label>
        <div class="transcript-capture-actions">
          <button class="btn secondary" type="button" data-transcript-discard>Εκκαθάριση</button>
          <button class="btn primary" type="button" data-transcript-submit>Εξαγωγή υποψηφίων</button>
        </div>
        <div class="transcript-status" data-transcript-status role="status" aria-live="polite"></div>
        <div data-transcript-results></div>
      </div>`;
    document.body.appendChild(dialog);
    textarea = dialog.querySelector("[data-transcript-input]");
    statusNode = dialog.querySelector("[data-transcript-status]");
    resultsNode = dialog.querySelector("[data-transcript-results]");
    submitButton = dialog.querySelector("[data-transcript-submit]");
    dialog.querySelector("[data-transcript-close]").addEventListener("click", closeAndClear);
    dialog.querySelector("[data-transcript-discard]").addEventListener("click", clearState);
    submitButton.addEventListener("click", submitTranscript);
    dialog.addEventListener("cancel", (event) => { event.preventDefault(); closeAndClear(); });
    dialog.addEventListener("close", clearState);
    return dialog;
  }

  function clearState() {
    candidates = [];
    if (textarea) textarea.value = "";
    if (statusNode) { statusNode.textContent = ""; statusNode.dataset.state = ""; }
    if (resultsNode) resultsNode.replaceChildren();
  }

  function closeAndClear() {
    clearState();
    if (dialog?.open) dialog.close();
  }

  function text(node, value) {
    node.textContent = String(value ?? "");
    return node;
  }

  function renderCandidate(candidate, index) {
    const article = document.createElement("article");
    article.className = "transcript-candidate";
    const heading = document.createElement("h3");
    text(heading, `${index + 1}. ${candidate.semantic_type || "candidate"}`);
    article.appendChild(heading);
    const dl = document.createElement("dl");
    const add = (label, value, className = "") => {
      const dt = document.createElement("dt"); text(dt, label);
      const dd = document.createElement("dd"); text(dd, value); if (className) dd.className = className;
      dl.append(dt, dd);
    };
    add("Speaker", candidate.source_assertion?.speaker || "unclear");
    add("Polarity", candidate.source_assertion?.polarity || "unclear");
    add("Temporality", candidate.source_assertion?.temporality || "unclear");
    add("Evidence", candidate.evidence_snippet || "—");
    const concepts = (candidate.components || []).map((item) => item.concept_key).join(", ") || "—";
    add("Concepts", concepts);
    const targets = (candidate.target_mappings || []).map((item) => `${item.status}: ${item.target_path || item.reason_code}`).join(" | ") || "unmapped";
    add("Targets", targets, "transcript-target");
    add("Warnings", (candidate.warnings || []).join(", ") || "—");
    article.appendChild(dl);
    return article;
  }

  function renderResults(response) {
    candidates = Array.isArray(response?.candidates) ? response.candidates : [];
    resultsNode.replaceChildren();
    if (!candidates.length) {
      const empty = document.createElement("p");
      text(empty, "Δεν εντοπίστηκαν δομημένα υποψήφια δεδομένα.");
      resultsNode.appendChild(empty);
      return;
    }
    candidates.forEach((candidate, index) => resultsNode.appendChild(renderCandidate(candidate, index)));
  }

  function safeErrorCode(payload) {
    const code = payload?.detail?.code;
    return typeof code === "string" ? code : "REQUEST_FAILED";
  }

  async function submitTranscript() {
    const transcript = String(textarea?.value || "").trim();
    if (!transcript) {
      statusNode.textContent = "Το transcript είναι κενό.";
      statusNode.dataset.state = "error";
      return;
    }
    submitButton.disabled = true;
    statusNode.textContent = "Γίνεται προσωρινή εξαγωγή…";
    statusNode.dataset.state = "busy";
    resultsNode.replaceChildren();
    candidates = [];
    try {
      const response = await fetch(ENDPOINT, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          schema_version: "clinical_transcript_extract_request_v1",
          source_type: "heidi_transcript",
          module: "osteoporosis",
          encounter_phase: "during_visit",
          language: "el",
          transcript,
          context: { encounter_archetype: document.querySelector("#encounterArchetype")?.value || null }
        })
      });
      let payload = null;
      try { payload = await response.json(); } catch { payload = null; }
      if (!response.ok) throw new Error(safeErrorCode(payload));
      textarea.value = "";
      renderResults(payload);
      statusNode.textContent = `Ολοκληρώθηκε: ${candidates.length} υποψήφια. Τίποτε δεν γράφτηκε στον κλινικό φάκελο.`;
      statusNode.dataset.state = "ok";
    } catch (error) {
      statusNode.textContent = `Η εξαγωγή δεν ολοκληρώθηκε (${error?.message || "REQUEST_FAILED"}). Το transcript παραμένει μόνο για άμεση διόρθωση/επανάληψη.`;
      statusNode.dataset.state = "error";
    } finally {
      submitButton.disabled = false;
    }
  }

  function openDialog() {
    makeDialog();
    clearState();
    dialog.showModal();
    textarea.focus();
  }

  addStyles();
  makeDialog();
  document.addEventListener("click", (event) => {
    const heidi = event.target.closest('[data-nav-action="heidi"]');
    if (heidi) { event.preventDefault(); openDialog(); return; }
    if (dialog?.open && event.target.closest('.side-item:not([data-nav-action="heidi"])')) closeAndClear();
  });
  window.addEventListener("pagehide", clearState);
  window.addEventListener("pageshow", clearState);
})();
