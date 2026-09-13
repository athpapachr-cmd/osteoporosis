(() => {
  "use strict";

  const API_BASE = "/clinical/clinic-utilities/medical-report";
  const $ = (id) => document.getElementById(id);
  const state = { draft: null, pending: null, researchStale: false };
  const nativeFetch = window.fetch.bind(window);

  function requestUrl(input) {
    return typeof input === "string" ? input : String(input?.url || "");
  }

  function displayDate(value) {
    const parts = String(value || "").split("-");
    return parts.length === 3 ? `${parts[2]}/${parts[1]}/${parts[0]}` : value;
  }

  window.fetch = async (input, init = {}) => {
    const url = requestUrl(input);
    let nextInit = init;
    if (state.draft && url.endsWith("/api/research") && typeof init.body === "string") {
      try {
        const payload = JSON.parse(init.body);
        payload.analysis = state.draft.analysis;
        nextInit = { ...init, body: JSON.stringify(payload) };
      } catch (_) {}
    }
    if (state.researchStale && (url.endsWith("/api/preview") || url.endsWith("/api/pdf")) && init.body instanceof FormData) {
      try {
        const raw = init.body.get("report_json");
        const payload = JSON.parse(String(raw || "{}"));
        payload.research_text = "";
        payload.citations = [];
        init.body.set("report_json", JSON.stringify(payload));
      } catch (_) {}
    }
    const response = await nativeFetch(input, nextInit);
    if (response.ok && url.endsWith("/api/analyze")) {
      response.clone().json().then((body) => {
        state.draft = body;
        state.pending = null;
        state.researchStale = false;
        setTimeout(() => {
          ensureUi();
          annotateVisualSources();
        }, 80);
      }).catch(() => {});
    }
    if (response.ok && url.endsWith("/api/research")) state.researchStale = false;
    return response;
  };

  function annotateVisualSources() {
    if (!state.draft) return;
    const boxes = Array.from(document.querySelectorAll("#sourcesPanel .source-item"));
    (state.draft.sources || []).forEach((source, index) => {
      const box = boxes[index];
      if (!box || !source.review_required || box.querySelector(".v11-vision-badge")) return;
      const badge = document.createElement("div");
      badge.className = "review-badge v11-vision-badge";
      badge.textContent = "Οπτική ανάγνωση AI · απαιτεί έλεγχο";
      box.append(badge);
    });
  }

  function syncEditedSections() {
    if (!state.draft) return;
    (state.draft.analysis.report_sections || []).forEach((section, index) => {
      const textarea = document.querySelector(`textarea[data-section-index="${index}"]`);
      if (textarea) section.draft_text = textarea.value.trim();
    });
  }

  function appendChat(role, text) {
    const root = $("v11ChatLog");
    if (!root) return;
    const box = document.createElement("div");
    box.className = `v11-chat-message ${role}`;
    const title = document.createElement("strong");
    title.textContent = role === "clinician" ? "Ιατρός" : "AI";
    const body = document.createElement("p");
    body.textContent = text;
    box.append(title, body);
    root.append(box);
    root.scrollTop = root.scrollHeight;
  }

  function renderSections() {
    if (!state.draft) return;
    (state.draft.analysis.report_sections || []).forEach((section, index) => {
      const textarea = document.querySelector(`textarea[data-section-index="${index}"]`);
      if (textarea) textarea.value = section.draft_text || "";
    });
  }

  function renderTimeline() {
    const root = $("timelinePanel");
    if (!root || !state.draft) return;
    root.replaceChildren();
    const heading = document.createElement("h2");
    heading.textContent = "Χρονολογική πορεία";
    root.append(heading);
    (state.draft.analysis.timeline || []).forEach((item) => {
      const box = document.createElement("div");
      box.className = `timeline-item${(item.conflict_flags || []).length ? " conflict" : ""}`;
      const title = document.createElement("strong");
      title.textContent = `${item.event_date ? displayDate(item.event_date) : item.date_text || "Χωρίς ακριβή ημερομηνία"} · ${item.title}`;
      const body = document.createElement("p");
      body.textContent = item.summary;
      box.append(title, body);
      root.append(box);
    });
  }

  function renderResolutions() {
    document.getElementById("v11ResolutionBlock")?.remove();
    const root = $("evidencePanel");
    const items = state.draft?.analysis?.clinician_resolutions || [];
    if (!root || !items.length) return;
    const block = document.createElement("div");
    block.id = "v11ResolutionBlock";
    block.className = "v11-resolutions";
    const heading = document.createElement("h3");
    heading.textContent = "Επιβεβαιωμένες διευκρινίσεις ιατρού";
    block.append(heading);
    items.forEach((item) => {
      const p = document.createElement("p");
      p.textContent = `${item.topic_or_conflict_key || "Διευκρίνιση"}: ${item.clinician_statement}`;
      block.append(p);
    });
    root.append(block);
  }

  function markResolvedWarnings() {
    const keys = new Set((state.draft?.analysis?.clinician_resolutions || []).map((x) => x.topic_or_conflict_key).filter(Boolean));
    document.querySelectorAll("#warningList .warning-chip").forEach((chip) => {
      for (const key of keys) {
        if (!chip.textContent.includes(key)) continue;
        chip.classList.add("v11-resolved-warning");
        if (!chip.textContent.startsWith("✓")) chip.textContent = `✓ Διευκρινίστηκε · ${chip.textContent}`;
        break;
      }
    });
  }

  function applyPending() {
    if (!state.pending || !state.draft) return;
    state.draft.analysis = state.pending.updated_analysis;
    state.draft.usage = state.pending.usage || state.draft.usage;
    state.pending = null;
    state.researchStale = true;
    $("v11Proposal").hidden = true;
    renderSections();
    renderTimeline();
    renderResolutions();
    markResolvedWarnings();
    const confirmed = $("clinicianConfirmed");
    if (confirmed) confirmed.checked = false;
    const research = $("researchResult");
    if (research) research.hidden = true;
    if ($("researchText")) $("researchText").value = "";
    if ($("citationList")) $("citationList").replaceChildren();
    if ($("summaryResearch")) $("summaryResearch").textContent = "Απαιτεί νέο έλεγχο";
  }

  async function sendRefinement() {
    if (!state.draft) return;
    const input = $("v11RefinementMessage");
    const message = input?.value.trim();
    if (!message) return;
    syncEditedSections();
    appendChat("clinician", message);
    input.value = "";
    const button = $("v11RefineButton");
    button.disabled = true;
    try {
      const response = await nativeFetch(`${API_BASE}/api/refine`, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case: state.draft.case, analysis: state.draft.analysis, clinician_message: message }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${response.status}`);
      }
      state.pending = await response.json();
      appendChat("assistant", state.pending.assistant_reply || "Η διευκρίνιση ολοκληρώθηκε.");
      const count = (state.pending.proposed_resolutions || []).length;
      $("v11ProposalText").textContent = count
        ? `Προτείνονται ${count} επιβεβαιωμένες διευκρινίσεις και αναθεώρηση των επηρεαζόμενων τμημάτων.`
        : "Υπάρχει προτεινόμενη αναθεώρηση για έλεγχο.";
      $("v11Proposal").hidden = false;
    } catch (error) {
      appendChat("assistant", `Αποτυχία διευκρίνισης: ${error.message}`);
    } finally {
      button.disabled = false;
    }
  }

  function ensureUi() {
    const workspace = $("analysisWorkspace");
    if (!workspace || $("v11RefinementCard")) return;
    const card = document.createElement("section");
    card.id = "v11RefinementCard";
    card.className = "card v11-refinement-card";
    card.innerHTML = `
      <div class="section-head"><div><p class="step">04B</p><h2>Διευκρινίσεις με το AI</h2></div><span class="review-badge">Session-only</span></div>
      <p class="microcopy">Διόρθωσε λάθος ημερομηνία άλλης πηγής, δήλωσε ότι μια εξέταση εκκρεμεί ή κάνε ερώτηση. Η αρχική πηγή δεν αλλάζει.</p>
      <div id="v11ChatLog" class="v11-chat-log"></div>
      <label class="field"><span>Μήνυμα προς το AI</span><textarea id="v11RefinementMessage" rows="4" placeholder="π.χ. Η σωστή ημερομηνία είναι ... Η άλλη ημερομηνία είναι λάθος αναδρομικής καταχώρισης και η πηγή πρέπει να παραμείνει αυτούσια."></textarea></label>
      <div class="actions"><button id="v11RefineButton" class="button secondary" type="button">Συζήτηση / διευκρίνιση</button></div>
      <div id="v11Proposal" class="notice" hidden><p id="v11ProposalText"></p><div class="actions"><button id="v11Apply" class="button primary" type="button">Εφαρμογή αλλαγών</button><button id="v11Discard" class="button quiet" type="button">Απόρριψη</button></div></div>`;
    const prognosisCard = Array.from(workspace.querySelectorAll(":scope > .card")).find((node) => node.textContent.includes("Πρόγνωση & βιβλιογραφία"));
    workspace.insertBefore(card, prognosisCard || workspace.lastElementChild);
    $("v11RefineButton").addEventListener("click", sendRefinement);
    $("v11Apply").addEventListener("click", applyPending);
    $("v11Discard").addEventListener("click", () => { state.pending = null; $("v11Proposal").hidden = true; });
  }

  window.MedicalReportV11Refine = Object.freeze({ getDraft: () => state.draft, ensureUi });
})();
