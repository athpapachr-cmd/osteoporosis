(() => {
  "use strict";

  const ACTIVE_PATIENT_KEY = "osteoporosis.clinical.activePatient.v1";
  const ACTIVE_CASE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const HIGH_VALUE_REASONS = new Set(["NEW_EVENT", "UNRESOLVED_PRIOR", "EXPLICIT_DUE_STATE", "TREATMENT_CONTEXT"]);

  let baselineKey = "";
  let previousTokens = null;
  let activeNewTokens = new Set();
  let reconcileTimer = null;

  function activePatientId() {
    return sessionStorage.getItem(ACTIVE_PATIENT_KEY) || "";
  }

  function activeCaseId() {
    return localStorage.getItem(ACTIVE_CASE_KEY) || "";
  }

  function currentKey() {
    return `${activePatientId() || "local"}|${activeCaseId() || "no-case"}`;
  }

  function ensureSummaryRoot() {
    let root = document.querySelector("#patientLongitudinalSummary");
    if (root) return root;
    root = document.createElement("section");
    root.id = "patientLongitudinalSummary";
    root.className = "patient-longitudinal-summary";
    const flow = document.querySelector("#progressiveGuidanceSummary") || document.querySelector(".step-tabs");
    if (flow?.parentNode) flow.parentNode.insertBefore(root, flow);
    return root;
  }

  function renderNoPatientPlaceholder() {
    if (activePatientId()) return;
    const root = ensureSummaryRoot();
    const hasMeaningfulContent = root.querySelector(".patient-summary-head") && root.textContent.includes("Σύνοψη ασθενούς");
    if (hasMeaningfulContent && !root.hidden) return;

    root.hidden = false;
    root.innerHTML = "";

    const head = document.createElement("div");
    head.className = "patient-summary-head";
    const title = document.createElement("h2");
    title.textContent = "Σύνοψη ασθενούς";
    const badge = document.createElement("span");
    badge.className = "patient-summary-readonly";
    badge.textContent = "Read-only longitudinal";
    head.append(title, badge);

    const status = document.createElement("div");
    status.className = "progressive-guidance-meta";
    status.textContent = "Δεν έχει επιλεγεί protected patient. Άνοιξε ασθενή από το Patient Registry για να εμφανιστεί η longitudinal σύνοψη από τις προηγούμενες ολοκληρωμένες/τροποποιημένες επισκέψεις.";

    root.append(head, status);
  }

  function itemTokens(item) {
    const domain = String(item?.card_id || "").trim();
    if (!domain) return [];
    const tokens = [];
    (Array.isArray(item?.evidence_rules) ? item.evidence_rules : []).forEach(rule => {
      const ruleId = String(rule?.rule_id || "").trim();
      if (ruleId) tokens.push(`E|${domain}|${ruleId}`);
    });
    (Array.isArray(item?.reason_codes) ? item.reason_codes : []).forEach(reason => {
      if (HIGH_VALUE_REASONS.has(reason)) tokens.push(`R|${domain}|${reason}`);
    });
    return tokens;
  }

  function tokenDomain(token) {
    return String(token || "").split("|")[1] || "";
  }

  function deriveTokenState(plan) {
    const items = Array.isArray(plan?.ordered_cards) ? plan.ordered_cards : [];
    const tokens = new Set(items.flatMap(itemTokens));
    const domains = new Set(items.map(item => String(item?.card_id || "").trim()).filter(Boolean));
    return { tokens, domains };
  }

  function newBadge() {
    const badge = document.createElement("span");
    badge.className = "progressive-guidance-new-badge g3-visibility-guard-badge";
    badge.textContent = "Νέο";
    badge.setAttribute("aria-label", "Νέα guidance ένδειξη");
    return badge;
  }

  function clearGuardDecoration() {
    document.querySelectorAll(".g3-visibility-guard-badge").forEach(node => node.remove());
    document.querySelectorAll(".g3-visibility-guard-new").forEach(node => {
      node.classList.remove("g3-visibility-guard-new", "is-newly-surfaced");
    });
  }

  function decorateDomains(domains) {
    clearGuardDecoration();
    domains.forEach(domain => {
      document.querySelectorAll(`.progressive-guidance-item`).forEach(item => {
        const name = item.querySelector("strong");
        const text = String(name?.firstChild?.textContent || name?.textContent || "").trim();
        const domainNode = document.querySelector(`article.card[data-guidance-domain="${CSS.escape(domain)}"]`);
        const domainLabel = domainNode?.dataset?.guidanceDomain || "";
        if (domainLabel === domain && !item.querySelector(".progressive-guidance-new-badge")) {
          // The top list is ordered identically to the plan; match by current guidance domain label when possible below.
        }
      });

      document.querySelectorAll(`article.card[data-guidance-domain="${CSS.escape(domain)}"]`).forEach(card => {
        card.classList.add("g3-visibility-guard-new", "is-newly-surfaced");
        const why = card.querySelector(":scope > .progressive-why-now");
        if (why && !why.querySelector(".progressive-guidance-new-badge")) why.appendChild(newBadge());
      });
    });

    const plan = window.ProgressiveGuidanceUI?.getLastPlan?.();
    const items = Array.isArray(plan?.ordered_cards) ? plan.ordered_cards : [];
    const topItems = Array.from(document.querySelectorAll("#progressiveGuidanceSummary .progressive-guidance-item"));
    items.slice(0, topItems.length).forEach((item, index) => {
      if (!domains.has(String(item?.card_id || ""))) return;
      const box = topItems[index];
      box.classList.add("g3-visibility-guard-new", "is-newly-surfaced");
      const name = box.querySelector("strong");
      if (name && !name.querySelector(".progressive-guidance-new-badge")) name.appendChild(newBadge());
    });
  }

  function reconcileSalience() {
    const plan = window.ProgressiveGuidanceUI?.getLastPlan?.();
    if (!plan) return;

    const key = currentKey();
    const { tokens, domains } = deriveTokenState(plan);
    if (baselineKey !== key || previousTokens === null) {
      baselineKey = key;
      previousTokens = tokens;
      activeNewTokens = new Set();
      clearGuardDecoration();
      return;
    }

    tokens.forEach(token => {
      if (!previousTokens.has(token)) activeNewTokens.add(token);
    });
    Array.from(activeNewTokens).forEach(token => {
      const domain = tokenDomain(token);
      if (!tokens.has(token) || !domains.has(domain)) activeNewTokens.delete(token);
    });
    previousTokens = tokens;

    decorateDomains(new Set(Array.from(activeNewTokens).map(tokenDomain).filter(Boolean)));
  }

  function scheduleReconcile(delay = 140) {
    if (reconcileTimer) clearTimeout(reconcileTimer);
    reconcileTimer = setTimeout(() => {
      reconcileTimer = null;
      renderNoPatientPlaceholder();
      reconcileSalience();
    }, delay);
  }

  const root = ensureSummaryRoot();
  if (typeof MutationObserver !== "undefined") {
    const observer = new MutationObserver(() => {
      if (!activePatientId() && (root.hidden || !root.textContent.includes("Σύνοψη ασθενούς"))) renderNoPatientPlaceholder();
    });
    observer.observe(root, { attributes: true, childList: true, subtree: true, attributeFilter: ["hidden"] });
  }

  document.addEventListener("input", () => scheduleReconcile());
  document.addEventListener("change", () => scheduleReconcile());
  document.addEventListener("click", event => {
    if (event.target.closest?.("[data-field][data-value], [data-nav-action], #clinicalSearchResults .btn, #clinicalCreatePatientBtn")) scheduleReconcile(220);
  });

  renderNoPatientPlaceholder();
  scheduleReconcile(260);
})();
