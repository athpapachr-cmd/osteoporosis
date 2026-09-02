(() => {
  "use strict";

  const ACTIVE_PATIENT_KEY = "osteoporosis.clinical.activePatient.v1";
  const ACTIVE_CASE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";

  let baselineKey = "";
  let previousTokens = null;
  let retainedNewTokens = [];
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

  function decorateDomains(domains, plan) {
    clearGuardDecoration();
    domains.forEach(domain => {
      document.querySelectorAll(`article.card[data-guidance-domain="${CSS.escape(domain)}"]`).forEach(card => {
        card.classList.add("g3-visibility-guard-new", "is-newly-surfaced");
        const why = card.querySelector(":scope > .progressive-why-now");
        if (why && !why.querySelector(".progressive-guidance-new-badge")) why.appendChild(newBadge());
      });
    });

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
    const core = window.G3SalienceTokenCore;
    if (!plan || !core) return;

    const key = currentKey();
    const initialize = baselineKey !== key || previousTokens === null;
    const state = core.advance({
      previousTokens,
      retainedNewTokens,
      items: Array.isArray(plan.ordered_cards) ? plan.ordered_cards : [],
      initialize
    });

    baselineKey = key;
    previousTokens = state.current_tokens;
    retainedNewTokens = state.retained_new_tokens;
    decorateDomains(new Set(state.newly_surfaced_domains), plan);
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
