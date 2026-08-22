(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  const A = "applicable";
  const U = "uncertain";
  const N = "not_applicable";

  const DOMAIN_LABELS = {
    fracture_history: "Ιστορικό καταγμάτων",
    formal_risk: "Formal fracture risk",
    dxa: "DXA / longitudinal DXA",
    vfa: "VFA / vertebral imaging",
    secondary_causes: "Secondary causes",
    laboratory_monitoring: "Εργαστηριακά / monitoring",
    falls_function: "Πτώσεις / frailty / function",
    sarcopenia: "Σαρκοπενία",
    treatment_history: "Θεραπευτικό ιστορικό",
    administrations: "Administrations / due dates",
    treatment_decision: "Κλινική απόφαση",
    transition_safety: "Transition / sequencing safety",
    followup_tasks: "Follow-up / care tasks",
    communication: "Επικοινωνία",
    understanding: "Κατανόηση / teach-back",
    reflection: "Post-visit reflection",
    documentation_capture: "Documentation / capture"
  };

  const CARD_ANCHORS = {
    fracture_history: ["#fractureHistoryReviewed"],
    formal_risk: ["#formalRiskIndicated", "#declaredRiskFramework", "#longitudinalRiskCard"],
    dxa: ["#s3DxaUsed", "#longitudinalDxaCard"],
    vfa: ["#s3VfaIndicated"],
    secondary_causes: ["#s3SecondaryIndicated"],
    laboratory_monitoring: ["#s3Ca"],
    falls_function: ["#s3FallsReviewed"],
    sarcopenia: ["#s3SarcApplicable"],
    treatment_history: ["#s4AddEpisode"],
    administrations: ["#s4AddAdministration"],
    treatment_decision: ["#s4DecisionType"],
    transition_safety: ["#s4TransitionRelevant"],
    followup_tasks: ["#s4AddTask", "#s4PlanComplete"],
    communication: ["#s5ConditionRisk"],
    understanding: ["#s5UnderstandCondition"],
    reflection: ["#s5WentWell"],
    documentation_capture: ["#s6Sources", "#s6GesyAvailable", "#s6HeidiFinalNote", "#s6Reliability"]
  };

  const MAP = {
    initial_assessment_new_or_uncertain_diagnosis: {
      fracture_history:A, formal_risk:A, dxa:A, vfa:A, secondary_causes:A, laboratory_monitoring:A,
      falls_function:A, sarcopenia:U, treatment_history:U, administrations:N, treatment_decision:A,
      transition_safety:N, followup_tasks:A, communication:A, understanding:A, reflection:A, documentation_capture:A
    },
    initial_assessment_known_osteoporosis_or_osteopenia: {
      fracture_history:A, formal_risk:U, dxa:A, vfa:U, secondary_causes:A, laboratory_monitoring:U,
      falls_function:A, sarcopenia:U, treatment_history:A, administrations:U, treatment_decision:A,
      transition_safety:U, followup_tasks:A, communication:A, understanding:A, reflection:A, documentation_capture:A
    },
    routine_followup_stable: {
      fracture_history:A, formal_risk:U, dxa:U, vfa:N, secondary_causes:N, laboratory_monitoring:U,
      falls_function:A, sarcopenia:U, treatment_history:A, administrations:U, treatment_decision:U,
      transition_safety:N, followup_tasks:A, communication:U, understanding:U, reflection:A, documentation_capture:A
    },
    treatment_start: {
      fracture_history:U, formal_risk:A, dxa:A, vfa:U, secondary_causes:A, laboratory_monitoring:A,
      falls_function:U, sarcopenia:U, treatment_history:A, administrations:U, treatment_decision:A,
      transition_safety:N, followup_tasks:A, communication:A, understanding:A, reflection:A, documentation_capture:A
    },
    treatment_continuation_or_due_monitoring: {
      fracture_history:A, formal_risk:U, dxa:U, vfa:N, secondary_causes:U, laboratory_monitoring:A,
      falls_function:U, sarcopenia:U, treatment_history:A, administrations:A, treatment_decision:A,
      transition_safety:N, followup_tasks:A, communication:A, understanding:U, reflection:A, documentation_capture:A
    },
    treatment_change_or_transition: {
      fracture_history:A, formal_risk:A, dxa:A, vfa:U, secondary_causes:U, laboratory_monitoring:A,
      falls_function:U, sarcopenia:U, treatment_history:A, administrations:A, treatment_decision:A,
      transition_safety:A, followup_tasks:A, communication:A, understanding:A, reflection:A, documentation_capture:A
    },
    post_fragility_fracture: {
      fracture_history:A, formal_risk:A, dxa:A, vfa:A, secondary_causes:A, laboratory_monitoring:A,
      falls_function:A, sarcopenia:U, treatment_history:A, administrations:U, treatment_decision:A,
      transition_safety:U, followup_tasks:A, communication:A, understanding:A, reflection:A, documentation_capture:A
    },
    fracture_on_treatment: {
      fracture_history:A, formal_risk:A, dxa:A, vfa:A, secondary_causes:A, laboratory_monitoring:A,
      falls_function:A, sarcopenia:U, treatment_history:A, administrations:A, treatment_decision:A,
      transition_safety:A, followup_tasks:A, communication:A, understanding:A, reflection:A, documentation_capture:A
    },
    adverse_effect_or_intolerance: {
      fracture_history:U, formal_risk:U, dxa:N, vfa:N, secondary_causes:U, laboratory_monitoring:U,
      falls_function:N, sarcopenia:N, treatment_history:A, administrations:A, treatment_decision:A,
      transition_safety:U, followup_tasks:A, communication:A, understanding:A, reflection:A, documentation_capture:A
    },
    treatment_completion_or_consolidation: {
      fracture_history:A, formal_risk:U, dxa:A, vfa:U, secondary_causes:U, laboratory_monitoring:A,
      falls_function:U, sarcopenia:U, treatment_history:A, administrations:A, treatment_decision:A,
      transition_safety:A, followup_tasks:A, communication:A, understanding:U, reflection:A, documentation_capture:A
    }
  };

  const ALL_DOMAINS = Object.keys(DOMAIN_LABELS);

  function getCases() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch { return []; }
  }
  function setCases(cases) { localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
  function activeUuid() { return localStorage.getItem(ACTIVE_KEY) || ""; }
  function activeCase() { const id = activeUuid(); return getCases().find(c => c.internal_uuid === id) || null; }

  function defaultStatus(archetype, domain) {
    if (!archetype || archetype === "other") return U;
    return MAP[archetype]?.[domain] || U;
  }

  function normalizeReview(raw, archetype) {
    const sameArchetype = raw?.archetype === archetype;
    const domains = {};
    ALL_DOMAINS.forEach(domain => {
      const def = defaultStatus(archetype, domain);
      const previousOverride = sameArchetype && raw?.domains?.[domain]?.override_status === A ? A : "";
      domains[domain] = {
        default_status: def,
        override_status: previousOverride,
        resolved_status: previousOverride === A ? A : def
      };
    });
    return { version: "applicability-v1", archetype, domains, updated_at: new Date().toISOString() };
  }

  function persistReview(review) {
    const id = activeUuid();
    if (!id) return;
    const cases = getCases();
    const i = cases.findIndex(c => c.internal_uuid === id);
    if (i < 0) return;
    cases[i] = { ...cases[i], applicability_review: review };
    setCases(cases);
  }

  function cardsForDomain(domain) {
    const found = [];
    (CARD_ANCHORS[domain] || []).forEach(selector => {
      const node = $(selector);
      const card = node?.matches?.("article.card") ? node : node?.closest?.("article.card");
      if (card && !found.includes(card)) found.push(card);
    });
    return found;
  }

  function ensureControl(card, domain) {
    card.dataset.applicabilityDomain = domain;
    let control = $(".adaptive-applicability-control", card);
    if (!control) {
      control = document.createElement("div");
      control.className = "adaptive-applicability-control";
      const heading = $(".card-heading", card);
      if (heading) heading.appendChild(control); else card.prepend(control);
    }
    return control;
  }

  function setCardState(card, domain, item, archetype) {
    const control = ensureControl(card, domain);
    const isOther = archetype === "other" || !archetype;
    const resolved = item.resolved_status;
    const overridden = item.override_status === A;
    const collapsed = !isOther && !overridden && resolved !== A;

    card.classList.toggle("adaptive-collapsed", collapsed);
    card.classList.toggle("adaptive-uncertain", !isOther && resolved === U && !overridden);
    card.classList.toggle("adaptive-not-applicable", !isOther && resolved === N && !overridden);
    card.classList.toggle("adaptive-overridden", overridden);

    let badgeText = "Case-by-case";
    let badgeClass = "uncertain";
    if (!isOther && resolved === A && !overridden) { badgeText = "Applicable"; badgeClass = "applicable"; }
    if (!isOther && resolved === U && !overridden) { badgeText = "Conditional"; badgeClass = "uncertain"; }
    if (!isOther && resolved === N && !overridden) { badgeText = "Usually N/A"; badgeClass = "na"; }
    if (overridden) { badgeText = "Applicable — override"; badgeClass = "override"; }

    control.innerHTML = `<span class="adaptive-badge ${badgeClass}">${badgeText}</span>${(!isOther && resolved !== A && !overridden) ? `<button type="button" class="adaptive-use-domain" data-domain="${domain}">Χρήση σήμερα</button>` : overridden ? `<button type="button" class="adaptive-reset-domain" data-domain="${domain}">Επαναφορά</button>` : ""}`;
  }

  function updateStepBanner(panel, review) {
    if (!panel) return;
    let banner = $(".adaptive-applicability-banner", panel);
    if (!banner) {
      banner = document.createElement("div");
      banner.className = "adaptive-applicability-banner";
      const context = $(".context-note", panel);
      if (context?.nextSibling) panel.insertBefore(banner, context.nextSibling); else panel.prepend(banner);
    }
    const domainsInPanel = new Set($$("article.card[data-applicability-domain]", panel).map(card => card.dataset.applicabilityDomain));
    let applicable = 0, conditional = 0, notApplicable = 0;
    domainsInPanel.forEach(domain => {
      const s = review.domains[domain]?.resolved_status;
      if (s === A) applicable += 1;
      else if (s === N) notApplicable += 1;
      else conditional += 1;
    });
    if (!review.archetype) {
      banner.textContent = "Επίλεξε encounter archetype στο Step 1 για adaptive applicability. Μέχρι τότε τα domains παραμένουν case-by-case.";
    } else if (review.archetype === "other") {
      banner.textContent = "Encounter type: Other — όλα τα domains παραμένουν ανοικτά και αξιολογούνται case-by-case.";
    } else {
      banner.innerHTML = `<strong>Adaptive applicability:</strong> ${applicable} applicable · ${conditional} conditional · ${notApplicable} usually N/A. Τα collapsed domains μπορούν να ενεργοποιηθούν για τη συγκεκριμένη επίσκεψη.`;
    }
  }

  function apply() {
    const c = activeCase();
    if (!c) return;
    const archetype = $("#encounterArchetype")?.value || c.encounter_archetype || "";
    const review = normalizeReview(c.applicability_review, archetype);

    ALL_DOMAINS.forEach(domain => {
      cardsForDomain(domain).forEach(card => setCardState(card, domain, review.domains[domain], archetype));
    });
    [2,3,4,5,6].forEach(step => updateStepBanner($(`[data-step-panel="${step}"]`), review));
    persistReview(review);
  }

  function overrideDomain(domain, enabled) {
    const c = activeCase();
    if (!c || !ALL_DOMAINS.includes(domain)) return;
    const archetype = $("#encounterArchetype")?.value || c.encounter_archetype || "";
    const review = normalizeReview(c.applicability_review, archetype);
    review.domains[domain].override_status = enabled ? A : "";
    review.domains[domain].resolved_status = enabled ? A : review.domains[domain].default_status;
    review.updated_at = new Date().toISOString();
    persistReview(review);
    apply();
  }

  document.addEventListener("click", event => {
    const use = event.target.closest(".adaptive-use-domain");
    const reset = event.target.closest(".adaptive-reset-domain");
    if (use) overrideDomain(use.dataset.domain, true);
    if (reset) overrideDomain(reset.dataset.domain, false);
    if (event.target.closest("[data-load-case]") || event.target.closest('[data-nav-action="new-case"]')) setTimeout(apply, 30);
  });

  $("#encounterArchetype")?.addEventListener("change", () => {
    const id = activeUuid();
    if (id) {
      const cases = getCases();
      const i = cases.findIndex(c => c.internal_uuid === id);
      if (i >= 0 && cases[i].applicability_review) {
        cases[i] = { ...cases[i], applicability_review: { ...cases[i].applicability_review, archetype: "", domains: {} } };
        setCases(cases);
      }
    }
    setTimeout(apply, 0);
  });

  $$(".step-tab").forEach(button => button.addEventListener("click", () => setTimeout(apply, 0)));
  ["#saveTopBtn", "#saveDraftBtn", "#finishVisitBtn"].forEach(selector => $(selector)?.addEventListener("click", () => setTimeout(apply, 0)));

  if (!document.querySelector('link[data-adaptive-applicability-style]')) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "./adaptive-applicability.css";
    link.dataset.adaptiveApplicabilityStyle = "true";
    document.head.appendChild(link);
  }

  setTimeout(apply, 0);
})();
