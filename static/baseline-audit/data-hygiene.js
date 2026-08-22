(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  const DXA_DETAIL_IDS = [
    "#s3SpineBmd", "#s3SpineT", "#s3TotalHipBmd", "#s3TotalHipT", "#s3FnBmd", "#s3FnT",
    "#s3ZScoreUsed", "#s3RoiIssue", "#s3Artifact", "#s3Longitudinal"
  ];
  const DXA_LONGITUDINAL_IDS = [
    "#s3ComparisonDate", "#s3ComparableMachine", "#s3LscKnown", "#s3SpineLsc", "#s3HipLsc", "#s3ChangeValid"
  ];
  const TRANSITION_IDS = [
    "#s4TransitionType", "#s4PriorAgentEnd", "#s4NextAgent", "#s4NextAgentDate", "#s4TransitionExplicit", "#s4SafetyUnresolved", "#s4SafetyNote"
  ];

  function getCases() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function setCases(cases) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cases));
  }

  function clearValue(selector) {
    const node = $(selector);
    if (node) node.value = "";
  }

  function sanitizeDom() {
    const dxaUsed = $("#s3DxaUsed")?.value || "";
    if (dxaUsed !== "yes") {
      DXA_DETAIL_IDS.forEach(clearValue);
      DXA_LONGITUDINAL_IDS.forEach(clearValue);
    } else if (($("#s3Longitudinal")?.value || "") !== "yes") {
      DXA_LONGITUDINAL_IDS.forEach(clearValue);
    }

    if (($("#s4TransitionRelevant")?.value || "") !== "yes") {
      TRANSITION_IDS.forEach(clearValue);
    }

    if (($("#s5InformationGiven")?.value || "") !== "yes") {
      $$("#s5InfoTypes input[type=\"checkbox\"]").forEach((node) => { node.checked = false; });
    }
    if (($("#s5Misunderstanding")?.value || "") !== "yes") {
      clearValue("#s5Corrected");
    }
  }

  function sanitizeCase(caseItem) {
    let changed = false;

    const dxa = caseItem?.step3?.dxa;
    if (dxa && dxa.used !== "yes") {
      Object.assign(dxa, {
        spine_bmd: null, spine_t: null, total_hip_bmd: null, total_hip_t: null,
        femoral_neck_bmd: null, femoral_neck_t: null, z_score_used: "", roi_issue: "",
        artifact: "", longitudinal: "", comparison_date: "", comparable_machine: "",
        lsc_known: "", spine_lsc_percent: null, hip_lsc_percent: null, change_valid: ""
      });
      changed = true;
    } else if (dxa && dxa.longitudinal !== "yes") {
      Object.assign(dxa, {
        comparison_date: "", comparable_machine: "", lsc_known: "",
        spine_lsc_percent: null, hip_lsc_percent: null, change_valid: ""
      });
      changed = true;
    }

    const transition = caseItem?.step4?.transition;
    if (transition && transition.relevant !== "yes") {
      Object.assign(transition, {
        type: "", prior_end_date: "", next_agent: "", next_agent_date: "",
        explicit_plan: "", unresolved_safety: "", note: ""
      });
      changed = true;
    }

    const understanding = caseItem?.step5?.understanding;
    if (understanding) {
      if (understanding.information_given !== "yes" && Array.isArray(understanding.information_types) && understanding.information_types.length) {
        understanding.information_types = [];
        changed = true;
      }
      if (understanding.misunderstanding_detected !== "yes" && understanding.misunderstanding_corrected) {
        understanding.misunderstanding_corrected = "";
        changed = true;
      }
    }

    return changed;
  }

  function sanitizeActiveStoredCase() {
    const id = localStorage.getItem(ACTIVE_KEY) || "";
    if (!id) return;
    const cases = getCases();
    const index = cases.findIndex((item) => item.internal_uuid === id);
    if (index < 0) return;
    if (sanitizeCase(cases[index])) setCases(cases);
  }

  function sanitizeNow() {
    sanitizeDom();
    sanitizeActiveStoredCase();
  }

  // Clean any legacy stale values already present when this module is loaded.
  sanitizeNow();

  // Capture phase runs before the individual step modules' bubble handlers, so
  // hidden child values are cleared before those modules collect/persist state.
  document.addEventListener("change", (event) => {
    const id = event.target?.id || "";
    if (["s3DxaUsed", "s3Longitudinal", "s4TransitionRelevant", "s5InformationGiven", "s5Misunderstanding"].includes(id)) {
      sanitizeDom();
      queueMicrotask(sanitizeActiveStoredCase);
    }
  }, true);

  // Save/finish must never re-persist hidden stale values from an older draft.
  ["#saveTopBtn", "#saveDraftBtn", "#finishVisitBtn"].forEach((selector) => {
    $(selector)?.addEventListener("click", sanitizeDom, true);
  });

  // Loading another case can hydrate legacy stale data; sanitize immediately after.
  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-load-case]") || event.target.closest('[data-nav-action="new-case"]')) {
      setTimeout(sanitizeNow, 0);
    }
  }, true);
})();
