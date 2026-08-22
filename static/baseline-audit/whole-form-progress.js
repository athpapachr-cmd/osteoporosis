(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  const explicit = (value) => value !== "" && value !== null && value !== undefined;
  const val = (selector) => $(selector)?.value ?? "";
  const choice = (field) => Boolean($(`[data-field="${field}"].selected`));
  const checkedAny = (selector) => $$(selector).some((node) => node.checked);

  function cardActive(anchor) {
    const node = $(anchor);
    if (!node) return false;
    const card = node.closest("article.card");
    if (!card) return true;
    return !card.classList.contains("adaptive-collapsed");
  }

  function add(items, applicable, complete) {
    if (applicable) items.push(Boolean(complete));
  }

  function calculateWholeFormProgress() {
    const items = [];

    // Step 1 — invariant encounter metadata / capture context.
    add(items, true, explicit(val("#encounterDate")));
    add(items, true, explicit(val("#ageYears")));
    add(items, true, choice("sex"));
    add(items, true, choice("patient_relationship_status"));
    add(items, true, explicit(val("#encounterArchetype")));
    add(items, true, choice("first_core_baseline_encounter_for_patient"));
    add(items, true, choice("osteoporosis_status"));
    add(items, true, choice("heidi_used"));
    if ($('[data-field="sex"][data-value="female"].selected')) add(items, true, choice("menopause_status"));
    if ($('[data-field="heidi_used"][data-value="yes"].selected')) {
      add(items, true, explicit(val("#heidiOutput")));
      add(items, true, explicit(val("#heidiReviewed")));
    }

    // Step 2 — only domains left active by adaptive applicability enter the denominator.
    add(items, cardActive("#fractureHistoryReviewed"), explicit(val("#fractureHistoryReviewed")) && explicit(val("#fractureReviewScope")));
    const formalRiskActive = cardActive("#formalRiskIndicated");
    add(items, formalRiskActive, explicit(val("#formalRiskIndicated")));
    if (formalRiskActive && val("#formalRiskIndicated") === "yes") {
      add(items, true, explicit(val("#formalRiskDone")));
      if (val("#formalRiskDone") === "yes") {
        add(items, true, explicit(val("#riskToolName")) && explicit(val("#declaredRiskFramework")) && explicit(val("#fraxMof")) && explicit(val("#fraxHip")));
      }
    }

    // Step 3 — completion markers, not quality judgements.
    add(items, cardActive("#s3DxaUsed"), explicit(val("#s3DxaUsed")));
    add(items, cardActive("#s3VfaIndicated"), explicit(val("#s3VfaIndicated")));
    add(items, cardActive("#s3SecondaryIndicated"), explicit(val("#s3SecondaryIndicated")));
    add(items, cardActive("#s3Ca"), explicit(val("#s3LabsReviewed")) || ["#s3Ca", "#s3VitD", "#s3Creat", "#s3Egfr"].some((s) => explicit(val(s))));
    add(items, cardActive("#s3FallsReviewed"), explicit(val("#s3FallsReviewed")) && explicit(val("#s3FunctionReviewed")));
    add(items, cardActive("#s3SarcApplicable"), explicit(val("#s3SarcApplicable")));

    // Step 4.
    add(items, cardActive("#s4AddEpisode"), Boolean($("#s4Episodes .repeat-item, #s4Episodes [data-episode-id]")) || explicit(val("#s4DecisionType")));
    add(items, cardActive("#s4AddAdministration"), Boolean($("#s4Administrations .repeat-item, #s4Administrations [data-administration-id]")) || explicit(val("#s4DecisionType")));
    add(items, cardActive("#s4DecisionType"), explicit(val("#s4DecisionType")) && explicit(val("#s4SafetyReview")) && explicit(val("#s4SequencingReview")));
    add(items, cardActive("#s4TransitionRelevant"), explicit(val("#s4TransitionRelevant")));
    add(items, cardActive("#s4PlanComplete"), explicit(val("#s4PlanComplete")) && explicit(val("#s4CriticalUnresolved")));

    // Step 5.
    add(items, cardActive("#s5ConditionRisk"), ["#s5ConditionRisk", "#s5ResultsStatus", "#s5MedicationPlan", "#s5Questions", "#s5Preferences"].every((s) => explicit(val(s))));
    add(items, cardActive("#s5UnderstandCondition"), explicit(val("#s5UnderstandPlan")) && explicit(val("#s5InformationGiven")));
    add(items, cardActive("#s5WentWell"), explicit(val("#s5MissedUncertain")));

    // Step 6 — provenance/capture quality is always part of pilot completion.
    add(items, cardActive("#s6Sources"), checkedAny("#s6Sources input[type=checkbox]") && explicit(val("#s6PrimarySource")));
    add(items, cardActive("#s6GesyAvailable"), explicit(val("#s6GesyAvailable")) && explicit(val("#s6GesyStatus")));
    add(items, cardActive("#s6HeidiFinalNote"), explicit(val("#s6HeidiFinalNote")));
    add(items, cardActive("#s6Reliability"), explicit(val("#s6Reliability")) && explicit(val("#s6MajorGap")) && explicit(val("#s6ReadyForAudit")));

    if (!items.length) return 0;
    return Math.round((items.filter(Boolean).length / items.length) * 100);
  }

  function render() {
    const fill = $("#progressFill");
    const text = $("#progressText");
    const label = $(".meta-progress .meta-label");
    if (!fill || !text) return;
    if (label) {
      label.textContent = "Συνολική συμπλήρωση";
      label.title = "Capture completion only — όχι KPI/performance score";
    }
    const progress = calculateWholeFormProgress();
    fill.style.width = `${progress}%`;
    text.textContent = `${progress}%`;
    text.title = "Capture completion only — όχι KPI/performance score";
  }

  document.addEventListener("input", () => setTimeout(render, 0));
  document.addEventListener("change", () => setTimeout(render, 0));
  document.addEventListener("click", (event) => {
    if (event.target.closest(".step-tab, [data-field][data-value], [data-load-case], [data-nav-action=\"new-case\"], .adaptive-use-domain, .adaptive-reset-domain, #saveTopBtn, #saveDraftBtn, #finishVisitBtn")) {
      setTimeout(render, 30);
    }
  });

  window.addEventListener("storage", render);
  setTimeout(render, 0);
})();
