(() => {
  "use strict";

  const preloadGuard = document.createElement("style");
  preloadGuard.id = "clinicalWorkspacePreloadGuard";
  preloadGuard.textContent = `
    #pilotPill,
    #caseIdDisplay,
    #privacyStrip,
    [data-nav-action="new-case"],
    [data-nav-action="cases"] { visibility: hidden; }
    .sampling-box { display: none !important; }
  `;
  document.head.appendChild(preloadGuard);

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src;
      script.async = false;
      script.onload = resolve;
      script.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(script);
    });
  }

  loadScript("./app-core.js")
    .then(() => loadScript("./clinical-workspace-shell.js"))
    .then(() => {
      document.querySelector("#clinicalWorkspacePreloadGuard")?.remove();
      return loadScript("./bmi-behavior.js");
    })
    .then(() => loadScript("./step3.js"))
    .then(() => loadScript("./dxa-machine-select.js"))
    .then(() => loadScript("./shared-risk-source.js"))
    .then(() => loadScript("./step4.js"))
    .then(() => loadScript("./data-hygiene.js"))
    .then(() => loadScript("./prior-dxa-inline.js"))
    .then(() => loadScript("./longitudinal.js"))
    .then(() => loadScript("./step5.js"))
    .then(() => loadScript("./step6.js"))
    .then(() => loadScript("./adaptive-applicability.js"))
    .then(() => loadScript("./progressive-guidance-core.js"))
    .then(() => loadScript("./osteoporosis-evidence-guidance-core.js"))
    .then(() => loadScript("./osteoporosis-longitudinal-summary-core.js"))
    .then(() => loadScript("./finalization-coordinator.js"))
    .then(() => loadScript("./patient-registry.js"))
    .then(() => loadScript("./progressive-guidance-ui.js"))
    .then(() => loadScript("./pilot-completion.js"))
    .then(() => loadScript("./whole-form-progress.js"))
    .then(() => loadScript("./lab-history-ui.js"))
    .then(() => loadScript("./calendar-link.js"))
    .catch((error) => {
      document.querySelector("#clinicalWorkspacePreloadGuard")?.remove();
      console.error("Clinical workspace bootstrap failed", error);
    });
})();
