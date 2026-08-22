(() => {
  "use strict";

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
    .then(() => loadScript("./step3.js"))
    .then(() => loadScript("./step4.js"))
    .then(() => loadScript("./longitudinal.js"))
    .then(() => loadScript("./step5.js"))
    .then(() => loadScript("./step6.js"))
    .then(() => loadScript("./pilot-completion.js"))
    .catch((error) => console.error("Baseline Audit bootstrap failed", error));
})();
