(() => {
  "use strict";

  const PREF_PREFIX = "osteoporosis.workspace.ui.v1.";
  function readCollapsed(key) {
    try {
      return sessionStorage.getItem(PREF_PREFIX + key) === "collapsed";
    } catch {
      return false;
    }
  }

  function writeCollapsed(key, collapsed) {
    try {
      const next = collapsed ? "collapsed" : "expanded";
      if (sessionStorage.getItem(PREF_PREFIX + key) !== next) {
        sessionStorage.setItem(PREF_PREFIX + key, next);
      }
    } catch {
      // UI preference persistence is optional and never clinical data.
    }
  }

  function setCollapsed(root, button, key, collapsed) {
    root.classList.toggle("g4-collapsed", collapsed);
    const expanded = collapsed ? "false" : "true";
    const text = collapsed ? "Ανάπτυξη" : "Σύμπτυξη";
    const title = collapsed ? "Εμφάνιση περιεχομένου" : "Απόκρυψη περιεχομένου";
    if (button.getAttribute("aria-expanded") !== expanded) button.setAttribute("aria-expanded", expanded);
    if (button.textContent !== text) button.textContent = text;
    if (button.title !== title) button.title = title;
    writeCollapsed(key, collapsed);
  }

  function ensureCollapseControl({ rootSelector, headSelector, key }) {
    const root = document.querySelector(rootSelector);
    const head = root?.querySelector(headSelector);
    if (!root || !head) return;

    let button = head.querySelector(`[data-g4-collapse="${key}"]`);
    if (!button) {
      button = document.createElement("button");
      button.type = "button";
      button.className = "g4-collapse-control";
      button.dataset.g4Collapse = key;
      button.setAttribute("aria-controls", root.id);
      button.addEventListener("click", () => {
        setCollapsed(root, button, key, !root.classList.contains("g4-collapsed"));
      });
      head.appendChild(button);
    }

    setCollapsed(root, button, key, readCollapsed(key));
  }

  function ensureWorkspaceControls() {
    ensureCollapseControl({
      rootSelector: "#patientLongitudinalSummary",
      headSelector: ".patient-summary-head",
      key: "patient-summary"
    });
    ensureCollapseControl({
      rootSelector: "#progressiveGuidanceSummary",
      headSelector: ".progressive-guidance-head",
      key: "current-flow"
    });
  }

  function refreshUi() {
    ensureWorkspaceControls();
  }

  let refreshQueued = false;
  function queueRefresh() {
    if (refreshQueued) return;
    refreshQueued = true;
    queueMicrotask(() => {
      refreshQueued = false;
      refreshUi();
    });
  }

  if (typeof MutationObserver !== "undefined") {
    const observer = new MutationObserver(queueRefresh);
    observer.observe(document.body, { childList: true, subtree: true });
  }

  refreshUi();
  setTimeout(refreshUi, 150);

  window.G4WorkspaceErgonomics = Object.freeze({
    refresh: refreshUi
  });
})();