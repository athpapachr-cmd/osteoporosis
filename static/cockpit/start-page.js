(() => {
  "use strict";

  // This preference contains only one allowlisted page id. It never stores a
  // credential, patient, appointment, message, case or selected record.
  const STORAGE_KEY = "clinical-excellence.start-page.v1";
  const pages = Object.freeze({
    day: Object.freeze({label: "Ημέρα μου", href: "/static/cockpit/?view=day"}),
    calendar: Object.freeze({label: "Εβδομαδιαίο ημερολόγιο", href: "/static/clinical-calendar/"}),
    inbox: Object.freeze({label: "Κλινικά εισερχόμενα", href: "/static/cockpit/#icloud-mail"}),
    cases: Object.freeze({label: "Περιστατικά ιατρείου", href: "/static/cockpit/#office-cases"}),
  });

  const valid = value => typeof value === "string" && Object.hasOwn(pages, value);

  function read() {
    try {
      const value = window.localStorage?.getItem(STORAGE_KEY);
      return valid(value) ? value : "day";
    } catch (_) {
      return "day";
    }
  }

  function write(value) {
    if (!valid(value)) return false;
    try {
      window.localStorage?.setItem(STORAGE_KEY, value);
      if (window.localStorage?.getItem(STORAGE_KEY) !== value) return false;
    } catch (_) {
      return false;
    }
    window.dispatchEvent?.(new CustomEvent("cockpit:start-page-changed", {detail:{value}}));
    return true;
  }

  function current() {
    const path = window.location?.pathname || "";
    if (path === "/static/clinical-calendar/" || path === "/static/clinical-calendar/index.html")
      return "calendar";
    if (path !== "/static/cockpit/" && path !== "/static/cockpit/index.html") return null;
    if (window.location.hash === "#icloud-mail") return "inbox";
    if (window.location.hash === "#office-cases") return "cases";
    if (window.location.hash || new URLSearchParams(window.location.search).has("chart")) return null;
    const main = document.getElementById("doctorMainView");
    const home = document.getElementById("visitWorkspaceHome");
    if ((main && main.hidden) || (home && home.hidden)) return null;
    return "day";
  }

  const href = value => pages[valid(value) ? value : read()].href;
  const label = value => pages[valid(value) ? value : read()].label;

  window.CockpitStartPage = Object.freeze({
    key: STORAGE_KEY,
    read,
    write,
    current,
    href,
    label,
    values: Object.freeze(Object.keys(pages)),
  });

  // Only the generic Cockpit entry follows the preference. Direct links and
  // selected clinical states remain untouched. The allowlist prevents an
  // arbitrary stored string from becoming a navigation target.
  const path = window.location?.pathname || "";
  const genericEntry = (path === "/static/cockpit/" || path === "/static/cockpit/index.html") &&
    !window.location.search && !window.location.hash;
  const preferred = read();
  if (genericEntry && preferred !== "day") window.location.replace(pages[preferred].href);
})();
