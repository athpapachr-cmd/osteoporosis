"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const app = fs.readFileSync("static/cockpit/visit-capture/app.js", "utf8");
const html = fs.readFileSync("static/cockpit/visit-capture/index.html", "utf8");
const promptMatch = html.match(/<template id="diaPromptTemplate">([\s\S]*?)<\/template>/);
assert.ok(promptMatch, "Dia template is present");

function harness(authenticated) {
  const events = new Map();
  const calls = [];
  const byId = new Map();
  const clipboard = [];
  const registry = Array.from({length: 151}, (_, number) => ({
    patient_id: "SYN-" + String(number).padStart(3, "0"),
    demographics: {full_name: number === 0 ? "Αθανάσιος Παπαχρήστου" : "Κοινός Δοκιμαστικός"},
  })).reverse();
  function folded(value) {
    return String(value).normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("el");
  }
  function element(id = "") {
    const callbacks = {};
    const classNames = new Set();
    const el = {
      id, hidden: false, disabled: false, value: "", textContent: "", className: "",
      dataset: {}, children: [], attributes: {},
      classList: {
        toggle(name, force) {
          if (force) classNames.add(name);
          else classNames.delete(name);
        },
        contains(name) { return classNames.has(name); },
      },
      addEventListener(name, fn) { callbacks[name] = fn; },
      dispatch(name, event = {}) {
        assert.ok(callbacks[name], "event handler exists: " + id + "/" + name);
        return callbacks[name](event);
      },
      setAttribute(name, value) { el.attributes[name] = value; },
      replaceChildren() { el.children = []; el.value = ""; },
      appendChild(child) { el.children.push(child); },
      focus() {},
    };
    return el;
  }
  const segments = ["snapshot", "brief", "detail"].map((level) => {
    const el = element(level);
    el.dataset.level = level;
    return el;
  });
  const document = {
    getElementById(id) {
      if (!byId.has(id)) byId.set(id, element(id));
      return byId.get(id);
    },
    createElement() { return element(); },
    querySelectorAll(selector) { return selector === ".segment" ? segments : []; },
    body: {appendChild() {}},
    execCommand(name) { assert.equal(name, "copy"); return true; },
  };
  document.getElementById("diaPromptTemplate").content = {textContent: promptMatch[1]};
  document.getElementById("authToggle").hidden = true;
  document.getElementById("saveRow").hidden = true;
  document.getElementById("patientPanel").hidden = true;
  document.getElementById("loginBox").hidden = true;
  document.getElementById("morePatientsBtn").hidden = true;
  const memory = new Map();
  async function fetch(url, options = {}) {
    calls.push({url, method: options.method || "GET"});
    let body = {};
    let status = 200;
    if (url === "/clinical/status") {
      if (!authenticated) status = 401;
    } else if (url.startsWith("/clinical/patients?query=")) {
      const parsed = new URL(url, "https://local.test");
      const term = folded(parsed.searchParams.get("query") || "");
      const tokens = term.split(/\s+/).filter(Boolean);
      const offset = Number(parsed.searchParams.get("offset") || 0);
      const limit = Number(parsed.searchParams.get("limit") || 20);
      body = registry.filter((patient) => {
        const content = folded(patient.patient_id + " " + patient.demographics.full_name);
        return tokens.every((token) => content.includes(token));
      }).slice(offset, offset + limit);
    } else if (url === "/clinical/visit-capture/context") {
      const request = JSON.parse(options.body);
      assert.ok(registry.some((row) => row.patient_id === request.patient_id), "context uses existing patient");
      body = {context_id: "context-123", patient_id: request.patient_id};
    } else if (url === "/clinical/visit-capture/preview") {
      body = {patient_id: "SYN-A", can_save: true, snapshot: "S", brief: "B", detail: "D"};
    } else if (url === "/clinical/visit-capture/save") {
      body = {encounter: {encounter_date: "2026-10-10"}, pending: []};
    } else if (url === "/clinical/login") {
      body = {authenticated: true};
    } else throw new Error("unexpected request: " + url);
    return {ok: status < 400, status, async json() {return status === 401 ? {detail: "Unauthorized"} : body;}};
  }
  const scope = {
    document, navigator: {clipboard: {async writeText(value) {clipboard.push(value);}}},
    fetch, sessionStorage: {
      getItem(key) {return memory.get(key) || null;},
      setItem(key, value) {memory.set(key, value);},
    },
    crypto: {randomUUID() {return "synthetic-session-uuid";}},
    location: {search: ""},
    URLSearchParams,
    setTimeout(fn) {fn(); return 1;},
    clearTimeout() {},
    console,
  };
  vm.runInNewContext(app, scope, {filename: "app.js"});
  return {
    get: (id) => document.getElementById(id),
    segment: (level) => segments.find((el) => el.dataset.level === level),
    calls, clipboard,
    async settle() { await new Promise((resolve) => setImmediate(resolve)); },
  };
}

(async () => {
  const demo = harness(false);
  await demo.settle();
  assert.equal(demo.get("patientPanel").hidden, true);
  assert.equal(demo.get("candidateInput").disabled, false);
  assert.equal(demo.get("saveRow").hidden, true);
  demo.get("exampleBtn").dispatch("click");
  assert.ok(demo.get("previewText").textContent.includes("Συνθετικό"));
  assert.equal(demo.get("saveBtn").disabled, true);
  demo.segment("detail").dispatch("click");
  assert.ok(demo.get("previewText").textContent.includes("Φάρμακα/κωδικοποίηση"));
  assert.ok(!demo.calls.some((c) => c.method === "POST"), "demo never posts data");
  demo.get("copyDiaPromptBtn").dispatch("click");
  await demo.settle();
  assert.ok(demo.clipboard[0].includes("ENCOUNTER DETAIL"));

  // Fully fabricated case: checks structural display, not medical correctness.
  // Every note below is user/source text; the app must never auto-diagnose.
  demo.get("candidateInput").value = [
    "SNAPSHOT",
    "Κλινική εικόνα: Συνθετική παρακολούθηση.",
    "Μεταβολή: Δεν υπάρχουν δεδομένα προηγούμενης επίσκεψης.",
    "Απόφαση: Παραγγέλθηκαν εργαστηριακές εξετάσεις.",
    "Εκκρεμότητες: Οι εξετάσεις δεν πραγματοποιήθηκαν και δεν υπάρχουν αποτελέσματα.",
    "",
    "VISIT BRIEF",
    "Λόγος επίσκεψης: Συνθετική επανεξέταση.",
    "Ευρήματα: FRAX MOF 6,2% και κίνδυνος ισχίου 1,3%, σύμφωνα με τη συνθετική πηγή.",
    "Αποφάσεις: Εργαστηριακός έλεγχος παραγγέλθηκε, χωρίς αποτέλεσμα.",
    "",
    "ENCOUNTER DETAIL",
    "Πηγές:",
    "Συνθετικό κείμενο ελέγχου.",
    "Ιστορικό:",
    "Δεν παρασχέθηκε προηγούμενη ημερομηνία.",
    "Εξετάσεις και κατάσταση:",
    "Παραγγέλθηκαν, δεν έχουν πραγματοποιηθεί, δεν υπάρχουν αποτελέσματα.",
    "Κλινική εκτίμηση:",
    "MOF και ισχίο αναφέρονται χωριστά.",
    "Σημεία προς έλεγχο (διοικητικά, όχι κλινική διάγνωση):",
    "1. Έλεγξε κωδικό παραπεμπτικού σύμφωνα με την αρχική πηγή.",
    "2. Διευκρίνισε την ετικέτα της ημερομηνίας <img src=x onerror=alert(1)>.",
  ].join("\n");
  demo.get("candidateInput").dispatch("input");
  demo.segment("snapshot").dispatch("click"); // previous legacy example left Detail selected
  assert.equal(demo.get("diaReviewPanel").hidden, false, "two supplied source alerts are surfaced");
  assert.ok(demo.get("diaReviewCount").textContent.startsWith("2 σημεία"));
  assert.equal(demo.get("diaReviewList").children.length, 2);
  assert.ok(demo.get("diaReviewList").children[1].textContent.includes("<img src=x"), "untrusted mark-up stays literal text");
  assert.equal(demo.get("structuredPreview").hidden, false, "labeled snapshot is structured");
  assert.equal(demo.get("structuredPreview").children.length, 4, "four supplied snapshot labels");

  demo.get("diaReviewDetails").open = true;
  demo.segment("brief").dispatch("click");
  assert.equal(demo.get("structuredPreview").hidden, false);
  assert.ok(demo.get("structuredPreview").children.some((node) =>
    node.children.some((child) => child.textContent === "Ευρήματα")), "Brief labels from source are preserved");
  assert.equal(demo.get("diaReviewDetails").open, true, "review notes stay open when changing tabs");

  demo.segment("detail").dispatch("click");
  const detailPanels = demo.get("structuredPreview").children;
  assert.ok(detailPanels.length >= 4, "Encounter Detail becomes separate explicit sections");
  assert.ok(detailPanels.every((panel) => typeof panel.open === "boolean"), "detail sections are collapsible");
  assert.equal(detailPanels[0].open, true, "first section is open for instant reading");
  assert.equal(detailPanels[1].open, false, "other sections start collapsed");
  assert.ok(demo.get("previewText").textContent.includes("δεν υπάρχουν αποτελέσματα"), "original Dia text not rewritten");
  assert.equal(demo.get("previewText").hidden, true, "structured detail displayed instead of a wall of text");

  // Old freeform notes must not be given invented clinical section titles or alerts.
  demo.get("candidateInput").value = "Ελεύθερο κείμενο χωρίς επισήμανση ή έγκυρη ενότητα.";
  demo.get("candidateInput").dispatch("input");
  assert.equal(demo.get("diaReviewPanel").hidden, true);
  assert.equal(demo.get("structuredPreview").hidden, true, "unlabeled text is shown verbatim");
  assert.equal(demo.get("previewText").hidden, false);

  demo.get("candidateInput").value = "";
  demo.get("candidateInput").dispatch("input");
  assert.equal(demo.get("diaReviewPanel").hidden, true, "clearing source clears review notes");
  assert.equal(demo.get("structuredPreview").hidden, true);
  assert.equal(demo.get("saveBtn").disabled, true);
  assert.ok(!demo.calls.some((c) => c.method === "POST"), "readability-only demo never posts clinical content");

  const live = harness(true);
  await live.settle();
  assert.equal(live.get("authCard").hidden, true, "authenticated panel automatically hides");
  assert.equal(live.get("loginBox").hidden, true);
  assert.equal(live.get("authToggle").hidden, false);
  live.get("authToggle").dispatch("click");
  assert.equal(live.get("authCard").hidden, false, "side button restores auth panel");
  live.get("authToggle").dispatch("click");
  assert.equal(live.get("authCard").hidden, true);
  live.get("recordModeBtn").dispatch("click");
  assert.equal(live.get("candidateInput").disabled, true, "record editor gated on patient confirmation");
  await live.settle();
  assert.ok(!live.calls.some((c) => c.url.startsWith("/clinical/patients?")), "no unrequested latest-100 fetch");
  assert.equal(live.get("patientSelect").children.length, 1, "picker empty until query entered");

  // The oldest patient is outside the latest 100 but remains fully searchable.
  live.get("patientSearch").value = "ΠΑΠΑΧΡΗΣΤΟΥ";
  live.get("patientSearch").dispatch("input");
  await live.settle();
  assert.ok(live.calls.some((c) => c.url.includes("query=") && c.url.includes("offset=0")));
  assert.ok(live.get("patientSelect").children.some((c) => c.value === "SYN-000"), "oldest patient returned by backend");

  // Common names are paged, not silently limited to first 20 or latest 100.
  live.get("patientSearch").value = "κοινος";
  live.get("patientSearch").dispatch("input");
  await live.settle();
  assert.equal(live.get("patientSelect").children.length, 21, "first 20 matches plus placeholder");
  assert.equal(live.get("morePatientsBtn").hidden, false);
  live.get("morePatientsBtn").dispatch("click");
  await live.settle();
  assert.equal(live.get("patientSelect").children.length, 41, "additional search page appended");
  assert.ok(live.calls.some((c) => c.url.includes("offset=20")), "next page requested from backend");

  live.get("patientSearch").value = "SYN-000";
  live.get("patientSearch").dispatch("input");
  await live.settle();
  live.get("patientSelect").value = "SYN-000";
  live.get("confirmPatientBtn").dispatch("click");
  await live.settle();
  assert.equal(live.get("candidateInput").disabled, false);
  assert.equal(live.get("saveBtn").disabled, true, "Save is not automatic");
  live.get("demoModeBtn").dispatch("click");
  assert.equal(live.get("saveRow").hidden, true);
  live.get("exampleBtn").dispatch("click");
  assert.equal(live.get("saveBtn").disabled, true);
  assert.ok(!live.calls.some((c) => c.url === "/clinical/visit-capture/save"), "no clinical write from test");
  assert.ok(!live.calls.some((c) => c.url === "/clinical/visit-capture/preview"), "demo never invokes protected patient preview");
  process.stdout.write("Visit Capture no-patient smoke UX: PASS\n");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
