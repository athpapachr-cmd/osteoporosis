"use strict";
// Synthetic protected-source regression: no patient/provider calls or persistence.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const app = fs.readFileSync("static/cockpit/app.js", "utf8");
const shell = fs.readFileSync("static/cockpit/doctor-shell.js", "utf8");
assert.equal((app.match(/apiJson\("\/clinical\/surgeries"\)/g) || []).length, 1,
  "Home must not add a second surgery reader");
for (const text of [app, shell]) assert.doesNotMatch(text, /localStorage|sessionStorage/);
assert.doesNotMatch(shell, /\/api\/outcomes\/[^"]+\/events/);

const nodes = new Map(), listeners = new Map(), calls = [];
function node(id = "", tag = "div") {
  const handlers = new Map(), classes = new Set();
  const el = {
    id, tagName: tag, textContent: "", value: "", hidden: false, open: false,
    dataset: {}, attributes: {}, children: [], firstChild: {textContent: "Πρόγραμμα σήμερα "},
    classList: {
      contains(k) { return classes.has(k); },
      add(k) { classes.add(k); },
      remove(k) { classes.delete(k); },
      toggle(k, v) { if (v === undefined) v = !classes.has(k);
        if (v) classes.add(k); else classes.delete(k); return v; }
    },
    setAttribute(k, v) { el.attributes[k] = String(v); },
    addEventListener(k, fn) { handlers.set(k, fn); },
    fire(k, event = {}) { assert.ok(handlers.has(k), "event " + id + "/" + k);
      return handlers.get(k)(event); },
    append(...items) { el.children.push(...items); },
    appendChild(item) { el.children.push(item); return item; },
    replaceChildren(...items) { el.children = items; },
    reset() { el.value = ""; },
    focus() {}, contains(other) { return other === el; }
  };
  return el;
}
const tabs = ["surgery","documents","modules","learning"].map(x => {
  const b = node("tab-" + x, "button"); b.dataset.doctorLibraryTab = x; return b;
});
const panels = ["surgery","documents","modules","learning"].map(x => {
  const b = node("panel-" + x); b.dataset.doctorLibraryPanel = x; return b;
});
const documentStub = {
  getElementById(id) { if (!nodes.has(id)) nodes.set(id, node(id)); return nodes.get(id); },
  createElement(tag) { return node("", tag); },
  querySelectorAll(selector) {
    if (selector === "[data-doctor-library-tab]") return tabs;
    if (selector === "[data-doctor-library-panel]") return panels;
    return [];
  },
  addEventListener() {},
  body: {classList: {contains() { return false; }, toggle() {}}}
};
const $ = id => documentStub.getElementById(id);
$("doctorPopover").hidden = true; $("doctorLibraryView").hidden = true;
const win = {
  CockpitSurgerySummary: null, CockpitHome: {},
  addEventListener(type, fn) {
    if (!listeners.has(type)) listeners.set(type, []);
    listeners.get(type).push(fn);
  },
  dispatchEvent(event) {
    for (const fn of listeners.get(event.type) || []) fn(event);
  },
  CustomEvent: class {
    constructor(type, options = {}) { this.type = type; this.detail = options.detail; }
  },
  setInterval() { return 0; }, confirm() { return true; }
};
function notifyAuth() { win.dispatchEvent(new win.CustomEvent("clinical:authenticated")); }
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
const ok = value => ({ok: true, status: 200, json: async () => value});
const denied = status => ({ok: false, status, json: async () => ({detail: "Unauthorized"})});
const surgeryResponses = [], rfResponses = [], writeResponses = [];
const CONTEXT = {
  generated_at: "2026-10-14T20:00:00Z", source_updated_at: null,
  today_total: 0, previous: null, current: null, next: null, current_conflict_count: 0
};
async function fakeFetch(url, options = {}) {
  calls.push({url, method: options.method || "GET", options});
  if (url === "/clinical/calendar/cockpit-context") return ok(CONTEXT);
  if (url === "/clinical/surgeries") {
    assert.ok(surgeryResponses.length, "missing synthetic surgery response");
    return surgeryResponses.shift();
  }
  if (url === "/clinical/clinic-utilities/rf/api/outcomes/search") {
    assert.equal(options.method, "POST");
    assert.equal(options.credentials, "same-origin");
    assert.deepEqual(JSON.parse(options.body), {query: "", scope: "all", page: 1});
    assert.ok(rfResponses.length, "missing synthetic RF response");
    return rfResponses.shift();
  }
  if (url.startsWith("/clinical/surgeries/")) {
    assert.ok(writeResponses.length, "unexpected synthetic surgery edit");
    return writeResponses.shift();
  }
  throw Error("Unexpected endpoint " + url);
}
class CyprusClock extends Date {
  constructor(...args) { super(...(args.length ? args : ["2026-10-14T22:30:00Z"])); }
  static now() { return Date.parse("2026-10-14T22:30:00Z"); }
}
function surgery(id, name, date, procedure = "Δοκιμαστική επέμβαση") {
  return {
    surgery_id: id, full_name: name, surgery_date: date, procedure_type: procedure,
    identity_number: "SYNTHETIC", date_of_birth: "1970-01-01",
    phone: "000000", laterality: "left", queue_position: Number(id.slice(1))
  };
}
const rows = [
  surgery("s1", "Συνθετικός Μόνος", "2026-10-15", "Αρθροσκόπηση"),
  surgery("s2", "Συνθετικός Δεύτερος", "2026-10-16"),
  surgery("s3", "Συνθετικός Τρίτος", "2026-10-16"),
  surgery("s4", "Συνθετικός Παλαιός", "2026-10-14"),
  surgery("s5", "Συνθετικός Χωρίς", null),
  surgery("s6", "Συνθετικός Άκυρος", "2026-02-30")
];
const rf = {
  scope: "all", page: 1, denominator: 6, truncated: false, applications: [],
  counts: {issued_unknown: 1, pending: 1, approved: 2, rejected: 1,
    scheduled: 1, performed: 1, mine_actionable: 1}
};
surgeryResponses.push(ok(rows)); rfResponses.push(ok(rf));
const env = {document: documentStub, window: win, fetch: fakeFetch,
  Date: CyprusClock, Intl, console, URLSearchParams};
vm.runInNewContext(app, env);
vm.runInNewContext(shell, env);
const tick = () => new Promise(resolve => setImmediate(resolve));
async function settle() { for (let i = 0; i < 5; i++) await tick(); }
function content(el) {
  return [String(el?.textContent || ""), ...(el?.children || []).map(content)].join(" ");
}
function attention(index) { return content($("doctorAttentionList").children[index]); }
async function refresh(surgeryResponse, rfResponse) {
  surgeryResponses.push(surgeryResponse);
  rfResponses.push(rfResponse);
  notifyAuth(); await settle();
}
(async () => {
  await settle();
  assert.equal($("doctorAttentionList").children.length, 4);
  assert.equal(win.CockpitSurgerySummary.total, 6);
  assert.equal(win.CockpitSurgerySummary.undated, 1);
  assert.equal(win.CockpitSurgerySummary.invalid, 1);
  assert.equal(win.CockpitSurgerySummary.overdue, 1);
  assert.equal(win.CockpitSurgerySummary.upcoming.length, 2);
  assert.equal(win.CockpitSurgerySummary.upcoming[0].date, "2026-10-15",
    "Cyprus date-only 15th, even when UTC is still the 14th");
  assert.equal(win.CockpitSurgerySummary.upcoming[0].name, "Συνθετικός Μόνος");
  assert.equal(win.CockpitSurgerySummary.upcoming[1].count, 2);
  assert.equal(win.CockpitSurgerySummary.upcoming[1].name, undefined);
  assert.match(attention(0), /Συνθετικός Μόνος/);
  assert.doesNotMatch(attention(0), /Συνθετικός Δεύτερος|Συνθετικός Τρίτος/);
  assert.doesNotMatch(attention(0), /Πραγματοποιήθηκαν/);
  assert.match(attention(0), /παλαιότερα εκκρεμή/);
  assert.match(attention(1), /Εκδοθείσες αιτήσεις 6/);
  assert.match(attention(1), /Αναμένουν απόφαση 1/);
  assert.match(attention(1), /Δικά μου προς παρακολούθηση 1/);
  assert.doesNotMatch(attention(1), /Υποβλήθηκαν|Συνθετικός/);
  assert.equal($("doctorAttentionList").children[1].href, "/clinical/clinic-utilities/rf");
  assert.equal($("doctorAttentionList").children[2].href, "/clinical/clinic-utilities/medical-report");
  assert.match(attention(2), /Δεν υπάρχει ακόμη αξιόπιστη καταμέτρηση/);
  assert.equal($("doctorInboxBadge").hidden, true);
  $("doctorAttentionList").children[0].fire("click");
  assert.equal($("doctorLibraryView").hidden, false, "one tap to existing queue");
  $("doctorBackHome").fire("click");
  assert.equal($("doctorMainView").hidden, false);

  // A delayed protected success cannot republish PHI after newer 401/403/503.
  const staleSuccess = deferred();
  surgeryResponses.push(staleSuccess.promise, denied(403));
  rfResponses.push(ok(rf), denied(403));
  notifyAuth(); notifyAuth();
  assert.equal(win.CockpitSurgerySummary, null, "auth refresh clears immediately");
  assert.doesNotMatch(attention(0), /Συνθετικός Μόνος/);
  await settle();
  staleSuccess.resolve(ok(rows)); await settle();
  assert.equal(win.CockpitSurgerySummary, null);
  assert.doesNotMatch(attention(0), /Συνθετικός/);
  assert.match(attention(1), /Μη διαθέσιμα/);

  // An older failure cannot clear a newer successful protected response.
  const staleFailure = deferred();
  surgeryResponses.push(staleFailure.promise, ok([rows[0]]));
  rfResponses.push(ok(rf), ok(rf));
  notifyAuth(); notifyAuth(); await settle();
  assert.equal(win.CockpitSurgerySummary.total, 1);
  staleFailure.resolve(denied(401)); await settle();
  assert.equal(win.CockpitSurgerySummary.upcoming[0].name, "Συνθετικός Μόνος");

  // Non-array and malformed row fail closed; only confirmed [] means zero.
  for (const bad of [null, {}, [null], [{surgery_id: "broken"}]]) {
    await refresh(ok(bad), ok(rf));
    assert.equal(win.CockpitSurgerySummary, null);
    assert.doesNotMatch(attention(0), /Συνθετικός/);
  }
  await refresh(ok([]), ok(rf));
  assert.equal(win.CockpitSurgerySummary.total, 0);
  assert.match(attention(0), /0 εκκρεμή/);

  // RF counts are an attempts-based snapshot, never an inferred submission ledger.
  await refresh(ok([]), ok({...rf, truncated: true}));
  assert.match(attention(1), /Μερική κάλυψη/);
  assert.doesNotMatch(attention(1), /Εκδοθείσες αιτήσεις 6|Αναμένουν απόφαση 1/);
  const invalidRF = [
    {...rf, truncated: "false"}, {...rf, truncated: undefined},
    {...rf, denominator: -1}, {...rf, counts: {...rf.counts, pending: -1}},
    {...rf, counts: {...rf.counts, approved: 1.5}},
    {...rf, counts: {...rf.counts, mine_actionable: undefined}},
    {...rf, counts: {}}, null
  ];
  for (const invalid of invalidRF) {
    await refresh(ok([]), ok(invalid));
    assert.match(attention(1), /Μη διαθέσιμα/);
    assert.doesNotMatch(attention(1), /Εκδοθείσες αιτήσεις 6/);
  }
  for (const status of [401, 403, 503]) {
    await refresh(ok([]), denied(status));
    assert.match(attention(1), /Μη διαθέσιμα/);
  }

  // Late RF success after newer rejection must not restore stale totals.
  const staleRF = deferred();
  surgeryResponses.push(ok([]), ok([]));
  rfResponses.push(staleRF.promise, denied(401));
  notifyAuth(); notifyAuth(); await settle();
  staleRF.resolve(ok(rf)); await settle();
  assert.match(attention(1), /Μη διαθέσιμα/);
  // Reverse ordering: late failure must not erase latest valid RF.
  const oldRF = deferred();
  surgeryResponses.push(ok([]), ok([]));
  rfResponses.push(oldRF.promise, ok(rf));
  notifyAuth(); notifyAuth(); await settle();
  oldRF.resolve(denied(503)); await settle();
  assert.match(attention(1), /Εκδοθείσες αιτήσεις 6/);

  // Existing surgery post-write actions MUST reload through the guarded reader.
  await refresh(ok([rows[0]]), ok(rf));
  const before = calls.filter(c => c.url === "/clinical/surgeries").length;
  writeResponses.push(ok([])); surgeryResponses.push(ok([rows[0]]));
  await $("surgeryTableBody").fire("click", {
    target: {closest() {return {dataset: {action: "up", surgeryId: "s1"}};}}
  });
  await settle();
  assert.equal(calls.filter(c => c.url === "/clinical/surgeries").length, before + 1);
  writeResponses.push(ok(rows[0])); surgeryResponses.push(ok([surgery("s1", "Συνθετικός Μόνος", "2026-10-16")]));
  await $("surgeryTableBody").fire("change", {
    target: {closest() {return {dataset: {surgeryId: "s1"}, value: "2026-10-16"};}}
  });
  await settle();
  assert.equal(win.CockpitSurgerySummary.upcoming[0].date, "2026-10-16");
  // A failed protected edit cannot leave a previously authorized Home snapshot.
  writeResponses.push(denied(401)); surgeryResponses.push(denied(401));
  await $("surgeryTableBody").fire("click", {
    target: {closest() {return {dataset: {action: "up", surgeryId: "s1"}};}}
  });
  await settle();
  assert.equal(win.CockpitSurgerySummary, null);
  assert.doesNotMatch(attention(0), /Συνθετικός/);

  assert.equal(calls.filter(c => c.url === "/clinical/surgeries").length,
    calls.filter(c => c.url === "/clinical/surgeries" && c.method === "GET").length);
  assert.ok(calls.filter(c => c.method === "POST" && c.url.includes("/rf/"))
    .every(c => c.url === "/clinical/clinic-utilities/rf/api/outcomes/search"));
  assert.ok(calls.every(c => !c.url.includes("patient_id") && !c.url.includes("identity_number")));
  console.log("PASS Cockpit Attention: surgery singleton/privacy, Cyprus dates, races, RF semantics/truncation, post-write guard");
})().catch(error => {console.error(error);process.exitCode = 1;});
