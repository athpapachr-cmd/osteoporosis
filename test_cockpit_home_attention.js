"use strict";
// Synthetic-only Home attention lifecycle and truthfulness regression.
const assert = require("node:assert/strict"), fs = require("node:fs"), vm = require("node:vm");
const script = fs.readFileSync("static/cockpit/home-attention.js", "utf8");
const handlers = new Map(), documentHandlers = new Map();
class Element {
  constructor(tag = "div") { this.tagName = tag; this.children = []; this.textContent = ""; this.events = {}; this.attributes = {}; }
  append(...x) { this.children.push(...x); }
  replaceChildren(...x) { this.children = x; }
  setAttribute(k, v) { this.attributes[k] = v; }
  addEventListener(k, v) { this.events[k] = v; }
  click() { this.events.click?.(); }
  focus() { document.activeElement = this; }
  contains(other) { return this === other || this.children.some(x => x.contains(other)); }
}
class SyntheticEvent { constructor(type, options = {}) { this.type = type; this.detail = options.detail; } }
const target = new Element(), document = {hidden: false,
  getElementById() { return target; }, createElement(t) { return new Element(t); },
  addEventListener(n, f) { documentHandlers.set(n, f); }};
const window = {addEventListener(n, f) { if (!handlers.has(n)) handlers.set(n, []); handlers.get(n).push(f); },
  dispatchEvent(e) { for (const f of handlers.get(e.type) || []) f(e); },
  CockpitHomeAttentionNavigation: {cases(t) { actions.push(t); }, obligations(t) { actions.push(t || "tasks"); }, calls() { actions.push("calls"); }} };
const actions = [];
let now = "2026-11-18T08:00:00Z";
class Clock extends Date { constructor(...a) { super(...(a.length ? a : [now])); } static now() { return Date.parse(now); } }
const pending = [], requests = [];
const ok = data => ({status: 200, ok: true, json: async () => data});
const bad = status => ({status, ok: false, json: async () => ({})});
const fixtures = {
  rf: {applications:[{id:"rf-1",patient_name:"Συνθετικός RF",stage:"approved",surgery_id:null}],has_more:true,snapshot_consistent:false},
  reports: {cases:[{patient_name:"Συνθετική Έκθεση",writing_started_date:"2026-11-17",report_sent_date:null},
    {patient_name:"Χωρίς Αίτημα",writing_started_date:null,report_sent_date:null,visit_date:"2026-09-01"}],has_more:false},
  tasks: {items:[{patient_id:"patient-1",patient_name:"Συνθετικός Ασθενής",description:"Ιδιωτικό όνομα Συνθετικός",pending_id:"task-1",status:"open",review_date:"2026-11-17"}]},
  calls: {source:"reception_dashboard",is_unread_count:false,coverage:"available_call_log_max_200",unhandled_count:2}
};
let mode = "bootstrap-pending";
function fetch(url) {
  requests.push(url);
  if (url === "/clinical/status") {
    if (mode === "bootstrap-pending") return new Promise(resolve => pending.push({key:"status",resolve}));
    return Promise.resolve(ok({}));
  }
  const key = url.includes("/office-cases/rf?") ? "rf" : url.includes("/accidents?") ? "reports" : url.includes("/clinical-updates?") ? "tasks" : "calls";
  if (mode === "hold") return new Promise(resolve => pending.push({key,resolve}));
  if (mode === "deny-rf") return Promise.resolve(key === "rf" ? bad(401) : ok(fixtures[key]));
  if (mode === "deny-tasks") return Promise.resolve(key === "tasks" ? bad(403) : ok(fixtures[key]));
  if (mode === "fail-calls") return Promise.resolve(key === "calls" ? bad(503) : ok(fixtures[key]));
  return Promise.resolve(ok(fixtures[key]));
}
const text = e => [e.textContent, ...e.children.map(text)].join(" ");
function buttons(e) { return [ ...(e.tagName === "button" ? [e] : []), ...e.children.flatMap(buttons) ]; }
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
async function flush() { await tick(); await tick(); }
function emit(type, detail) { window.dispatchEvent(new SyntheticEvent(type, {detail})); }
async function run() {
  vm.runInNewContext(script, {window, document, fetch, Event:SyntheticEvent, Date:Clock, Intl, URLSearchParams, console});
  emit("clinical:unauthenticated");
  pending.splice(0).forEach(p => p.resolve(ok({})));
  await flush();
  assert.equal(requests.length, 1, "late bootstrap cannot start protected loaders");
  assert.match(text(target), /Συνδέσου/);
  mode = "normal"; emit("clinical:authenticated"); await flush();
  emit("cockpit:surgery-counts", {attention:[
    {date:"2026-11-17",name:"Συνθετικός παλιός",procedure:"Επέμβαση",time:""},
    ...[1,2,3,4].map(i => ({date:"2026-11-1" + i,name:"Παλαιός " + i,procedure:"Επέμβαση",time:""})),
    {date:"2026-11-19",name:"Συνθετικός νέος",procedure:"Επέμβαση",time:"08:00"}
  ],total:6,undated:0,invalid:0});
  let output = text(target);
  assert.match(output, /Συνθετικός νέος/); assert.match(output, /Περασμένη ημερομηνία/);
  assert.match(text(buttons(target).find(b => text(b).includes("Συνθετικός νέος"))), /Συνθετικός νέος/);
  assert.match(output, /Να γίνουν · 4 από 8/);
  const expander = buttons(target).find(b => text(b).includes("Δες όλα τα 8"));
  expander.focus(); expander.click();
  assert.notEqual(document.activeElement, expander);
  assert.equal(document.activeElement.attributes["aria-expanded"], "true");
  assert.match(text(document.activeElement), /Δες λιγότερα/);
  output = text(target);
  assert.match(output, /Συνθετικός RF/); assert.match(output, /Μερική κάλυψη/);
  assert.match(output, /Συνθετική Έκθεση Σε σύνταξη/); assert.doesNotMatch(output, /Χωρίς Αίτημα/);
  assert.match(output, /Ιδιωτικό όνομα/); assert.match(output, /2/);
  const privacy = buttons(target).find(b => b.attributes["aria-label"] === "Απόκρυψη ονομάτων εδώ");
  privacy.focus(); privacy.click();
  assert.equal(document.activeElement.attributes["aria-label"], "Εμφάνιση ονομάτων εδώ");
  output = text(target); assert.doesNotMatch(output, /Συνθετικός|Συνθετική|Ιδιωτικό όνομα|Επέμβαση/);
  assert.match(output, /Απόκρυψη ονόματος/);
  buttons(target).find(b => b.attributes["aria-label"] === "Εμφάνιση ονομάτων εδώ").click();
  const focusedRF = buttons(target).find(b => text(b).includes("Συνθετικός RF"));
  focusedRF.focus();
  emit("cockpit:surgery-counts", {attention:[{date:"2026-11-19",name:"Συνθετικός νέος",procedure:"Επέμβαση",time:"08:00"}],total:1,undated:0,invalid:0});
  assert.equal(document.activeElement.attributes["aria-label"], "Εγκεκριμένη RF · Συνθετικός RF", "source redraw restores the same RF control");
  document.activeElement.click(); assert.equal(actions.pop(), "rf");
  buttons(target).find(b => text(b).includes("Συνθετικός Ασθενής")).click(); assert.equal(actions.pop(), "task-1");
  mode = "fail-calls"; window.CockpitHomeAttention.refresh(); await flush();
  assert.match(text(target), /Μη διαθέσιμο/); assert.doesNotMatch(text(target), /Τίποτα επείγον/);
  mode = "hold"; window.CockpitHomeAttention.refresh(); await tick();
  const before = requests.length; emit("clinical:unauthenticated");
  pending.splice(0).forEach(p => p.resolve(ok(fixtures[p.key]))); await flush();
  assert.doesNotMatch(text(target), /Συνθετικός|Συνθετική/); window.CockpitHomeAttention.refresh(); assert.equal(requests.length,before);
  mode = "deny-rf"; emit("clinical:authenticated"); await flush();
  assert.match(text(target), /Συνδέσου/); assert.doesNotMatch(text(target), /Συνθετικός/);
  mode = "deny-tasks"; emit("clinical:authenticated"); await flush();
  assert.match(text(target), /Συνδέσου/, "403 from any active source revokes the entire projection");
  mode = "hold"; emit("clinical:authenticated"); await tick(); const stale = pending.splice(0);
  mode = "normal"; emit("clinical:authenticated"); await flush();
  stale.forEach(p => p.resolve(bad(403))); await flush();
  assert.match(text(target), /Συνθετικός RF/, "old denial must not revoke a new session");
  fixtures.rf.applications = [];
  window.CockpitHomeAttention.refresh(); await flush();
  assert.doesNotMatch(text(target), /Συνθετικός RF/, "return/refresh invalidates old data");
  assert.match(text(target), /Μερική κάλυψη/); assert.doesNotMatch(text(target), /Τίποτα επείγον/);
  // Only proven patient ids group; each task retains its own date and exact target.
  fixtures.tasks.items = [
    {patient_id:"same-1",patient_name:"Ίδιο Όνομα",description:"Πρώτη δοκιμή",pending_id:"p1",status:"open",review_date:"2026-11-01"},
    {patient_id:"same-1",patient_name:"Ίδιο Όνομα",description:"Δεύτερη δοκιμή",pending_id:"p2",status:"open",review_date:"2026-11-15"},
    {patient_id:"same-1",patient_name:"Ίδιο Όνομα",description:"Τρίτη δοκιμή",pending_id:"p3",status:"open",review_date:null},
    {patient_id:"same-2",patient_name:"Ίδιο Όνομα",description:"Άλλος ασθενής",pending_id:"p4",status:"open",review_date:"2026-11-03"},
    {patient_id:"same-3",patient_name:"Χωρίς προθεσμία",description:"Άκυρη ημερομηνία",pending_id:"p5",status:"open",review_date:"2026-02-30"}
  ];
  fixtures.reports.cases = [{patient_name:"Συνθετική Έκθεση",writing_started_date:"2026-11-02",report_sent_date:null}];
  fixtures.rf.applications = [{id:"rf-2",patient_name:"Νέα RF",stage:"approved",surgery_id:null,decision_date:"2026-11-04"}];
  window.CockpitHomeAttention.refresh(); await flush();
  let list = target.children.find(n => n.className === "ha-section" && text(n).includes("Να γίνουν"));
  let visible = list.children.filter(n => (n.className || "").includes("ha-row"));
  assert.equal(visible.length,4); assert.match(text(visible[0]), /Πρώτη δοκιμή/);
  assert.match(text(visible[0]), /Δεύτερη δοκιμή/); assert.match(text(visible[0]), /\+1 ακόμη/);
  assert.match(text(visible[0]), /Νοε/); assert.match(text(visible[1]), /Συνθετική Έκθεση/);
  assert.match(text(visible[2]), /Άλλος ασθενής/); assert.match(text(visible[3]), /Νέα RF/);
  buttons(visible[0]).find(b => text(b).includes("Δεύτερη δοκιμή")).click(); assert.equal(actions.pop(), "p2");
  buttons(target).find(b => text(b).includes("Δες όλα τα 5")).click();
  assert.match(text(target), /Άκυρη ημερομηνία/); assert.match(text(target), /Μη έγκυρη ημερομηνία/);
  // An invalid patient identity disables all exact task actions, not just that row.
  fixtures.tasks.items.push({patient_id:null,patient_name:"Άγνωστη ταυτότητα",description:"Μη διαθέσιμο",pending_id:"p6",status:"open",review_date:null});
  window.CockpitHomeAttention.refresh(); await flush();
  assert.match(text(target), /ταυτότητα ορισμένων υποχρεώσεων δεν επιβεβαιώθηκε/);
  list = target.children.find(n => n.className === "ha-section" && text(n).includes("Να γίνουν"));
  visible = list.children.filter(n => (n.className || "").includes("ha-task-row"));
  assert.equal(visible.flatMap(buttons).length, 0, "mixed invalid identity must not promise exact navigation");
  let reprojections = 0;
  window.CockpitSurgeryQueue = {refreshAttention() {
    reprojections++;
    emit("cockpit:surgery-counts", {attention:[{date:"2026-11-26",name:"Νέα ημέρα",procedure:"Επέμβαση",time:""}],total:1,undated:0,invalid:0});
  }};
  now = "2026-11-19T00:05:00Z";
  emit("focus"); await flush();
  assert.ok(reprojections > 0, "return across midnight reprojects the existing surgery source");
  assert.match(text(target), /Νέα ημέρα/);
  // A surgery outside the seven-day projection still has an explicit list link.
  fixtures.tasks.items = []; fixtures.rf.applications = []; fixtures.reports.cases = [];
  fixtures.rf.has_more = false; fixtures.calls.unhandled_count = 0;
  window.CockpitSurgeryQueue = undefined; window.CockpitHomeAttention.refresh(); await flush();
  emit("cockpit:surgery-counts", {attention:[],total:1,undated:0,invalid:0});
  assert.match(text(target), /1 χειρουργείο μετά το επταήμερο/);
  buttons(target).find(b => text(b).includes("μετά το επταήμερο")).click(); assert.equal(actions.pop(), "surgery");
  // Mixed-age obligations of one confirmed patient stay together in the main group.
  fixtures.tasks.items = [
    {patient_id:"mixed",patient_name:"Δοκιμή Μικτή",description:"Παλιό",pending_id:"mixed-old",status:"open",review_date:"2026-07-01"},
    {patient_id:"mixed",patient_name:"Δοκιμή Μικτή",description:"Μελλοντικό",pending_id:"mixed-new",status:"open",review_date:"2026-11-20"},
    {patient_id:"other",patient_name:"Άλλη Δοκιμή",description:"Παλιό μόνο",pending_id:"old-only",status:"open",review_date:"2026-07-02"}
  ];
  fixtures.reports.cases=[{patient_name:"Παλιά Έκθεση",writing_started_date:"2026-07-03",report_sent_date:null}];
  fixtures.rf.applications=[{id:"rf-old",patient_name:"Παλιά RF",stage:"approved",surgery_id:null,decision_date:"2026-07-04"}];
  emit("clinical:unauthenticated"); emit("clinical:authenticated"); await flush();
  assert.match(text(target), /2 υποχρεώσεις · 1 εκπρόθεσμη/);
  buttons(target).find(b=>text(b).includes("Δες όλα τα 4"))?.click();
  // The compact threshold is >4: add one more report to enable filters.
  fixtures.reports.cases.push({patient_name:"Νέα Έκθεση",writing_started_date:"2026-11-01",report_sent_date:null});
  window.CockpitHomeAttention.refresh();await flush();
  buttons(target).find(b=>text(b).includes("Δες όλα τα 5")).click();
  assert.match(text(target), /Παλιότερη ημερομηνία/);
  assert.match(text(target), /Μελλοντικό/, "mixed-age group remains in main list");
  buttons(target).find(b=>text(b).includes("Μελλοντικό")).click();assert.equal(actions.pop(),"mixed-new");
  assert.match(text(target), /Παλιό μόνο/, "expansion preserves already visible old rows");
  buttons(target).find(b=>text(b).includes("Παλιότερη ημερομηνία")).click();
  assert.doesNotMatch(text(target), /Παλιό μόνο/, "old rows collapse explicitly");
  buttons(target).find(b=>text(b).includes("Παλιότερη ημερομηνία")).click();
  buttons(target).find(b=>text(b).includes("Υποχρεώσεις 2")).click();
  assert.doesNotMatch(text(target), /Παλιά Έκθεση/);
  assert.match(text(target), /3 ενεργές εγγραφές στην προβολή έως 100/);
  // All five old groups: expansion cannot hide any already visible row.
  fixtures.tasks.items=Array.from({length:5},(_,i)=>({patient_id:"old-"+i,patient_name:"Παλιός "+i,description:"Παλιά "+i,pending_id:"id-"+i,status:"open",review_date:"2026-07-0"+(i+1)}));
  fixtures.reports.cases=[];fixtures.rf.applications=[];
  emit("clinical:unauthenticated");emit("clinical:authenticated");await flush();
  assert.match(text(target),/Παλιός 0/);assert.match(text(target),/Παλιός 3/);assert.doesNotMatch(text(target),/Παλιός 4/);
  buttons(target).find(b=>text(b).includes("Δες όλα τα 5")).click();
  assert.match(text(target),/Παλιός 0/);assert.match(text(target),/Παλιός 3/);assert.match(text(target),/Παλιός 4/);
  console.log("PASS Home attention: source truth, partial coverage, privacy, navigation, revocation, races and refresh");
}
run().catch(e => { console.error(e); process.exitCode = 1; });
