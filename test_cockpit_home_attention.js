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
}
class SyntheticEvent { constructor(type, options = {}) { this.type = type; this.detail = options.detail; } }
const target = new Element(), document = {hidden: false,
  getElementById() { return target; }, createElement(t) { return new Element(t); },
  addEventListener(n, f) { documentHandlers.set(n, f); }};
const window = {addEventListener(n, f) { if (!handlers.has(n)) handlers.set(n, []); handlers.get(n).push(f); },
  dispatchEvent(e) { for (const f of handlers.get(e.type) || []) f(e); },
  CockpitHomeAttentionNavigation: {cases(t) { actions.push(t); }, obligations(t) { actions.push(t || "tasks"); }, calls() { actions.push("calls"); }} };
const actions = [];
class Clock extends Date { constructor(...a) { super(...(a.length ? a : ["2026-11-18T08:00:00Z"])); } static now() { return Date.parse("2026-11-18T08:00:00Z"); } }
const pending = [], requests = [];
const ok = data => ({status: 200, ok: true, json: async () => data});
const bad = status => ({status, ok: false, json: async () => ({})});
const fixtures = {
  rf: {applications:[{patient_name:"Συνθετικός RF",stage:"approved",surgery_id:null}],has_more:true,snapshot_consistent:false},
  reports: {cases:[{patient_name:"Συνθετική Έκθεση",writing_started_date:"2026-11-17",report_sent_date:null},
    {patient_name:"Χωρίς Αίτημα",writing_started_date:null,report_sent_date:null,visit_date:"2026-09-01"}],has_more:false},
  tasks: {items:[{patient_name:"Συνθετικός Ασθενής",description:"Ιδιωτικό όνομα Συνθετικός",pending_id:"task-1",status:"open",review_date:"2026-11-17"}]},
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
    {date:"2026-11-19",name:"Συνθετικός νέος",procedure:"Επέμβαση",time:"08:00"}
  ],undated:0,invalid:0});
  let output = text(target);
  assert.match(output, /Συνθετικός νέος/); assert.match(output, /Περασμένη ημερομηνία/);
  assert.match(output, /Συνθετικός RF/); assert.match(output, /Μερική κάλυψη/);
  assert.match(output, /Έκθεση σε σύνταξη/); assert.doesNotMatch(output, /Χωρίς Αίτημα/);
  assert.match(output, /Ιδιωτικό όνομα/); assert.match(output, /2/);
  buttons(target).find(b => text(b).includes("Απόκρυψη ονομάτων")).click();
  output = text(target); assert.doesNotMatch(output, /Συνθετικός|Συνθετική|Ιδιωτικό όνομα|Επέμβαση/);
  assert.match(output, /Απόκρυψη ονόματος/);
  buttons(target).find(b => text(b).includes("Εμφάνιση ονομάτων")).click();
  buttons(target).find(b => text(b).includes("Συνθετικός RF")).click(); assert.equal(actions.pop(), "rf");
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
  console.log("PASS Home attention: source truth, partial coverage, privacy, navigation, revocation, races and refresh");
}
run().catch(e => { console.error(e); process.exitCode = 1; });
