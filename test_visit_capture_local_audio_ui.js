"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const app = fs.readFileSync("static/cockpit/visit-capture/app.js", "utf8");
const entry = fs.readFileSync("static/cockpit/visit-capture/entry.js", "utf8");
const html = fs.readFileSync("static/cockpit/visit-capture/index.html", "utf8");
const promptMatch = html.match(/<template id="diaPromptTemplate">([\s\S]*?)<\/template>/);
assert.ok(promptMatch);

class Element {
  constructor(tag = "div", id = "") {
    this.tagName = tag.toUpperCase();
    this.id = id;
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.value = "";
    this.textContent = "";
    this.className = "";
    this.dataset = {};
    this.children = [];
    this.attributes = {};
    this.listeners = new Map();
    this.classList = {
      add: (name) => { if (!this.className.split(/\s+/).includes(name)) this.className = (this.className + " " + name).trim(); },
      remove: (name) => { this.className = this.className.split(/\s+/).filter((item) => item && item !== name).join(" "); },
      toggle: (name, force) => force ? this.classList.add(name) : this.classList.remove(name),
      contains: (name) => this.className.split(/\s+/).includes(name),
    };
  }
  addEventListener(name, handler) { this.listeners.set(name, handler); }
  dispatch(name, event = {}) {
    assert.ok(this.listeners.has(name), `missing ${this.id}/${name}`);
    return this.listeners.get(name)(event);
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  appendChild(child) {
    this.children.push(child);
    if (this.tagName === "SELECT" && child.selected) this.value = child.value;
    return child;
  }
  append(...children) { children.forEach((child) => this.appendChild(child)); }
  replaceChildren(...children) { this.children = []; children.forEach((child) => this.appendChild(child)); }
  focus() {}
  querySelector(selector) {
    const className = selector.startsWith(".") ? selector.slice(1) : "";
    for (const child of this.children) {
      if (child.className?.split(/\s+/).includes(className)) return child;
      const nested = child.querySelector?.(selector);
      if (nested) return nested;
    }
    return null;
  }
}

const byId = new Map();
const segments = ["snapshot", "brief", "detail"].map((level) => {
  const item = new Element("button", level);
  item.dataset.level = level;
  return item;
});
const document = {
  getElementById(id) {
    if (!byId.has(id)) byId.set(id, new Element("div", id));
    return byId.get(id);
  },
  createElement(tag) { return new Element(tag); },
  createTextNode(text) { const node = new Element("#text"); node.textContent = text; return node; },
  querySelectorAll(selector) { return selector === ".segment" ? segments : []; },
  body: new Element("body"),
  execCommand() { return true; },
};
document.getElementById("diaPromptTemplate").content = {textContent: promptMatch[1]};

const nonce = "N".repeat(43);
const localSessionId = "L".repeat(22);
const requestId = "R".repeat(22);

function runEntry(hash, search = "") {
  let replaced = "";
  const entryLocation = {
    href: "https://ortho-reception-backend.onrender.com/static/cockpit/visit-capture/index.html" + search + hash,
    origin: "https://ortho-reception-backend.onrender.com",
    hash,
    replace(target) { replaced = target; },
  };
  vm.runInNewContext(entry, {location: entryLocation, URL}, {filename: "entry.js"});
  return replaced;
}

assert.equal(runEntry("#local_visit_audio_v1=" + nonce), "", "valid local entry must retain its fragment and page");
assert.equal(
  runEntry("#local_visit_audio_v1=" + "N".repeat(42)),
  "https://ortho-reception-backend.onrender.com/static/cockpit/index.html?capture=1",
  "malformed local entry must not bypass the normal shell redirect",
);
assert.equal(
  runEntry("", "?patient_id=SYN-000"),
  "https://ortho-reception-backend.onrender.com/static/cockpit/index.html?capture=1&entry_patient=SYN-000",
  "ordinary Visit Capture entry must keep the integrated-shell patient handoff",
);
const windowEvents = new Map();
const calls = [];
const portMessages = [];
let candidate = null;
let pausePreview = false;
let releasePreview = null;
let pauseRenewal = false;
let releaseRenewal = null;

const port = {
  onmessage: null,
  onmessageerror: null,
  closed: false,
  start() {
    this.onmessage({data: {
      protocol: "local_visit_audio_v1", type: "companion_ready", nonce,
      local_session_id: localSessionId, payload: {},
    }});
  },
  close() { this.closed = true; },
  postMessage(message, transfer = []) {
    portMessages.push({message, transfer});
    if (message.type !== "job_start") return;
    assert.equal(transfer.length, 1);
    assert.equal(transfer[0], message.payload.audio);
    assert.equal(message.payload.mime_type, "audio/webm;codecs=opus");
    const result = {
      schema_version: "local_visit_audio_result_v1",
      protocol: "local_visit_audio_v1",
      local_session_id: localSessionId,
      request_id: message.request_id,
      source_ref: "local-audio:" + message.request_id,
      captured_at: "2026-10-26T10:00:00Z",
      duration_ms: message.payload.duration_ms,
      items: [
        {id: 0, text: "Πόνος στον δεξιό ώμο.", proposed_dispositions: ["reason"], risk_flags: ["laterality"], whisper_flags: []},
        {id: 1, text: "Προτάθηκε φυσιοθεραπεία.", proposed_dispositions: ["clinician_recommendation"], risk_flags: [], whisper_flags: []},
      ],
    };
    this.onmessage({data: {
      protocol: "local_visit_audio_v1", type: "job_result", nonce,
      local_session_id: localSessionId, request_id: message.request_id, payload: {result},
    }});
  },
};

const opener = {
  postMessage(message, origin) {
    assert.equal(origin, "http://127.0.0.1:8765");
    assert.equal(message.protocol, "local_visit_audio_v1");
    assert.equal(message.type, "bootstrap_ready");
    assert.equal(message.nonce, nonce);
    assert.deepEqual(Object.keys(message).sort(), ["nonce", "protocol", "type"]);
    windowEvents.get("message")({
      origin: "http://127.0.0.1:8765", source: opener, ports: [port],
      data: {protocol: "local_visit_audio_v1", type: "bootstrap_port", nonce, local_session_id: localSessionId},
    });
  },
};

async function fetch(url, options = {}) {
  calls.push({url, options});
  let body = {};
  if (url === "/clinical/status") body = {authenticated: true};
  else if (url.startsWith("/clinical/patients?query=")) body = [{patient_id: "SYN-000", demographics: {full_name: "Συνθετικός Ασθενής"}}];
  else if (url === "/clinical/visit-capture/context") body = {context_id: "context-before", patient_id: "SYN-000"};
  else if (url === "/clinical/visit-capture/context/renew") {
    if (pauseRenewal) {
      pauseRenewal = false;
      await new Promise((resolve) => { releaseRenewal = resolve; });
    }
    body = {context_id: "context-after", patient_id: "SYN-000", expires_at: "2026-10-26T11:00:00Z"};
  }
  else if (url === "/clinical/visit-capture/preview") {
    const payload = JSON.parse(options.body);
    candidate = payload.candidate;
    if (pausePreview) {
      pausePreview = false;
      await new Promise((resolve) => { releasePreview = resolve; });
    }
    body = {patient_id: "SYN-000", can_save: payload.recording_patient_confirmed === true, snapshot: "S", brief: "B", detail: "D", gesy_copy: "GESY"};
  } else if (url === "/clinical/visit-capture/save") {
    body = {encounter: {encounter_date: "2026-10-26"}, pending: []};
  } else throw new Error("unexpected request " + url);
  return {ok: true, status: 200, async json() { return body; }};
}

const stream = {getTracks() { return [{stop() {}}]; }};
class MediaRecorder {
  static isTypeSupported(value) { return value === "audio/webm;codecs=opus"; }
  constructor(receivedStream, options) {
    assert.equal(receivedStream, stream);
    assert.equal(options.mimeType, "audio/webm;codecs=opus");
    this.state = "inactive";
  }
  start() { this.state = "recording"; }
  stop() {
    this.state = "inactive";
    this.ondataavailable({data: new Blob(["synthetic-audio"], {type: "audio/webm;codecs=opus"})});
    this.onstop();
  }
}

const memory = new Map();
const crypto = {
  randomUUID() { return "synthetic-client-session"; },
  getRandomValues(array) { array.fill(17); return array; },
};
const location = {hash: "#local_visit_audio_v1=" + nonce, pathname: "/static/cockpit/visit-capture/index.html", search: ""};
const window = {
  opener,
  MediaRecorder,
  addEventListener(name, handler) { windowEvents.set(name, handler); },
  removeEventListener(name, handler) { if (windowEvents.get(name) === handler) windowEvents.delete(name); },
};
const scope = {
  document, window, location, history: {replaceState() { location.hash = ""; }},
  navigator: {mediaDevices: {async getUserMedia() { return stream; }}, clipboard: {async writeText() {}}},
  MediaRecorder, Blob, TextEncoder, fetch, crypto, URLSearchParams,
  sessionStorage: {getItem(key) { return memory.get(key) || null; }, setItem(key, value) { memory.set(key, value); }},
  btoa(value) { return Buffer.from(value, "binary").toString("base64"); },
  setTimeout(fn) { queueMicrotask(fn); return 1; }, clearTimeout() {},
  setInterval() { return 1; }, clearInterval() {}, console,
};

function get(id) { return document.getElementById(id); }
async function settle() { await new Promise((resolve) => setImmediate(resolve)); }

(async () => {
  vm.runInNewContext(app, scope, {filename: "app.js"});
  await settle();
  assert.equal(get("localAudioState").textContent, "Αναμονή ασθενούς");

  get("recordModeBtn").dispatch("click");
  get("patientSearch").value = "SYN-000";
  get("patientSearch").dispatch("input");
  await settle();
  get("patientSelect").value = "SYN-000";
  await get("confirmPatientBtn").dispatch("click");
  get("patientInformed").checked = true;
  get("patientInformed").dispatch("change");
  assert.equal(get("startRecordingBtn").disabled, false);

  await get("startRecordingBtn").dispatch("click");
  assert.equal(get("stopRecordingBtn").disabled, false);
  get("stopRecordingBtn").dispatch("click");
  await settle();
  assert.equal(get("localReviewItems").children.length, 2);
  assert.equal(get("localIdentityConfirmed").checked, false);
  assert.ok(portMessages.some(({message}) => message.type === "job_start"));
  assert.ok(portMessages.some(({message}) => message.type === "job_ack"));

  const recommendation = get("localReviewItems").children[1];
  recommendation.querySelector(".local-decision-type").value = "therapy";
  get("localIdentityConfirmed").checked = true;
  get("localIdentityConfirmed").dispatch("change");
  get("localRisksConfirmed").checked = true;
  get("localRisksConfirmed").dispatch("change");
  assert.equal(get("prepareLocalCandidateBtn").disabled, false);
  await get("prepareLocalCandidateBtn").dispatch("click");

  assert.ok(candidate, "local result reached protected preview only after explicit review");
  assert.equal(candidate.reason_for_visit.text, "Πόνος στον δεξιό ώμο.");
  assert.equal(candidate.recommendations[0].text, "Προτάθηκε φυσιοθεραπεία.");
  assert.equal(candidate.recommendations[0].type, "therapy");
  assert.equal(candidate.source_bindings[0].role, "local_visit_audio");
  const renewIndex = calls.findIndex(({url}) => url === "/clinical/visit-capture/context/renew");
  const previewIndex = calls.findIndex(({url}) => url === "/clinical/visit-capture/preview");
  assert.ok(renewIndex >= 0 && previewIndex > renewIndex, "CAS renewal precedes preview");
  assert.equal(JSON.parse(calls[previewIndex].options.body).context_id, "context-after");
  assert.equal(JSON.parse(calls[previewIndex].options.body).recording_patient_confirmed, true);
  assert.equal(get("gesyCopyRow").hidden, false);

  pausePreview = true;
  get("candidateInput").dispatch("input");
  await settle();
  assert.ok(releasePreview, "one Preview response remains in flight");
  get("localIdentityConfirmed").checked = false;
  get("localIdentityConfirmed").dispatch("change");
  assert.equal(get("saveBtn").disabled, true, "withdrawal immediately revokes Save");
  releasePreview();
  await settle();
  assert.equal(get("saveBtn").disabled, true, "late Preview success cannot restore Save");

  get("localIdentityConfirmed").checked = true;
  get("localIdentityConfirmed").dispatch("change");
  await settle();
  assert.equal(get("saveBtn").disabled, false, "fresh confirmed Preview restores Save");

  pauseRenewal = true;
  const interruptedSave = get("saveBtn").dispatch("click");
  await settle();
  assert.ok(releaseRenewal, "Save waits for exact-context renewal");
  get("localIdentityConfirmed").checked = false;
  get("localIdentityConfirmed").dispatch("change");
  get("localIdentityConfirmed").checked = true;
  get("localIdentityConfirmed").dispatch("change");
  releaseRenewal();
  await interruptedSave;
  await settle();
  assert.equal(calls.filter(({url}) => url === "/clinical/visit-capture/save").length, 0,
    "withdrawal during renewal cannot be undone by a quick recheck");

  assert.equal(get("saveBtn").disabled, false, "new Preview is required after withdrawal");
  await get("saveBtn").dispatch("click");
  await settle();
  const saveCall = calls.find(({url}) => url === "/clinical/visit-capture/save");
  assert.ok(saveCall);
  assert.equal(JSON.parse(saveCall.options.body).recording_patient_confirmed, true);
  console.log("Visit Capture local-audio reviewed flow: PASS");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
