"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const launcher = fs.readFileSync("local_visit_audio/launcher.js", "utf8");
const PROTOCOL = "local_visit_audio_v1";
const PRODUCTION_ORIGIN = "https://ortho-reception-backend.onrender.com";
const REQUEST = "R".repeat(22);

function element() {
  const listeners = new Map();
  return {
    textContent: "",
    addEventListener(name, handler) { listeners.set(name, handler); },
    dispatch(name, event = {}) { return listeners.get(name)(event); },
  };
}

function createHarness(mode) {
  const button = element();
  const status = element();
  const windowEvents = new Map();
  const fetchCalls = [];
  const order = [];
  const childMessages = [];
  let childPort = null;
  let sessionId = "";
  let openedUrl = "";

  class Port {
    constructor() { this.peer = null; this.onmessage = null; this.onmessageerror = null; this.closed = false; }
    postMessage(data) { this.peer?.onmessage?.({data, ports: []}); }
    close() { this.closed = true; }
  }
  class MessageChannel {
    constructor() {
      this.port1 = new Port();
      this.port2 = new Port();
      this.port1.peer = this.port2;
      this.port2.peer = this.port1;
    }
  }

  const child = {
    closed: false,
    postMessage(message, origin, ports) {
      assert.equal(origin, PRODUCTION_ORIGIN);
      assert.equal(message.type, "bootstrap_port");
      sessionId = message.local_session_id;
      childPort = ports[0];
      childPort.onmessage = (event) => childMessages.push(event.data);
    },
  };
  const document = {
    getElementById(id) { return id === "status" ? status : button; },
    querySelector() { return {content: "csrf-token"}; },
  };
  const window = {
    open(url) { openedUrl = url; return child; },
    addEventListener(name, handler) { windowEvents.set(name, handler); },
    removeEventListener(name, handler) { if (windowEvents.get(name) === handler) windowEvents.delete(name); },
  };

  function fetch(url, options) {
    fetchCalls.push({url, options});
    if (url === "/api/cancel") {
      order.push("cancel");
      return Promise.resolve({ok: true});
    }
    assert.equal(url, "/api/process");
    if (mode === "complete") {
      const result = {
        schema_version: "local_visit_audio_result_v1", protocol: PROTOCOL,
        local_session_id: sessionId, request_id: REQUEST,
        source_ref: "local-audio:" + REQUEST,
        captured_at: "2026-10-26T10:00:00Z", duration_ms: 1000,
        items: [{id: 0, text: "Συνθετικό", proposed_dispositions: ["reason"], risk_flags: [], whisper_flags: []}],
      };
      const bytes = new TextEncoder().encode(JSON.stringify(result));
      return Promise.resolve({ok: true, status: 200, async arrayBuffer() { return bytes.buffer; }});
    }
    return new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => {
        order.push("abort");
        const error = new Error("aborted");
        error.name = "AbortError";
        reject(error);
      });
    });
  }

  const timers = new Map();
  let nextTimer = 1;
  const scope = {
    document, window, MessageChannel, fetch, AbortController, ArrayBuffer,
    TextEncoder, TextDecoder, Date, JSON, Set,
    crypto: {getRandomValues(array) { array.fill(17); return array; }},
    btoa(value) { return Buffer.from(value, "binary").toString("base64"); },
    setTimeout(handler) { const id = nextTimer++; timers.set(id, handler); return id; },
    clearTimeout(id) { timers.delete(id); },
    console,
  };
  vm.runInNewContext(launcher, scope, {filename: "launcher.js"});

  function connectAndStart() {
    button.dispatch("click");
    const nonce = openedUrl.split("#local_visit_audio_v1=")[1];
    windowEvents.get("message")({
      origin: PRODUCTION_ORIGIN, source: child, ports: [],
      data: {protocol: PROTOCOL, type: "bootstrap_ready", nonce},
    });
    assert.ok(childPort);
    childPort.postMessage({
      protocol: PROTOCOL, type: "job_start", nonce, local_session_id: sessionId,
      request_id: REQUEST,
      payload: {mime_type: "audio/webm;codecs=opus", duration_ms: 1000, byte_length: 4, audio: new Uint8Array([1, 2, 3, 4]).buffer},
    });
  }
  function acknowledgeResult() {
    const result = childMessages.find((message) => message.type === "job_result");
    assert.ok(result);
    childPort.postMessage({
      protocol: PROTOCOL, type: "job_ack", nonce: result.nonce,
      local_session_id: sessionId, request_id: REQUEST, payload: {},
    });
  }
  function triggerPortError() { childPort.peer.onmessageerror(); }
  function sendInvalidActiveMessage() {
    const ready = childMessages.find((message) => message.type === "companion_ready");
    childPort.postMessage({
      protocol: PROTOCOL, type: "unexpected", nonce: ready.nonce,
      local_session_id: sessionId, request_id: REQUEST, payload: {},
    });
  }
  return {
    connectAndStart, acknowledgeResult, triggerPortError, sendInvalidActiveMessage,
    childMessages, fetchCalls, order, windowEvents, getSessionId: () => sessionId,
  };
}

async function settle() {
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
}

(async () => {
  async function verifyActiveClosure(trigger) {
    const active = createHarness("pending");
    active.connectAndStart();
    await settle();
    trigger(active);
    await settle();
    assert.deepEqual(active.order.slice(0, 2), ["cancel", "abort"], "cancel is issued before local fetch abort/state clear");
    const cancel = active.fetchCalls.find((call) => call.url === "/api/cancel");
    assert.ok(cancel);
    assert.equal(cancel.options.keepalive, true);
    assert.equal(cancel.options.headers["X-Local-Session-Id"], active.getSessionId());
    assert.equal(cancel.options.headers["X-Request-Id"], REQUEST);
    assert.ok(!active.childMessages.some((message) => message.type === "job_result"));
  }

  await verifyActiveClosure((active) => active.windowEvents.get("pagehide")());
  await verifyActiveClosure((active) => active.triggerPortError());
  await verifyActiveClosure((active) => active.sendInvalidActiveMessage());

  const completed = createHarness("complete");
  completed.connectAndStart();
  await settle();
  const result = completed.childMessages.find((message) => message.type === "job_result");
  assert.ok(result, "validated local result reaches child");
  // Normal terminal acknowledgement must close without a redundant cancellation.
  completed.acknowledgeResult();
  await settle();
  assert.equal(completed.fetchCalls.filter((call) => call.url === "/api/cancel").length, 0);
  console.log("Local launcher lifecycle cancellation: PASS");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
