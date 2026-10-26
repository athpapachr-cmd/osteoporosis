(() => {
  "use strict";

  const PROTOCOL = "local_visit_audio_v1";
  const PRODUCTION_ORIGIN = "https://ortho-reception-backend.onrender.com";
  const CAPTURE_URL = PRODUCTION_ORIGIN + "/static/cockpit/visit-capture/index.html";
  const ERROR_CODES = new Set([
    "filevault_unavailable", "invalid_request", "audio_limit", "decode_failed",
    "transcription_failed", "classification_failed", "validation_failed", "timeout",
    "cancelled", "cleanup_failed", "internal_failure",
  ]);
  const DISPOSITIONS = new Set([
    "reason", "patient_history", "objective_result", "clinician_interpretation",
    "option_discussed", "clinician_recommendation", "final_decision", "completed_action",
    "patient_accepted", "patient_declined", "patient_undecided", "follow_up",
    "pending_obligation", "safety_net", "uncertainty", "administrative", "unclassified",
  ]);
  const RISK_FLAGS = new Set([
    "digit_or_measurement", "medication", "laterality", "negation",
    "uncertainty_or_conflict", "unclassified", "semantic_status_conflict",
  ]);
  const WHISPER_FLAGS = new Set(["low_confidence", "no_speech_anomaly"]);
  const status = document.getElementById("status");
  const button = document.getElementById("openCockpit");
  const csrf = document.querySelector('meta[name="csrf-token"]').content;
  let child = null;
  let port = null;
  let nonce = "";
  let localSessionId = "";
  let activeRequest = "";
  let controller = null;
  let terminalTimer = 0;
  let processingTimer = 0;

  function randomBase64Url(bytes) {
    const data = new Uint8Array(bytes);
    crypto.getRandomValues(data);
    let binary = "";
    data.forEach((value) => { binary += String.fromCharCode(value); });
    return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function exactKeys(value, keys) {
    return value && typeof value === "object" && !Array.isArray(value)
      && Object.keys(value).sort().join("|") === [...keys].sort().join("|");
  }

  function envelope(type, requestId, payload) {
    const message = {protocol: PROTOCOL, type, nonce, local_session_id: localSessionId, payload};
    if (requestId) message.request_id = requestId;
    return message;
  }

  function closeChannel(message) {
    clearTimeout(terminalTimer);
    clearTimeout(processingTimer);
    if (controller) controller.abort();
    controller = null;
    activeRequest = "";
    if (port) port.close();
    port = null;
    status.textContent = message;
  }

  function cancelLocalProcessing() {
    if (!activeRequest) return;
    fetch("/api/cancel", {
      method: "POST",
      credentials: "same-origin",
      keepalive: true,
      headers: {
        "X-CSRF-Token": csrf,
        "X-Local-Session-Id": localSessionId,
        "X-Request-Id": activeRequest,
      },
    }).catch(() => {});
  }

  function sendError(requestId, code) {
    if (!port) return;
    if (!ERROR_CODES.has(code)) code = "internal_failure";
    port.postMessage(envelope("job_error", requestId, {code, request_id: requestId}));
    terminalTimer = setTimeout(() => closeChannel("Η τοπική εργασία έκλεισε χωρίς επιβεβαίωση."), 5000);
  }

  function uniqueAllowed(values, allowed) {
    return Array.isArray(values) && new Set(values).size === values.length
      && values.every((value) => allowed.has(value));
  }

  function validResult(result, requestId) {
    if (!exactKeys(result, ["schema_version", "protocol", "local_session_id", "request_id", "source_ref", "captured_at", "duration_ms", "items"])) return false;
    if (result.schema_version !== "local_visit_audio_result_v1" || result.protocol !== PROTOCOL
        || result.local_session_id !== localSessionId || result.request_id !== requestId
        || result.source_ref !== "local-audio:" + requestId
        || typeof result.captured_at !== "string" || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z$/.test(result.captured_at)
        || Number.isNaN(Date.parse(result.captured_at))
        || !Number.isInteger(result.duration_ms) || result.duration_ms < 1 || result.duration_ms > 1800000
        || !Array.isArray(result.items) || result.items.length < 1 || result.items.length > 512
        || new TextEncoder().encode(JSON.stringify(result)).byteLength > 1048576) return false;
    return result.items.every((item, index) => exactKeys(item, ["id", "text", "proposed_dispositions", "risk_flags", "whisper_flags"])
      && item.id === index && typeof item.text === "string" && item.text.length >= 1 && item.text.length <= 4000
      && new TextEncoder().encode(item.text).byteLength <= 16384
      && uniqueAllowed(item.proposed_dispositions, DISPOSITIONS) && item.proposed_dispositions.length > 0
      && uniqueAllowed(item.risk_flags, RISK_FLAGS)
      && uniqueAllowed(item.whisper_flags, WHISPER_FLAGS));
  }

  async function processJob(message) {
    const requestId = message.request_id;
    const payload = message.payload;
    if (
      activeRequest || !/^[A-Za-z0-9_-]{22}$/.test(requestId)
      || !exactKeys(payload, ["mime_type", "duration_ms", "byte_length", "audio"])
      || payload.mime_type !== "audio/webm;codecs=opus"
      || !(payload.audio instanceof ArrayBuffer)
      || payload.audio.byteLength !== payload.byte_length
      || payload.byte_length < 1 || payload.byte_length > 67108864
      || !Number.isInteger(payload.duration_ms) || payload.duration_ms < 1 || payload.duration_ms > 1800000
    ) {
      sendError(requestId || "", "invalid_request");
      return;
    }
    activeRequest = requestId;
    controller = new AbortController();
    let timedOut = false;
    processingTimer = setTimeout(() => {
      timedOut = true;
      cancelLocalProcessing();
      if (controller) controller.abort();
    }, 600000);
    try {
      status.textContent = "Τοπική μεταφορά και επεξεργασία…";
      port.postMessage(envelope("job_progress", requestId, {stage: "transferring"}));
      const response = await fetch("/api/process", {
        method: "POST",
        credentials: "same-origin",
        headers: {
          "Content-Type": payload.mime_type,
          "X-CSRF-Token": csrf,
          "X-Local-Session-Id": localSessionId,
          "X-Request-Id": requestId,
          "X-Duration-Ms": String(payload.duration_ms),
        },
        body: payload.audio,
        signal: controller.signal,
      });
      const encoded = await response.arrayBuffer();
      if (encoded.byteLength > 1048576) throw new Error("validation_failed");
      let result;
      try { result = JSON.parse(new TextDecoder("utf-8", {fatal: true}).decode(encoded)); }
      catch (_) { throw new Error("validation_failed"); }
      if (!response.ok) {
        if (!exactKeys(result, ["code", "request_id"]) || result.request_id !== requestId
            || !ERROR_CODES.has(result.code)) throw new Error("validation_failed");
        throw new Error(result.code);
      }
      if (!validResult(result, requestId)) throw new Error("validation_failed");
      port.postMessage(envelope("job_result", requestId, {result}));
      status.textContent = "Η τοπική επεξεργασία ολοκληρώθηκε. Αναμένεται η επιβεβαίωση στο Cockpit.";
      terminalTimer = setTimeout(() => closeChannel("Η επιβεβαίωση άργησε· ξεκινήστε νέα τοπική συνεδρία."), 5000);
    } catch (error) {
      const code = timedOut ? "timeout" : (error.name === "AbortError" ? "cancelled" : error.message);
      sendError(requestId, ERROR_CODES.has(code) ? code : "internal_failure");
    } finally {
      clearTimeout(processingTimer);
      processingTimer = 0;
      controller = null;
    }
  }

  function onPortMessage(event) {
    const message = event.data;
    if (!message || message.protocol !== PROTOCOL || message.nonce !== nonce
        || message.local_session_id !== localSessionId || message.request_id !== activeRequest
        || !exactKeys(message, ["protocol", "type", "nonce", "local_session_id", "request_id", "payload"])
        || (event.ports && event.ports.length !== 0)) {
      closeChannel("Απορρίφθηκε μη έγκυρο ή παλιό μήνυμα.");
      return;
    }
    if (message.type === "job_cancel" && exactKeys(message.payload, [])) {
      cancelLocalProcessing();
      if (controller) controller.abort();
      closeChannel("Η εργασία ακυρώθηκε και τα προσωρινά δεδομένα διαγράφηκαν.");
    } else if (message.type === "job_ack" && exactKeys(message.payload, [])) {
      closeChannel("Ολοκληρώθηκε. Για νέα επίσκεψη ανοίξτε νέα τοπική συνεδρία.");
    } else {
      closeChannel("Απορρίφθηκε μήνυμα εκτός σειράς.");
    }
  }

  function onBootstrap(event) {
    const message = event.data;
    if (
      event.origin !== PRODUCTION_ORIGIN || event.source !== child
      || !exactKeys(message, ["protocol", "type", "nonce"])
      || message.protocol !== PROTOCOL || message.type !== "bootstrap_ready" || message.nonce !== nonce
    ) {
      (event.ports || []).forEach((candidate) => candidate.close());
      window.removeEventListener("message", onBootstrap);
      closeChannel("Απορρίφθηκε μη έγκυρη σύνδεση Visit Capture.");
      return;
    }
    window.removeEventListener("message", onBootstrap);
    const channel = new MessageChannel();
    port = channel.port1;
    port.onmessage = (portEvent) => {
      const data = portEvent.data;
      if (data?.type === "job_start") {
        if (
          data.protocol !== PROTOCOL || data.nonce !== nonce || data.local_session_id !== localSessionId
          || !exactKeys(data, ["protocol", "type", "nonce", "local_session_id", "request_id", "payload"])
          || (portEvent.ports && portEvent.ports.length !== 0)
        ) {
          closeChannel("Απορρίφθηκε μη έγκυρη έναρξη εργασίας.");
          return;
        }
        processJob(data);
      } else {
        onPortMessage(portEvent);
      }
    };
    port.onmessageerror = () => closeChannel("Το ασφαλές τοπικό κανάλι έκλεισε.");
    child.postMessage(
      {protocol: PROTOCOL, type: "bootstrap_port", nonce, local_session_id: localSessionId},
      PRODUCTION_ORIGIN,
      [channel.port2],
    );
    port.postMessage(envelope("companion_ready", "", {}));
    status.textContent = "Το Visit Capture συνδέθηκε τοπικά.";
  }

  button.addEventListener("click", () => {
    if (child && !child.closed) return;
    nonce = randomBase64Url(32);
    localSessionId = randomBase64Url(16);
    child = window.open(CAPTURE_URL + "#local_visit_audio_v1=" + nonce, "_blank");
    if (!child) {
      status.textContent = "Ο browser απέκλεισε το νέο παράθυρο.";
      return;
    }
    window.addEventListener("message", onBootstrap);
    status.textContent = "Αναμονή ασφαλούς σύνδεσης με Visit Capture…";
    setTimeout(() => {
      if (!port) {
        window.removeEventListener("message", onBootstrap);
        status.textContent = "Η σύνδεση έληξε. Δοκιμάστε ξανά.";
      }
    }, 10000);
  });

  window.addEventListener("pagehide", () => closeChannel("Η τοπική συνεδρία έκλεισε."));
})();
