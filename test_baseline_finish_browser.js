"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = __dirname;
const coordinatorPath = path.join(ROOT, "static/baseline-audit/finalization-coordinator.js");
const pilotPath = path.join(ROOT, "static/baseline-audit/pilot-completion.js");
const registryPath = path.join(ROOT, "static/baseline-audit/patient-registry.js");
const appPath = path.join(ROOT, "static/baseline-audit/app.js");

const coordinatorSource = fs.readFileSync(coordinatorPath, "utf8");
const pilotSource = fs.readFileSync(pilotPath, "utf8");
const registrySource = fs.readFileSync(registryPath, "utf8");
const appSource = fs.readFileSync(appPath, "utf8");

class FakeButton {
  constructor() {
    this.handlers = [];
    this.disabled = false;
  }
  addEventListener(type, handler, options) {
    if (type !== "click") return;
    this.handlers.push({ handler, capture: options === true || Boolean(options?.capture) });
  }
  _dispatch() {
    let stopped = false;
    const event = {
      preventDefault() {},
      stopImmediatePropagation() { stopped = true; }
    };
    const ordered = [...this.handlers].sort((a, b) => Number(b.capture) - Number(a.capture));
    const results = [];
    for (const entry of ordered) {
      if (stopped) break;
      results.push(entry.handler(event));
    }
    return results;
  }
  click() { this._dispatch(); }
  async clickAndWait() {
    const results = this._dispatch();
    await Promise.all(results.filter((x) => x && typeof x.then === "function"));
  }
}

function createStorage(initial = {}) {
  const store = new Map(Object.entries(initial));
  return {
    getItem(key) { return store.has(key) ? store.get(key) : null; },
    setItem(key, value) { store.set(key, String(value)); },
    removeItem(key) { store.delete(key); }
  };
}

function assertRuntimeOwnershipContract() {
  assert.match(registrySource, /shouldSyncDraftOnSave\(\)/, "ordinary Save must consult the finalization guard");
  assert.match(registrySource, /window\.ClinicalRegistry\s*=\s*Object\.freeze/, "registry must export the strict finalization API");
  assert.doesNotMatch(registrySource, /#finishVisitBtn[\s\S]{0,120}addEventListener/, "patient registry must not own a second Finish click listener");

  const coordinatorIndex = appSource.indexOf('loadScript("./finalization-coordinator.js")');
  const registryIndex = appSource.indexOf('loadScript("./patient-registry.js")');
  const pilotIndex = appSource.indexOf('loadScript("./pilot-completion.js")');
  assert.ok(coordinatorIndex >= 0 && registryIndex > coordinatorIndex && pilotIndex > registryIndex,
    "load order must be coordinator -> registry -> single Finish owner");
}

function createHarness({ finalizeImpl }) {
  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const caseId = "case-1";
  const initialCase = {
    internal_uuid: caseId,
    baseline_phase: "pilot",
    encounter_date: "2026-08-30",
    step6: { capture_quality: { ready_for_audit: "yes", completion_time_minutes: 4 } }
  };
  const localStorage = createStorage({
    [STORAGE_KEY]: JSON.stringify([initialCase]),
    [ACTIVE_KEY]: caseId
  });

  const saveDraft = new FakeButton();
  const saveTop = new FakeButton();
  const finish = new FakeButton();
  const next = new FakeButton();
  const draftStatus = { textContent: "" };
  const pilotPill = { textContent: "" };
  const step6Panel = { hidden: false };
  const alerts = [];
  let draftSyncCount = 0;
  let finalizeCallCount = 0;
  let serverPayloadSnapshot = null;

  const document = {
    querySelector(selector) {
      return ({
        "#saveDraftBtn": saveDraft,
        "#saveTopBtn": saveTop,
        "#finishVisitBtn": finish,
        "#nextBtn": next,
        "#draftStatus": draftStatus,
        "#pilotPill": pilotPill,
        '[data-step-panel="6"]': step6Panel
      })[selector] || null;
    }
  };

  const context = {
    console,
    setTimeout,
    clearTimeout,
    Date,
    Promise,
    JSON,
    Object,
    Error,
    localStorage,
    document,
    getComputedStyle() { return { display: "block" }; }
  };
  context.window = context;
  context.globalThis = context;
  context.alert = (message) => alerts.push(message);
  vm.createContext(context);
  vm.runInContext(coordinatorSource, context, { filename: "finalization-coordinator.js" });

  // Mirror the real Steps 3-6 Save behavior: module persistence occurs in setTimeout(0).
  saveDraft.addEventListener("click", () => {
    setTimeout(() => {
      const cases = JSON.parse(localStorage.getItem(STORAGE_KEY));
      cases[0].step6.final_marker = "saved-before-server-finalization";
      localStorage.setItem(STORAGE_KEY, JSON.stringify(cases));
    }, 0);
  });

  // Mirror patient-registry ordinary Save using the same shared guard.
  saveDraft.addEventListener("click", () => {
    if (!context.window.BaselineFinalizationCoordinator.shouldSyncDraftOnSave()) return;
    draftSyncCount += 1;
  });

  context.window.ClinicalRegistry = {
    async finalizeActiveEncounter() {
      finalizeCallCount += 1;
      const cases = JSON.parse(localStorage.getItem(STORAGE_KEY));
      serverPayloadSnapshot = JSON.parse(JSON.stringify(cases[0]));
      return finalizeImpl({ payload: serverPayloadSnapshot });
    }
  };

  vm.runInContext(pilotSource, context, { filename: "pilot-completion.js" });

  return {
    finish,
    draftStatus,
    alerts,
    localStorage,
    context,
    get draftSyncCount() { return draftSyncCount; },
    get finalizeCallCount() { return finalizeCallCount; },
    get serverPayloadSnapshot() { return serverPayloadSnapshot; }
  };
}

async function testSuccessfulAuthoritativeFinish() {
  const h = createHarness({ finalizeImpl: async () => ({ status: "completed" }) });
  await h.finish.clickAndWait();

  assert.strictEqual(h.draftSyncCount, 0, "ordinary draft sync must be suppressed during authoritative Finish");
  assert.strictEqual(h.finalizeCallCount, 1, "server finalization must run exactly once");
  assert.strictEqual(h.serverPayloadSnapshot.step6.final_marker, "saved-before-server-finalization", "final module state must persist before server finalization");
  assert.strictEqual(h.serverPayloadSnapshot.pilot_completion.status, "complete", "server final payload must include pilot completion");
  assert.match(h.draftStatus.textContent, /protected server ως completed/);
  assert.strictEqual(h.alerts.length, 1);
  assert.strictEqual(h.context.window.BaselineFinalizationCoordinator.isAuthoritativeFinishInProgress(), false);
}

async function testFailedProtectedCompletionIsExplicit() {
  const h = createHarness({ finalizeImpl: async () => { throw new Error("No active protected patient"); } });
  await h.finish.clickAndWait();

  assert.strictEqual(h.draftSyncCount, 0, "failed Finish must not fall back to ordinary draft sync");
  assert.strictEqual(h.finalizeCallCount, 1);
  assert.match(h.draftStatus.textContent, /δεν επιβεβαιώθηκε protected completion/);
  assert.doesNotMatch(h.draftStatus.textContent, /protected server ως completed/);
  const cases = JSON.parse(h.localStorage.getItem("osteoporosis.baselineAuditPilot.v1_1"));
  assert.strictEqual(cases[0].pilot_completion.status, "complete", "local data should remain intact for retry");
  assert.strictEqual(h.context.window.BaselineFinalizationCoordinator.isAuthoritativeFinishInProgress(), false);
}

(async () => {
  assertRuntimeOwnershipContract();
  await testSuccessfulAuthoritativeFinish();
  await testFailedProtectedCompletionIsExplicit();
  console.log("baseline authoritative Finish browser regression: OK");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
