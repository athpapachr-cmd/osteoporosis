"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const registry = fs.readFileSync(path.join(ROOT, "static/baseline-audit/patient-registry.js"), "utf8");
const shell = fs.readFileSync(path.join(ROOT, "static/baseline-audit/clinical-workspace-shell.js"), "utf8");
const completion = fs.readFileSync(path.join(ROOT, "static/baseline-audit/pilot-completion.js"), "utf8");
const progress = fs.readFileSync(path.join(ROOT, "static/baseline-audit/whole-form-progress.js"), "utf8");
const app = fs.readFileSync(path.join(ROOT, "static/baseline-audit/app.js"), "utf8");

function assertServerAuthoritativeIdentity() {
  assert.match(registry, /expected_updated_at:\s*link\.updated_at/, "existing encounter writes must carry the last server version token");
  assert.match(registry, /err\.status\s*===\s*409/, "409 must have an explicit client conflict branch");
  assert.match(registry, /conflictedUuids\.add/, "conflicted encounter must enter a blocked autosave state");
  assert.match(registry, /conflictedUuids\.has/, "further sync must check conflict state");
  assert.match(registry, /Φόρτωση server έκδοσης/, "conflict UX must provide explicit server reload action");
  assert.match(registry, /confirmDiscard/, "server reload must require explicit local-discard confirmation");
  assert.match(registry, /patient_id:\s*row\.patient_id[\s\S]*encounter_id:\s*row\.encounter_id[\s\S]*updated_at:\s*row\.updated_at/, "local cache link must retain server patient/encounter/version identity");
}

function assertAutosaveAndSingleFlight() {
  assert.match(registry, /AUTOSAVE_DELAY_MS\s*=\s*900/, "meaningful edits should debounce before autosave");
  assert.match(registry, /syncQueue\s*=\s*Promise\.resolve\(\)/, "server writes need one serialized queue");
  assert.match(registry, /const task = syncQueue\.then\(/, "every sync must be sequenced behind the previous sync");
  assert.match(registry, /document\.addEventListener\("input", scheduleAutosave\)/, "form input must feed autosave");
  assert.match(registry, /document\.addEventListener\("change", scheduleAutosave\)/, "form changes must feed autosave");
  assert.match(registry, /shouldSyncDraftOnSave\(\)/, "autosave must preserve authoritative Finish coordination");
}

function assertNewVisitAndRecoverySemantics() {
  assert.match(registry, /Επίλεξε ή δημιούργησε protected patient πριν από Νέα επίσκεψη/, "orphan local clinical visit creation must be blocked");
  assert.match(registry, /recoverExistingEncounter/, "new-visit retry must recover an already-created server encounter before POSTing again");
  assert.match(registry, /row\.payload\?\.internal_uuid === current\.internal_uuid/, "retry recovery must use stable payload internal_uuid within the active patient");
  assert.match(registry, /workflow_mode = "clinical"/, "new server payloads must be explicitly clinical workflow payloads");
  assert.match(registry, /baseline_phase === "pilot"\) payload\.baseline_phase = "clinical"/, "legacy browser pilot defaults must not become server truth for new clinical visits");
}

function assertPilotShellRetiredWithoutSecondFinishOwner() {
  assert.match(shell, /Νέα επίσκεψη/, "early shell must present visit language");
  assert.match(shell, /Επισκέψεις/, "early shell must replace local Cases language");
  assert.match(shell, /Protected clinical mode:/, "privacy strip must reflect protected server mode");
  assert.match(shell, /sampleBox\.hidden = true/, "manual baseline-sample control must be removed from normal workflow");
  assert.match(shell, /cancel\.hidden = true/, "legacy local-case cancel control must not remain in the normal clinical workflow");
  assert.match(shell, /Privacy — clinical workspace/, "privacy modal must no longer describe pilot/local-only workflow");
  assert.match(shell, /browser cache είναι προσωρινό working cache/, "privacy modal must state that browser cache is not authoritative");
  assert.match(app, /clinicalWorkspacePreloadGuard/, "legacy pilot identifiers must be hidden before shell rewrite");
  assert.match(app, /loadScript\("\.\/clinical-workspace-shell\.js"\)/, "clinical shell rewrite must run immediately after app core");
  assert.ok(app.indexOf('loadScript("./clinical-workspace-shell.js")') < app.indexOf('loadScript("./bmi-behavior.js")'), "shell rewrite must precede later runtime modules");

  assert.match(completion, /encounter_completion/, "new completion metadata must be encounter-centric");
  assert.doesNotMatch(completion, /Pilot case|pilot case/, "clinician-facing Finish text must not mention pilot cases");
  assert.doesNotMatch(progress, /choice\("first_core_baseline_encounter_for_patient"\)/, "hidden baseline-sample field must not remain a completion requirement");
  assert.doesNotMatch(registry, /#finishVisitBtn[\s\S]{0,120}addEventListener/, "patient registry must not create another Finish listener");

  const coordinatorIndex = app.indexOf('loadScript("./finalization-coordinator.js")');
  const registryIndex = app.indexOf('loadScript("./patient-registry.js")');
  const completionIndex = app.indexOf('loadScript("./pilot-completion.js")');
  assert.ok(coordinatorIndex >= 0 && registryIndex > coordinatorIndex && completionIndex > registryIndex,
    "production-proven Finish owner load order must remain coordinator -> registry -> completion owner");
}

assertServerAuthoritativeIdentity();
assertAutosaveAndSingleFlight();
assertNewVisitAndRecoverySemantics();
assertPilotShellRetiredWithoutSecondFinishOwner();
console.log("C2 server-authoritative workspace wiring regression: OK");
