"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const html = fs.readFileSync("static/cockpit/pending-one-tap-synthetic.html", "utf8");
const js = fs.readFileSync("static/cockpit/pending-one-tap-synthetic.js", "utf8");
const home = fs.readFileSync("static/cockpit/index.html", "utf8");
assert.match(home, /pending-one-tap-synthetic\.html/);
assert.match(html, /Δοκιμαστικός ασθενής/);
assert.doesNotMatch(html, /staffKey|patientId|type="password"/);
// Strip source comments before checking executable code for network/storage access.\nconst executable = js.replace(/^\\s*\\/\\/.*$/gm, "");\nassert.doesNotMatch(executable, /fetch\\s*\\(|localStorage|sessionStorage|XMLHttpRequest/);
const els = new Map();
for (const name of ["question","yes","notYet","undo","reset","observation","patientStatus","briefStatus","audit"]) {
  const handlers = {};
  els.set(name, {
    textContent: "", disabled: false, handlers,
    classList: { toggle: () => {} },
    addEventListener: (name, fn) => { handlers[name] = fn; }
  });
}
const ctx = {document: {getElementById: id => els.get(id)}, fetch: () => { throw Error("Unexpected network"); }};
vm.runInNewContext(js, ctx, {filename: "pending-one-tap-synthetic.js"});
const el = id => els.get(id);
const click = id => el(id).handlers.click();
const same = () => assert.equal(el("patientStatus").textContent, el("briefStatus").textContent);
assert.match(el("question").textContent, /Προσκομίστηκε/);
same();
click("notYet"); assert.match(el("question").textContent, /Προσκομίστηκε/); same();
click("yes"); assert.match(el("question").textContent, /Εξετάστηκαν οι εικόνες/); same();
click("notYet"); assert.match(el("question").textContent, /Εξετάστηκαν οι εικόνες/); same();
click("yes"); assert.equal(el("yes").disabled, true); same();
assert.match(el("patientStatus").textContent, /Ολοκληρώθηκε/);
click("undo"); assert.equal(el("yes").disabled, false); same();
assert.match(el("question").textContent, /Εξετάστηκαν/);
click("reset"); assert.match(el("question").textContent, /Προσκομίστηκε/);
assert.match(el("audit").textContent, /κανένα/);
console.log("PASS: synthetic one-tap controller, two views, undo/reset, no auth or network");
