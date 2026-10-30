/* iCloud-specific frontend contract: no HTML interpolation, no mailbox mutations. */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const ui = fs.readFileSync("static/cockpit/icloud-inbox.js", "utf8");
const shell = fs.readFileSync("static/cockpit/doctor-shell.js", "utf8");
const html = fs.readFileSync("static/cockpit/index.html", "utf8");

assert.match(html, /id="doctorIcloudInboxView"/);
assert.match(html, /id="doctorIcloudInboxRows"/);
assert.match(html, /id="doctorIcloudInboxDetails"/);
assert.match(html, /icloud-inbox\.js/);
assert.match(ui, /credentials:"same-origin"/);
assert.match(ui, /cache:"no-store"/);
assert.match(ui, /textContent = String\(value\)/);
assert.match(ui, /clinical:unauthenticated/);
assert.match(ui, /state\.controller\?\.abort\(\)/);
assert.match(shell, /\/clinical\/icloud-mail\/peek/);
assert.doesNotMatch(ui, /innerHTML|outerHTML|localStorage|sessionStorage/);
assert.doesNotMatch(ui, /method:"POST"|method:"DELETE"|method:"PUT"/);
console.log("PASS iCloud viewer: protected alias reads, isolated state and text-only DOM");
