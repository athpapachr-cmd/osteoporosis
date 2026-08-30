"use strict";

const assert = require("assert");
const fs = require("fs");

const source = fs.readFileSync("static/baseline-audit/progressive-guidance-ui.js", "utf8");

const summaryStart = source.indexOf("cards.slice(0, 12).forEach(item => {");
const summaryEnd = source.indexOf("root.appendChild(list);", summaryStart);
assert(summaryStart >= 0 && summaryEnd > summaryStart, "G-1 summary renderer block not found");

const summaryRenderer = source.slice(summaryStart, summaryEnd);
assert.match(
  summaryRenderer,
  /why\.textContent\s*=\s*`Γιατί τώρα: \$\{item\.why_now \|\| "Σχετικό με τη σημερινή επίσκεψη\."\}`;/,
  "Σημερινή ροή must visibly prefix each deterministic reason with `Γιατί τώρα:`"
);

assert.match(
  source,
  /strong\.textContent\s*=\s*"Γιατί τώρα: ";/,
  "destination cards must retain their existing explicit `Γιατί τώρα:` label"
);

console.log("progressive guidance WHY-NOW presentation regression: OK");
