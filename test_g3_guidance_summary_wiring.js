"use strict";

const assert = require("assert");
const fs = require("fs");

const app = fs.readFileSync("static/baseline-audit/app.js", "utf8");
const ui = fs.readFileSync("static/baseline-audit/progressive-guidance-ui.js", "utf8");
const css = fs.readFileSync("static/baseline-audit/progressive-guidance.css", "utf8");
const core = fs.readFileSync("static/baseline-audit/osteoporosis-longitudinal-summary-core.js", "utf8");

// Load order: pure G3 summary core must exist before the single guidance UI owner.
const summaryCoreLoad = app.indexOf('loadScript("./osteoporosis-longitudinal-summary-core.js")');
const guidanceUiLoad = app.indexOf('loadScript("./progressive-guidance-ui.js")');
assert(summaryCoreLoad >= 0 && guidanceUiLoad > summaryCoreLoad, "G3 summary core must load before progressive-guidance-ui.js");

// Ownership: the pure core must not fetch, persist, or render.
assert(!/\bfetch\s*\(/.test(core), "G3 pure summary core must not own network fetch");
assert(!/localStorage|sessionStorage/.test(core), "G3 pure summary core must not own browser persistence");
assert(!/document\.|querySelector|createElement/.test(core), "G3 pure summary core must not own DOM rendering");

// Existing progressive guidance UI remains the network/render owner and fetches both protected axes.
assert(ui.includes("fetchHistoricalEncounters"), "existing protected encounter fetch owner missing");
assert(ui.includes("fetchHistoricalLabs"), "G3 must fetch labs through existing guidance history owner");
assert(ui.includes("Promise.allSettled"), "encounter and lab availability must be tracked independently");
assert(ui.includes("BaselineOsteoporosisLongitudinalSummary"), "G3 summary core not wired into guidance UI");
assert(ui.includes("patientLongitudinalSummary"), "always-visible patient summary root missing");
assert(ui.includes('title.textContent = "Σύνοψη ασθενούς"'), "patient summary title missing");
assert(ui.includes("current_non_historical"), "current visit must be visually distinguished from completed history");

// Salience: explicit textual marker plus visual class; initial plan baseline state is tracked.
assert(ui.includes("newlySurfacedDomains"), "newly surfaced state missing");
assert(ui.includes('badge.textContent = "Νέο"'), "new guidance must have a textual Νέο marker");
assert(ui.includes("is-newly-surfaced"), "new guidance visual class missing");
assert(ui.includes("salienceEligible"), "new guidance eligibility guard missing");
assert(ui.includes("planBaselineKey"), "per patient/case initial-plan baseline missing");
assert(css.includes(".progressive-guidance-item.is-newly-surfaced"), "top flow new-item visual emphasis missing");
assert(css.includes("article.card.guidance-surfaced.is-newly-surfaced"), "destination-card new-item visual emphasis missing");
assert(css.includes(".progressive-guidance-new-badge"), "text badge styling missing");

// Summary must explicitly represent uncertainty/absence rather than turning missing into a normal statement.
assert(ui.includes("Δεν έχει τεκμηριωθεί"), "explicit not-documented state missing");
assert(ui.includes("Μη διαθέσιμο — αποτυχία φόρτωσης protected laboratory history"), "lab unavailable state missing");
assert(ui.includes("Υπάρχει longitudinal ασυμφωνία"), "treatment conflict presentation missing");

// No new treatment selection/write path in G3 UI.
assert(!ui.includes("selected_agent ="), "G3 UI must not write treatment selection");
assert(!ui.includes("/clinical/encounter/") || !ui.includes('method: "PUT"'), "G3 UI must not introduce encounter write API ownership");

console.log("G3 guidance summary wiring/ownership regressions: PASS");
