/* Synthetic only. The Python event ledger owns state; no browser-side tasks. */
(() => {
  "use strict";
  if (window.CockpitSynthetic !== true ||
      window.CockpitClinicalBase !== "/clinical/synthetic-161") return;
  const container = document.getElementById("syntheticFollowthroughDemo");
  if (!container) return;
  const base = "/clinical/synthetic-161/followthrough";
  const stage = Object.freeze({
    awaiting_result: "Αναμένουμε αποτελέσματα",
    partially_received: "Παραλήφθηκε μέρος των εξετάσεων",
    awaiting_doctor_review: "Αναμένει ιατρικό έλεγχο",
    awaiting_secretary: "Η γραμματέας να ενημερώσει",
    awaiting_callback: "Αναμένει επικοινωνία με ιατρό",
    complete: "Ολοκληρώθηκε",
    source_invalidated: "Η υποχρέωση άλλαξε στην επίσκεψη · χρειάζεται επανέλεγχο"
  });
  const facets = Object.freeze({
    report: "γνωμάτευση", images: "εικόνες MRI", labs: "αναλύσεις"
  });
  let busy = false, generation = 0;

  const node = (tag, text, css = "") => {
    const n = document.createElement(tag);
    if (css) n.className = css;
    n.textContent = text;
    return n;
  };
  async function read(path = "", payload = null, interactive = false) {
    const response = await fetch(base + path, {
      method: payload === null ? "GET" : "POST",
      credentials: "same-origin", cache: "no-store",
      headers: {"Accept": "application/json",
                ...(payload === null ? {} : {"Content-Type": "application/json"})},
      ...(payload === null ? {} : {body: JSON.stringify(payload)})
    });
    if (!response.ok) {
      // Optional synthetic demo must never open/toggle Clinical Key on startup.
      // The existing shared-nav alone owns login; only a user-triggered action
      // may request it after a denied POST.
      if (interactive && (response.status === 401 || response.status === 403))
        window.dispatchEvent(new Event("clinical:request-login"));
      throw Error(response.status === 409 ?
        "Η εκκρεμότητα άλλαξε. Φόρτωσε την τρέχουσα κατάσταση." :
        "Η συνθετική δοκιμή δεν είναι διαθέσιμη.");
    }
    return response.json();
  }
  function next(caseData) {
    if(caseData.source_valid===false) return null;
    for (const part of caseData.required) {
      const source = caseData.received[part];
      if (!source) return {
        label: "Δοκιμή παραλαβής · " + facets[part],
        endpoint: "/simulate-arrival", part, choice: null, isFixture: true
      };
      if (caseData.reviewed[part] !== source.source_version) return {
        label: part === "report" && caseData.mode === "signed_pending" ?
          "Είδα τη γνωμάτευση — ενημέρωση ασθενούς" :
          "Είδα " + (facets[part] || part) + " — χωρίς επικοινωνία",
        endpoint: "/doctor-review", part,
        choice: part === "report" && caseData.mode === "signed_pending" ?
          "notify" : "no_contact"
      };
    }
    if (caseData.contact_status === "to_notify" ||
        caseData.contact_status === "attempted") return {
      label: "Γραμματεία · ο ασθενής ενημερώθηκε (δοκιμή)",
      endpoint: "/secretary-result", part: null, choice: "notified"
    };
    if (caseData.contact_status === "notified" &&
        !caseData.callback_complete) return {
      label: "Έγινε η επικοινωνία με τον ιατρό",
      endpoint: "/doctor-callback", part: null, choice: null
    };
    return null;
  }
  function card(caseData) {
    const box = node("article", "", "synthetic-cycle-card");
    box.append(node("small", caseData.mode === "signed_pending" ?
      "ΑΠΟ ΠΡΑΓΜΑΤΙΚΗ ΣΥΝΘΕΤΙΚΗ ΕΠΙΣΚΕΨΗ #161" :
      "ΕΠΙΣΚΕΨΗ ΧΩΡΙΣ ΗΛΕΚΤΡΟΝΙΚΟ ΦΑΚΕΛΟ"));
    box.append(node("h4", caseData.mode === "signed_pending" ?
      "MRI γόνατος" : "Εργαστηριακές εξετάσεις"));
    const stats = node("div", "", "synthetic-cycle-parts");
    for (const part of caseData.required) {
      const received = caseData.received[part];
      const reviewed = received && caseData.reviewed[part] === received.source_version;
      stats.append(node("span", (reviewed ? "✓ Εξετάστηκαν: " :
        received ? "• Παραλήφθηκαν: " : "◯ Αναμένονται: ") + facets[part]));
    }
    box.append(stats);
    box.append(node("p", stage[caseData.stage] || "Χρειάζεται έλεγχος",
      "synthetic-cycle-status" + (caseData.complete ? " done" : "")));
    const action = next(caseData);
    if (action) {
      const button = node("button", action.label,
        "synthetic-cycle-action" + (action.isFixture ? " fixture" : ""));
      button.type = "button";
      button.disabled = busy;
      button.addEventListener("click", async () => {
        if (busy) return;
        busy = true;
        button.disabled = true;
        try {
          await read(action.endpoint, {
            case_id: caseData.case_id, expected_version: caseData.version,
            idempotency_key: crypto.randomUUID(),
            part: action.part, choice: action.choice
          }, true);
          window.dispatchEvent(new Event("synthetic:followthrough-changed"));
          await reload();
        } catch (error) {
          const feedback = node("p", error.message, "synthetic-cycle-error");
          box.append(feedback);
          busy = false;
          button.disabled = false;
        }
      });
      box.append(button);
    }
    return box;
  }
  function draw(data) {
    container.replaceChildren();
    if (!data || data.mode !== "synthetic_only" ||
        data.live_provider_connected !== false ||
        data.real_messages_used !== false ||
        !Array.isArray(data.cases) || !data.cases.length) {
      container.hidden = true; return;
    }
    container.hidden = false;
    const header = node("div", "", "synthetic-cycle-header");
    header.append(node("h3", "Ένας κύκλος · από την επίσκεψη μέχρι την ολοκλήρωση"));
    header.append(node("p", "Μόνο συνθετική προσομοίωση. Καμία πραγματική κλήση, email ή αλλαγή φακέλου."));
    container.append(header);
    const grid = node("div", "", "synthetic-cycle-grid");
    for (const c of data.cases) grid.append(card(c));
    container.append(grid);
  }
  async function reload(seed = false) {
    const ticket = ++generation;
    try {
      const data = await read(seed ? "/prepare" : "", seed ? {} : null);
      if (ticket === generation) { busy = false; draw(data); }
    } catch (_) {
      if (ticket === generation) { busy = false; container.hidden = true; }
    }
  }
  window.addEventListener("synthetic:prepared", () => { void reload(true); });
  window.addEventListener("clinical:unauthenticated", () => {
    generation++; busy = false; container.replaceChildren(); container.hidden = true;
  });
  window.addEventListener("clinical:authenticated", () => { void reload(); });
  void reload();
})();
