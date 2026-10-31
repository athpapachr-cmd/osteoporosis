/* Live iCloud Mail viewer: read-only, alias-scoped, never a patient chart owner. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const view = $("doctorIcloudInboxView");
  if (!view) return;
  // Only enumerated, PHI-free diagnostic codes can appear in the clinician UI.
  const stageText = Object.freeze({
    CONFIG_DISABLED: "Η ανάγνωση iCloud είναι απενεργοποιημένη.",
    CONFIG_MISSING: "Λείπει υποχρεωτική ρύθμιση iCloud στον server.",
    TLS_CONNECTION_FAILED: "Απέτυχε η ασφαλής σύνδεση TLS με τον IMAP server.",
    IMAP_AUTH_FAILED: "Ο IMAP server απέρριψε τη σύνδεση με τα Apple credentials.",
    INBOX_SELECT_FAILED: "Ο IMAP server δεν επέτρεψε άνοιγμα του INBOX.",
    UIDVALIDITY_OR_UIDNEXT_FAILED: "Δεν επιστράφηκαν έγκυρα UID στοιχεία θυρίδας.",
    IMAP_SEARCH_FAILED: "Απέτυχε η περιορισμένη IMAP αναζήτηση.",
    IMAP_HEADER_FETCH_FAILED: "Απέτυχε η ανάγνωση κεφαλίδων IMAP.",
    IMAP_MESSAGE_FETCH_FAILED: "Απέτυχε η ανάγνωση επιλεγμένου μηνύματος.",
    IMAP_UNEXPECTED_FAILURE: "Απρόβλεπτη τεχνική αποτυχία της πηγής IMAP."
  });
  const stageLabel = code => typeof code === "string" &&
    Object.prototype.hasOwnProperty.call(stageText, code)
      ? stageText[code] + " [" + code + "]" : null;
  window.CockpitIcloudStageMessage = stageLabel;
  const state = {active:false, epoch:0, selection:0, id:null, items:[], controller:null,
    pages:[null], pageIndex:0, nextCursor:null, skipped:0, coverageComplete:false};
  const el = (tag, cls, value) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (value !== undefined) node.textContent = String(value);
    return node;
  };
  const current = ticket => state.active && !view.hidden && state.epoch === ticket;
  function clear() {
    state.epoch++;
    state.selection++;
    state.controller?.abort();
    state.controller = null;
    state.id = null;
    state.items = [];
    state.pages = [null];state.pageIndex = 0;state.nextCursor = null;
    state.skipped = 0;state.coverageComplete = false;
    $("doctorIcloudInboxRows").replaceChildren();
    $("doctorIcloudInboxDetails").replaceChildren();
    $("doctorIcloudInboxState").textContent = "";
    $("doctorIcloudInboxShown").textContent = "";
  }
  function close() {
    clear();
    state.active = false;
    view.hidden = true;
  }
  const validItem = row => row && /^[1-9][0-9]{0,18}-[1-9][0-9]{0,18}$/.test(row.id) &&
    typeof row.subject === "string" && typeof row.from === "string" &&
    typeof row.date === "string" && ["clinical","review"].includes(row.category);
  async function api(path, signal) {
    const response = await fetch("/clinical/icloud-mail" + path, {
      credentials:"same-origin", cache:"no-store", signal,
      headers:{"Accept":"application/json"}
    });
    if (!response.ok) {
      const error = new Error("Mail source unavailable");
      error.status = response.status;
      if (response.status === 503) {
        try {
          const failure = await response.json();
          error.code = failure?.detail?.code;
        } catch (_) { /* Unknown response: retain generic failure. */ }
      }
      throw error;
    }
    return response.json();
  }
  function sourceError(error, target) {
    const diagnosed = error.status === 503 && stageLabel(error.code);
    target.textContent = diagnosed
      ? "iCloud Mail: " + diagnosed + " Δεν γνωρίζουμε αν υπάρχουν νέα μηνύματα."
      : (error.status === 401 || error.status === 403)
      ? "Απαιτείται κλινική σύνδεση. Δεν εμφανίζονται μηνύματα."
      : error.status === 413
        ? "Το email υπερβαίνει το όριο ασφαλούς προβολής. Δες τα πλήρη συνημμένα στο αρχικό μήνυμα στο iCloud Mail."
        : error.status === 409
        ? "Η θυρίδα iCloud άλλαξε. Ανανέωσε από την αρχή για έγκυρη συνέχεια."
        : "Δεν έχει συνδεθεί ακόμη το iCloud Mail ή η πηγή δεν είναι διαθέσιμη. Δεν γνωρίζουμε αν υπάρχουν νέα μηνύματα.";
    if (error.status === 401 || error.status === 403) {
      window.dispatchEvent(new Event("clinical:unauthenticated"));
      window.dispatchEvent(new Event("clinical:request-login"));
    }
  }
  function rowTitle(item) {
    return item.category === "review" ? "Για έλεγχο · " + item.subject : item.subject;
  }
  function rows() {
    const root = $("doctorIcloudInboxRows");
    root.replaceChildren();
    for (const item of state.items) {
      const button = el("button", "doctor-action-row clinical-inbox-row");
      button.type = "button";
      button.setAttribute("aria-pressed", String(item.id === state.id));
      const body = el("span");
      body.append(el("strong", "", rowTitle(item)));
      body.append(el("small", "", item.from + (item.date ? " · " + new Date(item.date).toLocaleDateString("el-CY") : "")));
      button.append(body, el("span", "", "›"));
      button.addEventListener("click", () => select(item.id));
      root.append(button);
    }
    if (!state.items.length) root.append(el("p", "clinical-inbox-note",
      state.nextCursor
        ? "Δεν βρέθηκαν μηνύματα σε αυτή την ενότητα. Υπάρχουν παλαιότερα προς έλεγχο — η λίστα ΔΕΝ είναι επιβεβαιωμένα κενή."
        : "Δεν βρέθηκαν μηνύματα στην ελεγμένη ενότητα. Η προβολή καλύπτει μόνο To/Cc του alias στον κύριο φάκελο."));
    if (!state.coverageComplete) root.append(el("p", "clinical-inbox-warning",
      "Μερική κάλυψη iCloud: " +
      (state.skipped ? state.skipped + " προβληματικά μηνύματα δεν ήταν δυνατό να ελεγχθούν. " : "") +
      (state.nextCursor ? "Υπάρχουν παλαιότερα μηνύματα για έλεγχο." :
        "Η προβολή δεν είναι εγγυημένα πλήρης.")));
    const controls = el("div", "clinical-inbox-actions");
    if (state.pageIndex > 0) {
      const newer = el("button", "doctor-command-button", "← Νεότερα");
      newer.type = "button";
      newer.addEventListener("click", () => {state.pageIndex--;load();});
      controls.append(newer);
    }
    if (state.nextCursor) {
      const older = el("button", "doctor-command-button", "Παλαιότερα →");
      older.type = "button";
      older.addEventListener("click", () => {
        state.pages = state.pages.slice(0, state.pageIndex + 1);
        state.pages.push(state.nextCursor);
        state.pageIndex++;
        load();
      });
      controls.append(older);
    }
    if (controls.children.length) root.append(controls);
    $("doctorIcloudInboxShown").textContent =
      "Σελίδα " + (state.pageIndex + 1) + " · " + state.items.length +
      " μηνύματα προς το alias · πιθανώς κλινικά / για έλεγχο";
  }
  async function load(focusId) {
    if (!state.active) return;
    const cursor = state.pages[state.pageIndex];
    const ticket = ++state.epoch;
    ++state.selection;
    state.controller?.abort();
    state.id = null;state.items = [];
    $("doctorIcloudInboxRows").replaceChildren();
    $("doctorIcloudInboxDetails").replaceChildren();
    const controller = new AbortController();
    state.controller = controller;
    $("doctorIcloudInboxState").textContent = "Ανάγνωση προστατευμένων εισερχομένων iCloud…";
    try {
      const data = await api(cursor ? "?cursor=" + encodeURIComponent(cursor) : "", controller.signal);
      if (!current(ticket)) return;
      if (data?.source !== "icloud_mail" ||
          data.scope !== "recipient_alias_inbox_only" ||
          !Array.isArray(data.items) || data.items.length > 25 ||
          data.items.some(item => !validItem(item)) ||
          !(data.next_cursor === null ||
            (typeof data.next_cursor === "string" &&
             /^[1-9][0-9]{0,18}-[1-9][0-9]{0,18}$/.test(data.next_cursor))) ||
          !Number.isInteger(data.skipped_count) || data.skipped_count < 0 ||
          typeof data.coverage_complete !== "boolean")
        throw Error("Invalid iCloud source");
      state.items = data.items;
      state.nextCursor = data.next_cursor;
      state.skipped = data.skipped_count;
      state.coverageComplete = data.coverage_complete;
      $("doctorIcloudInboxState").textContent =
        "Μόνο ανάγνωση · δεν αλλάζει η κατάσταση των email και δεν γίνεται σύνδεση με ασθενή.";
      rows();
      if (focusId && data.items.some(i => i.id === focusId)) select(focusId);
    } catch (error) {
      if (!current(ticket)) return;
      clear();
      sourceError(error, $("doctorIcloudInboxState"));
    }
  }
  async function select(id) {
    if (!state.active || !state.items.some(i => i.id === id)) return;
    const ticket = state.epoch;
    const selected = ++state.selection;
    state.id = id;
    rows();
    const details = $("doctorIcloudInboxDetails");
    details.replaceChildren(el("p", "clinical-inbox-note", "Φόρτωση μηνύματος…"));
    try {
      const data = await api("/items/" + encodeURIComponent(id), state.controller?.signal);
      if (!current(ticket) || selected !== state.selection || state.id !== id) return;
      if (data?.source !== "icloud_mail" || !validItem(data.item) ||
          data.item.id !== id || typeof data.item.body !== "string" ||
          !Array.isArray(data.item.attachments) || data.item.attachments.length > 60 ||
          !Number.isInteger(data.item.attachments_unavailable) ||
          data.item.attachments_unavailable < 0 || data.item.attachments_unavailable > 10000)
        throw Error("Invalid message");
      details.replaceChildren();
      details.append(el("h3", "", rowTitle(data.item)),
        el("p", "clinical-inbox-note", "Από: " + data.item.from),
        el("p", "clinical-inbox-warning",
          "Μήνυμα μόνο για προβολή. Δεν αποτελεί επαληθευμένη ταυτότητα ή αποτέλεσμα ενταγμένο σε ιατρικό φάκελο."));
      const body = el("div", "doctor-icloud-body", data.item.body ||
        "Δεν υπάρχει ασφαλές απλό κείμενο για εμφάνιση. Ελέγξτε τυχόν συνημμένα.");
      details.append(body);
      if (data.item.attachments_unavailable > 0) {
        details.append(el("p", "clinical-inbox-warning",
          "Υπάρχουν " + data.item.attachments_unavailable +
          " συνημμένα που δεν μπορούν να προβληθούν εδώ λόγω τύπου ή μεγέθους. Έλεγξέ τα στο αρχικό iCloud Mail."));
      }
      if (data.item.attachments.length) {
        details.append(el("h4", "", "Συνημμένα · δεν έχουν ελεγχθεί για κακόβουλα αρχεία"));
        for (const attachment of data.item.attachments) {
          if (!attachment || !Number.isInteger(attachment.index) ||
              attachment.index < 0 || attachment.index > 100 ||
              typeof attachment.name !== "string" ||
              typeof attachment.content_type !== "string") continue;
          const link = el("a", "doctor-icloud-attachment",
            attachment.name + " · Λήψη (χωρίς έλεγχο κακόβουλου περιεχομένου)");
          link.href = "/clinical/icloud-mail/items/" + encodeURIComponent(id) +
            "/attachments/" + attachment.index;
          details.append(link);
        }
      }
    } catch (error) {
      if (!current(ticket) || selected !== state.selection) return;
      details.replaceChildren();
      sourceError(error, details.appendChild(el("p", "clinical-inbox-warning")));
    }
  }
  function open(id=null) {
    state.active = true;
    view.hidden = false;
    load(id);
  }
  $("doctorIcloudInboxReload").addEventListener("click", () => {
    state.pages = [null];state.pageIndex = 0;load();
  });
  window.addEventListener("clinical:unauthenticated", close);
  window.addEventListener("clinical:authenticated", close);
  window.CockpitIcloudInbox = {open,close};
})();
