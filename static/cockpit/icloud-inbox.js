/* Live iCloud Mail viewer: read-only, alias-scoped, never a patient chart owner. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const view = $("doctorIcloudInboxView");
  if (!view) return;
  const state = {active:false, epoch:0, selection:0, id:null, items:[], controller:null};
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
      throw error;
    }
    return response.json();
  }
  function sourceError(error, target) {
    target.textContent = (error.status === 401 || error.status === 403)
      ? "Απαιτείται κλινική σύνδεση. Δεν εμφανίζονται μηνύματα."
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
      "Δεν βρέθηκαν μηνύματα στα ελεγμένα εισερχόμενα του alias. Η κάλυψη δεν περιλαμβάνει κρυφούς παραλήπτες ή άλλους φακέλους."));
    $("doctorIcloudInboxShown").textContent =
      state.items.length + " μηνύματα προς το alias ιατρείου · πιθανώς κλινικά / για έλεγχο";
  }
  async function load(focusId) {
    if (!state.active) return;
    clear();
    const ticket = state.epoch;
    const controller = new AbortController();
    state.controller = controller;
    $("doctorIcloudInboxState").textContent = "Ανάγνωση προστατευμένων εισερχομένων iCloud…";
    try {
      const data = await api("", controller.signal);
      if (!current(ticket)) return;
      if (data?.source !== "icloud_mail" ||
          data.scope !== "recipient_alias_inbox_only" ||
          !Array.isArray(data.items) || data.items.length > 25 ||
          data.items.some(item => !validItem(item)))
        throw Error("Invalid iCloud source");
      state.items = data.items;
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
          !Array.isArray(data.item.attachments) || data.item.attachments.length > 60)
        throw Error("Invalid message");
      details.replaceChildren();
      details.append(el("h3", "", rowTitle(data.item)),
        el("p", "clinical-inbox-note", "Από: " + data.item.from),
        el("p", "clinical-inbox-warning",
          "Μήνυμα μόνο για προβολή. Δεν αποτελεί επαληθευμένη ταυτότητα ή αποτέλεσμα ενταγμένο σε ιατρικό φάκελο."));
      const body = el("div", "doctor-icloud-body", data.item.body ||
        "Δεν υπάρχει ασφαλές απλό κείμενο για εμφάνιση. Ελέγξτε τυχόν συνημμένα.");
      details.append(body);
      if (data.item.attachments.length) {
        details.append(el("h4", "", "Συνημμένα · ασφαλής λήψη"));
        for (const attachment of data.item.attachments) {
          if (!attachment || !Number.isInteger(attachment.index) ||
              attachment.index < 0 || attachment.index > 100 ||
              typeof attachment.name !== "string" ||
              typeof attachment.content_type !== "string") continue;
          const link = el("a", "doctor-icloud-attachment",
            attachment.name + " · Λήψη");
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
  $("doctorIcloudInboxReload").addEventListener("click", () => load());
  window.addEventListener("clinical:unauthenticated", close);
  window.addEventListener("clinical:authenticated", close);
  window.CockpitIcloudInbox = {open,close};
})();
