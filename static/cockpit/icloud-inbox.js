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
    IMAP_AUTH_FAILED: "Απέτυχε το στάδιο IMAP LOGIN. Δεν έχει επιβεβαιωθεί αν η αιτία είναι τα διαπιστευτήρια ή η επικοινωνία.",
    INBOX_SELECT_FAILED: "Απέτυχε το στάδιο ανοίγματος του INBOX. Δεν έχει επιβεβαιωθεί αν ο server αρνήθηκε το αίτημα ή διακόπηκε η επικοινωνία.",
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
  // Local role is a UX hint for blocking iCloud reads, not proof of a live
  // secretary session. Only the protected staff endpoint confirms that claim.
  const secretaryIcloudSession=()=>window.CockpitStaffRole==="secretary";
  async function verifiedStaffRole(){
    try{
      const response=await fetch("/clinical/staff-session",{
        credentials:"same-origin",cache:"no-store",
        headers:{"Accept":"application/json"}
      });
      if(response.status===401||response.status===403)return "invalid";
      if(!response.ok)return "unknown";
      const actor=await response.json();
      return actor?.staff_role==="secretary"||actor?.staff_role==="clinician"
        ?actor.staff_role:"unknown";
    }catch(_){return "unknown";}
  }
  function secretaryIcloudNotice(){
    $("doctorIcloudInboxState").textContent=
      "Τα iCloud email απαιτούν σύνδεση ιατρού. Η σύνδεση γραμματείας παραμένει ενεργή.";
  }
  function staffStatusUnavailable(){
    $("doctorIcloudInboxState").textContent=
      "Δεν επιβεβαιώθηκε η συνεδρία προσωπικού [STAFF_SESSION_UNVERIFIED]. "+
      "Δεν εμφανίζονται email ασθενών.";
  }
  const state = {active:false, epoch:0, selection:0, id:null, items:[], controller:null,
    pages:[null], pageIndex:0, nextCursor:null, skipped:0, coverageComplete:false,
    latest:null, latestAt:0};
  const el = (tag, cls, value) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (value !== undefined) node.textContent = String(value);
    return node;
  };
  const current = ticket => state.active && !view.hidden && state.epoch === ticket;
  function clear(purgeCache=false) {
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
    if(purgeCache){state.latest=null;state.latestAt=0;}
  }
  function close(purgeCache=false) {
    clear(purgeCache === true);
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
    try{return await response.json();}
    catch(_){const err=new Error("Invalid protected JSON");err.clientCode="LIST_JSON";throw err;}
  }
  function sourceError(error, target) {
    const diagnosed = error.status === 503 && stageLabel(error.code);
    const clientCode = ["LIST_JSON","LIST_SOURCE","LIST_ITEMS_MISSING",
      "LIST_ITEMS_LIMIT","LIST_ITEM_FIELDS","LIST_CURSOR","LIST_COVERAGE",
      "LIST_RENDER","LIST_AUTH_CHECK"].includes(error.clientCode)?error.clientCode:null;
    target.textContent = diagnosed
      ? "iCloud Mail: " + diagnosed + " Δεν γνωρίζουμε αν υπάρχουν νέα μηνύματα."
      : (error.status === 401 || error.status === 403)
      ? "Απαιτείται κλινική σύνδεση. Δεν εμφανίζονται μηνύματα."
      : clientCode
      ? "Το iCloud απάντησε, αλλά η πλήρης προβολή δεν μπόρεσε να επεξεργαστεί την απάντηση. ["+clientCode+"]"
      : error.status === 413
        ? "Το email υπερβαίνει το όριο ασφαλούς προβολής. Δες τα πλήρη συνημμένα στο αρχικό μήνυμα στο iCloud Mail."
        : error.status === 409
        ? "Η θυρίδα iCloud άλλαξε. Ανανέωσε από την αρχή για έγκυρη συνέχεια."
        : error.status
        ? "Η πηγή iCloud επέστρεψε σφάλμα HTTP ["+((error.status===503)?"LIST_SOURCE_503":"LIST_HTTP_ERROR")+"]."
        : "Δεν ολοκληρώθηκε η ανάγνωση στον browser [LIST_NETWORK].";
    if (error.status === 401 || error.status === 403) {
      // A denied clinician-only read may coexist with a valid secretary cookie,
      // including when shared-nav has not yet resolved the staff role.
      const ticket=state.epoch;
      void (async()=>{
        const role=await verifiedStaffRole();
        if(!current(ticket))return;
        if(role==="secretary"){
          window.CockpitStaffRole="secretary";
          secretaryIcloudNotice();
        }else if(role==="unknown"){
          staffStatusUnavailable();
        }else if(role==="clinician"){
          $("doctorIcloudInboxState").textContent=
            "Δεν επαληθεύτηκε πρόσβαση στα iCloud email [LIST_AUTH_DENIED].";
        }else{
          window.dispatchEvent(new Event("clinical:unauthenticated"));
          window.dispatchEvent(new Event("clinical:request-login"));
        }
      })();
    }
  }
  // If the browser remembers secretary, fail closed before checking the
  // actual cookie. A stale or unavailable staff session must never be
  // described as active; no provider request is sent while this check runs.
  async function guardSecretarySession(){
    if(!secretaryIcloudSession())return true;
    clear(true); // Purge any prior clinician's subject/body first.
    const ticket=state.epoch;
    $("doctorIcloudInboxState").textContent="Έλεγχος συνεδρίας προσωπικού…";
    const role=await verifiedStaffRole();
    if(!current(ticket))return false;
    if(role==="secretary"){secretaryIcloudNotice();return false;}
    if(role==="unknown"){staffStatusUnavailable();return false;}
    if(role==="invalid"){
      window.dispatchEvent(new Event("clinical:unauthenticated"));
      window.dispatchEvent(new Event("clinical:request-login"));
      return false;
    }
    // An explicit verified clinician login may replace a stale UI hint.
    window.CockpitStaffRole="clinician";
    return true;
  }
  function listContractCode(data) {
    if(data?.source!=="icloud_mail"||
       data.scope!=="recipient_alias_inbox_only"||data.filter!=="all")return "LIST_SOURCE";
    if(!Array.isArray(data.items))return "LIST_ITEMS_MISSING";
    if(data.items.length>25)return "LIST_ITEMS_LIMIT";
    if(data.items.some(item=>!validItem(item)))return "LIST_ITEM_FIELDS";
    if(!(data.next_cursor===null||
         (typeof data.next_cursor==="string"&&
          /^[1-9][0-9]{0,18}-[1-9][0-9]{0,18}$/.test(data.next_cursor))))
      return "LIST_CURSOR";
    if(!Number.isInteger(data.skipped_count)||data.skipped_count<0||
       typeof data.coverage_complete!=="boolean")return "LIST_COVERAGE";
    return null;
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
    if (!state.active || !(await guardSecretarySession())) return;
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
    let sourceReturned=false;
    try {
      const data = await api(cursor ? "?cursor=" + encodeURIComponent(cursor) : "", controller.signal);
      sourceReturned=true;
      if (!current(ticket)) return;
      const invalid=listContractCode(data);
      if(invalid){const err=new Error("Invalid list contract");err.clientCode=invalid;throw err;}
      if(cursor===null){state.latest=data;state.latestAt=Date.now();}
      state.items = data.items;
      state.nextCursor = data.next_cursor;
      state.skipped = data.skipped_count;
      state.coverageComplete = data.coverage_complete;
      $("doctorIcloudInboxState").textContent =
        "Μόνο ανάγνωση · δεν αλλάζει η κατάσταση των email και δεν γίνεται σύνδεση με ασθενή.";
      rows();
      if (focusId && /^[1-9][0-9]{0,18}-[1-9][0-9]{0,18}$/.test(focusId)) select(focusId);
    } catch (error) {
      if (!current(ticket)) return;
      if(sourceReturned&&!error.status&&!error.clientCode)error.clientCode="LIST_RENDER";
      // Authentication loss is never a transient source error: do not revive
      // protected email subjects/senders from a cached list after 401/403.
      if(error?.status===401||error?.status===403){
        clear(true);
        sourceError(error, $("doctorIcloudInboxState"));
        return;
      }
      clear();
      if(cursor===null && state.latest){
        const data=state.latest;
        state.items=data.items;state.nextCursor=data.next_cursor;
        state.skipped=data.skipped_count;state.coverageComplete=data.coverage_complete;
        rows();
        $("doctorIcloudInboxState").textContent=
          "Η ανανέωση απέτυχε. Εμφανίζεται παλαιότερη επιβεβαιωμένη λίστα (όχι τρέχουσα).";
      }else sourceError(error, $("doctorIcloudInboxState"));
    }
  }
  async function select(id) {
    if (!state.active || !/^[1-9][0-9]{0,18}-[1-9][0-9]{0,18}$/.test(id) ||
        !(await guardSecretarySession())) return;
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
      if(error?.status===401||error?.status===403){
        clear(true); // Do not retain the previous page during staff-role lookup.
        sourceError(error,$("doctorIcloudInboxState"));
        return;
      }
      details.replaceChildren();
      sourceError(error, details.appendChild(el("p", "clinical-inbox-warning")));
    }
  }
  function open(id=null) {
    state.active = true;
    view.hidden = false;
    if(secretaryIcloudSession()){
      void (async()=>{
        if(await guardSecretarySession())open(id);
      })();
      return;
    }
    // Open an unread peek selection directly. The full archive may place
    // this UID many pages later; never depend on fetching page one first.
    if(typeof id==="string" && /^[1-9][0-9]{0,18}-[1-9][0-9]{0,18}$/.test(id)){
      state.epoch++;state.selection++;
      state.controller?.abort();state.controller=new AbortController();
      state.id=null;state.items=[];state.pages=[null];state.pageIndex=0;
      $("doctorIcloudInboxRows").replaceChildren();
      $("doctorIcloudInboxDetails").replaceChildren();
      $("doctorIcloudInboxShown").textContent="";
      $("doctorIcloudInboxState").textContent=
        "Απευθείας προβολή επιλεγμένου email · μόνο ανάγνωση.";
      select(id);
      return;
    }
    if(state.latest && Date.now()-state.latestAt < 5*60*1000){
      // A cached header list must not bypass session expiry. Validate the
      // same-origin Clinical Key cookie before resurfacing cached PHI.
      const ticket=++state.epoch;
      state.controller?.abort();
      const controller=new AbortController();
      state.controller=controller;
      $("doctorIcloudInboxRows").replaceChildren();
      $("doctorIcloudInboxDetails").replaceChildren();
      $("doctorIcloudInboxState").textContent="Έλεγχος κλινικής πρόσβασης…";
      fetch("/clinical/status",{
        credentials:"same-origin",cache:"no-store",signal:controller.signal
      }).then(response=>{
        if(!current(ticket))return;
        if(!response.ok){
          clear(true);
          sourceError({status:response.status},$("doctorIcloudInboxState"));
          return;
        }
        const data=state.latest;
        if(!data)return;
        state.pages=[null];state.pageIndex=0;
        state.items=data.items;state.nextCursor=data.next_cursor;
        state.skipped=data.skipped_count;state.coverageComplete=data.coverage_complete;
        $("doctorIcloudInboxState").textContent=
          "Μόνο ανάγνωση · τελευταία ενημέρωση "+
          new Date(state.latestAt).toLocaleTimeString("el-CY",
            {hour:"2-digit",minute:"2-digit"});
        rows();
      }).catch(()=>{
        if(!current(ticket)||controller.signal.aborted)return;
        clear();
        sourceError({clientCode:"LIST_AUTH_CHECK"},$("doctorIcloudInboxState"));
      });
    }else load(id);
  }
  $("doctorIcloudInboxReload").addEventListener("click", () => {
    state.pages = [null];state.pageIndex = 0;load();
  });
  window.addEventListener("clinical:unauthenticated", ()=>close(true));
  window.addEventListener("clinical:authenticated", ()=>close(true));
  // Automatic 5-minute refresh belongs to the unread preview owner.
  // Never replace an actively opened full-message view on a preview tick.
  window.CockpitIcloudInbox = {open,close};
})();
