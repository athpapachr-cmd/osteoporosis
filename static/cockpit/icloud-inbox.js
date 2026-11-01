/* Live iCloud Mail viewer: read-only, alias-scoped, never a patient chart owner. */
(() => {
  "use strict";
  const triage=window.CockpitIcloudTriage;
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
    IMAP_FLAGS_FETCH_FAILED: "Απέτυχε η ανάγνωση της κατάστασης αναγνωσμένου/μη αναγνωσμένου στο IMAP.",
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
    view:"clinical",
    pages:[null], pageIndex:0, nextCursor:null, skipped:0, coverageComplete:false,
    latest:null, latestAt:0, projection:null};
  const el = (tag, cls, value) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (value !== undefined) node.textContent = String(value);
    return node;
  };
  const current = ticket => state.active && !view.hidden && state.epoch === ticket;
  function emptySelection() {
    if (!state.active || view.hidden) return;
    $("doctorIcloudInboxDetails").replaceChildren(el("p", "clinical-inbox-empty-detail",
      "Επίλεξε ένα μήνυμα από τη λίστα για να δεις το περιεχόμενό του. Η προβολή δεν αλλάζει την κατάστασή του στο iCloud."));
  }
  function clear(purgeCache=false) {
    state.epoch++;
    state.selection++;
    state.controller?.abort();
    state.controller = null;
    state.id = null;
    state.projection=null;
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
    typeof row.date === "string" &&
    ["clinical","review","administrative","other"].includes(row.category);
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
          // Staff clinician status does not override clinician-only iCloud 401.
          window.dispatchEvent(new Event("clinical:unauthenticated"));
          window.dispatchEvent(new Event("clinical:request-login"));
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
       data.scope!=="recipient_alias_inbox_only"||data.filter!=="all"||
       (data.view!==undefined&&data.view!==state.view))return "LIST_SOURCE";
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
  const categoryName = Object.freeze({
    clinical:"Κλινικό", administrative:"Διοικητικό",
    other:"Άλλο", review:"Χρειάζεται χαρακτηρισμό"
  });
  function rowTitle(item) {
    return categoryName[item.category] + " · " + item.subject;
  }
  function updateViews() {
    document.querySelectorAll("[data-icloud-view]").forEach(button => {
      const chosen=button.dataset.icloudView===state.view;
      button.setAttribute("aria-pressed",String(chosen));
      button.classList.toggle("selected",chosen);
    });
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
      " μηνύματα · " + (state.view==="clinical"?"Κλινικά / αταξινόμητα":
      state.view==="administrative"?"Διοικητικά / λοιπά":"Όλα προς το email του ιατρείου");
  }
  async function load(focusId) {
    if (!state.active)return;
    if(secretaryIcloudSession() && !(await guardSecretarySession()))return;
    if(!state.active||view.hidden)return; // Closed while role verification ran.
    const cursor = state.pages[state.pageIndex];
    const ticket = ++state.epoch;
    ++state.selection;
    state.controller?.abort();
    state.id = null;state.items = [];
    $("doctorIcloudInboxRows").replaceChildren();
    emptySelection();
    const controller = new AbortController();
    state.controller = controller;
    $("doctorIcloudInboxState").textContent = "Ανάγνωση προστατευμένων εισερχομένων iCloud…";
    let sourceReturned=false;
    try {
      const status=await fetch("/clinical/status",{credentials:"same-origin",cache:"no-store",signal:controller.signal});
      if(!status.ok)throw {status:status.status};
      if(!current(ticket))return;
      const params=new URLSearchParams({view:state.view});
      if(cursor)params.set("cursor",cursor);
      if(triage){try{await triage.qualify();}catch(error){if(error.status===401||error.status===403)throw error;}}
      if(!current(ticket))return;
      const data = await api("?" + params.toString(), controller.signal);
      sourceReturned=true;
      if (!current(ticket)) return;
      const invalid=listContractCode(data);
      if(invalid){const err=new Error("Invalid list contract");err.clientCode=invalid;throw err;}
      if(triage && data.triage?.available && !triage.accept(data.triage))throw {status:409};
      if(data.triage?.available===false){state.view="all";updateViews();}
      if(cursor===null && (!triage||data.triage?.available)){state.latest=data;state.latestAt=Date.now();}
      state.projection=data.triage;
      state.items = data.items;
      state.nextCursor = data.next_cursor;
      state.skipped = data.skipped_count;
      state.coverageComplete = data.coverage_complete;
      $("doctorIcloudInboxState").textContent =
        "Οι κατηγορίες είναι προτάσεις ταξινόμησης, όχι ιατρική αξιολόγηση ή σύνδεση με φάκελο.";
      if(data.triage?.available===false)$("doctorIcloudInboxState").textContent=
        "Οι αποθηκευμένες κατηγορίες δεν επιβεβαιώθηκαν. Προβάλλονται όλα τα email χωρίς φιλτράρισμα· οι αλλαγές δεν είναι διαθέσιμες.";
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
      if(cursor===null && state.latest && !triage && !state.latest.triage){
        const data=state.latest;
        state.items=data.items;state.nextCursor=data.next_cursor;
        state.skipped=data.skipped_count;state.coverageComplete=data.coverage_complete;
        rows();
        $("doctorIcloudInboxState").textContent=
          "Η ανανέωση απέτυχε. Εμφανίζεται παλαιότερη επιβεβαιωμένη λίστα (όχι τρέχουσα).";
      }else sourceError(error, $("doctorIcloudInboxState"));
    }
  }
  // The iCloud inbox is a source VIEWER, not a second clinical task engine.
  // The clinician chooses an existing patient and signed Pending explicitly.
  // No email-to-patient link, receipt, medical review or contact is saved here.
  function renderExistingPending(id,ticket,selection,_item,details) {
    const stillCurrent=()=>current(ticket)&&selection===state.selection&&state.id===id;
    const panel=el("section","doctor-icloud-workflow");
    panel.append(
      el("h4","","Άνοιγμα υπάρχουσας κλινικής εκκρεμότητας"),
      el("p","clinical-inbox-note",
        "Επίλεξε εσύ τον ασθενή και την εκκρεμότητά του. Το email δεν συνδέεται " +
        "αυτόματα ούτε θεωρείται παραληφθέν ή αξιολογημένο.")
    );
    const input=el("input");
    input.type="search";input.autocomplete="off";input.maxLength=120;
    input.placeholder="Ονοματεπώνυμο από τον υπάρχοντα φάκελο";
    input.setAttribute("aria-label","Αναζήτηση υπάρχοντος ασθενούς για εκκρεμότητα");
    const results=el("div","doctor-icloud-candidates");
    const note=el("p","clinical-inbox-note","");
    note.setAttribute("role","status");
    let requestNumber=0;
    const makeButton=(label,handler)=>{
      const button=el("button","doctor-command-button",label);
      button.type="button";button.addEventListener("click",handler);return button;
    };
    const searchButton=makeButton("Βρες υπάρχουσα εκκρεμότητα",searchPatients);
    const inputArea=el("div","doctor-icloud-linking");
    inputArea.append(input,searchButton);
    panel.append(inputArea,note,results);details.append(panel);
    const protectedRead=async(path)=>{
      const response=await fetch(path,{
        credentials:"same-origin",cache:"no-store",signal:state.controller?.signal,
        headers:{"Accept":"application/json"}
      });
      if(!response.ok){
        const error=Error("Existing clinical source unavailable");
        error.status=response.status;throw error;
      }
      return response.json();
    };
    const handleError=error=>{
      if(!stillCurrent())return;
      if(error?.status===401||error?.status===403){
        clear(true);sourceError(error,$("doctorIcloudInboxState"));return;
      }
      results.replaceChildren();
      note.textContent=error?.status===503?
        "Η υπάρχουσα κλινική πηγή δεν είναι διαθέσιμη. Δεν έγινε καμία καταγραφή.":
        "Η αναζήτηση δεν ολοκληρώθηκε. Δεν έγινε σύνδεση ή αλλαγή ασθενούς.";
    };
    async function loadPendings(patientId,patientName){
      if(!stillCurrent()||typeof patientId!=="string"||!patientId||patientId.length>120)return;
      const version=++requestNumber;
      results.replaceChildren();
      note.textContent="Έλεγχος υπαρχουσών εκκρεμοτήτων…";
      try{
        // This existing endpoint filters by CURRENT signed-visit membership
        // and exact patient/Pending identity; never show an unsigned legacy item.
        const items=await protectedRead("/clinical/patient/"+encodeURIComponent(patientId)+"/pending?signed_only=true");
        if(!stillCurrent()||version!==requestNumber)return;
        if(!Array.isArray(items)||items.length>500)throw Error("Invalid Pending list");
        const active=items.filter(row=>row&&row.patient_id===patientId&&
          typeof row.pending_id==="string"&&row.pending_id.length>0&&row.pending_id.length<=120&&
          typeof row.description==="string"&&
          ["open","awaiting_patient","awaiting_result"].includes(row.status));
        note.textContent=active.length?
          "Εκκρεμότητες · "+patientName+" · επίλεξε τη σωστή":
          "Δεν υπάρχει ενεργή εκκρεμότητα στον επιλεγμένο φάκελο. Δεν δημιουργήθηκε νέα.";
        for(const row of active.slice(0,20)){
          results.append(makeButton("Άνοιγμα εκκρεμότητας · "+row.description,()=>{
            if(!stillCurrent())return;
            if(typeof window.CockpitOpenPendingFromIcloud!=="function"){
              note.textContent="Η προβολή της υπάρχουσας εκκρεμότητας δεν είναι διαθέσιμη.";return;
            }
            window.CockpitOpenPendingFromIcloud(patientId,row.pending_id);
          }));
        }
        if(active.length>20)results.append(el("p","clinical-inbox-note",
          "Εμφανίζονται οι πρώτες 20 εκκρεμότητες. Άνοιξε τον φάκελο για τις υπόλοιπες."));
      }catch(error){handleError(error);}
    }
    async function searchPatients(){
      const query=input.value.trim();
      if(!stillCurrent())return;
      const version=++requestNumber;
      results.replaceChildren();
      if(query.length<3){
        note.textContent="Γράψε τουλάχιστον τρεις χαρακτήρες ονοματεπωνύμου.";return;
      }
      note.textContent="Αναζήτηση στον υπάρχοντα κλινικό κατάλογο…";
      try{
        const people=await protectedRead(
          "/clinical/patients?query="+encodeURIComponent(query.slice(0,120))+"&limit=20");
        if(!stillCurrent()||version!==requestNumber||query!==input.value.trim())return;
        if(!Array.isArray(people)||people.length>20)throw Error("Invalid patient list");
        note.textContent=people.length?
          "Επίλεξε τον σωστό ασθενή. Η αναζήτηση δεν επιβεβαιώνει την ταυτότητα του email.":
          "Δεν βρέθηκε υπάρχων φάκελος. Δεν δημιουργήθηκε νέος ασθενής.";
        for(const person of people){
          if(!person||typeof person.patient_id!=="string"||
             !person.patient_id||person.patient_id.length>120)continue;
          const demographics=person.demographics||{};
          const name=demographics.full_name||demographics.fullName||
            demographics.name||"Κλινικός φάκελος";
          if(typeof name!=="string")continue;
          const dob=typeof demographics.dob==="string"?demographics.dob:"";
          const entry=el("div","doctor-icloud-candidate");
          entry.append(el("p","clinical-inbox-note",name+(dob?" · "+dob:"")));
          entry.append(makeButton("Δες εκκρεμότητες",()=>loadPendings(person.patient_id,name)));
          results.append(entry);
        }
        if(people.length===20)results.append(el("p","clinical-inbox-note",
          "Εμφανίζονται έως 20 αποτελέσματα. Εξειδίκευσε το όνομα αν χρειάζεται."));
      }catch(error){handleError(error);}
    }
    // Editing the query invalidates all prior patient/Pending results, even
    // when an earlier protected request is still completing.
    input.addEventListener("input",()=>{
      ++requestNumber;
      results.replaceChildren();
      note.textContent="Η αναζήτηση άλλαξε. Επίλεξε εκ νέου τον σωστό ασθενή.";
    });
    input.addEventListener("keydown",event=>{
      if(event.key==="Enter"){event.preventDefault();searchPatients();}
    });
  }
  async function select(id) {
    if (!state.active || !/^[1-9][0-9]{0,18}-[1-9][0-9]{0,18}$/.test(id))return;
    if(secretaryIcloudSession() && !(await guardSecretarySession()))return;
    if(!state.active||view.hidden)return; // Never fetch after closing.
    const ticket = state.epoch;
    const selected = ++state.selection;
    state.id = id;
    rows();
    const details = $("doctorIcloudInboxDetails");
    details.replaceChildren(el("p", "clinical-inbox-note", "Φόρτωση μηνύματος…"));
    try {
      if(triage){try{await triage.qualify();}catch(error){if(error.status===401||error.status===403)throw error;}}
      if(!current(ticket)||selected!==state.selection)return;
      const data = await api("/items/" + encodeURIComponent(id), state.controller?.signal);
      if (!current(ticket) || selected !== state.selection || state.id !== id) return;
      if (data?.source !== "icloud_mail" || !validItem(data.item) ||
          data.item.id !== id || typeof data.item.body !== "string" ||
          !Array.isArray(data.item.attachments) || data.item.attachments.length > 60 ||
          !Number.isInteger(data.item.attachments_unavailable) ||
          data.item.attachments_unavailable < 0 || data.item.attachments_unavailable > 10000)
        throw Error("Invalid message");
      if(triage && data.triage?.available && !triage.accept(data.triage))throw {status:409};
      state.projection=data.triage;
      details.replaceChildren();
      details.append(el("h3", "", rowTitle(data.item)),
        el("p", "clinical-inbox-note", "Από: " + data.item.from),
        el("p", "clinical-inbox-warning",
          "Μήνυμα μόνο για προβολή. Δεν αποτελεί επαληθευμένη ταυτότητα ή αποτέλεσμα ενταγμένο σε ιατρικό φάκελο."));
      if(triage && data.triage?.available){
        const controls=el("div");details.append(controls);
        void triage.editor(controls,data.item,()=>current(ticket)&&selected===state.selection&&state.id===id,data.triage);
      }
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
      renderExistingPending(id,ticket,selected,data.item,details);
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
  function open(id=null, initialView="clinical") {
    state.active = true;
    view.hidden = false;
    if(["clinical","administrative","all"].includes(initialView))
      state.view=initialView;
    updateViews();
    emptySelection();
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
    load(id); // Reopen obtains a newly qualified source/category projection.

  }
  document.querySelectorAll("[data-icloud-view]").forEach(button => {
    button.addEventListener("click",()=>{
      if(!state.active)return;
      const value=button.dataset.icloudView;
      if(!["clinical","administrative","all"].includes(value))return;
      state.view=value;updateViews();
      state.pages=[null];state.pageIndex=0;state.nextCursor=null;
      state.latest=null;state.latestAt=0;
      load();
    });
  });
  $("doctorIcloudInboxReload").addEventListener("click", () => {
    state.pages = [null];state.pageIndex = 0;load();
  });
  $("doctorIcloudRules")?.addEventListener("click",()=>{
    if(!state.active||!triage)return;
    clear(true);const ticket=state.epoch;
    void triage.manager($("doctorIcloudInboxDetails"),()=>current(ticket)).then(meta=>{if(current(ticket))state.projection=meta;});
  });
  window.addEventListener("cockpit:icloud-triage-changed",()=>{
    clear(true);
    if(state.active&&!view.hidden)void load();
  });
  async function qualifyReturn(){
    if(!triage||!state.active||view.hidden||document.hidden)return;
    const ticket=state.epoch;
    try{const prefs=await triage.qualify();if(current(ticket)&&state.projection&&!triage.same(prefs.triage,state.projection)){clear(true);void load();}}
    catch(_){if(current(ticket)){clear(true);$("doctorIcloudInboxState").textContent="Οι κατηγορίες δεν επιβεβαιώθηκαν. Επίλεξε Ανανέωση για όλα τα email.";}}
  }
  window.addEventListener("focus",qualifyReturn);
  document.addEventListener("visibilitychange",qualifyReturn);
  window.addEventListener("clinical:unauthenticated", ()=>close(true));
  window.addEventListener("clinical:authenticated", ()=>close(true));
  // Automatic 5-minute refresh belongs to the unread preview owner.
  // Never replace an actively opened full-message view on a preview tick.
  window.CockpitIcloudInbox = {open,close};
})();
