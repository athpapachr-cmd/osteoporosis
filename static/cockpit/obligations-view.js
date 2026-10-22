/* Standalone Clinical Obligations view — projection ONLY over existing protected
 * /clinical/clinical-updates and existing CockpitStoredChart.openPending authority.
 * No new task store, no email ingestion, no clinical status mutations here.
 */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const view = $("doctorObligationsView");
  if (!view) return;
  const dialog=$("doctorObligationDialog");
  const feedback=$("doctorObligationsFeedback");

  const state = { active: false, generation: 0, controller: null, rows: [],
    scope: "real", grouping: "patient", openPending: null, undo: null,
    modalToken: 0, modalBusy: false };
  const statuses = {
    open: "Ανοικτή", awaiting_patient: "Αναμένεται από ασθενή",
    awaiting_result: "Αναμένεται αποτέλεσμα"
  };
  const categories = {
    test: "Εξετάσεις", result: "Αποτελέσματα", investigation: "Εξετάσεις",
    document: "Έγγραφα", follow_up: "Επανεξέταση", referral: "Παραπομπή",
    other: "Λοιπές"
  };
  const categoryOrder = ["test", "investigation", "result", "document",
    "follow_up", "referral", "other"];
  const dateGroups = [
    ["overdue", "Εκπρόθεσμες"], ["today", "Για σήμερα"],
    ["future", "Μελλοντικές"], ["undated", "Χωρίς προθεσμία"]
  ];
  const fold = v => String(v || "").normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("el").trim();
  const iso = v => typeof v === "string" && /^\d{4}-\d{2}-\d{2}$/.test(v) ? v : null;
  const dateText = v => iso(v) ? v.slice(8,10) + "/" + v.slice(5,7) + "/" + v.slice(0,4) : "χωρίς προθεσμία";
  const localeName = (a,b) => String(a||"").localeCompare(String(b||""),"el");
  function cyprusToday() {
    const p = Object.fromEntries(new Intl.DateTimeFormat("en-GB", {
      timeZone:"Asia/Nicosia",year:"numeric",month:"2-digit",day:"2-digit"
    }).formatToParts(new Date()).filter(p => p.type !== "literal")
      .map(p => [p.type, p.value]));
    return p.year + "-" + p.month + "-" + p.day;
  }
  function dateClass(item,today) {
    const due = iso(item.review_date);
    if (!due) return "undated";
    if (due < today) return "overdue";
    return due === today ? "today" : "future";
  }
  function categoryKey(item) { return item.item_type==="investigation" ? "test" : (item.item_type||"other"); }
  function label(item) { return categories[categoryKey(item)] || categories.other; }
  function validItem(row) {
    return row && typeof row === "object" &&
      typeof row.pending_id === "string" && row.pending_id &&
      typeof row.patient_id === "string" && row.patient_id &&
      typeof row.description === "string" &&
      typeof row.patient_name === "string" &&
      Object.prototype.hasOwnProperty.call(statuses,row.status);
  }
  function node(tag,cls,text) {
    const n=document.createElement(tag);
    if(cls)n.className=cls;
    if(text!==undefined)n.textContent=String(text);
    return n;
  }
  function summary() {
    const today=cyprusToday(),n=state.rows.length;
    $("doctorObligationsTotal").textContent=String(n);
    $("doctorObligationsOverdue").textContent=String(state.rows.filter(x=>dateClass(x,today)==="overdue").length);
    $("doctorObligationsToday").textContent=String(state.rows.filter(x=>dateClass(x,today)==="today").length);
  }
  function rowsToShow(){
    const q=fold($("doctorObligationsSearch").value);
    const status=$("doctorObligationsStatusFilter").value;
    const category=$("doctorObligationsTypeFilter").value;
    return state.rows.filter(item =>
      (!status || item.status===status) &&
      (!category || categoryKey(item)===category) &&
      (!q || fold(item.patient_name+" "+item.description+" "+label(item)).includes(q)));
  }
  function groupRows(items){
    const mode=state.grouping,today=cyprusToday(),groups=new Map();
    for(const item of items){
      const category=categoryKey(item);
      const key=mode==="patient"?item.patient_id:
        mode==="category"?category:dateClass(item,today);
      if(!groups.has(key))groups.set(key,{key,heading:mode==="patient"?
        item.patient_name:mode==="category"?label(item):
        dateGroups.find(g=>g[0]===key)[1],items:[]});
      groups.get(key).items.push(item);
    }
    const result=[...groups.values()];
    const dateRank=k=>dateGroups.findIndex(x=>x[0]===k);
    result.sort((a,b)=>mode==="patient"?localeName(a.heading,b.heading):
      mode==="category"?categoryOrder.indexOf(a.key)-categoryOrder.indexOf(b.key):
      dateRank(a.key)-dateRank(b.key));
    result.forEach(g=>g.items.sort((a,b)=>String(a.review_date||"9999").localeCompare(String(b.review_date||"9999")) ||
      localeName(a.patient_name,b.patient_name) || String(a.pending_id).localeCompare(String(b.pending_id))));
    return result;
  }
  function makeButton(parent,title,fn,className){
    const b=node("button",className||"",title);b.type="button";
    b.addEventListener("click",fn);parent.append(b);return b;
  }
  function isWritable(item){
    return item?.checkpoint_write_allowed===true &&
      Number.isInteger(item?.lifecycle?.revision)&&item.lifecycle.revision>=0 &&
      ["open","awaiting_patient","awaiting_result"].includes(item.status);
  }
  function rowFor(item){
    const entry=node("div","doctor-obligation-entry");
    const row=node("button","doctor-obligation-row");
    row.type="button";
    row.dataset.pendingId=item.pending_id;
    row.setAttribute("aria-label","Άνοιγμα υποχρέωσης "+item.description+" · "+item.patient_name);
    const primary=node("span","doctor-obligation-main");
    primary.append(node("strong","",item.description));
    if(state.grouping!=="patient")
      primary.append(node("span","doctor-obligation-patient",item.patient_name));
    const secondary=node("span","doctor-obligation-meta");
    secondary.append(node("span","",label(item)+" · "+statuses[item.status]));
    if(iso(item.review_date)){
      secondary.append(node("span","doctor-obligation-date","Προθεσμία: "+dateText(item.review_date)));
      const group=dateClass(item,cyprusToday());
      if(group==="overdue"||group==="today")
        secondary.append(node("span","doctor-obligation-urgent",group==="overdue"?"Εκπρόθεσμη":"Για σήμερα"));
    } else secondary.append(node("span","doctor-obligation-no-date","Χωρίς προθεσμία"));
    primary.append(secondary);
    row.append(primary,node("span","doctor-obligation-chevron","›"));
    row.addEventListener("click",()=>{
      if(!state.active)return;
      // Historical #161 uses an isolated synthetic chart endpoint with no
      // normal metadata/lifecycle routes. It stays read-only in this view.
      if(window.CockpitSynthetic===true){
        state.openPending?.(item.patient_id,item.pending_id);
      }else showTask(item,false,row);
    });
    entry.append(row);
    if(window.CockpitSynthetic!==true && isWritable(item)){
      const edit=makeButton(entry,"✎ Επεξεργασία",()=>showTask(item,true,edit),
        "doctor-obligation-edit");
      edit.setAttribute("aria-label","Επεξεργασία κατηγορίας και προθεσμίας: "+item.description);
    }
    return entry;
  }
  function render() {
    if(!state.active)return;
    const root=$("doctorObligationsRows");
    root.replaceChildren();
    const visible=rowsToShow();
    $("doctorObligationsShown").textContent=String(visible.length)+" από "+state.rows.length+
      " εγγραφές της τρέχουσας προβολής";
    if(!visible.length){
      root.append(node("p","doctor-obligations-empty",state.rows.length?
        "Δεν υπάρχουν υποχρεώσεις με αυτά τα φίλτρα.":
        "Δεν υπάρχουν καταγεγραμμένες ενεργές υποχρεώσεις στην τρέχουσα προβολή."));
      return;
    }
    for(const group of groupRows(visible)){
      const section=node("section","doctor-obligation-group");
      const heading=node("h3","",group.heading);
      const count=node("span","doctor-obligation-count",group.items.length);
      const bar=node("div","doctor-obligation-group-head");
      bar.append(heading,count);section.append(bar);
      const buckets=state.grouping==="category"?
        [
          {name:"Με προθεσμία",items:group.items.filter(i=>Boolean(iso(i.review_date)))},
          {name:"Χωρίς προθεσμία",items:group.items.filter(i=>!iso(i.review_date))}
        ] : [{name:null,items:group.items}];
      for(const bucket of buckets){
        if(!bucket.items.length)continue;
        const region=node("div","doctor-obligation-subgroup");
        if(bucket.name){
          const subhead=node("div","doctor-obligation-subhead");
          subhead.append(node("h4","",bucket.name),
            node("span","doctor-obligation-count",bucket.items.length));
          region.append(subhead);
        }
        for(const item of bucket.items)region.append(rowFor(item));
        section.append(region);
      }
      root.append(section);
    }
  }

  function closeDialog(){
    state.modalToken++;
    state.modalBusy=false;
    if(dialog.open)dialog.close();
    dialog.replaceChildren(); // patient identity and editable form never remain hidden
  }
  function resetFeedback(){
    feedback.replaceChildren();
    state.undo=null;
  }
  function setFeedback(message,undo){
    feedback.replaceChildren();
    feedback.append(node("span","",message));
    state.undo=undo||null;
    if(!undo)return;
    makeButton(feedback,"Αναίρεση διαγραφής",undoLast,"doctor-obligation-undo");
  }
  function writeErrorText(err){
    if(err?.status===409)return "Η υποχρέωση άλλαξε σε άλλη συνεδρία. Έλεγξε ξανά την καταγεγραμμένη κατάσταση.";
    if(err?.status===401||err?.status===403)return "Για την ενέργεια χρειάζεται έγκυρη πρόσβαση γιατρού.";
    if(err?.status===503)return "Η αποθήκευση δεν είναι διαθέσιμη αυτή τη στιγμή. Ανανέωσε την κατάσταση.";
    if(err?.status===422)return "Έλεγξε την κατηγορία και την ημερομηνία.";
    return "Δεν επιβεβαιώθηκε η ενέργεια. Ανανέωσε την κατάσταση πριν επιχειρήσεις ξανά.";
  }
  async function postTask(item,suffix,body){
    const url="/clinical/patient/"+encodeURIComponent(item.patient_id)+
      "/pending/"+encodeURIComponent(item.pending_id)+suffix;
    const response=await fetch(url,{
      method:"POST",credentials:"same-origin",cache:"no-store",
      headers:{"Content-Type":"application/json","Accept":"application/json"},
      body:JSON.stringify(body)
    });
    if(!response.ok){
      const error=new Error("Clinical task change refused");error.status=response.status;
      if(response.status===401)window.dispatchEvent(new Event("clinical:request-login"));
      throw error;
    }
    return response.json();
  }
  async function undoLast(){
    const u=state.undo;
    if(!state.active||!u)return;
    resetFeedback();
    feedback.append(node("span","","Επαναφορά υποχρέωσης…"));
    // Existing reversible Pending lifecycle owner; no second task writer.
    try{
      await postTask(u,"/lifecycle",{
        action:"restore",expected_revision:u.revision,
        cancellation_event_id:u.eventId,idempotency_key:crypto.randomUUID()
      });
      if(!state.active)return;
      setFeedback("Η υποχρέωση επανήλθε.");
      load(true);
    }catch(err){
      if(!state.active)return;
      setFeedback(writeErrorText(err));
      load(true);
    }
  }
  function showTask(item,expandEditor=false,opener=null){
    if(!state.active||window.CockpitSynthetic===true)return;
    closeDialog();
    const ticket=state.generation,modalToken=state.modalToken;
    const writeAllowed=isWritable(item);
    const header=node("div","doctor-obligation-dialog-heading");
    const identity=node("div","doctor-obligation-dialog-identity");
    identity.append(node("p","doctor-obligation-patient",item.patient_name));
    identity.append(node("h3","",item.description));
    const headActions=node("div","doctor-obligation-dialog-head-actions");
    makeButton(headActions,"×",closeDialog,"doctor-obligation-dialog-close")
      .setAttribute("aria-label","Κλείσιμο υποχρέωσης");
    header.append(identity,headActions);dialog.append(header);
    const details=node("div","doctor-obligation-dialog-context");
    details.append(node("span","",label(item)+" · "+statuses[item.status]));
    details.append(node("span","",iso(item.review_date)?
      "Προθεσμία: "+dateText(item.review_date):"Χωρίς προθεσμία"));
    dialog.append(details);
    const actions=node("div","doctor-obligation-dialog-actions");
    const status=node("p","doctor-obligation-dialog-status","");
    status.setAttribute("role","status");
    if(writeAllowed){
      const editor=node("details","pending-metadata-editor");
      editor.open=Boolean(expandEditor);
      editor.append(node("summary","","Αλλαγή κατηγορίας / προθεσμίας"));
      const fields=node("div","pending-metadata-fields");
      const catLabel=node("label","","Κατηγορία");
      const cat=node("select");cat.setAttribute("aria-label","Κατηγορία υποχρέωσης");
      const opts=[
        ["test","Εξετάσεις"],["result","Αποτελέσματα"],
        ["document","Έγγραφα"],["follow_up","Επανεξέταση"],
        ["referral","Παραπομπή"],["other","Λοιπές"]
      ];
      const selected=categoryKey(item);
      if(!opts.some(v=>v[0]===selected)){
        const opt=node("option","","Άλλη υπάρχουσα κατηγορία");
        opt.value=item.item_type;cat.append(opt);
      }
      for(const [val,title]of opts){
        const opt=node("option","",title);opt.value=val;cat.append(opt);
      }
      cat.value=opts.some(v=>v[0]===selected)?selected:item.item_type;
      catLabel.append(cat);
      const dueLabel=node("label","","Προθεσμία");
      const due=node("input");due.type="date";due.value=iso(item.review_date)||"";
      due.setAttribute("aria-label","Προθεσμία υποχρέωσης");dueLabel.append(due);
      fields.append(catLabel,dueLabel);editor.append(fields);
      const suggested=!item.review_date&&typeof window.CockpitObligationDateSuggestion==="function"?
        window.CockpitObligationDateSuggestion(item.description,item.source_date):null;
      if(suggested){
        const suggestion=node("div","pending-metadata-suggestion");
        suggestion.append(node("small","",
          (suggested.approximate?"Ενδεικτική ημερομηνία (~15 ημέρες για μισό μήνα)":"Πρόταση από τίτλο")+
          " · από επίσκεψη "+dateText(suggested.sourceDate)));
        makeButton(suggestion,"Χρήση πρότασης · "+dateText(suggested.date),()=>{
          due.value=suggested.date;due.focus();
        },"doctor-obligation-suggestion");
        editor.append(suggestion);
      }
      editor.append(node("p","doctor-obligation-editor-help",
        "Η ημερομηνία δεν ολοκληρώνει την υποχρέωση. Το αρχικό κείμενο παραμένει αμετάβλητο."));
      const saveArea=node("div","pending-metadata-actions");
      makeButton(saveArea,"Αποθήκευση",async ()=>{
        if(state.modalBusy)return;
        const chosen=cat.value===selected?item.item_type:cat.value;
        const review_date=due.value||null;
        if(chosen===item.item_type&&review_date===(item.review_date||null)){
          status.textContent="Δεν υπάρχουν αλλαγές.";return;
        }
        if(review_date&&!iso(review_date)){
          status.textContent="Επίλεξε έγκυρη ημερομηνία.";return;
        }
        state.modalBusy=true;saveArea.querySelectorAll("button").forEach(b=>b.disabled=true);
        status.textContent="Αποθήκευση…";
        const userTurn=state.generation;
        try{
          await postTask(item,"/metadata",{
            item_type:chosen,review_date,expected_revision:item.lifecycle.revision,
            idempotency_key:crypto.randomUUID()
          });
          if(!state.active||userTurn!==state.generation)return;
          closeDialog();
          setFeedback("Η κατηγορία / προθεσμία αποθηκεύτηκε.");
          window.dispatchEvent(new Event("cockpit:clinical-updates-changed"));
        }catch(err){
          if(!state.active||userTurn!==state.generation)return;
          state.modalBusy=false;
          saveArea.querySelectorAll("button").forEach(b=>b.disabled=false);
          status.textContent=writeErrorText(err);
          // Error/409 retains clinician inputs; never retry with new key blindly.
        }
      },"doctor-obligation-save");
      makeButton(saveArea,"Χωρίς προθεσμία",()=>{due.value="";due.focus();});
      makeButton(saveArea,"Άκυρο",()=>{
        cat.value=opts.some(v=>v[0]===selected)?selected:item.item_type;
        due.value=iso(item.review_date)||"";
        editor.open=false;status.textContent="";
      });
      editor.append(saveArea);dialog.append(editor);
      const confirm=node("div","doctor-obligation-delete-confirmation");
      confirm.hidden=true;
      confirm.append(node("p","","Να διαγραφεί αυτή η υποχρέωση; Μπορείς να την επαναφέρεις."));
      const confirmActions=node("div","pending-metadata-actions");
      makeButton(confirmActions,"Διαγραφή",async()=>{
        if(state.modalBusy)return;
        state.modalBusy=true;
        confirmActions.querySelectorAll("button").forEach(b=>b.disabled=true);
        status.textContent="Διαγραφή…";
        const userTurn=state.generation;
        try{
          const data=await postTask(item,"/lifecycle",{
            action:"cancel",expected_revision:item.lifecycle.revision,
            idempotency_key:crypto.randomUUID()
          });
          if(!state.active||userTurn!==state.generation)return;
          closeDialog();
          const v=data?.lifecycle?.revision;
          const undo=typeof data?.event_id==="string" && Number.isInteger(v)?
            {patient_id:item.patient_id,pending_id:item.pending_id,
              eventId:data.event_id,revision:v}:null;
          setFeedback("Η υποχρέωση διαγράφηκε.",undo);
          window.dispatchEvent(new Event("cockpit:clinical-updates-changed"));
        }catch(err){
          if(!state.active||userTurn!==state.generation)return;
          state.modalBusy=false;
          confirmActions.querySelectorAll("button").forEach(b=>b.disabled=false);
          status.textContent=writeErrorText(err);
        }
      },"doctor-obligation-confirm-delete");
      makeButton(confirmActions,"Διατήρηση",()=>{confirm.hidden=true;status.textContent="";});
      confirm.append(confirmActions);
      makeButton(actions,"Διαγραφή",()=>{
        if(state.modalBusy)return;
        confirm.hidden=false;confirm.querySelector("button")?.focus();
      },"doctor-obligation-delete");
      dialog.append(confirm);
    }else{
      dialog.append(node("p","doctor-obligation-readonly",
        "Για αλλαγές απαιτείται ενεργή υποχρέωση και επιβεβαιωμένη πρόσβαση γιατρού."));
    }
    makeButton(actions,"Άνοιγμα στον φάκελο ασθενούς",()=>{
      if(state.modalBusy)return;
      closeDialog();
      state.openPending?.(item.patient_id,item.pending_id);
    },"doctor-obligation-open-chart");
    makeButton(actions,"Κλείσιμο",closeDialog);
    dialog.append(status,actions);
    dialog.showModal();
    // Keep item identifiers only while modal is visible; closing purges DOM.
    if(expandEditor)dialog.querySelector(".pending-metadata-editor select")?.focus();
    else dialog.querySelector(".doctor-obligation-dialog-close")?.focus();
  }
  dialog.addEventListener("click",e=>{
    if(e.target===dialog&&!state.modalBusy)closeDialog();
  });
  dialog.addEventListener("cancel",e=>{
    if(state.modalBusy)e.preventDefault();
  });

  function resetContents() {
    state.rows=[];
    $("doctorObligationsRows").replaceChildren();
    $("doctorObligationsState").textContent="";
    $("doctorObligationsShown").textContent="";
    for(const id of ["doctorObligationsTotal","doctorObligationsOverdue","doctorObligationsToday"])
      $(id).textContent="—";
  }
  function cancel() {
    state.generation++;
    if(state.controller)state.controller.abort();
    state.controller=null;
  }
  function close() {
    cancel();
    closeDialog();
    resetFeedback();
    state.active=false;
    state.openPending=null;
    // Clearing result rows is insufficient: a typed patient name remains in
    // the hidden search field otherwise, including after clinician logout.
    $("doctorObligationsSearch").value="";
    $("doctorObligationsStatusFilter").value="";
    $("doctorObligationsTypeFilter").value="";
    resetContents();
  }
  async function load(preserveRows=false) {
    cancel();
    const ticket=state.generation;
    const controller=new AbortController();
    state.controller=controller;
    if(!preserveRows)resetContents();
    $("doctorObligationsState").textContent="Φόρτωση προστατευμένων υποχρεώσεων…";
    $("doctorObligationsScope").textContent=state.scope==="real"?
      "Προβολή δοκιμαστικού ασθενούς":"Επιστροφή σε πραγματικούς φακέλους";
    try {
      // The existing #161 sandbox has its own strictly synthetic clinical base;
      // normal Cockpit always uses the actual protected /clinical owner.
      const base=window.CockpitSynthetic===true && window.CockpitClinicalBase==="/clinical/synthetic-161"
        ? window.CockpitClinicalBase : "/clinical";
      const response=await fetch(base+"/clinical-updates?scope="+
        encodeURIComponent(state.scope)+"&active_only=true",{
        credentials:"same-origin",cache:"no-store",signal:controller.signal,
        headers:{"Accept":"application/json"}
      });
      if(!response.ok) {
        const err=new Error("Request failed");err.status=response.status;throw err;
      }
      const body=await response.json();
      if(!state.active||ticket!==state.generation||view.hidden)return;
      if(!body||!Array.isArray(body.items))throw Error("Invalid protected projection");
      // Synthetic #161 predates active_only and returns resolved history too.
      // The clinic endpoint filters active tasks *before* its 100-row limit.
      const active=body.items.filter(x=>["open","awaiting_patient","awaiting_result"].includes(x?.status));
      if(active.some(x=>!validItem(x))||active.length>100)
        throw Error("Invalid protected projection");
      state.rows=active;
      $("doctorObligationsState").textContent=state.scope==="test"?
        "Δοκιμαστικός φάκελος · χωρίς ανάμειξη με πραγματικές εγγραφές.":
        "Ενεργές κλινικές υποχρεώσεις · οι ενέργειες γίνονται μέσα στην καρτέλα.";
      summary();render();
    }catch(err){
      if(!state.active||ticket!==state.generation||view.hidden)return;
      resetContents();
      $("doctorObligationsState").textContent=
        err.status===401||err.status===403?
          "Απαιτείται έγκυρη κλινική σύνδεση. Δεν εμφανίζονται στοιχεία ασθενών.":
        err.status===503?"Η κλινική πηγή δεν είναι διαθέσιμη.":
          "Οι υποχρεώσεις δεν φορτώθηκαν. Δοκίμασε Ανανέωση.";
    }
  }
  function open(onItem) {
    state.active=true;
    state.openPending=onItem;
    closeDialog();
    resetFeedback();
    state.scope="real";
    state.grouping="patient";
    $("doctorObligationsSearch").value="";
    $("doctorObligationsStatusFilter").value="";
    $("doctorObligationsTypeFilter").value="";
    document.querySelectorAll("[data-doctor-obligation-group]").forEach(button=>{
      const selected=button.dataset.doctorObligationGroup==="patient";
      button.setAttribute("aria-pressed",String(selected));
      button.classList.toggle("selected",selected);
    });
    load();
  }
  $("doctorObligationsSearch").addEventListener("input",render);
  $("doctorObligationsStatusFilter").addEventListener("change",render);
  $("doctorObligationsTypeFilter").addEventListener("change",render);
  document.querySelectorAll("[data-doctor-obligation-group]").forEach(button=>{
    button.addEventListener("click",()=>{
      if(!state.active)return;
      state.grouping=button.dataset.doctorObligationGroup;
      document.querySelectorAll("[data-doctor-obligation-group]").forEach(other=>{
        const selected=other===button;
        other.classList.toggle("selected",selected);
        other.setAttribute("aria-pressed",String(selected));
      });
      render();
    });
  });
  $("doctorObligationsReload").addEventListener("click",()=>{if(state.active)load();});
  $("doctorObligationsScope").addEventListener("click",()=>{
    if(!state.active)return;
    state.scope=state.scope==="real"?"test":"real";load();
  });
  window.addEventListener("clinical:unauthenticated",close);
  window.addEventListener("clinical:authenticated",()=>{if(state.active)load();});
  window.addEventListener("cockpit:clinical-updates-changed",()=>{if(state.active)load(true);});
  window.CockpitObligations={open,close};
})();
