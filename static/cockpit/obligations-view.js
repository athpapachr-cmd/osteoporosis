/* Standalone Clinical Obligations view — projection ONLY over existing protected
 * /clinical/clinical-updates and existing CockpitStoredChart.openPending authority.
 * No new task store, no email ingestion, no clinical status mutations here.
 */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const view = $("doctorObligationsView");
  if (!view) return;

  const state = { active: false, generation: 0, controller: null, rows: [],
    scope: "real", grouping: "patient", openPending: null };
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
  function label(item) { return categories[item.item_type] || categories.other; }
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
      (!category || (item.item_type||"other")===category) &&
      (!q || fold(item.patient_name+" "+item.description+" "+label(item)).includes(q)));
  }
  function groupRows(items){
    const mode=state.grouping,today=cyprusToday(),groups=new Map();
    for(const item of items){
      const category=item.item_type||"other";
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
  function render() {
    if(!state.active) return;
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
      for(const item of group.items){
        const row=node("button","doctor-obligation-row");
        row.type="button";
        row.dataset.pendingId=item.pending_id;
        row.setAttribute("aria-label","Άνοιγμα υποχρέωσης "+item.description+
          " · "+item.patient_name);
        const primary=node("span","doctor-obligation-main");
        primary.append(node("strong","",item.description));
        if(state.grouping!=="patient")primary.append(node("span","doctor-obligation-patient",item.patient_name));
        const secondary=node("span","doctor-obligation-meta");
        secondary.append(node("span","",label(item)+" · "+statuses[item.status]));
        if(iso(item.review_date))secondary.append(node("span","doctor-obligation-date","Προθεσμία: "+dateText(item.review_date)));
        primary.append(secondary);row.append(primary,node("span","doctor-obligation-chevron","›"));
        row.addEventListener("click",()=>{
          if(!state.active||typeof state.openPending!=="function")return;
          // Navigation first, then the existing chart owner performs a protected
          // patient/pending read. No capability is created by this list row.
          state.openPending(item.patient_id,item.pending_id);
        });
        section.append(row);
      }
      root.append(section);
    }
  }
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
    state.active=false;
    state.openPending=null;
    resetContents(); // identifiers never persist in a hidden screen
  }
  async function load() {
    cancel();
    const ticket=state.generation;
    const controller=new AbortController();
    state.controller=controller;
    resetContents();
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
  window.addEventListener("cockpit:clinical-updates-changed",()=>{if(state.active)load();});
  window.CockpitObligations={open,close};
})();
