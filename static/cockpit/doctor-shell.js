(() => {
  "use strict";
  // Read-only Cockpit shell. Appointment content comes from the EXISTING
  // protected clinical calendar; summaries or unread counts are never invented.
  const $ = id => document.getElementById(id);
  const buttons = {day: $("doctorDateButton"), inbox: $("doctorInboxButton"),
    tasks: $("doctorTasksButton"), actions: $("doctorActionsButton")};
  const popover = $("doctorPopover"), content = $("doctorPopoverContent");
  const state = {panel: null, opener: null, request: 0, surgeries: null};
  const zone = "Asia/Nicosia";
  const dateLabel = new Intl.DateTimeFormat("el-CY", {
    timeZone: zone, weekday: "short", day: "numeric", month: "short"
  });
  const timeLabel = new Intl.DateTimeFormat("el-CY", {
    timeZone: zone, hour: "2-digit", minute: "2-digit", hour12: false
  });
  const fullDate = new Intl.DateTimeFormat("en-GB", {
    timeZone: zone, year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23"
  });

  function localParts(date) {
    const p = Object.fromEntries(fullDate.formatToParts(date)
      .filter(x => x.type !== "literal").map(x => [x.type, Number(x.value)]));
    return p;
  }
  function midnightUTC(year, month, day) {
    // Probe at UTC midnight for the local offset before any Cyprus DST shift
    // later in the day. Start/end are computed independently.
    const probe = new Date(Date.UTC(year, month - 1, day));
    const local = localParts(probe);
    const offset = Date.UTC(local.year, local.month - 1, local.day,
      local.hour, local.minute, local.second) - probe.getTime();
    return new Date(Date.UTC(year, month - 1, day) - offset);
  }
  function todayRange(now = new Date()) {
    const p = localParts(now);
    const next = new Date(Date.UTC(p.year, p.month - 1, p.day + 1));
    return {
      start: midnightUTC(p.year, p.month, p.day).toISOString(),
      end: midnightUTC(next.getUTCFullYear(), next.getUTCMonth() + 1,
        next.getUTCDate()).toISOString()
    };
  }
  function text(message) {
    const p = document.createElement("p");
    p.className = "doctor-help";
    p.textContent = message;
    return p;
  }
  function newLink(label, href, meta) {
    const a = document.createElement("a");
    a.className = "doctor-action-row";
    a.href = href;
    if (/^https:\/\//.test(href)) {
      a.target = "_blank"; a.rel = "noopener noreferrer";
    }
    const body = document.createElement("span");
    const strong = document.createElement("strong");strong.textContent = label;
    body.append(strong);
    if (meta) {const small = document.createElement("small");small.textContent = meta;body.append(small);}
    const arrow = document.createElement("span");arrow.textContent = "›";
    a.append(body, arrow);
    return a;
  }
  function action(label, meta, fn) {
    const b = document.createElement("button");b.type = "button";b.className = "doctor-action-row";
    const copy=document.createElement("span"),strong=document.createElement("strong");
    strong.textContent=label;copy.append(strong);
    if(meta){const small=document.createElement("small");small.textContent=meta;copy.append(small);}
    const arrow=document.createElement("span");arrow.textContent="›";
    b.append(copy,arrow);b.addEventListener("click",fn);return b;
  }
  function sectionTitle(label) {
    const p = document.createElement("p");p.className="doctor-panel-kicker";p.textContent=label;return p;
  }
  function close() {
    state.request++;
    if(state.opener)state.opener.setAttribute("aria-expanded","false");
    const focus=state.opener;
    state.panel=null;state.opener=null;popover.hidden=true;
    content.replaceChildren();
    if(focus && typeof focus.focus==="function")focus.focus();
  }
  function openPanel(kind, opener) {
    if(state.panel===kind && !popover.hidden){close();return;}
    if(state.opener)state.opener.setAttribute("aria-expanded","false");
    state.request++;
    state.panel=kind;state.opener=opener;opener.setAttribute("aria-expanded","true");
    popover.hidden=false;
    popover.dataset.kind=kind;
    content.replaceChildren();
    $("doctorPopoverTitle").textContent={
      day:"Σημερινό πρόγραμμα",inbox:"Κλινικά εισερχόμενα",
      tasks:"Εκκρεμότητες ιατρείου",actions:"Ενέργειες"
    }[kind];
    if(kind==="day")loadDay();
    else if(kind==="inbox")inbox();
    else if(kind==="tasks")tasks();
    else commands();
  }
  async function loadDay() {
    const ticket=state.request;
    content.append(text("Φόρτωση κλινικού ημερολογίου…"));
    try {
      const range=todayRange();
      const q=new URLSearchParams({start:range.start,end:range.end});
      const response=await fetch("/clinical/calendar/appointments?"+q, {
        credentials:"same-origin",headers:{"Accept":"application/json"}
      });
      if(!response.ok) {const error=new Error("Unavailable");error.status=response.status;throw error;}
      const rows=await response.json();
      if(ticket!==state.request||state.panel!=="day")return;
      if(!Array.isArray(rows))throw Error("Invalid schedule response");
      content.replaceChildren();
      content.append(text("Επισκέψεις που καλύπτει το υπάρχον κλινικό ημερολόγιο. Δεν περιλαμβάνει κατ’ ανάγκη όλο το πρόγραμμα της Γραμματείας."));
      content.append(sectionTitle("Σήμερα · "+rows.length+" εγγραφές"));
      if(!rows.length)content.append(text("Δεν υπάρχουν διαθέσιμα ραντεβού σε αυτή την κλινική προβολή."));
      rows.sort((a,b)=>String(a.start_at).localeCompare(String(b.start_at)))
        .forEach(row=>{
          const start=new Date(/(?:Z|[+-]\d{2}:?\d{2})$/.test(row.start_at)?row.start_at:row.start_at+"Z");
          const time=Number.isNaN(start.getTime())?"—":timeLabel.format(start);
          const name=String(row.patient_display_name||"Ραντεβού χωρίς καταχωρισμένο όνομα");
          const reason=String(row.reason||row.category||"Ραντεβού");
          content.append(action(time+" · "+name,reason,()=>{
            close();
            if(window.CockpitHome?.openAppointment)window.CockpitHome.openAppointment({
              kind:"appointment",name,reason,time,appointmentId:row.appointment_id,
              patientId:null
            });
          }));
        });
    }catch(error){
      if(ticket!==state.request||state.panel!=="day")return;
      content.replaceChildren();
      content.append(text(error.status===401||error.status===403
        ?"Απαιτείται σύνδεση με το Clinical Data Key."
        :"Το κλινικό ημερολόγιο δεν είναι διαθέσιμο ή δεν καλύπτει πλήρως την ημέρα."));
    }
    if(ticket===state.request&&state.panel==="day"){
      content.append(newLink("Πλήρες πρόγραμμα Γραμματείας",
        "https://ortho-reception-backend-v2.onrender.com/dashboard",
        "Για όλες τις κρατήσεις και την τρέχουσα διαθεσιμότητα"));
      content.append(newLink("Εβδομαδιαίο κλινικό ημερολόγιο",
        "/static/clinical-calendar/"));
    }
  }
  function inbox() {
    content.append(text("Δεν έχει συνδεθεί ακόμη το Clinical Inbox με το Cockpit. Δεν υπάρχει αξιόπιστη μέτρηση νέων κλινικών μηνυμάτων, συνεπώς δεν εμφανίζεται αριθμός."));
    content.append(newLink("Άνοιγμα Gmail","https://mail.google.com/","Έλεγχος μηνυμάτων απευθείας στο Gmail"));
  }
  function tasks() {
    content.replaceChildren();
    const s=state.surgeries;
    content.append(sectionTitle("Κλινική εργασία"));
    if(s){
      content.append(action("Εκκρεμή χειρουργεία · "+s.total,
        s.undated+" χωρίς ορισμένη ημερομηνία",()=>showLibrary("surgery")));
    }else{
      content.append(action("Χειρουργεία","Άνοιγμα της προστατευμένης λίστας",()=>showLibrary("surgery")));
    }
    content.append(newLink("Ιατρική έκθεση","/clinical/clinic-utilities/medical-report"));
    content.append(newLink("Ραδιοκύματα / RF","/clinical/clinic-utilities/rf"));
    content.append(text("Συγκεντρωμένη παρακολούθηση RF, emails και εργασιών Dia/Heidi δεν έχει ενεργοποιηθεί. Δεν παρουσιάζονται εικονικές εκκρεμότητες."));
  }
  const links=[
    {name:"Καταγραφή επίσκεψης",hint:"Visit Capture",href:"/static/cockpit/visit-capture/"},
    {name:"Παραπεμπτικό φυσιοθεραπείας",hint:"Κείμενο για ΓεΣΥ",href:"/clinical/clinic-utilities/physio-referral"},
    {name:"Αναρρωτική άδεια",hint:"Κλινικό έγγραφο",href:"/clinical/clinic-utilities/sick-leave"},
    {name:"RF · Ραδιοκύματα",hint:"Αίτηση",href:"/clinical/clinic-utilities/rf"},
    {name:"Ιατρική έκθεση",hint:"Σύνταξη έκθεσης",href:"/clinical/clinic-utilities/medical-report"},
    {name:"Οστεοπόρωση",hint:"Module 01",href:"/static/baseline-audit/"},
    {name:"Εβδομαδιαίο ημερολόγιο",hint:"Οστεοπόρωση και κλινικό πρόγραμμα",href:"/static/clinical-calendar/"},
    {name:"Γραμματεία",hint:"Κλήσεις και πλήρες πρόγραμμα",href:"https://ortho-reception-backend-v2.onrender.com/dashboard"},
    {name:"Εκπαίδευση",hint:"Clinical Learning",href:"/static/clinical-learning/"}
  ];
  function commands() {
    const input=document.createElement("input");input.type="search";input.id="doctorCommandSearch";
    input.className="doctor-command-search";input.placeholder="Βρες ενέργεια…";
    input.autocomplete="off";input.setAttribute("aria-label","Αναζήτηση ενεργειών");
    const list=document.createElement("div");list.className="doctor-command-results";
    content.append(input,list);
    const draw=()=>{
      list.replaceChildren();const q=input.value.toLocaleLowerCase("el");
      for(const item of links.filter(x=>(x.name+" "+x.hint).toLocaleLowerCase("el").includes(q))){
        list.append(newLink(item.name,item.href,item.hint));
      }
      if(!list.children.length)list.append(text("Δεν βρέθηκε ενέργεια."));
      list.append(action("Εργασίες ιατρείου", "Χειρουργεία, έγγραφα, modules και εκπαίδευση",
        ()=>showLibrary("surgery")));
    };
    input.addEventListener("input",draw);draw();input.focus();
  }
  function showLibrary(category="surgery") {
    close();
    $("doctorMainView").hidden=true;
    $("doctorLibraryView").hidden=false;
    const panels=document.querySelectorAll("[data-doctor-library-panel]");
    panels.forEach(p=>p.hidden=p.dataset.doctorLibraryPanel!==category);
    document.querySelectorAll("[data-doctor-library-tab]").forEach(b=>{
      const selected=b.dataset.doctorLibraryTab===category;
      b.classList.toggle("selected",selected);b.setAttribute("aria-pressed",String(selected));
    });
    $("doctorBackHome").focus();
  }
  function showHome() {
    close();$("doctorLibraryView").hidden=true;$("doctorMainView").hidden=false;
    $("doctorActionsButton").focus();
  }
  function attention() {
    const target=$("doctorAttentionList");target.replaceChildren();
    const s=state.surgeries;
    target.append(action(
      s?"Χειρουργεία · "+s.total+" εκκρεμή":"Εκκρεμή χειρουργεία",
      s?s.undated+" χωρίς ημερομηνία":"Άνοιγμα λίστας και προγραμματισμού",
      ()=>showLibrary("surgery")));
    target.append(action("Έγγραφα · RF και εκθέσεις",
      "Άμεση πρόσβαση στη σύνταξη",()=>showLibrary("documents")));
    target.append(action("Κλινικές ενημερώσεις",
      "Η σύνδεση με το Clinical Inbox εκκρεμεί",
      ()=>openPanel("inbox",buttons.inbox)));
  }
  buttons.day.addEventListener("click",()=>openPanel("day",buttons.day));
  buttons.inbox.addEventListener("click",()=>openPanel("inbox",buttons.inbox));
  buttons.tasks.addEventListener("click",()=>openPanel("tasks",buttons.tasks));
  buttons.actions.addEventListener("click",()=>openPanel("actions",buttons.actions));
  $("doctorPopoverClose").addEventListener("click",close);
  $("doctorBackHome").addEventListener("click",showHome);
  document.querySelectorAll("[data-doctor-library-tab]").forEach(b=>{
    b.addEventListener("click",()=>showLibrary(b.dataset.doctorLibraryTab));
  });
  document.addEventListener("keydown",e=>{
    if(e.key==="Escape"&&!popover.hidden){e.preventDefault();close();}
    else if(e.key.toLowerCase()==="k"&&(e.metaKey||e.ctrlKey)){
      e.preventDefault();openPanel("actions",buttons.actions);
    }
  });
  document.addEventListener("pointerdown",e=>{
    if(!popover.hidden&&!popover.contains(e.target)&&
       !Object.values(buttons).some(b=>b.contains(e.target)))close();
  });
  $("doctorRailToggle").addEventListener("click",()=>{
    const next=!document.body.classList.contains("doctor-rail-open");
    document.body.classList.toggle("doctor-rail-open",next);
    $("doctorRailToggle").setAttribute("aria-expanded",String(next));
  });
  function updateSurgeryCounts(detail) {
    if(!detail||!Number.isInteger(detail.total)||detail.total<0||
       !Number.isInteger(detail.undated)||detail.undated<0)return;
    state.surgeries={total:detail.total,undated:detail.undated};
    $("doctorTasksBadge").hidden=detail.total===0;
    $("doctorTasksBadge").textContent=String(detail.total);
    attention();
    if(state.panel==="tasks")tasks();
  }
  window.addEventListener("cockpit:surgery-counts",e=>updateSurgeryCounts(e.detail));
  // No fake unread-mail badge. It remains hidden until the real Inbox owner exists.
  $("doctorInboxBadge").hidden=true;
  buttons.day.firstChild.textContent=dateLabel.format(new Date())+" ";
  attention();
  updateSurgeryCounts(window.CockpitSurgerySummary);
})();