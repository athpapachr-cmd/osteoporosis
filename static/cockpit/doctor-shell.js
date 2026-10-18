(() => {
  "use strict";
 const clinicalURL=path=>(window.CockpitClinicalBase||"/clinical")+path.slice(9);
  // Read-only Cockpit shell. Appointment content comes from the EXISTING
  // protected clinical calendar; summaries or unread counts are never invented.
  const $ = id => document.getElementById(id);
  const buttons = {day: $("doctorDateButton"), inbox: $("doctorInboxButton"),
    tasks: $("doctorTasksButton"), actions: $("doctorActionsButton")};
  const popover = $("doctorPopover"), content = $("doctorPopoverContent");
  const state = {panel: null, opener: null, request: 0, day: null, surgeries: null, rf: null};
  let rfGeneration = 0;
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
  function dayKey(now=new Date()) {
    const p=localParts(now);return [p.year,String(p.month).padStart(2,'0'),String(p.day).padStart(2,'0')].join('-');
  }
  function selectedDate(){return new Date(state.day+'T12:00:00Z');}
  function dayHeading(){
    const today=dayKey(),tomorrow=new Date(today+'T12:00:00Z');tomorrow.setUTCDate(tomorrow.getUTCDate()+1);
    const relative=state.day===today?'Σήμερα':state.day===dayKey(tomorrow)?'Αύριο':'';
    const label=new Intl.DateTimeFormat('el-CY',{timeZone:zone,weekday:'long',day:'numeric',month:'long',year:'numeric'}).format(selectedDate());
    return (relative?relative+' · ':'')+label;
  }
  function dayNavigation(){
    const nav=document.createElement('div');nav.className='doctor-day-navigation';
    const line=document.createElement('div');line.className='doctor-day-stepper';
    function arrow(id,label,symbol,shift){const b=document.createElement('button');b.id=id;b.type='button';b.className='doctor-day-arrow';b.textContent=symbol;b.setAttribute('aria-label',label);b.title=label;
      b.addEventListener('click',()=>{const d=selectedDate();d.setUTCDate(d.getUTCDate()+shift);state.day=dayKey(d);loadDay(id);});return b;}
    const heading=document.createElement('p');heading.id='doctorDayLabel';heading.className='doctor-day-label';heading.textContent=dayHeading();heading.setAttribute('aria-live','polite');
    line.append(arrow('doctorDayPrevious','Προηγούμενη ημέρα','‹',-1),heading,arrow('doctorDayNext','Επόμενη ημέρα','›',1));
    const today=document.createElement('button');today.id='doctorDayToday';today.type='button';today.className='doctor-day-today';today.textContent='Σήμερα';today.disabled=state.day===dayKey();
    today.addEventListener('click',()=>{state.day=dayKey();loadDay('doctorDayNext');});nav.append(line,today);return nav;
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
      day:"Πρόγραμμα ημέρας",inbox:"Κλινικά εισερχόμενα",
      tasks:"Εκκρεμότητες ιατρείου",actions:"Ενέργειες"
    }[kind];
    if(kind==="day"){state.day=dayKey();loadDay();}
    else if(kind==="inbox")inbox();
    else if(kind==="tasks")tasks();
    else commands();
  }
  async function loadDay(focusId) {
    const ticket=++state.request;
    const navigation=dayNavigation();
    const clear=()=>{const focused=navigation.contains(document.activeElement)?document.activeElement:null;content.replaceChildren();content.append(navigation);focused?.focus({preventScroll:true});};
    clear();if(focusId)$(focusId)?.focus({preventScroll:true});
    content.append(text("Φόρτωση κλινικού ημερολογίου…"));
    try {
      const range=todayRange(selectedDate());
      const q=new URLSearchParams({start:range.start,end:range.end});
      const response=await fetch(clinicalURL("/clinical/calendar/appointments?"+q), {
        credentials:"same-origin",headers:{"Accept":"application/json"}
      });
      if(!response.ok) {const error=new Error("Unavailable");error.status=response.status;throw error;}
      const rows=await response.json();
      if(ticket!==state.request||state.panel!=="day")return;
      if(!Array.isArray(rows))throw Error("Invalid schedule response");
      clear();
      content.append(text("Επισκέψεις που καλύπτει το υπάρχον κλινικό ημερολόγιο. Δεν περιλαμβάνει κατ’ ανάγκη όλο το πρόγραμμα της Γραμματείας."));
      content.append(sectionTitle(rows.length+" ραντεβού"));
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
              kind:"appointment",name,reason,time,date:Number.isNaN(start.getTime())?null:new Intl.DateTimeFormat("en-CA",{timeZone:"Asia/Nicosia",year:"numeric",month:"2-digit",day:"2-digit"}).format(start),appointmentId:row.appointment_id,
              patientId:null
            });
          }));
        });
    }catch(error){
      if(ticket!==state.request||state.panel!=="day")return;
      clear();
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
    content.replaceChildren();
    content.append(text("Δεν έχει συνδεθεί ακόμη το Clinical Inbox με το Cockpit. Δεν υπάρχει αξιόπιστη μέτρηση νέων κλινικών μηνυμάτων, συνεπώς δεν εμφανίζεται αριθμός."));
    content.append(newLink("Άνοιγμα Gmail","https://mail.google.com/","Έλεγχος μηνυμάτων απευθείας στο Gmail"));
  }
  function tasks() {
    if(window.CockpitStoredChart){
      const ticket=state.request;
      $("doctorPopoverTitle").textContent="Υποχρεώσεις ασθενών";
      window.CockpitStoredChart.renderUpdates(content,()=>state.panel==="tasks"&&ticket===state.request,close);
      return;
    }
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
    content.append(text("Το Clinical Inbox και οι εργασίες Dia/Heidi δεν έχουν συνδεθεί με την παρακολούθηση."));
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
  const rfCounterNames = [
    "issued_unknown", "pending", "approved", "rejected",
    "scheduled", "performed", "mine_actionable"
  ];
  function validatedRF(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)
        || typeof value.truncated !== "boolean" || value.scope !== "all"
        || value.page !== 1 || !Number.isInteger(value.denominator)
        || value.denominator < 0 || !value.counts || typeof value.counts !== "object"
        || Array.isArray(value.counts)) return null;
    const numbers = {};
    for (const key of rfCounterNames) {
      const count = value.counts[key];
      if (!Number.isInteger(count) || count < 0 || count > value.denominator) return null;
      numbers[key] = count;
    }
    if (value.truncated) return { partial: true }; // Never suggest an exhaustive total.
    return { denominator: value.denominator, ...numbers };
  }

  async function loadRfAttention() {
    const generation = ++rfGeneration;
    state.rf = null; // Auth refresh invalidates prior RF counts synchronously.
    attention();
    try {
      const response = await fetch(clinicalURL("/clinical/clinic-utilities/rf/api/outcomes/search"), {
        method: "POST", credentials: "same-origin", cache: "no-store",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ query: "", scope: "all", page: 1 })
      });
      if (!response.ok) throw new Error("Protected RF source unavailable");
      const parsed = validatedRF(await response.json());
      if (generation !== rfGeneration) return;
      if (!parsed) throw new Error("Invalid RF count source");
      state.rf = parsed; // No individual RF applications, names or IDs in Home state.
    } catch (_) {
      if (generation !== rfGeneration) return;
      state.rf = null;
    }
    if (generation === rfGeneration) attention();
  }

  function surgeryAttentionLines(summary) {
    const lines = [];
    const remainder = [];
    if (summary.undated) remainder.push(summary.undated + " χωρίς ημερομηνία");
    if (summary.invalid) remainder.push(summary.invalid + " μη έγκυρες ημερομηνίες");
    if (summary.overdue) remainder.push(summary.overdue + " παλαιότερα εκκρεμή");
    if (remainder.length) lines.push(remainder.join(" · "));
    for (const group of summary.upcoming) {
      const day = dateLabel.format(new Date(group.date + "T12:00:00Z"));
      lines.push(group.count === 1
        ? day + " · " + group.name + " · " + group.procedure
        : day + " · " + group.count + " χειρουργεία");
    }
    if (!lines.length) lines.push(summary.total === 0
      ? "Δεν υπάρχουν εκκρεμή χειρουργεία" : "Δεν υπάρχουν επόμενες ημερομηνίες");
    return lines.join("\n");
  }

  function attention() {
    const target = $("doctorAttentionList"); target.replaceChildren();
    const s = state.surgeries, rf = state.rf;
    target.append(action(
      s ? "Χειρουργεία · " + s.total + " εκκρεμή" : "Χειρουργεία · Μη διαθέσιμα",
      s ? surgeryAttentionLines(s) : "Άνοιγμα προστατευμένης λίστας",
      () => showLibrary("surgery")));
    const rfTitle = rf?.partial ? "RF · Μερική κάλυψη"
      : rf ? "RF · Εκδοθείσες αιτήσεις " + rf.denominator
        + " · Αναμένουν απόφαση " + rf.pending : "RF · Μη διαθέσιμα";
    const rfInfo = rf?.partial ? "Μερική κάλυψη — άνοιξε RF"
      : rf ? "Εγκρίθηκαν " + rf.approved + " · Απορρίφθηκαν " + rf.rejected
        + "\nΠρογραμματίστηκαν " + rf.scheduled + " · Πραγματοποιήθηκαν " + rf.performed
        + " · Δικά μου προς παρακολούθηση " + rf.mine_actionable
      : "Άνοιξε RF για αναζήτηση";
    target.append(newLink(rfTitle, "/clinical/clinic-utilities/rf/tracking", rfInfo));
    target.append(newLink("Ιατρικές εκθέσεις", "/clinical/clinic-utilities/medical-report",
      "Δεν υπάρχει ακόμη αξιόπιστη καταμέτρηση."));
    target.append(action("Υποχρεώσεις ασθενών",
      "Ανοικτές εκκρεμότητες ανά ασθενή",
      () => openPanel("tasks", buttons.tasks)));
  }

    buttons.day.addEventListener("click",()=>openPanel("day",buttons.day));
  buttons.inbox.addEventListener("click",()=>openPanel("inbox",buttons.inbox));
  window.addEventListener("cockpit:clinical-updates-changed",()=>{if(state.panel==="inbox")inbox();else if(state.panel==="tasks")tasks();});
  window.addEventListener("clinical:unauthenticated",()=>{close();});
  buttons.tasks.addEventListener("click",()=>openPanel("tasks",buttons.tasks));
  buttons.actions.addEventListener("click",()=>openPanel("actions",buttons.actions));
  $("doctorPopoverClose").addEventListener("click",close);
  if (typeof window !== "undefined" && typeof window.addEventListener === "function") window.addEventListener("clinical:authenticated", () => {
    if(state.panel === "day") loadDay();
    loadRfAttention();
  });
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
    const counts = ["total", "undated", "invalid", "overdue"];
    const validGroups = detail && Array.isArray(detail.upcoming)
      && detail.upcoming.length <= 3 && detail.upcoming.every(group =>
        group && /^\d{4}-\d{2}-\d{2}$/.test(group.date)
        && Number.isInteger(group.count) && group.count >= 1
        && (group.count !== 1 || (typeof group.name === "string"
          && !!group.name.trim() && typeof group.procedure === "string"
          && !!group.procedure.trim())));
    const validCounts = detail && counts.every(k => Number.isInteger(detail[k]) && detail[k] >= 0);
    if (detail?.unavailable || !validCounts || !validGroups
        || detail.undated + detail.invalid + detail.overdue
          + (detail.upcoming || []).reduce((sum, group) => sum + group.count, 0) > detail.total) {
      state.surgeries = null;
      $("doctorTasksBadge").hidden = true;
      attention();
      if (state.panel === "tasks") tasks();
      return;
    }
    state.surgeries = {
      total: detail.total, undated: detail.undated, invalid: detail.invalid,
      overdue: detail.overdue,
      upcoming: detail.upcoming.map(group => group.count === 1
        ? { date: group.date, count: 1, name: group.name, procedure: group.procedure }
        : { date: group.date, count: group.count })
    };
    $("doctorTasksBadge").hidden = detail.total === 0;
    $("doctorTasksBadge").textContent = String(detail.total);
    attention();
    if (state.panel === "tasks") tasks();
  }
    window.addEventListener("cockpit:surgery-counts",e=>updateSurgeryCounts(e.detail));
  // No fake unread-mail badge. It remains hidden until the real Inbox owner exists.
  $("doctorInboxBadge").hidden=true;
  buttons.day.firstChild.textContent=dateLabel.format(new Date())+" ";
  attention();
  updateSurgeryCounts(window.CockpitSurgerySummary);
  loadRfAttention();
})();