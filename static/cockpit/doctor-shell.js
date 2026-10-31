(() => {
  "use strict";
 const clinicalURL=path=>(window.CockpitClinicalBase||"/clinical")+path.slice(9);
  // Read-only Cockpit shell. Appointment content comes from the EXISTING
  // protected clinical calendar; summaries or unread counts are never invented.
  const $ = id => document.getElementById(id);
  const buttons = {day: $("doctorDateButton"), calls: $("doctorCallsButton"), inbox: $("doctorInboxButton"),
    tasks: $("doctorTasksButton"), notion: $("doctorNotionButton"), actions: $("doctorActionsButton")};
  const popover = $("doctorPopover"), content = $("doctorPopoverContent");
  const state = {panel: null, opener: null, request: 0, day: null, surgeries: null, rf: null};
  let rfGeneration = 0;
  let notionRecorder=null,notionStream=null,notionTimer=null;
  function clearNotionMedia(){
    if(notionTimer){clearTimeout(notionTimer);notionTimer=null;}
    if(notionRecorder&&notionRecorder.state!=="inactive")notionRecorder.stop();
    notionRecorder=null;
    notionStream?.getTracks().forEach(track=>track.stop());notionStream=null;
  }
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
    clearNotionMedia();
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
    clearNotionMedia();
    state.panel=kind;state.opener=opener;opener.setAttribute("aria-expanded","true");
    popover.hidden=false;
    popover.dataset.kind=kind;
    content.replaceChildren();
    $("doctorPopoverTitle").textContent={
      day:"Πρόγραμμα ημέρας",calls:"Κλήσεις γραμματέα",inbox:"Κλινικά εισερχόμενα",
      tasks:"Εκκρεμότητες ιατρείου",notion:"Οι εργασίες μου",actions:"Ενέργειες"
    }[kind];
    if(kind==="day"){state.day=dayKey();loadDay();}
    else if(kind==="calls")callPeek();
    else if(kind==="inbox")inbox();
    else if(kind==="tasks")tasks();
    else if(kind==="notion")notionTasks();
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
      const response=await fetch(clinicalURL("/clinical/calendar/agenda?"+q), {
        credentials:"same-origin",headers:{"Accept":"application/json"}
      });
      if(!response.ok) {const error=new Error("Unavailable");error.status=response.status;throw error;}
      const rows=await response.json();
      if(ticket!==state.request||state.panel!=="day")return;
      if(!Array.isArray(rows))throw Error("Invalid schedule response");
      clear();
      content.append(text("Ραντεβού από τη διαθέσιμη προβολή της Γραμματείας για αυτή την ημέρα."));
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
  // The icons are PEEKS, not navigation shortcuts. Separate full-screen
  // actions remain explicit, and every read is scoped to the active popover.
  const syntheticInbox=()=>window.CockpitSynthetic===true &&
    window.CockpitClinicalBase==="/clinical/synthetic-161";
  const activePeek=(kind,ticket)=>state.panel===kind &&
    !popover.hidden && state.request===ticket;

  async function callPeek() {
    const ticket=++state.request;
    content.replaceChildren();
    content.append(text("Φόρτωση πρόσφατων κλήσεων από τη Γραμματεία…"));
    const dashboard="https://ortho-reception-backend-v2.onrender.com/dashboard";
    try {
      const response=await fetch(clinicalURL("/clinical/reception-calls/peek"),{
        credentials:"same-origin",cache:"no-store",headers:{"Accept":"application/json"}
      });
      if(!response.ok){const error=new Error("Unavailable");error.status=response.status;throw error;}
      const data=await response.json();
      if(!activePeek("calls",ticket))return;
      if(!data||data.source!=="reception_dashboard"||data.is_unread_count!==false||
          !["available_call_log_max_200","ambiguous_call_identity"].includes(data.coverage)||
          !(data.unhandled_count===null||(Number.isInteger(data.unhandled_count)&&
            data.unhandled_count>=0&&data.unhandled_count<=200))||
          !Array.isArray(data.calls)||data.calls.length>4||
          data.calls.some(c=>!c||(c.call_id!==null&&typeof c.call_id!=="string")||
            typeof c.logged_at!=="string"||typeof c.name!=="string"||
            typeof c.summary!=="string"||typeof c.kind!=="string"||
            ![true,false,null].includes(c.handled)||
            !(c.todo_id===null||(typeof c.todo_id==="string"&&c.todo_id.length>0))))
        throw Error("Invalid Reception source");
      content.replaceChildren();
      content.append(text(data.unhandled_count===null?
        "Ο αριθμός μη σημειωμένων κλήσεων δεν είναι διαθέσιμος λόγω ασαφούς ταυτότητας στην πηγή.":
        data.unhandled_count+" μη σημειωμένες ως διεκπεραιωμένες στη διαθέσιμη προβολή (έως 200 καταγεγραμμένες κλήσεις)."));
      if(!data.calls.length)content.append(text("Δεν υπάρχουν κλήσεις στην τρέχουσα προβολή της Γραμματείας."));
      for(const row of data.calls){
        const date=new Date(row.logged_at);
        const when=Number.isNaN(date.getTime())?"Χρόνος άγνωστος":
          dateLabel.format(date)+" · "+timeLabel.format(date);
        const status=row.summary_pending?" · Η τελική σύνοψη ετοιμάζεται":"";
        const kind=row.kind==="message"?" · Μήνυμα":"";
        const name=row.name+(row.phone?" · "+row.phone:"");
        const entry=document.createElement("div");entry.className="doctor-call-entry";
        entry.append(newLink(name,dashboard,when+kind+status+" · "+row.summary));
        const controls=document.createElement("div");controls.className="doctor-call-controls";
        const handled=text(row.handled===true?"✓ Σημειώθηκε ως διεκπεραιωμένη":
          row.handled===false?"Δεν έχει σημειωθεί ως διεκπεραιωμένη":
          "Η ατομική κατάσταση δεν είναι διαθέσιμη");
        handled.className="doctor-call-state";controls.append(handled);
        const dial=String(row.phone||"").trim();
        const dialTarget=dial.replace(/[^0-9+]/g,"");
        if(/^\+?[0-9 ()-]{5,40}$/.test(dial)&&dialTarget.replace(/\D/g,"").length>=5){
          const tel=document.createElement("a");tel.className="doctor-call-control";
          tel.href="tel:"+dialTarget;tel.textContent="Κλήση";
          tel.setAttribute("aria-label","Κλήση στο "+dial);controls.append(tel);
        }
        if(row.handled===false&&row.todo_id){
          const button=document.createElement("button");button.type="button";
          button.className="doctor-call-control";button.textContent="Σημείωση ολοκλήρωσης";
          button.addEventListener("click",async()=>{
            button.disabled=true;handled.textContent="Καταχώριση στη Γραμματεία…";
            try{
              const response=await fetch(clinicalURL("/clinical/reception-calls/handled"),{
                method:"POST",credentials:"same-origin",cache:"no-store",
                headers:{"Accept":"application/json","Content-Type":"application/json"},
                body:JSON.stringify({todo_id:row.todo_id,handled:true})
              });
              if((response.status===401||response.status===403)&&activePeek("calls",ticket)){
                window.dispatchEvent(new Event("clinical:unauthenticated"));return;
              }
              if(!response.ok)throw Error("Reception write failed");
              const result=await response.json();
              if(result?.handled!==true)throw Error("Reception write not confirmed");
              if(activePeek("calls",ticket))callPeek();
            }catch(_error){
              if(activePeek("calls",ticket)){
                handled.textContent="Η ολοκλήρωση δεν επιβεβαιώθηκε. Ελέγξτε ξανά την πηγή.";
                button.disabled=false;
              }
            }
          });controls.append(button);
        }
        entry.append(controls);content.append(entry);
      }
      content.append(newLink("Όλες οι κλήσεις και ο χειρισμός τους",dashboard,
        "Το dashboard του γραμματέα παραμένει η μοναδική πηγή κατάστασης."));
    }catch(error){
      if(!activePeek("calls",ticket))return;
      content.replaceChildren();
      content.append(text(error.status===401||error.status===403?
        "Απαιτείται κλινική σύνδεση. Δεν εμφανίζονται στοιχεία κλήσεων.":
        "Οι κλήσεις της Γραμματείας δεν είναι συνδεδεμένες ή η πηγή δεν είναι διαθέσιμη. Δεν συμπεραίνουμε ότι δεν υπάρχουν κλήσεις."));
      if(error.status===401||error.status===403)
        content.append(action("Σύνδεση","Χρήση του υπάρχοντος Clinical Key",
          ()=>window.dispatchEvent(new Event("clinical:request-login"))));
      content.append(newLink("Άνοιγμα dashboard Γραμματείας",dashboard));
    }
  }

  // Clinician-session, volatile unread-summary cache. No local/session storage,
  // Postgres, cross-principal data, raw IMAP, or background fetch without auth.
  const ICLOUD_PEEK_TTL_MS = 5 * 60 * 1000;
  // Only the currently open inbox popover may render its verified cache.
  // A background IMAP response cannot bypass a pending Clinical Key check.
  let verifiedUnreadPanelTicket = null;
  const unreadPeek = {
    authorized:false, generation:0, data:null, updatedAt:0, checkedAt:0,
    error:null, pending:null, controller:null
  };
  function icloudPeekContractCode(data) {
    if(data?.source!=="icloud_mail"||data.scope!=="recipient_alias_inbox_only"||
       data.filter!=="unseen")return "PEEK_SOURCE";
    if(!Array.isArray(data.items))return "PEEK_ITEMS_MISSING";
    if(data.items.length>4)return "PEEK_ITEMS_LIMIT";
    if(!Number.isInteger(data.skipped_count)||data.skipped_count<0||
       typeof data.coverage_complete!=="boolean")return "PEEK_COVERAGE";
    if(!(data.next_cursor===null||
      (typeof data.next_cursor==="string"&&
       /^[1-9][0-9]{0,18}-[1-9][0-9]{0,18}$/.test(data.next_cursor))))
      return "PEEK_CURSOR";
    if(data.items.some(x=>!x||typeof x.id!=="string"||
       typeof x.subject!=="string"||typeof x.from!=="string"||
       !["clinical","review"].includes(x.category)))return "PEEK_ITEM_FIELDS";
    return null;
  }
  function updateUnreadBadge() {
    // A dot means at least one verified UNSEEN message, never a false total.
    const badge=$("doctorInboxBadge");
    const hasUnread=!!(unreadPeek.data?.items?.length);
    badge.hidden=!hasUnread;
    badge.textContent=hasUnread?"•":"";
    buttons.inbox.setAttribute("aria-label",hasUnread?
      "Υπάρχουν μη αναγνωσμένα κλινικά email":"Μη αναγνωσμένα κλινικά email");
  }
  function clearUnreadPeek() {
    verifiedUnreadPanelTicket=null;
    unreadPeek.generation++;
    unreadPeek.controller?.abort();
    unreadPeek.authorized=false;unreadPeek.pending=null;unreadPeek.controller=null;
    unreadPeek.data=null;unreadPeek.updatedAt=0;unreadPeek.checkedAt=0;
    unreadPeek.error=null;updateUnreadBadge();
  }
  function displayUnreadError(err) {
    if(err?.status===401||err?.status===403) {
      content.append(text("Απαιτείται κλινική σύνδεση. Δεν εμφανίζονται μηνύματα."));
      content.append(action("Σύνδεση","Clinical Key",
        ()=>window.dispatchEvent(new Event("clinical:request-login"))));
      return;
    }
    if(err?.status===503) {
      const msg=window.CockpitIcloudStageMessage?.(err.code);
      content.append(text(msg?"iCloud Mail: "+msg+" Δεν γνωρίζουμε αν υπάρχουν νέα μηνύματα.":
        "Η πηγή iCloud δεν είναι διαθέσιμη [PEEK_SOURCE_503]. Δεν γνωρίζουμε αν υπάρχουν νέα μηνύματα."));
    } else if(err?.clientCode) {
      const safe=["PEEK_SOURCE","PEEK_ITEMS_MISSING","PEEK_ITEMS_LIMIT",
        "PEEK_COVERAGE","PEEK_CURSOR","PEEK_ITEM_FIELDS","PEEK_JSON"];
      const code=safe.includes(err.clientCode)?err.clientCode:"PEEK_RENDER";
      content.append(text("Το iCloud απάντησε HTTP 200 αλλά η προβολή μη αναγνωσμένων απέτυχε ["+
        code+"]. Δεν συμπεραίνουμε ότι η θυρίδα είναι άδεια."));
    } else {
      content.append(text(err?.status?
        "Αποτυχία αιτήματος iCloud [PEEK_HTTP_ERROR]. Δεν γνωρίζουμε αν υπάρχουν νέα μηνύματα.":
        "Αποτυχία δικτύου iCloud [PEEK_NETWORK]. Δεν γνωρίζουμε αν υπάρχουν νέα μηνύματα."));
    }
  }
  function displayUnreadPeek() {
    if(verifiedUnreadPanelTicket!==state.request ||
       !activePeek("inbox",state.request))return;
    content.replaceChildren();
    const data=unreadPeek.data;
    const outdated=unreadPeek.updatedAt &&
      Date.now()-unreadPeek.updatedAt>=ICLOUD_PEEK_TTL_MS;
    if(data){
      const checked=new Date(unreadPeek.updatedAt).toLocaleTimeString("el-CY",
        {hour:"2-digit",minute:"2-digit"});
      content.append(text("Μη αναγνωσμένα στο iCloud · ενημέρωση "+checked+
        (outdated?" · απαιτείται ανανέωση":"")));
      if(unreadPeek.error){
        content.append(text("Η τελευταία ανανέωση απέτυχε. Προβάλλεται παλαιότερη λίστα."));
        displayUnreadError(unreadPeek.error);
      }
      if(!data.coverage_complete)content.append(text(
        "Μερική κάλυψη: "+(data.skipped_count?
        data.skipped_count+" μηνύματα δεν ελέγχθηκαν. ":"")+
        (data.next_cursor?"Υπάρχουν παλαιότερα μη αναγνωσμένα προς έλεγχο.":
        "Ο έλεγχος δεν είναι επιβεβαιωμένα πλήρης.")));
      for(const item of data.items)content.append(action(
        (item.category==="review"?"Για έλεγχο · ":"")+item.subject,
        item.from+" · Μη αναγνωσμένο στο iCloud",()=>showIcloudInbox(item.id)));
      if(!data.items.length)content.append(text(data.coverage_complete?
        "Δεν υπάρχουν μη αναγνωσμένα μηνύματα προς το alias στο ελεγμένο INBOX.":
        "Δεν βρέθηκαν μη αναγνωσμένα στην ελεγμένη ενότητα. Η θυρίδα μπορεί να έχει παλαιότερα."));
    } else if(unreadPeek.error){
      displayUnreadError(unreadPeek.error);
    } else {
      content.append(text("Φόρτωση μη αναγνωσμένων iCloud για το ιατρείο…"));
    }
    if(unreadPeek.pending && data) content.append(text("Ανανέωση στο παρασκήνιο…"));
    content.append(action("Ανανέωση τώρα","Μη αναγνωσμένα στο iCloud",
      ()=>{unreadPeek.checkedAt=0;void fetchUnreadPeek(true,true);}));
    content.append(text("Όλα τα email: Συντομεύσεις → Εισερχόμενα ασθενών."));
  }
  function fetchUnreadPeek(force=false, manual=false) {
    if(!manual && (!unreadPeek.authorized || document.hidden))return null;
    if(unreadPeek.pending)return unreadPeek.pending;
    if(!force && unreadPeek.checkedAt &&
       Date.now()-unreadPeek.checkedAt<ICLOUD_PEEK_TTL_MS)return null;
    const generation=unreadPeek.generation;
    const controller=typeof AbortController==="function" ? new AbortController() :
      {signal:{aborted:false},abort(){this.signal.aborted=true;}};
    unreadPeek.controller=controller;
    const pending=(async()=>{
      let err=null;
      try{
        const response=await fetch("/clinical/icloud-mail/peek",{
          credentials:"same-origin",cache:"no-store",signal:controller.signal,
          headers:{"Accept":"application/json"}
        });
        if(!response.ok){
          err={status:response.status};
          if(response.status===503){
            try{err.code=(await response.json())?.detail?.code;}catch(_){}
          }
          throw err;
        }
        let data;
        try{data=await response.json();}
        catch(_){throw {clientCode:"PEEK_JSON"};}
        const code=icloudPeekContractCode(data);
        if(code)throw {clientCode:code};
        if(unreadPeek.generation!==generation)return;
        unreadPeek.data=data;unreadPeek.updatedAt=Date.now();unreadPeek.error=null;
        if(typeof window.dispatchEvent==="function")
          window.dispatchEvent(new Event("cockpit:icloud-peek-refreshed"));
      }catch(error){
        if(unreadPeek.generation!==generation)return;
        if(controller.signal.aborted)return;
        unreadPeek.error={
          status:Number.isInteger(error?.status)?error.status:0,
          code:typeof error?.code==="string"?error.code:null,
          clientCode:typeof error?.clientCode==="string"?error.clientCode:null
        };
        if(unreadPeek.error.status===401||unreadPeek.error.status===403){
          window.dispatchEvent(new Event("clinical:unauthenticated"));
          window.dispatchEvent(new Event("clinical:request-login"));
        }
      }finally{
        if(unreadPeek.generation!==generation)return;
        unreadPeek.checkedAt=Date.now();
        unreadPeek.pending=null;unreadPeek.controller=null;
        updateUnreadBadge();
        if(state.panel==="inbox"&&!popover.hidden)displayUnreadPeek();
      }
    })();
    unreadPeek.pending=pending;
    return pending;
  }
  function maybeRefreshUnread(force=false){
    // Synthetic-161 and real provider paths must never cross, including
    // during successful synthetic Clinical Key/status bootstrapping.
    if(syntheticInbox())return;
    if(unreadPeek.authorized && !document.hidden)void fetchUnreadPeek(force);
  }
  async function inbox() {
    const ticket=++state.request;
    if(!syntheticInbox()){
      // P1-01: even a fresh in-memory email snapshot is protected PHI.
      // Revalidate the existing same-origin Clinical Key before rendering it.
      // This is a lightweight /clinical/status GET, NOT an Apple IMAP fetch.
      verifiedUnreadPanelTicket=null;
      const generation=unreadPeek.generation;
      content.replaceChildren();
      content.append(text("Έλεγχος κλινικής πρόσβασης στα μη αναγνωσμένα…"));
      try{
        const status=await fetch("/clinical/status",{
          credentials:"same-origin",cache:"no-store",
          headers:{"Accept":"application/json"}
        });
        if(!activePeek("inbox",ticket) ||
           unreadPeek.generation!==generation)return;
        if(status.status===401||status.status===403){
          clearUnreadPeek(); // Includes the dot and all sender/subject memory.
          window.dispatchEvent(new Event("clinical:unauthenticated"));
          window.dispatchEvent(new Event("clinical:request-login"));
          return;
        }
        if(!status.ok){
          content.replaceChildren();
          content.append(text(
            "Δεν επιβεβαιώθηκε η κλινική σύνδεση [PEEK_AUTH_CHECK]. "+
            "Δεν εμφανίζονται αποθηκευμένα email."));
          return;
        }
        unreadPeek.authorized=true;
        verifiedUnreadPanelTicket=ticket;
        displayUnreadPeek();
        if(!unreadPeek.data && unreadPeek.error)
          void fetchUnreadPeek(true,true);
        else maybeRefreshUnread();
      }catch(_){
        if(!activePeek("inbox",ticket) ||
           unreadPeek.generation!==generation)return;
        content.replaceChildren();
        content.append(text(
          "Δεν ήταν δυνατό να επαληθευτεί η κλινική σύνδεση "+
          "[PEEK_AUTH_CHECK]. Δεν εμφανίζονται αποθηκευμένα email."));
      }
      return;
    }
    content.replaceChildren();
    content.append(text("Φόρτωση συνθετικών εισερχομένων…"));
    try{
      const response=await fetch(window.CockpitClinicalBase+"/attention",{
        credentials:"same-origin",cache:"no-store",headers:{"Accept":"application/json"}
      });
      if(!response.ok){const e=new Error("Unavailable");e.status=response.status;throw e;}
      const data=await response.json();
      if(!activePeek("inbox",ticket))return;
      if(data?.mode!=="synthetic_only"||data.live_provider_connected!==false ||
         !Array.isArray(data.items)||!Array.isArray(data.accounts) ||
         data.items.some(i=>!i||typeof i.item_id!=="string"||
           typeof i.source_title!=="string"||typeof i.review_state!=="string"))
        throw Error("Invalid synthetic source");
      content.replaceChildren();
      content.append(text("Μόνο συνθετικές εγγραφές · καμία πραγματική θυρίδα συνδεδεμένη."));
      for(const item of data.items.slice(0,4)){
        const account=data.accounts.find(a=>a.mailbox_account_id===item.mailbox_account_ref);
        content.append(action(item.source_title,
          (account?.label||"Συνθετική πηγή")+" · "+
          (item.review_state==="needs_re_review"?"Επανέλεγχος":
           item.review_state==="reviewed"?"Επισκοπήθηκε":"Δεν επισκοπήθηκε"),
          ()=>showClinicalInbox(item.item_id)));
      }
      if(!data.items.length)content.append(text("Δεν υπάρχουν συνθετικές εγγραφές στην τρέχουσα δοκιμή."));
      content.append(action("Όλα τα κλινικά εισερχόμενα",
        "Άνοιγμα πλήρους συνθετικής δοκιμής",()=>showClinicalInbox()));
    }catch(err){
      if(!activePeek("inbox",ticket))return;
      content.replaceChildren();
      content.append(text(err.status===401||err.status===403?
        "Απαιτείται κλινική σύνδεση. Δεν εμφανίζονται εισερχόμενα.":
        "Η συνθετική πηγή δεν είναι διαθέσιμη. Δεν γνωρίζουμε εάν υπάρχουν νέα στοιχεία."));
      content.append(action("Πρόσβαση","Σύνδεση με το υπάρχον Clinical Key",
        ()=>window.dispatchEvent(new Event("clinical:request-login"))));
    }
  }
  async function tasks() {
    const ticket=++state.request;
    $("doctorPopoverTitle").textContent="Υποχρεώσεις ασθενών";
    content.replaceChildren();
    content.append(text("Φόρτωση πρόσφατων ανοικτών υποχρεώσεων…"));
    const base=syntheticInbox()?window.CockpitClinicalBase:"/clinical";
    try{
      // Existing protected ClinicalPending projection owns these records.
      // First 100 active rows, sorted by updated_at; no new task store/count.
      const response=await fetch(base+"/clinical-updates?scope=real&active_only=true",{
        credentials:"same-origin",cache:"no-store",headers:{"Accept":"application/json"}
      });
      if(!response.ok){const e=new Error("Unavailable");e.status=response.status;throw e;}
      const data=await response.json();
      if(!activePeek("tasks",ticket))return;
      if(!data||!Array.isArray(data.items)||data.items.length>100 ||
         data.items.some(i=>!i||typeof i.status!=="string"||
           typeof i.patient_id!=="string"||typeof i.pending_id!=="string"||
           typeof i.description!=="string"||typeof i.patient_name!=="string"))
        throw Error("Invalid clinical source");
      const active=data.items.filter(i=>
        ["open","awaiting_patient","awaiting_result"].includes(i.status));
      active.sort((a,b)=>String(b.updated_at||"").localeCompare(String(a.updated_at||"")));
      content.replaceChildren();
      content.append(text("Πρόσφατες ανοικτές υποχρεώσεις (έως 4, από την προστατευμένη προβολή έως 100 εγγραφών)."));
      if(!active.length)content.append(text(
        "Δεν υπάρχουν ανοικτές υποχρεώσεις στην τρέχουσα προβολή."));
      for(const item of active.slice(0,4)){
        const when=/^\d{4}-\d{2}-\d{2}$/.test(item.review_date||"")?
          " · Προθεσμία "+item.review_date:"";
        content.append(action(item.patient_name,item.description+when,
          ()=>showObligations(item.pending_id)));
      }
      content.append(action("Όλες οι υποχρεώσεις",
        "Πλήρης προβολή, αλλαγή κατηγορίας και προθεσμίας",()=>showObligations()));
    }catch(err){
      if(!activePeek("tasks",ticket))return;
      content.replaceChildren();
      content.append(text(err.status===401||err.status===403?
        "Απαιτείται κλινική σύνδεση. Δεν εμφανίζονται υποχρεώσεις.":
        "Η προστατευμένη λίστα δεν είναι διαθέσιμη. Δεν θεωρούμε ότι είναι κενή."));
      content.append(action("Όλες οι υποχρεώσεις",
        "Άνοιγμα της υπάρχουσας προστατευμένης λίστας",()=>showObligations()));
    }
  }
  const notionURL=href=>{
    try{const url=new URL(href);return url.protocol==="https:"&&
      ["www.notion.so","notion.so","app.notion.com","www.notion.com"].includes(url.hostname)?href:null;
    }catch(_){return null;}
  };
  function notionAttemptId(){
    if(window.crypto?.randomUUID)return window.crypto.randomUUID();
    const bytes=new Uint8Array(16);window.crypto.getRandomValues(bytes);
    bytes[6]=(bytes[6]&15)|64;bytes[8]=(bytes[8]&63)|128;
    const hex=Array.from(bytes,b=>b.toString(16).padStart(2,"0")).join("");
    return hex.slice(0,8)+"-"+hex.slice(8,12)+"-"+hex.slice(12,16)+"-"+hex.slice(16,20)+"-"+hex.slice(20);
  }
  function notionField(label,value,multiline=false){
    const wrap=document.createElement("label");wrap.className="doctor-notion-field";
    const caption=document.createElement("span");caption.textContent=label;
    const control=document.createElement(multiline?"textarea":"input");
    if(!multiline)control.type="text";
    control.value=value||"";control.autocomplete="off";
    wrap.append(caption,control);return {wrap,control};
  }
  function notionSelect(label,options,value){
    const wrap=document.createElement("label");wrap.className="doctor-notion-field";
    const caption=document.createElement("span");caption.textContent=label;
    const control=document.createElement("select");
    for(const optionValue of ["",...options]){
      const option=document.createElement("option");option.value=optionValue;
      option.textContent=optionValue||"—";control.append(option);
    }
    control.value=value||"";wrap.append(caption,control);return {wrap,control};
  }
  function notionReview(task,suggestion={},transcript="",attemptId=null){
    if(state.panel!=="notion")return;
    const ticket=++state.request;
    content.replaceChildren();
    content.append(sectionTitle("Επιβεβαίωση για συγκεκριμένη εργασία"),
      text(task.title));
    if(transcript){
      const spoken=text("Υπαγόρευση: "+transcript);spoken.className="doctor-notion-transcript";
      content.append(spoken);
    }
    const form=document.createElement("div");form.className="doctor-notion-form";
    const done=document.createElement("label");done.className="doctor-notion-checkbox";
    const doneInput=document.createElement("input");doneInput.type="checkbox";
    doneInput.checked=suggestion.mark_done===true;
    const doneLabel=document.createElement("span");doneLabel.textContent="Σήμανση ως Done στο Notion";
    done.append(doneInput,doneLabel);
    const note=notionField("Ενημέρωση / σημείωση στην υπάρχουσα εργασία",suggestion.completion_note,true);
    note.control.maxLength=800;
    const title=notionField("Νέα εργασία παρακολούθησης (προαιρετική)",suggestion.followup_title);
    title.control.maxLength=180;
    const details=notionField("Σημειώσεις νέας εργασίας (προαιρετικές)",suggestion.followup_notes,true);
    details.control.maxLength=1000;
    const status=text("Ελέγξτε και διορθώστε τα πεδία πριν την καταχώριση. Η επιβεβαίωση ενημερώνει το Notion.");
    status.setAttribute("role","status");
    const controls=document.createElement("div");controls.className="doctor-notion-controls";
    const cancel=document.createElement("button");cancel.type="button";cancel.textContent="Άκυρο";
    cancel.addEventListener("click",()=>{if(activePeek("notion",ticket))notionTasks();});
    const confirm=document.createElement("button");confirm.type="button";
    confirm.className="doctor-notion-confirm";confirm.textContent="Επιβεβαίωση στο Notion";
    confirm.addEventListener("click",async()=>{
      const completion=note.control.value.trim(),followup=title.control.value.trim();
      if((!doneInput.checked&&!completion)||(details.control.value.trim()&&!followup)){
        status.textContent="Χρειάζεται σημείωση ή Done στην υπάρχουσα εργασία· οι σημειώσεις νέας εργασίας χρειάζονται τίτλο.";
        return;
      }
      confirm.disabled=true;cancel.disabled=true;status.textContent="Καταχώριση στο Notion…";
      try{
        const response=await fetch(clinicalURL("/clinical/notion-tasks/confirm"),{
          method:"POST",credentials:"same-origin",cache:"no-store",
          headers:{"Content-Type":"application/json","Accept":"application/json"},
          body:JSON.stringify({attempt_id:attemptId||notionAttemptId(),task_id:task.id,
            revision:task.revision,mark_done:doneInput.checked,completion_note:completion,
            followup_title:followup,followup_notes:details.control.value.trim()})
        });
        if(!activePeek("notion",ticket))return;
        if(response.status===401||response.status===403){
          window.dispatchEvent(new Event("clinical:unauthenticated"));return;
        }
        if(!response.ok)throw Error("Notion confirmation uncertain");
        const result=await response.json();
        if(result?.source!=="notion"||result.updated!==true)throw Error("Notion confirmation uncertain");
        notionTasks();
      }catch(_){
        if(activePeek("notion",ticket)){
          status.textContent="Η έκβαση δεν επιβεβαιώθηκε. Ελέγξτε την εργασία και το follow-up στο Notion πριν από νέα ενέργεια.";
          const url=notionURL(task.url);if(url)form.append(newLink("Άνοιγμα της εργασίας στο Notion",url));
          // Never repeat an uncertain external write from this review.
        }
      }
    });
    controls.append(cancel,confirm);
    form.append(done,note.wrap,title.wrap,details.wrap,status,controls);
    content.append(form);
    const url=notionURL(task.url);if(url)content.append(newLink("Άνοιγμα εργασίας στο Notion",url));
  }
  function notionCreateReview(suggestion={},transcript="",attemptId=null){
    if(state.panel!=="notion")return;
    const ticket=++state.request;
    content.replaceChildren();content.append(sectionTitle("Νέα εργασία στο Notion"));
    if(transcript){
      const spoken=text("Υπαγόρευση: "+transcript);spoken.className="doctor-notion-transcript";
      content.append(spoken);
    }
    const form=document.createElement("div");form.className="doctor-notion-form";
    const title=notionField("Τίτλος εργασίας",suggestion.title);
    title.control.maxLength=180;
    const notes=notionField("Σημειώσεις (προαιρετικές)",suggestion.notes,true);
    notes.control.maxLength=1000;
    const due=notionField("Προθεσμία (προαιρετική)",suggestion.due_date);
    due.control.type="date";
    const category=notionSelect("Κατηγορία (προαιρετική)",
      ["Ιατρείο","Cockpit","Προσωπικά"],suggestion.category);
    const priority=notionSelect("Προτεραιότητα (προαιρετική)",
      ["Υψηλή","Κανονική","Χαμηλή"],suggestion.priority);
    const status=text("Η νέα εργασία θα ξεκινήσει ως Not started. Ελέγξτε και διορθώστε πριν την καταχώριση.");
    status.setAttribute("role","status");
    const controls=document.createElement("div");controls.className="doctor-notion-controls";
    const cancel=document.createElement("button");cancel.type="button";cancel.textContent="Άκυρο";
    cancel.addEventListener("click",()=>{if(activePeek("notion",ticket))notionTasks();});
    const confirm=document.createElement("button");confirm.type="button";
    confirm.className="doctor-notion-confirm";confirm.textContent="Δημιουργία στο Notion";
    confirm.addEventListener("click",async()=>{
      const heading=title.control.value.trim(),dueDate=due.control.value.trim();
      const chosenCategory=category.control.value,chosenPriority=priority.control.value;
      if(!heading||heading.length>180||!/^\d{4}-\d{2}-\d{2}$|^$/.test(dueDate)||
         !["","Ιατρείο","Cockpit","Προσωπικά"].includes(chosenCategory)||
         !["","Υψηλή","Κανονική","Χαμηλή"].includes(chosenPriority)){
        status.textContent="Χρειάζεται τίτλος και έγκυρη προθεσμία, κατηγορία ή προτεραιότητα.";
        return;
      }
      confirm.disabled=true;cancel.disabled=true;status.textContent="Δημιουργία στο Notion…";
      try{
        const response=await fetch(clinicalURL("/clinical/notion-tasks/create"),{
          method:"POST",credentials:"same-origin",cache:"no-store",
          headers:{"Content-Type":"application/json","Accept":"application/json"},
          body:JSON.stringify({attempt_id:attemptId||notionAttemptId(),title:heading,
            notes:notes.control.value.trim(),due_date:dueDate,
            category:chosenCategory,priority:chosenPriority})
        });
        if(!activePeek("notion",ticket))return;
        if(response.status===401||response.status===403){
          window.dispatchEvent(new Event("clinical:unauthenticated"));return;
        }
        if(!response.ok)throw Error("Notion creation uncertain");
        const result=await response.json(),url=notionURL(result?.task_url);
        if(result?.source!=="notion"||result.created!==true||!url)
          throw Error("Notion creation uncertain");
        content.replaceChildren(sectionTitle("Η εργασία δημιουργήθηκε στο Notion"),
          newLink("Άνοιγμα νέας εργασίας",url),
          action("Πίσω στη λίστα","Ανανέωση από το Notion",()=>notionTasks()));
      }catch(_){
        if(activePeek("notion",ticket)){
          status.textContent="Η έκβαση δεν επιβεβαιώθηκε. Ελέγξτε πρώτα τη βάση Notion· μην επαναλάβετε τη δημιουργία από αυτή την πρόταση.";
          form.append(newLink("Έλεγχος στη βάση Notion",
            "https://app.notion.com/p/241379c6877a40fc8ab4a0efc042c887"));
        }
      }
    });
    controls.append(cancel,confirm);
    form.append(title.wrap,notes.wrap,due.wrap,category.wrap,priority.wrap,status,controls);
    content.append(form);
  }
  async function notionDictate(task=null){
    if(state.panel!=="notion")return;
    const ticket=++state.request;
    content.replaceChildren(sectionTitle(task?"Σύντομη υπαγόρευση":"Υπαγόρευση νέας εργασίας"));
    if(task)content.append(text(task.title));
    if(!window.MediaRecorder||!navigator.mediaDevices?.getUserMedia||
       !window.MediaRecorder.isTypeSupported?.("audio/webm")){
      content.append(text("Η συσκευή δεν υποστηρίζει αυτή την καταγραφή. Μπορείτε να γράψετε την ενημέρωση."));
      content.append(action("Συμπλήρωση με κείμενο","Χωρίς μικρόφωνο",
        ()=>task?notionReview(task):notionCreateReview()));
      return;
    }
    const status=text("Αναμονή άδειας μικροφώνου…");status.setAttribute("role","status");
    content.append(status);
    try{
      const stream=await navigator.mediaDevices.getUserMedia({audio:true});
      if(!activePeek("notion",ticket)){stream.getTracks().forEach(t=>t.stop());return;}
      notionStream=stream;
      const chunks=[],recorder=new window.MediaRecorder(stream,{mimeType:"audio/webm"});
      notionRecorder=recorder;const start=Date.now();
      const stop=document.createElement("button");stop.type="button";
      stop.className="doctor-notion-confirm";stop.textContent="Σταμάτημα και πρόταση";
      recorder.ondataavailable=event=>{if(event.data?.size)chunks.push(event.data);};
      recorder.onstop=async()=>{
        stream.getTracks().forEach(t=>t.stop());
        if(!activePeek("notion",ticket))return;
        stop.disabled=true;status.textContent="Μεταγραφή και προετοιμασία πρότασης…";
        const duration=Date.now()-start,blob=new Blob(chunks,{type:"audio/webm"});
        if(duration<1000||duration>60000||blob.size>2_000_000){
          status.textContent="Η καταγραφή πρέπει να είναι 1–60 δευτερόλεπτα και έως 2 MB.";return;
        }
        try{
          const response=await fetch(clinicalURL(task?
            "/clinical/notion-tasks/propose":"/clinical/notion-tasks/propose-new"),{
            method:"POST",credentials:"same-origin",cache:"no-store",
            headers:{"Content-Type":"audio/webm","Accept":"application/json",
              ...(task?{"X-Task-Id":task.id,"X-Task-Revision":task.revision}:{}),
              "X-Audio-Duration-Ms":String(duration)},body:blob
          });
          if(!activePeek("notion",ticket))return;
          if(response.status===401||response.status===403){window.dispatchEvent(new Event("clinical:unauthenticated"));return;}
          if(!response.ok)throw Error("Voice unavailable");
          const data=await response.json();
          if(typeof data?.transcript!=="string"||typeof data.attempt_id!=="string"||
             !data.suggestion||typeof data.suggestion!=="object"||
             (task&&(data.task?.id!==task.id||data.task?.revision!==task.revision)))
            throw Error("Voice response invalid");
          if(task)notionReview(task,data.suggestion,data.transcript,data.attempt_id);
          else notionCreateReview(data.suggestion,data.transcript,data.attempt_id);
        }catch(_){
          if(activePeek("notion",ticket)){
            status.textContent="Η πρόταση δεν είναι διαθέσιμη. Δεν έγινε αλλαγή στο Notion.";
            content.append(action("Συμπλήρωση με κείμενο","Χωρίς μεταγραφή",
              ()=>task?notionReview(task):notionCreateReview()));
          }
        }
      };
      stop.addEventListener("click",()=>{if(recorder.state!=="inactive")recorder.stop();});
      recorder.start();status.textContent="Η καταγραφή ξεκίνησε. Μέγιστη διάρκεια 60 δευτερόλεπτα.";
      content.append(stop);
      notionTimer=setTimeout(()=>{if(recorder.state!=="inactive")recorder.stop();},60000);
    }catch(_){
      if(activePeek("notion",ticket)){
        if(notionRecorder){notionRecorder.onstop=null;notionRecorder.ondataavailable=null;}
        clearNotionMedia();
        status.textContent="Δεν επιτράπηκε το μικρόφωνο. Μπορείτε να γράψετε την ενημέρωση.";
        content.append(action("Συμπλήρωση με κείμενο","Χωρίς μικρόφωνο",
          ()=>task?notionReview(task):notionCreateReview()));
      }
    }
  }
  async function notionTasks(){
    const ticket=++state.request;
    content.replaceChildren();content.append(text("Φόρτωση από τις εργασίες μου στο Notion…"));
    try{
      const response=await fetch(clinicalURL("/clinical/notion-tasks/peek"),{
        credentials:"same-origin",cache:"no-store",headers:{"Accept":"application/json"}
      });
      if(!response.ok){const e=Error("Notion unavailable");e.status=response.status;throw e;}
      const data=await response.json();
      if(!activePeek("notion",ticket))return;
      if(data?.source!=="notion"||!Array.isArray(data.tasks)||data.tasks.length>4||
         !notionURL(data.database_url)||data.tasks.some(t=>!t||typeof t.id!=="string"||
           typeof t.title!=="string"||typeof t.revision!=="string"||!notionURL(t.url)||
           !["Not started","In progress"].includes(t.status)))throw Error("Invalid Notion source");
      content.replaceChildren();
      const topActions=document.createElement("div");
      topActions.className="doctor-notion-controls doctor-notion-top-actions";
      topActions.append(action("Νέα εργασία","Δημιουργία στο Notion",()=>notionCreateReview()));
      if(data.voice_available===true)
        topActions.append(action("Υπαγόρευση νέας","Πείτε τη νέα εργασία",()=>notionDictate()));
      content.append(topActions);
      content.append(text("Έως 4 ενεργές εργασίες από την υπάρχουσα προβολή «Εκκρεμείς»."));
      if(!data.tasks.length)content.append(text("Δεν υπάρχουν ενεργές εργασίες στην τρέχουσα προβολή."));
      for(const task of data.tasks){
        const item=document.createElement("div");item.className="doctor-notion-entry";
        item.append(newLink(task.title,task.url,
          (task.status==="In progress"?"Σε εξέλιξη":"Δεν ξεκίνησε")+
          (task.due?" · Προθεσμία "+task.due:"")));
        const controls=document.createElement("div");controls.className="doctor-notion-controls";
        controls.append(action("Ενημέρωση","Σύντομη επιβεβαίωση",()=>notionReview(task)));
        if(data.voice_available===true)controls.append(action("Υπαγόρευση","Έως 60 δευτερόλεπτα",()=>notionDictate(task)));
        item.append(controls);content.append(item);
      }
      content.append(newLink("Όλες οι εργασίες μου",data.database_url,"Άνοιγμα της βάσης στο Notion"));
    }catch(error){
      if(!activePeek("notion",ticket))return;
      content.replaceChildren();content.append(text(error.status===401||error.status===403?
        "Απαιτείται κλινική σύνδεση. Δεν εμφανίζονται εργασίες.":
        "Η σύνδεση Notion δεν είναι διαθέσιμη. Δεν θεωρούμε ότι η λίστα είναι κενή."));
    }
  }
  const links=[
    {name:"Εισερχόμενα ασθενών",hint:"Όλα τα iCloud email · μόνο ανάγνωση",href:"/static/cockpit/#icloud-mail"},
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
      list.append(action("Περιστατικά ιατρείου", "Χειρουργεία, τροχαία και RF",
        ()=>showLibrary("cases")));
    };
    input.addEventListener("input",draw);draw();input.focus();
  }
  function showIcloudInbox(itemId=null){
    // The synthetic Clinical Inbox can never navigate into real iCloud data,
    // including through the global #icloud-mail bookmark/shortcut.
    if(syntheticInbox()){openPanel("inbox",buttons.inbox);return;}
    if(!window.CockpitIcloudInbox)return;
    if(typeof window.CockpitStoredChart?.canLeave==="function" &&
       !window.CockpitStoredChart.canLeave())return;
    close();window.CockpitObligations?.close();
    window.CockpitClinicalInbox?.close();
    $("doctorClinicalInboxView").hidden=true;
    $("doctorObligationsView").hidden=true;
    $("doctorLibraryView").hidden=true;
    $("doctorMainView").hidden=true;
    $("doctorIcloudInboxView").hidden=false;
    // An isolated synthetic Chromium harness uses about:blank (null origin),
    // where History.replaceState with an application path is forbidden.
    if(window.location?.origin && window.location.origin!=="null" &&
       window.location.pathname?.startsWith("/static/cockpit/") &&
       window.location.hash!=="#icloud-mail")
      window.history?.replaceState?.(null,"","/static/cockpit/#icloud-mail");
    window.CockpitIcloudInbox.open(itemId);
  }
  function showClinicalInbox(itemId=null){
    // Normal live Cockpit retains the honest unconnected source placeholder.
    if(window.CockpitSynthetic!==true ||
       window.CockpitClinicalBase!=="/clinical/synthetic-161" ||
       !window.CockpitClinicalInbox){
      openPanel("inbox",buttons.inbox);return;
    }
    if(typeof window.CockpitStoredChart?.canLeave==="function" &&
       !window.CockpitStoredChart.canLeave())return;
    close();window.CockpitObligations?.close();
    window.CockpitIcloudInbox?.close();
    $("doctorIcloudInboxView").hidden=true;
    $("doctorObligationsView").hidden=true;
    $("doctorLibraryView").hidden=true;
    $("doctorMainView").hidden=true;
    $("doctorClinicalInboxView").hidden=false;
    window.CockpitClinicalInbox.open(itemId);
  }
  function showObligations(pendingId=null) {
    // Preserve an in-progress clinical visit draft. Reuse the chart owner's
    // existing guard and its patient-visible reason, never silently hide it.
    if (typeof window.CockpitStoredChart?.canLeave==="function"
        && !window.CockpitStoredChart.canLeave()) return;
    close();
    window.CockpitClinicalInbox?.close();
    $("doctorClinicalInboxView").hidden=true;
    window.CockpitIcloudInbox?.close();
    $("doctorIcloudInboxView").hidden=true;
    const screen=$("doctorObligationsView");
    if(!screen)return;
    $("doctorMainView").hidden=true;
    $("doctorLibraryView").hidden=true;
    screen.hidden=false;
    window.CockpitObligations?.open((id,pendingId)=>{
      // Reuse the exact existing chart+pending owner. Opening a listed task
      // cannot grant a capability or write a clinical record by itself.
      showHome();
      const chart=window.CockpitStoredChart;
      if(typeof chart?.openPending==="function")chart.openPending(id,pendingId);
      else if(window.CockpitSynthetic===true && typeof chart?.open==="function")chart.open(id);
    },pendingId);
    $("doctorObligationsSearch").focus();
  }
  function showLibrary(category="cases") {
    close();
    window.CockpitClinicalInbox?.close();
    $("doctorClinicalInboxView").hidden=true;
    window.CockpitIcloudInbox?.close();
    $("doctorIcloudInboxView").hidden=true;
    window.CockpitObligations?.close();
    $("doctorObligationsView").hidden=true;
    $("doctorMainView").hidden=true;
    $("doctorLibraryView").hidden=false;
    const panels=document.querySelectorAll("[data-doctor-library-panel]");
    panels.forEach(p=>p.hidden=p.dataset.doctorLibraryPanel!==category);
    document.querySelectorAll("[data-doctor-library-tab]").forEach(b=>{
      const selected=b.dataset.doctorLibraryTab===category;
      b.classList.toggle("selected",selected);b.setAttribute("aria-pressed",String(selected));
    });
    if(category === "cases") window.CockpitOfficeCases?.open?.();
    $("doctorBackHome").focus();
  }
  function showHome() {
    close();
    window.CockpitClinicalInbox?.close();
    $("doctorClinicalInboxView").hidden=true;
    window.CockpitIcloudInbox?.close();
    $("doctorIcloudInboxView").hidden=true;
    window.CockpitObligations?.close();
    $("doctorObligationsView").hidden=true;
    $("doctorLibraryView").hidden=true;
    $("doctorMainView").hidden=false;
    if(window.location?.hash==="#icloud-mail"){
      if(window.location.origin!=="null" &&
         window.location.pathname?.startsWith("/static/cockpit/"))
        window.history?.replaceState?.(null,"","/static/cockpit/");
      else window.location.hash="";
    }
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
      () => showLibrary("cases")));
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
      showObligations));
  }

    buttons.day.addEventListener("click",()=>openPanel("day",buttons.day));
  buttons.calls.addEventListener("click",()=>openPanel("calls",buttons.calls));
  buttons.inbox.addEventListener("click",()=>openPanel("inbox",buttons.inbox));
  $("doctorClinicalInboxBack").addEventListener("click",showHome);
  $("doctorIcloudInboxBack").addEventListener("click",showHome);
  window.addEventListener("cockpit:clinical-updates-changed",()=>{if(state.panel==="inbox")inbox();else if(state.panel==="tasks")tasks();});
  window.addEventListener("clinical:unauthenticated",()=>{showHome();});
  buttons.tasks.addEventListener("click",()=>openPanel("tasks",buttons.tasks));
  buttons.notion.addEventListener("click",()=>openPanel("notion",buttons.notion));
  $("doctorObligationsBack").addEventListener("click",showHome);
  buttons.actions.addEventListener("click",()=>openPanel("actions",buttons.actions));
  $("doctorPopoverClose").addEventListener("click",close);
  if (typeof window !== "undefined" && typeof window.addEventListener === "function") window.addEventListener("clinical:authenticated", () => {
    if(state.panel === "day") loadDay();
    if(state.panel === "tasks") tasks();
    if(state.panel === "inbox" && syntheticInbox()) inbox();
    if(state.panel === "calls") callPeek();
    if(state.panel === "notion") notionTasks();
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
      $("doctorTasksBadge").textContent = "";
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
    // The top icon now opens PATIENT obligations; this surgical count
    // is not a patient task count. Keep surgery numbers in Home/library only.
    $("doctorTasksBadge").hidden = true;
    $("doctorTasksBadge").textContent = "";
    attention();
    if (state.panel === "tasks") tasks();
  }
    window.addEventListener("cockpit:surgery-counts",e=>updateSurgeryCounts(e.detail));
  // The badge is an existence dot only, never an invented unread total.
  updateUnreadBadge();
  buttons.day.firstChild.textContent=dateLabel.format(new Date())+" ";
  attention();
  updateSurgeryCounts(window.CockpitSurgerySummary);
  // One protected bootstrap check; background email requests only after
  // existing clinician auth. A five-minute interval operates only in a
  // visible Cockpit tab. Hidden or logged-out tabs make no provider requests.
  if(typeof window.setInterval==="function")
    window.setInterval(()=>maybeRefreshUnread(),ICLOUD_PEEK_TTL_MS);
  document.addEventListener("visibilitychange",()=>maybeRefreshUnread());
  window.addEventListener("focus",()=>maybeRefreshUnread());
  window.addEventListener("clinical:authenticated",()=>{
    clearUnreadPeek();unreadPeek.authorized=true;
    if(state.panel==="inbox"&&!popover.hidden&&!syntheticInbox())void inbox();
    maybeRefreshUnread(true);
  });
  window.addEventListener("clinical:unauthenticated",clearUnreadPeek);
  const bootstrapGeneration=unreadPeek.generation;
  fetch("/clinical/status",{credentials:"same-origin",cache:"no-store"})
    .then(r=>{
      if(r.ok&&bootstrapGeneration===unreadPeek.generation){
        unreadPeek.authorized=true;maybeRefreshUnread(true);
      }
    }).catch(()=>{});
  // One stable deep link shared by all Cockpit modules and browser bookmarks.
  function openIcloudRoute(){
    if(window.location?.hash==="#icloud-mail" && $("doctorIcloudInboxView").hidden)
      showIcloudInbox();
  }
  window.addEventListener("hashchange",openIcloudRoute);
  openIcloudRoute();
  loadRfAttention();
})();
