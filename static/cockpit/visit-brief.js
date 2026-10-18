/* Bounded Visit Brief. No automatic patient GET from UID/name; no browser PHI cache. */
(() => {
  "use strict";
  const $=id=>document.getElementById(id);
  const state={
    ticket:0, view:null, ready:false, revision:0, linked:false,
    pending:null, chosen:null, patientId:null, inFlight:false, chartMode:false, replaceRequired:false
  };
  const endpoint=(window.CockpitClinicalBase||"/clinical")+"/calendar/visit-brief/";

  async function request(action,body){
    const response=await fetch(endpoint+action,{
      method:"POST",credentials:"same-origin",cache:"no-store",
      headers:{"Content-Type":"application/json","Accept":"application/json"},
      body:JSON.stringify(body)
    });
    if(!response.ok){
      const err=new Error("HTTP "+response.status);err.status=response.status;throw err;
    }
    return response.json();
  }
  function evidence(id,values,empty){
    const el=$(id);el.replaceChildren();
    if(!values.length){
      const p=document.createElement("p");p.className="clinical-muted";p.textContent=empty;
      el.append(p);return;
    }
    const list=document.createElement("ul");
    values.forEach(value=>{
      const sourceLabel={heidi_today:"Heidi",gesy_today:"ΓεΣΥ σήμερα",
        gesy_previous:"ΓεΣΥ προηγούμενο",dia_summary:"Σύνοψη Dia",clinician_edit:"Ιατρική καταγραφή"};
      let printed="";
      if(value&&typeof value==="object"){
        const p=(value.source_roles||[]).slice(0,3).map(x=>sourceLabel[x]||"Μη προσδιορισμένη πηγή");
        const provenance=p.length?" · Πηγή: "+p.join(", "):"";
        const uncertain=value.certainty&&value.certainty!=="certain"?
          " · Βεβαιότητα: "+value.certainty:"";
        const unverified=value.verification&&value.verification!=="clinician_confirmed"?
          " · Καταγραφή από πηγή":"";
        printed=String(value.text||"")+provenance+uncertain+unverified;
      }else printed=String(value||"");
      if(!printed.trim())return;
      const li=document.createElement("li");li.textContent=printed;list.append(li);
    });
    el.append(list);
  }
  function clearProtected(){
    state.patientId=null;
    $("visitBriefClinical").hidden=true;
    $("visitBriefOpenChart").hidden=true;
    document.getElementById("visitBriefOpenLatest")?.remove?.();
    if($('visitBriefPreviousHeading'))$('visitBriefPreviousHeading').textContent='Προηγούμενη επιβεβαιωμένη επίσκεψη';
    if($('visitBriefLinkNote'))$('visitBriefLinkNote').hidden=false;
    $('visitBriefChecksTitle').hidden=false;$('visitBriefChecks').hidden=false;
    $("visitBriefUnlink").textContent="Αποσύνδεση ραντεβού";
    $("visitBriefUnlink").dataset.armed="";
    $("visitBriefUnlink").hidden=true;
    for(const id of ["visitBriefPrevious","visitBriefPlan","visitBriefPending","visitBriefChecks"])
      $(id).replaceChildren();
  }
  function invalidate(){
    state.ticket++;
    state.ready=false;state.view=null;state.pending=null;state.chosen=null;
    state.revision=0;state.linked=false;state.inFlight=false;state.chartMode=false;state.replaceRequired=false;
    clearProtected();
    // Hiding the overlay alone must not leave patient names or confirmation
    // identifiers in the DOM after close, logout, re-login or context switch.
    for(const id of ["visitBriefName","visitBriefWhen","visitBriefReason",
      "visitBriefFreshness","visitBriefIdentity","visitBookingContext",
      "visitBookingConfirmStatus"])$(id).textContent="";
    $("visitBriefChecksTitle").textContent="Για το ραντεβού";
    $("visitBookingIndependentCheck").checked=false;
    $("visitBookingCorrection").checked=false;
    $("visitBookingCorrectionWrap").hidden=true;
    $("visitConfirmBooking").disabled=false;
    $("visitBriefChoosePatient").hidden=true;
    $("visitBriefChoosePatient").disabled=true;
    $("visitBookingConfirmation").hidden=true;
  }
  function closed(){
    // The overlay owner hides it/restores focus; purge pending/chosen identities.
    invalidate();
  }
  function identityDenied(){
    invalidate();
    $("visitBriefIdentity").textContent="Απαιτείται νέα προστατευμένη σύνδεση προσωπικού.";
    if(typeof Event==="function"&&typeof window.dispatchEvent==="function"){
      window.dispatchEvent(new Event("clinical:unauthenticated"));
      window.dispatchEvent(new Event("clinical:request-login"));
    }
  }
  async function open(view){
    if(!view||!view.uid)return;
    if(window.CockpitStoredChart?.canLeave&&!window.CockpitStoredChart.canLeave())return;
    invalidate();
    const ticket=state.ticket;
    state.view={...view};
    $("visitBriefName").textContent=view.name;
    $("visitBriefWhen").textContent=view.time;
    $("visitBriefReason").textContent=view.reason;
    $("visitBriefFreshness").textContent=view.freshness;
    $("visitBriefIdentity").textContent="Επαλήθευση ραντεβού από τη Γραμματεία…";
    $("visitBriefChoosePatient").hidden=false;
    $("visitBriefChoosePatient").disabled=true;
    $("visitBriefOverlay").hidden=false;
    $("visitBriefClose").focus();
    try{
      if(window.CockpitSynthetic){
        const response=await fetch(window.CockpitClinicalBase+"/open-booking", {
          method:"POST",credentials:"same-origin",cache:"no-store",
          headers:{"Content-Type":"application/json"},body:JSON.stringify({booking_uid:view.uid})
        });
        if(!response.ok){const e=new Error("Read failed");e.status=response.status;throw e;}
        const data=await response.json();
        if(ticket!==state.ticket||$("visitBriefOverlay").hidden)return;
        state.chosen={name:data.patient_name};
        $("visitBriefFreshness").textContent="";
        renderConfirmed(data);
        return;
      }
      if(window.CockpitPatientEntry){
        const normal=await fetch('/clinical/status',{credentials:'same-origin',cache:'no-store'});
        if(ticket!==state.ticket||$('visitBriefOverlay').hidden)return;
        if(normal.ok){
          state.chartMode=true;
          const data=await request('chart-read',{booking_uid:view.uid,target_date:view.date||null});
          if(ticket!==state.ticket||$('visitBriefOverlay').hidden)return;
          renderChart(data);return;
        }
      }
      const data=await request("read",{booking_uid:view.uid});
      if(ticket!==state.ticket||$("visitBriefOverlay").hidden)return;
      if(data.mode!=="booking_only"||data.booking?.booking_uid!==view.uid)
        throw new Error("Incorrect booking context");
      state.revision=data.revision||0;
      state.linked=Boolean(data.linked);
      state.ready=data.reason!=="disabled";
      $("visitBriefName").textContent=data.booking.patient_display_name||"Ραντεβού";
      $("visitBriefReason").textContent=data.booking.reason||"";
      state.view.name=data.booking.patient_display_name||view.name;
      state.view.reason=data.booking.reason||view.reason;
      state.view.observation=data.booking.observation||null;
      // Fresh Reception owns any reschedule; never reuse stale Home slot time.
      const started=new Date(data.booking.start_at);
      if(Number.isFinite(started.getTime())){
        const local=new Intl.DateTimeFormat("el-CY",{timeZone:"Asia/Nicosia",
          day:"2-digit",month:"2-digit",year:"numeric",hour:"2-digit",minute:"2-digit",hour12:false}).format(started);
        state.view.time=local;
        $("visitBriefWhen").textContent=local;
      }
      $("visitBriefIdentity").textContent=data.reason==="disabled"?
        "Η επιβεβαίωση σύνδεσης ραντεβού με κλινικό φάκελο δεν έχει ενεργοποιηθεί.":
        data.reason==="needs_reconfirmation"?
        "Η προηγούμενη σύνδεση απαιτεί νέα ανεξάρτητη επιβεβαίωση. Κανένα ιστορικό δεν φορτώθηκε.":
        "Δεν υπάρχει επιβεβαιωμένη σύνδεση με φάκελο. Το όνομα του ραντεβού δεν αρκεί.";
      $("visitBriefChoosePatient").hidden=!state.ready;
      $("visitBriefChoosePatient").disabled=!state.ready;
      $("visitBriefUnlink").hidden=!state.linked||window.CockpitStaffRole!=="clinician";
    }catch(err){
      if(ticket!==state.ticket||$("visitBriefOverlay").hidden)return;
      clearProtected();
      if(state.chartMode){
        $('visitBriefChoosePatient').hidden=true;
        $('visitBriefIdentity').textContent=err.status===401?'Συνδέσου από την Πρόσβαση.':'Δεν ανανεώθηκε το ραντεβού. Μπορείς να ανοίξεις την καρτέλα με αναζήτηση.';
        if(err.status===401)identityDenied();return;
      }
      $("visitBriefChoosePatient").hidden=true;
      if((err.status===401||err.status===403)&&window.CockpitPatientEntry&&!window.CockpitSynthetic){
        // Optional staff Brief denial does not revoke an independently valid registry session.
        let registry;
        try{registry=await fetch('/clinical/status',{credentials:'same-origin',cache:'no-store'});}catch(e){}
        if(ticket!==state.ticket||$('visitBriefOverlay').hidden)return;
        if(registry?.ok){
          state.ready=false;state.linked=false;
          $('visitBriefIdentity').textContent='Η προεπισκόπηση δεν είναι διαθέσιμη. Μπορείς να αναζητήσεις ή να δημιουργήσεις καρτέλα.';
          return;
        }
      }
      if(err.status===401||err.status===403)identityDenied();
      else $("visitBriefIdentity").textContent=
        "Δεν επιβεβαιώνεται πλήρης πρόσφατη πηγή. Δεν ανοίγει κλινικό ιστορικό.";
    }
  }
  function selectionContext(){
    if(!state.view||!state.ready||$("visitBriefOverlay").hidden)return null;
    // The caller uses this immutable per-opening scope, not a reusable session grant.
    return {uid:state.view.uid, name:state.view.name,
      time:state.view.time,reason:state.view.reason,
      revision:state.revision, linked:state.linked,
      observation:state.view.observation};
  }
  function activateSelection(context){
    invalidate();
    if(!context||!context.uid)return;
    state.pending={...context};
  }
  function bookingVerification(){
    if(!state.pending)return null;
    return {booking_uid:state.pending.uid,
            observed_booking:state.pending.observation};
  }
  function selectPatient(row){
    const p=state.pending;
    if(!p||!row?.patientId)return;
    state.chosen={id:row.patientId,name:row.name};
    $("visitBookingConfirmation").hidden=false;
    $("visitBookingIndependentCheck").checked=false;
    $("visitBookingCorrection").checked=false;
    // Revoked links remain versioned: revision>0 is not a fresh/never-linked UID.
    // The explicit correction checkbox must also be available after unlink.
    $("visitBookingCorrectionWrap").hidden=!(p.linked||p.revision>0);
    $("visitBookingContext").textContent=
      "Ραντεβού "+p.time+" · "+p.name+" — επιλεγμένος φάκελος: "+row.name+
      ". Επιβεβαίωσε μόνο αφού ελέγξεις ανεξάρτητα τα στοιχεία ταυτότητας.";
    $("visitBookingConfirmStatus").textContent="";
  }
  function renderConfirmed(response){
    const b=response.brief,booking=response.booking;
    clearProtected();
    state.patientId=b.patient_id;
    state.revision=response.revision;state.linked=true;
    $("visitBriefName").textContent=booking.patient_display_name||"Ραντεβού";
    $("visitBriefReason").textContent=booking.reason||"";
    $("visitBriefIdentity").textContent=
      "Προστατευμένος φάκελος: "+(state.chosen?.name||"επιβεβαιωμένος ασθενής")+
      ". Ανεξάρτητα επιβεβαιωμένος μόνο για αυτό το άνοιγμα — χωρίς μόνιμο δικαίωμα αυτόματης προβολής.";
    if(window.CockpitSynthetic) $("visitBriefIdentity").textContent=
      "";
    const visit=b.previous_visit;
    evidence("visitBriefPrevious",visit?
      [visit.date+" · "+(visit.reason||"Υπογεγραμμένη επίσκεψη")]
        .concat(visit.findings||[]):[],
      "Δεν επιβεβαιώνεται προηγούμενη κλινική καταγραφή πριν από αυτό το ραντεβού.");
    const plan=b.prior_plan;
    const acts={prescribed:"Συνταγογραφήθηκε",continued:"Συνεχίζεται",stopped:"Διακόπηκε"};
    const medications=plan?(plan.medications||[]).map(m=>({
      text:(acts[m.action]||m.action)+": "+m.name+(m.dose?" · "+m.dose:""),
      source_roles:m.source_roles||[],certainty:m.certainty,
      verification:m.verification
    })):[];
    evidence("visitBriefPlan",plan?(plan.decisions||[]).concat(medications):[],
      "Δεν υπάρχει τεκμηριωμένη προηγούμενη απόφαση ή αγωγή.");
    if(plan?.next_contact){
      const note=document.createElement("p");note.className="clinical-muted";
      note.textContent="Επόμενη επαφή στο προηγούμενο σχέδιο: "+
        (plan.next_contact.when||"χωρίς ημερομηνία")+" · "+plan.next_contact.purpose;
      $("visitBriefPlan").append(note);
    }
    if(b.historical_plan){
      const note=document.createElement("p");note.className="clinical-muted";
      note.textContent="Προηγούμενο σχέδιο — χρόνος/συνάφεια μη επιβεβαιωμένα: "+
        (b.historical_plan.when||"χωρίς ημερομηνία")+" · "+b.historical_plan.purpose;
      $("visitBriefPlan").append(note);
    }
    if(window.CockpitSynthetic){
      window.CockpitStoredChart.renderPending($("visitBriefPending"), b.patient_id, b.pending||[]);
    }else{
      const labels={open:"Σε εκκρεμότητα",awaiting_patient:"Αναμένεται από τον ασθενή",awaiting_result:"Αναμένεται αποτέλεσμα"};
      evidence("visitBriefPending",(b.pending||[]).map(p=>
        p.description+" · "+(labels[p.status]||"Σε εκκρεμότητα")+(p.review_date?" · "+p.review_date:"")),
        "Δεν υπάρχουν ανοικτές εκκρεμότητες.");
    }
    $("visitBriefChecksTitle").textContent=b.check_heading||"Για το ραντεβού";
    evidence("visitBriefChecks",(b.appointment_checks||[]).map(c=>c.text),
      "Δεν έχει τεκμηριωθεί συγκεκριμένος έλεγχος για αυτό το ραντεβού.");
    if(window.CockpitSynthetic){
      $("visitBriefChecksTitle").hidden=!(b.appointment_checks||[]).length;
      $("visitBriefChecks").hidden=!(b.appointment_checks||[]).length;
    }
    $("visitBriefClinical").hidden=false;
    $("visitBriefChoosePatient").hidden=true;
    $("visitBriefOpenChart").hidden=window.CockpitStaffRole!=="clinician"&&!window.CockpitSynthetic;
    $("visitBriefUnlink").hidden=window.CockpitSynthetic||window.CockpitStaffRole!=="clinician";
  }
  async function confirmSelection(){
    const p=state.pending,chosen=state.chosen;
    if(!p||!chosen||state.inFlight)return;
    if(!/^[0-9a-f]{64}$/.test(p.observation||"")){
      $("visitBookingConfirmStatus").textContent="Το ραντεβού δεν έχει έγκυρη νέα επαλήθευση. Άνοιξέ το ξανά.";
      return;
    }
    if(!$("visitBookingIndependentCheck").checked){
      $("visitBookingConfirmStatus").textContent=
        "Χρειάζεται ανεξάρτητη επαλήθευση του συγκεκριμένου προσώπου.";
      return;
    }
    if(!p.linked&&p.revision>0&&!$("visitBookingCorrection").checked){
      $("visitBookingConfirmStatus").textContent=
        "Μετά την αποσύνδεση απαιτείται ρητή επιβεβαίωση νέας σύνδεσης.";
      return;
    }
    const ticket=++state.ticket;
    state.inFlight=true;$("visitConfirmBooking").disabled=true;
    $("visitBookingConfirmStatus").textContent="Επιβεβαίωση σύνδεσης…";
    try{
      const data=await request("confirm",{
        booking_uid:p.uid,patient_id:chosen.id,expected_revision:p.revision,
        independently_verified:true,evidence_type:"protected_chart_identifiers",
        observed_booking:p.observation,
        replace_existing:$("visitBookingCorrection").checked
      });
      if(ticket!==state.ticket||state.pending!==p||state.chosen!==chosen)return;
      if(data.mode!=="confirmed_once"||data.brief?.patient_id!==chosen.id||
         data.booking?.booking_uid!==p.uid)throw Error("Invalid confirmation result");
      state.pending=null;
      state.view={uid:p.uid,name:p.name,reason:p.reason,time:p.time,
        observation:p.observation};
      $("visitBookingConfirmation").hidden=true;
      $("visitBriefWhen").textContent=p.time;
      $("visitBriefFreshness").textContent=
        "Επιβεβαιωμένη πηγή ραντεβού · signed clinical source";
      renderConfirmed(data);
      $("visitBriefOverlay").hidden=false;
      $("visitBriefClose").focus();
    }catch(err){
      if(ticket!==state.ticket)return;
      clearProtected();
      if(err.status===401||err.status===403)identityDenied();
      else $("visitBookingConfirmStatus").textContent=err.status===409?
        "Άλλαξε η σύνδεση ή το ραντεβού. Επέστρεψε στην αρχική και άνοιξέ το ξανά.":
        "Δεν ολοκληρώθηκε η επιβεβαίωση. Δεν εμφανίζεται κλινικό ιστορικό.";
    }finally{
      if(ticket===state.ticket){
        state.inFlight=false;$("visitConfirmBooking").disabled=false;
      }
    }
  }
  async function revoke(){
    if(!state.linked||!state.view&&!state.patientId)return;
    const button=$("visitBriefUnlink");
    if(button.dataset.armed!=="yes"){
      button.dataset.armed="yes";button.textContent="Πάτησε ξανά για αποσύνδεση";return;
    }
    const uid=state.view?.uid||state.pending?.uid;
    if(!uid)return;
    const revision=state.revision,ticket=++state.ticket;
    clearProtected();
    // After revocation, the previously selected protected patient must not
    // remain as a usable pending confirmation or hidden context.
    state.pending=null;state.chosen=null;
    $("visitBookingContext").textContent="";
    $("visitBookingIndependentCheck").checked=false;
    $("visitBookingCorrection").checked=false;
    $("visitBookingConfirmation").hidden=true;
    $("visitBriefChoosePatient").hidden=true;
    $("visitBriefIdentity").textContent="Αποσύνδεση…";
    try{
      if(state.chartMode){await request('chart-unlink',{booking_uid:uid,expected_revision:revision,request_id:crypto.randomUUID()});if(ticket!==state.ticket)return;state.linked=false;state.revision=revision+1;state.replaceRequired=true;$('visitBriefPatientEntry').textContent='Σύνδεση καρτέλας / Νέος ασθενής';$('visitBriefIdentity').textContent='Η σύνδεση αφαιρέθηκε.';return;}
      await request("unlink",{booking_uid:uid,expected_revision:revision,explicit_unlink:true});
      if(ticket!==state.ticket)return;
      state.linked=false;
      $("visitBriefIdentity").textContent="Η σύνδεση ανακλήθηκε. Για νέο φάκελο απαιτείται ανεξάρτητη επιβεβαίωση.";
    }catch(err){
      if(ticket!==state.ticket)return;
      if(err.status===401||err.status===403)identityDenied();
      else $("visitBriefIdentity").textContent="Η αποσύνδεση δεν επιβεβαιώθηκε. Άνοιξε ξανά το ραντεβού.";
    }
  }

  $("visitConfirmBooking").addEventListener("click",confirmSelection);
  $("visitBriefUnlink").addEventListener("click",revoke);
  function renderChart(data){
    clearProtected();state.revision=data.revision;state.linked=Boolean(data.linked);state.replaceRequired=Boolean(data.replace_required);
    state.ready=true;state.view.observation=data.booking.observation;state.view.uid=data.booking.booking_uid;
    $('visitBriefWhen').textContent=new Intl.DateTimeFormat('el-CY',{timeZone:'Asia/Nicosia',dateStyle:'short',timeStyle:'short'}).format(new Date(data.booking.start_at));
    $('visitBriefName').textContent=data.booking.patient_display_name||'Ραντεβού';
    state.view.name=data.booking.patient_display_name;state.view.date=new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Nicosia',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(data.booking.start_at));
    $('visitBriefReason').textContent=data.booking.reason||'';
    $('visitBriefChoosePatient').hidden=true;
    const entry=$('visitBriefPatientEntry');entry.hidden=false;entry.className=data.mode==='chart_linked'?'clinical-secondary':'clinical-primary';entry.textContent=data.mode==='chart_linked'?'Αλλαγή συνδεδεμένης καρτέλας':'Σύνδεση καρτέλας / Νέος ασθενής';
    if(data.mode!=='chart_linked'){$('visitBriefIdentity').textContent='Επίλεξε την καρτέλα μία φορά για αυτό το ραντεβού.';return;}
    state.patientId=data.patient_id;$('visitBriefIdentity').textContent='Συνδεδεμένη καρτέλα · '+data.patient_name;
    $('visitBriefClinical').hidden=false;$('visitBriefOpenChart').hidden=false;$('visitBriefUnlink').hidden=false;
    if($('visitBriefLinkNote'))$('visitBriefLinkNote').hidden=true;
    if($('visitBriefPreviousHeading'))$('visitBriefPreviousHeading').textContent='Τελευταία αποθηκευμένη επίσκεψη';
    const latest=data.latest_visit;
    evidence('visitBriefPrevious',latest?[latest.date+' · '+latest.reason].concat(latest.findings||[]):[],'Δεν έχει αποθηκευτεί επίσκεψη.');
    evidence('visitBriefPlan',latest?.decisions||[],'Δεν έχει καταγραφεί προηγούμενο σχέδιο.');
    $('visitBriefPending').replaceChildren();$('visitBriefChecks').replaceChildren();$('visitBriefChecksTitle').hidden=true;$('visitBriefChecks').hidden=true;
    const pending=data.pending||[];if(!pending.length)evidence('visitBriefPending',[],'Δεν υπάρχουν ανοικτές υποχρεώσεις.');
    for(const item of pending){const b=document.createElement('button');b.type='button';b.className='clinical-secondary';b.textContent=item.description;b.onclick=()=>{if(!window.CockpitStoredChart.canLeave())return;const id=state.patientId;$('visitBriefClose').click();window.CockpitStoredChart.openPending(id,item.pending_id);};$('visitBriefPending').append(b);}
    if(latest){const open=document.createElement('button');open.id='visitBriefOpenLatest';open.type='button';open.className='clinical-secondary';open.textContent='Άνοιγμα τελευταίας επίσκεψης';open.onclick=async()=>{
      if(!window.CockpitStoredChart.canLeave())return;const id=state.patientId,visit=latest.source_encounter_id;$('visitBriefClose').click();
      const opened=await window.CockpitStoredChart.open(id);if(opened!==id)return;const node=document.getElementById('visit-'+visit);if(node){node.hidden=false;node.open=true;node.scrollIntoView({behavior:'smooth'});}
    };$('visitBriefClinical').append(open);}
  }
  const manualEntry=document.getElementById('visitBriefPatientEntry');
  if(manualEntry)manualEntry.addEventListener('click',()=>{
    const name=String(state.view?.name||$('visitBriefName').textContent||'');
    if(window.CockpitStoredChart?.canLeave&&!window.CockpitStoredChart.canLeave())return;
    const context=state.chartMode&&state.ready&&state.view?.observation?{...state.view,revision:state.revision,linked:state.linked,replace:state.replaceRequired}:null;
    $('visitBriefClose').click();
    const token=state.ticket;
    const requestId=crypto.randomUUID();
    window.CockpitPatientEntry?.open({name,...(context?{onChoose:async(id,valid)=>{
      if(!valid()||token!==state.ticket)throw Error('Closed selection');
      await request('chart-select',{booking_uid:context.uid,target_date:context.date||null,patient_id:id,expected_revision:context.revision,
        observed_booking:context.observation,replace_existing:context.replace||context.linked,request_id:requestId});
      if(!valid()||token!==state.ticket)throw Error('Closed selection');
    }}:{})});
  });
  $("visitBriefOpenChart").addEventListener("click",()=>{
    if(!state.patientId)return;
    if(window.CockpitStoredChart){
      const patientId=state.patientId;
      $("visitBriefClose").click();
      window.CockpitStoredChart?.open(patientId);
      return;
    }
    $("visitBriefClose").click();
    $("visitWorkspaceHome").hidden=true;
    $("visitWorkspacePatient").hidden=false;
  });
  window.addEventListener("clinical:authenticated",invalidate);
  window.addEventListener("clinical:unauthenticated",()=>{
    invalidate();$("visitBriefOverlay").hidden=true;
  });
  async function bookingFromURL(){
    const ticket=state.ticket,url=new URL(location.href),uid=url.searchParams.get('booking');if(!uid||uid.length>160)return;
    try{const r=await fetch('/clinical/status',{credentials:'same-origin',cache:'no-store'});if(ticket!==state.ticket)return;if(!r.ok){if(r.status===401)window.dispatchEvent(new Event('clinical:request-login'));return;}
      const date=url.searchParams.get('booking_date');url.searchParams.delete('booking');url.searchParams.delete('booking_date');history.replaceState(null,'',url);
      open({uid,name:'Ραντεβού',time:'',reason:'',freshness:'',date:/^\d{4}-\d{2}-\d{2}$/.test(date||'')?date:null});
    }catch(e){}
  }
  if(typeof URL==='function'&&typeof location!=='undefined'){
    window.addEventListener('clinical:authenticated',bookingFromURL);
    // Defer URL entry until all deferred owners (chart and patient entry) are installed.
    if(document.readyState!=='complete')document.addEventListener('DOMContentLoaded',bookingFromURL);else bookingFromURL();
  }
  window.CockpitVisitBrief={
    open, selectionContext, bookingVerification,
    activateSelection, selectPatient, invalidate, closed,
  };
})();
