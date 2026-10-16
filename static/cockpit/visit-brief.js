/* Bounded Visit Brief. No automatic patient GET from UID/name; no browser PHI cache. */
(() => {
  "use strict";
  const $=id=>document.getElementById(id);
  const state={
    ticket:0, view:null, ready:false, revision:0, linked:false,
    pending:null, chosen:null, patientId:null, inFlight:false
  };
  const endpoint="/clinical/calendar/visit-brief/";

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
      if(!String(value||"").trim())return;
      const li=document.createElement("li");li.textContent=String(value);list.append(li);
    });
    el.append(list);
  }
  function clearProtected(){
    state.patientId=null;
    $("visitBriefClinical").hidden=true;
    $("visitBriefOpenChart").hidden=true;
    $("visitBriefUnlink").textContent="Αποσύνδεση ραντεβού";
    $("visitBriefUnlink").dataset.armed="";
    $("visitBriefUnlink").hidden=true;
    for(const id of ["visitBriefPrevious","visitBriefPlan","visitBriefPending","visitBriefChecks"])
      $(id).replaceChildren();
  }
  function invalidate(){
    state.ticket++;
    state.ready=false;state.view=null;state.pending=null;state.chosen=null;
    state.revision=0;state.linked=false;state.inFlight=false;
    clearProtected();
    $("visitBookingConfirmation").hidden=true;
  }
  function closed(){
    // The existing overlay owner handles hiding/focus; sensitive content dies here.
    state.ticket++;
    state.ready=false;state.view=null;state.patientId=null;
    clearProtected();
  }
  function identityDenied(){
    invalidate();
    $("visitBriefIdentity").textContent="Απαιτείται νέα σύνδεση με Clinical Key.";
    if(typeof Event==="function"&&typeof window.dispatchEvent==="function")
      window.dispatchEvent(new Event("clinical:request-login"));
  }
  async function open(view){
    if(!view||!view.uid)return;
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
      $("visitBriefIdentity").textContent=data.reason==="disabled"?
        "Η επιβεβαίωση σύνδεσης ραντεβού με κλινικό φάκελο δεν έχει ενεργοποιηθεί.":
        data.reason==="needs_reconfirmation"?
        "Η προηγούμενη σύνδεση απαιτεί νέα ανεξάρτητη επιβεβαίωση. Κανένα ιστορικό δεν φορτώθηκε.":
        "Δεν υπάρχει επιβεβαιωμένη σύνδεση με φάκελο. Το όνομα του ραντεβού δεν αρκεί.";
      $("visitBriefChoosePatient").hidden=!state.ready;
      $("visitBriefChoosePatient").disabled=!state.ready;
      $("visitBriefUnlink").hidden=!state.linked;
    }catch(err){
      if(ticket!==state.ticket||$("visitBriefOverlay").hidden)return;
      clearProtected();
      $("visitBriefChoosePatient").hidden=true;
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
      revision:state.revision, linked:state.linked};
  }
  function activateSelection(context){
    invalidate();
    if(!context||!context.uid)return;
    state.pending={...context};
  }
  function selectPatient(row){
    const p=state.pending;
    if(!p||!row?.patientId)return;
    state.chosen={id:row.patientId,name:row.name};
    $("visitBookingConfirmation").hidden=false;
    $("visitBookingIndependentCheck").checked=false;
    $("visitBookingCorrection").checked=false;
    $("visitBookingCorrectionWrap").hidden=!p.linked;
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
      "Ανεξάρτητα επιβεβαιωμένος φάκελος για αυτό το άνοιγμα — χωρίς μόνιμο δικαίωμα αυτόματης προβολής.";
    const visit=b.previous_visit;
    evidence("visitBriefPrevious",visit?
      [visit.date+" · "+(visit.reason||"Υπογεγραμμένη επίσκεψη")]
        .concat(visit.findings||[]):[],
      "Δεν επιβεβαιώνεται προηγούμενη κλινική καταγραφή πριν από αυτό το ραντεβού.");
    const plan=b.prior_plan;
    const acts={prescribed:"Συνταγογραφήθηκε",continued:"Συνεχίζεται",stopped:"Διακόπηκε"};
    const medications=plan?(plan.medications||[]).map(m=>
      (acts[m.action]||m.action)+": "+m.name+(m.dose?" · "+m.dose:"")):[];
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
    evidence("visitBriefPending",(b.pending||[]).map(p=>
      p.description+" · "+p.status+" · "+(p.review_date||"χωρίς ημερομηνία")+
      " (τελευταία καταγεγραμμένη κατάσταση, επιβεβαίωσε αν ισχύει)"),
      "Δεν υπάρχουν καταγεγραμμένες ανοικτές εκκρεμότητες.");
    $("visitBriefChecksTitle").textContent=b.check_heading||"Για το ραντεβού";
    evidence("visitBriefChecks",(b.appointment_checks||[]).map(c=>c.text),
      "Δεν έχει τεκμηριωθεί συγκεκριμένος έλεγχος για αυτό το ραντεβού.");
    $("visitBriefClinical").hidden=false;
    $("visitBriefChoosePatient").hidden=true;
    $("visitBriefOpenChart").hidden=false;
    $("visitBriefUnlink").hidden=false;
  }
  async function confirmSelection(){
    const p=state.pending,chosen=state.chosen;
    if(!p||!chosen||state.inFlight)return;
    if(!$("visitBookingIndependentCheck").checked){
      $("visitBookingConfirmStatus").textContent=
        "Χρειάζεται ανεξάρτητη επαλήθευση του συγκεκριμένου προσώπου.";
      return;
    }
    const ticket=++state.ticket;
    state.inFlight=true;$("visitConfirmBooking").disabled=true;
    $("visitBookingConfirmStatus").textContent="Επιβεβαίωση σύνδεσης…";
    try{
      const data=await request("confirm",{
        booking_uid:p.uid,patient_id:chosen.id,expected_revision:p.revision,
        independently_verified:true,evidence_type:"protected_chart_identifiers",
        replace_existing:$("visitBookingCorrection").checked
      });
      if(ticket!==state.ticket||state.pending!==p||state.chosen!==chosen)return;
      if(data.mode!=="confirmed_once"||data.brief?.patient_id!==chosen.id||
         data.booking?.booking_uid!==p.uid)throw Error("Invalid confirmation result");
      state.pending=null;
      state.view={uid:p.uid,name:p.name,reason:p.reason,time:p.time};
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
    $("visitBriefChoosePatient").hidden=true;
    $("visitBriefIdentity").textContent="Αποσύνδεση…";
    try{
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
  $("visitBriefOpenChart").addEventListener("click",()=>{
    if(!state.patientId)return;
    $("visitBriefClose").click();
    $("visitWorkspaceHome").hidden=true;
    $("visitWorkspacePatient").hidden=false;
  });
  window.addEventListener("clinical:authenticated",invalidate);
  window.addEventListener("clinical:unauthenticated",()=>{
    invalidate();$("visitBriefOverlay").hidden=true;
  });
  window.CockpitVisitBrief={
    open, selectionContext, activateSelection, selectPatient, invalidate, closed,
  };
})();
