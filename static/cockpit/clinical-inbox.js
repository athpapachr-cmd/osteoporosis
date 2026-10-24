/* Phase A: protected #161 synthetic fixture ONLY. No provider data or browser storage. */
(() => {
  "use strict";
  const $=id=>document.getElementById(id);
  const view=$("doctorClinicalInboxView");
  if(!view)return;
  const base=window.CockpitClinicalBase;
  const enabled=window.CockpitSynthetic===true&&base==="/clinical/synthetic-161";
  const state={active:false,generation:0,selection:0,id:null,items:[],accounts:[],busy:false,controller:null};
  const make=(tag,cls,value)=>{
    const e=document.createElement(tag);if(cls)e.className=cls;
    if(value!==undefined)e.textContent=String(value);return e;
  };
  function button(parent,label,fn){
    const b=make("button","doctor-command-button",label);
    b.type="button";b.addEventListener("click",fn);parent.append(b);return b;
  }
  function clear(){
    ++state.generation;++state.selection;
    state.controller?.abort();state.controller=null;state.id=null;
    state.items=[];state.accounts=[];state.busy=false;
    $("doctorClinicalInboxRows").replaceChildren();
    $("doctorClinicalInboxDetails").replaceChildren();
    $("doctorClinicalInboxFeedback").textContent="";
    $("doctorClinicalInboxState").textContent="";
    $("doctorClinicalInboxShown").textContent="";
    $("doctorClinicalInboxAccount").replaceChildren(new Option("Όλες οι δοκιμαστικές πηγές",""));
    $("doctorClinicalInboxReviewFilter").value="";
  }
  function close(){clear();state.active=false;view.hidden=true;}
  const current=(g,id)=>state.active&&!view.hidden&&g===state.generation&&
    (id===undefined||id===state.id);
  async function api(path,options={}){
    const response=await fetch(base+"/attention"+path,{
      credentials:"same-origin",cache:"no-store",
      headers:{"Accept":"application/json",...(options.body?{"Content-Type":"application/json"}:{})},
      ...options
    });
    if(!response.ok){const e=new Error("Request rejected");e.status=response.status;throw e;}
    return response.json();
  }
  function rows(){
    const box=$("doctorClinicalInboxRows");box.replaceChildren();
    const a=$("doctorClinicalInboxAccount").value,r=$("doctorClinicalInboxReviewFilter").value;
    const items=state.items.filter(x=>(!a||x.mailbox_account_ref===a)&&
      (!r||x.review_state===r));
    $("doctorClinicalInboxShown").textContent=items.length+" συνθετικά αντικείμενα · όχι πραγματικά emails";
    for(const item of items){
      const b=make("button","doctor-action-row clinical-inbox-row");b.type="button";
      b.dataset.attentionId=item.item_id;b.setAttribute("aria-pressed",String(state.id===item.item_id));
      const body=make("span");body.append(make("strong","",item.source_title));
      const label=state.accounts.find(x=>x.mailbox_account_id===item.mailbox_account_ref)?.label||"Δοκιμή";
      body.append(make("small","",label+" · "+item.review_state+
        (item.source_subject_scope!=="single_subject"?" · μόνο Inbox":"")));
      b.append(body,make("span","","›"));b.addEventListener("click",()=>select(item.item_id));
      box.append(b);
    }
    if(!items.length)box.append(make("p","clinical-inbox-note","Δεν υπάρχουν αντικείμενα σε αυτό το δοκιμαστικό φίλτρο."));
  }
  async function load(focusItemId=null){
    if(!enabled||!state.active)return;
    const g=++state.generation;
    state.controller?.abort();const ac=new AbortController();state.controller=ac;
    $("doctorClinicalInboxState").textContent="Φόρτωση συνθετικού Inbox…";
    try{
      const data=await api("",{signal:ac.signal});
      if(!current(g))return;
      if(data.mode!=="synthetic_only"||data.live_provider_connected!==false||
          !Array.isArray(data.items)||!Array.isArray(data.accounts))throw Error("Invalid scope");
      state.items=data.items;state.accounts=data.accounts;
      const accountSelect=$("doctorClinicalInboxAccount"),old=accountSelect.value;
      accountSelect.replaceChildren(new Option("Όλες οι δοκιμαστικές πηγές",""));
      for(const account of data.accounts)accountSelect.append(new Option(account.label,account.mailbox_account_id));
      accountSelect.value=data.accounts.some(x=>x.mailbox_account_id===old)?old:"";
      $("doctorClinicalInboxState").textContent="Μόνο συνθετικά δεδομένα. Καμία πραγματική θυρίδα συνδεδεμένη.";
      rows();
      if(focusItemId&&data.items.some(i=>i.item_id===focusItemId))select(focusItemId);
    }catch(e){
      if(!current(g))return;
      if(e.status===401){
        // Only the current clinician's denied read may invalidate a session.
        window.dispatchEvent(new Event("clinical:unauthenticated"));
        window.dispatchEvent(new Event("clinical:request-login"));
        return;
      }
      // Never keep a formerly authorized patient projection after an error.
      state.items=[];state.accounts=[];state.id=null;
      $("doctorClinicalInboxRows").replaceChildren();
      $("doctorClinicalInboxDetails").replaceChildren();
      $("doctorClinicalInboxState").textContent="Η συνθετική πηγή δεν είναι διαθέσιμη. Δεν δηλώνεται ότι δεν υπάρχουν μηνύματα.";
    }
  }
  async function select(id){
    if(!state.active||state.busy)return;
    const g=state.generation,ticket=++state.selection;state.id=id;
    const box=$("doctorClinicalInboxDetails");box.replaceChildren(make("p","clinical-inbox-note","Φόρτωση…"));
    rows();
    try{
      const data=await api("/"+encodeURIComponent(id));
      if(!current(g,id)||ticket!==state.selection)return;
      if(data.mode!=="synthetic_only"||data.item?.item_id!==id)throw Error("Unexpected item");
      render(data.item,g);
    }catch(e){
      if(!current(g,id)||ticket!==state.selection)return;
      if(e.status===401){
        window.dispatchEvent(new Event("clinical:unauthenticated"));
        window.dispatchEvent(new Event("clinical:request-login"));
        return;
      }
      box.replaceChildren(make("p","clinical-inbox-note","Δεν επιβεβαιώθηκε το αντικείμενο."));
    }
  }
  async function action(item,kind,extra={}){
    if(state.busy||!state.active||state.id!==item.item_id)return;
    const id=item.item_id,g=state.generation;state.busy=true;
    $("doctorClinicalInboxFeedback").textContent="Αποθήκευση συνθετικής ενέργειας…";
    try{
      const data=await api("/"+encodeURIComponent(id)+"/action",{
        method:"POST",body:JSON.stringify({action:kind,expected_version:item.version,
          idempotency_key:crypto.randomUUID(),...extra})
      });
      if(!current(g,id))return;
      if(data.item?.item_id!==id)throw Error("Unexpected response");
      state.busy=false;state.id=null;
      $("doctorClinicalInboxDetails").replaceChildren();await load();
      if(state.active)select(id);
      $("doctorClinicalInboxFeedback").textContent="Η συνθετική ενέργεια καταγράφηκε.";
    }catch(e){
      if(!current(g,id))return;
      state.busy=false;
      if(e.status===401){
        window.dispatchEvent(new Event("clinical:unauthenticated"));
        window.dispatchEvent(new Event("clinical:request-login"));return;
      }
      $("doctorClinicalInboxFeedback").textContent=e.status===409?
        "Η εγγραφή άλλαξε. Ανανέωσε πριν ενεργήσεις ξανά.":
        "Η ενέργεια δεν επιβεβαιώθηκε. Ανανέωσε για να δεις την καταγεγραμμένη κατάσταση.";
    }
  }
  function render(item,g){
    const box=$("doctorClinicalInboxDetails");box.replaceChildren();
    box.append(make("h3","",item.source_title),
      make("p","clinical-inbox-note","Πηγή: "+item.mailbox_account_ref+" · "+
        item.review_state+" · έκδοση "+item.active_source_revision));
    if(item.source_type==="gesy_notice"&&item.source_status==="results_available")
      box.append(make("p","clinical-inbox-note",
        "Αποτελέσματα διαθέσιμα — εκκρεμεί ιατρικός έλεγχος. Δεν υπάρχουν τιμές εξετάσεων."));
    const safe=item.source_subject_scope==="single_subject";
    box.append(make("p",safe?"clinical-inbox-note":"clinical-inbox-warning",
      safe?(item.candidate_summary||"Προηγούμενη εξαγωγή άκυρη· απαιτείται επανέλεγχος."):
        "Πολλοί/άγνωστοι ασθενείς. Το έγγραφο παραμένει μόνο στο Inbox."));
    box.append(make("p","clinical-inbox-note",item.identity_state==="clinician_confirmed"?
      "Συνδεδεμένος συνθετικός ασθενής: "+item.patient_id+" · "+
        (item.link_evidence_kind==="verified_referral"?"προηγουμένως επιβεβαιωμένο παραπεμπτικό":
         "επαληθευμένη σύνδεση"):
      item.identity_state==="conflict"?"Σύγκρουση τεκμηρίων — παραμένει χωρίς σύνδεση.":
      "Δεν υπάρχει επιβεβαιωμένη σύνδεση με ασθενή."));
    const actions=make("div","clinical-inbox-actions");box.append(actions);
    button(actions,"Ελέγχθηκε",()=>action(item,"mark_reviewed"));
    button(actions,"Απαιτείται επανέλεγχος",()=>action(item,"needs_review"));
    if(item.review_state==="reviewed"&&item.identity_state==="clinician_confirmed")
      button(actions,"Χωρίς επικοινωνία",()=>action(item,"no_communication"));
    if(!safe||item.identity_state==="clinician_confirmed"||item.identity_state==="conflict")return;
    const form=make("form","clinical-inbox-link");
    const strong=item.verified_link_candidate;
    form.append(make("p","clinical-inbox-note",strong?
      strong.reason+" — σύγκρινε με τον συνθετικό φάκελο και επιβεβαίωσε μία φορά. Δεν απαιτείται ημερομηνία γέννησης.":
      "Μόνο πιθανή ταυτοποίηση. Το όνομα δεν αρκεί· απαιτείται ανεξάρτητο τεκμήριο από τον προστατευμένο φάκελο ή παραμένει ασύνδετο."));
    const patient=make("select");patient.required=true;patient.setAttribute("aria-label","Δοκιμαστικός ασθενής");
    patient.append(new Option("Επιλογή συνθετικού φακέλου",""));
    const pLabel=make("label","","Δοκιμαστικός ασθενής");pLabel.append(patient);
    form.append(pLabel);
    let evidence=null,confirm=null;
    if(!strong){
      const eLabel=make("label","","Κωδικός ανεξάρτητου συνθετικού τεκμηρίου");
      evidence=make("input");evidence.type="text";evidence.maxLength=64;evidence.required=true;
      evidence.autocomplete="off";evidence.placeholder="Αρχικό παραπεμπτικό από φάκελο";
      evidence.setAttribute("aria-label","Ανεξάρτητο συνθετικό τεκμήριο");
      eLabel.append(evidence);form.append(eLabel);
      const confirmLabel=make("label","clinical-inbox-checkbox");
      confirm=make("input");confirm.type="checkbox";confirm.required=true;
      confirmLabel.append(confirm,make("span","",
        "Σύγκρινα το αρχικό τεκμήριο με τον προστατευμένο συνθετικό φάκελο."));
      form.append(confirmLabel);
    }
    const submit=make("button","doctor-command-button","Επιβεβαίωση ασθενούς");
    submit.type="submit";submit.disabled=true;form.append(submit);
    form.addEventListener("submit",e=>{
      e.preventDefault();
      if(!patient.value||submit.disabled)return;
      if(strong){
        if(patient.value!==strong.patient_id)return;
        action(item,"confirm_link",{patient_id:patient.value,
          evidence_kind:strong.evidence_kind,independently_verified:true});
      }else{
        if(!confirm.checked||!evidence.value.trim())return;
        action(item,"confirm_link",{patient_id:patient.value,
          evidence_kind:"clinician_source_comparison",
          evidence_ref:evidence.value.trim(),independently_verified:true});
      }
    });
    box.append(form);
    api("/eligible-patients").then(d=>{
      if(!current(g,item.item_id)||!box.contains(form))return;
      if(d.mode!=="synthetic_only"||!Array.isArray(d.patients))
        throw Error("Invalid synthetic patient scope");
      for(const p of d.patients){
        if(!strong||p.patient_id===strong.patient_id)patient.append(new Option(p.name,p.patient_id));
      }
      if(strong){
        if(!d.patients.some(p=>p.patient_id===strong.patient_id)){
          form.remove();box.append(make("p","clinical-inbox-warning",
            "Ο επαληθευμένος φάκελος δεν βρέθηκε· η σύνδεση δεν επιτρέπεται."));return;
        }
        patient.value=strong.patient_id;patient.disabled=true;
      }
      submit.disabled=false;
    }).catch(e=>{
      // A superseded read must not revoke the replacement clinician session.
      if(!current(g,item.item_id)||!box.contains(form))return;
      form.remove(); // No patient-link action survives failed identity coverage.
      if(e.status===401){
        window.dispatchEvent(new Event("clinical:unauthenticated"));
        window.dispatchEvent(new Event("clinical:request-login"));
      }else{
        box.append(make("p","clinical-inbox-warning",
          "Δεν επιβεβαιώθηκαν οι συνθετικοί ασθενείς. Ανανέωσε πριν από σύνδεση."));
      }
    });
  }
  function open(focusItemId=null){
    clear();state.active=true;view.hidden=false;
    if(enabled)load(focusItemId);
    else $("doctorClinicalInboxState").textContent="Καμία πραγματική πηγή email δεν είναι συνδεδεμένη.";
  }
  $("doctorClinicalInboxAccount").addEventListener("change",rows);
  $("doctorClinicalInboxReviewFilter").addEventListener("change",rows);
  $("doctorClinicalInboxReload").addEventListener("click",()=>{state.id=null;
    $("doctorClinicalInboxDetails").replaceChildren();load();});
  window.addEventListener("clinical:unauthenticated",close);
  window.addEventListener("clinical:authenticated",()=>{
    if(!state.active)return;
    clear();state.active=true;load();
  });
  window.CockpitClinicalInbox={open,close};
})();
