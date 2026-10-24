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
  const reviewLabel=status=>({
    unreviewed:"Δεν επισκοπήθηκε",
    reviewed:"✓ Επισκοπήθηκε",
    needs_re_review:"Απαιτείται επανέλεγχος"
  })[status]||"Άγνωστη κατάσταση";
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
      body.append(make("small","",label+" · "+reviewLabel(item.review_state)+
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
    const id=item.item_id,g=state.generation,ticket=++state.selection;
    state.busy=true;
    $("doctorClinicalInboxFeedback").textContent="Αποθήκευση…";
    try{
      const data=await api("/"+encodeURIComponent(id)+"/action",{
        method:"POST",body:JSON.stringify({action:kind,expected_version:item.version,
          idempotency_key:crypto.randomUUID(),...extra})
      });
      if(!current(g,id)||ticket!==state.selection)return;
      if(data.item?.item_id!==id||data.item.version<=item.version)
        throw Error("Unexpected action response");
      // Display the authoritative updated state at once. Avoid an async
      // list/detail reload that can hide an otherwise successful one-tap action.
      state.items=state.items.map(row=>row.item_id===id?data.item:row);
      state.busy=false;
      rows();
      render(data.item,g);
      $("doctorClinicalInboxFeedback").textContent={
        mark_reviewed:"✓ Επισκοπήθηκε το εισερχόμενο. Δεν σημαίνει ιατρικό έλεγχο αποτελεσμάτων.",
        needs_review:"Καταγράφηκε ότι απαιτείται επανέλεγχος.",
        confirm_link:"✓ Επιβεβαιώθηκε η σύνδεση με τον συνθετικό ασθενή.",
        no_communication:"Καταγράφηκε ότι δεν απαιτείται επικοινωνία."
      }[kind]||"Η ενέργεια καταγράφηκε.";
    }catch(e){
      if(!current(g,id)||ticket!==state.selection)return;
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
        reviewLabel(item.review_state)+" · έκδοση "+item.active_source_revision));
    if(item.source_type==="gesy_notice"&&item.source_status==="results_available")
      box.append(make("p","clinical-inbox-note",
        "Αποτελέσματα διαθέσιμα — εκκρεμεί ιατρικός έλεγχος. Δεν υπάρχουν τιμές εξετάσεων."));
    const safe=item.source_subject_scope==="single_subject";
    box.append(make("p",safe?"clinical-inbox-note":"clinical-inbox-warning",
      safe?(item.source_type==="gesy_notice"?
          "Ειδοποίηση διαθεσιμότητας — δεν περιλαμβάνει κλινική ερμηνεία ή αποτελέσματα.":
          (item.candidate_summary||"Προηγούμενη εξαγωγή άκυρη· απαιτείται επανέλεγχος.")):
        "Πολλοί/άγνωστοι ασθενείς. Το έγγραφο παραμένει μόνο στο Inbox."));
    box.append(make("p","clinical-inbox-note",item.identity_state==="clinician_confirmed"?
      "Συνδεδεμένος συνθετικός ασθενής: "+item.patient_id+" · "+
        (item.link_evidence_kind==="verified_referral"?"προηγουμένως επιβεβαιωμένο παραπεμπτικό":
         "επαληθευμένη σύνδεση"):
      item.identity_state==="conflict"?"Σύγκρουση τεκμηρίων — παραμένει χωρίς σύνδεση.":
      "Δεν υπάρχει επιβεβαιωμένη σύνδεση με ασθενή."));
    if(item.review_state==="reviewed")
      box.append(make("p","clinical-inbox-note",
        "✓ Το εισερχόμενο επισκοπήθηκε. Αυτό δεν είναι ιατρική αποδοχή αποτελεσμάτων."));
    const actions=make("div","clinical-inbox-actions");box.append(actions);
    if(item.review_state!=="reviewed")
      button(actions,"Το είδα",()=>action(item,"mark_reviewed"));
    button(actions,"Απαιτείται επανέλεγχος",()=>action(item,"needs_review"));
    if(item.review_state==="reviewed"&&item.identity_state==="clinician_confirmed")
      button(actions,"Χωρίς επικοινωνία",()=>action(item,"no_communication"));
    if(!safe||item.identity_state==="clinician_confirmed"||item.identity_state==="conflict")return;
    const strong=item.verified_link_candidate;
    if(!strong){
      // No typed synthetic secret or checkbox can create qualified evidence.
      // Keep the weak source Inbox-only until the protected source owner
      // independently establishes a current, unique patient association.
      box.append(make("p","clinical-inbox-warning",
        "Δεν υπάρχει επαληθευμένη αντιστοίχιση με ασθενή. " +
        "Το έγγραφο παραμένει στα εισερχόμενα. Δεν χρειάζεται να εισαγάγεις κωδικό."));
      return;
    }
    const form=make("div","clinical-inbox-link");
    form.append(make("p","clinical-inbox-note",
      strong.reason+" — μία επιβεβαίωση, χωρίς καταχώριση στοιχείων."));
    const patientLabel=make("p","clinical-inbox-note","Έλεγχος συνθετικού φακέλου…");
    const submit=make("button","doctor-command-button","Επιβεβαίωση ασθενούς");
    submit.type="button";submit.disabled=true;
    submit.addEventListener("click",()=>action(item,"confirm_link",{
      patient_id:strong.patient_id,evidence_kind:strong.evidence_kind,
      independently_verified:true
    }));
    form.append(patientLabel,submit);box.append(form);
    api("/eligible-patients").then(d=>{
      if(!current(g,item.item_id)||!box.contains(form))return;
      if(d.mode!=="synthetic_only"||!Array.isArray(d.patients))
        throw Error("Invalid synthetic patient scope");
      const matches=d.patients.filter(p=>p.patient_id===strong.patient_id);
      if(matches.length!==1){
        form.remove();box.append(make("p","clinical-inbox-warning",
          "Δεν υπάρχει μοναδικός επαληθευμένος συνθετικός φάκελος. Η σύνδεση δεν επιτρέπεται."));
        return;
      }
      patientLabel.textContent="Ασθενής: "+matches[0].name;
      submit.disabled=false;
    }).catch(e=>{
      if(!current(g,item.item_id)||!box.contains(form))return;
      form.remove();
      if(e.status===401){
        window.dispatchEvent(new Event("clinical:unauthenticated"));
        window.dispatchEvent(new Event("clinical:request-login"));
      }else{
        box.append(make("p","clinical-inbox-warning",
          "Δεν επιβεβαιώθηκε ο συνθετικός φάκελος. Ανανέωσε πριν από σύνδεση."));
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
