/* One protected registry entry for Home, appointment drafts and natural capture. */
(() => {
  'use strict';
  if(window.CockpitSynthetic)return;
  const $=id=>document.getElementById(id);
  const el=(parent,tag,text,cls)=>{const n=document.createElement(tag);if(text)n.textContent=text;if(cls)n.className=cls;parent.append(n);return n;};
  const button=(parent,text,fn)=>{const b=el(parent,'button',text,'clinical-secondary');b.type='button';b.onclick=fn;return b;};
  let revision=0,searchRevision=0,timer=null,active=false,creating=false,capture=false,creationId=null,opener=null;
  const dialog=el(document.body,'dialog','','patient-entry-dialog');dialog.id='patientEntry';dialog.setAttribute('aria-labelledby','patientEntryTitle');
  const heading=el(dialog,'div','','clinical-row-between');el(heading,'h2','Καρτέλα ασθενούς').id='patientEntryTitle';button(heading,'Κλείσιμο',dismiss);
  el(dialog,'p','Άνοιξε υπάρχουσα καρτέλα ή δημιούργησε νέα.','clinical-muted');
  const form=el(dialog,'form');
  function field(title,id,type,required=false){const label=el(form,'label',title),input=el(label,'input');input.id=id;input.type=type;input.required=required;input.autocomplete='off';return input;}
  const name=field('Ονοματεπώνυμο','patientEntryName','text',true);name.maxLength=200;
  const results=el(form,'div','','patient-entry-results');results.id='patientEntryMatches';results.setAttribute('aria-live','polite');
  const fields=el(form,'div','','patient-entry-demographics');
  const dob=field('Ημερομηνία γέννησης · προαιρετικά','patientEntryDob','date');fields.append(dob.parentElement);
  const phone=field('Τηλέφωνο · προαιρετικά','patientEntryPhone','tel');phone.maxLength=60;fields.append(phone.parentElement);
  const submit=el(form,'button','Δημιουργία νέας καρτέλας','clinical-primary');submit.type='submit';submit.id='patientEntryCreate';
  const status=el(dialog,'p','','clinical-muted');status.id='patientEntryStatus';status.setAttribute('role','status');
  button(dialog,'Πρόσβαση',()=>{dismiss();window.dispatchEvent(new Event('clinical:request-login'));});
  function clear(){revision++;searchRevision++;clearTimeout(timer);active=false;creating=false;name.value='';dob.value='';phone.value='';results.replaceChildren();status.textContent='';for(const control of form.elements)control.disabled=false;}
  function dismiss(){clear();if(dialog.open)dialog.close();opener?.focus?.();opener=null;}
  dialog.onclose=()=>{if(active)clear();};
  async function api(path,body){
    const r=await fetch('/clinical'+path,{method:body===undefined?'GET':'POST',credentials:'same-origin',cache:'no-store',headers:{'Content-Type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)})});
    if(!r.ok){const e=Error(r.status===401||r.status===403?'Συνδέσου από την Πρόσβαση για την καρτέλα.':r.status===409?'Το αίτημα έχει ήδη αποθηκευτεί με άλλα στοιχεία. Αναζήτησε την αποθηκευμένη καρτέλα πριν δημιουργήσεις άλλη.':'Η ενέργεια δεν ολοκληρώθηκε. Τα στοιχεία διατηρήθηκαν· μπορείς να δοκιμάσεις ξανά.');e.status=r.status;throw e;}return r.json();
  }
  async function choose(id,ticket=revision){
    if(ticket!==revision||!active)return;
    const startCapture=capture;dismiss();const after=revision;
    window.CockpitVisitBrief?.invalidate();
    const opened=await window.CockpitStoredChart.open(id);
    if(after===revision&&opened===id&&startCapture)window.CockpitStoredChart.beginVisit();
  }
  function displayName(p){const d=p.demographics||{};return d.full_name||d.fullName||d.name||[d.first_name||d.firstName,d.last_name||d.lastName].filter(Boolean).join(' ')||'Καταχωρισμένος ασθενής';}
  async function search(append=false,offset=0){
    const term=name.value.trim(),ticket=revision,lookup=++searchRevision;
    if(!term||!active)return;
    try{const rows=await api('/patients?query='+encodeURIComponent(term.slice(0,120))+'&limit=20&offset='+offset);
      if(ticket!==revision||lookup!==searchRevision||!active||term!==name.value.trim())return;
      if(!append)results.replaceChildren();
      if(!rows.length&&!append)el(results,'p','Δεν βρέθηκε καρτέλα. Μπορείς να δημιουργήσεις νέα με τα παρακάτω στοιχεία.','clinical-muted');
      for(const p of rows){const row=el(results,'div','','patient-entry-match');const copy=el(row,'div');el(copy,'strong',displayName(p));const d=p.demographics||{};el(copy,'small',[d.dob||d.date_of_birth||d.birth_date,d.phone].filter(Boolean).join(' · '));button(row,capture?'Καταγραφή επίσκεψης':'Άνοιγμα καρτέλας',()=>choose(p.patient_id,ticket));}
      if(rows.length===20){const more=button(results,'Περισσότερα αποτελέσματα',()=>{more.remove();search(true,offset+rows.length);});}
    }catch(e){if(ticket===revision&&lookup===searchRevision&&active){results.replaceChildren();el(results,'p',e.message,'clinical-muted');}}
  }
  name.oninput=()=>{clearTimeout(timer);searchRevision++;results.replaceChildren();status.textContent='';timer=setTimeout(()=>search(),250);};
  form.onsubmit=async e=>{
    e.preventDefault();if(creating||!active||!form.reportValidity())return;
    creating=true;searchRevision++;clearTimeout(timer);results.replaceChildren();const ticket=revision;for(const control of form.elements)control.disabled=true;
    status.textContent='Δημιουργία καρτέλας…';
    try{const p=await api('/patients/create',{creation_id:creationId,full_name:name.value,dob:dob.value||null,phone:phone.value||null});if(ticket===revision&&active)await choose(p.patient_id,ticket);}
    catch(error){if(ticket===revision&&active)status.textContent=error.message;}
    finally{if(ticket===revision&&active){creating=false;for(const control of form.elements)control.disabled=false;}}
  };
  function open(options={}){
    if(window.CockpitStoredChart?.canLeave&&!window.CockpitStoredChart.canLeave())return;
    window.CockpitStoredChart?.cancelPendingRead();
    dismiss();active=true;capture=Boolean(options.capture);creationId=crypto.randomUUID();opener=document.activeElement;
    name.value=String(options.name||'').slice(0,200);submit.textContent=capture?'Δημιουργία και καταγραφή επίσκεψης':'Δημιουργία νέας καρτέλας';
    dialog.showModal();name.focus();if(name.value.trim())search();
  }
  const homeButton=button($('visitPatientSearch').parentElement,'Νέος ασθενής',()=>open({name:$('visitPatientSearch').value}));homeButton.id='patientEntryHome';
  window.CockpitPatientEntry={open};
  window.addEventListener('clinical:unauthenticated',dismiss);
  window.addEventListener('clinical:authenticated',()=>{dismiss();fromURL();});
  window.addEventListener('cockpit:patient-opening',()=>{if(active)dismiss();});
  async function fromURL(){
    const url=new URL(location.href);if(url.searchParams.get('capture')!=='1')return;
    const ticket=revision;
    try{await api('/status');if(ticket!==revision)return;
      const id=url.searchParams.get('entry_patient');url.searchParams.delete('capture');url.searchParams.delete('entry_patient');history.replaceState(null,'',url);
      if(id&&id.length<=120){const opened=await window.CockpitStoredChart.open(id);if(opened===id&&ticket===revision)window.CockpitStoredChart.beginVisit();}
      else open({capture:true});
    }catch(e){if(ticket===revision&&e.status===401)window.dispatchEvent(new Event('clinical:request-login'));}
  }
  fromURL();
})();
