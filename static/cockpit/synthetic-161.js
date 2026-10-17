/* Same Cockpit shell and controllers; server limits every test read/write. */
(() => {
  "use strict";
  if (!window.CockpitSynthetic) return;
  const $ = id => document.getElementById(id);
  const base = window.CockpitClinicalBase;
  document.querySelector(".clinical-visit-actions").hidden = true;
  let generation = 0, activePatient = null;
  const chart = document.createElement("section");
  chart.id = "synthetic161Chart";
  chart.setAttribute("aria-label", "Πλήρης αποθηκευμένη καρτέλα");
  $("visitWorkspacePatient").append(chart);
  const status = $("synthetic161Status");

  async function request(path, method = "GET") {
    const r = await fetch(base + path, {method, credentials:"same-origin", cache:"no-store"});
    if (!r.ok) {
      if (r.status === 401) window.dispatchEvent(new Event("clinical:request-login"));
      throw Error(r.status === 401 ? "Συνδέσου από την υπάρχουσα Πρόσβαση." :
        "Η αποθηκευμένη δοκιμή δεν είναι διαθέσιμη. Πάτησε Προετοιμασία δοκιμής.");
    }
    return r.json();
  }
  function text(parent, tag, value) {
    const el = document.createElement(tag); el.textContent = value; parent.append(el); return el;
  }
  function clear() { generation++; activePatient=null; chart.replaceChildren(); }
  function date(value, time=false){
    if(!value)return "";
    const raw=String(value), stamp=new Date(raw.length===10?raw+"T12:00:00Z":/(?:Z|[+-]\d{2}:?\d{2})$/.test(raw)?raw:raw+"Z");
    return new Intl.DateTimeFormat("el-CY",{timeZone:"Asia/Nicosia",day:"numeric",month:"long",year:"numeric",
      ...(time?{hour:"2-digit",minute:"2-digit",hour12:false}:{})}).format(stamp);
  }
  const eventTitle={created:"Δημιουργήθηκε η υποχρέωση",received:"Παραλήφθηκε το CD",reopened:"Η υποχρέωση άνοιξε ξανά"};
  function facts(parent, label, values){
    if(!values?.length)return;
    text(parent,"h4",label);
    const list=text(parent,"ul","");
    for(const fact of values)text(list,"li",typeof fact==="string"?fact:fact.text);
  }
  function revealSource(encounterId){
    const source=document.getElementById("visit-"+encounterId);
    if(!source)return;
    source.open=true;source.scrollIntoView({behavior:"smooth",block:"start"});
    source.querySelector("summary").focus();
  }
  async function renderUpdates(target, active, close){
    const ticket=generation;
    target.replaceChildren();text(target,"p","Φόρτωση ενημερώσεων…");
    try{
      const data=await request("/clinical-updates");
      if(ticket!==generation||!active())return;
      target.replaceChildren();
      text(target,"p",data.open_count+" ανοικτές υποχρεώσεις · από τις αποθηκευμένες επισκέψεις");
      for(const item of data.items){
        const row=text(target,"button","");row.type="button";row.className="doctor-action-row";
        const copy=text(row,"span","");text(copy,"strong",item.patient_name);
        text(copy,"small",item.description);
        text(copy,"small",(item.status==="resolved"?"Παραλήφθηκε":"Αναμένεται από τον ασθενή")+" · "+date(item.updated_at,true));
        text(copy,"small","Από την επίσκεψη "+date(item.source_date));
        text(row,"span","›");
        row.addEventListener("click",()=>{close();open(item.patient_id);});
      }
    }catch(e){if(ticket===generation&&active()){target.replaceChildren();text(target,"p",e.message);}}
  }
  function renderPending(target, patientId, items) {
    target.replaceChildren();
    if(!items.length) { text(target,"p","Δεν υπάρχουν ανοικτές εκκρεμότητες."); return; }
    for(const item of items){
      const row=text(target,"div",""); row.className="synthetic-pending-row";
      const copy=text(row,"div","");
      text(copy,"strong",item.description);
      text(copy,"p",item.status==="resolved"?"Παραλήφθηκε":"Αναμένεται από τον ασθενή");
      const origin=text(copy,"button","Δημιουργήθηκε στην επίσκεψη · "+date(item.source_date));
      origin.type="button";origin.className="clinical-source-link";
      origin.addEventListener("click",async()=>{
        if(!$("visitBriefOverlay").hidden)$("visitBriefClose").click();
        if(activePatient!==patientId)await open(patientId);
        revealSource(item.source_encounter_id);
      });
      const action=text(row,"button",item.status==="resolved"?"Αναίρεση":"Παραλήφθηκε");
      action.type="button";action.className="pending-action";
      const feedback=text(row,"span",""); feedback.setAttribute("role","status");
      action.addEventListener("click",async()=>{
        action.disabled=true;
        try {
          const response=await fetch(base+"/patient/"+encodeURIComponent(patientId)+"/pending/"+encodeURIComponent(item.pending_id),{
            method:"POST",credentials:"same-origin",cache:"no-store",headers:{"Content-Type":"application/json"},
            body:JSON.stringify({status:item.status==="resolved"?"awaiting_patient":"resolved",expected_updated_at:item.updated_at})
          });
          if(!response.ok) throw Error(response.status===409?"Η εκκρεμότητα άλλαξε σε άλλη συνεδρία. Άνοιξε ξανά τον φάκελο.":"Δεν αποθηκεύτηκε. Δοκίμασε ξανά.");
          const updated=await response.json();
          if(!row.isConnected)return;
          renderPending(target,patientId,items.map(p=>p.pending_id===updated.pending_id?updated:p));
          if(activePatient===patientId)await open(patientId);
          window.dispatchEvent(new Event("cockpit:clinical-updates-changed"));
        } catch(e){feedback.textContent=e.message;action.disabled=false;}
      });
    }
  }
  async function open(patientId) {
    clear(); const ticket = generation;
    text(chart,"p","Φόρτωση καρτέλας…");
    try {
      const data = await request("/patient/" + encodeURIComponent(patientId) + "/chart");
      if (ticket !== generation) return;
      chart.replaceChildren();
      $("visitWorkspaceHome").hidden = true;
      $("visitWorkspacePatient").hidden = false;
      $("visitSelectedName").textContent = data.demographics.full_name;
      $("visitSelectedSource").textContent = "ΚΑΡΤΕΛΑ ΑΣΘΕΝΟΥΣ";
      $("visitSelectedReason").textContent = "Γέννηση: " + data.demographics.dob;
      $("visitIdentityStatus").textContent = "";
      $("visitIdentityStatus").hidden = true;
      $("visitIdentityNote").textContent = "";
      $("visitIdentityNote").hidden = true;
      $("visitBookingConfirmation").hidden = true;
      $("visitDiaComposer").hidden = true;
      $("visitDiaOpen").hidden = true;
      activePatient=patientId;
      const latest=data.visits[0];
      const heading=text(chart,"div","");heading.className="patient-chart-heading";
      text(heading,"h2","Κλινική συνέχεια");
      text(heading,"p",data.visits.length+" αποθηκευμένες επισκέψεις · Τελευταία επίσκεψη "+date(latest?.date));
      const grid=text(chart,"div","");grid.className="patient-chart-grid";
      const overview=text(grid,"section","");overview.className="patient-chart-card";
      text(overview,"h3","Κλινική εικόνα");
      text(overview,"p","Από την επίσκεψη "+date(latest?.date));
      const clinical=latest?.clinical||{};
      facts(overview,"Λόγος παρακολούθησης",clinical.reason_for_visit?[clinical.reason_for_visit]:[]);
      facts(overview,"Εκτίμηση",clinical.clinical_impression);
      facts(overview,"Συμφωνημένο σχέδιο",clinical.decisions);
      const obligations=text(grid,"section","");obligations.className="patient-chart-card";
      const openItems=data.pending.filter(p=>p.status!=="resolved"&&p.status!=="cancelled");
      text(obligations,"h3","Υποχρεώσεις · "+openItems.length+(openItems.length===1?" ανοικτή":" ανοικτές"));
      const pendingBox=text(obligations,"div","");
      renderPending(pendingBox,patientId,data.pending);
      const timeline=text(chart,"section","");timeline.className="patient-timeline";
      text(timeline,"h3","Πορεία ασθενούς");
      const entries=data.visits.map(visit=>({at:visit.date+"T08:00:00",kind:"visit",visit}));
      for(const item of data.pending){
        const changes=(item.history||[]).filter(e=>e.kind!=="created");
        if(changes.length)entries.push({...changes[changes.length-1],item,earlier:changes.slice(0,-1)});
      }
      entries.sort((a,b)=>b.at.localeCompare(a.at));
      for(const entry of entries){
        if(entry.kind!=="visit"){
          const event=text(timeline,"article","");event.className="patient-timeline-event";
          text(event,"small",date(entry.at,true));
          text(event,"h4",eventTitle[entry.kind]);
          text(event,"p",entry.item.description);
          if(entry.earlier.length){
            const earlier=text(event,"details","");
            text(earlier,"summary","Προηγούμενες αλλαγές · "+entry.earlier.length);
            for(const change of [...entry.earlier].reverse())text(earlier,"p",date(change.at,true)+" · "+eventTitle[change.kind]);
          }
          continue;
        }
        const visit=entry.visit,c=visit.clinical;
        const block=text(timeline,"details","");block.className="patient-visit";block.id="visit-"+visit.encounter_id;
        const summary=text(block,"summary","");
        text(summary,"span",date(visit.date));
        text(summary,"strong",c.reason_for_visit?.text||"Κλινική επίσκεψη");
        const originated=data.pending.filter(p=>p.source_encounter_id===visit.encounter_id);
        for(const item of originated){
          text(summary,"small","Δημιουργήθηκε υποχρέωση: "+item.description);
        }
        const body=text(block,"div","");body.className="patient-visit-body";
        facts(body,"Ευρήματα",c.findings);
        facts(body,"Κλινική εκτίμηση",c.clinical_impression);
        facts(body,"Αποφάσεις",c.decisions);
        facts(body,"Αγωγή",(c.medications||[]).map(m=>({prescribed:"Έναρξη",continued:"Συνέχιση",stopped:"Διακοπή"}[m.action]||"")+" · "+m.name+(m.dose?" · "+m.dose:"")));
        if(originated.length){
          const origin=text(body,"section","");origin.className="patient-obligation-origin";
          text(origin,"h4","Υποχρεώσεις που δημιουργήθηκαν στην επίσκεψη");
          for(const item of originated){
            text(origin,"strong",item.description);
            text(origin,"p","Καταγράφηκε "+date(item.created_at)+" · Ανατέθηκε στον ασθενή");
            text(origin,"p","Τρέχουσα κατάσταση: "+(item.status==="resolved"?"Παραλήφθηκε":"Αναμένεται από τον ασθενή"));
          }
        }
        facts(body,"Οδηγίες επανεπικοινωνίας",c.safety_net);
        facts(body,"Προς διευκρίνιση κατά την επίσκεψη",c.uncertainties);
      }
      const url = new URL(location.href); url.searchParams.set("chart", patientId);
      history.replaceState(null, "", url);
    } catch (e) { if(ticket === generation) status.textContent = e.message; }
  }
  $("synthetic161Prepare").addEventListener("click", async () => {
    const button = $("synthetic161Prepare"); button.disabled = true;
    status.textContent = "Αποθήκευση ελεγχόμενου σεναρίου…";
    try {
      await request("/prepare", "POST");
      status.textContent = "Η δοκιμή αποθηκεύτηκε. Επίλεξε ένα από τα τρία ραντεβού.";
      window.dispatchEvent(new Event("clinical:authenticated"));
    } catch (e) { status.textContent = e.message; }
    finally { button.disabled = false; }
  });
  $("visitBackHome").addEventListener("click", () => {
    clear(); const url = new URL(location.href); url.searchParams.delete("chart");
    history.replaceState(null, "", url);
  });
  window.addEventListener("clinical:unauthenticated", clear);
  window.addEventListener("clinical:authenticated", () => {
    clear();
    const patient = new URL(location.href).searchParams.get("chart");
    if(patient) open(patient);
    window.dispatchEvent(new Event("cockpit:clinical-updates-changed"));
  });
  window.CockpitStoredChart = {open, renderPending, renderUpdates, reset() {
    clear();
    const url = new URL(location.href); url.searchParams.delete("chart");
    history.replaceState(null, "", url);
  }};
  // Restores by a fresh authenticated backend read, never browser storage.
  const patientId = new URL(location.href).searchParams.get("chart");
  if (patientId) open(patientId);
})();
