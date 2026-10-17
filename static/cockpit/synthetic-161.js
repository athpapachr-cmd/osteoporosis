/* Same Cockpit shell and controllers; server limits every test read/write. */
(() => {
  "use strict";
  if (!window.CockpitSynthetic) return;
  const $ = id => document.getElementById(id);
  const base = window.CockpitClinicalBase;
  document.querySelector(".clinical-visit-actions").hidden = true;
  let generation = 0;
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
  function clear() { generation++; chart.replaceChildren(); }
  function renderPending(target, patientId, items) {
    target.replaceChildren();
    if(!items.length) { text(target,"p","Δεν υπάρχουν ανοικτές εκκρεμότητες."); return; }
    for(const item of items){
      const row=text(target,"div",""); row.className="synthetic-pending-row";
      const copy=text(row,"div","");
      text(copy,"strong",item.description);
      text(copy,"p",item.status==="resolved"?"Παραλήφθηκε":"Αναμένεται από τον ασθενή");
      const action=text(row,"button",item.status==="resolved"?"Αναίρεση":"Παραλήφθηκε");
      action.type="button";
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
        } catch(e){feedback.textContent=e.message;action.disabled=false;}
      });
    }
  }
  async function open(patientId) {
    clear(); const ticket = generation;
    try {
      const data = await request("/patient/" + encodeURIComponent(patientId) + "/chart");
      if (ticket !== generation) return;
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
      text(chart, "h2", "Πλήρης καρτέλα · " + data.visits.length + " αποθηκευμένες επισκέψεις");
      text(chart, "h3", "Καταγεγραμμένες εκκρεμότητες");
      const pendingBox=text(chart,"div","");
      renderPending(pendingBox,patientId,data.pending);
      for (const visit of data.visits) {
        const block = text(chart, "article", "");
        text(block, "h3", "Επίσκεψη " + visit.date);
        for (const [title, field] of [["Συνοπτική εικόνα","snapshot"],["Σύνοψη επίσκεψης","brief"],["Πλήρης καταγραφή","detail"]]) {
          const details = text(block, "details", "");
          details.open = field === "brief" && visit === data.visits[0];
          text(details, "summary", title);
          const pre = text(details, "p", visit[field]); pre.style.whiteSpace = "pre-wrap";
        }
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
  });
  window.CockpitStoredChart = {open, renderPending, reset() {
    clear();
    const url = new URL(location.href); url.searchParams.delete("chart");
    history.replaceState(null, "", url);
  }};
  // Restores by a fresh authenticated backend read, never browser storage.
  const patientId = new URL(location.href).searchParams.get("chart");
  if (patientId) open(patientId);
})();
