"use strict";
/* #162 synthetic-only demo. Render strings with textContent; no protected storage. */
(() => {
  const el = id => document.getElementById(id);
  const state = {pendingId: "", patientId: "", checkpoint: null, busy: false};
  const say = message => { el("message").textContent = message; };
  async function json(path, options = {}) {
    const response = await fetch(path, {credentials:"same-origin", cache:"no-store", ...options});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error("Αποτυχία " + response.status);
    return data;
  }
  function show(view, daily) {
    state.checkpoint = view;
    el("checkpoint").classList.toggle("hidden", !view);
    const next = view?.next_checkpoint;
    el("itemStatus").textContent = view ? "Εκκρεμότητα · revision " + view.revision + " · " + view.status : "";
    el("question").textContent = next?.question || (view ? "Όλα τα εγκεκριμένα στάδια ολοκληρώθηκαν." : "");
    el("verified").textContent = view ? "Επιβεβαιώθηκαν: " + (view.confirmed_keys.join(", ") || "κανένα") : "";
    el("yes").disabled = state.busy || !next;
    el("notYet").disabled = state.busy || !next;
    const same = view && daily?.items?.some(item =>
      item.pending_id === view.pending_id && item.revision === view.revision);
    el("comparison").textContent = view ?
      "Patient: " + view.pending_id + " · " + view.revision +
      "\nDaily Brief: " + (same ? "ίδιο pending/revision" :
        view.status === "resolved" ? "ολοκληρώθηκε — εκτός ανοικτών" : "μη διαθέσιμο") :
      "Δεν έχουν φορτωθεί δεδομένα.";
  }
  async function refresh() {
    if (!state.patientId || !state.pendingId) return;
    const [items, daily] = await Promise.all([
      json("/clinical/patient/" + encodeURIComponent(state.patientId) + "/pending"),
      json("/clinical/daily-brief/pending"),
    ]);
    const row = items.find(item => item.pending_id === state.pendingId);
    if (!row?.checkpoint) throw new Error("Δεν υπάρχει εγκεκριμένο checkpoint plan");
    show(row.checkpoint, daily);
  }
  function setBusy(value) {
    state.busy = value;
    el("yes").disabled = value || !state.checkpoint?.next_checkpoint;
    el("notYet").disabled = value || !state.checkpoint?.next_checkpoint;
  }
  async function action(observation) {
    const step = state.checkpoint?.next_checkpoint;
    if (!step || state.busy) return;
    setBusy(true);
    try {
      await json("/clinical/patient/" + encodeURIComponent(state.patientId) +
        "/pending/" + encodeURIComponent(state.pendingId) + "/checkpoint", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({checkpoint_key:step.key, observation,
          expected_revision:state.checkpoint.revision,
          idempotency_key:"syn-" + crypto.randomUUID().replaceAll("-","")}),
      });
      await refresh(); say("Καταγράφηκε μόνο το συγκεκριμένο στάδιο.");
    } catch (error) {
      show(null, null); say(error.message + " · Ανανέωσε πριν απαντήσεις.");
    } finally { setBusy(false); }
  }
  el("signIn").addEventListener("click", async () => {
    try {
      const key = el("staffKey").value;
      await json("/clinical/login", {method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({key})});
      el("staffKey").value = "";say("Συνδέθηκες στο synthetic περιβάλλον.");
    } catch (error) {el("staffKey").value="";say(error.message);}
  });
  el("signOut").addEventListener("click", async () => {
    try{await json("/clinical/logout",{method:"POST"});}catch(_){}
    show(null,null);state.patientId="";state.pendingId="";say("Αποσυνδέθηκες.");
  });
  el("load").addEventListener("click",async()=>{
    state.patientId=el("patientId").value.trim();state.pendingId="";
    if (!state.patientId){say("Συμπλήρωσε synthetic patient ID.");return;}
    try{
      const rows=await json("/clinical/patient/"+encodeURIComponent(state.patientId)+"/pending");
      const row=rows.find(item=>item.checkpoint);
      if (!row)throw new Error("Δεν υπάρχει synthetic pending με εγκεκριμένο plan");
      state.pendingId=row.pending_id;await refresh();say("Φορτώθηκε η εκκρεμότητα.");
    }catch(error){show(null,null);say(error.message);}
  });
  el("yes").addEventListener("click",()=>action("yes_confirmed"));
  el("notYet").addEventListener("click",()=>action("not_yet_observed"));
})();
