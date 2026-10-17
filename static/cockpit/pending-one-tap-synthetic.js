"use strict";
// Browser-only design evaluation. No fetch, credentials, localStorage or clinical writes.
(() => {
  const $ = (id) => document.getElementById(id);
  const steps = [
    "Προσκομίστηκε το CD της μαγνητικής;",
    "Εξετάστηκαν οι εικόνες από τον γιατρό;",
  ];
  let stage = 0;
  let observation = "";
  let events = [];
  function render() {
    const done = stage === steps.length;
    $("question").textContent = done ?
      "Ολοκληρώθηκαν τα δύο εγκεκριμένα στάδια." : steps[stage];
    $("question").classList.toggle("finished", done);
    $("yes").disabled = done;
    $("notYet").disabled = done;
    $("undo").disabled = stage === 0;
    $("observation").textContent = observation;
    const state = done ? "Ολοκληρώθηκε — παραλαβή CD και εξέταση εικόνων επιβεβαιώθηκαν" :
      stage === 1 ? "Παραλήφθηκε το CD · Εκκρεμεί η εξέταση εικόνων" :
      "Ανοικτή — η παραλαβή του CD δεν έχει επιβεβαιωθεί";
    $("patientStatus").textContent = state;
    $("briefStatus").textContent = state;
    $("audit").textContent = "Ενδεικτικά γεγονότα στη μνήμη: " + events.length +
      " · " + (events.length ? events.at(-1) : "κανένα");
  }
  $("yes").addEventListener("click", () => {
    if (stage >= steps.length) return;
    events.push("Ναι — " + steps[stage]);
    stage += 1;
    observation = "Επιβεβαιώθηκε μόνο το προηγούμενο στάδιο.";
    render();
  });
  $("notYet").addEventListener("click", () => {
    if (stage >= steps.length) return;
    events.push("Όχι ακόμη — " + steps[stage]);
    observation = "Δεν έχει ακόμη επιβεβαιωθεί. Η εκκρεμότητα παραμένει ανοικτή.";
    render();
  });
  $("undo").addEventListener("click", () => {
    if (stage === 0) return;
    stage -= 1;
    events.push("Αναίρεση — " + steps[stage]);
    observation = "Η προηγούμενη επιβεβαίωση αναιρέθηκε στο demo.";
    render();
  });
  $("reset").addEventListener("click", () => {
    stage = 0; observation = ""; events = []; render();
  });
  render();
})();
