/* Osteoporosis Treatment Strategy — synthetic-only deterministic view model.
 * NO medication selection, FRAX calculation, prescribing, persistence or patient API.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.StrategyCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const copy = value => JSON.parse(JSON.stringify(value));
  const preferences = new Set(['not_discussed', 'accepts', 'declines', 'alternative', 'undecided']);
  const outcomes = new Set(['pending', 'agreed', 'deferred', 'declined']);

  function historyStatus(patient, discussion) {
    if (discussion.history_confirmation === 'confirmed_none') return 'confirmed_none';
    if (discussion.history_confirmation === 'unknown') return 'not_recorded';
    if (discussion.history_confirmation === 'previous_therapy') return 'known_incomplete';
    if (patient.history_conflict) return 'conflicting';
    if ((patient.episodes || []).some(ep => ep.exposure === 'confirmed')) return 'known';
    if (patient.confirmed_no_prior_therapy) return 'confirmed_none';
    return 'not_recorded';
  }
  function actualAdministration(patient, agent) {
    const doses = (patient.administrations || []).filter(a => a.agent === agent && a.status === 'administered' && /^\d{4}-\d{2}-\d{2}$/.test(a.date || ''));
    return doses.length ? doses.map(a => a.date).sort().at(-1) : null;
  }
  function getTreatmentContext(patient, discussion) {
    const status = historyStatus(patient, discussion);
    const priorDenosumab = (patient.episodes || []).some(ep => ep.agent === 'denosumab' && ep.exposure === 'confirmed');
    const lastActualDose = actualAdministration(patient, 'denosumab');
    const actualDoseConflict = !!patient.administration_conflict;
    const intent = discussion.therapy_course || patient.therapy_course || 'not_established';
    return {status, priorDenosumab, lastActualDose, actualDoseConflict, intent};
  }
  function currentQuestion(patient, discussion) {
    const c = getTreatmentContext(patient, discussion);
    if (patient.new_fracture_during_treatment) return 'Τι χρειάζεται να επανεκτιμήσουμε πριν συνεχίσουμε ή αλλάξουμε τη θεραπεία;';
    if (c.priorDenosumab && c.intent === 'subsequent_treatment' && patient.current_treatment) return 'Τι χρειάζεται να επανεκτιμήσουμε στη σημερινή θεραπεία;';
    if (c.priorDenosumab && c.intent === 'ongoing') return 'Πώς διασφαλίζουμε τη συνέχεια της σημερινής θεραπείας;';
    if (c.priorDenosumab) return 'Τι σημαίνει η προηγούμενη Prolia για τη σημερινή απόφαση;';
    if (patient.completed_anabolic) return 'Πώς διατηρούμε το όφελος που έχει επιτευχθεί;';
    if (patient.current_treatment) return 'Εξακολουθεί η σημερινή θεραπεία να καλύπτει τις ανάγκες του ασθενούς;';
    if (patient.lower_risk) return 'Ποια φροντίδα έχει μεγαλύτερη σημασία σήμερα;';
    return 'Ποια θεραπευτική προσέγγιση αξίζει να εξετάσουμε;';
  }
  function contextLine(patient, discussion) {
    const c = getTreatmentContext(patient, discussion);
    if (c.priorDenosumab) return 'Προηγούμενη Prolia · χρειάζεται να καταλάβουμε τι ακολούθησε.';
    if (c.status === 'confirmed_none') return 'Επιβεβαιώθηκε ότι δεν έχει προηγηθεί θεραπεία.';
    if (c.status === 'known_incomplete') return 'Έχει προηγηθεί αγωγή, αλλά δεν γνωρίζουμε ακόμη ποια.';
    if (c.status === 'not_recorded') return 'Δεν έχει καταγραφεί προηγούμενη θεραπεία — δεν σημαίνει ότι δεν υπήρξε.';
    if (c.status === 'conflicting') return 'Υπάρχουν αντικρουόμενα στοιχεία θεραπευτικού ιστορικού.';
    if (patient.current_treatment) return 'Σήμερα λαμβάνει ' + patient.current_treatment.label + '.';
    if (patient.completed_anabolic) return 'Έχει καταγραφεί ολοκλήρωση προηγούμενης αναβολικής θεραπείας.';
    return 'Υπάρχει καταγεγραμμένο θεραπευτικό ιστορικό.';
  }
  function makeDiscussion() {
    return {history_confirmation:null,therapy_course:null,confirmed_last_dose:null,dose_declared_unknown:false,
      selected_focus:null,considered:[],recommendation:'',patient_preference:'not_discussed',decision:'pending',rationale:'',saved:false};
  }
  function createSession(fixture) {return {patient:copy(fixture), discussion:makeDiscussion(),stage:'rest',opened:'',revision:fixture.revision,synthetic:true};}
  function replacePatient(session, fixture) {return createSession(fixture);}
  function updateSource(session, fixture) {return createSession(fixture);}
  function setHistory(session,value) {
    if (!['confirmed_none','previous_therapy','unknown'].includes(value)) throw Error('invalid history answer');
    session.discussion.history_confirmation = value;
    session.discussion.saved=false;
    return session;
  }
  function setCourse(session,value) {
    if (!['ongoing','transition_considered','subsequent_treatment','not_established'].includes(value)) throw Error('invalid course');
    session.discussion.therapy_course=value;session.discussion.saved=false;return session;
  }
  function setDose(session,value) {
    if(value === null){session.discussion.dose_declared_unknown=true;session.discussion.confirmed_last_dose=null;return session;}
    if(!/^\d{4}-\d{2}-\d{2}$/.test(value) || Number.isNaN(Date.parse(value))) throw Error('invalid date');
    session.discussion.confirmed_last_dose=value;session.discussion.dose_declared_unknown=false;session.discussion.saved=false;return session;
  }
  function markConsidered(session,approach){
    if(!['bone_building','bone_loss_reduction','nonpharm','continuity_review','current_treatment_review'].includes(approach))throw Error('invalid option');
    if(!session.discussion.considered.includes(approach))session.discussion.considered.push(approach);
    session.discussion.saved=false;return session;
  }
  function setDecision(session, fields) {
    if (!preferences.has(fields.patient_preference) || !outcomes.has(fields.decision)) throw Error('invalid decision');
    session.discussion.recommendation=String(fields.recommendation||'').slice(0,800);
    session.discussion.patient_preference=fields.patient_preference;
    session.discussion.decision=fields.decision;
    session.discussion.rationale=String(fields.rationale||'').slice(0,800);
    session.discussion.saved=true;
    return session;
  }
  function approachInfo(key,patient){
    const reviewed={
      bone_building:{name:'Θεραπείες που ενισχύουν τον σχηματισμό οστού',why:'Σε ορισμένους ασθενείς με πολύ υψηλό κίνδυνο αξίζει να εξεταστεί αυτή η προσέγγιση.',matters:'Το πρόσφατο σπονδυλικό κάταγμα είναι σημαντικό στην κλινική συζήτηση.',later:'Αν επιλεγεί, θα χρειαστεί και σχεδιασμός της συνέχειας μετά τη θεραπεία.',source:'NOGG 2024 §6 — very-high-risk treatment consideration'},
      bone_loss_reduction:{name:'Θεραπείες που περιορίζουν την οστική απώλεια',why:'Αποτελούν σημαντική θεραπευτική προσέγγιση σε κατάλληλο κλινικό πλαίσιο.',matters:'Η προηγούμενη αγωγή, η νεφρική κατάσταση και οι προτιμήσεις επηρεάζουν την επιλογή.',later:'Η συνέχεια και η παρακολούθηση εξαρτώνται από τη συγκεκριμένη θεραπεία.',source:'NOGG 2024 §6 — treatment choice factors'},
      nonpharm:{name:'Άσκηση, πτώσεις και διατροφική επάρκεια',why:'Η φροντίδα αυτή είναι ουσιαστικό μέρος της αντιμετώπισης.',matters:'Οι ανάγκες και δυνατότητες του συγκεκριμένου ασθενούς καθορίζουν τον στόχο.',later:'Ένας συμφωνημένος στόχος μπορεί αργότερα να συνδεθεί με ενέργεια και επανεκτίμηση.',source:'NOGG 2024 §5 — non-pharmacological care'},
      current_treatment_review:{name:'Επανεκτίμηση της σημερινής θεραπείας',why:'Η καταγεγραμμένη σημερινή αγωγή και η προηγούμενη έκθεση έχουν σημασία για τη συζήτηση.',matters:'Αξιολογούνται η τρέχουσα πορεία και όσα πραγματικά γνωρίζουμε από προηγούμενες θεραπείες.',later:'Η τελική απόφαση θα καθορίσει τη συνέχεια της φροντίδας χωρίς να δημιουργεί αυτόματα χορήγηση.',source:'NOGG 2024 §6 — treatment choice factors'},
      continuity_review:{name:'Ασφαλής συνέχεια μετά από Prolia',why:'Η αλλαγή ή διακοπή απαιτεί προσοχή όταν έχει προηγηθεί denosumab.',matters:'Πρέπει να γνωρίζουμε την πραγματική τελευταία χορήγηση και τι έχει ακολουθήσει.',later:'Το τελικό πλάνο χρειάζεται να διατηρεί τις πραγματικές υποχρεώσεις συνέχειας.',source:'NOGG 2024 §6 — denosumab long-term plan'}
    };
    return reviewed[key]||null;
  }
  function makeHandoff(session) {
    if(!session.discussion.saved)return null;
    const patient=session.patient,c=getTreatmentContext(patient,session.discussion);
    return {type:'synthetic_strategy_handoff_preview',patient_fixture:patient.id,source_revision:patient.revision,
      source_refs:copy(patient.source_refs||[]),risk_assessment_reference:patient.risk?.source_ref||null,
      question:currentQuestion(patient,session.discussion),treatment_context:c,clinician_reported_actual_dose:session.discussion.confirmed_last_dose,
      considered:copy(session.discussion.considered),recommendation:session.discussion.recommendation,
      patient_preference:session.discussion.patient_preference,decision:session.discussion.decision,
      rationale:session.discussion.rationale,unresolved:[]
        .concat(['not_recorded','known_incomplete'].includes(c.status)?['unverified_treatment_history']:[])
        .concat(c.priorDenosumab&&!c.lastActualDose&&!session.discussion.confirmed_last_dose?['last_actual_denosumab_dose_not_established']:[])
        .concat(c.actualDoseConflict?['conflicting_administration_records']:[]),
      authoritative_write:false,medication_selected_by_engine:false,administration_created:false,plan_tasks_created:false};
  }
  return {historyStatus,actualAdministration,getTreatmentContext,currentQuestion,contextLine,
    createSession,replacePatient,updateSource,setHistory,setCourse,setDose,markConsidered,
    setDecision,approachInfo,makeHandoff};
});