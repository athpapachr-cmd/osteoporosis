'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const C=require('./strategy-core.js');
const F=require('./fixtures.js');

test('J distinguishes no record from confirmed treatment-naive',()=>{
  let s=C.createSession(F.J);assert.equal(C.historyStatus(s.patient,s.discussion),'not_recorded');
  C.setHistory(s,'confirmed_none');assert.equal(C.historyStatus(s.patient,s.discussion),'confirmed_none');
  assert.equal(F.J.confirmed_no_prior_therapy,false);
});
test('J explicit previous therapy with details missing is not treatment-naive',()=>{
  const s=C.createSession(F.J);C.setHistory(s,'previous_therapy');
  assert.equal(C.historyStatus(s.patient,s.discussion),'known_incomplete');
});
test('J an unanswered/unknown history is not explicit negative',()=>{
  const s=C.createSession(F.J);C.setHistory(s,'unknown');
  assert.equal(C.historyStatus(s.patient,s.discussion),'not_recorded');
});
test('JD last actual dose remains unknown until proven; scheduled dose excluded',()=>{
  const s=C.createSession(F.JD);assert.equal(C.getTreatmentContext(s.patient,s.discussion).lastActualDose,null);
  s.patient.administrations.push({agent:'denosumab',date:'2026-08-12',status:'planned'});
  assert.equal(C.getTreatmentContext(s.patient,s.discussion).lastActualDose,null);
  C.setDose(s,'2026-04-12');assert.equal(s.discussion.confirmed_last_dose,'2026-04-12');
  assert.equal(C.getTreatmentContext(s.patient,s.discussion).lastActualDose,null);
});
test('JD known administered exposure is reused and not inferred from prescription',()=>{
  const s=C.createSession(F.JDknown);assert.equal(C.getTreatmentContext(s.patient,s.discussion).lastActualDose,'2026-04-11');
  assert.equal(C.currentQuestion(s.patient,s.discussion),'Πώς διασφαλίζουμε τη συνέχεια της σημερινής θεραπείας;');
});
test('JD conflict is not resolved by picking a date automatically',()=>{
  const s=C.createSession(F.JDconflict);assert.equal(C.getTreatmentContext(s.patient,s.discussion).actualDoseConflict,true);
  assert.equal(C.getTreatmentContext(s.patient,s.discussion).lastActualDose,null);
});
test('prior Prolia alone does not imply ongoing therapy',()=>{
  const s=C.createSession(F.JD);assert.equal(C.getTreatmentContext(s.patient,s.discussion).intent,'not_established');
  assert.match(C.currentQuestion(s.patient,s.discussion),/προηγούμενη Prolia/);
});
test('new fracture under bisphosphonate promotes reassessment, not failure',()=>{
  const s=C.createSession(F.S3);assert.match(C.currentQuestion(s.patient,s.discussion),/επανεκτιμήσουμε/);
  assert.doesNotMatch(C.currentQuestion(s.patient,s.discussion),/απέτυχε/);
});
test('post-anabolic completion focuses on preserving benefit',()=>{
  const s=C.createSession(F.S4);assert.match(C.currentQuestion(s.patient,s.discussion),/διατηρούμε/);
});
test('lower risk has non-drug work as first question',()=>{
  const s=C.createSession(F.S8);assert.match(C.currentQuestion(s.patient,s.discussion),/φροντίδα/);
});
test('option consideration is not clinician recommendation or final decision',()=>{
  const s=C.createSession(F.J0);C.markConsidered(s,'bone_building');
  assert.equal(s.discussion.considered.length,1);assert.equal(s.discussion.recommendation,'');
  assert.equal(s.discussion.decision,'pending');assert.equal(C.makeHandoff(s),null);
});
test('recommendation, patient view and actual decision remain distinct',()=>{
  const s=C.createSession(F.J0);C.markConsidered(s,'nonpharm');
  C.setDecision(s,{recommendation:'Να συζητηθεί η προσέγγιση.',patient_preference:'declines',decision:'deferred',rationale:'Εκκρεμεί συζήτηση.'});
  const h=C.makeHandoff(s);assert.equal(h.recommendation,'Να συζητηθεί η προσέγγιση.');
  assert.equal(h.patient_preference,'declines');assert.equal(h.decision,'deferred');
  assert.deepEqual(h.considered,['nonpharm']);assert.equal(h.authoritative_write,false);
  assert.equal(h.administration_created,false);assert.equal(h.plan_tasks_created,false);
});
test('unknown history and last denosumab administration remain open in handoff',()=>{
  const s=C.createSession(F.JD);C.setDecision(s,{recommendation:'',patient_preference:'undecided',decision:'pending',rationale:''});
  const h=C.makeHandoff(s);assert.deepEqual(h.unresolved,['last_actual_denosumab_dose_not_established']);
});
test('patient switch and source revision clear transient recommendation and preferences',()=>{
  let s=C.createSession(F.J0);C.setDecision(s,{recommendation:'A',patient_preference:'accepts',decision:'agreed',rationale:'B'});
  s=C.replacePatient(s,F.JD);assert.equal(s.discussion.recommendation,'');assert.equal(s.discussion.patient_preference,'not_discussed');
  s=C.updateSource(s,{...F.JD,revision:'fixture-2'});assert.equal(s.revision,'fixture-2');assert.equal(s.discussion.saved,false);
});