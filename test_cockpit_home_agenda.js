"use strict";
const assert=require("node:assert/strict"), fs=require("node:fs"), vm=require("node:vm");
const handlers={}, root={};
const document={hidden:false,getElementById(id){return id==="homeAgendaRows"?root:null;},addEventListener(){}};
const window={addEventListener(n,fn){handlers[n]=fn;},setInterval(){}};
vm.runInNewContext(fs.readFileSync("static/cockpit/home-agenda.js","utf8"),{window,document,Intl,Date,URLSearchParams,Event});
const api=window.CockpitHomeAgenda, now=Date.parse("2026-11-18T10:52:00Z");
const rows=[
 {appointment_id:"past",start_at:"2026-11-18T08:00:00Z",end_at:"2026-11-18T08:20:00Z"},
 {appointment_id:"one",start_at:"2026-11-18T10:45:00Z",end_at:"2026-11-18T11:05:00Z"},
 {appointment_id:"two",start_at:"2026-11-18T10:50:00Z",end_at:"2026-11-18T11:10:00Z"},
 {appointment_id:"future",start_at:"2026-11-18T11:40:00Z",end_at:"2026-11-18T12:00:00Z"},
 {appointment_id:"unknown",start_at:"2026-11-18T10:35:00Z",end_at:null}];
const v=api.classify(rows,now);
assert.deepEqual(Array.from(v.past,x=>x.raw.appointment_id),["past"]);
assert.deepEqual(Array.from(v.current,x=>x.raw.appointment_id),["one","two"]);
assert.deepEqual(Array.from(v.future,x=>x.raw.appointment_id),["future"]);
assert.deepEqual(Array.from(v.unknown,x=>x.raw.appointment_id),["unknown"]);
assert.equal(api.classify(rows,Date.parse("2026-11-18T11:05:00Z")).past.length,2);
const spring=api.tomorrowRange(Date.parse("2026-03-28T12:00:00Z"));
const autumn=api.tomorrowRange(Date.parse("2026-10-24T12:00:00Z"));
assert.equal((Date.parse(spring.end)-Date.parse(spring.start))/3600000,23);
assert.equal((Date.parse(autumn.end)-Date.parse(autumn.start))/3600000,25);
assert.equal(api.cleanReason("Πόνος · Ραντεβού για: ΣΥΝΘΕΤΙΚΟΣ ΑΣΘΕΝΗΣ","Συνθετικός Ασθενής"),"Πόνος");
assert.equal(api.cleanReason("Ραντεβού για: Άλλος Ασθενής","Συνθετικός Ασθενής"),"Ραντεβού για: Άλλος Ασθενής");
console.log("PASS home agenda intervals, simultaneous bookings, uncertainty, DST and source text");
