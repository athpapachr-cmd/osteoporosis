"use strict";
const fs=require("fs"),vm=require("vm"),assert=require("node:assert/strict");
const calls=[], events=[],win={CockpitStaffRole:null};
win.fetch=async u=>{calls.push(String(u));throw Error("REAL FETCH ATTEMPT")};
const context={window:win,document:{addEventListener:()=>{}},location:{href:"https://preview.invalid/"},
  fetch:win.fetch,URL,Response,Event,console};
vm.runInNewContext(fs.readFileSync("preview/visit-brief-synthetic/mock-api.js","utf8"),context);
assert.equal(win.CockpitStaffRole,"clinician");
(async()=>{
 const fetch=win.fetch;const post=(s,data)=>fetch(s,{method:"POST",body:JSON.stringify(data)});
 const ctx=await(await fetch("/clinical/calendar/cockpit-context")).json();
 assert.equal(ctx.today_total,3);
 for(const appt of [ctx.previous,ctx.current,ctx.next]){
   const uid=appt.appointment_id;
   const preview=await(await post("/clinical/calendar/visit-brief/read",{booking_uid:uid})).json();
   assert.equal(preview.mode,"booking_only");
   assert.equal(preview.reason,"unlinked");
   const search=await(await post("/clinical/calendar/visit-brief/patient-search",
     {booking_uid:uid,observed_booking:preview.booking.observation,query:"Δοκιμασ"})).json();
   assert.equal(search.length,1);
   const wrong=await post("/clinical/calendar/visit-brief/confirm",
     {booking_uid:uid,patient_id:"INVALID",expected_revision:0,observed_booking:preview.booking.observation,independently_verified:true});
   assert.equal(wrong.status,409);
   const r=await post("/clinical/calendar/visit-brief/confirm",
     {booking_uid:uid,patient_id:search[0].patient_id,expected_revision:0,observed_booking:preview.booking.observation,independently_verified:true});
   assert.equal(r.status,200);
   const confirm=await r.json();
   assert.equal(confirm.mode,"confirmed_once");
   assert(confirm.brief.previous_visit?.date);
   assert(confirm.brief.pending.length>=1);
   const reopened=await(await post("/clinical/calendar/visit-brief/read",{booking_uid:uid})).json();
   assert.equal(reopened.reason,"needs_reconfirmation");
   assert.equal(reopened.brief,undefined,"reopened booking must not disclose patient history");
 }
 const outbound=await fetch("https://ortho-reception-backend-v2.onrender.com/dashboard");
 assert.equal(outbound.status,403);
 assert.deepEqual(calls,[]);
 console.log("PASS synthetic Cockpit Visit Brief preview: 3 appointments, scoped search, confirm, reconfirm, no external IO");
})().catch(e=>{console.error(e);process.exitCode=1});
