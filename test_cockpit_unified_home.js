"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const html=fs.readFileSync("static/cockpit/index.html","utf8");
const source=fs.readFileSync("static/cockpit/clinical-workspace.js","utf8");
assert.match(html,/id="clinicalWorkspace"/);
assert.match(html,/id="visitRecentRows"/);
assert.match(html,/id="visitUpcomingRows"/);
assert.match(html,/id="visitDiaPrompt"/);
assert.match(html,/id="visitDiaText"/);
assert.match(html,/href="\/static\/baseline-audit\/"/);
assert.match(html,/id="surgeryTableBody"/);
for(const id of ["visitBriefOverlay","visitSidebarPrevious","visitSidebarCurrent","visitSidebarNext","visitBriefChoosePatient"])assert.match(html,new RegExp('id="'+id+'"'));
for(const route of ["/clinical/clinic-utilities/physio-referral","/clinical/clinic-utilities/sick-leave","/clinical/clinic-utilities/rf","/clinical/clinic-utilities/medical-report"])assert.equal(html.split('href="'+route+'"').length-1,1);
assert.match(source,/Περισσότερα αποτελέσματα/);
assert.match(source,/limit=20&offset=/);
assert.doesNotMatch(source,/fetch\(.+method:\s*"(POST|PUT|PATCH|DELETE)"/);
const scheduleNow="2026-10-10T07:00:00Z";
const upcoming=[
 {appointment_id:"AP1",patient_display_name:"Μαρία Δοκιμαστική",start_at:"2026-10-10T07:30:00Z",reason:"Οστεοπόρωση"},
 {appointment_id:"AP2",patient_display_name:"Δήμος Πρωί",start_at:"2026-10-10T08:00:00Z",reason:"Prolia"},
 {appointment_id:"AP3",patient_display_name:"Ελένη Τρίτη",start_at:"2026-10-10T09:00:00Z",reason:"Αξιολόγηση"}
];
const recent=[
 {patient_id:"P001",patient_display_name:"Ιωάννης Πρώτος",encounter_date:"2026-10-09",visit_type:"Επανέλεγχος"},
 {patient_id:"P002",patient_display_name:"Άννα Δεύτερη",encounter_date:"2026-10-08",visit_type:"Οστεοπόρωση"},
 {patient_id:"P003",patient_display_name:"Γιώργος Τρίτος",encounter_date:"2026-10-07",visit_type:"Ανασκόπηση"}
];
const events=new Map(), nodes=new Map();
function create(id=""){
 const callbacks={}, classes=new Set();
 const obj={
   id, textContent:"",value:"",hidden:false,children:[],attributes:{},dataset:{},disabled:false,
   classList:{toggle(name,enabled){if(enabled)classes.add(name);else classes.delete(name);}},
   setAttribute(name,value){obj.attributes[name]=String(value);},
   addEventListener(name,callback){callbacks[name]=callback;},
   fire(name){assert.ok(callbacks[name],"missing "+id+"/"+name);return callbacks[name]();},
   replaceChildren(){obj.children=[];},
   append(...items){obj.children.push(...items);},
   focus(){}, scrollIntoView(){}
 };
 return obj;
}
const tabs=["snapshot","brief","detail"].map(key=>{const el=create(key);el.dataset.clinicalTab=key;return el;});
const doc={
 getElementById(id){if(!nodes.has(id))nodes.set(id,create(id));return nodes.get(id);},
 addEventListener(){},
 activeElement:null,
 querySelectorAll(selector){return selector==="[data-clinical-tab]"?tabs:[];},
 createElement(){return create();},
 body:{append(){}}
};
const $=id=>doc.getElementById(id);
$("visitWorkspacePatient").hidden=true;
$("visitDiaComposer").hidden=true;$("visitDiaPreview").hidden=true;
$("visitDiaEditPane").hidden=true;$("visitPatientMatches").hidden=true;
$("visitDiaPrompt").textContent="SNAPSHOT\n[μία]\nVISIT BRIEF\n[δύο]\nENCOUNTER DETAIL\n[τρία]";
const requests=[],clipboard=[];
const fixture=async(url,options)=>{
 requests.push([url,options]);
 let body;
 if(url==="/clinical/recent-encounters?limit=3")body=recent;
 else if(url==="/clinical/calendar/cockpit-context")
   body={generated_at:scheduleNow,upcoming_today:upcoming,next:upcoming[0]};
 else if(url.startsWith("/clinical/patients?query="))
   body=[{patient_id:"P004",demographics:{full_name:"Μαρία Δοκιμαστική",date_of_birth:"1970-01-01"}}];
 else throw Error("Unexpected GET "+url);
 return {ok:true,status:200,json:async()=>body};
};
vm.runInNewContext(source,{document:doc,fetch:fixture,
 navigator:{clipboard:{writeText:async value=>{clipboard.push(value);}}},
 setTimeout(fn){fn();return 1;},clearTimeout(){},console,URL,encodeURIComponent,Intl,Date});
const tick=()=>new Promise(ok=>setImmediate(ok));
(async()=>{
 await tick();await tick();
 assert.equal($("visitRecentRows").children.length,3,"three recent completed visits");
 assert.equal($("visitUpcomingRows").children.length,3,"three upcoming today");
 $("previousAppointmentPatient").dataset.appointmentId="AP0";
 $("previousAppointmentPatient").textContent="Μαρία Δοκιμαστική";
 $("previousAppointmentTime").textContent="09:00";
 $("previousAppointmentType").textContent="Αξιολόγηση";
 $("calendarNote").textContent="Το πρόγραμμα μπορεί να έχει παλιώσει.";
 $("visitSidebarPrevious").fire("click");
 assert.equal($("visitBriefOverlay").hidden,false);
 assert.equal($("visitBriefName").textContent,"Μαρία Δοκιμαστική");
 assert.match($("visitBriefIdentity").textContent,/δεν αρκεί/i);
 assert.equal($("visitWorkspacePatient").hidden,true,"sidebar click stays over Home");
 $("visitBriefClose").fire("click");
 assert.equal($("visitBriefOverlay").hidden,true);
 $("visitSidebarCurrent").fire("click");
 assert.equal($("visitBriefOverlay").hidden,true,"no popup for an empty slot");
 $("visitUpcomingRows").children[0].fire("click");
 assert.equal($("visitWorkspacePatient").hidden,false);
 assert.equal($("visitIdentityStatus").textContent,"Απαιτείται επιλογή φακέλου");
 assert.equal($("visitDiaComposer").hidden,true,"Dia begins hidden");
 $("visitDiaOpen").fire("click");
 assert.equal($("visitDiaComposer").hidden,false);
 $("visitDiaCopy").fire("click");await tick();
 assert.equal(clipboard.length,1);
 $("visitDiaText").value="SNAPSHOT\nΣύντομο.\nVISIT BRIEF\nΕπανεξέταση.\nENCOUNTER DETAIL\nΚλινικό ιστορικό.";
 $("visitDiaText").fire("input");
 assert.equal($("visitDiaPreview").hidden,false);
 assert.equal($("visitDiaPreviewText").textContent,"Σύντομο.");
 tabs[2].fire("click");
 assert.equal($("visitDiaPreviewText").textContent,"Κλινικό ιστορικό.");
 $("visitDiaEdit").fire("click");
 $("visitDiaEditText").value="Διορθώθηκε.";
 $("visitDiaApplyEdit").fire("click");
 assert.equal($("visitDiaPreviewText").textContent,"Διορθώθηκε.");
 $("visitBackHome").fire("click");
 assert.equal($("visitDiaText").value,"","patient switch clears candidate");
 $("visitRecentRows").children[0].fire("click");
 assert.equal($("visitIdentityStatus").textContent,"Υπάρχει κλινικός φάκελος");
 $("visitBackHome").fire("click");
 $("visitPatientSearch").value="Μαρία";
 $("visitPatientSearch").fire("input");await tick();await tick();
 assert.equal($("visitPatientMatches").children.length,1);
 $("visitPatientMatches").children[0].fire("click");
 assert.equal($("visitSelectedName").textContent,"Μαρία Δοκιμαστική");
 assert.ok(requests.every(([,opts])=>!opts.method||opts.method==="GET"),"no writes");
 console.log("PASS Cockpit V3 Stage A: hybrid sidebar, 3+3, identity, Dia, edits and reset");
})().catch(e=>{console.error(e);process.exitCode=1;});
