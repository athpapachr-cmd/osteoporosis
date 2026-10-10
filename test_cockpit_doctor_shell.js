"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const html=fs.readFileSync("static/cockpit/index.html","utf8");
const script=fs.readFileSync("static/cockpit/doctor-shell.js","utf8");
assert.match(html,/id="doctorDateButton"/);
assert.match(html,/id="doctorInboxButton"/);
assert.match(html,/id="doctorTasksButton"/);
assert.match(html,/id="doctorLibraryView"/);
assert.match(html,/id="visitBriefOverlay"/);
assert.doesNotMatch(script,/localStorage|sessionStorage/);

const nodes=new Map(),listener=new Map(),calls=[];
function el(id="",tag="div"){
 const handlers=new Map(),classes=new Set();
 const obj={id,tagName:tag,textContent:"",value:"",hidden:false,children:[],attributes:{},
  dataset:{},firstChild:{textContent:"Πρόγραμμα σήμερα "},className:"",
  classList:{contains(name){return classes.has(name);},toggle(name,val){if(val===undefined)val=!classes.has(name);if(val)classes.add(name);else classes.delete(name);return val;}},
  setAttribute(name,value){obj.attributes[name]=String(value);},
  addEventListener(name,fn){handlers.set(name,fn);},
  fire(name,event={}){const fn=handlers.get(name);assert.ok(fn,"handler "+id+"/"+name);return fn(event);},
  replaceChildren(){obj.children=[];},
  append(...children){obj.children.push(...children);},
  focus(){},
  contains(target){return target===obj||obj.children.some(c=>c===target||typeof c.contains==="function"&&c.contains(target));}
 };
 return obj;
}
const tabs=["surgery","documents","modules","learning"].map(c=>{
 const b=el("tab-"+c,"button");b.dataset.doctorLibraryTab=c;return b;
});
const panels=["surgery","documents","modules","learning"].map(c=>{
 const div=el("panel-"+c);div.dataset.doctorLibraryPanel=c;return div;
});
const doc={
 getElementById(id){if(!nodes.has(id))nodes.set(id,el(id));return nodes.get(id);},
 createElement(tag){return el("",tag);},
 querySelectorAll(q){
  if(q==="[data-doctor-library-tab]")return tabs;
  if(q==="[data-doctor-library-panel]")return panels;
  return [];
 },
 addEventListener(name,fn){listener.set(name,fn);},
 body:{classList:{contains(){return false;},toggle(){}}}
};
const $=id=>doc.getElementById(id);
$("doctorPopover").hidden=true;$("doctorLibraryView").hidden=true;
$("doctorInboxBadge").hidden=true;$("doctorTasksBadge").hidden=true;
const windowHandlers=new Map(),windowStub={
 CockpitHome:{openAppointment(row){windowStub.lastAppointment=row;}},
 addEventListener(name,fn){windowHandlers.set(name,fn);},
 CockpitSurgerySummary:null
};
let deferred=null,authenticated=true;
const mockAppointments=[
 {appointment_id:"cal.com:synthetic-1",patient_display_name:"Συνθετικός Ασθενής Α",
  start_at:"2026-10-11T07:20:00Z",reason:"Συνθετική οστεοπόρωση"},
 {appointment_id:"cal.com:synthetic-2",patient_display_name:"Συνθετικός Ασθενής Β",
  start_at:"2026-10-11T08:40:00Z",reason:"Prolia"}
];
async function fakeFetch(url,opts){
 calls.push({url,method:opts.method||"GET",options:opts});
 if(!url.startsWith("/clinical/calendar/appointments?"))throw Error("unexpected URL "+url);
 if(deferred)return deferred.promise;
 return {ok:authenticated,status:authenticated?200:401,json:async()=>mockAppointments};
}
vm.runInNewContext(script,{document:doc,window:windowStub,fetch:fakeFetch,
 URLSearchParams,Date,Intl,console});
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function findButtonContent(title) {
 return $("doctorPopoverContent").children.find(item=>
   item.tagName==="button" && item.children?.some(c=>
     c.children?.some(n=>n.textContent?.includes(title))));
}
(async()=>{
 assert.equal($("doctorLibraryView").hidden,true);
 assert.equal($("doctorInboxBadge").hidden,true,"no invented unread email count");
 assert.equal($("doctorAttentionList").children.length,3);
 $("doctorDateButton").fire("click");
 await tick();await tick();
 assert.equal($("doctorPopover").hidden,false);
 assert.equal($("doctorDateButton").attributes["aria-expanded"],"true");
 assert.equal(calls.length,1);
 assert.match(calls[0].url,/\/clinical\/calendar\/appointments\?start=/);
 assert.match(calls[0].url,/&end=/);
 const first=findButtonContent("Συνθετικός Ασθενής Α");
 assert.ok(first,"actual protected appointment is actionable");
 first.fire("click");
 assert.equal($("doctorPopover").hidden,true);
 assert.equal(windowStub.lastAppointment.patientId,null,"appointment name cannot link a patient");
 assert.equal(windowStub.lastAppointment.appointmentId,"cal.com:synthetic-1");

 $("doctorInboxButton").fire("click");
 assert.equal($("doctorInboxBadge").hidden,true);
 assert.ok($("doctorPopoverContent").children.some(x=>x.textContent?.includes("Δεν έχει συνδεθεί")));
 $("doctorInboxButton").fire("click");
 assert.equal($("doctorPopover").hidden,true);

 windowHandlers.get("cockpit:surgery-counts")({detail:{total:2,undated:1}});
 assert.equal($("doctorTasksBadge").hidden,false);
 assert.equal($("doctorTasksBadge").textContent,"2");
 $("doctorTasksButton").fire("click");
 const surgical=findButtonContent("Εκκρεμή χειρουργεία");
 assert.ok(surgical);
 surgical.fire("click");
 assert.equal($("doctorLibraryView").hidden,false);
 assert.equal($("doctorMainView").hidden,true);
 assert.equal(panels[0].hidden,false);
 assert.equal(panels[1].hidden,true);
 tabs[1].fire("click");
 assert.equal(panels[1].hidden,false);
 $("doctorBackHome").fire("click");
 assert.equal($("doctorMainView").hidden,false);
 assert.equal($("doctorLibraryView").hidden,true);

 $("doctorActionsButton").fire("click");
 const palette=$("doctorPopoverContent").children[0];
 assert.equal(palette.id,"doctorCommandSearch");
 palette.value="Ραδιο";
 palette.fire("input");
 const matches=$("doctorPopoverContent").children[1].children;
 assert.ok(matches.some(a=>a.href==="/clinical/clinic-utilities/rf"));
 $("doctorActionsButton").fire("click");
 assert.equal($("doctorPopover").hidden,true);

 authenticated=false;
 $("doctorDateButton").fire("click");await tick();await tick();
 assert.ok($("doctorPopoverContent").children.some(x=>x.textContent?.includes("Clinical Data Key")));
 $("doctorDateButton").fire("click");

 authenticated=true;
 let resolve;
 deferred={promise:new Promise(done=>{resolve=done;})};
 $("doctorDateButton").fire("click");
 $("doctorInboxButton").fire("click");
 const inboxState=$("doctorPopoverContent").children.length;
 resolve({ok:true,status:200,json:async()=>mockAppointments});
 await tick();await tick();
 assert.equal($("doctorPopover").dataset.kind,"inbox","late day response cannot replace Inbox");
 assert.equal($("doctorPopoverContent").children.length,inboxState);
 assert.ok(calls.every(x=>x.method==="GET"),"no new clinical writes");
 assert.ok(calls.every(x=>!x.url.includes("patient_id")),"no patient identifiers in URLs");
 console.log("PASS Cockpit Doctor Shell: contextual day, Peek route, Inbox truth, tasks, actions, library, stale isolation");
})().catch(e=>{console.error(e);process.exitCode=1;});
