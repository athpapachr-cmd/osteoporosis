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
  start_at:"2026-10-10T22:20:00Z",reason:"Συνθετική οστεοπόρωση"},
 {appointment_id:"cal.com:synthetic-2",patient_display_name:"Συνθετικός Ασθενής Β",
  start_at:"2026-10-11T08:40:00Z",reason:"Prolia"}
];
async function fakeFetch(url,opts){
 calls.push({url,method:opts.method||"GET",options:opts});
 if(url==="/clinical/clinic-utilities/rf/api/outcomes/search"){
   assert.equal(opts.method,"POST","RF endpoint is read-intent POST only");
   assert.deepEqual(JSON.parse(opts.body),{query:"",scope:"all",page:1});
   return {ok:true,status:200,json:async()=>({
     denominator:5,scope:"all",page:1,truncated:false,applications:[],
     counts:{issued_unknown:1,pending:2,approved:1,rejected:0,scheduled:1,performed:1,mine_actionable:1}
   })};
 }
 if(!url.startsWith("/clinical/calendar/agenda?"))throw Error("unexpected URL "+url);
 if(deferred)return deferred.promise;
 return {ok:authenticated,status:authenticated?200:401,json:async()=>mockAppointments};
}
class CalendarClock extends Date{constructor(...args){super(...(args.length?args:['2026-12-31T12:00:00Z']));}}
vm.runInNewContext(script.replace("  function text(message) {","  window.dayRangeForTest=todayRange;\n  function text(message) {"),{document:doc,window:windowStub,fetch:fakeFetch,
 URLSearchParams,Date:CalendarClock,Intl,console});
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function findButtonContent(title) {
 return $("doctorPopoverContent").children.find(item=>
   item.tagName==="button" && item.children?.some(c=>
     c.children?.some(n=>n.textContent?.includes(title))));
}
function byId(id,node=$('doctorPopoverContent')){if(node.id===id)return node;for(const child of node.children||[]){const found=byId(id,child);if(found)return found;}return null;}
(async()=>{
 for(const [day,hours] of [['2026-03-29',23],['2026-10-25',25],['2026-12-31',24]]){
   const range=windowStub.dayRangeForTest(new Date(day+'T12:00:00Z'));
   assert.equal((new Date(range.end)-new Date(range.start))/3600000,hours,'Cyprus calendar boundaries '+day);
 }

 assert.equal($("doctorLibraryView").hidden,true);
 assert.equal($("doctorInboxBadge").hidden,true,"no invented unread email count");
 assert.equal($("doctorAttentionList").children.length,4);
 $("doctorDateButton").fire("click");
 await tick();await tick();
 assert.equal($("doctorPopover").hidden,false);
 assert.equal($("doctorDateButton").attributes["aria-expanded"],"true");
 const calendarCalls=calls.filter(x=>x.url.startsWith("/clinical/calendar/agenda?"));
 assert.equal(calendarCalls.length,1);
 assert.match(calendarCalls[0].url,/\/clinical\/calendar\/agenda\?start=/);
 assert.match(calendarCalls[0].url,/&end=/);
 const todayStart=new URLSearchParams(calendarCalls[0].url.split('?')[1]).get('start');
 assert.equal(todayStart,'2026-12-30T22:00:00.000Z');
 byId('doctorDayNext').fire('click');await tick();await tick();
 assert.match(byId('doctorDayLabel').textContent,/Αύριο/);
 const tomorrowCall=calls.filter(c=>c.url.startsWith('/clinical/calendar/agenda?')).at(-1);
 const tomorrowStart=new URLSearchParams(tomorrowCall.url.split('?')[1]).get('start');
 assert.equal(tomorrowStart,new URLSearchParams(calendarCalls[0].url.split('?')[1]).get('end'));
 assert.equal(tomorrowStart,'2026-12-31T22:00:00.000Z','next arrow crosses month and year to Jan1');
 byId('doctorDayPrevious').fire('click');await tick();await tick();
 assert.equal(new URLSearchParams(calls.filter(c=>c.url.startsWith('/clinical/calendar/agenda?')).at(-1).url.split('?')[1]).get('start'),todayStart);
 byId('doctorDayNext').fire('click');await tick();await tick();byId('doctorDayToday').fire('click');await tick();await tick();
 assert.match(byId('doctorDayLabel').textContent,/Σήμερα/);
 const first=findButtonContent("Συνθετικός Ασθενής Α");
 assert.ok(first,"actual protected appointment is actionable");
 first.fire("click");
 assert.equal($("doctorPopover").hidden,true);
 assert.equal(windowStub.lastAppointment.patientId,null,"appointment name cannot link a patient");
 assert.equal(windowStub.lastAppointment.appointmentId,"cal.com:synthetic-1");
 assert.equal(windowStub.lastAppointment.date,"2026-10-11","appointment selection carries Cyprus day across UTC midnight");

 let pendingReads=0;
 windowStub.CockpitStoredChart={renderUpdates(){pendingReads++;}};
 $("doctorInboxButton").fire("click");
 assert.equal(pendingReads,0,"stored obligations cannot substitute for incoming source updates");
 assert.equal($("doctorInboxBadge").hidden,true);
 assert.ok($("doctorPopoverContent").children.some(x=>x.textContent?.includes("Δεν έχει συνδεθεί")));
 const incomingRows=$("doctorPopoverContent").children.length;
 windowHandlers.get("cockpit:clinical-updates-changed")();
 assert.equal($("doctorPopoverContent").children.length,incomingRows,"refresh replaces rather than repeats incoming state");
 assert.equal(pendingReads,0);
 $("doctorInboxButton").fire("click");
 assert.equal($("doctorPopover").hidden,true);

 windowStub.CockpitStoredChart=null;
 windowHandlers.get("cockpit:surgery-counts")({detail:{total:2,undated:1,invalid:0,overdue:0,
  upcoming:[{date:"2099-11-01",count:1,name:"Συνθετικός Ασθενής Μ",procedure:"Αρθροσκόπηση"}]}});
 assert.equal($("doctorTasksBadge").hidden,true,
   "surgery totals must no longer masquerade as patient tasks");
 windowHandlers.get("cockpit:ccc01-attention-count")({detail:{total:2}});
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
 let resolveOld;
 deferred={promise:new Promise(done=>{resolveOld=done;})};
 $('doctorDateButton').fire('click');
 deferred=null;byId('doctorDayNext').fire('click');await tick();await tick();
 resolveOld({ok:true,status:200,json:async()=>[{...mockAppointments[0],patient_display_name:'STALE_TODAY'}]});await tick();await tick();
 assert.match(byId('doctorDayLabel').textContent,/Αύριο/);assert.ok(!findButtonContent('STALE_TODAY'));
 $('doctorDateButton').fire('click');
 let resolve;
 deferred={promise:new Promise(done=>{resolve=done;})};
 $("doctorDateButton").fire("click");
 $("doctorInboxButton").fire("click");
 const inboxState=$("doctorPopoverContent").children.length;
 resolve({ok:true,status:200,json:async()=>mockAppointments});
 await tick();await tick();
 assert.equal($("doctorPopover").dataset.kind,"inbox","late day response cannot replace Inbox");
 assert.equal($("doctorPopoverContent").children.length,inboxState);
 assert.ok(calls.every(x=>x.method==="GET" || (x.method==="POST"
   && x.url==="/clinical/clinic-utilities/rf/api/outcomes/search")),
   "the RF search is the only new read-intent POST");
 assert.ok(calls.every(x=>!x.url.includes("patient_id")),"no patient identifiers in URLs");
 console.log("PASS Cockpit Doctor Shell: contextual day, Peek route, Inbox truth, tasks, actions, library, stale isolation");
})().catch(e=>{console.error(e);process.exitCode=1;});
