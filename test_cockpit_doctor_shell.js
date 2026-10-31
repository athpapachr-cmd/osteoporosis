"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const html=fs.readFileSync("static/cockpit/index.html","utf8");
const script=fs.readFileSync("static/cockpit/doctor-shell.js","utf8");
assert.match(html,/id="doctorDateButton"/);
assert.match(html,/id="doctorInboxButton"/);
assert.match(html,/id="doctorCallsButton"/);
assert.match(html,/id="doctorTasksButton"/);
assert.match(html,/id="doctorNotionButton"/);
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
  replaceChildren(...children){obj.children=[...children];},
  append(...children){obj.children.push(...children);},
  focus(){},
  contains(target){return target===obj||obj.children.some(c=>c===target||typeof c.contains==="function"&&c.contains(target));}
 };
 return obj;
}
const tabs=["cases","documents","modules","learning"].map(c=>{
 const b=el("tab-"+c,"button");b.dataset.doctorLibraryTab=c;return b;
});
const panels=["cases","documents","modules","learning"].map(c=>{
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
$("doctorPopover").hidden=true;$("doctorLibraryView").hidden=true;$("doctorObligationsView").hidden=true;
$("doctorInboxBadge").hidden=true;$("doctorTasksBadge").hidden=true;
const windowHandlers=new Map(),windowStub={
 CockpitHome:{openAppointment(row){windowStub.lastAppointment=row;}},
 crypto:{randomUUID(){return "00000000-0000-4000-8000-000000000010";}},
 addEventListener(name,fn){windowHandlers.set(name,fn);},
 CockpitSurgerySummary:null
};
const navigatorStub={mediaDevices:{async getUserMedia(){return {getTracks(){return [{stop(){}}];}};}}};
class SyntheticRecorder{
 static isTypeSupported(type){return type==="audio/webm";}
 constructor(){this.state="inactive";}
 start(){this.state="recording";}
 stop(){this.state="inactive";this.ondataavailable({data:new Blob(["x".repeat(100)],{type:"audio/webm"})});this.onstop();}
}
windowStub.MediaRecorder=SyntheticRecorder;
let deferred=null,deferredVoice=null,authenticated=true,callHandled=false,notionDone=false,notionVoice=false;
const notionTask={id:"00000000-0000-4000-8000-000000000001",title:"Συνθετική εργασία",
 status:"Not started",revision:"2026-10-25T08:00:00.000Z",due:"2026-10-30",
 url:"https://www.notion.so/00000000000040008000000000000001"};
const mockAppointments=[
 {appointment_id:"cal.com:synthetic-1",patient_display_name:"Συνθετικός Ασθενής Α",
  start_at:"2026-10-10T22:20:00Z",reason:"Συνθετική οστεοπόρωση"},
 {appointment_id:"cal.com:synthetic-2",patient_display_name:"Συνθετικός Ασθενής Β",
  start_at:"2026-10-11T08:40:00Z",reason:"Prolia"}
];
async function fakeFetch(url,opts){
 calls.push({url,method:opts.method||"GET",options:opts});
 if(url==="/clinical/status"){
   assert.equal(opts.credentials,"same-origin");
   assert.equal(opts.cache,"no-store");
   return {ok:authenticated,status:authenticated?200:401,
     json:async()=>({protected:authenticated})};
 }
 if(url==="/clinical/clinic-utilities/rf/api/outcomes/search"){
   assert.equal(opts.method,"POST","RF endpoint is read-intent POST only");
   assert.deepEqual(JSON.parse(opts.body),{query:"",scope:"all",page:1});
   return {ok:true,status:200,json:async()=>({
     denominator:5,scope:"all",page:1,truncated:false,applications:[],
     counts:{issued_unknown:1,pending:2,approved:1,rejected:0,scheduled:1,performed:1,mine_actionable:1}
   })};
 }
 if(url.startsWith("/clinical/clinical-updates?")){
   assert.equal(opts.method||"GET","GET");
   assert.equal(opts.cache,"no-store");
   return {ok:authenticated,status:authenticated?200:401,json:async()=>({
     items:[0,1,2,3,4].map(i=>({
       pending_id:"SYN-PENDING-"+i,patient_id:"SYN-PATIENT-"+i,
       patient_name:"Συνθετικός πρόσφατος "+i,description:"Συνθετική υποχρέωση "+i,
       status:"open",updated_at:"2026-10-"+String(10+i).padStart(2,"0")+"T08:00:00Z"
     })).concat([{pending_id:"DONE",patient_id:"SYN-PATIENT-9",
       patient_name:"Παλιός ασθενής",description:"Ολοκληρωμένη υποχρέωση",
       status:"resolved",updated_at:"2026-10-24T18:00:00Z"}])
   })};
 }
 if(url==="/clinical/reception-calls/peek"){
   assert.equal(opts.cache,"no-store");
   return {ok:authenticated,status:authenticated?200:401,json:async()=>({
     source:"reception_dashboard",is_unread_count:false,
     coverage:"available_call_log_max_200",unhandled_count:callHandled?0:1,calls:[{
       call_id:"RM-TEST",todo_id:"td_RM-TEST",handled:callHandled,
       logged_at:"2026-10-25T07:00:00+02:00",
       name:"Συνθετικός καλών",phone:"0000000000",
       summary:"Ζήτησε τηλεφώνημα",summary_pending:false,
       kind:"message"
     }]
   })};
 }
 if(url==="/clinical/reception-calls/handled"){
   assert.equal(opts.method,"POST");
   assert.deepEqual(JSON.parse(opts.body),{todo_id:"td_RM-TEST",handled:true});
   callHandled=true;
   return {ok:true,status:200,json:async()=>({handled:true})};
 }
 if(url==="/clinical/notion-tasks/peek"){
   return {ok:authenticated,status:authenticated?200:401,json:async()=>({
     source:"notion",database_url:"https://app.notion.com/p/241379c6877a40fc8ab4a0efc042c887",
     voice_available:notionVoice,tasks:notionDone?[]:[notionTask]})};
 }
 if(url==="/clinical/notion-tasks/create"){
   assert.equal(opts.method,"POST");
   const body=JSON.parse(opts.body);
   assert.equal(body.title,"Συνθετική νέα εργασία");
   assert.equal(body.notes,"Συνθετική σημείωση");
   assert.equal(body.due_date,"2026-10-30");
   assert.equal(body.category,"Ιατρείο");
   assert.equal(body.priority,"Κανονική");
   return {ok:true,status:200,json:async()=>({source:"notion",created:true,
     task_url:"https://www.notion.so/00000000000040008000000000000002"})};
 }
 if(url==="/clinical/notion-tasks/propose-new"){
   assert.equal(opts.method,"POST");
   assert.equal(opts.headers["Content-Type"],"audio/webm");
   if(deferredVoice)return deferredVoice;
   return {ok:true,status:200,json:async()=>({attempt_id:"00000000-0000-4000-8000-000000000011",
     transcript:"Συνθετική νέα εργασία μέχρι τις 30 Οκτωβρίου",
     suggestion:{title:"Συνθετική νέα εργασία",notes:"Συνθετική σημείωση",
       due_date:"2026-10-30",category:"Ιατρείο",priority:"Κανονική"}})};
 }
 if(url==="/clinical/notion-tasks/confirm"){
   assert.equal(opts.method,"POST");
   const body=JSON.parse(opts.body);
   assert.equal(body.task_id,notionTask.id);
   assert.equal(body.revision,notionTask.revision);
   assert.equal(body.mark_done,true);
   assert.equal(body.completion_note,"Συνθετική ολοκλήρωση");
   notionDone=true;
   return {ok:true,status:200,json:async()=>({source:"notion",updated:true,
     followup_created:false,task_url:notionTask.url})};
 }
 if(!url.startsWith("/clinical/calendar/agenda?"))throw Error("unexpected URL "+url);
 if(deferred)return deferred.promise;
 return {ok:authenticated,status:authenticated?200:401,json:async()=>mockAppointments};
}
let voiceClock=0;
class CalendarClock extends Date{
 constructor(...args){super(...(args.length?args:['2026-12-31T12:00:00Z']));}
 static now(){return voiceClock?voiceClock+=2500:Date.now();}
}
vm.runInNewContext(script.replace("  function text(message) {","  window.dayRangeForTest=todayRange;\n  function text(message) {"),{document:doc,window:windowStub,fetch:fakeFetch,
 URL,URLSearchParams,Date:CalendarClock,Intl,console,navigator:navigatorStub,Blob,setTimeout,clearTimeout});
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function findButtonContent(title) {
 return $("doctorPopoverContent").children.find(item=>
   item.tagName==="button" && item.children?.some(c=>
     c.children?.some(n=>n.textContent?.includes(title))));
}
function byId(id,node=$('doctorPopoverContent')){if(node.id===id)return node;for(const child of node.children||[]){const found=byId(id,child);if(found)return found;}return null;}
function findTextNode(value,node=$('doctorPopoverContent')){
 if(node.textContent===value)return node;
 for(const child of node.children||[]){const found=findTextNode(value,child);if(found)return found;}
 return null;
}
function findAction(label,node=$('doctorPopoverContent')){
 if(node.tagName==="button" && node.children?.some(c=>
   c.children?.some(n=>n.textContent===label)))return node;
 for(const child of node.children||[]){const found=findAction(label,child);if(found)return found;}
 return null;
}
(async()=>{
 for(const [day,hours] of [['2026-03-29',23],['2026-10-25',25],['2026-12-31',24]]){
   const range=windowStub.dayRangeForTest(new Date(day+'T12:00:00Z'));
   assert.equal((new Date(range.end)-new Date(range.start))/3600000,hours,'Cyprus calendar boundaries '+day);
 }

 assert.equal($("doctorLibraryView").hidden,true);
 $("doctorCallsButton").fire("click");await tick();await tick();
 assert.equal($("doctorPopover").hidden,false,"phone popover opens");
 assert.equal($("doctorCallsButton").attributes["aria-expanded"],"true");
 const entry=$("doctorPopoverContent").children.find(n=>n.className==="doctor-call-entry");
 const callLink=entry.children[0];
 assert.match(callLink.children[0].children[0].textContent,/Συνθετικός καλών/);
 assert.doesNotMatch(callLink.children[0].children[1].textContent,/Χρειάζεται προσοχή/);
 assert.match(callLink.children[0].children[1].textContent,/Μήνυμα/);
 assert.match($("doctorPopoverContent").children[0].textContent,/μη σημειωμένες/);
 const complete=entry.children[1].children.find(n=>n.tagName==="button");
 assert.ok(complete,"explicit Reception completion action");
 await complete.fire("click");await tick();await tick();
 assert.equal(callHandled,true);
 assert.ok(calls.some(c=>c.url==="/clinical/reception-calls/handled"&&c.method==="POST"));
 assert.ok(calls.some(c=>c.url==="/clinical/reception-calls/peek"),"protected call route");
 $("doctorCallsButton").fire("click");
 assert.equal($("doctorPopover").hidden,true);
 $("doctorNotionButton").fire("click");await tick();await tick();
 assert.equal($("doctorPopover").hidden,false);
 assert.ok(findTextNode("Συνθετική εργασία"),"actual Notion row shown");
 assert.ok(!findTextNode("Υπαγόρευση"),"voice hidden without separate provider gate");
 const notionEntry=$("doctorPopoverContent").children.find(n=>n.className==="doctor-notion-entry");
 notionEntry.children[1].children[0].fire("click");
 const form=$("doctorPopoverContent").children.find(n=>n.className==="doctor-notion-form");
 form.children[0].children[0].checked=true;
 form.children[1].children[1].value="Συνθετική ολοκλήρωση";
 form.children.at(-1).children[1].fire("click");await tick();await tick();
 assert.equal(notionDone,true,"explicit confirmation updates Notion source");
 assert.ok($("doctorPopoverContent").children.some(n=>n.textContent?.includes("Δεν υπάρχουν ενεργές")));
 assert.ok(findAction("Νέα εργασία"),"general create remains above an empty task list");
 findAction("Νέα εργασία").fire("click");
 const newForm=$("doctorPopoverContent").children.find(n=>n.className==="doctor-notion-form");
 assert.ok(newForm,"new task review opens without selecting an existing row");
 newForm.children[0].children[1].value="Συνθετική νέα εργασία";
 newForm.children[1].children[1].value="Συνθετική σημείωση";
 newForm.children[2].children[1].value="2026-10-30";
 newForm.children[3].children[1].value="Ιατρείο";
 newForm.children[4].children[1].value="Κανονική";
 assert.equal(calls.filter(c=>c.url==="/clinical/notion-tasks/create").length,0,
   "editing is not a Notion write");
 await newForm.children.at(-1).children[1].fire("click");await tick();
 assert.equal(calls.filter(c=>c.url==="/clinical/notion-tasks/create").length,1,
   "one explicit confirmation creates a task");
 assert.ok(findTextNode("Η εργασία δημιουργήθηκε στο Notion"));
 $("doctorNotionButton").fire("click");
 notionVoice=true;
 $("doctorNotionButton").fire("click");await tick();await tick();
 assert.ok(findAction("Υπαγόρευση νέας"),"general dictation action is visible when voice is enabled");
 voiceClock=1;
 findAction("Υπαγόρευση νέας").fire("click");await tick();
 const stopVoice=$("doctorPopoverContent").children.find(n=>n.tagName==="button"&&
   n.textContent==="Σταμάτημα και πρόταση");
 assert.ok(stopVoice,"voice capture started for a new task");
 stopVoice.fire("click");await tick();await tick();
 const dictatedForm=$("doctorPopoverContent").children.find(n=>n.className==="doctor-notion-form");
 assert.equal(dictatedForm.children[0].children[1].value,"Συνθετική νέα εργασία");
 assert.equal(dictatedForm.children[1].children[1].value,"Συνθετική σημείωση");
 assert.equal(dictatedForm.children[2].children[1].value,"2026-10-30");
 assert.equal(dictatedForm.children[3].children[1].value,"Ιατρείο");
 assert.equal(dictatedForm.children[4].children[1].value,"Κανονική");
 assert.equal(calls.filter(c=>c.url==="/clinical/notion-tasks/create").length,1,
   "dictation prefills review without a second write");
 $("doctorNotionButton").fire("click");
 let trackStops=0;
 navigatorStub.mediaDevices.getUserMedia=async()=>({getTracks(){return [{stop(){trackStops++;}}];}});
 const proposalsBeforeFailure=calls.filter(c=>c.url==="/clinical/notion-tasks/propose-new").length;
 for(const Failure of [
   class ConstructorFailure extends SyntheticRecorder{constructor(){super();throw Error("synthetic");}},
   class StartFailure extends SyntheticRecorder{start(){throw Error("synthetic");}}
 ]){
   windowStub.MediaRecorder=Failure;
   $("doctorNotionButton").fire("click");await tick();await tick();
   findAction("Υπαγόρευση νέας").fire("click");await tick();
   assert.ok(findAction("Συμπλήρωση με κείμενο"),"recorder failure offers manual review");
   $("doctorNotionButton").fire("click");
 }
 assert.equal(trackStops,2,"acquired microphone tracks stop on constructor and start failure");
 assert.equal(calls.filter(c=>c.url==="/clinical/notion-tasks/propose-new").length,
   proposalsBeforeFailure,"recorder failure sends no audio");
 windowStub.MediaRecorder=SyntheticRecorder;
 let resolveStaleVoice;
 deferredVoice=new Promise(done=>{resolveStaleVoice=done;});
 $("doctorNotionButton").fire("click");await tick();await tick();
 findAction("Υπαγόρευση νέας").fire("click");await tick();
 $("doctorPopoverContent").children.find(n=>n.tagName==="button"&&
   n.textContent==="Σταμάτημα και πρόταση").fire("click");await tick();
 $("doctorNotionButton").fire("click");
 resolveStaleVoice({ok:true,status:200,json:async()=>({
   attempt_id:"00000000-0000-4000-8000-000000000012",transcript:"Συνθετική όψιμη πρόταση",
   suggestion:{title:"Συνθετική όψιμη εργασία",notes:"",due_date:"",category:"",priority:""}})});
 deferredVoice=null;await tick();await tick();
 assert.equal($("doctorPopover").hidden,true,"late new-task proposal cannot reopen a closed panel");
 assert.equal($("doctorPopoverContent").children.length,0,"late proposal cannot expose task details");
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
 $("doctorInboxButton").fire("click");await tick();await tick();
 assert.equal(pendingReads,0,"stored obligations cannot substitute for incoming source updates");
 assert.equal($("doctorInboxBadge").hidden,true);
 assert.ok($("doctorPopoverContent").children.some(x=>x.textContent?.includes("PEEK_NETWORK")), "browser fetch failure cannot masquerade as disconnected iCloud");
 const incomingRows=$("doctorPopoverContent").children.length;
 windowHandlers.get("cockpit:clinical-updates-changed")();
 await tick();await tick(); // Protected status revalidation is asynchronous.
 assert.equal($("doctorPopoverContent").children.length,incomingRows,"refresh replaces rather than repeats incoming state");
 assert.equal(pendingReads,0);
 $("doctorInboxButton").fire("click");
 assert.equal($("doctorPopover").hidden,true);

 let obligationsOpened=0,obligationsCleared=0,focused=null;
 windowStub.CockpitObligations={open(fn,pendingId){obligationsOpened++;focused=pendingId;},close(){obligationsCleared++}};
 windowStub.CockpitStoredChart=null;
 windowHandlers.get("cockpit:surgery-counts")({detail:{total:2,undated:1,invalid:0,overdue:0,
   upcoming:[{date:"2099-11-01",count:1,name:"Συνθετικός Ασθενής Μ",procedure:"Αρθροσκόπηση"}]}});
 assert.equal($("doctorTasksBadge").hidden,true,"surgery count cannot masquerade as patient obligations");
 assert.equal($("doctorTasksBadge").textContent,"");
 windowStub.CockpitStoredChart={canLeave(){return false;}};
 $("doctorTasksButton").fire("click");await tick();await tick();
 assert.equal($("doctorPopover").hidden,false,"top-right icon opens a peek");
 assert.equal($("doctorPopover").dataset.kind,"tasks");
 assert.equal($("doctorObligationsView").hidden,true,"no automatic full-screen navigation");
 assert.equal($("doctorTasksButton").attributes["aria-expanded"],"true");
 const peekButtons=$("doctorPopoverContent").children.filter(x=>x.tagName==="button");
 assert.equal(peekButtons.length,5,"four most-recent active tasks and full-list action");
 assert.ok(findButtonContent("Συνθετικός πρόσφατος 4"));
 assert.ok(!findButtonContent("Συνθετικός πρόσφατος 0"),"only four newest");
 assert.ok(!findButtonContent("Παλιός ασθενής"),"resolved row cannot masquerade as active");
 findButtonContent("Συνθετικός πρόσφατος 4").fire("click");
 assert.equal(obligationsOpened,0,"unsaved draft blocks full-screen item navigation");
 assert.equal($("doctorPopover").hidden,false);
 windowStub.CockpitStoredChart={canLeave(){return true;}};
 findButtonContent("Συνθετικός πρόσφατος 4").fire("click");
 assert.equal(focused,"SYN-PENDING-4","preview routes to exact on-screen pending dialog");
 assert.equal($("doctorObligationsView").hidden,false);
 assert.equal($("doctorMainView").hidden,true);
 assert.equal($("doctorPopover").hidden,true);
 assert.equal(obligationsOpened,1);
 $("doctorObligationsBack").fire("click");
 assert.equal($("doctorMainView").hidden,false);
 assert.equal($("doctorObligationsView").hidden,true);
 assert.ok(obligationsCleared>0,"hiding the screen purges cached patient data");
 $("doctorTasksButton").fire("click");await tick();await tick();
 findButtonContent("Όλες οι υποχρεώσεις").fire("click");
 assert.equal(obligationsOpened,2);
 assert.equal(focused,null,"full list does not force an arbitrary patient");
 $("doctorObligationsBack").fire("click");
 $("doctorActionsButton").fire("click");
 const libraryAction=$("doctorPopoverContent").children[1]?.children?.at(-1);
 assert.equal(libraryAction?.children?.[0]?.children?.[0]?.textContent,
   "Περιστατικά ιατρείου","office-cases launcher remains in Actions");
 libraryAction.fire("click");
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
 assert.ok($("doctorPopoverContent").children.length>=inboxState,
    "Protected asynchronous inbox revalidation must not be overwritten by stale calendar");
 assert.ok(!$("doctorPopoverContent").children.some(x=>
    (x.textContent||"").includes("STALE_TODAY")),
    "Late calendar response must not replace inbox or render stale appointments");
 assert.ok(calls.every(x=>x.method==="GET" || (x.method==="POST"
   && ["/clinical/clinic-utilities/rf/api/outcomes/search",
       "/clinical/reception-calls/handled", "/clinical/notion-tasks/confirm",
       "/clinical/notion-tasks/create", "/clinical/notion-tasks/propose-new"].includes(x.url))),
   "only named source actions use POST");
 assert.ok(calls.every(x=>!x.url.includes("patient_id")),"no patient identifiers in URLs");
 console.log("PASS Cockpit Doctor Shell: contextual day, Peek route, Inbox truth, tasks, actions, library, stale isolation");
})().catch(e=>{console.error(e);process.exitCode=1;});
