"use strict";
/* Synthetic actual-controller test: no PHI, network, provider or browser storage. */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const js=fs.readFileSync("static/cockpit/visit-brief.js","utf8");
const nodes=new Map();
const emitted=[];
const requests=[];
let readHandler=null;
let confirmHandler=null;
let unlinkHandler=null;

function element(id=""){
  const handlers={};
  const el={
    id,hidden:false,disabled:false,checked:false,dataset:{},textContent:"",
    children:[],className:"",
    addEventListener(kind,cb){handlers[kind]=cb;},
    fire(kind){assert.ok(handlers[kind],id+" has no "+kind);return handlers[kind]();},
    click(){return el.fire("click");},
    append(...nodes){el.children.push(...nodes);},
    replaceChildren(...parts){el.children=[...parts];},
    focus(){},setAttribute(){},
  };
  return el;
}
const get=id=>{
  if(!nodes.has(id))nodes.set(id,element(id));
  return nodes.get(id);
};
get("visitBriefOverlay").hidden=true;
get("visitBookingConfirmation").hidden=true;
get("visitBriefClinical").hidden=true;
get("visitBriefChoosePatient").hidden=true;
get("visitBriefOpenChart").hidden=true;
get("visitBriefUnlink").hidden=true;
const listeners={};
const win={
  addEventListener(kind,cb){listeners[kind]=cb;},
  dispatchEvent(event){emitted.push(event.type);}
};
const resp=data=>({ok:true,status:200,json:async()=>data});
const reject=status=>({ok:false,status});
const booking=uid=>({
  booking_uid:uid,patient_display_name:"Συνθετικός Ασθενής",
  start_at:"2026-10-16T14:00:00+03:00",clinic:"Limassol",reason:"Επανεξέταση"
});
const bookingOnly=(uid,linked=false,revision=0)=>({
  mode:"booking_only",reason:linked?"needs_reconfirmation":"unlinked",
  booking:booking(uid),revision,linked
});
const confirmed=uid=>({
  mode:"confirmed_once",booking:booking(uid),revision:1,
  brief:{
    patient_id:"P1",selected_booking_date:"2026-10-16",check_heading:"Σήμερα",
    previous_visit:{date:"2026-10-15",reason:"Αξιολόγηση",findings:["Προηγούμενο εύρημα"],
      source_encounter_id:"e1",signed_at:"2026-10-15T08:00:00+00:00"},
    prior_plan:{decisions:["Επανέλεγχος"],medications:[],
      next_contact:{when:"2026-10-16",purpose:"Έλεγχος"}},
    historical_plan:null,
    pending:[{description:"Εργαστηριακό",status:"awaiting_result",review_date:"2026-10-16"}],
    appointment_checks:[{text:"Έλεγχος κίνησης",kind:"previous_next_contact"}],
    pending_truth:"last_recorded_at_read",partial:false,
  },
});
let linked=false;
const fetch=async(url,options)=>{
  assert.ok(url.startsWith("/clinical/calendar/visit-brief/"));
  assert.equal(options.credentials,"same-origin");
  assert.equal(options.method,"POST");
  assert.equal(options.cache,"no-store");
  const action=url.split("/").at(-1),body=JSON.parse(options.body);
  requests.push({action,body});
  if(action==="read")return resp(await (readHandler?readHandler(body):bookingOnly(body.booking_uid,linked,linked?1:0)));
  if(action==="confirm")return resp(await (confirmHandler?confirmHandler(body):confirmed(body.booking_uid)));
  if(action==="unlink")return resp(await (unlinkHandler?unlinkHandler(body):{
    mode:"booking_only",reason:"unlinked",revision:2
  }));
  throw Error("Unknown action "+action);
};
const context={document:{getElementById:get,createElement:()=>element()},window:win,Event:class Event{constructor(type){this.type=type}},fetch,console};
vm.runInNewContext(js,context,{filename:"visit-brief.js"});
const controller=win.CockpitVisitBrief;
assert.equal(typeof controller.open,"function");
const tick=()=>new Promise(ok=>setImmediate(ok));
const view=uid=>({uid,name:"Συνθετικός Ασθενής",time:"14:00",
  reason:"Επανεξέταση",freshness:"Πρόσφατη πηγή"});

(async()=>{
  // Unlinked: first open shows context without protected chart or prefetch.
  await controller.open(view("UID-1"));
  assert.equal(get("visitBriefOverlay").hidden,false);
  assert.equal(get("visitBriefClinical").hidden,true);
  assert.equal(get("visitBriefChoosePatient").disabled,false);
  assert.equal(requests.filter(x=>x.action==="read").length,1);
  assert.equal(requests.filter(x=>x.action==="confirm").length,0);
  const context=controller.selectionContext();
  assert.equal(context.uid,"UID-1");
  controller.closed();
  controller.activateSelection(context);
  controller.selectPatient({patientId:"P1",name:"Πρόσωπο P1"});
  assert.equal(get("visitBookingConfirmation").hidden,false);
  await get("visitConfirmBooking").fire("click");
  assert.equal(requests.filter(x=>x.action==="confirm").length,0,
    "No clinical access without explicit independent verification");
  get("visitBookingIndependentCheck").checked=true;
  await get("visitConfirmBooking").fire("click");
  assert.equal(requests.filter(x=>x.action==="confirm").length,1);
  assert.equal(get("visitBriefClinical").hidden,false);
  assert.equal(get("visitBriefOpenChart").hidden,false);
  assert.equal(get("visitBriefPrevious").children[0].children[0].textContent.includes("Αξιολόγηση"),true);
  assert.equal(get("visitBriefChecks").children[0].children[0].textContent,"Έλεγχος κίνησης");

  // A subsequent same-UID read MUST NOT obtain a reusable patient grant.
  linked=true;
  await controller.open(view("UID-1"));
  assert.equal(get("visitBriefClinical").hidden,true);
  assert.match(get("visitBriefIdentity").textContent,/νέα ανεξάρτητη/);
  assert.equal(requests.filter(x=>x.action==="confirm").length,1);

  // UID A slow response cannot revive protected content after switching to B.
  let finishA;
  readHandler=body=>body.booking_uid==="UID-A"?
    new Promise(resolve=>{finishA=resolve}):bookingOnly(body.booking_uid);
  const pendingA=controller.open(view("UID-A"));
  await tick();
  await controller.open(view("UID-B"));
  finishA(bookingOnly("UID-A",true,8));
  await pendingA;
  assert.equal(get("visitBriefName").textContent,"Συνθετικός Ασθενής");
  assert.equal(controller.selectionContext().uid,"UID-B");
  assert.equal(get("visitBriefClinical").hidden,true);

  // An old CONFIRM response arriving after navigation cannot populate B.
  readHandler=null;
  await controller.open(view("UID-1"));
  const pending=controller.selectionContext();
  controller.closed();controller.activateSelection(pending);
  controller.selectPatient({patientId:"P1",name:"P1"});
  get("visitBookingIndependentCheck").checked=true;
  let finishConfirm;
  confirmHandler=body=>new Promise(resolve=>{finishConfirm=resolve});
  const req=get("visitConfirmBooking").fire("click");
  await tick();
  controller.invalidate();
  await controller.open(view("UID-B"));
  finishConfirm(confirmed("UID-1"));
  await req;
  assert.equal(get("visitBriefClinical").hidden,true);
  assert.equal(get("visitBriefOpenChart").hidden,true);
  assert.equal(controller.selectionContext().uid,"UID-B");

  // 401 immediately clears patient state, requiring the shared Clinical Key.
  readHandler=()=>reject(401);confirmHandler=null;
  await controller.open(view("UID-1"));
  assert.equal(get("visitBriefClinical").hidden,true);
  assert.ok(emitted.includes("clinical:request-login"));
  assert.equal(controller.selectionContext(),null);

  // Every patient action is directly scoped and deliberate, never a URL token.
  assert.ok(requests.every(x=>!Object.keys(x.body).some(k=>/token|name|phone/i.test(k))));
  assert.doesNotMatch(js,/localStorage|sessionStorage/);
  console.log("PASS Visit Brief: explicit per-open verification, clinical evidence, stale responses, 401 and privacy");
})().catch(e=>{console.error(e);process.exitCode=1;});
