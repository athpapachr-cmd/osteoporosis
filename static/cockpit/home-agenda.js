(() => {
  "use strict";
  // Read-only, start-in-day agenda. Never treat elapsed time as attendance or
  // infer availability outside the bounded source projection.
  const root = document.getElementById("homeAgendaRows");
  if (!root) return;
  const zone = "Asia/Nicosia";
  const dayFmt = new Intl.DateTimeFormat("en-CA", {timeZone:zone,year:"numeric",month:"2-digit",day:"2-digit"});
  const clockFmt = new Intl.DateTimeFormat("el-CY", {timeZone:zone,hour:"2-digit",minute:"2-digit",hour12:false});
  const dateFmt = new Intl.DateTimeFormat("el-CY", {timeZone:zone,weekday:"short",day:"numeric",month:"short"});
  const fullFmt = new Intl.DateTimeFormat("en-GB", {timeZone:zone,year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hourCycle:"h23"});
  const url = path => (window.CockpitClinicalBase || "/clinical") + path.slice(9);
  const parts = date => Object.fromEntries(fullFmt.formatToParts(date).filter(x=>x.type!=="literal").map(x=>[x.type,Number(x.value)]));
  function midnight(y,m,d) {
    const probe = new Date(Date.UTC(y,m-1,d)), p = parts(probe);
    return new Date(Date.UTC(y,m-1,d) - (Date.UTC(p.year,p.month-1,p.day,p.hour,p.minute,p.second) - probe.getTime()));
  }
  function tomorrowRange(now) {
    const p=parts(new Date(now)), t=new Date(Date.UTC(p.year,p.month-1,p.day+1));
    const n=new Date(Date.UTC(t.getUTCFullYear(),t.getUTCMonth(),t.getUTCDate()+1));
    return {key:t.toISOString().slice(0,10), label:dateFmt.format(new Date(t.getTime()+12*3600000)),
      start:midnight(t.getUTCFullYear(),t.getUTCMonth()+1,t.getUTCDate()).toISOString(),
      end:midnight(n.getUTCFullYear(),n.getUTCMonth()+1,n.getUTCDate()).toISOString()};
  }
  const dayKey = now => dayFmt.format(new Date(now));
  function parseDate(value) {
    if (typeof value !== "string" || !value) return null;
    const n = Date.parse(/(?:Z|[+-]\d\d:?\d\d)$/.test(value) ? value : value + "Z");
    return Number.isFinite(n) ? n : null;
  }
  function tidy(value) {
    const s=String(value || "");
    return s && s === s.toLocaleUpperCase("el") && s !== s.toLocaleLowerCase("el")
      ? s.toLocaleLowerCase("el").replace(/(^|[\s-])(\p{L})/gu,(_m,a,b)=>a+b.toLocaleUpperCase("el")) : s;
  }
  const norm = s => String(s || "").normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLocaleLowerCase("el").trim();
  function cleanReason(reason, name) {
    let value=String(reason || "").replace(/\s+/g," ").trim();
    // Remove only a matching final patient-name suffix; keep all other clinical text.
    const match=value.match(/\s*Ραντεβού\s+για\s*:\s*(.+)$/i);
    if (match && norm(match[1]) === norm(name)) value=value.slice(0,match.index).replace(/[·\s-]+$/,"").trim();
    return value;
  }
  function normalise(row,index) {
    if (!row || typeof row !== "object") return {raw:{},start:null,end:null,index,valid:false};
    const start=parseDate(row.start_at),end=parseDate(row.end_at);
    return {raw:row,start,end,index,valid:start!==null && end!==null && end>start};
  }
  function classify(rows,now) {
    const all=rows.map(normalise), valid=all.filter(x=>x.valid).sort((a,b)=>a.start-b.start||a.index-b.index);
    return {past:valid.filter(x=>x.end<=now),current:valid.filter(x=>x.start<=now&&now<x.end),
      future:valid.filter(x=>x.start>now),unknown:all.filter(x=>!x.valid)};
  }
  const state={rows:null,day:null,checked:0,generation:0,tomorrow:null,pastOpen:false,tomorrowOpen:false, reload:null,surgery:null};
  const focusKeys=new WeakMap(); let targets=new Map();
  function el(tag,cls,value) {const e=document.createElement(tag); if(cls)e.className=cls;if(value!==undefined)e.textContent=value;return e;}
  function track(e,key){if(key){focusKeys.set(e,key);targets.set(key,e);}return e;}
  function button(label,cls,click,key){const b=track(el("button",cls,label),key);b.type="button";b.addEventListener("click",click);return b;}
  function clock(x){return x===null?"—":clockFmt.format(new Date(x));}
  function clinic(value){const n=norm(value);return ["evrychou","evrichou","ευρυχου"].includes(n)?"Ευρύχου":["limassol","λεμεσος"].includes(n)?"Λεμεσός":String(value||"");}
  function row(item,style="") {
    const r=item.raw, rawName=String(r.patient_display_name || "Ραντεβού χωρίς καταχωρισμένο όνομα"), rawReason=String(r.reason || r.category || "Ραντεβού");
    const b=button("","hg-row "+style,()=>{
      const view={uid:r.appointment_id,name:rawName,reason:rawReason,time:clock(item.start),
        date:item.start===null?null:dayKey(item.start),freshness:""};
      if(window.CockpitVisitBrief?.open) window.CockpitVisitBrief.open(view);
      else window.CockpitHome?.openAppointment?.({kind:"appointment",name:view.name,reason:view.reason,
        time:view.time,date:view.date,appointmentId:view.uid,patientId:null});
    },"appointment:"+String(r.appointment_id||"")+":"+String(item.start)+":"+item.index+":"+style);
    if (typeof r.appointment_id !== "string" || !r.appointment_id.trim()) {
      b.disabled=true; b.title="Δεν υπάρχει επιβεβαιωμένο αναγνωριστικό ραντεβού";
    }
    const body=el("span","hg-text");body.append(el("strong","hg-name",tidy(rawName)));
    const why=cleanReason(r.reason,rawName); if(why) {const line=el("span","hg-why",why);line.title=String(r.reason);body.append(line);}
    b.append(el("span","hg-time",clock(item.start)),body,el("span","hg-right",clinic(r.clinic) || "›"));return b;
  }
  function clear(){state.generation++;state.rows=null;state.day=null;state.checked=0;state.tomorrow=null;state.surgery=null;state.pastOpen=false;state.tomorrowOpen=false;}
  function readTomorrow(info) {
    const gen=state.generation;state.tomorrow={key:info.key,info,loading:true,rows:null};
    const q=new URLSearchParams({start:info.start,end:info.end});
    fetch(url("/clinical/calendar/agenda?"+q),{credentials:"same-origin",cache:"no-store",headers:{Accept:"application/json"}})
      .then(async response=>{
        if(gen!==state.generation || !state.rows) return;
        if(response.status===401||response.status===403){window.dispatchEvent(new Event("clinical:unauthenticated"));return;}
        if(!response.ok)throw Error("unavailable");
        const data=await response.json();if(gen!==state.generation||!state.rows)return;
        if(!Array.isArray(data))throw Error("invalid");
        state.tomorrow={key:info.key,info,loading:false,rows:data.map(normalise).sort((a,b)=>(a.start??Infinity)-(b.start??Infinity))};draw();
      }).catch(()=>{if(gen===state.generation&&state.rows){state.tomorrow={key:info.key,info,loading:false,rows:null};draw();}});
  }
  function tomorrowBlock(info){
    if(!state.tomorrow || state.tomorrow.key!==info.key)readTomorrow(info);
    const t=state.tomorrow, box=el("section","hg-tomorrow");box.append(el("h3","hg-label","Αύριο · "+info.label));
    if(t.loading)box.append(el("p","hg-empty","Φόρτωση αυριανού προγράμματος…"));
    else if(!t.rows)box.append(el("p","hg-empty","Το αυριανό πρόγραμμα δεν είναι διαθέσιμο. Δεν σημαίνει ότι δεν υπάρχουν ραντεβού."));
    else {
      const surgeries=state.surgery && Array.isArray(state.surgery.attention) ? state.surgery.attention.filter(x=>x.date===info.key) : [];
      const items=t.rows.map(x=>({sort:x.start===null?"99:99":clock(x.start),node:()=>row(x,"tomorrow")}));
      surgeries.forEach((s,i)=>{const time=/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(s.time)?s.time:"";
        items.push({sort:time||"99:99",node:()=>button((time?time+" · ":"")+tidy(s.name)+" · Χειρουργείο", "hg-surgery",()=>window.CockpitHomeAttentionNavigation?.cases?.("surgery"),"surgery:"+i)});});
      items.sort((a,b)=>a.sort.localeCompare(b.sort));const shown=state.tomorrowOpen?items:items.slice(0,4);shown.forEach(x=>box.append(x.node()));
      if(!t.rows.length)box.append(el("p","hg-empty","Δεν βρέθηκαν ραντεβού αύριο στη διαθέσιμη πηγή."));
      if(!state.surgery)box.append(el("p","hg-empty","Τα αυριανά χειρουργεία δεν επιβεβαιώθηκαν εδώ."));
      if(items.length>4){const more=button(state.tomorrowOpen?"Λιγότερα":"Δες και τα "+items.length,"hg-more",()=>{state.tomorrowOpen=!state.tomorrowOpen;draw();},"tomorrow-more");more.setAttribute("aria-expanded",String(state.tomorrowOpen));box.append(more);}
    }
    return box;
  }
  function draw(){
    if(!state.rows)return;
    const active=document.activeElement, previous=active?focusKeys.get(active):null, inside=!!(active&&root.contains(active));targets=new Map();
    const priorScroll=root.querySelector(".hg-list")?.scrollTop||0, now=Date.now(), view=classify(state.rows,now);
    const wrap=el("div","hg-agenda"),heading=track(el("h3","hg-label","Ραντεβού με έναρξη σήμερα"),"heading");heading.tabIndex=-1;wrap.append(heading);
    if(view.past.length){const past=button(view.past.length+" περασμένα ραντεβού", "hg-more",()=>{state.pastOpen=!state.pastOpen;draw();},"past");past.setAttribute("aria-expanded",String(state.pastOpen));wrap.append(past);if(state.pastOpen)view.past.forEach(x=>wrap.append(row(x,"past")));}
    if(view.current.length){wrap.append(el("p","hg-label",view.current.length>1?"Τώρα · ταυτόχρονα ραντεβού":"Τώρα"));view.current.forEach(x=>wrap.append(row(x,"current")));}
    if(view.future.length){const [next,...rest]=view.future;const minutes=Math.max(0,Math.ceil((next.start-now)/60000));
      if(!view.current.length)wrap.append(el("p","hg-now",clock(now)+" · Επόμενο με επιβεβαιωμένη ώρα σε "+minutes+" λεπτά"));
      wrap.append(row(next,"next"));if(rest.length){const list=el("div","hg-list");list.tabIndex=0;list.setAttribute("aria-label","Υπόλοιπα σημερινά ραντεβού");rest.forEach(x=>list.append(row(x)));wrap.append(list);}}
    if(view.unknown.length){wrap.append(el("p","hg-warning","Ραντεβού με μη επιβεβαιωμένη ώρα: "+view.unknown.length));view.unknown.forEach(x=>wrap.append(row(x,"unknown")));}
    if(!view.current.length&&!view.future.length&&!view.unknown.length){
      wrap.append(el("p","hg-empty",state.rows.length?"Δεν απομένουν άλλα ραντεβού με έναρξη σήμερα στη διαθέσιμη πηγή.":"Δεν βρέθηκαν ραντεβού με έναρξη σήμερα στη διαθέσιμη πηγή."));
      wrap.append(tomorrowBlock(tomorrowRange(now)));
    }
    root.replaceChildren(wrap);const list=root.querySelector(".hg-list");if(list)list.scrollTop=priorScroll;
    const target=(previous&&targets.get(previous))||(inside&&targets.get("heading"));target?.focus?.({preventScroll:true});
  }
  function validSurgery(d){return !!d && !d.unavailable && Array.isArray(d.attention) && d.attention.every(x=>x && typeof x.date === "string" && typeof x.name === "string" && typeof x.procedure === "string" && typeof x.time === "string");}
  function render(rows,options){clear();state.rows=rows;state.day=options.day;state.checked=Date.now();state.reload=options.reload;state.surgery=validSurgery(window.CockpitSurgerySummary)?window.CockpitSurgerySummary:null;draw();}
  function tick(){if(!state.rows||document.hidden)return;if(dayKey(Date.now())!==state.day||Date.now()-state.checked>=300000){state.reload?.();return;}draw();}
  window.addEventListener("clinical:unauthenticated",clear);
  window.addEventListener("cockpit:surgery-counts",e=>{if(state.rows){state.surgery=validSurgery(e.detail)?e.detail:null;if(state.tomorrow)draw();}});
  document.addEventListener("visibilitychange",()=>{if(!document.hidden)tick();});window.addEventListener("focus",tick);
  window.setInterval(tick,30000);
  window.CockpitHomeAgenda={render,clear,classify,tomorrowRange,cleanReason};
})();
