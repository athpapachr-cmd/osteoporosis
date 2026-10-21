(() => {
  "use strict";
  // CCC-01: presentation only. Single protected /clinical/clinical-updates owner;
  // no local persistence, clinical writes, inferred external events or duplicate tasks.
  const $ = id => document.getElementById(id);
  const labels = {
    open:"Ανοικτή",awaiting_patient:"Αναμένεται από τον ασθενή",
    awaiting_result:"Αναμένεται αποτέλεσμα",resolved:"Ολοκληρώθηκε",
    cancelled:"Ακυρώθηκε"
  };
  const types = {
    investigation:"Εξέταση",test:"Εξέταση",result:"Αποτέλεσμα",
    document:"Έγγραφο",follow_up:"Επανεξέταση",
    referral:"Παραπομπή",other:"Άλλο"
  };
  let epoch=0,popupToken=0,fullToken=0,attentionToken=0,fullReady=false;
  let lastFull=null;
  const full={scope:"real",offset:0,limit:40,query:"",category:"",status:"",
    due:"all",sort:"updated",group:"patient",active:null,leave:null};
  function node(parent,tag,label="",css=""){
    const n=document.createElement(tag);
    if(label)n.textContent=label;
    if(css)n.className=css;
    parent.append(n);
    return n;
  }
  function button(parent,label,act,css=""){
    const b=node(parent,"button",label,css);
    b.type="button";b.addEventListener("click",act);return b;
  }
  function statusText(item){return labels[item.status]||item.status||"Άγνωστη";}
  function validDay(value){
    if(typeof value!=="string"||! /^\d{4}-\d{2}-\d{2}$/.test(value))return null;
    const d=new Date(value+"T12:00:00Z");
    return Number.isFinite(d.getTime())&&d.toISOString().slice(0,10)===value?value:null;
  }
  function dueBucket(item,today){
    const d=validDay(item.review_date);
    if(!d)return "undated";
    if(d<today)return "overdue";
    if(d===today)return "today";
    return "future";
  }
  function bucketName(v){
    return {overdue:"Εκπρόθεσμες",today:"Σήμερα",
      future:"Μελλοντικές",undated:"Χωρίς τεκμηριωμένη προθεσμία"}[v]||v;
  }
  function formatDay(v){const d=validDay(v);return d?d.split("-").reverse().join("/"):"";}
  function buildUrl(options){
    const q=new URLSearchParams({
      scope:options.scope||"real",active_only:String(options.active_only!==false),
      limit:String(options.limit||40),offset:String(options.offset||0),
      query:options.query||"",category:options.category||"",
      status:options.status||"",due:options.due||"all",
      sort:options.sort||"updated"
    });
    return "/clinical/clinical-updates?"+q;
  }
  async function load(options){
    const response=await fetch(buildUrl(options),{
      credentials:"same-origin",cache:"no-store",headers:{"Accept":"application/json"}
    });
    if(!response.ok){const err=new Error(response.status===401||response.status===403?
      "Χρειάζεται επανασύνδεση στο Cockpit.":"Δεν είναι διαθέσιμη η προστατευμένη λίστα υποχρεώσεων.");
      err.status=response.status;throw err;}
    const data=await response.json();
    if(!data||!Array.isArray(data.items)||!Number.isInteger(data.total)||
      data.total<0||!Number.isInteger(data.shown)||data.shown!==data.items.length||
      typeof data.has_more!=="boolean"||!validDay(data.today)||
      data.items.some(x=>!x||typeof x.patient_id!=="string"||
        typeof x.pending_id!=="string"||typeof x.description!=="string"))
      throw Error("Η πηγή επέστρεψε μη έγκυρες υποχρεώσεις.");
    return data;
  }
  function attentionBadge(total){
    // Total is based ONLY on explicit due/overdue review_date for this date.
    window.dispatchEvent(new CustomEvent("cockpit:ccc01-attention-count",{
      detail:{total:Number.isInteger(total)?total:null}
    }));
  }
  async function refreshAttention(){
    const ticket=++attentionToken,generation=epoch;
    attentionBadge(null);
    try{
      const data=await load({scope:"real",active_only:true,limit:4,due:"today"});
      if(ticket!==attentionToken||generation!==epoch)return;
      attentionBadge(data.total);
    }catch{
      if(ticket===attentionToken&&generation===epoch)attentionBadge(null);
    }
  }
  function itemButton(parent,item,close){
    const b=button(parent,"",async()=>{
      if(!window.CockpitStoredChart?.openPending)return;
      if(window.CockpitStoredChart.canLeave&&!window.CockpitStoredChart.canLeave())return;
      close();
      await window.CockpitStoredChart.openPending(item.patient_id,item.pending_id);
    },"doctor-action-row ccc-obligation-row");
    b.dataset.pendingId=item.pending_id;
    const body=node(b,"span");
    node(body,"strong",item.description);
    const meta=[statusText(item),types[item.item_type]||item.item_type,
      item.review_date?"Έλεγχος "+formatDay(item.review_date):"Χωρίς ημερομηνία"];
    node(body,"small",meta.filter(Boolean).join(" · "));
    node(b,"span","›");
    return b;
  }
  async function renderPopover(target,active,close,openAll,scope="real"){
    const ticket=++popupToken,generation=epoch;
    target.replaceChildren();node(target,"p","Φόρτωση σημερινών εργασιών…");
    try{
      const data=await load({scope,active_only:true,due:"today",limit:4,offset:0});
      if(ticket!==popupToken||generation!==epoch||!active())return;
      target.replaceChildren();
      node(target,"p",data.total?
        "Σημερινές και εκπρόθεσμες υποχρεώσεις με τεκμηριωμένη ημερομηνία · "+data.total:
        "Δεν υπάρχουν υποχρεώσεις με τεκμηριωμένη σημερινή ή παλαιότερη προθεσμία.","ccc-popup-lead");
      for(const item of data.items){
        const group=node(target,"section","","patient-updates-group");
        node(group,"h3",item.patient_name||"Κλινικός φάκελος");
        itemButton(group,item,close);
      }
      if(data.has_more)node(target,"p","Εμφανίζονται οι πρώτες 4 · οι υπόλοιπες βρίσκονται στην πλήρη οθόνη.","ccc-popup-lead");
      button(target,"Όλες οι υποχρεώσεις →",openAll,"ccc-open-full");
      node(target,"p","Email και κλήσεις δεν έχουν ακόμη συνδεθεί. Εργασίες χωρίς προθεσμία παραμένουν στο πλήρες μητρώο.","ccc-popup-note");
    }catch(e){
      if(ticket!==popupToken||generation!==epoch||!active())return;
      target.replaceChildren();
      node(target,"p",e.message||"Οι εργασίες δεν είναι διαθέσιμες.");
      button(target,"Όλες οι υποχρεώσεις →",openAll,"ccc-open-full");
    }
  }
  function groupKey(item,group,today){
    if(group==="category")return item.item_type||"other";
    if(group==="due")return dueBucket(item,today);
    return item.patient_id;
  }
  function groupTitle(item,group,today){
    if(group==="category")return types[item.item_type]||item.item_type||"Άλλο";
    if(group==="due")return bucketName(dueBucket(item,today));
    return item.patient_name||"Κλινικός φάκελος";
  }
  function paintFull(data){
    const root=$("cccObligationsList");
    root.replaceChildren();
    const info=$("cccObligationsStatus");
    info.textContent=data.total?
      "Υποχρεώσεις "+(full.offset+1)+"–"+(full.offset+data.items.length)+" από "+data.total:
      "Δεν βρέθηκαν υποχρεώσεις με αυτά τα φίλτρα.";
    const groups=new Map();
    for(const item of data.items){
      const key=groupKey(item,full.group,data.today);
      let section=groups.get(key);
      if(!section){
        section=node(root,"section","","ccc-obligation-group patient-updates-group");
        node(section,"h3",groupTitle(item,full.group,data.today));
        groups.set(key,section);
      }
      itemButton(section,item,full.leave);
      if(item.trigger||item.external_dependency){
        // Informational cue, never an automatically observed trigger event.
        const last=section.lastElementChild;
        const small=node(last.querySelector("span"),"small",
          item.trigger||item.external_dependency?.condition||"Αναμονή εξωτερικού συμβάντος");
        small.className="ccc-trigger-note";
      }
    }
    $("cccPrev").disabled=full.offset===0;
    $("cccNext").disabled=!data.has_more;
    $("cccPage").textContent="Σελίδα "+(Math.floor(full.offset/full.limit)+1);
  }
  function readyForFull(ticket,generation){
    return ticket===fullToken&&generation===epoch&&
      typeof full.active==="function"&&full.active();
  }
  async function loadFull(){
    const ticket=++fullToken,generation=epoch;
    const list=$("cccObligationsList");
    list.replaceChildren();$("cccObligationsStatus").textContent="Φόρτωση υποχρεώσεων…";
    $("cccPrev").disabled=true;$("cccNext").disabled=true;
    try{
      const data=await load({...full,active_only:!["resolved","cancelled"].includes(full.status)});
      if(!readyForFull(ticket,generation))return;
      lastFull=data;paintFull(data);
    }catch(e){
      if(!readyForFull(ticket,generation))return;
      lastFull=null;list.replaceChildren();
      $("cccObligationsStatus").textContent=e.message||"Δεν είναι διαθέσιμες οι υποχρεώσεις.";
      $("cccPage").textContent="";
    }
  }
  function mountFull(active,leave){
    full.active=active;full.leave=leave;
    if(!fullReady){
      fullReady=true;
      const query=$("cccSearch");
      const fields=[
        ["cccCategory","category"],["cccStatus","status"],
        ["cccDue","due"],["cccSort","sort"],["cccGroup","group"]
      ];
      for(const [id,name] of fields){
        $(id).addEventListener("change",()=>{
          full[name]=$(id).value;full.offset=0;
          if(name==="group"&&lastFull){paintFull(lastFull);return;}
          loadFull();
        });
      }
      let debounce=0;
      query.addEventListener("input",()=>{
        window.clearTimeout(debounce);
        debounce=window.setTimeout(()=>{
          full.query=query.value.trim();full.offset=0;loadFull();
        },250);
      });
      $("cccPrev").addEventListener("click",()=>{
        full.offset=Math.max(0,full.offset-full.limit);loadFull();
      });
      $("cccNext").addEventListener("click",()=>{
        if(!lastFull?.has_more)return;
        full.offset=lastFull.next_offset;loadFull();
      });
      $("cccSynthetic").addEventListener("click",()=>{
        full.scope=full.scope==="real"?"test":"real";
        full.offset=0;$("cccSynthetic").textContent=full.scope==="real"?
          "Δοκιμαστικός φάκελος":"Πραγματικοί φάκελοι";
        loadFull();
      });
    }
    loadFull();
  }
  window.addEventListener("clinical:unauthenticated",()=>{
    epoch++;popupToken++;fullToken++;attentionToken++;lastFull=null;
    $("cccObligationsList")?.replaceChildren();
    if($("cccObligationsStatus"))$("cccObligationsStatus").textContent="Απαιτείται σύνδεση.";
    attentionBadge(null);
  });
  window.addEventListener("clinical:authenticated",()=>{
    epoch++;refreshAttention();
    if(full.active?.())loadFull();
  });
  window.addEventListener("cockpit:clinical-updates-changed",()=>{
    refreshAttention();if(full.active?.())loadFull();
  });
  window.CockpitCCC01={renderPopover,mountFull,refreshAttention};
  refreshAttention();
})();