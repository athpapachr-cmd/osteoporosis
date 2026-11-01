/* Private, volatile projection revisions and explicit sorting preferences. */
(() => {
  "use strict";
  const endpoint="/clinical/icloud-mail/preferences";
  let observed=null,authGeneration=0;
  const drafts=new Map();
  let saving=false;
  function lockEditors(){
    saving=true;
    const controls=Array.from(document.querySelectorAll(".doctor-icloud-preferences input,.doctor-icloud-preferences select,.doctor-icloud-preferences button"));
    const disabled=controls.map(n=>n.disabled);
    controls.forEach(n=>{n.disabled=true;});
    return ()=>{controls.forEach((n,i)=>{n.disabled=disabled[i];});saving=false;};
  }
  const values=fields=>fields.map(n=>n.type==="checkbox"?n.checked:n.value);
  function clearSubmitted(key,reference,submitted){
    if(drafts.get(key)===reference&&JSON.stringify(reference)===JSON.stringify(submitted))drafts.delete(key);
  }
  const floors=new Map();
  const valid=m=>m&&m.available===true&&/^[a-f0-9]{64}$/.test(m.mailbox_key)&&
    /^[1-9][0-9]{0,18}$/.test(m.uidvalidity)&&Number.isSafeInteger(m.revision)&&m.revision>=0;
  const key=m=>m.mailbox_key+":"+m.uidvalidity;
  const same=(a,b)=>valid(a)&&valid(b)&&key(a)===key(b)&&a.revision===b.revision;
  const accept=m=>valid(m)&&(!observed||key(m)===key(observed))&&m.revision>=(floors.get(key(m))||0);
  async function request(body){
    const generation=authGeneration;
    const response=await fetch(endpoint,{method:body?"POST":"GET",credentials:"same-origin",cache:"no-store",
      headers:{"Accept":"application/json",...(body?{"Content-Type":"application/json"}:{})},
      ...(body?{body:JSON.stringify(body)}:{})});
    if(!response.ok){
      const error=new Error(response.status===409?
        "Οι επιλογές άλλαξαν ή η αποθήκευση δεν επιβεβαιώθηκε. Έλεγξε ξανά πριν αποθηκεύσεις.":
        response.status===422?"Έλεγξε τον αποστολέα και τις λέξεις του κανόνα.":
        "Οι προτιμήσεις δεν είναι διαθέσιμες. Δεν επιβεβαιώθηκε αποθήκευση.");
      error.status=response.status;
      if(response.status===401||response.status===403)window.dispatchEvent(new Event("clinical:unauthenticated"));
      throw error;
    }
    const data=await response.json();
    if(generation!==authGeneration)throw Error("Η κλινική συνεδρία άλλαξε.");
    if(!valid(data.triage)||!Array.isArray(data.rules))throw Error("Μη έγκυρη απάντηση προτιμήσεων.");
    return data;
  }
  async function qualify(){
    const data=await request();
    if(data.triage.revision<(floors.get(key(data.triage))||0))throw Error("Οι προτιμήσεις δεν επιβεβαιώθηκαν.");
    observed=data.triage;floors.set(key(observed),observed.revision);
    return data;
  }
  let channel;
  try{channel=new BroadcastChannel("cockpit-icloud-triage");}catch(_){}
  function invalidate(meta, remote=false){
    if(!valid(meta))return;
    floors.set(key(meta),Math.max(meta.revision,floors.get(key(meta))||0));
    if(!remote)observed=meta;
    window.dispatchEvent(new CustomEvent("cockpit:icloud-triage-changed",{detail:meta}));
    if(!remote)channel?.postMessage(meta);
  }
  if(channel)channel.onmessage=event=>invalidate(event.data,true);
  for(const event of ["clinical:unauthenticated","clinical:authenticated"])window.addEventListener(event,()=>{authGeneration++;observed=null;floors.clear();drafts.clear();});
  const labels={clinical:"Κλινικό",administrative:"Διοικητικό",other:"Άλλο",review:"Αχαρακτήριστο"};
  const origins={manual:"Δική σου επιλογή",rule:"Προσωπικός κανόνας",conflict:"Αντικρουόμενοι κανόνες — χρειάζεται χαρακτηρισμό",
    suggestion:"Αυτόματη πρόταση",unavailable:"Ο χαρακτηρισμός δεν επιβεβαιώθηκε"};
  const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const button=(text,handler)=>{const n=el("button",text);n.type="button";n.className="doctor-command-button";n.addEventListener("click",handler);return n;};
  function field(parent,label,node){node.setAttribute("aria-label",label);const wrap=el("label",label);wrap.append(node);parent.append(wrap);return node;}
  function category(parent,value,reset=false){
    const select=el("select");
    for(const [v,label] of Object.entries(labels)){const o=el("option",label);o.value=v;select.append(o);}
    if(reset){const o=el("option","Επαναφορά αυτόματης πρότασης");o.value="automatic";select.append(o);}
    select.value=value;return field(parent,"Κατηγορία",select);
  }
  function criteria(parent,rule={}){
    const sender=el("input");sender.maxLength=254;sender.value=rule.sender||"";
    field(parent,"Ακριβής διεύθυνση αποστολέα ή domain",sender);
    const domain=el("input");domain.type="checkbox";domain.checked=!!rule.domain;
    field(parent,"Κανόνας για ολόκληρο το domain",domain);
    const terms=el("input");terms.maxLength=647;terms.value=(rule.terms||[]).join(", ");
    field(parent,"Λέξεις θέματος, χωρισμένες με κόμμα",terms);
    parent.append(el("p","Ο κανόνας αφορά επόμενα παρόμοια μηνύματα. Οι υπηρεσίες μεταφοράς αρχείων χρειάζονται και λέξεις θέματος."));
    return target=>({sender:sender.value,domain:domain.checked,terms:terms.value.split(",").map(x=>x.trim()).filter(Boolean),category:target,
      ...(rule.rule_id?{rule_id:rule.rule_id}:{})});
  }
  const notice=()=>{
    let n=document.getElementById("doctorIcloudTriageNotice");
    if(!n){n=el("div");n.id="doctorIcloudTriageNotice";n.setAttribute("role","status");
      document.getElementById("doctorIcloudInboxState").after(n);}
    return n;
  };
  function confirmed(data,kind){
    invalidate(data.triage);
    const n=notice();n.replaceChildren(el("p","Η επιλογή αποθηκεύτηκε. Οι προβολές email ενημερώθηκαν."));
    if(kind!=="undo" && data.committed_revision===data.triage.revision){
      const undo={kind:"undo",operation_id:crypto.randomUUID(),expected_revision:data.triage.revision,undo_id:data.operation_id};
      n.append(button("Αναίρεση τελευταίας αλλαγής",async()=>{
        const b=n.querySelector("button");b.disabled=true;
        try{const result=await request(undo);confirmed(result,"undo");}
        catch(error){n.append(el("p",error.message));b.disabled=false;}
      }));
    }
  }
  async function editor(parent,item,current,projectionMeta){
    const box=el("section");box.className="doctor-icloud-preferences";
    parent.append(box);
    let prefs;
    try{prefs=await qualify();}catch(error){if(current())box.append(el("p",error.message));return;}
    if(!current())return;
    if(saving){box.append(el("p","Η αποθήκευση είναι σε εξέλιξη. Περίμενε την επιβεβαίωση."));return;}
    if(!same(prefs.triage,projectionMeta)){invalidate(prefs.triage);return;}
    box.append(el("h4","Χαρακτηρισμός μηνύματος"),el("p",origins[item.category_origin]||"Αυτόματη πρόταση"));
    const choose=category(box,item.category,true);
    const useRule=el("input");useRule.type="checkbox";
    field(box,"Κανόνας και για επόμενα παρόμοια",useRule);
    const ruleBox=el("div");ruleBox.hidden=true;box.append(ruleBox);
    const readRule=criteria(ruleBox);
    useRule.addEventListener("change",()=>{ruleBox.hidden=!useRule.checked;});
    const status=el("p");status.setAttribute("role","status");box.append(status);
    let pending=null,submittedDraft,submittedFields;
    const fields=Array.from(box.querySelectorAll("input,select"));
    const draft=drafts.get(item.id);
    if(draft){fields.forEach((n,i)=>{if(n.type==="checkbox")n.checked=!!draft[i];else n.value=draft[i];});ruleBox.hidden=!useRule.checked;status.textContent="Οι προηγούμενες επιλογές σου διατηρήθηκαν. Έλεγξέ τες πριν αποθηκεύσεις.";}
    box.addEventListener("input",()=>{pending=null;drafts.set(item.id,fields.map(n=>n.type==="checkbox"?n.checked:n.value));});
    const save=button("Αποθήκευση χαρακτηρισμού",async()=>{
      if(!current()||saving)return;
      if(useRule.checked&&choose.value==="automatic"){status.textContent="Επίλεξε κατηγορία για τον κανόνα.";return;}
      if(!pending){submittedDraft=drafts.get(item.id);submittedFields=values(fields);}
      pending=pending||{kind:"category",operation_id:crypto.randomUUID(),expected_revision:prefs.triage.revision,item_id:item.id,
        category:choose.value==="automatic"?null:choose.value,...(useRule.checked?{rule:readRule(choose.value)}:{})};
      const unlock=lockEditors();
      try{const result=await request(pending);clearSubmitted(item.id,submittedDraft,submittedFields);confirmed(result,"category");}
      catch(error){if(current())status.textContent=error.message;}
      finally{unlock();}
    });
    box.append(save,button("Έλεγχος νεότερων επιλογών",async()=>{
      try{const updated=await qualify();if(current()){prefs=updated;pending=null;status.textContent="Φορτώθηκαν οι νεότερες επιλογές. Έλεγξε τα πεδία και επίλεξε ξανά Αποθήκευση.";}}
      catch(error){if(current())status.textContent=error.message;}
    }));
  }
  async function manager(parent,current){
    let prefs;
    try{prefs=await qualify();}catch(error){if(current())parent.replaceChildren(el("p",error.message));return;}
    if(!current())return;
    parent.replaceChildren(el("h3","Προσωπικοί κανόνες email"));
    const renderForm=(rule={})=>{
      if(saving)return;
      const form=el("section");form.className="doctor-icloud-preferences";
      form.append(el("h4",rule.rule_id?"Τροποποίηση κανόνα":"Νέος κανόνας"));
      const readRule=criteria(form,rule);const choose=category(form,rule.category||"clinical");
      const status=el("p");form.append(status);let pending=null,submittedDraft,submittedFields;
      const draftKey="rule:"+(rule.rule_id||"new");const fields=Array.from(form.querySelectorAll("input,select"));
      const draft=drafts.get(draftKey);
      if(draft){fields.forEach((n,i)=>{if(n.type==="checkbox")n.checked=!!draft[i];else n.value=draft[i];});status.textContent="Διατηρήθηκαν μη αποθηκευμένες επιλογές. Έλεγξέ τες πριν από την Αποθήκευση.";}
      form.addEventListener("input",()=>{pending=null;drafts.set(draftKey,fields.map(n=>n.type==="checkbox"?n.checked:n.value));});
      const save=button("Αποθήκευση κανόνα",async()=>{
        if(saving)return;
        if(!pending){submittedDraft=drafts.get(draftKey);submittedFields=values(fields);}
        pending=pending||{kind:"rule",operation_id:crypto.randomUUID(),expected_revision:prefs.triage.revision,rule:readRule(choose.value)};
        const unlock=lockEditors();
        try{const result=await request(pending);clearSubmitted(draftKey,submittedDraft,submittedFields);confirmed(result,"rule");}
        catch(error){if(current())status.textContent=error.message;}
        finally{unlock();}
      });form.append(save,button("Έλεγχος νεότερων επιλογών",async()=>{
        try{const updated=await qualify();if(current()){prefs=updated;pending=null;status.textContent="Ελέγχθηκαν οι νεότερες επιλογές. Επιβεβαίωσε ξανά με Αποθήκευση.";}}
        catch(error){if(current())status.textContent=error.message;}
      }));parent.append(form);
    };
    for(const rule of prefs.rules){
      const row=el("section");row.className="doctor-icloud-preferences";
      row.append(el("p",rule.sender+" · "+rule.terms.join(", ")+" → "+labels[rule.category]),
        el("p",rule.active?"Ενεργός για επόμενα μηνύματα":rule.enabled?"Χρειάζεται επανενεργοποίηση μετά από αλλαγή θυρίδας":"Απενεργοποιημένος"));
      row.append(button("Τροποποίηση / ενεργοποίηση",()=>renderForm(rule)));
      if(rule.enabled)row.append(button("Απενεργοποίηση κανόνα",()=>{
        const disable={kind:"disable",operation_id:crypto.randomUUID(),expected_revision:prefs.triage.revision,rule_id:rule.rule_id};
        row.append(el("p","Τα μηνύματα που επηρεάζει θα ταξινομούνται από τους υπόλοιπους κανόνες και τις αυτόματες προτάσεις. Οι ατομικές επιλογές σου παραμένουν."),
          button("Επιβεβαίωση απενεργοποίησης",async()=>{
            try{const result=await request(disable);confirmed(result,"disable");}
            catch(error){if(current())row.append(el("p",error.message));}
          }));
      }));parent.append(row);
    }
    renderForm();
    return prefs.triage;
  }
  window.addEventListener("cockpit:icloud-triage-changed",()=>document.getElementById("doctorIcloudTriageNotice")?.replaceChildren());
  window.addEventListener("clinical:unauthenticated",()=>document.getElementById("doctorIcloudTriageNotice")?.replaceChildren());
  window.CockpitIcloudTriage={qualify,accept,same,invalidate,editor,manager,origins};
})();
