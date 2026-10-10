(() => {
 "use strict";
 // Read-only UI. Existing Calendar/Clinical Data owners retain authority.
 const $=id=>document.getElementById(id);
 const state={selected:null,appointment:null,parts:null,tab:"snapshot",revision:0,timer:null,briefReturnFocus:null};
 const localTime=new Intl.DateTimeFormat("el-CY",{hour:"2-digit",minute:"2-digit",hour12:false,timeZone:"Asia/Nicosia"});
 const localDay=new Intl.DateTimeFormat("el-CY",{day:"2-digit",month:"2-digit",year:"numeric",timeZone:"Asia/Nicosia"});
 const dateTime=value=>{if(!value)return null;const s=String(value);const d=new Date(/(?:Z|[+-]\d{2}:?\d{2})$/.test(s)?s:s+"Z");return Number.isFinite(d.getTime())?d:null;};
 const fold=x=>String(x||"").normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLocaleLowerCase("el");
 function nameOf(p){
   const d=p.demographics||{};
   return String(d.full_name||d.fullName||d.name||d["ονοματεπώνυμο"]||
     [d.first_name||d.firstName||d["όνομα"],d.last_name||d.lastName||d["επώνυμο"]].filter(Boolean).join(" ")||
     "Ασθενής χωρίς καταχωρισμένο όνομα").trim();
 }
 async function get(url){
   const r=await fetch(url,{credentials:"same-origin",headers:{"Accept":"application/json"}});
   if(!r.ok){const e=new Error("HTTP "+r.status);e.status=r.status;throw e;}
   return r.json();
 }
 function info(target,text){
   target.replaceChildren();const p=document.createElement("p");p.className="clinical-muted";p.textContent=text;target.append(p);
 }
 function addRow(target,title,subtitle,leading,fn){
   const b=document.createElement("button");b.type="button";b.className="clinical-patient-row";
   const lead=document.createElement("span");lead.className="clinical-row-time";lead.textContent=leading;
   const body=document.createElement("span");body.className="clinical-row-copy";
   const strong=document.createElement("strong");strong.textContent=title;
   const small=document.createElement("small");small.textContent=subtitle;
   body.append(strong,small);
   const arrow=document.createElement("span");arrow.className="clinical-row-arrow";arrow.textContent="›";
   b.append(lead,body,arrow);b.addEventListener("click",fn);target.append(b);
 }
 function resetDraft(){
   state.parts=null;state.tab="snapshot";
   $("visitDiaText").value="";$("visitDiaEditText").value="";
   $("visitDiaComposer").hidden=true;$("visitDiaPreview").hidden=true;
   $("visitDiaEditPane").hidden=true;$("visitDiaCopyStatus").textContent="";
   $("visitDiaOpen").textContent="Σύνοψη από το Dia";
   $("visitDiaOpen").setAttribute("aria-expanded","false");
   document.querySelectorAll("[data-clinical-tab]").forEach(b=>{
     const active=b.dataset.clinicalTab==="snapshot";b.classList.toggle("selected",active);
     b.setAttribute("aria-selected",String(active));
   });
 }
 function open(row){
   state.revision++;clearTimeout(state.timer);
   resetDraft();
   state.selected=row;
   if(row.kind==="appointment")state.appointment=row;
   $("visitWorkspaceHome").hidden=true;$("visitWorkspacePatient").hidden=false;
   $("visitSelectedName").textContent=row.name;
   $("visitSelectedReason").textContent=[row.time||"",row.reason||""].filter(Boolean).join(" · ");
   $("visitSelectedSource").textContent=row.kind==="appointment"?"ΑΠΟ ΤΟ ΗΜΕΡΟΛΟΓΙΟ":
     row.kind==="recent"?"ΠΡΟΣΦΑΤΗ ΚΛΙΝΙΚΗ ΕΠΙΣΚΕΨΗ":"ΑΠΟ ΤΟ ΜΗΤΡΩΟ";
   $("visitIdentityStatus").textContent=row.patientId?"Υπάρχει κλινικός φάκελος":"Απαιτείται επιλογή φακέλου";
   $("visitIdentityNote").textContent=row.patientId ?
     "Επίλεξες υπάρχουσα προστατευμένη κλινική εγγραφή. Έλεγξε τα στοιχεία πριν από οριστική καταγραφή." :
     "Το όνομα ενός ραντεβού δεν επιβεβαιώνει τον κλινικό φάκελο. Επίλεξε τον σωστό ασθενή από το μητρώο.";
   $("visitChoosePatient").hidden=Boolean(row.patientId);
   $("visitPatientMatches").hidden=true;
   $("visitPatientSearch").setAttribute("aria-expanded","false");
 }
 function back(){
   state.revision++;clearTimeout(state.timer);
   resetDraft();state.selected=null;state.appointment=null;
   $("visitWorkspacePatient").hidden=true;$("visitWorkspaceHome").hidden=false;
   $("visitPatientSearch").value="";$("visitPatientMatches").hidden=true;
   $("visitPatientSearch").setAttribute("aria-expanded","false");
 }
 async function recent(){
   const target=$("visitRecentRows");
   try{
     const list=await get("/clinical/recent-encounters?limit=3");
     if(!Array.isArray(list))throw Error("invalid response");
     target.replaceChildren();
     if(!list.length)return info(target,"Δεν υπάρχουν ολοκληρωμένες επισκέψεις στο μητρώο.");
     list.slice(0,3).forEach(r=>{
       const name=r.patient_display_name||"Ασθενής χωρίς καταχωρισμένο όνομα";
       const context=[r.encounter_date,r.visit_type||"Κλινική επίσκεψη"].filter(Boolean).join(" · ");
       addRow(target,name,context,"↶",()=>open({kind:"recent",name,reason:context,patientId:r.patient_id}));
     });
   }catch(e){info(target,e.status===401?"Συνδέσου για να δεις τις κλινικές επισκέψεις.":"Οι πρόσφατες κλινικές επισκέψεις δεν είναι ακόμη διαθέσιμες.");}
 }
 async function upcoming(){
   const target=$("visitUpcomingRows");
   try{
     const c=await get("/clinical/calendar/cockpit-context");
     const now=dateTime(c.generated_at)||new Date();
     const rows=Array.isArray(c.upcoming_today)?c.upcoming_today:(c.next?[c.next]:[]);
     const future=rows.filter(a=>{const d=dateTime(a.start_at);return d&&d>now&&localDay.format(d)===localDay.format(now);})
       .sort((a,b)=>dateTime(a.start_at)-dateTime(b.start_at)).slice(0,3);
     target.replaceChildren();
     if(!future.length)return info(target,"Δεν υπάρχουν άλλες διαθέσιμες σημερινές επισκέψεις.");
     future.forEach(a=>{
       const name=a.patient_display_name||"Ραντεβού χωρίς όνομα",reason=a.reason||a.category||"Ραντεβού";
       const time=localTime.format(dateTime(a.start_at));
       addRow(target,name,reason,time,()=>open({kind:"appointment",name,reason,time,appointmentId:a.appointment_id,patientId:null}));
     });
     if(!Array.isArray(c.upcoming_today))$("visitHomeNote").textContent=
       "Το ημερολόγιο επιστρέφει προσωρινά μόνο την επόμενη επίσκεψη. Δεν κατασκευάζονται επιπλέον ραντεβού.";
   }catch(e){info(target,e.status===401?"Συνδέσου για να δεις το πρόγραμμα.":"Το σημερινό πρόγραμμα δεν είναι διαθέσιμο.");}
 }
 async function loadMatches(term,ticket,offset){
   const box=$("visitPatientMatches");
   try{
     const rows=await get("/clinical/patients?query="+encodeURIComponent(term)+"&limit=20&offset="+offset);
     if(ticket!==state.revision)return;
     if(!Array.isArray(rows))throw Error("invalid results");
     if(offset===0)box.replaceChildren();
     if(!rows.length&&offset===0)return info(box,"Δεν βρέθηκε ασθενής.");
     rows.forEach(p=>{
       const name=nameOf(p),d=p.demographics||{},birthday=d.date_of_birth||d.birth_date||d.dob||"";
       const meta=birthday?"Γέννηση: "+birthday:"Καταχωρισμένος φάκελος — επιβεβαίωσε τα στοιχεία πριν την επιλογή";
       addRow(box,name,meta,"↗",()=>{
         const prev=state.appointment;
         open({kind:"registry",name,reason:prev?prev.reason:meta,time:prev?prev.time:"",patientId:p.patient_id});
       });
     });
     if(rows.length===20){
       const more=document.createElement("button");more.type="button";
       more.className="clinical-more-results";more.textContent="Περισσότερα αποτελέσματα";
       more.addEventListener("click",()=>{more.disabled=true;more.remove();loadMatches(term,ticket,offset+rows.length);});
       box.append(more);
     }
   }catch(e){if(ticket===state.revision)info(box,e.status===401?"Απαιτείται σύνδεση.":"Η αναζήτηση δεν είναι διαθέσιμη.");}
 }
 function search(){
   clearTimeout(state.timer);
   const term=$("visitPatientSearch").value.trim(),ticket=++state.revision,box=$("visitPatientMatches");
   box.hidden=!term;
   $("visitPatientSearch").setAttribute("aria-expanded",String(Boolean(term)));
   if(!term){box.replaceChildren();return;}
   info(box,"Αναζήτηση σε όλο το υπάρχον μητρώο…");
   state.timer=setTimeout(()=>loadMatches(term,ticket,0),250);
 }
 function closeBrief(){
   $("visitBriefOverlay").hidden=true;
   const opener=state.briefReturnFocus;state.briefReturnFocus=null;
   if(opener)opener.focus();
 }
 function showBrief(prefix){
   const nameNode=$(prefix+"AppointmentPatient");
   if(!nameNode.dataset.appointmentId)return;
   const overlay=$("visitBriefOverlay");
   state.briefReturnFocus=$("visitSidebar"+prefix[0].toUpperCase()+prefix.slice(1));
   $("visitBriefName").textContent=nameNode.textContent;
   $("visitBriefWhen").textContent=$(prefix+"AppointmentTime").textContent;
   $("visitBriefReason").textContent=$(prefix+"AppointmentType").textContent;
   $("visitBriefFreshness").textContent=$("calendarNote").textContent;
   $("visitBriefIdentity").textContent="Δεν υπάρχει επιβεβαιωμένη κλινική σύνδεση από το ημερολόγιο. Το όνομα δεν αρκεί για πρόσβαση στον φάκελο.";
   overlay.hidden=false;
   $("visitBriefClose").focus();
 }
 function selectFromBrief(){
   const name=$("visitBriefName").textContent,reason=$("visitBriefReason").textContent,time=$("visitBriefWhen").textContent;
   closeBrief();back();
   state.appointment={kind:"appointment",name,reason,time,patientId:null};
   $("visitPatientSearch").value=name;$("visitPatientSearch").focus();search();
 }
 function parse(raw){
   const parts={snapshot:"",brief:"",detail:""},keys={"SNAPSHOT":"snapshot","VISIT BRIEF":"brief","ENCOUNTER DETAIL":"detail"};
   let active=null;const preface=[];
   String(raw).replace(/\r\n?/g,"\n").split("\n").forEach(line=>{
     const key=line.trim().replace(/^#{1,6}\s*/,"").replace(/:$/,"").trim().toUpperCase();
     if(Object.prototype.hasOwnProperty.call(keys,key)){active=keys[key];return;}
     if(active)parts[active]+=(parts[active]?"\n":"")+line;
     else preface.push(line);
   });
   if(!Object.values(parts).some(v=>v.trim()))parts.brief=preface.join("\n").trim();
   Object.keys(parts).forEach(k=>parts[k]=parts[k].trim());
   return parts;
 }
 function render(){
   const has=state.parts&&Object.values(state.parts).some(v=>v.trim());
   $("visitDiaPreview").hidden=!has;
   $("visitDiaPreviewText").textContent=has?(state.parts[state.tab]||"Δεν δόθηκε κείμενο για αυτή την ενότητα."):"";
 }
 async function copy(){
   const content=$("visitDiaPrompt").textContent.trim();
   try{
     if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(content);
     else{
       const t=document.createElement("textarea");t.value=content;document.body.append(t);t.select();
       const okay=document.execCommand("copy");t.remove();if(!okay)throw Error("copy refused");
     }
     $("visitDiaCopyStatus").textContent="Αντιγράφηκε. Επικόλλησε την οδηγία στο Dia.";
   }catch(e){$("visitDiaCopyStatus").textContent="Η αντιγραφή δεν επιτράπηκε.";}
 }
 ["previous","current","next"].forEach(prefix=>{
   $("visitSidebar"+prefix[0].toUpperCase()+prefix.slice(1)).addEventListener("click",()=>showBrief(prefix));
 });
 $("visitBriefClose").addEventListener("click",closeBrief);
 $("visitBriefChoosePatient").addEventListener("click",selectFromBrief);
 $("visitBriefOverlay").addEventListener("click",e=>{if(e.target===$("visitBriefOverlay"))closeBrief();});
 document.addEventListener("keydown",e=>{
   if($("visitBriefOverlay").hidden)return;
   if(e.key==="Escape"){e.preventDefault();closeBrief();}
   if(e.key==="Tab"){
     const a=$("visitBriefClose"),b=$("visitBriefChoosePatient");
     if(e.shiftKey&&document.activeElement===a){e.preventDefault();b.focus();}
     else if(!e.shiftKey&&document.activeElement===b){e.preventDefault();a.focus();}
   }
 });
 $("visitBackHome").addEventListener("click",back);
 $("visitChoosePatient").addEventListener("click",()=>{
   $("visitWorkspacePatient").hidden=true;$("visitWorkspaceHome").hidden=false;
   $("visitPatientSearch").value=state.selected?.name||"";$("visitPatientSearch").focus();search();
 });
 $("visitPatientSearch").addEventListener("input",search);
 $("visitPatientSearch").addEventListener("keydown",e=>{if(e.key==="Escape"){$("visitPatientSearch").value="";search();}});
 $("visitDiaOpen").addEventListener("click",()=>{
   if(!state.selected)return;
   const expand=$("visitDiaComposer").hidden;
   $("visitDiaComposer").hidden=!expand;
   $("visitDiaOpen").textContent=expand?"Κλείσιμο εισαγωγής Dia":"Σύνοψη από το Dia";
   $("visitDiaOpen").setAttribute("aria-expanded",String(expand));
   if(expand)$("visitDiaText").focus();
 });
 $("visitDiaClose").addEventListener("click",()=>{
   $("visitDiaComposer").hidden=true;$("visitDiaOpen").textContent="Σύνοψη από το Dia";
   $("visitDiaOpen").setAttribute("aria-expanded","false");
 });
 $("visitDiaCopy").addEventListener("click",copy);
 $("visitDiaText").addEventListener("input",()=>{
   state.parts=$("visitDiaText").value.trim()?parse($("visitDiaText").value):null;render();
 });
 document.querySelectorAll("[data-clinical-tab]").forEach(b=>b.addEventListener("click",()=>{
   state.tab=b.dataset.clinicalTab;
   document.querySelectorAll("[data-clinical-tab]").forEach(t=>{
     const active=t===b;t.classList.toggle("selected",active);t.setAttribute("aria-selected",String(active));
   });
   render();
 }));
 $("visitDiaEdit").addEventListener("click",()=>{
   if(!state.parts)return;$("visitDiaEditPane").hidden=false;
   $("visitDiaEditText").value=state.parts[state.tab]||"";$("visitDiaEditText").focus();
 });
 $("visitDiaApplyEdit").addEventListener("click",()=>{
   if(!state.parts)return;state.parts[state.tab]=$("visitDiaEditText").value.trim();
   $("visitDiaEditPane").hidden=true;render();
 });
 recent();upcoming();
})();
