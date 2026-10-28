/* Normal Cockpit: existing chart/atomic VisitCapture owners, transient drafts. */
(() => {
  'use strict';
  if(window.CockpitSynthetic)return;
  const $=id=>document.getElementById(id);
  let draftGeneration=0,journeyObserver=null,pendingBusy=false,chartPending=[],amendment=null,pendingBindings=new Map();
  let generation=0,patientId=null,contextId=null,candidate=null,source='',busy=false,chartScroll=0,chartReturnPoint=null;
  let audioRecorder=null,audioStream=null,audioChunks=[],audioTimer=null,audioAbort=null;
  let audioStarted=0,audioDiscard=false,audioDraft=false,audioReady=false,audioReviewUnits=[],audioSuggestions=[],audioSuggestionDraft='',audioToken=0,audioPreviewed=null,audioContextExpiresAt='';
  const MAX_AUDIO_BYTES=23*1024*1024;
  const session=crypto.randomUUID();
  const el=(parent,tag,value,cls)=>{const n=document.createElement(tag);if(value)n.textContent=value;if(cls)n.className=cls;parent.append(n);return n;};
  const button=(parent,label,fn)=>{const b=el(parent,'button',label,'clinical-secondary');b.type='button';b.onclick=fn;return b;};
  const date=v=>v?new Intl.DateTimeFormat('el-CY',{dateStyle:'medium',timeZone:'Asia/Nicosia'}).format(new Date(v.includes('T')?v:v.slice(0,10)+'T12:00:00Z')):'';
  const statuses={open:'Ανοικτή',awaiting_patient:'Αναμένεται από τον ασθενή',awaiting_result:'Αναμένεται αποτέλεσμα',resolved:'Ολοκληρώθηκε',cancelled:'Ακυρώθηκε',not_yet_indicated:'Δεν έχει ακόμη ένδειξη'};
  const roles={patient:'Ασθενής',clinician:'Γιατρός',secretary:'Γραμματεία',unspecified:'Δεν προσδιορίστηκε'};
  const chart=el($('visitWorkspacePatient'),'section','','patient-chart');chart.id='patientChart';
  const composer=el($('visitWorkspacePatient'),'section','','clinical-dia-composer');composer.id='captureComposer';composer.hidden=true;
  const composerTitle=el(composer,'h3','Νέα επίσκεψη από τη σύνοψη Dia');
  const amendmentNote=el(composer,'p','','clinical-muted');amendmentNote.hidden=true;
  const promptText=$('visitDiaNaturalPrompt').content.textContent.trim();
  const promptTools=el(composer,'section','','capture-dia-prompt');
  const promptCopy=button(promptTools,'Αντιγραφή οδηγίας για το Dia',async()=>{if(!await window.CockpitHome.copyDiaPrompt(promptText,promptStatus))promptDetails.open=true;});promptCopy.id='captureCopyDiaPrompt';
  const promptDetails=el(promptTools,'details');el(promptDetails,'summary','Δες την οδηγία για το Dia');
  const promptDisplay=el(promptDetails,'pre',promptText);promptDisplay.id='captureDiaPromptText';promptDisplay.tabIndex=0;
  const promptStatus=el(promptTools,'p','','clinical-muted');promptStatus.id='captureDiaPromptStatus';promptStatus.setAttribute('role','status');

  const dateLabel=el(composer,'label','Ημερομηνία επίσκεψης');
  const visitDate=el(dateLabel,'input');visitDate.type='date';visitDate.id='captureDate';
  const audioPanel=el(composer,'section','','capture-web-audio');audioPanel.id='captureWebAudio';
  el(audioPanel,'h4','Καταγραφή επίσκεψης με μικρόφωνο');
  el(audioPanel,'p','Μετά τη Λήξη, ο ήχος αποστέλλεται στην OpenAI για μεταγραφή. Η μεταγραφή και, όταν υπάρχουν, το όνομα και η ημερομηνία γέννησης της επιλεγμένης καρτέλας αποστέλλονται στην OpenAI για το προσχέδιο.','clinical-muted');
  el(audioPanel,'p','Η OpenAI μπορεί να διατηρεί τη μεταγραφή και τα στοιχεία της καρτέλας σε αρχεία ελέγχου έως 30 ημέρες ή περισσότερο στις εξαιρέσεις της πολιτικής της. Έλεγξε το κλινικό κείμενο πριν την Αποθήκευση.','clinical-muted');
  const informedLabel=el(audioPanel,'label','');
  const informed=el(informedLabel,'input');informed.type='checkbox';informed.id='capturePatientInformed';
  informedLabel.append(document.createTextNode(' Ο ασθενής ενημερώθηκε για την ηχογράφηση, την επεξεργασία από OpenAI και τη διατήρηση που περιγράφεται παραπάνω, και συμφώνησε'));
  const audioStart=button(audioPanel,'Έναρξη',startAudio);audioStart.id='captureAudioStart';audioStart.disabled=true;
  const audioStop=button(audioPanel,'Λήξη',stopAudio);audioStop.id='captureAudioStop';audioStop.disabled=true;
  const audioStatus=el(audioPanel,'p','Έλεγχος διαθεσιμότητας…');audioStatus.id='captureAudioStatus';audioStatus.setAttribute('role','status');
  const audioReview=el(composer,'section','','capture-audio-review');audioReview.hidden=true;
  el(audioReview,'h4','Έλεγχος ηχογράφησης και κλινικού κειμένου');
  const audioPatient=el(audioReview,'strong','');
  const audioConfirmLabel=el(audioReview,'label','');
  const audioConfirm=el(audioConfirmLabel,'input');audioConfirm.type='checkbox';audioConfirm.id='captureAudioIdentity';
  audioConfirmLabel.append(document.createTextNode(' Έλεγξα τα παραπάνω και επιβεβαιώνω ότι η ηχογράφηση ανήκει στον εμφανιζόμενο ασθενή'));
  const audioFocused=el(audioReview,'div','','capture-audio-focused');
  const inputLabel=el(composer,'label','Σύνοψη επίσκεψης');
  const input=el(inputLabel,'textarea');input.id='captureText';input.rows=7;input.maxLength=30000;
  input.placeholder='Επικόλλησε το φυσικό κείμενο της σύνοψης. Δεν χρειάζονται ειδικές επικεφαλίδες.';
  const extractButton=button(composer,'Προετοιμασία καταγραφής',extract);
  const feedback=el(composer,'p');feedback.id='captureStatus';feedback.setAttribute('role','status');
  const review=el(composer,'div');review.id='captureReview';
  const previews=el(composer,'div');
  const saveButton=button(composer,'Αποθήκευση επίσκεψης και υποχρεώσεων',save);saveButton.id='captureSave';saveButton.hidden=true;
  const closeComposer=button(composer,'Κλείσιμο',()=>{const previous=amendment;resetDraft();composer.hidden=true;chart.hidden=false;if(previous)showVisit(patientId,previous.encounter_id);});
  const testButton=button($('visitPatientSearch').parentElement,'Δοκιμαστικός ασθενής',async()=>{
    try{const data=await request('/visit-capture/test-patient',{});await open(data.patient_id);}catch(e){testButton.textContent=e.message;}
  });testButton.id='captureTestPatient';
  async function request(path,body){
    const r=await fetch('/clinical'+path,{method:body===undefined?'GET':'POST',credentials:'same-origin',cache:'no-store',
      headers:{'Content-Type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)})});
    if(!r.ok){
      if(r.status===503){
        const error=await r.json().catch(()=>null);
        if(error?.detail?.code==='provider_budget_exhausted')throw Error('Το διαθέσιμο υπόλοιπο επεξεργασίας έχει εξαντληθεί. Το κείμενό σου διατηρήθηκε· χρειάζεται ανανέωση του υπολοίπου του παρόχου.');
      }
      if(r.status===401)window.dispatchEvent(new Event('clinical:request-login'));
      const failure=Error(r.status===401?'Συνδέσου από την Πρόσβαση.':r.status===409?'Ο φάκελος άλλαξε. Άνοιξέ τον ξανά πριν συνεχίσεις.':
        r.status===502||r.status===503?'Η μετατροπή δεν είναι διαθέσιμη αυτή τη στιγμή. Το κείμενο διατηρήθηκε.':'Η ενέργεια δεν ολοκληρώθηκε. Δοκίμασε ξανά.');failure.status=r.status;throw failure;
    }return r.json();
  }
  // A title's relative interval is only a suggested date. The clinician
  // explicitly adopts it and saves; no NLP or silent deadline mutation.
  function dateSuggestion(description, sourceDate){
    if(typeof sourceDate!=='string'||! /^\d{4}-\d{2}-\d{2}$/.test(sourceDate))return null;
    const [year,month,day]=sourceDate.split('-').map(Number);
    const valid=new Date(Date.UTC(year,month-1,day,12));
    if(valid.getUTCFullYear()!==year||valid.getUTCMonth()!==month-1||valid.getUTCDate()!==day)return null;
    const words=String(description||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLocaleLowerCase('el');
    const match=words.match(/(?:^|\s)(?:σε|μετα\s+απο)\s+(\d+(?:[,.]\d+)?)\s+(μηνα|μηνες|χρονο|χρονια|ετος|ετη)(?=$|[^\p{L}])/u);
    if(!match)return null;
    const quantity=Number(match[1].replace(',','.')),unit=match[2];
    if(!Number.isFinite(quantity)||quantity<=0)return null;
    const years=/^(χρονο|χρονια|ετος|ετη)$/.test(unit);
    if(quantity>(years?10:120))return null;
    const totalMonths=years?quantity*12:quantity;
    let whole=Math.floor(totalMonths),fraction=totalMonths-whole;
    // A half-month has no exact calendar definition; suggest ~15 days.
    if(!(Math.abs(fraction)<1e-9||Math.abs(fraction-0.5)<1e-9))return null;
    const first=new Date(Date.UTC(year,month-1+whole,1,12));
    const y=first.getUTCFullYear(),m=first.getUTCMonth();
    const last=new Date(Date.UTC(y,m+1,0,12)).getUTCDate();
    const result=new Date(Date.UTC(y,m,Math.min(day,last)+(fraction>0.25?15:0),12));
    return {date:result.toISOString().slice(0,10),
      approximate:fraction>0.25, sourceDate};
  }
  // Shared *non-authoritative* suggestion helper for chart and Obligations UI.
  window.CockpitObligationDateSuggestion=dateSuggestion;
  function pendingControls(disabled){for(const b of chart.querySelectorAll('.pending-checkpoint-actions button,.pending-checkpoint-actions select,.pending-stage button,.pending-open,.pending-metadata-editor input,.pending-metadata-editor select,.pending-metadata-editor button'))b.disabled=disabled;}
  function resetDraft(){discardAudio();amendment=null;pendingBindings=new Map();composerTitle.textContent='Νέα επίσκεψη';amendmentNote.hidden=true;saveButton.textContent='Αποθήκευση επίσκεψης και υποχρεώσεων';closeComposer.disabled=false;chart.hidden=false;promptStatus.textContent='';pendingControls(false);draftGeneration++;contextId=null;candidate=null;source='';busy=false;input.value='';review.replaceChildren();previews.replaceChildren();feedback.textContent='';saveButton.hidden=true;extractButton.disabled=false;input.disabled=false;visitDate.disabled=false;audioPanel.hidden=false;audioReview.hidden=true;audioConfirm.checked=false;audioFocused.replaceChildren();audioDraft=false;audioReviewUnits=[];audioSuggestions=[];audioSuggestionDraft='';audioPreviewed=null;audioContextExpiresAt='';audioReady=false;audioStart.disabled=true;audioStop.disabled=true;informed.checked=false;audioStatus.textContent='Έλεγχος διαθεσιμότητας…';}
  function reset(){for(const dialog of chart.querySelectorAll('dialog[open]'))dialog.close();pendingBusy=false;chartPending=[];chartScroll=0;chartReturnPoint=null;journeyObserver?.disconnect();journeyObserver=null;generation++;patientId=null;chart.replaceChildren();resetDraft();composer.hidden=true;}
  function facts(parent,title,items){if(!items?.length)return;el(parent,'h4',title);const list=el(parent,'ul');for(const i of items)el(list,'li',typeof i==='string'?i:i.text);}
  function showPending(id,item,editMetadata=false){
    if(!composer.hidden||pendingBusy||patientId!==id)return;
    chart.querySelector('.patient-pending-dialog')?.remove();
    const dialog=el(chart,'dialog','','patient-journey-preview patient-pending-dialog');dialog.setAttribute('aria-label','Ενημέρωση υποχρέωσης');
    button(dialog,'Κλείσιμο',()=>dialog.close()).classList.add('patient-journey-close');
    el(dialog,'h3','Ενημέρωση υποχρέωσης');renderPending(el(dialog,'div'),id,[item]);
    if(editMetadata)dialog.querySelector('.pending-metadata-editor')?.setAttribute('open','');
    dialog.onclick=e=>{if(e.target===dialog)dialog.close();};dialog.onclose=()=>dialog.remove();dialog.showModal();
  }
  function renderPending(target,id,items,branch=false){
    target.replaceChildren();if(!items.length)el(target,'p','Δεν έχουν καταγραφεί υποχρεώσεις.');
    for(const item of items){const row=el(target,'article','','synthetic-pending-row'),copy=el(row,'div');
      row.dataset.pendingId=item.pending_id;
      if(branch){const title=button(copy,item.description,()=>showPending(id,item));title.classList.add('pending-open');title.setAttribute('aria-haspopup','dialog');}else el(copy,'strong',item.description);
      if(!branch||!item.checkpoints||item.checkpoints.steps.length>1)el(copy,'p',(statuses[item.status]||item.status)+' · '+(roles[item.responsible_role]||item.responsible_role));
      if(item.review_date)el(copy,'p','Επανέλεγχος: '+date(item.review_date));
      const progress=item.checkpoints;
      const actions=el(copy,'div','','pending-checkpoint-actions');let lastStageControls=null;
      async function mutate(path,body){
        if(!composer.hidden||pendingBusy||patientId!==id)return;pendingBusy=true;pendingControls(true);
        const captureOpen=$('captureOpen');if(captureOpen)captureOpen.disabled=true;
        const ticket=generation;for(const b of copy.querySelectorAll('button'))b.disabled=true;
        const status=el(copy,'p','Αποθήκευση…');status.setAttribute('role','status');
        try{await request('/patient/'+encodeURIComponent(id)+'/pending/'+encodeURIComponent(item.pending_id)+path,{...body,idempotency_key:crypto.randomUUID()});
          if(ticket!==generation)return;await open(id);window.dispatchEvent(new Event('cockpit:clinical-updates-changed'));
        }catch(e){if(ticket!==generation)return;status.textContent=e.message+' Έλεγξε την αποθηκευμένη κατάσταση.';button(copy,'Ανανέωση κατάστασης',()=>{if(composer.hidden)open(id);});}
        finally{if(ticket===generation){pendingBusy=false;if(captureOpen)captureOpen.disabled=false;}}
      }
      function stage(parent,step){
        parent=el(parent,'div','','pending-stage');parent.dataset.state=step.state;
        if(branch){const dot=button(parent,'',()=>showPending(id,item));dot.className='patient-followthrough-dot pending-open';dot.setAttribute('aria-label',step.question);dot.setAttribute('aria-haspopup','dialog');}
        el(parent,'p',step.state==='confirmed'?'✓ '+step.meaning:step.question);
        if(step.state==='not_yet')el(parent,'small','Όχι ακόμη · καταγράφηκε '+date(step.observed_not_yet_at));
        if(!item.checkpoint_write_allowed||item.status==='cancelled')return;
        const controls=el(parent,'div','','pending-checkpoint-actions');lastStageControls=controls;
        const act=(observation)=>mutate('/checkpoint',{checkpoint_key:step.key,observation,expected_revision:progress.revision,retracts_event_id:observation==='retracted'?step.event_id:null});
        if(step.state==='confirmed')button(controls,'Αναίρεση',()=>act('retracted'));
        else{button(controls,'Ναι',()=>act('yes_confirmed'));button(controls,'Όχι ακόμη',()=>act('not_yet_observed'));}
      }
      if(progress&&branch){
        actions.classList.add('patient-followthrough-rail');actions.setAttribute('aria-label','Βήματα υποχρέωσης');
        for(const step of progress.steps)stage(actions,step);
        const end=el(actions,'div','','patient-followthrough-end');el(end,'span','','patient-followthrough-stop');el(end,'small','Τέλος');
        end.dataset.complete=String(progress.completed);
      }else if(progress){
        for(const step of progress.steps.filter(s=>s.state==='confirmed'))stage(actions,step);
        if(progress.next_checkpoint)stage(actions,progress.next_checkpoint);
        const later=progress.steps.filter(s=>s.state!=='confirmed'&&s.key!==progress.next_checkpoint?.key);
        if(later.length){const other=el(copy,'details');el(other,'summary','Έγινε ήδη επόμενο βήμα;');for(const step of later)stage(other,step);}
      }else if(item.checkpoint_write_allowed&&['open','awaiting_patient','awaiting_result'].includes(item.status)){
        el(actions,'p','Ολοκληρώθηκε η ενέργεια;');
        for(const [label,observation]of [['Ναι','yes_confirmed'],['Όχι ακόμη','not_yet_observed']])
          button(actions,label,()=>mutate('/checkpoint-plan',{expected_revision:item.lifecycle?.revision||0,template:'completion',first_observation:observation}));

      }else if(!item.checkpoint_storage_ready)el(copy,'small','Η ενημέρωση των βημάτων δεν έχει ενεργοποιηθεί ακόμη.');
      else if(!item.checkpoint_write_allowed)el(copy,'small','Για ενημέρωση χρειάζεται η πρόσβαση γιατρού.');

      if(branch&&!progress){
        const existing=[...actions.childNodes];actions.classList.add('patient-followthrough-rail');
        const step=el(actions,'div','','pending-stage');step.dataset.state='unknown';const dot=button(step,'',()=>showPending(id,item));dot.className='patient-followthrough-dot pending-open';dot.setAttribute('aria-label','Ενημέρωση: '+item.description);dot.setAttribute('aria-haspopup','dialog');
        for(const child of existing)step.append(child);
        if(!existing.length)el(step,'p',statuses[item.status]||item.status);
        const end=el(actions,'div','','patient-followthrough-end');end.dataset.complete=String(item.status==='resolved');el(end,'span','','patient-followthrough-stop');el(end,'small',item.status==='resolved'?'Ολοκληρώθηκε':item.status==='cancelled'?'Ακυρώθηκε':'Τέλος');
      }
      if(item.checkpoint_write_allowed){
        const tools=lastStageControls||el(copy,'div','','pending-checkpoint-actions');
        if(item.status==='cancelled'&&item.lifecycle?.cancellation_event_id){
          button(tools,'Επαναφορά υποχρέωσης',()=>mutate('/lifecycle',{action:'restore',expected_revision:item.lifecycle.revision,cancellation_event_id:item.lifecycle.cancellation_event_id}));
          const permanent=button(tools,'Μόνιμη διαγραφή',()=>{permanent.hidden=true;warning.hidden=false;});
          const warning=el(tools,'div','','pending-removal-confirmation');warning.hidden=true;
          el(warning,'p','Η υποχρέωση θα αφαιρεθεί οριστικά και δεν θα μπορεί να επανέλθει. Η αποθηκευμένη επίσκεψη διατηρείται.');
          button(warning,'Οριστική διαγραφή',()=>mutate('/lifecycle',{action:'delete_permanently',expected_revision:item.lifecycle.revision,cancellation_event_id:item.lifecycle.cancellation_event_id}));
          button(warning,'Διατήρηση',()=>{warning.hidden=true;permanent.hidden=false;});
        }else if(['open','awaiting_patient','awaiting_result','resolved'].includes(item.status)){
          const remove=button(tools,branch?'Διαγραφή':'Διαγραφή υποχρέωσης',()=>{remove.hidden=true;confirmation.hidden=false;});
          remove.setAttribute('aria-label','Διαγραφή υποχρέωσης');
          const confirmation=el(tools,'div','','pending-removal-confirmation');confirmation.hidden=true;
          el(confirmation,'p','Να διαγραφεί αυτή η υποχρέωση; Μπορείς να την επαναφέρεις.');
          button(confirmation,'Διαγραφή',()=>mutate('/lifecycle',{action:'cancel',expected_revision:item.lifecycle?.revision||0}));
          button(confirmation,'Διατήρηση',()=>{confirmation.hidden=true;remove.hidden=false;});
        }
      }
      if(!branch && item.checkpoint_write_allowed &&
          ['open','awaiting_patient','awaiting_result','resolved'].includes(item.status) &&
          target.closest('.patient-pending-dialog')){
        const editor=el(copy,'details','','pending-metadata-editor');
        el(editor,'summary','Κατηγορία / Προθεσμία');
        const fields=el(editor,'div','','pending-metadata-fields');
        const catLabel=el(fields,'label','Κατηγορία');
        const category=el(catLabel,'select');category.setAttribute('aria-label','Κατηγορία υποχρέωσης');
        const known=[
          ['test','Εξετάσεις'],['result','Αποτελέσματα'],['document','Έγγραφα'],
          ['follow_up','Επανεξέταση'],['referral','Παραπομπή'],['other','Λοιπές']
        ];
        const initialCategory=item.item_type==='investigation'?'test':item.item_type;
        if(!known.some(v=>v[0]===initialCategory)){
          const legacy=el(category,'option','Άλλη υφιστάμενη κατηγορία');legacy.value=initialCategory;
        }
        for(const [key,label] of known){
          const option=el(category,'option',label);option.value=key;
        }
        category.value=initialCategory;
        const dateLabel=el(fields,'label','Προθεσμία');
        const due=el(dateLabel,'input');due.type='date';due.value=item.review_date||'';
        due.setAttribute('aria-label','Προθεσμία υποχρέωσης');
        const suggestion=!item.review_date?dateSuggestion(item.description,item.source_date):null;
        if(suggestion){
          const proposed=el(editor,'div','','pending-metadata-suggestion');
          el(proposed,'small',suggestion.approximate?
            'Ενδεικτική ημερομηνία (περίπου), από την καταγεγραμμένη επίσκεψη '+date(suggestion.sourceDate):
            'Πρόταση από τίτλο, με αφετηρία την καταγεγραμμένη επίσκεψη '+date(suggestion.sourceDate));
          button(proposed,'Χρήση πρότασης · '+date(suggestion.date),()=>{due.value=suggestion.date;due.focus();});
        }
        el(editor,'p','Η προθεσμία είναι ανεξάρτητη από το αν η υποχρέωση παραμένει ανοικτή. Το αρχικό κείμενο της επίσκεψης δεν αλλάζει.','clinical-muted');
        const buttons=el(editor,'div','','pending-metadata-actions');
        const validation=el(editor,'p','','clinical-muted');validation.setAttribute('role','status');
        button(buttons,'Αποθήκευση',()=>{
          const chosen=category.value===initialCategory?item.item_type:category.value;
          const reviewDate=due.value||null;
          if(chosen===item.item_type&&reviewDate===(item.review_date||null)){
            validation.textContent='Δεν έχουν γίνει αλλαγές.';return;
          }
          validation.textContent='';
          mutate('/metadata',{item_type:chosen,review_date:reviewDate,
            expected_revision:item.lifecycle?.revision??0});
        });
        button(buttons,'Χωρίς προθεσμία',()=>{due.value='';due.focus();});
        button(buttons,'Άκυρο',()=>{
          category.value=initialCategory;due.value=item.review_date||'';validation.textContent='';
          editor.open=false;
        });
      }
      if(!branch){
      if(item.source_excerpt)el(copy,'blockquote',item.source_excerpt);
      button(copy,'Δημιουργήθηκε στην επίσκεψη · '+date(item.source_date),async()=>{
        if(patientId!==id){if(!canLeave())return;await open(id);}showVisit(id,item.source_encounter_id);
      });
      }
    }
  }
  function showVisit(id,encounter){
    if(patientId!==id||pendingBusy||!canLeave())return false;
    const reader=chart.querySelector('.patient-visit-reader'),overview=chart.querySelector('.patient-chart-overview');
    const visit=[...chart.querySelectorAll('.patient-visit')].find(n=>n.id==='visit-'+encounter);
    if(!reader||!overview||!visit)return false;
    if(!overview.hidden)chartScroll=window.scrollY;
    chartReturnPoint=[...overview.querySelectorAll('.patient-journey-node')].find(n=>n.dataset.encounterId===encounter)?.querySelector('.patient-journey-point');
    for(const dialog of chart.querySelectorAll('dialog[open]'))dialog.close();
    for(const other of reader.querySelectorAll('.patient-visit')){other.hidden=other!==visit;other.open=other===visit;}
    overview.hidden=true;reader.hidden=false;
    visit.querySelector('summary').focus({preventScroll:true});reader.scrollIntoView({block:'start'});return true;
  }
  function returnToChart(){
    const reader=chart.querySelector('.patient-visit-reader'),overview=chart.querySelector('.patient-chart-overview');
    if(!reader||!overview||pendingBusy||!canLeave())return;
    for(const visit of reader.querySelectorAll('.patient-visit')){visit.hidden=true;visit.open=false;}
    reader.hidden=true;overview.hidden=false;chartReturnPoint?.focus({preventScroll:true});
    window.scrollTo({top:chartScroll,behavior:'auto'});
  }
  function canLeave(){if(!composer.hidden){feedback.textContent='Κλείσε ή αποθήκευσε την τρέχουσα καταγραφή πριν αλλάξεις ασθενή.';composer.scrollIntoView({behavior:'smooth'});return false;}return true;}
  function discardAudio(){
    audioToken++;audioDiscard=true;audioAbort?.abort();audioAbort=null;
    clearInterval(audioTimer);audioTimer=null;
    if(audioRecorder&&audioRecorder.state!=='inactive'){try{audioRecorder.stop();}catch{}}
    audioStream?.getTracks().forEach(track=>track.stop());audioStream=null;audioRecorder=null;audioChunks=[];
  }
  async function loadAudioCapability(){
    const ticket=generation,draftTicket=draftGeneration,id=patientId;
    audioReady=false;audioStart.disabled=true;
    if(!navigator.mediaDevices?.getUserMedia||typeof MediaRecorder==='undefined'){
      audioStatus.textContent='Το μικρόφωνο δεν υποστηρίζεται από αυτό το πρόγραμμα περιήγησης.';return;
    }
    try{const result=await request('/visit-capture/audio/capability');
      if(ticket!==generation||draftTicket!==draftGeneration||id!==patientId||composer.hidden)return;
      audioReady=result.enabled===true;
      audioStatus.textContent=audioReady?'Έτοιμο για ηχογράφηση.':'Η ασφαλής επεξεργασία ήχου δεν έχει ενεργοποιηθεί ακόμη.';
      audioStart.disabled=!audioReady||!informed.checked;
    }catch(e){if(ticket===generation&&draftTicket===draftGeneration)audioStatus.textContent=e.message;}
  }
  function beginVisit(){if(!patientId||pendingBusy)return;resetDraft();composer.hidden=false;pendingControls(true);visitDate.value=new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Nicosia',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());loadAudioCapability();input.focus();composer.scrollIntoView({behavior:'smooth'});}
  async function startAudio(){
    if(!audioReady||!informed.checked||!patientId||composer.hidden||amendment||audioRecorder||busy||candidate||!visitDate.value)return;
    draftGeneration++;contextId=null;source='';input.value='';review.replaceChildren();previews.replaceChildren();saveButton.hidden=true;
    const ticket=generation,draftTicket=draftGeneration,id=patientId,token=++audioToken;
    audioDiscard=false;audioStart.disabled=true;audioStatus.textContent='Άνοιγμα μικροφώνου…';
    let stream=null;
    try{
      const mime=['audio/webm;codecs=opus','audio/mp4','audio/webm'].find(type=>MediaRecorder.isTypeSupported(type));
      if(!mime)throw Error('Δεν υποστηρίζεται ασφαλής μορφή ήχου.');
      stream=await navigator.mediaDevices.getUserMedia({audio:true});
      if(!informed.checked)throw Error('Η επιβεβαίωση ενημέρωσης αποσύρθηκε. Ο ήχος απορρίφθηκε.');
      if(ticket!==generation||draftTicket!==draftGeneration||id!==patientId||composer.hidden||token!==audioToken)throw Error('Η καρτέλα άλλαξε.');
      const context=await request('/visit-capture/context',{patient_id:id,client_session_id:session});
      if(!informed.checked)throw Error('Η επιβεβαίωση ενημέρωσης αποσύρθηκε. Ο ήχος απορρίφθηκε.');
      if(ticket!==generation||draftTicket!==draftGeneration||id!==patientId||composer.hidden||token!==audioToken||context.patient_id!==id)throw Error('Η καρτέλα άλλαξε.');
      contextId=context.context_id;audioContextExpiresAt=context.expires_at||'';
      audioChunks=[];
      const recorder=new MediaRecorder(stream,{mimeType:mime,audioBitsPerSecond:64000});
      recorder.ondataavailable=event=>{if(token===audioToken&&event.data?.size)audioChunks.push(event.data);};
      recorder.onerror=()=>{if(token===audioToken){discardAudio();audioStatus.textContent='Η ηχογράφηση διακόπηκε. Δεν κρατήθηκε ήχος.';input.disabled=false;extractButton.disabled=false;visitDate.disabled=false;audioStop.disabled=true;audioStart.disabled=!audioReady||!informed.checked;}};
      recorder.onstop=()=>{
        stream.getTracks().forEach(track=>track.stop());clearInterval(audioTimer);audioTimer=null;
        if(token!==audioToken||audioDiscard)return;
        const blob=new Blob(audioChunks,{type:mime.split(';')[0]});audioChunks=[];audioRecorder=null;audioStream=null;
        finishAudio(blob,token,id,ticket,draftTicket);
      };
      if(!informed.checked)throw Error('Η επιβεβαίωση ενημέρωσης αποσύρθηκε. Ο ήχος απορρίφθηκε.');
      audioStream=stream;audioRecorder=recorder;recorder.start(1000);audioStarted=Date.now();audioStop.disabled=false;
      input.disabled=true;extractButton.disabled=true;visitDate.disabled=true;
      audioStatus.textContent='● Ηχογράφηση · 0:00';
      audioTimer=setInterval(()=>{if(token!==audioToken)return;const seconds=Math.floor((Date.now()-audioStarted)/1000);
        audioStatus.textContent='● Ηχογράφηση · '+Math.floor(seconds/60)+':'+String(seconds%60).padStart(2,'0');
        if(seconds>=1800){discardAudio();audioStatus.textContent='Το όριο ηχογράφησης έληξε. Ο ήχος απορρίφθηκε.';audioStop.disabled=true;input.disabled=false;extractButton.disabled=false;visitDate.disabled=false;}
      },1000);
    }catch(e){stream?.getTracks().forEach(track=>track.stop());if(token===audioToken){audioStream=null;audioRecorder=null;audioStatus.textContent=e.message||'Δεν ξεκίνησε η ηχογράφηση.';audioStart.disabled=!audioReady||!informed.checked;}}
  }
  function stopAudio(){if(!audioRecorder||audioRecorder.state==='inactive')return;audioStop.disabled=true;audioStatus.textContent='Λήξη και επεξεργασία…';audioRecorder.stop();}
  async function finishAudio(blob,token,id,ticket,draftTicket){
    if(token!==audioToken)return;
    if(!informed.checked){audioStatus.textContent='Η επιβεβαίωση ενημέρωσης αποσύρθηκε. Ο ήχος απορρίφθηκε.';input.disabled=false;extractButton.disabled=false;visitDate.disabled=false;return;}
    if(!blob.size||blob.size>MAX_AUDIO_BYTES){audioStatus.textContent='Η ηχογράφηση υπερβαίνει το όριο μεγέθους. Ο ήχος απορρίφθηκε.';input.disabled=false;extractButton.disabled=false;visitDate.disabled=false;return;}
    busy=true;audioAbort=new AbortController();
    try{
      const renewed=await request('/visit-capture/context/renew',{prior_context_id:contextId,patient_id:id,client_session_id:session});
      if(token!==audioToken||ticket!==generation||draftTicket!==draftGeneration||id!==patientId||renewed.patient_id!==id)throw Error('Η καρτέλα άλλαξε.');
      contextId=renewed.context_id;audioContextExpiresAt=renewed.expires_at||'';
      const usedContext=contextId,usedDate=visitDate.value;
      if(!informed.checked)throw Error('Η επιβεβαίωση ενημέρωσης αποσύρθηκε. Ο ήχος απορρίφθηκε.');
      const r=await fetch('/clinical/visit-capture/audio/draft',{method:'POST',credentials:'same-origin',cache:'no-store',
        headers:{'Content-Type':blob.type,'X-Visit-Context':usedContext,'X-Encounter-Date':usedDate,'X-Visit-Retention-Acknowledged':'standard'},body:blob,signal:audioAbort.signal});
      if(!r.ok){if(r.status===401)window.dispatchEvent(new Event('clinical:request-login'));throw Error(r.status===409?'Ο φάκελος άλλαξε. Άνοιξέ τον ξανά.':'Η ηχογράφηση δεν επεξεργάστηκε. Δοκίμασε ξανά.');}
      const result=await r.json();
      if(token!==audioToken||ticket!==generation||draftTicket!==draftGeneration||id!==patientId||contextId!==usedContext||visitDate.value!==usedDate)return;
      if(result.patient_id!==id||result.context_id!==usedContext||result.candidate?.schema_version!=='visit_capture_candidate_v1'||
          result.candidate?.encounter_date!==usedDate||!result.candidate?.source_bindings?.some(binding=>binding.role==='web_visit_audio')||!Array.isArray(result.review_units)||!Array.isArray(result.suggested_next_checks))throw Error('Η ταυτότητα ή η καταγραφή δεν επαληθεύτηκε.');
      candidate=result.candidate;audioDraft=true;audioReviewUnits=result.review_units;audioSuggestions=result.suggested_next_checks.slice(0,6);
      audioSuggestionDraft=audioSuggestions.map(item=>item.text).join('\n');audioPreviewed=null;
      bindClinician();audioPatient.textContent=$('visitSelectedName').textContent;
      audioReview.hidden=false;audioConfirm.checked=false;drawReview();renderAudioFocused();
      saveButton.hidden=false;saveButton.disabled=true;audioStatus.textContent='Το κλινικό κείμενο είναι έτοιμο για έλεγχο. Το προσωρινό αντίγραφο ήχου στον server αφαιρέθηκε.';
      feedback.textContent='Έλεγξε και διόρθωσε μόνο ό,τι χρειάζεται. Οι αμφίβολες φράσεις παραμένουν εμφανείς στις Αβεβαιότητες.';
    }catch(e){if(token===audioToken&&ticket===generation&&draftTicket===draftGeneration){audioStatus.textContent=e.name==='AbortError'?'Η επεξεργασία ακυρώθηκε.':e.message;}}
    finally{if(token===audioToken){busy=false;audioAbort=null;input.disabled=false;extractButton.disabled=false;visitDate.disabled=false;audioStart.disabled=!audioReady||!informed.checked||Boolean(candidate);}}
  }
  informed.onchange=()=>{if(!informed.checked&&audioRecorder){discardAudio();audioStop.disabled=true;input.disabled=false;extractButton.disabled=false;visitDate.disabled=false;audioStatus.textContent='Η επιβεβαίωση ενημέρωσης αποσύρθηκε. Ο ήχος απορρίφθηκε.';}audioStart.disabled=!audioReady||!informed.checked||Boolean(audioRecorder)||busy;};
  audioConfirm.onchange=()=>{audioPreviewed=null;saveButton.disabled=true;previews.replaceChildren();if(audioConfirm.checked)preview();};
  async function open(id){
    window.dispatchEvent(new Event('cockpit:patient-opening'));
    reset();const ticket=generation;el(chart,'p','Φόρτωση καρτέλας…');
    $('visitWorkspaceHome').hidden=true;$('visitWorkspacePatient').hidden=false;
    for(const name of ['visitIdentityStatus','visitIdentityNote','visitBookingConfirmation','visitChoosePatient','visitDiaComposer','visitDiaPreview'])$(name).hidden=true;
    document.querySelector('.clinical-visit-actions').hidden=true;
    try{const data=await request('/patient/'+encodeURIComponent(id)+'/chart');if(ticket!==generation)return;
      patientId=id;chartPending=data.pending;chart.replaceChildren();$('visitSelectedName').textContent=data.demographics.full_name||data.demographics.name||'Κλινικός φάκελος';
      $('visitSelectedReason').textContent=data.demographics.dob?'Γέννηση: '+date(data.demographics.dob):'';
      $('visitSelectedSource').textContent=data.demographics.synthetic_scenario?'ΔΟΚΙΜΑΣΤΙΚΗ ΚΑΡΤΕΛΑ':'ΚΑΡΤΕΛΑ ΑΣΘΕΝΟΥΣ';
      const chartOverview=el(chart,'div','','patient-chart-overview');
      const heading=el(chartOverview,'div','','patient-chart-heading');el(heading,'h2','Κλινική συνέχεια');
      button(heading,'Νέα επίσκεψη · Σύνοψη Dia',beginVisit).id='captureOpen';
      const grid=el(chartOverview,'div','','patient-chart-grid'),overview=el(grid,'section','','patient-chart-card');el(overview,'h3','Κλινική εικόνα');
      const latest=data.visits[0];const c=latest?.clinical;
      if(c){el(overview,'p','Από την επίσκεψη '+date(latest.date));facts(overview,'Λόγος παρακολούθησης',[c.reason_for_visit]);facts(overview,'Εκτίμηση',c.clinical_impression);facts(overview,'Σχέδιο',c.decisions);}
      else el(overview,'p',latest?'Η προηγούμενη επίσκεψη δεν διαθέτει δομημένη σύνοψη.':'Η πρώτη επίσκεψη θα εμφανιστεί εδώ μετά την αποθήκευση.');
      const pending=el(grid,'section','','patient-chart-card');el(pending,'h3','Υποχρεώσεις');
      const activePending=data.pending.filter(p=>p.status!=='cancelled');
      if(!activePending.length)el(pending,'p','Δεν υπάρχουν ενεργές υποχρεώσεις.');
      function obligationLink(parent,item){const row=el(parent,'div','','patient-obligation-link');
        const b=button(row,item.description,()=>showPending(id,item));b.classList.add('pending-open');b.setAttribute('aria-haspopup','dialog');
        el(row,'small',statuses[item.status]||item.status);
      }
      for(const item of activePending)obligationLink(pending,item);
      const deleted=data.pending.filter(p=>p.status==='cancelled');
      if(deleted.length){const history=el(pending,'details','','patient-deleted-obligations');el(history,'summary','Διαγραμμένες υποχρεώσεις · '+deleted.length);for(const item of deleted)obligationLink(history,item);}
      const timeline=el(chartOverview,'section','','patient-timeline');el(timeline,'h3','Πορεία ασθενούς');
      const scrollHint=el(timeline,'small','Σύρε οριζόντια για τη συνέχεια →','patient-journey-scroll-hint');scrollHint.hidden=true;
      const rail=el(timeline,'div','','patient-journey-rail');rail.setAttribute('aria-label','Επισκέψεις και συνέχεια');
      const track=el(rail,'div','','patient-journey-track');
      const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.classList.add('patient-journey-lines');svg.setAttribute('aria-hidden','true');track.append(svg);
      const selected=el(timeline,'dialog','','patient-journey-preview');selected.setAttribute('aria-label','Σύνοψη επίσκεψης');
      selected.onclick=e=>{if(e.target===selected)selected.close();};
      selected.onclose=()=>{for(const dot of rail.querySelectorAll('.patient-journey-point'))dot.setAttribute('aria-expanded','false');};
      const points=[];
      function selectPoint(point,b){
        for(const dot of rail.querySelectorAll('.patient-journey-point'))dot.setAttribute('aria-pressed',String(dot===b));
        selected.replaceChildren();button(selected,'Κλείσιμο',()=>selected.close()).classList.add('patient-journey-close');el(selected,'small',date(point.date));el(selected,'h4',point.title);
        for(const text of point.summary||[])el(selected,'p',text);
        button(selected,'Άνοιγμα',()=>showVisit(id,point.source));
        b.setAttribute('aria-expanded','true');selected.showModal();
      }
      for(const visit of data.visits)points.push({date:visit.date,title:visit.clinical?.reason_for_visit?.text||'Καταγεγραμμένη επίσκεψη',source:visit.encounter_id,summary:[...(visit.clinical?.findings||[]).slice(0,2).map(f=>'Ευρήματα: '+f.text),...(visit.clinical?.clinical_impression||[]).slice(0,2).map(f=>'Εκτίμηση: '+f.text),...(visit.clinical?.decisions||[]).slice(0,2).map(f=>'Απόφαση: '+f.text),...data.pending.filter(p=>p.source_encounter_id===visit.encounter_id).map(p=>'Υποχρέωση: '+p.description)]});
      points.sort((a,b)=>a.date.localeCompare(b.date));
      for(const point of points){
        const origin=activePending.filter(p=>p.source_encounter_id===point.source);
        const node=el(track,'section','','patient-journey-node');node.dataset.encounterId=point.source;
        node.classList.toggle('has-followthrough',Boolean(origin.length));
        const b=button(node,'',()=>selectPoint(point,b));b.className='clinical-secondary patient-journey-point';b.setAttribute('aria-pressed','false');b.setAttribute('aria-haspopup','dialog');b.setAttribute('aria-expanded','false');
        el(b,'small',date(point.date));el(b,'span',point.title);el(b,'span','','patient-journey-dot');
        if(!origin.length)continue;
        const branch=el(node,'details','','patient-followthrough-branch');branch.id='followthrough-'+point.source;branch.open=true;
        const heading=el(branch,'summary');
        const active=origin.filter(p=>!['resolved','cancelled'].includes(p.status)).length;
        el(heading,'small',active?'Συνέχεια επίσκεψης · '+active+' σε εκκρεμότητα':'Συνέχεια επίσκεψης · Κλειστή συνέχεια');
        renderPending(el(branch,'div'),id,origin,true);
        branch.ontoggle=()=>requestAnimationFrame(drawJourney);
      }
      if(!points.length)el(timeline,'p','Δεν έχουν καταγραφεί επισκέψεις.');
      // One drawing: primary line and each finite fork use their actual source-dot coordinates.
      function drawJourney(){
        if(!track.isConnected)return;
        svg.replaceChildren();scrollHint.hidden=track.scrollWidth<=rail.clientWidth;const bounds=track.getBoundingClientRect();
        svg.setAttribute('width',track.scrollWidth);svg.setAttribute('height',track.offsetHeight);
        const center=n=>{const r=n.getBoundingClientRect();return {x:r.left-bounds.left+r.width/2,y:r.top-bounds.top+r.height/2};};
        function path(d,cls,source){const p=document.createElementNS(svg.namespaceURI,'path');p.setAttribute('d',d);p.setAttribute('class',cls);if(source)p.dataset.source=source;svg.append(p);}
        const nodes=[...track.querySelectorAll('.patient-journey-node')];if(!nodes.length)return;
        const first=center(nodes[0].querySelector('.patient-journey-dot')),last=center(nodes.at(-1).querySelector('.patient-journey-dot'));
        path(`M ${first.x} ${first.y} H ${last.x}`,'journey-primary');
        path(`M ${last.x} ${last.y} H ${track.scrollWidth-16}`,'journey-future');
        for(const node of nodes){const from=center(node.querySelector('.patient-journey-dot'));
          for(const stages of node.querySelectorAll('.patient-followthrough-rail')){
            if(!stages.closest('.patient-followthrough-branch').open)continue;
            const dot=stages.querySelector('.patient-followthrough-dot'),stop=stages.querySelector('.patient-followthrough-stop');
            if(!dot||!stop||!dot.getClientRects().length)continue;
            const to=center(dot),end=center(stop);
            path(`M ${from.x} ${from.y} V ${to.y-12} Q ${from.x} ${to.y} ${from.x+12} ${to.y} H ${end.x}`,'journey-fork',node.dataset.encounterId);
          }
        }
      }
      journeyObserver=new ResizeObserver(drawJourney);journeyObserver.observe(track);
      for(const node of track.querySelectorAll('.patient-journey-node'))journeyObserver.observe(node);
      requestAnimationFrame(drawJourney);

      const reader=el(chart,'section','','patient-visit-reader');reader.hidden=true;reader.setAttribute('aria-label','Επιλεγμένη επίσκεψη');
      button(reader,'Επιστροφή στην καρτέλα',returnToChart).id='patientVisitBack';
      for(const visit of data.visits){const block=el(reader,'details','','patient-visit');block.id='visit-'+visit.encounter_id;block.hidden=true;
        const summary=el(block,'summary');el(summary,'span',date(visit.date));el(summary,'strong',visit.clinical?.reason_for_visit?.text||'Καταγεγραμμένη επίσκεψη');
        const origin=data.pending.filter(p=>p.source_encounter_id===visit.encounter_id);for(const p of origin)el(summary,'small','Δημιουργήθηκε υποχρέωση: '+p.description);
        const body=el(block,'div','','patient-visit-body');const v=visit.clinical;
        if(v){button(body,'Επεξεργασία επίσκεψης',()=>beginAmend(visit.encounter_id)).classList.add('visit-edit');
          if(visit.revision)el(body,'small','Διορθωμένη καταγραφή · '+visit.revision);
          if(visit.amendments?.length){const history=el(body,'details','','visit-amendment-history');el(history,'summary','Ιστορικό διορθώσεων');
            for(const prior of [...visit.amendments].reverse()){const d=el(history,'details');el(d,'summary','Προηγούμενη καταγραφή · '+date(prior.date)+' · διορθώθηκε '+date(prior.amended_at));el(d,'p',prior.detail,'clinical-preview-text');}}
        }
        if(!v){el(body,'p','Παλαιότερη καταγραφή χωρίς δομημένη σύνοψη.');continue;}
        facts(body,'Ιστορικό / μεταβολές',v.what_changed);facts(body,'Ευρήματα',v.findings);facts(body,'Κλινική εκτίμηση',v.clinical_impression);facts(body,'Αποφάσεις',v.decisions);
        facts(body,'Αγωγή',(v.medications||[]).map(m=>[({prescribed:'Έναρξη',continued:'Συνέχιση',stopped:'Διακοπή'})[m.action],m.name,m.dose,m.frequency,m.route,m.duration?m.duration.value+' '+m.duration.unit:null].filter(Boolean).join(' · ')));
        if(v.next_contact)facts(body,'Επόμενη επαφή',[ [v.next_contact.when,v.next_contact.time,v.next_contact.purpose,...v.next_contact.assess].filter(Boolean).join(' · ')]);
        facts(body,'Οδηγίες επανεπικοινωνίας',v.safety_net);facts(body,'Αβεβαιότητες',v.uncertainties);
        if(origin.length){el(body,'h4','Υποχρεώσεις που δημιουργήθηκαν');renderPending(el(body,'div'),id,origin);}
      }
      const url=new URL(location.href);url.searchParams.set('chart',id);history.replaceState(null,'',url);return id;
    }catch(e){if(ticket===generation){chart.replaceChildren();el(chart,'p',e.message);}}
  }
  const labels={reason_for_visit:'Λόγος επίσκεψης',visit_type:'Τύπος επίσκεψης',specialty:'Ειδικότητα',what_changed:'Μεταβολές',findings:'Ιστορικό και ευρήματα',history:'Ιστορικό',clinical_impression:'Κλινική εκτίμηση',options_discussed:'Επιλογές που συζητήθηκαν',recommendations:'Συστάσεις',completed_actions:'Ενέργειες που ολοκληρώθηκαν',patient_preferences:'Προτιμήσεις ασθενούς',follow_up:'Παρακολούθηση',administrative_notes:'Διοικητικές σημειώσεις',coding:'Κωδικοποίηση',gesy_coded:'Κωδικοί',coding_complete:'Πλήρης κωδικοποίηση',decisions:'Αποφάσεις',medications:'Φάρμακα',pending:'Προτεινόμενες υποχρεώσεις',next_contact:'Επόμενη επαφή',safety_net:'Οδηγίες επανεπικοινωνίας',uncertainties:'Αβεβαιότητες',text:'Περιγραφή',type:'Είδος',responsible_role:'Υπεύθυνος',certainty:'Βεβαιότητα',region:'Περιοχή',name:'Φάρμακο',action:'Ενέργεια',dose:'Δόση',frequency:'Συχνότητα',route:'Οδός',duration:'Διάρκεια',value:'Τιμή',unit:'Μονάδα',approximate:'Κατά προσέγγιση',item_type:'Είδος υποχρέωσης',description:'Περιγραφή',trigger:'Προϋπόθεση',status:'Κατάσταση',external_dependency:'Εξωτερική εξάρτηση',actor:'Από ποιον',condition:'Προϋπόθεση',review_date:'Ημερομηνία επανελέγχου',when:'Ημερομηνία',time:'Ώρα',purpose:'Σκοπός',assess:'Προς αξιολόγηση',code:'Κωδικός',system:'Σύστημα',title:'Τίτλος',direction:'Μεταβολή'};
  const audioLabels={history:'Ιστορικό και συμπτώματα',findings:'Ευρήματα / εξέταση',recommendations:'Θεραπεία και συστάσεις',decisions:'Αποφάσεις / πλάνο',pending:'Επόμενα βήματα / υποχρεώσεις',follow_up:'Παρακολούθηση / επόμενος έλεγχος'};
  const enums={responsible_role:roles,item_type:{investigation:'Εξέταση',test:'Εξέταση',result:'Αποτέλεσμα',document:'Έγγραφο',follow_up:'Επανέλεγχος',referral:'Παραπομπή',other:'Άλλο'},certainty:{certain:'Βέβαιο',uncertain:'Αβέβαιο',conflicting:'Αντικρουόμενο'},action:{prescribed:'Έναρξη',continued:'Συνέχιση',stopped:'Διακοπή'},status:{open:statuses.open,awaiting_patient:statuses.awaiting_patient,awaiting_result:statuses.awaiting_result},type:{therapy:'Θεραπεία',referral:'Παραπομπή',medication:'Αγωγή',investigation:'Εξέταση',advice:'Οδηγία',sick_leave:'Άδεια ασθενείας',admin:'Διοικητικό'}};
  function edited(obj){if(obj&&typeof obj==='object'&&!Array.isArray(obj)&&obj.provenance){if(!obj.provenance.includes('clinician_edit'))obj.provenance.push('clinician_edit');if('verification'in obj)obj.verification='clinician_confirmed';}previews.replaceChildren();if(audioDraft){audioConfirm.checked=false;audioPreviewed=null;saveButton.disabled=true;}}
  function fields(parent,obj){
    let extra=null;
    for(const [key,value] of Object.entries(obj)){
      if(['schema_version','source_identity_state','source_bindings','comparison_basis','encounter_date','provenance','verification'].includes(key))continue;
      if(key==='source_excerpt'){if(value)el(parent,'blockquote',value);continue;}
      if(Array.isArray(value)){const group=el(parent,key==='uncertainties'&&audioDraft?'details':'section');group.dataset.field=key;el(group,key==='uncertainties'&&audioDraft?'summary':'h4',labels[key]||key);const rows=el(group,'div');
        for(const item of [...value]){const row=el(rows,'div','','capture-field-group');
          if(typeof item==='object')fields(row,item);else{const l=el(row,'label',labels[key]||key);const t=el(l,'input');t.value=item;t.oninput=()=>{const index=Array.from(rows.querySelectorAll(':scope > .capture-field-group')).indexOf(row);value[index]=t.value;edited(obj);};}
          button(row,'Αφαίρεση',()=>{const index=Array.from(rows.querySelectorAll(':scope > .capture-field-group')).indexOf(row);if(index>=0)value.splice(index,1);row.remove();edited(obj);});
        }
        if(['what_changed','findings','clinical_impression','decisions','recommendations','completed_actions','patient_preferences','medications','pending','safety_net','uncertainties','gesy_coded','assess'].includes(key))
          button(group,'Προσθήκη · '+(labels[key]||key),()=>{value.push(newField(key));edited(obj);drawReview();});
        continue;}
      if(value&&typeof value==='object'){const group=el(parent,'fieldset');el(group,'legend',labels[key]||key);fields(group,value);if(['duration','external_dependency','next_contact'].includes(key))button(group,'Αφαίρεση · '+labels[key],()=>{obj[key]=null;edited(obj);drawReview();});continue;}
      // Empty optional objects cannot be edited as strings.
      if(value===null&&['duration','external_dependency','next_contact'].includes(key)){
        button(parent,'Προσθήκη · '+(labels[key]||key),()=>{obj[key]=key==='next_contact'?{when:null,time:null,purpose:'',assess:[],provenance:['clinician_edit']}:key==='duration'?{value:1,unit:'ημέρες',approximate:false}:{actor:'',condition:''};edited(obj);drawReview();});continue;}
      let destination=parent;
      if(['type','certainty','region','item_type','specialty','visit_type','system','coding_complete'].includes(key)){
        if(!extra){extra=el(parent,'details');el(extra,'summary','Λεπτομέρειες');}destination=extra;
      }
      const label=el(destination,'label',labels[key]||key);let control;
      const choices=key==='status'&&'text'in obj?{accepted:'Αποδέχθηκε',declined:'Αρνήθηκε',undecided:'Δεν αποφάσισε'}:enums[key];
      if(choices){control=el(label,'select');for(const [v,title]of Object.entries({...choices,...(value&&!choices[value]?{[value]:value}:{})})){const option=el(control,'option',title);option.value=v;}}
      else{control=el(label,['text','description','purpose'].includes(key)?'textarea':'input');if(typeof value==='boolean')control.type='checkbox';else if(typeof value==='number')control.type='number';else if(['review_date','when'].includes(key))control.type='date';}
      if(typeof value==='boolean')control.checked=value;else control.value=value??'';
      if(control.tagName==='TEXTAREA')control.rows=2;
      control.oninput=()=>{obj[key]=typeof value==='boolean'?control.checked:typeof value==='number'?Number(control.value):control.value||null;edited(obj);};
    }
  }
  function newField(key){
    const fact={text:'',provenance:['clinician_edit'],verification:'clinician_confirmed',certainty:'certain'};
    if(key==='pending')return {source_excerpt:null,item_type:'other',description:'',trigger:null,status:'open',responsible_role:'clinician',external_dependency:null,review_date:null,provenance:['clinician_edit']};
    if(key==='medications')return {name:'',action:'prescribed',dose:null,frequency:null,route:null,duration:null,provenance:['clinician_edit'],verification:'clinician_confirmed',certainty:'certain'};
    if(key==='gesy_coded')return {system:'ICD10',code:'',title:'',provenance:['clinician_edit']};
    if(key==='assess')return '';
    if(key==='decisions')return {...fact,type:'advice',responsible_role:null};
    if(key==='recommendations')return {...fact,type:'advice'};
    if(key==='patient_preferences')return {...fact,status:'undecided'};
    if(key==='findings')return {...fact,region:null};
    if(key==='what_changed')return {...fact,direction:'unchanged'};
    return fact;
  }
  function bindClinician(){if(!candidate.source_bindings.some(b=>b.source_ref==='clinician_edit'))candidate.source_bindings.push({source_ref:'clinician_edit',role:'clinician_edit'});}
  const audioCompactKeys=['reason_for_visit','what_changed','history','findings','clinical_impression','options_discussed','recommendations','completed_actions','decisions','patient_preferences','pending','follow_up','safety_net','administrative_notes','uncertainties'];
  const audioMetadataKeys=new Set(['schema_version','source_identity_state','source_bindings','comparison_basis','encounter_date']);
  function hasReviewContent(value){
    if(value===null||value===undefined||value===''||value===false)return false;
    if(Array.isArray(value))return value.some(hasReviewContent);
    if(typeof value==='object')return Object.entries(value).some(([key,item])=>!['provenance','verification','certainty'].includes(key)&&hasReviewContent(item));
    return true;
  }
  function markClinicianFact(fact){
    if(!fact||typeof fact!=='object')return;
    if(!Array.isArray(fact.provenance))fact.provenance=[];
    if(!fact.provenance.includes('clinician_edit'))fact.provenance.push('clinician_edit');
    if('verification'in fact)fact.verification='clinician_confirmed';
  }
  function audioItemText(key,item){return key==='pending'?item.description:item.text;}
  function syncAudioLines(key,value,raw){
    const lines=raw.split('\n').map(line=>line.trim()).filter(Boolean),available=value.map(item=>({item,used:false}));
    const next=lines.map((text,index)=>{
      let match=available.find(entry=>!entry.used&&entry.item&&typeof entry.item==='object'&&audioItemText(key,entry.item)===text);
      if(match){match.used=true;return match.item;}
      match=available[index]&&!available[index].used?available[index]:available.find(entry=>!entry.used);
      let fact;
      if(match&&match.item&&typeof match.item==='object'){match.used=true;fact={...match.item,provenance:[...(match.item.provenance||[])]};}
      else fact=newField(key);
      if(key==='pending')fact.description=text;else fact.text=text;markClinicianFact(fact);return fact;
    });
    value.splice(0,value.length,...next);edited();
  }
  function compactAudioFields(parent,obj){
    const expected=new Set([...audioCompactKeys,...audioMetadataKeys,'coding','medications','next_contact']);
    const unsupported=Object.entries(obj).some(([key,value])=>!expected.has(key)&&hasReviewContent(value))||
      ['coding','medications','next_contact'].some(key=>hasReviewContent(obj[key]));
    if(unsupported){fields(parent,obj);return;}
    const intro=el(parent,'p','Μόνο οι ενότητες που περιέχουν κείμενο εμφανίζονται. Κάθε γραμμή είναι ένα στοιχείο· διόρθωσε ή αφαίρεσε μόνο ό,τι χρειάζεται.','clinical-muted');
    intro.id='captureAudioCompactHelp';
    for(const key of audioCompactKeys){
      const value=obj[key];
      if(key==='reason_for_visit'){
        if(!value||!String(value.text||'').trim())continue;
        const section=el(parent,'section','','capture-audio-compact-section');section.dataset.field=key;
        el(section,'h4',audioLabels[key]||labels[key]||key);const textarea=el(section,'textarea');textarea.value=value.text;textarea.rows=2;
        textarea.setAttribute('aria-describedby',intro.id);textarea.oninput=()=>{value.text=textarea.value;markClinicianFact(value);edited();};
        continue;
      }
      if(!Array.isArray(value)||!value.length)continue;
      const section=el(parent,'section','','capture-audio-compact-section');section.dataset.field=key;
      el(section,'h4',audioLabels[key]||labels[key]||key);const textarea=el(section,'textarea');textarea.value=value.map(item=>audioItemText(key,item)).join('\n');textarea.rows=Math.min(6,Math.max(2,value.length));
      textarea.setAttribute('aria-describedby',intro.id);textarea.oninput=()=>syncAudioLines(key,value,textarea.value);
    }
  }
  function renderAudioSuggestions(parent){
    if(!audioSuggestions.length)return;
    const section=el(parent,'section','','capture-audio-suggestions');
    el(section,'h4','Προτάσεις AI για την επόμενη επίσκεψη');
    el(section,'p','Δεν είναι καταγραφή όσων ειπώθηκαν και δεν αποθηκεύονται αυτόματα. Διέγραψε όποια γραμμή δεν θέλεις.','clinical-muted');
    const textarea=el(section,'textarea');textarea.value=audioSuggestionDraft;textarea.rows=Math.min(6,Math.max(2,audioSuggestions.length));
    textarea.oninput=()=>{audioSuggestionDraft=textarea.value;};
    const details=el(section,'details');el(details,'summary','Γιατί προτείνονται');
    const list=el(details,'ul');for(const item of audioSuggestions){const row=el(list,'li');el(row,'strong',item.text);row.append(document.createTextNode(' — '+item.rationale));}
    button(section,'Προσθήκη στο πλάνο',()=>{
      const lines=audioSuggestionDraft.split('\n').map(line=>line.trim()).filter(Boolean);
      const known=new Set(candidate.follow_up.map(item=>item.text));
      for(const text of lines)if(!known.has(text)){const item=newField('follow_up');item.text=text;candidate.follow_up.push(item);known.add(text);}
      audioSuggestions=[];audioSuggestionDraft='';edited();drawReview();
    });
  }
  function drawReview(){
    review.replaceChildren();
    if(audioDraft){compactAudioFields(review,candidate);renderAudioSuggestions(review);review.append(audioConfirmLabel);}else fields(review,candidate);
    button(review,'Προεπισκόπηση καταγραφής',preview);
  }
  function renderAudioFocused(){
    audioFocused.replaceChildren();
    if(!audioDraft)return;
    const unique=[...new Map(audioReviewUnits.map(item=>[item.text,item])).values()];
    if(!unique.length)return;
    el(audioFocused,'p','Δεν χρειάζεται ξεχωριστή επιβεβαίωση για κάθε σημείο. Οι αμφίβολες φράσεις παραμένουν στις Αβεβαιότητες.','clinical-muted');
    const details=el(audioFocused,'details');el(details,'summary','Προαιρετικός επανέλεγχος '+unique.length+' σημείων');
    const list=el(details,'ul');for(const item of unique)el(list,'li',item.text);
  }
  function taskReview(){
    if(!amendment)return;
    const section=el(previews,'section','','amendment-task-review');el(section,'h4','Έλεγχος υποχρεώσεων');
    const kept=new Set();
    for(const item of candidate.pending){const id=pendingBindings.get(item),original=amendment.originals.get(id);const same=original&&JSON.stringify(item)===original;
      if(same)kept.add(id);el(section,'p',(same?'Διατηρείται με την κατάστασή της: ':id?'Αντικατάσταση με νέα εκκρεμότητα: ':'Νέα εκκρεμότητα: ')+item.description);}
    for(const [id,description] of amendment.descriptions)if(!kept.has(id))el(section,'p','Στις διαγραμμένες: '+description);
    if(!candidate.pending.length&&!amendment.descriptions.size)el(section,'p','Δεν υπάρχουν υποχρεώσεις.');
  }
  async function beginAmend(encounter){
    if(!patientId||pendingBusy||!canLeave())return;
    const id=patientId,ticket=generation,draftTicket=++draftGeneration;
    try{const result=await request('/patient/'+encodeURIComponent(id)+'/visit/'+encodeURIComponent(encounter)+'/edit');
      if(ticket!==generation||draftTicket!==draftGeneration||!composer.hidden||pendingBusy)return;
      resetDraft();amendment={...result,request_id:crypto.randomUUID(),originals:new Map(),descriptions:new Map(),submitted:null};
      candidate=result.candidate;
      candidate.pending.forEach((item,index)=>{const pid=result.pending_bindings[index];pendingBindings.set(item,pid);amendment.originals.set(pid,JSON.stringify(item));amendment.descriptions.set(pid,item.description);});
      bindClinician();composerTitle.textContent='Επεξεργασία επίσκεψης · '+date(candidate.encounter_date);
      audioPanel.hidden=true;
      amendmentNote.hidden=false;amendmentNote.textContent='Διόρθωσε την υπάρχουσα καταγραφή ή επικόλλησε ξανά τη σύνοψη και πάτησε «Προετοιμασία καταγραφής». Αμετάβλητες υποχρεώσεις διατηρούν την κατάστασή τους. Αφαιρούμενες πηγαίνουν στις διαγραμμένες· τροποποιημένες αντικαθίστανται ως νέες εκκρεμότητες.';
      visitDate.value=candidate.encounter_date;saveButton.textContent='Αποθήκευση διορθώσεων';saveButton.hidden=false;saveButton.disabled=false;
      composer.hidden=false;chart.hidden=true;pendingControls(true);drawReview();feedback.textContent='Η αποθηκευμένη επίσκεψη είναι έτοιμη για επεξεργασία. Δεν έχει αλλάξει τίποτα ακόμη.';
      composer.scrollIntoView({block:'start'});
    }catch(e){if(ticket===generation&&draftTicket===draftGeneration){const status=el(chart,'p',e.message);status.setAttribute('role','status');}}
  }
  async function extract(){
    if(busy||!patientId||input.value.trim().length<10||!visitDate.value)return;
    const ticket=generation,draftTicket=++draftGeneration;busy=true;extractButton.disabled=true;saveButton.hidden=true;review.replaceChildren();previews.replaceChildren();feedback.textContent='Προετοιμασία καταγραφής…';
    source=input.value;input.disabled=true;visitDate.disabled=true;
    try{const context=await request('/visit-capture/context',{patient_id:patientId,client_session_id:session});if(ticket!==generation||draftTicket!==draftGeneration)return;contextId=context.context_id;
      const result=await request('/visit-capture/extract',{context_id:contextId,text:source,encounter_date:visitDate.value});if(ticket!==generation||draftTicket!==draftGeneration)return;
      candidate=result.candidate;pendingBindings=new Map();bindClinician();
      drawReview();feedback.textContent=candidate.source_identity_state==='consistent'?'Διόρθωσε ό,τι χρειάζεται και αφαίρεσε όποια υποχρέωση δεν αποδέχεσαι.':'Η σύνοψη περιέχει ασυμφωνία ή αβεβαιότητα ταυτότητας. Διόρθωσε το αρχικό κείμενο και προετοίμασε ξανά.';
      saveButton.hidden=candidate.source_identity_state!=='consistent';saveButton.disabled=false;await preview();
    }catch(e){if(ticket===generation&&draftTicket===draftGeneration)feedback.textContent=e.message;}
    finally{if(ticket===generation&&draftTicket===draftGeneration){busy=false;extractButton.disabled=false;input.disabled=false;visitDate.disabled=false;}}
  }
  async function refreshAudioContextIfNeeded(){
    if(!audioDraft)return true;
    // Clinical context timestamps are naive UTC from the existing API.
    const expires=Date.parse(/[zZ]|[+-]\d\d:\d\d$/.test(audioContextExpiresAt)?audioContextExpiresAt:audioContextExpiresAt+'Z');
    if(!Number.isFinite(expires)){feedback.textContent='Η ισχύς της καρτέλας δεν επαληθεύτηκε. Άνοιξε ξανά την επίσκεψη.';return false;}
    if(expires-Date.now()>120000)return true;
    const old=contextId,id=patientId,ticket=generation,draftTicket=draftGeneration;
    try{const renewed=await request('/visit-capture/context/renew',{prior_context_id:old,patient_id:id,client_session_id:session});
      if(ticket!==generation||draftTicket!==draftGeneration||id!==patientId||old!==contextId||renewed.patient_id!==id)return false;
      contextId=renewed.context_id;audioContextExpiresAt=renewed.expires_at||'';
      audioConfirm.checked=false;audioPreviewed=null;saveButton.disabled=true;previews.replaceChildren();
      feedback.textContent='Η ασφαλής σύνδεση του φακέλου ανανεώθηκε. Επιβεβαίωσε ξανά τον ασθενή και δες την προεπισκόπηση.';
      return false;
    }catch(e){feedback.textContent=e.message;return false;}
  }
  async function preview(){
    if(audioDraft&&!audioConfirm.checked){feedback.textContent='Επιβεβαίωσε πρώτα ότι η ηχογράφηση ανήκει στον εμφανιζόμενο ασθενή.';return;}
    if(audioDraft&&!await refreshAudioContextIfNeeded())return;
    if(amendment&&!contextId){const ticket=generation,draftTicket=draftGeneration;try{const c=await request('/visit-capture/context',{patient_id:patientId,client_session_id:session});if(ticket!==generation||draftTicket!==draftGeneration)return;contextId=c.context_id;}catch(e){if(ticket===generation&&draftTicket===draftGeneration)feedback.textContent=e.message;return;}}
    const ticket=generation,draftTicket=draftGeneration;if(!candidate)return;const submitted=JSON.stringify(candidate);try{
    const p=await request('/visit-capture/preview',{context_id:contextId,candidate,recording_patient_confirmed:audioDraft&&audioConfirm.checked});if(ticket!==generation||draftTicket!==draftGeneration||JSON.stringify(candidate)!==submitted||(audioDraft&&!audioConfirm.checked))return;
    previews.replaceChildren();taskReview();for(const [key,label]of [['snapshot','Συνοπτική εικόνα'],['brief','Προετοιμασία επίσκεψης'],['detail','Πλήρης καταγραφή']]){const d=el(previews,'details');el(d,'summary',label);el(d,'p',p[key],'clinical-preview-text');}
    if(audioDraft){audioPreviewed=p.can_save?submitted:null;saveButton.disabled=!p.can_save;}
  }catch(e){if(ticket===generation&&draftTicket===draftGeneration)feedback.textContent=e.message;}}
  input.oninput=()=>{draftGeneration++;if(amendment){saveButton.hidden=true;previews.replaceChildren();feedback.textContent='Πάτησε «Προετοιμασία καταγραφής» για να χρησιμοποιήσεις το νέο κείμενο.';return;}if(audioDraft){audioDraft=false;audioReviewUnits=[];audioSuggestions=[];audioSuggestionDraft='';audioReview.hidden=true;audioConfirm.checked=false;audioPreviewed=null;}candidate=null;saveButton.hidden=true;review.replaceChildren();previews.replaceChildren();};visitDate.oninput=()=>{if(amendment){candidate.encounter_date=visitDate.value;previews.replaceChildren();}else input.oninput();};
  async function save(){if(amendment)return saveAmendment();if(busy||!candidate||!contextId)return;if(audioDraft&&(!audioConfirm.checked||audioPreviewed!==JSON.stringify(candidate))){feedback.textContent='Επιβεβαίωσε τον ασθενή και ανανέωσε την προεπισκόπηση πριν αποθηκεύσεις.';return;}if(audioDraft&&!await refreshAudioContextIfNeeded())return;const ticket=generation,draftTicket=draftGeneration,id=patientId;busy=true;saveButton.disabled=true;extractButton.disabled=true;
    feedback.textContent='Αποθήκευση…';
    try{await request('/visit-capture/save',{context_id:contextId,candidate,recording_patient_confirmed:audioDraft&&audioConfirm.checked});if(ticket!==generation||draftTicket!==draftGeneration)return;await open(id);window.dispatchEvent(new Event('cockpit:clinical-updates-changed'));
    }catch(e){if(ticket!==generation||draftTicket!==draftGeneration)return;
      // A lost response may follow a committed save. Never offer a fresh write
      // blindly; the user can inspect the authoritative chart first.
      feedback.textContent=e.message+' Έλεγξε την καρτέλα πριν επαναλάβεις την καταγραφή.';
      button(review,'Έλεγχος αποθηκευμένης καρτέλας',()=>open(id));
    }
  }
  async function saveAmendment(){
    if(busy||!candidate)return;
    const ticket=generation,draftTicket=draftGeneration,id=patientId,editing=amendment;
    const active=()=>ticket===generation&&draftTicket===draftGeneration&&id===patientId&&editing===amendment&&!composer.hidden;
    if(!contextId){await preview();if(!active()||!contextId)return;}
    // First click reviews the task effects; the same button then explicitly saves.
    if(!previews.querySelector('.amendment-task-review')){await preview();if(active())feedback.textContent='Έλεγξε τις υποχρεώσεις και πάτησε ξανά «Αποθήκευση διορθώσεων».';return;}
    if(!active())return;
    const body={context_id:contextId,expected_token:editing.expected_token,request_id:editing.request_id,candidate,pending_bindings:candidate.pending.map(p=>pendingBindings.get(p)||null)};
    const serialized=JSON.stringify(body);
    if(editing.submitted&&editing.submitted!==serialized){feedback.textContent='Η προηγούμενη αποθήκευση έχει αβέβαιο αποτέλεσμα. Έλεγξε την καρτέλα πριν κάνεις νέα διόρθωση.';return;}
    editing.submitted=serialized;busy=true;
    const controls=[...composer.querySelectorAll('button,input,textarea,select')];const disabled=controls.map(c=>c.disabled);controls.forEach(c=>c.disabled=true);
    feedback.textContent='Αποθήκευση διορθώσεων…';
    try{await request('/patient/'+encodeURIComponent(id)+'/visit/'+encodeURIComponent(editing.encounter_id)+'/amend',body);
      if(ticket!==generation||draftTicket!==draftGeneration)return;
      await open(id);showVisit(id,editing.encounter_id);window.dispatchEvent(new Event('cockpit:clinical-updates-changed'));
    }catch(e){if(active()){if(e.status===422)editing.submitted=null;feedback.textContent=(e.status===422?'Έλεγξε τα πεδία και κράτησε τις νέες υποχρεώσεις ανοικτές. ':e.message+' ')+'Το πρόχειρο διατηρήθηκε. Μπορείς να επαναλάβεις την ίδια αποθήκευση ή να κλείσεις και να ελέγξεις την καρτέλα.';busy=false;controls.forEach((c,i)=>c.disabled=disabled[i]);}}
  }
  async function renderUpdates(target,active,close,scope=patientId==='cockpit-cp1-test'?'test':'real'){const ticket=generation;target.replaceChildren();el(target,'p','Φόρτωση ενημερώσεων…');
    try{const data=await request('/clinical-updates?scope='+scope+'&active_only=true');if(ticket!==generation||!active())return;target.replaceChildren();
      button(target,scope==='test'?'Πραγματικοί φάκελοι':'Δοκιμαστικός ασθενής',()=>renderUpdates(target,active,close,scope==='test'?'real':'test'));
      const items=data.items.filter(p=>['open','awaiting_patient','awaiting_result'].includes(p.status));
      el(target,'p',items.length?'Ανοικτές υποχρεώσεις ανά ασθενή · έως 100':'Δεν υπάρχουν ανοικτές υποχρεώσεις.');
      const groups=new Map();
      for(const item of items){
        let group=groups.get(item.patient_id);
        if(!group){group=el(target,'section','','patient-updates-group');el(group,'h3',item.patient_name);groups.set(item.patient_id,group);}
        const b=button(group,'',async()=>{close();if(!canLeave())return;await open(item.patient_id);const current=chartPending.find(p=>p.pending_id===item.pending_id);if(current)showPending(item.patient_id,current);});
        b.className='doctor-action-row';b.dataset.pendingId=item.pending_id;
        const copy=el(b,'span');el(copy,'strong',item.description);el(copy,'small',(item.checkpoints?.next_checkpoint?.question||statuses[item.status]||item.status)+' · '+date(item.source_date));el(b,'span','›');
      }
    }catch(e){if(ticket===generation&&active()){target.replaceChildren();el(target,'p',e.message);}}
  }
  $('visitBackHome').addEventListener('click',()=>{if(!canLeave())return;reset();const u=new URL(location.href);u.searchParams.delete('chart');history.replaceState(null,'',u);});
  window.addEventListener('clinical:unauthenticated',reset);
  window.addEventListener('clinical:authenticated',()=>{reset();const id=new URL(location.href).searchParams.get('chart');if(id)open(id);});
  window.CockpitStoredChart={async openVisit(id,encounter){if(!canLeave()||pendingBusy)return false;if(await open(id)!==id)return false;return showVisit(id,encounter);},async openPending(id,pending,options={}){if(!canLeave())return;if(patientId!==id&&await open(id)!==id)return;const item=chartPending.find(p=>p.pending_id===pending);if(item)showPending(id,item,Boolean(options?.editMetadata));},open(id){if(!canLeave())return;return open(id);},reset,renderPending,renderUpdates,beginVisit,canLeave,cancelPendingRead(){generation++;}};
  window.addEventListener('pagehide',discardAudio);
  window.addEventListener('beforeunload',e=>{if(!composer.hidden&&(input.value.trim()||candidate||busy||audioRecorder)){e.preventDefault();e.returnValue='';}});
  const id=new URL(location.href).searchParams.get('chart');if(id)open(id);
})();
