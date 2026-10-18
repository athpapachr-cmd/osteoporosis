/* Normal Cockpit: existing chart/atomic VisitCapture owners, transient drafts. */
(() => {
  'use strict';
  if(window.CockpitSynthetic)return;
  const $=id=>document.getElementById(id);
  let draftGeneration=0,journeyObserver=null,pendingBusy=false,chartPending=[];
  let generation=0,patientId=null,contextId=null,candidate=null,source='',busy=false;
  const session=crypto.randomUUID();
  const el=(parent,tag,value,cls)=>{const n=document.createElement(tag);if(value)n.textContent=value;if(cls)n.className=cls;parent.append(n);return n;};
  const button=(parent,label,fn)=>{const b=el(parent,'button',label,'clinical-secondary');b.type='button';b.onclick=fn;return b;};
  const date=v=>v?new Intl.DateTimeFormat('el-CY',{dateStyle:'medium',timeZone:'Asia/Nicosia'}).format(new Date(v.includes('T')?v:v.slice(0,10)+'T12:00:00Z')):'';
  const statuses={open:'Ανοικτή',awaiting_patient:'Αναμένεται από τον ασθενή',awaiting_result:'Αναμένεται αποτέλεσμα',resolved:'Ολοκληρώθηκε',cancelled:'Ακυρώθηκε',not_yet_indicated:'Δεν έχει ακόμη ένδειξη'};
  const roles={patient:'Ασθενής',clinician:'Γιατρός',secretary:'Γραμματεία',unspecified:'Δεν προσδιορίστηκε'};
  const chart=el($('visitWorkspacePatient'),'section','','patient-chart');chart.id='patientChart';
  const composer=el($('visitWorkspacePatient'),'section','','clinical-dia-composer');composer.id='captureComposer';composer.hidden=true;
  el(composer,'h3','Νέα επίσκεψη από τη σύνοψη Dia');
  const promptText=$('visitDiaNaturalPrompt').content.textContent.trim();
  const promptTools=el(composer,'section','','capture-dia-prompt');
  const promptCopy=button(promptTools,'Αντιγραφή οδηγίας για το Dia',async()=>{if(!await window.CockpitHome.copyDiaPrompt(promptText,promptStatus))promptDetails.open=true;});promptCopy.id='captureCopyDiaPrompt';
  const promptDetails=el(promptTools,'details');el(promptDetails,'summary','Δες την οδηγία για το Dia');
  const promptDisplay=el(promptDetails,'pre',promptText);promptDisplay.id='captureDiaPromptText';promptDisplay.tabIndex=0;
  const promptStatus=el(promptTools,'p','','clinical-muted');promptStatus.id='captureDiaPromptStatus';promptStatus.setAttribute('role','status');

  const dateLabel=el(composer,'label','Ημερομηνία επίσκεψης');
  const visitDate=el(dateLabel,'input');visitDate.type='date';visitDate.id='captureDate';
  const inputLabel=el(composer,'label','Σύνοψη επίσκεψης');
  const input=el(inputLabel,'textarea');input.id='captureText';input.rows=7;input.maxLength=30000;
  input.placeholder='Επικόλλησε το φυσικό κείμενο της σύνοψης. Δεν χρειάζονται ειδικές επικεφαλίδες.';
  const extractButton=button(composer,'Προετοιμασία καταγραφής',extract);
  const feedback=el(composer,'p');feedback.id='captureStatus';feedback.setAttribute('role','status');
  const review=el(composer,'div');review.id='captureReview';
  const previews=el(composer,'div');
  const saveButton=button(composer,'Αποθήκευση επίσκεψης και υποχρεώσεων',save);saveButton.id='captureSave';saveButton.hidden=true;
  button(composer,'Κλείσιμο',()=>{resetDraft();composer.hidden=true;});
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
      throw Error(r.status===401?'Συνδέσου από την Πρόσβαση.':r.status===409?'Ο φάκελος άλλαξε. Άνοιξέ τον ξανά πριν συνεχίσεις.':
        r.status===502||r.status===503?'Η μετατροπή δεν είναι διαθέσιμη αυτή τη στιγμή. Το κείμενο διατηρήθηκε.':'Η ενέργεια δεν ολοκληρώθηκε. Δοκίμασε ξανά.');
    }return r.json();
  }
  function pendingControls(disabled){for(const b of chart.querySelectorAll('.pending-checkpoint-actions button,.pending-checkpoint-actions select,.pending-stage button,.pending-open'))b.disabled=disabled;}
  function resetDraft(){promptStatus.textContent='';pendingControls(false);draftGeneration++;contextId=null;candidate=null;source='';busy=false;input.value='';review.replaceChildren();previews.replaceChildren();feedback.textContent='';saveButton.hidden=true;extractButton.disabled=false;input.disabled=false;visitDate.disabled=false;}
  function reset(){for(const dialog of chart.querySelectorAll('dialog[open]'))dialog.close();pendingBusy=false;chartPending=[];journeyObserver?.disconnect();journeyObserver=null;generation++;patientId=null;chart.replaceChildren();resetDraft();composer.hidden=true;}
  function facts(parent,title,items){if(!items?.length)return;el(parent,'h4',title);const list=el(parent,'ul');for(const i of items)el(list,'li',typeof i==='string'?i:i.text);}
  function showPending(id,item){
    if(!composer.hidden||pendingBusy||patientId!==id)return;
    chart.querySelector('.patient-pending-dialog')?.remove();
    const dialog=el(chart,'dialog','','patient-journey-preview patient-pending-dialog');dialog.setAttribute('aria-label','Ενημέρωση υποχρέωσης');
    button(dialog,'Κλείσιμο',()=>dialog.close()).classList.add('patient-journey-close');
    el(dialog,'h3','Ενημέρωση υποχρέωσης');renderPending(el(dialog,'div'),id,[item]);
    dialog.onclick=e=>{if(e.target===dialog)dialog.close();};dialog.onclose=()=>dialog.remove();dialog.showModal();
  }
  function renderPending(target,id,items,branch=false){
    target.replaceChildren();if(!items.length)el(target,'p','Δεν έχουν καταγραφεί υποχρεώσεις.');
    for(const item of items){const row=el(target,'article','','synthetic-pending-row'),copy=el(row,'div');
      row.dataset.pendingId=item.pending_id;
      if(branch){const title=button(copy,item.description,()=>showPending(id,item));title.classList.add('pending-open');title.setAttribute('aria-haspopup','dialog');}else el(copy,'strong',item.description);el(copy,'p',(statuses[item.status]||item.status)+' · '+(roles[item.responsible_role]||item.responsible_role));
      if(item.review_date)el(copy,'p','Επανέλεγχος: '+date(item.review_date));
      const progress=item.checkpoints;
      const actions=el(copy,'div','','pending-checkpoint-actions');
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
        const controls=el(parent,'div','','pending-checkpoint-actions');
        const act=(observation)=>mutate('/checkpoint',{checkpoint_key:step.key,observation,expected_revision:progress.revision,retracts_event_id:observation==='retracted'?step.event_id:null});
        if(step.state==='confirmed')button(controls,'Αναίρεση',()=>act('retracted'));
        else{button(controls,'Ναι',()=>act('yes_confirmed'));button(controls,'Όχι ακόμη',()=>act('not_yet_observed'));}
      }
      if(progress&&branch){
        actions.classList.add('patient-followthrough-rail');actions.setAttribute('aria-label','Βήματα υποχρέωσης');
        for(const step of progress.steps)stage(actions,step);
        const end=el(actions,'div','','patient-followthrough-end');el(end,'span','','patient-followthrough-stop');el(end,'small',progress.completed?'Ολοκληρώθηκε':'Τέλος');
        end.dataset.complete=String(progress.completed);
      }else if(progress){
        for(const step of progress.steps.filter(s=>s.state==='confirmed'))stage(actions,step);
        if(progress.next_checkpoint)stage(actions,progress.next_checkpoint);
        const later=progress.steps.filter(s=>s.state!=='confirmed'&&s.key!==progress.next_checkpoint?.key);
        if(later.length){const other=el(copy,'details');el(other,'summary','Έγινε ήδη επόμενο βήμα;');for(const step of later)stage(other,step);}
      }else if(item.checkpoint_write_allowed&&['open','awaiting_patient','awaiting_result'].includes(item.status)){
        const choices={material_images:['CD: παραλαβή → εξέταση εικόνων','Παραλήφθηκε το CD;'],document:['Πόρισμα: παραλαβή → ανάγνωση','Παραλήφθηκε το πόρισμα;'],completion:['Μία ενέργεια','Ολοκληρώθηκε η ενέργεια;']};
        const label=el(actions,'label','Παρακολούθηση '),select=el(label,'select');
        for(const [key,[title]]of Object.entries(choices)){const option=el(select,'option',title);option.value=key;}
        select.value=/\bcd\b/i.test(item.description)?'material_images':/πόρισμα/i.test(item.description)?'document':'completion';
        const question=el(actions,'p',choices[select.value][1]);select.onchange=()=>question.textContent=choices[select.value][1];
        button(actions,'Ναι',()=>mutate('/checkpoint-plan',{expected_revision:item.lifecycle?.revision||0,template:select.value,first_observation:'yes_confirmed'}));
        button(actions,'Όχι ακόμη',()=>mutate('/checkpoint-plan',{expected_revision:item.lifecycle?.revision||0,template:select.value,first_observation:'not_yet_observed'}));
        const prior=el(actions,'details');el(prior,'summary','Έγινε ήδη επόμενο βήμα;');
        const priorChoices=el(prior,'div');
        function showPrior(){priorChoices.replaceChildren();prior.hidden=select.value==='completion';
          const key=select.value==='material_images'?'images_reviewed':'report_read';
          el(priorChoices,'p',key==='images_reviewed'?'Εξετάστηκαν οι εικόνες;':'Διαβάστηκε το πόρισμα;');
          for(const [label,observation]of [['Ναι','yes_confirmed'],['Όχι ακόμη','not_yet_observed']])button(priorChoices,label,()=>mutate('/checkpoint-plan',{expected_revision:item.lifecycle?.revision||0,template:select.value,first_checkpoint_key:key,first_observation:observation}));
        }
        select.onchange=()=>{question.textContent=choices[select.value][1];showPrior();};showPrior();

      }else if(!item.checkpoint_storage_ready)el(copy,'small','Η ενημέρωση των βημάτων δεν έχει ενεργοποιηθεί ακόμη.');
      else if(!item.checkpoint_write_allowed)el(copy,'small','Για ενημέρωση χρειάζεται η πρόσβαση γιατρού.');

      if(branch&&!progress){
        const existing=[...actions.childNodes];actions.classList.add('patient-followthrough-rail');
        const step=el(actions,'div','','pending-stage');step.dataset.state='unknown';const dot=button(step,'',()=>showPending(id,item));dot.className='patient-followthrough-dot pending-open';dot.setAttribute('aria-label','Ενημέρωση: '+item.description);dot.setAttribute('aria-haspopup','dialog');
        for(const child of existing)step.append(child);
        if(!existing.length)el(step,'p',statuses[item.status]||item.status);
        const end=el(actions,'div','','patient-followthrough-end');end.dataset.complete=String(item.status==='resolved');el(end,'span','','patient-followthrough-stop');el(end,'small',item.status==='resolved'?'Ολοκληρώθηκε':item.status==='cancelled'?'Ακυρώθηκε':'Τέλος');
      }
      if(target.closest('.patient-pending-dialog')&&item.checkpoint_write_allowed){
        const tools=el(copy,'div','','pending-checkpoint-actions');
        if(item.status==='cancelled'&&item.lifecycle?.cancellation_event_id){
          button(tools,'Επαναφορά υποχρέωσης',()=>mutate('/lifecycle',{action:'restore',expected_revision:item.lifecycle.revision,cancellation_event_id:item.lifecycle.cancellation_event_id}));
        }else if(['open','awaiting_patient','awaiting_result'].includes(item.status)){
          button(tools,'Διαγραφή υποχρέωσης',()=>{
            tools.replaceChildren();el(tools,'p','Να αφαιρεθεί από τις ενεργές υποχρεώσεις; Η καταγραφή της επίσκεψης διατηρείται και μπορείς να την επαναφέρεις.');
            button(tools,'Διαγραφή',()=>mutate('/lifecycle',{action:'cancel',expected_revision:item.lifecycle?.revision||0}));
            button(tools,'Διατήρηση',()=>{const dialog=target.closest('dialog');dialog.close();showPending(id,item);});
          });
        }
      }
      if(item.source_excerpt)el(copy,'blockquote',item.source_excerpt);
      button(copy,'Δημιουργήθηκε στην επίσκεψη · '+date(item.source_date),async()=>{
        if(patientId!==id){if(!canLeave())return;await open(id);}const target=$('visit-'+item.source_encounter_id);if(target){target.hidden=false;target.open=true;target.scrollIntoView({behavior:'smooth'});}
      });
    }
  }
  function canLeave(){if(!composer.hidden){feedback.textContent='Κλείσε ή αποθήκευσε την τρέχουσα καταγραφή πριν αλλάξεις ασθενή.';composer.scrollIntoView({behavior:'smooth'});return false;}return true;}
  function beginVisit(){if(!patientId||pendingBusy)return;resetDraft();composer.hidden=false;pendingControls(true);visitDate.value=new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Nicosia',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());input.focus();composer.scrollIntoView({behavior:'smooth'});}
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
      const heading=el(chart,'div','','patient-chart-heading');el(heading,'h2','Κλινική συνέχεια');
      button(heading,'Νέα επίσκεψη · Σύνοψη Dia',beginVisit).id='captureOpen';
      const grid=el(chart,'div','','patient-chart-grid'),overview=el(grid,'section','','patient-chart-card');el(overview,'h3','Κλινική εικόνα');
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
      const timeline=el(chart,'section','','patient-timeline');el(timeline,'h3','Πορεία ασθενούς');
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
        button(selected,'Άνοιγμα',()=>{selected.close();const visit=$('visit-'+point.source);if(visit){visit.hidden=false;visit.open=true;visit.scrollIntoView({behavior:'smooth'});visit.querySelector('summary').focus();}});
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

      for(const visit of data.visits){const block=el(timeline,'details','','patient-visit');block.id='visit-'+visit.encounter_id;block.hidden=true;
        const summary=el(block,'summary');el(summary,'span',date(visit.date));el(summary,'strong',visit.clinical?.reason_for_visit?.text||'Καταγεγραμμένη επίσκεψη');
        const origin=data.pending.filter(p=>p.source_encounter_id===visit.encounter_id);for(const p of origin)el(summary,'small','Δημιουργήθηκε υποχρέωση: '+p.description);
        const body=el(block,'div','','patient-visit-body');const v=visit.clinical;
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
  const labels={reason_for_visit:'Λόγος επίσκεψης',visit_type:'Τύπος επίσκεψης',specialty:'Ειδικότητα',what_changed:'Μεταβολές',findings:'Ιστορικό και ευρήματα',clinical_impression:'Κλινική εκτίμηση',coding:'Κωδικοποίηση',gesy_coded:'Κωδικοί',coding_complete:'Πλήρης κωδικοποίηση',decisions:'Αποφάσεις',medications:'Φάρμακα',pending:'Προτεινόμενες υποχρεώσεις',next_contact:'Επόμενη επαφή',safety_net:'Οδηγίες επανεπικοινωνίας',uncertainties:'Αβεβαιότητες',text:'Περιγραφή',type:'Είδος',responsible_role:'Υπεύθυνος',certainty:'Βεβαιότητα',region:'Περιοχή',name:'Φάρμακο',action:'Ενέργεια',dose:'Δόση',frequency:'Συχνότητα',route:'Οδός',duration:'Διάρκεια',value:'Τιμή',unit:'Μονάδα',approximate:'Κατά προσέγγιση',item_type:'Είδος υποχρέωσης',description:'Περιγραφή',trigger:'Προϋπόθεση',status:'Κατάσταση',external_dependency:'Εξωτερική εξάρτηση',actor:'Από ποιον',condition:'Προϋπόθεση',review_date:'Ημερομηνία επανελέγχου',when:'Ημερομηνία',time:'Ώρα',purpose:'Σκοπός',assess:'Προς αξιολόγηση',code:'Κωδικός',system:'Σύστημα',title:'Τίτλος',direction:'Μεταβολή'};
  const enums={responsible_role:roles,item_type:{investigation:'Εξέταση',test:'Εξέταση',result:'Αποτέλεσμα',document:'Έγγραφο',follow_up:'Επανέλεγχος',referral:'Παραπομπή',other:'Άλλο'},certainty:{certain:'Βέβαιο',uncertain:'Αβέβαιο',conflicting:'Αντικρουόμενο'},action:{prescribed:'Έναρξη',continued:'Συνέχιση',stopped:'Διακοπή'},status:{open:statuses.open,awaiting_patient:statuses.awaiting_patient,awaiting_result:statuses.awaiting_result},type:{therapy:'Θεραπεία',referral:'Παραπομπή',medication:'Αγωγή',advice:'Οδηγία',sick_leave:'Άδεια ασθενείας',admin:'Διοικητικό'}};
  function edited(obj){if(obj&&typeof obj==='object'&&!Array.isArray(obj)&&obj.provenance){if(!obj.provenance.includes('clinician_edit'))obj.provenance.push('clinician_edit');if('verification'in obj)obj.verification='clinician_confirmed';}previews.replaceChildren();}
  function fields(parent,obj){
    let extra=null;
    for(const [key,value] of Object.entries(obj)){
      if(['schema_version','source_identity_state','source_bindings','comparison_basis','encounter_date','provenance','verification'].includes(key))continue;
      if(key==='source_excerpt'){if(value)el(parent,'blockquote',value);continue;}
      if(Array.isArray(value)){if(!value.length)continue;const group=el(parent,'section');el(group,'h4',labels[key]||key);
        for(const item of [...value]){const row=el(group,'div','','capture-field-group');
          if(typeof item==='object')fields(row,item);else{const l=el(row,'label',labels[key]||key);const t=el(l,'input');t.value=item;t.oninput=()=>{const index=Array.from(group.querySelectorAll(':scope > .capture-field-group')).indexOf(row);value[index]=t.value;edited(obj);};}
          button(row,'Αφαίρεση',()=>{const index=Array.from(group.querySelectorAll(':scope > .capture-field-group')).indexOf(row);if(index>=0)value.splice(index,1);row.remove();edited(obj);});
        }continue;}
      if(value&&typeof value==='object'){const group=el(parent,'fieldset');el(group,'legend',labels[key]||key);fields(group,value);continue;}
      // Empty optional objects cannot be edited as strings.
      if(value===null&&['duration','external_dependency','next_contact'].includes(key))continue;
      let destination=parent;
      if(['type','certainty','region','item_type','specialty','visit_type','system','coding_complete'].includes(key)){
        if(!extra){extra=el(parent,'details');el(extra,'summary','Λεπτομέρειες');}destination=extra;
      }
      const label=el(destination,'label',labels[key]||key);let control;
      if(enums[key]){control=el(label,'select');for(const [v,title]of Object.entries({...enums[key],...(value&&!enums[key][value]?{[value]:value}:{})})){const option=el(control,'option',title);option.value=v;}}
      else{control=el(label,['text','description','purpose'].includes(key)?'textarea':'input');if(typeof value==='boolean')control.type='checkbox';else if(typeof value==='number')control.type='number';else if(['review_date','when'].includes(key))control.type='date';}
      if(typeof value==='boolean')control.checked=value;else control.value=value??'';
      if(control.tagName==='TEXTAREA')control.rows=2;
      control.oninput=()=>{obj[key]=typeof value==='boolean'?control.checked:typeof value==='number'?Number(control.value):control.value||null;edited(obj);};
    }
  }
  async function extract(){
    if(busy||!patientId||input.value.trim().length<10||!visitDate.value)return;
    const ticket=generation,draftTicket=++draftGeneration;busy=true;extractButton.disabled=true;saveButton.hidden=true;review.replaceChildren();previews.replaceChildren();feedback.textContent='Προετοιμασία καταγραφής…';
    source=input.value;input.disabled=true;visitDate.disabled=true;
    try{const context=await request('/visit-capture/context',{patient_id:patientId,client_session_id:session});if(ticket!==generation||draftTicket!==draftGeneration)return;contextId=context.context_id;
      const result=await request('/visit-capture/extract',{context_id:contextId,text:source,encounter_date:visitDate.value});if(ticket!==generation||draftTicket!==draftGeneration)return;
      candidate=result.candidate;candidate.source_bindings.push({source_ref:'clinician_edit',role:'clinician_edit'});
      fields(review,candidate);feedback.textContent=candidate.source_identity_state==='consistent'?'Διόρθωσε ό,τι χρειάζεται και αφαίρεσε όποια υποχρέωση δεν αποδέχεσαι.':'Η σύνοψη περιέχει ασυμφωνία ή αβεβαιότητα ταυτότητας. Διόρθωσε το αρχικό κείμενο και προετοίμασε ξανά.';
      button(review,'Προεπισκόπηση καταγραφής',preview);saveButton.hidden=candidate.source_identity_state!=='consistent';saveButton.disabled=false;await preview();
    }catch(e){if(ticket===generation&&draftTicket===draftGeneration)feedback.textContent=e.message;}
    finally{if(ticket===generation&&draftTicket===draftGeneration){busy=false;extractButton.disabled=false;input.disabled=false;visitDate.disabled=false;}}
  }
  async function preview(){const ticket=generation,draftTicket=draftGeneration;if(!candidate)return;const submitted=JSON.stringify(candidate);try{
    const p=await request('/visit-capture/preview',{context_id:contextId,candidate});if(ticket!==generation||draftTicket!==draftGeneration||JSON.stringify(candidate)!==submitted)return;
    previews.replaceChildren();for(const [key,label]of [['snapshot','Συνοπτική εικόνα'],['brief','Προετοιμασία επίσκεψης'],['detail','Πλήρης καταγραφή']]){const d=el(previews,'details');el(d,'summary',label);el(d,'p',p[key],'clinical-preview-text');}
  }catch(e){if(ticket===generation&&draftTicket===draftGeneration)feedback.textContent=e.message;}}
  input.oninput=()=>{draftGeneration++;candidate=null;saveButton.hidden=true;review.replaceChildren();previews.replaceChildren();};visitDate.oninput=input.oninput;
  async function save(){if(busy||!candidate||!contextId)return;const ticket=generation,draftTicket=draftGeneration,id=patientId;busy=true;saveButton.disabled=true;extractButton.disabled=true;
    feedback.textContent='Αποθήκευση…';
    try{await request('/visit-capture/save',{context_id:contextId,candidate});if(ticket!==generation||draftTicket!==draftGeneration)return;await open(id);window.dispatchEvent(new Event('cockpit:clinical-updates-changed'));
    }catch(e){if(ticket!==generation||draftTicket!==draftGeneration)return;
      // A lost response may follow a committed save. Never offer a fresh write
      // blindly; the user can inspect the authoritative chart first.
      feedback.textContent=e.message+' Έλεγξε την καρτέλα πριν επαναλάβεις την καταγραφή.';
      button(review,'Έλεγχος αποθηκευμένης καρτέλας',()=>open(id));
    }
  }
  async function renderUpdates(target,active,close,scope=patientId==='cockpit-cp1-test'?'test':'real'){const ticket=generation;target.replaceChildren();el(target,'p','Φόρτωση ενημερώσεων…');
    try{const data=await request('/clinical-updates?scope='+scope);if(ticket!==generation||!active())return;target.replaceChildren();
      button(target,scope==='test'?'Πραγματικοί φάκελοι':'Δοκιμαστικός ασθενής',()=>renderUpdates(target,active,close,scope==='test'?'real':'test'));
      el(target,'p',data.items.filter(p=>p.status!=='cancelled').length?'Υποχρεώσεις από αποθηκευμένες επισκέψεις · έως 100 πρόσφατες':'Δεν έχουν καταγραφεί υποχρεώσεις.');
      for(const item of data.items.filter(p=>p.status!=='cancelled')){const b=button(target,'',async()=>{close();if(!canLeave())return;await open(item.patient_id);const current=chartPending.find(p=>p.pending_id===item.pending_id);if(current)showPending(item.patient_id,current);});b.className='doctor-action-row';const copy=el(b,'span');el(copy,'strong',item.patient_name);el(copy,'small',item.description);el(copy,'small',(item.checkpoints?.next_checkpoint?.question||statuses[item.status]||item.status)+' · '+date(item.source_date));el(b,'span','›');}
    }catch(e){if(ticket===generation&&active()){target.replaceChildren();el(target,'p',e.message);}}
  }
  $('visitBackHome').addEventListener('click',()=>{if(!canLeave())return;reset();const u=new URL(location.href);u.searchParams.delete('chart');history.replaceState(null,'',u);});
  window.addEventListener('clinical:unauthenticated',reset);
  window.addEventListener('clinical:authenticated',()=>{reset();const id=new URL(location.href).searchParams.get('chart');if(id)open(id);});
  window.CockpitStoredChart={open(id){if(!canLeave())return;return open(id);},reset,renderPending,renderUpdates,beginVisit,canLeave,cancelPendingRead(){generation++;}};
  window.addEventListener('beforeunload',e=>{if(!composer.hidden&&(input.value.trim()||candidate||busy)){e.preventDefault();e.returnValue='';}});
  const id=new URL(location.href).searchParams.get('chart');if(id)open(id);
})();
