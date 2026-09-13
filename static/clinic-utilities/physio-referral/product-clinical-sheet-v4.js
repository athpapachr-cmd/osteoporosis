'use strict';
// Compact clinical-picture interaction. Presentation only: clinical IDs,
// evidence, safety and projection authority remain unchanged.
const V4_CLINICAL = Object.freeze({
  pain:{label:'Πόνος',prompt:'Πού εντοπίζεται;'},
  stiffness:{label:'Δυσκαμψία',prompt:'Πότε εμφανίζεται;'},
  weakness:{label:'Αδυναμία',prompt:'Τι έχει διαπιστωθεί;'},
  function:{label:'Λειτουργικότητα',prompt:'Σε τι υπάρχει δυσκολία;'},
});

function v4Count(kind) {
  if(kind==='pain') return qualifierState.pain_locations.length;
  if(kind==='stiffness') return qualifierState.stiffness_patterns.length + (qualifierState.morning_stiffness_duration?1:0);
  if(kind==='weakness') return qualifierState.weakness_detail?1:0;
  return state?.functional_impairments?.length || 0;
}
function v4Active(kind) {
  if(!state) return false;
  if(kind==='pain') return qHasPain();
  if(kind==='stiffness') return qHasStiffness();
  if(kind==='weakness') return qHasWeakness();
  return !!state.functional_impairments.length;
}
function v4ClinicalButton(kind) {
  const b=btn('',{class:'clinical-card-v4','data-clinical-v4':kind,'aria-pressed':'false','aria-label':V4_CLINICAL[kind].label});
  b.append(
    make('span',{class:'clinical-card-label-v4',text:V4_CLINICAL[kind].label}),
    make('span',{class:'clinical-card-count-v4','data-clinical-count-v4':kind,hidden:''}),
    make('span',{class:'clinical-card-detail-v4','data-clinical-detail-v4':kind,'aria-hidden':'true',text:'›',hidden:''})
  );
  return b;
}
function v4InstallClinicalGrid() {
  const box=$('#phenotype'); if(!box || box.dataset.v4Installed==='1') return;
  box.dataset.v4Installed='1'; box.className='clinical-grid-v4';
  box.replaceChildren(...['pain','stiffness','weakness','function'].map(v4ClinicalButton));
  // Base draft/reset/BFCache lifecycle code still addresses this legacy anchor.
  box.append(btn('',{id:'functionToggle',class:'legacy-qualifiers-v4-hidden',hidden:'','aria-hidden':'true','aria-expanded':'false','aria-controls':'functionChoices',tabindex:'-1'}));
  document.querySelector('.smart-qualifiers')?.classList.add('legacy-qualifiers-v4-hidden');
  $('#functionChoices')?.classList.add('legacy-qualifiers-v4-hidden');
}
function v4RenderGrid() {
  if(!state) return;
  $$('[data-clinical-v4]').forEach(b=>{
    const kind=b.dataset.clinicalV4, count=v4Count(kind), active=v4Active(kind);
    b.setAttribute('aria-pressed',String(active));
    b.classList.toggle('is-active',active);
    const badge=b.querySelector('[data-clinical-count-v4]');
    if(badge){badge.hidden=count===0;badge.textContent=count?`· ${count}`:'';}
    const detail=b.querySelector('[data-clinical-detail-v4]');
    if(detail) detail.hidden=!(kind==='function'||active);
    const ariaDetail=count
      ? ` · ${count} ενεργοί προσδιορισμοί`
      : active&&kind!=='function'
        ? ' · επιλεγμένο · πάτησε ξανά για λεπτομέρειες'
        : kind==='function' ? ' · άνοιγμα επιλογών' : '';
    b.setAttribute('aria-label',V4_CLINICAL[kind].label+ariaDetail);
  });
}
function v4EnsureParent(kind) {
  if(kind==='pain' && !qHasPain()) { state.findings=[...state.findings,'pain']; return true; }
  if(kind==='stiffness' && !qHasStiffness()) { state.phenotype.stiffness_symptom=true; return true; }
  if(kind==='weakness' && !qHasWeakness()) { state.phenotype.weakness_symptom_or_context=true; return true; }
  return false;
}
function v4RemoveParent(kind) {
  if(kind==='pain'){state.findings=state.findings.filter(v=>v!=='pain');qResetPain();}
  else if(kind==='stiffness'){state.phenotype.stiffness_symptom=false;qResetStiffness();}
  else if(kind==='weakness'){state.phenotype.weakness_symptom_or_context=false;qResetWeakness();}
  else state.functional_impairments=[];
  changed('clinical_picture_'+kind+'_removed');
  closeSheet();
}
function v4Option(text,attrs){return btn(text,{class:'clinical-option-v4',...attrs});}
function v4SheetNodes(kind) {
  const nodes=[make('p',{class:'clinical-sheet-prompt-v4',text:V4_CLINICAL[kind].prompt})];
  if(kind==='pain') {
    nodes.push(make('div',{class:'clinical-options-v4'},[
      v4Option('Έσω μεσάρθρια',{'data-q-pain':'medial_joint_line'}),v4Option('Έξω μεσάρθρια',{'data-q-pain':'lateral_joint_line'}),
      v4Option('Πρόσθιος / περιεπιγονατιδικός',{'data-q-pain':'anterior_peripatellar'}),v4Option('Χήνειος πόδας',{'data-q-pain':'pes_anserine_region'}),
      v4Option('Οπίσθιος',{'data-q-pain':'posterior'}),v4Option('Διάχυτος',{'data-q-pain':'diffuse'}),
    ]),make('p',{class:'footnote',text:'Η εντόπιση είναι προαιρετική και περιγράφει μόνο το σύμπτωμα.'}));
  } else if(kind==='stiffness') {
    nodes.push(make('div',{class:'clinical-options-v4'},[
      v4Option('Πρωινή',{'data-q-stiffness':'morning'}),v4Option('Μετά από ακινησία',{'data-q-stiffness':'after_inactivity'}),
    ]),make('div',{id:'v4MorningDuration',class:'clinical-nested-v4'},[
      make('p',{class:'subtle small',text:'Διάρκεια πρωινής δυσκαμψίας'}),make('div',{class:'clinical-options-v4'},[
        v4Option('≤30′',{'data-q-duration':'le_30'}),v4Option('>30′',{'data-q-duration':'gt_30'}),
      ])
    ]),make('div',{id:'v4StiffnessClue',class:'review-clue-inline'},[
      make('strong',{text:'Μη τυπικό χαρακτηριστικό για το συνήθη OA phenotype'}),
      make('span',{text:'Η παρατεταμένη πρωινή δυσκαμψία χρειάζεται κλινική επανεκτίμηση, όχι αυτόματη αλλαγή θεραπείας.'})
    ]));
  } else if(kind==='weakness') {
    nodes.push(make('div',{class:'clinical-options-v4'},[
      v4Option('Μυϊκή αδυναμία στην εξέταση',{'data-q-weakness':'objective'}),
      v4Option('Αδυναμία τετρακεφάλου στην εξέταση',{'data-q-weakness':'quadriceps_exam'}),
    ]),make('p',{class:'footnote',text:'Οι επιλογές αυτές είναι προαιρετικά ευρήματα εξέτασης και δεν απαιτούνται για να καταγραφεί απλώς αδυναμία.'}));
  } else {
    nodes.push(make('div',{class:'clinical-options-v4'},[
      v4Option('Βάδιση',{'data-function':'walking_tolerance'}),v4Option('Σκάλες',{'data-function':'stairs'}),
      v4Option('Έγερση',{'data-function':'sit_to_stand'}),v4Option('Άσκηση',{'data-function':'sport_gym'}),
    ]));
  }
  nodes.push(make('div',{class:'clinical-sheet-footer-v4'},[
    btn(kind==='function'?'Καθαρισμός':'Αφαίρεση από την κλινική εικόνα',{class:'quiet danger-quiet-v4','data-clinical-remove-v4':kind}),
    btn('Τέλος',{class:'primary','data-clinical-done-v4':''}),
  ]));
  return nodes;
}
function v4SyncClinicalSheet() {
  if(sheetView?.type!=='clinical-v4' || !state) return;
  $$(`#sheetBody [data-q-pain]`).forEach(b=>b.setAttribute('aria-pressed',String(qualifierState.pain_locations.includes(b.dataset.qPain))));
  $$(`#sheetBody [data-q-stiffness]`).forEach(b=>b.setAttribute('aria-pressed',String(qualifierState.stiffness_patterns.includes(b.dataset.qStiffness))));
  $$(`#sheetBody [data-q-duration]`).forEach(b=>b.setAttribute('aria-pressed',String(qualifierState.morning_stiffness_duration===b.dataset.qDuration)));
  $$(`#sheetBody [data-q-weakness]`).forEach(b=>b.setAttribute('aria-pressed',String(qualifierState.weakness_detail===b.dataset.qWeakness)));
  $$(`#sheetBody [data-function]`).forEach(b=>b.setAttribute('aria-pressed',String(state.functional_impairments.includes(b.dataset.function))));
  const duration=$('#v4MorningDuration');if(duration)duration.hidden=!qualifierState.stiffness_patterns.includes('morning');
  const clue=$('#v4StiffnessClue');if(clue)clue.hidden=qualifierState.morning_stiffness_duration!=='gt_30';
  v4RenderGrid();
}
function v4OpenClinicalSheet(kind,source) {
  if(!state) return;
  const activated=v4EnsureParent(kind);
  invoker=source; evidenceBack=null; sheetView={type:'clinical-v4',item:kind};
  const dialog=$('#sheet'); dialog.classList.add('clinical-v4-sheet');
  $('#sheetSafety').hidden=true; $('#sheetTitle').textContent=V4_CLINICAL[kind].label;
  $('#sheetBody').replaceChildren(...v4SheetNodes(kind));
  v4SyncClinicalSheet();
  if(!dialog.open) dialog.showModal();
  dialog.scrollTop=0; $('#sheetTitle').focus({preventScroll:true});
  if(activated) changed('clinical_picture_'+kind+'_selected'); else paintStatus();
}

v4InstallClinicalGrid();
const v4BasePaint=paint;
paint=function(){v4BasePaint();v4RenderGrid();v4SyncClinicalSheet();};
const v4BaseOpenSheet=openSheet;
openSheet=function(type,item=null,back=null){$('#sheet').classList.remove('clinical-v4-sheet');return v4BaseOpenSheet(type,item,back);};
const v4BaseCloseSheet=closeSheet;
closeSheet=function(){$('#sheet').classList.remove('clinical-v4-sheet');return v4BaseCloseSheet();};

document.addEventListener('click',event=>{
  const b=event.target.closest('button');if(!b||!state)return;
  if(b.dataset.clinicalV4){
    const kind=b.dataset.clinicalV4;
    if(kind!=='function'&&!v4Active(kind)){
      if(v4EnsureParent(kind)) changed('clinical_picture_'+kind+'_selected');
      return;
    }
    v4OpenClinicalSheet(kind,b);return;
  }
  if(b.dataset.clinicalRemoveV4){v4RemoveParent(b.dataset.clinicalRemoveV4);return;}
  if(b.hasAttribute('data-clinical-done-v4')){closeSheet();return;}
  if(sheetView?.type==='clinical-v4' && (b.dataset.qPain||b.dataset.qStiffness||b.dataset.qDuration||b.dataset.qWeakness||b.dataset.function)) {
    queueMicrotask(v4SyncClinicalSheet);
  }
});

let v4InitialPaintFrames=0;
function v4ReconcileInitialPaint(){
  if(state){paintStatus();v4RenderGrid();return;}
  if(v4InitialPaintFrames++<120) requestAnimationFrame(v4ReconcileInitialPaint);
}
queueMicrotask(v4ReconcileInitialPaint);

window.addEventListener('pagehide',()=>{$('#sheet')?.classList.remove('clinical-v4-sheet');});
