'use strict';
// Knee-OA V5.1: bounded post-use discoverability/examination refinement.
// Presentation and product-local qualifier state only. Evidence resolution,
// safety, selection authority and persistence boundaries remain server-owned.

const V51_QUALIFIER_DEFAULTS = Object.freeze({
  rom_restriction_present:false,
  active_flexion_restricted:false,
  passive_flexion_restricted:false,
  crepitus:false,
  stability_findings:[],
});
const V51_STABILITY_LABELS = Object.freeze({
  valgus_instability:'Αστάθεια σε βλαισότητα',
  varus_instability:'Αστάθεια σε ραιβότητα',
  anterior_instability_acl:'Πρόσθια αστάθεια / ΠΧΣ',
  posterior_instability_pcl:'Οπίσθια αστάθεια / ΟΧΣ',
});

Object.assign(QLABELS.weakness,{
  knee_extension_exam:'Αδυναμία έκτασης γόνατος / τετρακεφάλου',
  knee_flexion_exam:'Αδυναμία κάμψης γόνατος / ισχιοκνημιαίων',
  knee_extension_flexion_exam:'Αδυναμία κάμψης και έκτασης γόνατος',
});
Object.assign(QLABELS.weakness_summary,{
  knee_extension_exam:'Αδυναμία έκτασης / τετρακεφάλου',
  knee_flexion_exam:'Αδυναμία κάμψης / ισχιοκνημιαίων',
  knee_extension_flexion_exam:'Αδυναμία κάμψης + έκτασης',
});
Object.assign(QLABELS.tenderness,{
  medial_joint_line:'Αρθρική · έσω',
  lateral_joint_line:'Αρθρική · έξω',
  medial_bony:'Οστική · έσω',
  lateral_bony:'Οστική · έξω',
  pes_anserine_region:'Χήνειος πόδας',
  extensor_mechanism:'Εκτατικός μηχανισμός',
});
for(const id of ['active_rom_restricted','passive_rom_restricted']) V3_DUPLICATED_FINDINGS.add(id);

// V5 established atrophy as an objective Examination finding. Removing the
// reported/generic weakness symptom must therefore clear only weakness detail,
// not a separately documented atrophy finding.
qResetWeakness=function(){
  qualifierState.weakness_detail=null;
  if(state?.findings)state.findings=state.findings.filter(id=>!['objective_weakness','quadriceps_weakness'].includes(id));
};

function v51EnsureQualifierState(){
  if(!qualifierState)return;
  for(const [key,value] of Object.entries(V51_QUALIFIER_DEFAULTS)){
    if(!(key in qualifierState)) qualifierState[key]=Array.isArray(value)?[]:value;
  }
}
function v51RomHasSpecificDetail(){
  return !!(
    state?.findings?.includes('extension_lag') ||
    qualifierState.fixed_flexion_deformity ||
    qualifierState.active_flexion_restricted ||
    qualifierState.passive_flexion_restricted
  );
}
function v51RomActive(){
  return !!(
    qualifierState.rom_restriction_present ||
    v51RomHasSpecificDetail() ||
    state?.findings?.includes('active_rom_restricted') ||
    state?.findings?.includes('passive_rom_restricted')
  );
}
function v51ToggleQualifierArray(key,value){
  const values=qualifierState[key]||[];
  qualifierState[key]=values.includes(value)?values.filter(v=>v!==value):[...values,value];
}
function v51ClinicalHintText(kind){
  if(!['pain','stiffness','weakness'].includes(kind))return '';
  return 'Πατήστε ξανά για προαιρετικές λεπτομέρειες';
}
function v51InstallClinicalHints(){
  const box=$('#phenotype');if(!box)return false;
  for(const button of [...box.querySelectorAll('[data-clinical-v4]')]){
    if(button.closest('.v51-clinical-wrap'))continue;
    const kind=button.dataset.clinicalV4;
    const wrap=make('div',{class:'v51-clinical-wrap','data-v51-clinical-wrap':kind});
    button.replaceWith(wrap);wrap.append(button);
    if(kind!=='function')wrap.append(make('span',{class:'v51-second-tap-hint','data-v51-second-tap-hint':kind,text:v51ClinicalHintText(kind),hidden:''}));
  }
  return true;
}
function v51RenderClinicalHints(){
  if(!state)return;
  v51InstallClinicalHints();
  for(const hint of $$('[data-v51-second-tap-hint]')){
    const kind=hint.dataset.v51SecondTapHint;
    const active=v4Active(kind), count=v4Count(kind);
    hint.hidden=!(active&&count===0);
  }
}
function v51InstallReviewBubble(){
  if($('#v51ReviewBubble'))return;
  const grid=$('#phenotype');if(!grid)return;
  grid.parentElement?.insertBefore(make('div',{id:'v51ReviewBubble',class:'v51-review-bubble',role:'status',hidden:''}),grid.nextSibling);
}
function v51SafeLink(href,label,text='Οδηγία ↗'){
  if(!href)return null;
  try{
    const url=new URL(href);
    if(url.protocol!=='https:'||url.username||url.password)return null;
    return make('a',{href:url.href,target:'_blank',rel:'noopener noreferrer','aria-label':label,text});
  }catch(_error){return null;}
}
function v51RenderReviewBubble(){
  v51InstallReviewBubble();
  const host=$('#v51ReviewBubble');if(!host)return;
  const clues=response?.clinical_review_clues||[];
  if(!clues.length){host.hidden=true;host.replaceChildren();return;}
  const clue=clues[0];
  const children=[
    make('strong',{text:clue.label}),
    make('span',{text:'Περαιτέρω κλινική εκτίμηση πριν θεωρηθεί το εύρημα τυπική εικόνα ΟΑ.'}),
  ];
  const link=v51SafeLink(clue.source_url,'Άνοιγμα πηγής για '+clue.label,'Πηγή ↗');if(link)children.push(link);
  host.replaceChildren(...children);host.hidden=false;
}

const v51BaseV4SheetNodes=v4SheetNodes;
v4SheetNodes=function(kind){
  if(kind!=='weakness')return v51BaseV4SheetNodes(kind);
  return [
    make('p',{class:'clinical-sheet-prompt-v4',text:'Τι έχει διαπιστωθεί κατά την εξέταση;'}),
    make('div',{class:'clinical-options-v4'},[
      v4Option(QLABELS.weakness.knee_extension_exam,{'data-q-weakness':'knee_extension_exam'}),
      v4Option(QLABELS.weakness.knee_flexion_exam,{'data-q-weakness':'knee_flexion_exam'}),
      v4Option(QLABELS.weakness.knee_extension_flexion_exam,{'data-q-weakness':'knee_extension_flexion_exam'}),
    ]),
    make('p',{class:'footnote',text:'Προαιρετικά αντικειμενικά ευρήματα. Η απλή αναφορά αδυναμίας παραμένει ξεχωριστή από την εξέταση.'}),
    make('div',{class:'clinical-sheet-footer-v4'},[
      btn('Αφαίρεση από την κλινική εικόνα',{class:'quiet danger-quiet-v4','data-clinical-remove-v4':'weakness'}),
      btn('Τέλος',{class:'primary','data-clinical-done-v4':''}),
    ]),
  ];
};

const v51BaseV3SelectedForCategory=v3SelectedForCategory;
v3SelectedForCategory=function(id){
  const values=v51BaseV3SelectedForCategory(id);
  if(id!=='exam')return values;
  if(qualifierState.crepitus)values.push('Κριγμός');
  if(qualifierState.active_flexion_restricted)values.push('Ενεργητική κάμψη');
  if(qualifierState.passive_flexion_restricted)values.push('Παθητική κάμψη');
  if(qualifierState.rom_restriction_present&&!v51RomHasSpecificDetail())values.push('Περιορισμός εύρους κίνησης');
  for(const id of qualifierState.stability_findings||[])if(V51_STABILITY_LABELS[id])values.push(V51_STABILITY_LABELS[id]);
  return [...new Set(values)];
};

function v51Choice(text,attrs){return btn(text,{class:'v3-qualifier-choice',...attrs});}
function v51ExamSheet(){
  v51EnsureQualifierState();
  const sections=[];
  if(qHasWeakness()){
    sections.push(make('section',{class:'v3-sheet-section','data-v3-focus-target':'weakness'},[
      make('h3',{class:'v3-sheet-subtitle',text:'Δύναμη στην εξέταση'}),
      make('p',{class:'v3-sheet-note',text:'Επίλεξε μόνο το αντικειμενικό πρότυπο που διαπιστώθηκε.'}),
      make('div',{class:'v3-inline-choices'},[
        v51Choice(QLABELS.weakness.knee_extension_exam,{'data-q-weakness':'knee_extension_exam','aria-pressed':String(qualifierState.weakness_detail==='knee_extension_exam')}),
        v51Choice(QLABELS.weakness.knee_flexion_exam,{'data-q-weakness':'knee_flexion_exam','aria-pressed':String(qualifierState.weakness_detail==='knee_flexion_exam')}),
        v51Choice(QLABELS.weakness.knee_extension_flexion_exam,{'data-q-weakness':'knee_extension_flexion_exam','aria-pressed':String(qualifierState.weakness_detail==='knee_extension_flexion_exam')}),
      ]),
    ]));
  }
  sections.push(make('section',{class:'v3-sheet-section'},[
    make('h3',{class:'v3-sheet-subtitle',text:'Μυϊκή μάζα'}),
    make('div',{class:'v3-inline-choices'},[
      v51Choice('Ατροφία τετρακεφάλου',{'data-v3-atrophy-quadriceps':'','aria-pressed':String(qualifierState.visible_atrophy&&qualifierState.atrophy_location==='quadriceps')}),
    ]),
  ]));
  sections.push(v3SheetSection('Βασικά ευρήματα',[
    v3OptionRow('effusion','findings'),
    make('div',{class:'v3-option-row'},[v3Button({class:'v3-option-button','data-v51-crepitus':'','aria-pressed':String(qualifierState.crepitus)},[
      make('span',{class:'v3-option-mark','aria-hidden':'true',text:qualifierState.crepitus?'✓':''}),make('span',{text:'Κριγμός στην κίνηση'})
    ])]),
  ],'Μόνο όσα έχουν πράγματι εξεταστεί.'));

  const romActive=v51RomActive();
  sections.push(make('section',{class:'v3-sheet-section','data-v3-focus-target':'rom'},[
    make('h3',{class:'v3-sheet-subtitle',text:'Εύρος κίνησης'}),
    make('p',{class:'v3-sheet-note',text:'Ο γενικός περιορισμός μπορεί να εξειδικευτεί μόνο αν έχει εξεταστεί.'}),
    make('div',{class:'v3-option-list'},[make('div',{class:'v3-option-row'},[
      v3Button({class:'v3-option-button','data-v51-rom-parent':'','aria-pressed':String(romActive)},[
        make('span',{class:'v3-option-mark','aria-hidden':'true',text:romActive?'✓':''}),make('span',{text:'Περιορισμός εύρους κίνησης'})
      ])
    ])]),
    make('div',{class:'v51-rom-details',id:'v51RomDetails',hidden:!romActive},[
      make('div',{class:'v3-inline-choices'},[
        v51Choice('Υστέρηση ενεργητικής έκτασης',{'data-v51-extension-lag':'','aria-pressed':String(state.findings.includes('extension_lag'))}),
        v51Choice('Παθητικό έλλειμμα έκτασης',{'data-q-ffd':'','aria-pressed':String(qualifierState.fixed_flexion_deformity)}),
        v51Choice('Περιορισμός ενεργητικής κάμψης',{'data-v51-active-flexion':'','aria-pressed':String(qualifierState.active_flexion_restricted)}),
        v51Choice('Περιορισμός παθητικής κάμψης',{'data-v51-passive-flexion':'','aria-pressed':String(qualifierState.passive_flexion_restricted)}),
      ]),
      make('label',{class:'field v3-ffd-field',id:'v3FfdDegreesWrap'},['Παθητικό έλλειμμα έκτασης σε μοίρες (αν μετρήθηκε)',make('input',{id:'v3FfdDegrees',type:'number',min:'1',max:'60',step:'1',inputmode:'numeric','aria-label':'Παθητικό έλλειμμα έκτασης σε μοίρες'})]),
    ]),
  ]));

  sections.push(make('section',{class:'v3-sheet-section','data-v3-focus-target':'tenderness'},[
    make('h3',{class:'v3-sheet-subtitle',text:'Ευαισθησία στην ψηλάφηση'}),
    make('p',{class:'v3-sheet-note',text:'Εντόπιση ευρήματος, όχι ξεχωριστή διάγνωση.'}),
    make('p',{class:'v51-mini-heading',text:'Αρθρική ευαισθησία'}),
    make('div',{class:'v3-inline-choices'},[
      v51Choice('Έσω',{'data-q-tenderness':'medial_joint_line','aria-pressed':String(qualifierState.focal_tenderness_locations.includes('medial_joint_line'))}),
      v51Choice('Έξω',{'data-q-tenderness':'lateral_joint_line','aria-pressed':String(qualifierState.focal_tenderness_locations.includes('lateral_joint_line'))}),
    ]),
    make('p',{class:'v51-mini-heading',text:'Οστική ευαισθησία'}),
    make('div',{class:'v3-inline-choices'},[
      v51Choice('Έσω',{'data-q-tenderness':'medial_bony','aria-pressed':String(qualifierState.focal_tenderness_locations.includes('medial_bony'))}),
      v51Choice('Έξω',{'data-q-tenderness':'lateral_bony','aria-pressed':String(qualifierState.focal_tenderness_locations.includes('lateral_bony'))}),
    ]),
    make('p',{class:'v51-mini-heading',text:'Άλλη εστιακή ευαισθησία'}),
    make('div',{class:'v3-inline-choices'},[
      v51Choice('Χήνειος πόδας',{'data-q-tenderness':'pes_anserine_region','aria-pressed':String(qualifierState.focal_tenderness_locations.includes('pes_anserine_region'))}),
      v51Choice('Εκτατικός μηχανισμός',{'data-q-tenderness':'extensor_mechanism','aria-pressed':String(qualifierState.focal_tenderness_locations.includes('extensor_mechanism'))}),
    ]),
  ]));

  sections.push(make('section',{class:'v3-sheet-section','data-v3-focus-target':'stability'},[
    make('h3',{class:'v3-sheet-subtitle',text:'Σταθερότητα άρθρωσης'}),
    make('p',{class:'v3-sheet-note',text:'Αντικειμενικό εύρημα εξέτασης. Δεν ταυτίζεται με υποκειμενικό giving-way.'}),
    make('div',{class:'v3-inline-choices'},Object.entries(V51_STABILITY_LABELS).map(([id,text])=>
      v51Choice(text,{'data-v51-stability':id,'aria-pressed':String((qualifierState.stability_findings||[]).includes(id))})
    )),
  ]));

  const other=Object.keys(meta.labels.findings).filter(x=>!V3_DUPLICATED_FINDINGS.has(x));
  if(other.length)sections.push(v3SheetSection('Άλλα ευρήματα',other.map(x=>v3OptionRow(x,'findings'))));
  return sections;
}
v3ExamSheet=v51ExamSheet;

const v51BaseV3SyncSheetState=v3SyncSheetState;
v3SyncSheetState=function(){
  v51EnsureQualifierState();
  v51BaseV3SyncSheetState();
  if(sheetView?.type!=='advanced-v3'||sheetView.item!=='exam')return;
  const romActive=v51RomActive();
  const parent=$('#sheet [data-v51-rom-parent]');if(parent){parent.setAttribute('aria-pressed',String(romActive));const mark=parent.querySelector('.v3-option-mark');if(mark)mark.textContent=romActive?'✓':'';}
  const details=$('#v51RomDetails');if(details)details.hidden=!romActive;
  const lag=$('#sheet [data-v51-extension-lag]');if(lag)lag.setAttribute('aria-pressed',String(state.findings.includes('extension_lag')));
  const activeFlex=$('#sheet [data-v51-active-flexion]');if(activeFlex)activeFlex.setAttribute('aria-pressed',String(qualifierState.active_flexion_restricted));
  const passiveFlex=$('#sheet [data-v51-passive-flexion]');if(passiveFlex)passiveFlex.setAttribute('aria-pressed',String(qualifierState.passive_flexion_restricted));
  const crepitus=$('#sheet [data-v51-crepitus]');if(crepitus){crepitus.setAttribute('aria-pressed',String(qualifierState.crepitus));const mark=crepitus.querySelector('.v3-option-mark');if(mark)mark.textContent=qualifierState.crepitus?'✓':'';}
  $$('#sheet [data-v51-stability]').forEach(b=>b.setAttribute('aria-pressed',String((qualifierState.stability_findings||[]).includes(b.dataset.v51Stability))));
  const wrap=$('#v3FfdDegreesWrap');if(wrap)wrap.hidden=!qualifierState.fixed_flexion_deformity;
};

const v51BaseAdvancedCount=advancedCount;
advancedCount=function(){
  v51EnsureQualifierState();
  let count=v51BaseAdvancedCount();
  if(qualifierState.crepitus)count++;
  if(qualifierState.active_flexion_restricted)count++;
  if(qualifierState.passive_flexion_restricted)count++;
  if(qualifierState.rom_restriction_present&&!v51RomHasSpecificDetail())count++;
  count+=(qualifierState.stability_findings||[]).length;
  return count;
};

function v51RenderSourceShortcuts(item){
  if(sheetView?.type!=='evidence'||!item)return;
  const view=response?.evidence?.[item], body=$('#sheetBody');if(!view||!body)return;
  body.querySelector('#v51SourceShortcuts')?.remove();
  const links=[],seen=new Set();
  for(const position of view.positions||[]){
    const link=v51SafeLink(position.locator,'Άνοιγμα οδηγίας: '+position.source_label,position.source_label+' ↗');
    const href=link?.getAttribute('href');
    if(link&&href&&!seen.has(href)){seen.add(href);link.className='v51-source-shortcut';links.push(link);}
  }
  const local=view.jurisdiction;
  if(local){
    const link=v51SafeLink(local.source_provenance?.source_url,'Άνοιγμα κυπριακής οδηγίας','Κύπρος · ΟΑΥ ↗');
    const href=link?.getAttribute('href');
    if(link&&href&&!seen.has(href)){seen.add(href);link.className='v51-source-shortcut';links.push(link);}
  }
  if(!links.length)return;
  const strip=make('div',{id:'v51SourceShortcuts',class:'v51-source-shortcuts','aria-label':'Άμεσοι σύνδεσμοι οδηγιών'},[
    make('span',{class:'v51-source-shortcuts-label',text:'Πηγές'}),...links,
  ]);
  const stateNode=body.querySelector('.sheet-state');
  if(stateNode)stateNode.insertAdjacentElement('afterend',strip);else body.prepend(strip);
}
function v51PromoteSourceLinks(item){
  if(sheetView?.type!=='evidence'||!item)return;
  const view=response?.evidence?.[item];if(!view)return;
  v51RenderSourceShortcuts(item);
  const rows=[...$('#sheetBody')?.querySelectorAll('.source-position:not(.jurisdiction-position-v1)')||[]];
  for(const position of view.positions||[]){
    const row=rows.find(node=>node.querySelector('.source-heading span')?.textContent===position.source_label);
    const heading=row?.querySelector('.source-heading');
    if(!heading||heading.querySelector('.v51-source-link'))continue;
    const link=v51SafeLink(position.locator,'Άνοιγμα οδηγίας: '+position.source_label,'Οδηγία ↗');
    if(link){link.className='v51-source-link';heading.append(link);}
  }
  const local=view.jurisdiction;
  const localHeading=$('#sheetBody .jurisdiction-position-v1 .source-heading');
  if(local&&localHeading&&!localHeading.querySelector('.v51-source-link')){
    const link=v51SafeLink(local.source_provenance?.source_url,'Άνοιγμα τοπικής οδηγίας','Οδηγία ↗');
    if(link){link.className='v51-source-link';localHeading.append(link);}
  }
}

const v51BaseOpenSheet=openSheet;
openSheet=function(type,item=null,back=null){
  const result=v51BaseOpenSheet(type,item,back);
  if(type==='evidence')queueMicrotask(()=>v51PromoteSourceLinks(item));
  return result;
};

const v51BasePaint=paint;
paint=function(){
  v51EnsureQualifierState();
  v51BasePaint();
  v51RenderClinicalHints();
  v51RenderReviewBubble();
  v3SyncSheetState();
};
const v51BaseNewDraft=newDraft;
newDraft=function(){
  v51BaseNewDraft();
  v51EnsureQualifierState();
  queueMicrotask(()=>{v51RenderClinicalHints();v51RenderReviewBubble();});
};

v51EnsureQualifierState();
let v51InstallFrames=0;
function v51Install(){
  if(v51InstallClinicalHints()){
    v51InstallReviewBubble();v51RenderClinicalHints();v51RenderReviewBubble();return;
  }
  if(v51InstallFrames++<120)requestAnimationFrame(v51Install);
}
queueMicrotask(v51Install);

document.addEventListener('click',event=>{
  const b=event.target.closest('button');if(!b||!state)return;
  v51EnsureQualifierState();
  if(b.hasAttribute('data-v51-rom-parent')&&sheetView?.type==='advanced-v3'){
    const active=v51RomActive();
    if(active){
      qualifierState.rom_restriction_present=false;
      qualifierState.active_flexion_restricted=false;
      qualifierState.passive_flexion_restricted=false;
      qualifierState.fixed_flexion_deformity=false;
      qualifierState.fixed_flexion_deformity_deg=null;
      state.findings=state.findings.filter(id=>!['extension_lag','active_rom_restricted','passive_rom_restricted'].includes(id));
    }else qualifierState.rom_restriction_present=true;
    changed('rom_restriction_exam');queueMicrotask(v3SyncSheetState);return;
  }
  if(b.hasAttribute('data-v51-extension-lag')&&sheetView?.type==='advanced-v3'){
    const active=state.findings.includes('extension_lag');
    state.findings=active?state.findings.filter(id=>id!=='extension_lag'):[...state.findings,'extension_lag'];
    if(!active)qualifierState.rom_restriction_present=true;
    changed('extension_lag_exam');queueMicrotask(v3SyncSheetState);return;
  }
  if(b.hasAttribute('data-v51-active-flexion')&&sheetView?.type==='advanced-v3'){
    qualifierState.active_flexion_restricted=!qualifierState.active_flexion_restricted;
    if(qualifierState.active_flexion_restricted)qualifierState.rom_restriction_present=true;
    changed('active_flexion_restriction_exam');queueMicrotask(v3SyncSheetState);return;
  }
  if(b.hasAttribute('data-v51-passive-flexion')&&sheetView?.type==='advanced-v3'){
    qualifierState.passive_flexion_restricted=!qualifierState.passive_flexion_restricted;
    if(qualifierState.passive_flexion_restricted)qualifierState.rom_restriction_present=true;
    changed('passive_flexion_restriction_exam');queueMicrotask(v3SyncSheetState);return;
  }
  if(b.hasAttribute('data-v51-crepitus')&&sheetView?.type==='advanced-v3'){
    qualifierState.crepitus=!qualifierState.crepitus;changed('crepitus_exam');queueMicrotask(v3SyncSheetState);return;
  }
  if(b.dataset.v51Stability&&sheetView?.type==='advanced-v3'){
    v51ToggleQualifierArray('stability_findings',b.dataset.v51Stability);changed('stability_exam');queueMicrotask(v3SyncSheetState);return;
  }
  if(b.hasAttribute('data-q-ffd')&&sheetView?.type==='advanced-v3'){
    if(!qualifierState.fixed_flexion_deformity)qualifierState.rom_restriction_present=true;
    queueMicrotask(v3SyncSheetState);
  }
});

window.addEventListener('pagehide',()=>{
  if(!qualifierState)return;
  for(const [key,value] of Object.entries(V51_QUALIFIER_DEFAULTS))qualifierState[key]=Array.isArray(value)?[]:value;
});
