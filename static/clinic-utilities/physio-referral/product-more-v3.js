'use strict';
// Knee-OA usability v3: redesign `Περισσότερα` as a calm scan-first surface.
// This layer changes discoverability/visual hierarchy only. It does not own
// clinical semantics, evidence, safety, suggestions or persistence.

const V3_ROUTINE_FUNCTIONS = new Set(['walking_tolerance','stairs','sit_to_stand','sport_gym']);
const V3_DUPLICATED_FINDINGS = new Set(['pain','objective_weakness','quadriceps_weakness','extension_lag','effusion','tenderness']);
const V3_MAX_FAVORITES = 6;
const V3_CATEGORY_DEFS = Object.freeze([
  {id:'exam', title:'Εξέταση', icon:'⊙', hint:'Αντικειμενικά ευρήματα'},
  {id:'function', title:'Λειτουργία & στόχοι', icon:'↗', hint:'Πέρα από τη γρήγορη ροή'},
  {id:'rehab', title:'Αποκατάσταση', icon:'○', hint:'Πρόσθετες κατευθύνσεις'},
  {id:'adjuncts', title:'Συμπληρωματικά', icon:'＋', hint:'Επιλογές πέρα από τον ενεργητικό πυρήνα'},
  {id:'notes', title:'Περιορισμοί & σημείωση', icon:'✎', hint:'Ρητές οδηγίες και πληροφορία'},
  {id:'safety', title:'Κλινικός έλεγχος', icon:'!', hint:'Μόνο αν υπάρχει ανεπίλυτη ανησυχία'},
]);
let v3CustomizeFavorites = false;

const v3Button = (attrs, children) => make('button',{type:'button',...attrs},children);

function v3EligibleFavorite(category,item) {
  return V2_FAVORITE_CATEGORIES.has(category) && Array.isArray(state?.[category]) && !!label(item);
}
function v3FavoriteParts(key) {
  const i=key.indexOf(':'); if(i<1)return null;
  const category=key.slice(0,i),item=key.slice(i+1);
  return v3EligibleFavorite(category,item)?{category,item}:null;
}
function v3ToggleFavorite(category,item) {
  const key=v2FavoriteKey(category,item);
  if(v2Favorites.includes(key)) v2Favorites=v2Favorites.filter(v=>v!==key);
  else {
    if(v2Favorites.length>=V3_MAX_FAVORITES) { notice('Τα Συχνά μένουν σκόπιμα μικρά · έως '+V3_MAX_FAVORITES+' επιλογές.'); return; }
    v2Favorites=[...v2Favorites,key];
  }
  v3UpdateAdvancedOverview();
  v3SyncFavoriteButtons();
}
function v3PinButton(category,item) {
  const active=v2IsFavorite(category,item);
  return btn(active?'★':'☆',{class:'v3-pin','data-favorite-v3':'','data-favorite-category':category,'data-favorite-item':item,
    'aria-pressed':String(active),'aria-label':(active?'Αφαίρεση από τα Συχνά: ':'Προσθήκη στα Συχνά: ')+label(item)});
}
function v3SyncFavoriteButtons() {
  $$('[data-favorite-v3]').forEach(b=>{
    const active=v2IsFavorite(b.dataset.favoriteCategory,b.dataset.favoriteItem);
    b.textContent=active?'★':'☆'; b.setAttribute('aria-pressed',String(active));
    b.setAttribute('aria-label',(active?'Αφαίρεση από τα Συχνά: ':'Προσθήκη στα Συχνά: ')+label(b.dataset.favoriteItem));
  });
}

function v3SelectedForCategory(id) {
  const values=[];
  if(id==='exam') {
    if(qualifierState.weakness_detail) values.push(QLABELS.weakness_summary[qualifierState.weakness_detail] || 'Αδυναμία στην εξέταση');
    if(state.findings.includes('extension_lag')) values.push(label('extension_lag'));
    if(state.findings.includes('effusion')) values.push(label('effusion'));
    if(qualifierState.fixed_flexion_deformity) values.push('Παθητικό έλλειμμα έκτασης');
    if(qualifierState.focal_tenderness_locations.length) values.push('Εστιακή ευαισθησία');
    Object.keys(meta.labels.findings).filter(x=>!V3_DUPLICATED_FINDINGS.has(x)).forEach(x=>{if(state.findings.includes(x))values.push(label(x));});
  } else if(id==='function') {
    Object.keys(meta.labels.functional_impairments).filter(x=>!V3_ROUTINE_FUNCTIONS.has(x)).forEach(x=>{if(state.functional_impairments.includes(x))values.push(label(x));});
    Object.keys(meta.labels.goals).forEach(x=>{if(state.goals.includes(x))values.push(label(x));});
  } else if(id==='rehab') {
    Object.keys(meta.labels.rehab_directions).filter(x=>!meta.defaults.includes(x)).forEach(x=>{if(state.rehab_directions.includes(x))values.push(label(x));});
  } else if(id==='adjuncts') {
    Object.keys(meta.labels.adjuncts).forEach(x=>{if(state.adjunct_options.includes(x))values.push(label(x));});
  } else if(id==='notes') {
    state.explicit_restrictions.forEach(x=>values.push(meta.restrictions[x.restriction_id]||'Περιορισμός'));
    if(state.clinician_free_text_optional.trim()) values.push('Κλινική σημείωση');
  } else if(id==='safety') {
    state.safety_flags.forEach(x=>values.push(meta.safety_labels[x]||x));
  }
  return [...new Set(values)];
}
function v3SummaryText(values) {
  if(!values.length)return '';
  if(values.length<=2)return values.join(' · ');
  return values.slice(0,2).join(' · ')+' · +'+(values.length-2);
}

function v3RelevantNow() {
  if(!state)return [];
  const result=[];
  if(qHasWeakness() && !qualifierState.weakness_detail) result.push({title:'Αδυναμία στην εξέταση',detail:'Μόνο αν την έχεις εξετάσει',category:'exam',focus:'weakness'});
  const focalPain=(qualifierState.pain_locations||[]).some(x=>x!=='diffuse');
  if(focalPain && !qualifierState.focal_tenderness_locations.length) result.push({title:'Εστιακή ευαισθησία',detail:'Αν έγινε ψηλάφηση',category:'exam',focus:'tenderness'});
  const romFinding=state.findings.includes('active_rom_restricted')||state.findings.includes('passive_rom_restricted');
  if(romFinding && !qualifierState.fixed_flexion_deformity) result.push({title:'Παθητικό έλλειμμα έκτασης',detail:'Αν μετρήθηκε',category:'exam',focus:'ffd'});
  return result.slice(0,3);
}

function v3FavoriteShortcut(category,item) {
  const selected=state[category].includes(item);
  return v3Button({class:'v3-favorite-shortcut','data-select':item,'data-category':category,'aria-pressed':String(selected),'data-key':'v3fav:'+category+':'+item},[
    make('span',{class:'v3-favorite-check','aria-hidden':'true',text:selected?'✓':'○'}),
    make('span',{class:'v3-favorite-label',text:label(item)}),
  ]);
}
function v3RenderFavoritesSection() {
  const host=$('#v3Favorites'); if(!host||!state)return;
  const items=v2Favorites.map(v3FavoriteParts).filter(Boolean).slice(0,V3_MAX_FAVORITES);
  const head=make('div',{class:'v3-section-head'},[
    make('div',{},[make('p',{class:'v3-kicker',text:'ΠΡΟΣΩΠΙΚΗ ΣΥΝΤΟΜΕΥΣΗ'}),make('h3',{text:'★ Συχνά'})]),
    btn(v3CustomizeFavorites?'Τέλος':'Προσαρμογή Συχνών',{class:'v3-text-action','data-v3-customize-favorites':'','aria-pressed':String(v3CustomizeFavorites)}),
  ]);
  if(!items.length){
    host.replaceChildren(head,make('p',{class:'v3-empty',text:'Καρφίτσωσε έως '+V3_MAX_FAVORITES+' επιλογές που χρησιμοποιείς συχνά.'}));
    return;
  }
  host.replaceChildren(head,make('div',{class:'v3-favorites-grid'},items.map(x=>v3FavoriteShortcut(x.category,x.item))));
  if(v3CustomizeFavorites) host.append(make('p',{class:'v3-customize-note',text:'Άνοιξε μια κατηγορία και πάτησε ☆ για να αλλάξεις τα Συχνά.'}));
}
function v3RenderRelevantSection() {
  const host=$('#v3Relevant'); if(!host)return;
  const items=v3RelevantNow();
  if(!items.length){host.hidden=true;host.replaceChildren();return;}
  host.hidden=false;
  host.replaceChildren(
    make('div',{class:'v3-section-head'},[make('div',{},[make('p',{class:'v3-kicker',text:'ΜΕ ΒΑΣΗ ΟΣΑ ΕΧΕΙΣ ΗΔΗ ΔΗΛΩΣΕΙ'}),make('h3',{text:'Σχετικά τώρα'})])]),
    make('div',{class:'v3-relevant-list'},items.map(x=>v3Button({class:'v3-relevant-row','data-v3-category':x.category,'data-v3-focus':x.focus},[
      make('span',{class:'v3-relevant-copy'},[make('strong',{text:x.title}),make('span',{text:x.detail})]),make('span',{class:'v3-chevron','aria-hidden':'true',text:'›'})
    ])))
  );
}
function v3CategoryRow(def) {
  const selected=v3SelectedForCategory(def.id); const count=selected.length;
  return v3Button({class:'v3-category-row','data-v3-category':def.id,'aria-label':def.title+(count?' · '+count+' ενεργά':'')},[
    make('span',{class:'v3-category-icon','aria-hidden':'true',text:def.icon}),
    make('span',{class:'v3-category-copy'},[
      make('strong',{text:def.title}),
      make('span',{class:'v3-category-summary',text:count?v3SummaryText(selected):def.hint}),
    ]),
    make('span',{class:'v3-category-side'},[
      make('span',{class:'v3-category-count',text:count?count+' ενεργά':''}),
      make('span',{class:'v3-chevron','aria-hidden':'true',text:'›'}),
    ])
  ]);
}
function v3RenderCategoryList() {
  const host=$('#v3Categories'); if(!host)return;
  host.replaceChildren(
    make('div',{class:'v3-section-head v3-all-head'},[make('div',{},[make('p',{class:'v3-kicker',text:'ΠΛΗΡΗΣ ΠΡΟΣΒΑΣΗ'}),make('h3',{text:'Όλα'})])]),
    make('div',{class:'v3-category-list'},V3_CATEGORY_DEFS.map(v3CategoryRow))
  );
}
function v3BuildAdvanced() {
  const advanced=$('#advanced'); if(!advanced)return;
  advanced.replaceChildren(make('div',{class:'advanced-v3','data-v3-more':''},[
    make('section',{id:'v3Favorites',class:'v3-block','aria-label':'Συχνά'}),
    make('section',{id:'v3Relevant',class:'v3-block v3-relevant-block','aria-label':'Σχετικά τώρα',hidden:''}),
    make('section',{id:'v3Categories',class:'v3-block','aria-label':'Όλες οι κατηγορίες'}),
  ]));
  v3UpdateAdvancedOverview();
}
function v3UpdateAdvancedOverview() {
  if(!$('#advanced [data-v3-more]'))return;
  v3RenderFavoritesSection(); v3RenderRelevantSection(); v3RenderCategoryList(); v3SyncFavoriteButtons();
}

function v3OptionRow(item,category,{evidence=false,pinnable=true}={}) {
  const selected=state[category].includes(item);
  const select=v3Button({class:'v3-option-button','data-select':item,'data-category':category,'data-key':'v3select:'+item,'aria-pressed':String(selected)},[
    make('span',{class:'v3-option-mark','aria-hidden':'true',text:selected?'✓':''}),make('span',{text:label(item)})
  ]);
  const actions=[];
  if(evidence) actions.push(evidenceButton(item));
  if(v3CustomizeFavorites&&pinnable&&v3EligibleFavorite(category,item)) actions.push(v3PinButton(category,item));
  return make('div',{class:'v3-option-row','data-v3-item':item},[select,make('div',{class:'v3-option-actions'},actions)]);
}
function v3SheetSection(title,nodes,note='') {
  const children=[make('h3',{class:'v3-sheet-subtitle',text:title})];
  if(note)children.push(make('p',{class:'v3-sheet-note',text:note}));
  children.push(make('div',{class:'v3-option-list'},nodes));
  return make('section',{class:'v3-sheet-section'},children);
}
function v3ExamSheet() {
  const sections=[];
  if(qHasWeakness()) {
    sections.push(make('section',{class:'v3-sheet-section','data-v3-focus-target':'weakness'},[
      make('h3',{class:'v3-sheet-subtitle',text:'Αδυναμία στην εξέταση'}),
      make('p',{class:'v3-sheet-note',text:'Δήλωσέ την μόνο όταν πρόκειται για εύρημα εξέτασης.'}),
      make('div',{class:'v3-inline-choices'},[
        btn(QLABELS.weakness.objective,{class:'v3-qualifier-choice','data-q-weakness':'objective','aria-pressed':String(qualifierState.weakness_detail==='objective')}),
        btn(QLABELS.weakness.quadriceps_exam,{class:'v3-qualifier-choice','data-q-weakness':'quadriceps_exam','aria-pressed':String(qualifierState.weakness_detail==='quadriceps_exam')}),
        btn('Ατροφία',{class:'v3-qualifier-choice','data-q-atrophy':'','aria-pressed':String(qualifierState.visible_atrophy)}),
      ]),
      make('div',{id:'v3AtrophyLocation',class:'v3-inline-choices v3-nested'},Object.entries(QLABELS.atrophy).map(([id,name])=>btn(name,{class:'v3-qualifier-choice','data-q-atrophy-location':id,'aria-pressed':String(qualifierState.atrophy_location===id)}))),
    ]));
  }
  sections.push(v3SheetSection('Βασικά ευρήματα',[
    v3OptionRow('extension_lag','findings'),v3OptionRow('effusion','findings')
  ],'Μόνο όσα έχουν πράγματι εξεταστεί.'));
  sections.push(make('section',{class:'v3-sheet-section','data-v3-focus-target':'ffd'},[
    make('h3',{class:'v3-sheet-subtitle',text:'Έκταση γόνατος'}),
    make('div',{class:'v3-option-list'},[make('div',{class:'v3-option-row'},[
      v3Button({class:'v3-option-button','data-q-ffd':'','aria-pressed':String(qualifierState.fixed_flexion_deformity)},[
        make('span',{class:'v3-option-mark','aria-hidden':'true',text:qualifierState.fixed_flexion_deformity?'✓':''}),make('span',{text:'Παθητικό έλλειμμα έκτασης'})
      ])
    ])]),
    make('label',{class:'field v3-ffd-field',id:'v3FfdDegreesWrap'},['Έλλειμμα σε μοίρες (αν μετρήθηκε)',make('input',{id:'v3FfdDegrees',type:'number',min:'1',max:'60',step:'1',inputmode:'numeric','aria-label':'Παθητικό έλλειμμα έκτασης σε μοίρες'})]),
  ]));
  sections.push(make('section',{class:'v3-sheet-section','data-v3-focus-target':'tenderness'},[
    make('h3',{class:'v3-sheet-subtitle',text:'Εστιακή ευαισθησία'}),make('p',{class:'v3-sheet-note',text:'Περιγράφει εύρημα ψηλάφησης, όχι ξεχωριστή διάγνωση.'}),
    make('div',{class:'v3-inline-choices'},Object.entries(QLABELS.tenderness).map(([id,name])=>btn(name,{class:'v3-qualifier-choice','data-q-tenderness':id,'aria-pressed':String(qualifierState.focal_tenderness_locations.includes(id))})))
  ]));
  const other=Object.keys(meta.labels.findings).filter(x=>!V3_DUPLICATED_FINDINGS.has(x));
  if(other.length)sections.push(v3SheetSection('Άλλα ευρήματα',other.map(x=>v3OptionRow(x,'findings'))));
  return sections;
}
function v3FunctionSheet() {
  const advancedFunctions=Object.keys(meta.labels.functional_impairments).filter(x=>!V3_ROUTINE_FUNCTIONS.has(x));
  return [
    v3SheetSection('Λειτουργία',advancedFunctions.map(x=>v3OptionRow(x,'functional_impairments')),'Οι τέσσερις συχνές λειτουργίες παραμένουν στη βασική οθόνη.'),
    v3SheetSection('Στόχοι',Object.keys(meta.labels.goals).map(x=>v3OptionRow(x,'goals'))),
  ];
}
function v3RehabSheet() {
  return [v3SheetSection('Πρόσθετες κατευθύνσεις',Object.keys(meta.labels.rehab_directions).filter(x=>!meta.defaults.includes(x)).map(x=>v3OptionRow(x,'rehab_directions',{evidence:true})),
    'Οι βασικές προτεραιότητες παραμένουν στο κύριο πλάνο. Πρόταση ≠ επιλογή.')];
}
function v3AdjunctSheet() {
  return [v3SheetSection('Συμπληρωματικές επιλογές',Object.keys(meta.labels.adjuncts).map(x=>v3OptionRow(x,'adjunct_options',{evidence:true})),
    'Η τεκμηρίωση μπορεί να διαφέρει ανά οδηγία. Η επιλογή παραμένει δική σου.')];
}
function v3NotesSheet() {
  const select=make('select',{id:'v3RestrictionId','aria-label':'Είδος περιορισμού'});
  select.append(make('option',{value:'',text:'Χωρίς καταγεγραμμένο περιορισμό'}));
  for(const [id,name] of Object.entries(meta.restrictions))select.append(make('option',{value:id,text:name}));
  const restriction=make('textarea',{id:'v3RestrictionText',maxlength:300,rows:2,autocomplete:'off',spellcheck:'false','aria-label':'Ρητή οδηγία περιορισμού',placeholder:'Μόνο η ρητή κλινική οδηγία'});
  if(state.explicit_restrictions.length){select.value=state.explicit_restrictions[0].restriction_id;restriction.value=state.explicit_restrictions[0].state_or_value;}
  const note=make('textarea',{id:'v3ClinicalNote',maxlength:800,rows:3,autocomplete:'off',spellcheck:'false','aria-label':'Κλινική σημείωση',placeholder:'Δοκιμαστικό κείμενο, χωρίς ονόματα ή στοιχεία ασθενών'});note.value=state.clinician_free_text_optional;
  const restrictionChange=()=>{state.explicit_restrictions=select.value?[{restriction_id:select.value,state_or_value:restriction.value,source:'clinician_entered'}]:[];changed();};
  select.addEventListener('change',restrictionChange);restriction.addEventListener('input',restrictionChange);
  note.addEventListener('input',()=>{state.clinician_free_text_optional=note.value;changed();});
  return [
    make('section',{class:'v3-sheet-section'},[make('h3',{class:'v3-sheet-subtitle',text:'Περιορισμοί'}),make('label',{class:'field'},['Είδος περιορισμού',select]),make('label',{class:'field'},['Οδηγία',restriction])]),
    make('section',{class:'v3-sheet-section'},[make('h3',{class:'v3-sheet-subtitle',text:'Κλινική σημείωση'}),make('label',{class:'field'},['Πρόσθετη πληροφορία',note])]),
  ];
}
function v3SafetySheet() {
  return [v3SheetSection('Ανεπίλυτες ανησυχίες',Object.entries(meta.safety_labels).map(([id,name])=>{
    const selected=state.safety_flags.includes(id);
    return make('div',{class:'v3-option-row'},[v3Button({class:'v3-option-button','data-select':id,'data-category':'safety_flags','aria-pressed':String(selected)},[
      make('span',{class:'v3-option-mark','aria-hidden':'true',text:selected?'✓':''}),make('span',{text:name})
    ])]);
  }),'Η μη επιλογή δεν αποτελεί φυσιολογικό έλεγχο.')];
}
function v3CategoryContent(id) {
  if(id==='exam')return v3ExamSheet();
  if(id==='function')return v3FunctionSheet();
  if(id==='rehab')return v3RehabSheet();
  if(id==='adjuncts')return v3AdjunctSheet();
  if(id==='notes')return v3NotesSheet();
  if(id==='safety')return v3SafetySheet();
  return [];
}
function v3SyncSheetState() {
  if(sheetView?.type!=='advanced-v3')return;
  $$('#sheet [data-select][data-category]').forEach(b=>{
    const list=state[b.dataset.category]; if(Array.isArray(list)){
      const active=list.includes(b.dataset.select);b.setAttribute('aria-pressed',String(active));
      const mark=b.querySelector('.v3-option-mark');if(mark)mark.textContent=active?'✓':'';
    }
  });
  const ffd=$('#sheet [data-q-ffd]');if(ffd){ffd.setAttribute('aria-pressed',String(qualifierState.fixed_flexion_deformity));const mark=ffd.querySelector('.v3-option-mark');if(mark)mark.textContent=qualifierState.fixed_flexion_deformity?'✓':'';}
  const wrap=$('#v3FfdDegreesWrap');if(wrap)wrap.hidden=!qualifierState.fixed_flexion_deformity;
  const input=$('#v3FfdDegrees');if(input&&document.activeElement!==input)input.value=qualifierState.fixed_flexion_deformity_deg??'';
  const atrophy=$('#v3AtrophyLocation');if(atrophy)atrophy.hidden=!qualifierState.visible_atrophy;
  v3SyncFavoriteButtons();
}
function v3OpenCategorySheet(id,focus='') {
  const def=V3_CATEGORY_DEFS.find(x=>x.id===id);if(!def)return;
  const dialog=$('#sheet');if(!dialog.open)invoker=document.activeElement;
  sheetView={type:'advanced-v3',item:id};evidenceBack=null;
  $('#closeSheet').textContent='×';$('#closeSheet').setAttribute('aria-label','Κλείσιμο');
  const body=$('#sheetBody');body.replaceChildren(...v3CategoryContent(id));
  $('#sheetTitle').textContent=def.title;
  if(!dialog.open)dialog.showModal();dialog.scrollTop=0;$('#sheetTitle').focus({preventScroll:true});
  const input=$('#v3FfdDegrees');
  input?.addEventListener('input',event=>{
    const raw=event.target.value;
    if(raw===''){event.target.setCustomValidity('');event.target.removeAttribute('aria-invalid');qualifierState.fixed_flexion_deformity_deg=null;}
    else {const value=Number.parseInt(raw,10);if(!Number.isInteger(value)||value<1||value>60){event.target.setCustomValidity('Καταχώρισε θετικό έλλειμμα 1–60° ή άφησέ το κενό.');event.target.setAttribute('aria-invalid','true');return;}event.target.setCustomValidity('');event.target.removeAttribute('aria-invalid');qualifierState.fixed_flexion_deformity_deg=value;}
    if(!qualifierState.fixed_flexion_deformity)qualifierState.fixed_flexion_deformity=true;changed('fixed_flexion_deformity');
  });
  v3SyncSheetState();paintStatus();
  if(focus)queueMicrotask(()=>body.querySelector(`[data-v3-focus-target="${focus}"] button, [data-v3-focus-target="${focus}"] input`)?.focus({preventScroll:true}));
}

const v3BaseOpenSheet=openSheet;
openSheet=function(type,item=null,back=null){
  if(type==='advanced-v3'){v3OpenCategorySheet(item);return;}
  v3BaseOpenSheet(type,item,back);
};

// Supersede the v2 dense advanced renderer while preserving v2 direct-edit and
// secondary-suggestion behavior.
buildAdvanced=v3BuildAdvanced;
v2InstallFavoriteControls=function(){};
v2RenderFavorites=function(){v3UpdateAdvancedOverview();};
v2SyncFavoriteToggles=function(){v3SyncFavoriteButtons();};
const v3BasePaint=paint;
paint=function(){v3BasePaint();v3UpdateAdvancedOverview();v3SyncSheetState();};
const v3BaseNewDraft=newDraft;
newDraft=function(){v3CustomizeFavorites=false;v3BaseNewDraft();};

// Rebuild only the advanced overview on interaction. Category sheets remain the
// deliberate detail surface.
document.addEventListener('click',event=>{
  const b=event.target.closest('button');if(!b||!state)return;
  if(b.hasAttribute('data-v3-customize-favorites')){
    v3CustomizeFavorites=!v3CustomizeFavorites;v3UpdateAdvancedOverview();return;
  }
  if(b.hasAttribute('data-favorite-v3')){
    event.stopImmediatePropagation();v3ToggleFavorite(b.dataset.favoriteCategory,b.dataset.favoriteItem);return;
  }
  if(b.dataset.v3Category){v3OpenCategorySheet(b.dataset.v3Category,b.dataset.v3Focus||'');return;}
  if(b.closest('#sheet')&&sheetView?.type==='advanced-v3')queueMicrotask(v3SyncSheetState);
});
window.addEventListener('pagehide',()=>{v3CustomizeFavorites=false;});
