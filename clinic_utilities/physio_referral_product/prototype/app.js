'use strict';
// In-memory synthetic prototype. The server owns validation, evidence and text projection.
const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const make = (tag, attrs = {}, children = []) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'text') node.textContent = value;
    else if (key === 'class') node.className = value;
    else if (value !== undefined && value !== null) node.setAttribute(key, String(value));
  }
  for (const child of children) node.append(typeof child === 'string' ? document.createTextNode(child) : child);
  return node;
};
const btn = (text, attrs = {}) => make('button', {type:'button', text, ...attrs});
const LABELS = {
  therapeutic_exercise:'Θεραπευτική άσκηση', progressive_strengthening:'Προοδευτική ενδυνάμωση',
  education_and_self_management:'Εκπαίδευση & αυτοδιαχείριση', mobility_exercise_when_restricted:'Κινητικότητα',
  graded_activity_exposure:'Σταδιακή δραστηριότητα', progressive_endurance_or_capacity_work:'Αντοχή & λειτουργική ικανότητα',
  neuromuscular_proprioceptive_training:'Νευρομυϊκή επανεκπαίδευση', balance_stepping_recovery_training:'Ισορροπία',
  gait_walking_practice:'Επανεκπαίδευση βάδισης', functional_task_retraining:'Λειτουργική επανεκπαίδευση',
  home_exercise_programme:'Ασκήσεις στο σπίτι', manual_therapy:'Χειροθεραπεία', soft_tissue_techniques:'Τεχνικές μαλακών μορίων',
  acupuncture:'Βελονισμός', taping:'Περίδεση (taping)', orthosis_or_brace_context:'Νάρθηκας / ορθωτικό',
};
const CUES = {recommended_or_supported:'●',conditional_or_context_dependent:'◇',limited_or_insufficient_evidence:'?',
  guideline_conflict_or_mixed:'⇄',recommendation_against_routine_use:'⊖',not_yet_assessed:'◌'};
const DIRECTIONS = {strong_for:'Ισχυρή σύσταση υπέρ',for:'Υπέρ',conditional_for:'Υπέρ υπό προϋποθέσεις',weak_for:'Ασθενής σύσταση υπέρ',
  neutral_or_insufficient:'Ανεπαρκή δεδομένα υπέρ ή κατά',conditional_against:'Υπό όρους κατά',weak_against:'Ασθενής σύσταση κατά',against:'Κατά',not_addressed:'Δεν εξετάζεται'};
function cueNode(code) {
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  for(const [key,value] of Object.entries({viewBox:'0 0 20 20',width:'14',height:'14',fill:'none',stroke:'currentColor','stroke-width':'1.6','stroke-linecap':'round','aria-hidden':'true'}))svg.setAttribute(key,value);
  const shapes={recommended_or_supported:['circle',{cx:10,cy:10,r:4,fill:'currentColor'}],conditional_or_context_dependent:['path',{d:'M10 2 18 10 10 18 2 10Z'}],limited_or_insufficient_evidence:['path',{d:'M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16M7.5 7a2.5 2.5 0 0 1 5 0c0 2-2.5 2-2.5 4M10 14h.01'}],guideline_conflict_or_mixed:['path',{d:'M3 6h14m-4-3 4 3-4 3M17 14H3m4-3-4 3 4 3'}],recommendation_against_routine_use:['path',{d:'M6 2h8l4 4v8l-4 4H6l-4-4V6ZM6 10h8'}],not_yet_assessed:['circle',{cx:10,cy:10,r:7,'stroke-dasharray':'1 3'}]};
  const [tag,attrs]=shapes[code];const shape=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [key,value] of Object.entries(attrs))shape.setAttribute(key,String(value));svg.append(shape);
  return make('span',{class:'evidence-cue','data-cue':code,'aria-hidden':'true'},[svg]);
}
const DEMO = 'ΔΟΚΙΜΑΣΤΙΚΟ · ΟΧΙ ΓΙΑ ΚΛΙΝΙΚΗ ΧΡΗΣΗ\n\n';
let meta, state, draftId, revision = 0, response = null, pending = true, seq = 0, controller = null;
let dismissed = [], manual = null, advancedOpen = false, bubbleItem = null, sheetView = null, invoker = null;
let evidenceBack = null, timer = null;
const label = id => LABELS[id] || Object.values(meta?.labels || {}).map(group=>group[id]).find(Boolean) || id;
const chosen = () => [...state.rehab_directions, ...state.adjunct_options];
const fresh = () => response && !pending && response.draft_id === draftId && response.revision === revision && response.package_version === meta.package_version;
const manualStale = () => manual && manual.baseRevision !== revision;
const canExport = () => !!(fresh() && response.gate.allowed && !response.gate.blocked && !manualStale() && effectiveText().trim());
const effectiveText = () => manual ? manual.text : response?.text || '';
const payload = () => ({draft_id:draftId,revision,package_version:meta.package_version,synthetic_only:true,state,dismissed});
function announce(text) { $('#status').textContent = text; }
function notice(text) {
  const n=$('#notice'); n.replaceChildren(document.createTextNode(text),btn('×',{'aria-label':'Κλείσιμο μηνύματος','data-hide-notice':''})); n.hidden=false;
}
function focusedKey() { return document.activeElement?.getAttribute('data-key'); }
function restoreKey(key) {
  if (!key || (document.activeElement !== document.body && document.activeElement?.isConnected && document.activeElement.getClientRects().length)) return;
  const target=$$('[data-key]').find(n=>n.getAttribute('data-key')===key && n.getClientRects().length);
  target?.focus({preventScroll:true});
}
function newDraft() {
  controller?.abort(); clearTimeout(timer); seq++;
  draftId=crypto.randomUUID(); revision=0; response=null; pending=true; dismissed=[]; manual=null; bubbleItem=null;
  state={laterality:'not_stated',formal_assertion_state:'not_stated',phenotype:{stiffness_symptom:false,weakness_symptom_or_context:false},
    findings:[],functional_impairments:[],rehab_directions:[...meta.defaults],adjunct_options:[],goals:[],explicit_restrictions:[],clinician_free_text_optional:'',safety_flags:[]};
  advancedOpen=false; $('#advanced').hidden=true; $('#advanced').replaceChildren(); $('#advancedToggle').setAttribute('aria-expanded','false');
  $('#functionChoices').hidden=true; $('#functionToggle').setAttribute('aria-expanded','false'); $('#notice').hidden=true;
  closeAll(); paint(); refresh();
}
function changed(item=null) {
  revision++; bubbleItem=item; pending=true; paint(); clearTimeout(timer); timer=setTimeout(()=>refresh(),65);
}
function toggle(category,id) {
  const list=state[category]; state[category]=list.includes(id) ? list.filter(v=>v!==id) : [...list,id]; changed(id);
}
async function refresh(candidate=null) {
  controller?.abort(); controller=new AbortController(); const token=++seq, oldId=draftId, oldRevision=revision;
  pending=true; paintStatus(); const data=JSON.parse(JSON.stringify(payload()));
  if(candidate) data.candidate=candidate;
  try {
    const r=await fetch('/api/project',{method:'POST',headers:{'Content-Type':'application/json','X-Physio-Prototype':'1'},body:JSON.stringify(data),signal:controller.signal,cache:'no-store'});
    if(!r.ok) throw new Error('projection_rejected');
    const next=await r.json();
    if(token!==seq || oldId!==draftId || oldRevision!==revision) return;
    if(next.draft_id!==draftId || next.package_version!==meta.package_version || next.revision!==revision+(candidate?1:0)) throw new Error('stale_projection');
    if(candidate){ state=next.state; revision=next.revision; bubbleItem=candidate.item_id; }
    const previousLabel=response?.readiness?.label; response=next; pending=false; paint(); if(candidate && !sheetView) $$('#plan [data-select]').find(b=>b.dataset.select===candidate.item_id)?.focus({preventScroll:true}); if(previousLabel!==next.readiness.label)announce(next.readiness.label);
  } catch(error) {
    if(error.name==='AbortError' || token!==seq) return;
    response=null; pending=false; paint(); announce('Η παραπομπή δεν είναι διαθέσιμη για εξαγωγή. Ελέγξτε τα στοιχεία ή τη σύνδεση με το τοπικό prototype.');
  }
}
function evidenceButton(item) {
  const b=btn('',{class:'info','data-evidence':item,'data-key':'info:'+item,'aria-label':'Τεκμηρίωση για '+label(item)});
  b.append(make('span',{'aria-hidden':'true',text:'i'})); return b;
}
function row(item,category) {
  const view=response?.evidence?.[item]; const code=view?.evidence_state || (meta.defaults.includes(item)?'recommended_or_supported':'conditional_or_context_dependent');
  const title=make('span',{class:'row-title '+(view?.all_sources_active?'e-'+code:'unavailable')},[
    cueNode(code),label(item),
    make('span',{class:'sr-only',text:' · '+meta.states[code].label+(view?.all_sources_active===false?' · η τεκμηρίωση χρειάζεται έλεγχο':'')})]);
  const selected=state[category].includes(item);
  const select=btn('',{class:'row-select','data-select':item,'data-category':category,'data-key':'select:'+item,'aria-pressed':selected});
  select.append(make('span',{class:'selection-mark','aria-hidden':'true'}),title);
  const node=make('div',{class:'row','data-row-item':item},[make('div',{class:'row-main'},[select,evidenceButton(item)])]);
  if(item===bubbleItem && selected && view && (!view.all_sources_active || ['guideline_conflict_or_mixed','limited_or_insufficient_evidence','recommendation_against_routine_use','not_yet_assessed'].includes(code))) {
    node.append(make('div',{class:'bubble'},[make('span',{text:view.all_sources_active?view.evidence_label:'Η τεκμηρίωση χρειάζεται έλεγχο'}),btn('×',{'data-dismiss-bubble':'','aria-label':'Κλείσιμο επισήμανσης'})]));
  }
  return node;
}
function renderPlan() {
  const nodes=meta.defaults.map(id=>row(id,'rehab_directions'));
  for(const category of ['rehab_directions','adjunct_options']) for(const id of state[category]) if(!meta.defaults.includes(id)) nodes.push(row(id,category));
  $('#plan').replaceChildren(...nodes);
  // Advanced selected interventions move into the visible plan, avoiding duplicate controls.
  $$('#advanced [data-row-item]').forEach(n=>{n.hidden=chosen().includes(n.dataset.rowItem);});
}
function smallSelection(id,category) {
  return btn(label(id),{'data-select':id,'data-category':category,'data-key':'select:'+id,'aria-pressed':state[category].includes(id)});
}
function advancedGroup(title,nodes) {
  const d=make('details',{class:'advanced-group'},[make('summary',{text:title}),...nodes]);
  d.addEventListener('toggle',()=>renderSuggestions()); return d;
}
function buildAdvanced() {
  const groups=[];
  for(const [title,category,source,excluded] of [
    ['Ευρήματα','findings','findings',['pain']],['Λειτουργία','functional_impairments','functional_impairments',['walking_tolerance','stairs','sit_to_stand','sport_gym']],
    ['Στόχοι','goals','goals',[]]]) {
    groups.push(advancedGroup(title,[make('div',{class:'chips'},Object.keys(meta.labels[source]).filter(id=>!excluded.includes(id)).map(id=>smallSelection(id,category)))]));
  }
  groups.push(advancedGroup('Παρεμβάσεις',[make('div',{class:'rows'},Object.keys(meta.labels.rehab_directions).filter(id=>!meta.defaults.includes(id)).map(id=>row(id,'rehab_directions')))]));
  groups.push(advancedGroup('Συμπληρωματικά',[make('div',{class:'rows'},Object.keys(meta.labels.adjuncts).map(id=>row(id,'adjunct_options')))]));
  const select=make('select',{id:'restrictionId','aria-label':'Είδος περιορισμού'});
  select.append(make('option',{value:'',text:'Χωρίς καταγεγραμμένο περιορισμό'}));
  for(const [id,name] of Object.entries(meta.restrictions)) select.append(make('option',{value:id,text:name}));
  const restriction=make('textarea',{id:'restrictionText',maxlength:300,rows:2,autocomplete:'off',spellcheck:'false','aria-label':'Ρητή οδηγία περιορισμού',placeholder:'Μόνο η ρητή κλινική οδηγία'});
  groups.push(advancedGroup('Περιορισμοί',[make('label',{class:'field'},['Είδος περιορισμού',select]),make('label',{class:'field'},['Οδηγία',restriction])]));
  const note=make('textarea',{id:'clinicalNote',maxlength:800,rows:3,autocomplete:'off',spellcheck:'false','aria-label':'Κλινική σημείωση',placeholder:'Δοκιμαστικό κείμενο, χωρίς ονόματα ή στοιχεία ασθενών'});
  note.value=state.clinician_free_text_optional;
  groups.push(advancedGroup('Κλινική σημείωση',[make('label',{class:'field'},['Πρόσθετη πληροφορία',note])]));
  groups.push(advancedGroup('Κλινικός έλεγχος',[make('p',{class:'footnote',text:'Δήλωσε μόνο αν υπάρχει ανεπίλυτη ανησυχία. Η μη επιλογή δεν αποτελεί φυσιολογικό έλεγχο.'}),
    make('div',{class:'chips'},Object.entries(meta.safety_labels).map(([id,name])=>btn(name,{'data-select':id,'data-category':'safety_flags','data-key':'select:'+id,'aria-pressed':state.safety_flags.includes(id)})))]));
  $('#advanced').replaceChildren(...groups);
  if(state.explicit_restrictions.length){select.value=state.explicit_restrictions[0].restriction_id;restriction.value=state.explicit_restrictions[0].state_or_value;}
  const restrictionChange=()=>{state.explicit_restrictions=select.value?[{restriction_id:select.value,state_or_value:restriction.value,source:'clinician_entered'}]:[];changed();};
  select.addEventListener('change',restrictionChange); restriction.addEventListener('input',restrictionChange);
  note.addEventListener('input',()=>{state.clinician_free_text_optional=note.value;changed();});
}
function advancedCount() {
  const routine=['walking_tolerance','stairs','sit_to_stand','sport_gym'];
  return new Set([...state.findings.filter(x=>x!=='pain'),...state.functional_impairments.filter(x=>!routine.includes(x)),...state.goals,
    ...chosen().filter(x=>!meta.defaults.includes(x)),...state.safety_flags,...state.explicit_restrictions.map(x=>x.restriction_id)]).size;
}
function currentCandidate(item) { return fresh()?response.suggestions.find(c=>c.item_id===item):null; }
function suggestionCard(candidate, inSheet=false) {
  const item=candidate.item_id; const b=btn('+',{class:'add','data-add':item,'aria-label':'Προσθήκη: '+label(item),'data-key':'add:'+item}); b.disabled=!fresh();
  const controls=[make('strong',{class:'suggestion-title',text:label(item)}),b];
  const existing=$$('[data-evidence]').find(n=>n.dataset.evidence===item && n.getClientRects().length && !n.closest('#suggestions') && !n.closest('#sheet'));
  if(inSheet || !existing) controls.push(evidenceButton(item));
  const card=make('div',{class:'suggestion'},[make('div',{class:'suggestion-top'},controls),make('p',{class:'suggestion-caption',text:candidate.source_caption}),
    make('div',{class:'suggestion-bottom'},[make('span',{class:'small subtle',text:candidate.reason_codes.includes('core_omitted')?'Βασική επιλογή':'Με βάση τις επιλογές σου'}),
      btn('Παράλειψη',{class:'suggestion-dismiss','data-dismiss-suggestion':item,'aria-label':'Παράλειψη πρότασης: '+label(item)})])]);
  return card;
}
function renderSuggestions() {
  const box=$('#suggestions'); const candidates=response?.suggestions||[];
  if(!candidates.length){box.replaceChildren();return;}
  box.replaceChildren(make('p',{class:'eyebrow',text:'ΠΡΟΤΑΣΗ'}),suggestionCard(candidates[0]));
  if(candidates.length>1) box.append(btn('Προτάσεις · '+candidates.length,{class:'suggestions-more','data-all-suggestions':''}));
}
function statusText() {
  if(pending) return 'Η παραπομπή ενημερώνεται';
  if(!response) return 'Ελέγξτε τα στοιχεία ή την τοπική σύνδεση';
  if(response.gate.blocked) return 'Απαιτείται κλινικός έλεγχος';
  if(!response.gate.allowed) return 'Συμπληρώστε τα απαραίτητα στοιχεία';
  if(manualStale()) return 'Ελέγξτε τις αλλαγές στο κείμενο';
  return response.readiness.label;
}
function paintStatus() {
  const text=statusText(); $('#reviewStatus').textContent=text; $('#mobileStatus').textContent=text;
  $$('[data-copy]').forEach(b=>b.disabled=!canExport());
  $('#sheetSafety').hidden=!response?.gate?.blocked;
  $('#manualReconcile').hidden=!manualStale(); $('#manualBadge').hidden=!manual;
  const output=$('#referralText');
  // Never retain an apparently current/exportable text after a blocking or stale transition.
  const content=fresh() && response.gate.allowed ? effectiveText() : 'Επιβεβαίωσε τη διάγνωση, επίλεξε πλευρά και έλεγξε τυχόν εκκρεμότητες.';
  output.textContent=content; output.classList.toggle('placeholder',!fresh() || !response?.gate?.allowed);
  if(sheetView?.type==='preview') {const node=$('#sheetReferralText'); if(node) node.textContent=content;}
  if(!canExport()) $('#printArea').textContent='Η εξαγωγή δεν είναι διαθέσιμη.';
}
function paint() {
  if(!meta || !state) return;
  const key=focusedKey();
  $('#assertion').setAttribute('aria-pressed',state.formal_assertion_state==='yes');
  $$('[data-side]').forEach(b=>b.setAttribute('aria-pressed',state.laterality===b.dataset.side));
  $$('[data-finding]').forEach(b=>b.setAttribute('aria-pressed',state.findings.includes(b.dataset.finding)));
  $$('[data-phenotype]').forEach(b=>b.setAttribute('aria-pressed',!!state.phenotype[b.dataset.phenotype]));
  $$('[data-function]').forEach(b=>b.setAttribute('aria-pressed',state.functional_impairments.includes(b.dataset.function)));
  renderPlan();
  $$('#advanced [data-select]').forEach(b=>b.setAttribute('aria-pressed',state[b.dataset.category].includes(b.dataset.select)));
  const count=advancedCount(); $('#advancedLabel').textContent='Περισσότερα'+(count?' · '+count+' ενεργά':'')+(state.clinician_free_text_optional?' · σημείωση':'');
  renderSuggestions(); paintStatus(); restoreKey(key);
}
function closeAll() { if($('#sheet').open) $('#sheet').close(); sheetView=null;evidenceBack=null; }
function openSheet(type,item=null,back=null) {
  const dialog=$('#sheet'); if(!dialog.open) invoker=document.activeElement;
  sheetView={type,item}; evidenceBack=back;
  $('#closeSheet').textContent=back?'‹':'×'; $('#closeSheet').setAttribute('aria-label',back?'Επιστροφή':'Κλείσιμο');
  const body=$('#sheetBody'); body.replaceChildren();
  let title='Παραπομπή';
  if(type==='evidence') {
    const view=response?.evidence?.[item]; if(!view) return notice('Η τεκμηρίωση δεν είναι διαθέσιμη ακόμη.');
    title=label(item);
    body.append(make('p',{class:'sheet-state '+(view.all_sources_active?'e-'+view.evidence_state:'unavailable')},[cueNode(view.evidence_state),view.all_sources_active?view.evidence_label:'Τελευταία καταγεγραμμένη θέση: '+view.evidence_label]));
    if(!view.all_sources_active) body.append(make('p',{class:'scope-caption',text:'Η τεκμηρίωση χρειάζεται έλεγχο. Διατηρείται η τελευταία καταγεγραμμένη θέση.'}));
    body.append(make('p',{class:'purpose',text:view.purpose}));
    for(const p of view.positions) {
      const original=make('details',{},[make('summary',{text:'Τεκμηρίωση'}),make('p',{text:p.summary,lang:'en'}),
        make('p',{text:'Ισχύς στην πηγή: '+(p.native_strength || 'Δεν διαβαθμίζεται στην καταγεγραμμένη πηγή')}),
        make('p',{class:'scope-caption',text:p.scope_label+(p.standalone_strength_allowed?'':' · Η ισχύς αφορά την ευρύτερη σύσταση, όχι αυτοτελώς αυτή την επιλογή.')})]);
      try {const url=new URL(p.locator);if(url.protocol==='https:'&&!url.search&&!url.username) original.append(make('a',{href:url.href,target:'_blank',rel:'noopener noreferrer',text:'Παραπομπή στην πηγή ↗'}));}catch(_error){}
      const source=make('div',{class:'source-position'},[make('div',{class:'source-heading'},[make('span',{text:p.source_label}),make('span',{class:'source-direction',text:DIRECTIONS[p.direction]})]),
        make('p',{class:'scope-caption',text:p.scope_label})]);
      if(p.summary_el) source.append(make('p',{class:'source-summary',text:p.summary_el}));
      if(p.availability!=='active_reviewed') source.append(make('p',{class:'scope-caption',text:'Διαθεσιμότητα πηγής: χρειάζεται έλεγχος'}));
      source.append(original); body.append(source);
    }
    const date=view.clinical_reviewed_on.split('-').reverse().join('/');
    body.append(make('p',{class:'reviewed-date',text:'Κλινική ανασκόπηση περιεχομένου: '+date}),make('p',{class:'footnote',text:'Οι σύνδεσμοι αφορούν ολόκληρες πηγές, όχι επαληθευμένο αριθμό σύστασης. Οι ελληνικές μεταφράσεις του prototype παραμένουν προς κλινική αποδοχή.'}));
    body.append(make('details',{class:'legend'},[make('summary',{text:'Ενδείξεις'}),...Object.entries(meta.states).map(([code,v])=>make('p',{class:'e-'+code},[cueNode(code),v.label]))]));
  } else if(type==='preview') {
    title='Η παραπομπή σου'; body.append(make('p',{class:'demo-stamp',text:'ΔΟΚΙΜΑΣΤΙΚΟ ΚΕΙΜΕΝΟ'}),make('p',{id:'sheetReferralText',class:'referral-text',text:effectiveText()}),
      btn(statusText(),{class:'review-status','data-review':''}),btn('Αντιγραφή',{class:'primary sheet-copy','data-copy':''}),btn('Ενέργειες',{class:'quiet','data-menu':''}));
  } else if(type==='review') {
    title='Έλεγχος παραπομπής';body.append(make('p',{text:statusText()}));
    if(!response?.gate.allowed) body.append(make('p',{class:'footnote',text:'Χρειάζεται επιβεβαιωμένη διάγνωση, πλευρά και ολοκλήρωση του ελέγχου. Ανεπίλυτες κλινικές ανησυχίες δεν αίρονται από αυτή την οθόνη.'}));
    for(const note of response?.notes||[]) body.append(make('div',{class:'row-main'},[make('div',{class:'suggestion-title'},[make('strong',{text:label(note.item_id)}),make('p',{class:'scope-caption',text:note.label})]),evidenceButton(note.item_id)]));
    if(manualStale()) body.append(btn('Έλεγχος δικού σου κειμένου',{class:'quiet','data-manual-review':''}),btn('Χρήση νέου κειμένου',{class:'quiet','data-use-generated':''}));
  } else if(type==='suggestions') {
    title='Προτάσεις';for(const c of response?.suggestions||[]) body.append(suggestionCard(c,true));
  } else if(type==='menu') {
    title='Παραπομπή';for(const [caption,key] of [['Επεξεργασία κειμένου','data-edit'],['Εκτύπωση / PDF','data-print'],['Νέα παραπομπή','data-new']]) {const action=btn(caption,{class:'menu-action',[key]:''});action.disabled=key!=='data-new'&&(!fresh()||!response.gate.allowed);body.append(action);}
  } else if(type==='manual') {
    title='Επεξεργασία';
    if(!manual){if(!fresh()||!response.gate.allowed)return notice('Συμπλήρωσε πρώτα τα απαραίτητα στοιχεία.');manual={text:response.text,baseRevision:revision};}
    body.append(make('p',{class:'footnote',text:'Το κείμενο είναι δικό σου. Οι δομημένες επιλογές δεν αλλάζουν και οι χειροκίνητες προσθήκες δεν ελέγχονται βιβλιογραφικά.'}));
    if(manualStale()) body.append(make('details',{},[make('summary',{text:'Νέο κείμενο από τις τρέχουσες επιλογές'}),make('p',{class:'source-summary',text:response?.text||'Μη διαθέσιμο'})]));
    const area=make('textarea',{class:'manual-text',id:'manualText',maxlength:5000,autocomplete:'off',spellcheck:'false','aria-label':'Επεξεργασμένο παραπεμπτικό'});area.value=manual.text;
    area.addEventListener('input',()=>{manual.text=area.value;paintStatus();});body.append(area,
      make('div',{class:'sheet-actions'},[btn('Επιβεβαίωση κειμένου',{class:'primary','data-confirm-manual':''}),btn('Χρήση αυτόματου κειμένου',{class:'quiet','data-use-generated':''})]));
  } else if(type==='copyFallback') {
    title='Αντιγραφή';const area=make('textarea',{class:'manual-text',readonly:'','aria-label':'Δοκιμαστικό κείμενο για αντιγραφή'});area.value=DEMO+effectiveText();body.append(make('p',{class:'footnote',text:'Το πρόγραμμα περιήγησης δεν επέτρεψε αυτόματη αντιγραφή. Επίλεξε το κείμενο και αντέγραψέ το.'}),area);
  } else if(type==='reset') {
    title='Νέα παραπομπή';body.append(make('p',{text:'Οι τρέχουσες επιλογές και τυχόν δικό σου κείμενο θα διαγραφούν από τη μνήμη.'}),btn('Νέα παραπομπή',{class:'primary','data-confirm-reset':''}));
  }
  $('#sheetTitle').textContent=title;
  if(!dialog.open)dialog.showModal(); dialog.scrollTop=0; $('#sheetTitle').focus({preventScroll:true});paintStatus();
}
function closeSheet() {
  if(evidenceBack){const back=evidenceBack;openSheet(back.type,back.item,back.back||null);return;}
  closeAll();
  if(invoker?.isConnected && invoker.getClientRects().length)invoker.focus({preventScroll:true});
  else $('#assertion').focus({preventScroll:true});
}
async function copyReferral() {
  if(!canExport())return notice('Η παραπομπή δεν είναι έτοιμη για εξαγωγή.');
  const atRevision=revision;
  try {await navigator.clipboard.writeText(DEMO+effectiveText());notice(atRevision===revision?'Αντιγράφηκε το δοκιμαστικό κείμενο.':'Αντιγράφηκε προηγούμενη εκδοχή. Οι επιλογές άλλαξαν.');}
  catch(_error){if(canExport())openSheet('copyFallback');}
}
function preparePrint() { $('#printArea').textContent=canExport()?DEMO+effectiveText():'Η παραπομπή δεν είναι διαθέσιμη για εξαγωγή.'; }
function resetRequest(){if(manual||state?.findings.length||state?.laterality!=='not_stated')openSheet('reset');else newDraft();}

document.addEventListener('click',async event=>{
  const b=event.target.closest('button');if(!b||b.disabled||!meta)return;
  if(b.dataset.side){state.laterality=b.dataset.side;changed();}
  else if(b.id==='assertion'){state.formal_assertion_state=state.formal_assertion_state==='yes'?'not_stated':'yes';changed();}
  else if(b.dataset.finding)toggle('findings',b.dataset.finding);
  else if(b.dataset.phenotype){const id=b.dataset.phenotype;state.phenotype[id]=!state.phenotype[id];changed();}
  else if(b.dataset.function)toggle('functional_impairments',b.dataset.function);
  else if(b.dataset.select)toggle(b.dataset.category,b.dataset.select);
  else if(b.dataset.evidence){const back=sheetView?{...sheetView,back:evidenceBack}:null;openSheet('evidence',b.dataset.evidence,back);}
  else if(b.id==='functionToggle') {const open=b.getAttribute('aria-expanded')!=='true';b.setAttribute('aria-expanded',open);$('#functionChoices').hidden=!open;}
  else if(b.id==='advancedToggle'){advancedOpen=!advancedOpen;b.setAttribute('aria-expanded',advancedOpen);$('#advanced').hidden=!advancedOpen;if(advancedOpen&&!$('#advanced').children.length)buildAdvanced();paint();}
  else if(b.dataset.add){const c=currentCandidate(b.dataset.add);if(c){const mode=sheetView?.type;await refresh(c);if(mode==='suggestions'&&sheetView?.type===mode)openSheet('suggestions');}else notice('Η πρόταση άλλαξε. Έλεγξε τις τρέχουσες επιλογές.');}
  else if(b.dataset.dismissSuggestion){const c=currentCandidate(b.dataset.dismissSuggestion);if(c){dismissed.push(c.dismiss_key);await refresh();if(sheetView?.type==='suggestions')openSheet('suggestions');}}
  else if(b.hasAttribute('data-dismiss-bubble')){bubbleItem=null;paint();}
  else if(b.hasAttribute('data-all-suggestions'))openSheet('suggestions');
  else if(b.hasAttribute('data-review'))openSheet('review',null,sheetView?.type==='preview'?{type:'preview'}:null);
  else if(b.hasAttribute('data-menu'))openSheet('menu',null,sheetView?.type==='preview'?{type:'preview'}:null);
  else if(b.hasAttribute('data-copy'))copyReferral();
  else if(b.id==='openPreview')openSheet('preview');
  else if(b.id==='closeSheet')closeSheet();
  else if(b.hasAttribute('data-edit')||b.hasAttribute('data-manual-review'))openSheet('manual');
  else if(b.hasAttribute('data-confirm-manual')){if(fresh()&&response.gate.allowed&&manual.text.trim()){manual.baseRevision=revision;closeSheet();paint();}else notice('Ολοκλήρωσε τον έλεγχο και το κείμενο πριν επιβεβαιώσεις.');}
  else if(b.hasAttribute('data-use-generated')){manual=null;closeAll();paint();}
  else if(b.hasAttribute('data-print')){if(canExport()){preparePrint();window.print();}else notice('Η παραπομπή δεν είναι έτοιμη για εξαγωγή.');}
  else if(b.id==='reset'||b.hasAttribute('data-new'))resetRequest();
  else if(b.hasAttribute('data-confirm-reset'))newDraft();
  else if(b.hasAttribute('data-hide-notice'))$('#notice').hidden=true;
});
$('#sheet').addEventListener('cancel',event=>{event.preventDefault();closeSheet();});
window.addEventListener('beforeprint',preparePrint);
window.addEventListener('afterprint',()=>{$('#printArea').textContent='';});
window.addEventListener('pagehide',()=>{
  controller?.abort();clearTimeout(timer);seq++;closeAll();response=null;manual=null;dismissed=[];state=null;pending=true;
  draftId=null;revision=0;invoker=null;$$('[aria-pressed]').forEach(n=>n.setAttribute('aria-pressed','false'));$$('[data-copy]').forEach(n=>n.disabled=true);$('#functionChoices').hidden=true;$('#advanced').hidden=true;$('#notice').hidden=true;$('#manualBadge').hidden=true;$('#manualReconcile').hidden=true;$('#reviewStatus').textContent='Συμπληρώστε τα απαραίτητα στοιχεία';$('#mobileStatus').textContent='Συμπληρώστε τα στοιχεία';$$('textarea,input').forEach(n=>{n.value='';});$('#sheetBody').replaceChildren();$('#referralText').textContent='';$('#printArea').textContent='';$('#plan').replaceChildren();$('#suggestions').replaceChildren();$('#advanced').replaceChildren();
});
window.addEventListener('pageshow',event=>{if(event.persisted&&meta)newDraft();});
(async()=>{try{const r=await fetch('/api/bootstrap',{cache:'no-store'});if(!r.ok)throw new Error('bootstrap');meta=await r.json();newDraft();}catch(_error){pending=false;notice('Δεν συνδέθηκε το τοπικό prototype. Ξεκίνα τον τοπικό server και ανανέωσε τη σελίδα.');}})();
