'use strict';
// Post-review product-owner refinement. This classic script intentionally shares
// the global lexical environment with app.js; it does not create persistence or
// a second clinical engine.

const ACTIVE_JURISDICTION = Object.freeze({
  id: 'CY_GESY',
  label: 'Κύπρος · ΓεΣΥ',
  note: 'Ενεργό τοπικό πλαίσιο. Οι κυπριακές συστάσεις δεν έχουν ακόμη ενσωματωθεί ανά παρέμβαση· η διεθνής τεκμηρίωση παραμένει ο κλινικός πυρήνας.',
});

const qualifierDefaults = () => ({
  pain_locations: [],
  stiffness_patterns: [],
  morning_stiffness_duration: null,
  weakness_detail: null,
  visible_atrophy: false,
  atrophy_location: null,
  fixed_flexion_deformity: false,
  fixed_flexion_deformity_deg: null,
  focal_tenderness_locations: [],
});
let qualifierState = qualifierDefaults();

const QLABELS = {
  pain: {
    medial_joint_line: 'Έσω μεσάρθρια', lateral_joint_line: 'Έξω μεσάρθρια',
    anterior_peripatellar: 'Πρόσθιος / περιεπιγονατιδικός', pes_anserine_region: 'Χήνειος πόδας',
    posterior: 'Οπίσθιος', diffuse: 'Διάχυτος',
  },
  stiffness: {morning: 'Πρωινή', after_inactivity: 'Μετά από ακινησία'},
  duration: {le_30: '≤30′', gt_30: '>30′'},
  weakness: {objective: 'Αντικειμενική στην εξέταση', quadriceps_exam: 'Τετρακέφαλος στην εξέταση'},
  weakness_summary: {objective: 'Αντικειμενική', quadriceps_exam: 'Τετρακέφαλος · εξέταση'},
  atrophy: {quadriceps: 'Τετρακέφαλος', peri_knee_general: 'Περιαρθρικά'},
  tenderness: {medial_joint_line: 'Έσω μεσάρθρια', lateral_joint_line: 'Έξω μεσάρθρια', pes_anserine_region: 'Χήνειος πόδας'},
};

function qJoin(values) { return values.filter(Boolean).join(' · '); }
function qHasPain() { return !!state?.findings?.includes('pain'); }
function qHasStiffness() { return !!state?.phenotype?.stiffness_symptom; }
function qHasWeakness() { return !!state?.phenotype?.weakness_symptom_or_context; }
function qResetPain() { qualifierState.pain_locations = []; }
function qResetStiffness() { qualifierState.stiffness_patterns = []; qualifierState.morning_stiffness_duration = null; }
function qResetWeakness() {
  qualifierState.weakness_detail = null; qualifierState.visible_atrophy = false; qualifierState.atrophy_location = null;
  if (state?.findings) state.findings = state.findings.filter(id => !['objective_weakness','quadriceps_weakness'].includes(id));
}
function qResetExam() {
  qualifierState.fixed_flexion_deformity = false; qualifierState.fixed_flexion_deformity_deg = null;
  qualifierState.focal_tenderness_locations = [];
  if (state?.findings) state.findings = state.findings.filter(id => id !== 'tenderness');
}

function installAmendmentUI() {
  const intro = $('.intro');
  const title = intro?.querySelector('h1');
  const diagnosis = $('#assertion');
  if (title) title.textContent = 'Παραπομπή φυσιοθεραπείας';
  if (diagnosis) {
    diagnosis.className = 'diagnosis-choice';
    diagnosis.setAttribute('aria-label', 'Επιλογή διάγνωσης: Οστεοαρθρίτιδα γόνατος');
    diagnosis.replaceChildren(
      make('span',{class:'diagnosis-choice-copy'},[
        make('span',{class:'diagnosis-choice-label',text:'Διάγνωση'}),
        make('strong',{text:'Οστεοαρθρίτιδα γόνατος'}),
      ]),
      make('span',{id:'diagnosisChoiceState',class:'diagnosis-choice-state',text:'Επίλεξε'}),
    );
    if (!$('#diagnosisRequiredHint')) diagnosis.after(make('p',{id:'diagnosisRequiredHint',class:'required-hint',text:'Απαιτείται επιλογή διάγνωσης.'}));
  }
  const side = $('#side');
  const sideSection = side?.closest('section');
  if (sideSection) {
    sideSection.id = 'lateralitySection';
    if (!$('#sideRequiredHint')) side.after(make('p',{id:'sideRequiredHint',class:'required-hint',text:'Επίλεξε πλευρά για την παραπομπή.'}));
  }
  if (intro && !$('#jurisdictionProfile')) {
    const context = make('div',{id:'jurisdictionProfile',class:'jurisdiction-profile','data-jurisdiction':ACTIVE_JURISDICTION.id},[
      make('span',{class:'jurisdiction-label',text:'Πλαίσιο · '+ACTIVE_JURISDICTION.label}),
      btn('i',{class:'jurisdiction-info','data-jurisdiction-info':'','aria-label':'Πληροφορίες για το τοπικό πλαίσιο'}),
    ]);
    intro.append(context);
  }
  const quad = $('[data-q-weakness="quadriceps"]');
  if (quad) { quad.dataset.qWeakness='quadriceps_exam'; quad.textContent=QLABELS.weakness.quadriceps_exam; }
  const objective = $('[data-q-weakness="objective"]');
  if (objective) objective.textContent=QLABELS.weakness.objective;
  const mobileDock = $('#mobileDock');
  if (mobileDock && !$('#mobileManualReview')) {
    const review = btn('Έλεγχος αλλαγών',{id:'mobileManualReview',class:'mobile-manual-review','data-manual-review':''});
    review.hidden=true;
    mobileDock.insertBefore(review,mobileDock.querySelector('[data-copy]'));
  }
}

// Inject the bounded product-local qualifier overlay into the existing local
// projection request. All validation and final text remain server-owned.
const qFetch = window.fetch.bind(window);
window.fetch = function(input, init = {}) {
  const url = typeof input === 'string' ? input : input?.url;
  if (url === '/api/project' && init?.body) {
    try {
      const body = JSON.parse(init.body);
      if (body?.state) body.state.qualifiers = JSON.parse(JSON.stringify(qualifierState));
      init = {...init, body: JSON.stringify(body)};
    } catch (_error) {}
  }
  return qFetch(input, init);
};

function diagnosisMissing() { return state?.formal_assertion_state !== 'yes'; }
function sideMissing() { return !!state && !['right','left','bilateral'].includes(state.laterality); }
function missingRequirement() {
  if (diagnosisMissing()) return {id:'diagnosis', label:'Επίλεξε διάγνωση', target:$('#assertion')};
  if (sideMissing()) return {id:'laterality', label:'Επίλεξε πλευρά', target:$('#side')};
  return null;
}
function focusMissingRequirement() {
  const missing=missingRequirement();
  if (!missing) return false;
  missing.target?.scrollIntoView({block:'center',behavior:'smooth'});
  const focusTarget = missing.id==='laterality' ? $('#side button') : missing.target;
  focusTarget?.focus({preventScroll:true});
  notice(missing.label+'.');
  return true;
}
function renderRequiredUI() {
  if (!state) return;
  const diagnosisIsMissing=diagnosisMissing();
  // Keep only the next unresolved prerequisite in the error state so the screen
  // does not become a forest of red controls.
  const sideIsMissing=!diagnosisIsMissing && sideMissing();
  const diagnosis=$('#assertion');
  diagnosis?.classList.toggle('required-missing',diagnosisIsMissing);
  diagnosis?.setAttribute('aria-invalid',String(diagnosisIsMissing));
  const diagnosisHint=$('#diagnosisRequiredHint'); if(diagnosisHint) diagnosisHint.hidden=!diagnosisIsMissing;
  const indicator=$('#diagnosisChoiceState');
  if(indicator) indicator.textContent=diagnosisIsMissing?'Επίλεξε':'Επιλεγμένη ✓';
  $('#lateralitySection')?.classList.toggle('required-missing',sideIsMissing);
  $('#side')?.setAttribute('aria-invalid',String(sideIsMissing));
  const sideHint=$('#sideRequiredHint'); if(sideHint) sideHint.hidden=!sideIsMissing;
  const mobileReview=$('#mobileManualReview'); if(mobileReview) mobileReview.hidden=!manualStale();
}

const qBaseStatusText = statusText;
statusText = function() {
  if (state) {
    const missing=missingRequirement();
    if (missing) return missing.label;
  }
  return qBaseStatusText();
};

const qBasePaintStatus = paintStatus;
paintStatus = function() {
  qBasePaintStatus();
  if (!state) return;
  const missing=missingRequirement();
  if (missing && $('#referralText')) {
    $('#referralText').textContent = missing.id==='diagnosis'
      ? 'Επίλεξε τη διάγνωση για να δημιουργηθεί η παραπομπή.'
      : 'Η διάγνωση έχει επιλεγεί. Επίλεξε πλευρά για να δημιουργηθεί η παραπομπή.';
  }
  renderRequiredUI();
};

const qBaseNewDraft = newDraft;
newDraft = function() {
  qualifierState = qualifierDefaults();
  qBaseNewDraft();
  queueMicrotask(()=>{renderQualifierUI();renderRequiredUI();});
};
const qBasePaint = paint;
paint = function() { qBasePaint(); renderQualifierUI(); renderRequiredUI(); };
const qBaseAdvancedCount = advancedCount;
advancedCount = function() {
  let count = qBaseAdvancedCount();
  if (qualifierState.fixed_flexion_deformity) count += 1;
  if (qualifierState.focal_tenderness_locations.length) count += 1;
  return count;
};
const qBaseBuildAdvanced = buildAdvanced;
buildAdvanced = function() {
  qBaseBuildAdvanced();
  appendExamGroup();
  hideDuplicatedFindingControls();
  renderQualifierUI();
};

function simplifyEvidenceSheet(item) {
  const view=response?.evidence?.[item];
  if (!view || view.evidence_state==='guideline_conflict_or_mixed') return;
  const body=$('#sheetBody');
  const positions=[...body.querySelectorAll('.source-position')];
  if (!positions.length) return;
  const deep=make('details',{class:'evidence-deep'},[make('summary',{text:'Αναλυτικές θέσεις πηγών'})]);
  positions.forEach(node=>deep.append(node));
  const reviewed=body.querySelector('.reviewed-date');
  if (reviewed) reviewed.before(deep); else body.append(deep);
}

const qBaseOpenSheet = openSheet;
openSheet = function(type, item = null, back = null) {
  if (type==='review' && missingRequirement()) { focusMissingRequirement(); return; }
  qBaseOpenSheet(type, item, back);
  if (type === 'review') appendClinicalReviewClues();
  if (type === 'evidence') simplifyEvidenceSheet(item);
};

function smartText(kind) {
  if (kind === 'pain') {
    const values = qualifierState.pain_locations.map(id => QLABELS.pain[id]);
    return values.length ? qJoin(values) : 'Εντόπιση ›';
  }
  if (kind === 'stiffness') {
    const values = qualifierState.stiffness_patterns.map(id => QLABELS.stiffness[id]);
    if (qualifierState.stiffness_patterns.includes('morning') && qualifierState.morning_stiffness_duration) {
      const i = values.indexOf('Πρωινή'); if (i >= 0) values[i] += ' ' + QLABELS.duration[qualifierState.morning_stiffness_duration];
    }
    return values.length ? qJoin(values) : 'Χαρακτήρας ›';
  }
  const values = [];
  if (qualifierState.weakness_detail) values.push(QLABELS.weakness_summary[qualifierState.weakness_detail]);
  if (qualifierState.visible_atrophy) values.push('Ατροφία' + (qualifierState.atrophy_location ? ' · ' + QLABELS.atrophy[qualifierState.atrophy_location] : ''));
  return values.length ? qJoin(values) : 'Προσδιορισμός ›';
}

function collapseQualifier(kind) {
  const root=document.querySelector(`[data-smart-root="${kind}"]`); if(!root)return;
  const panel=root.querySelector('[data-smart-panel]'); const summary=root.querySelector('[data-smart-summary]');
  if(panel) panel.hidden=true;
  if(summary) summary.setAttribute('aria-expanded','false');
}
function renderQualifierUI() {
  if (!state) return;
  const parents = {pain:qHasPain(), stiffness:qHasStiffness(), weakness:qHasWeakness()};
  for (const [kind, active] of Object.entries(parents)) {
    const root = document.querySelector(`[data-smart-root="${kind}"]`);
    if (!root) continue;
    root.hidden = !active;
    const summary = root.querySelector('[data-smart-summary]');
    if (summary) summary.textContent = smartText(kind);
    if (!active) collapseQualifier(kind);
  }
  $$('[data-q-pain]').forEach(b => b.setAttribute('aria-pressed', qualifierState.pain_locations.includes(b.dataset.qPain)));
  $$('[data-q-stiffness]').forEach(b => b.setAttribute('aria-pressed', qualifierState.stiffness_patterns.includes(b.dataset.qStiffness)));
  $$('[data-q-duration]').forEach(b => b.setAttribute('aria-pressed', qualifierState.morning_stiffness_duration === b.dataset.qDuration));
  $$('[data-q-weakness]').forEach(b => b.setAttribute('aria-pressed', qualifierState.weakness_detail === b.dataset.qWeakness));
  $$('[data-q-atrophy]').forEach(b => b.setAttribute('aria-pressed', qualifierState.visible_atrophy));
  $$('[data-q-atrophy-location]').forEach(b => b.setAttribute('aria-pressed', qualifierState.atrophy_location === b.dataset.qAtrophyLocation));
  $$('[data-q-ffd]').forEach(b => b.setAttribute('aria-pressed', qualifierState.fixed_flexion_deformity));
  $$('[data-q-tenderness]').forEach(b => b.setAttribute('aria-pressed', qualifierState.focal_tenderness_locations.includes(b.dataset.qTenderness)));
  const morningDuration = $('#morningDuration'); if (morningDuration) morningDuration.hidden = !qualifierState.stiffness_patterns.includes('morning');
  const atrophyLocation = $('#atrophyLocation'); if (atrophyLocation) atrophyLocation.hidden = !qualifierState.visible_atrophy;
  const ffdDegrees = $('#ffdDegreesWrap'); if (ffdDegrees) ffdDegrees.hidden = !qualifierState.fixed_flexion_deformity;
  const ffdInput = $('#ffdDegrees'); if (ffdInput && document.activeElement !== ffdInput) ffdInput.value = qualifierState.fixed_flexion_deformity_deg ?? '';
  const clue = $('#stiffnessReviewClue'); if (clue) clue.hidden = !(qHasStiffness() && qualifierState.morning_stiffness_duration === 'gt_30');
}

function toggleArray(key, value) {
  const list = qualifierState[key];
  qualifierState[key] = list.includes(value) ? list.filter(v => v !== value) : [...list, value];
}

function appendExamGroup() {
  const advanced = $('#advanced'); if (!advanced || $('#examQualifierGroup')) return;
  const group = make('details',{class:'advanced-group',id:'examQualifierGroup'},[
    make('summary',{text:'Εξέταση'}),
    make('p',{class:'footnote',text:'Αντικειμενικά ευρήματα μόνο όταν έχουν εξεταστεί. Η δυσκαμψία ως σύμπτωμα παραμένει ξεχωριστή.'}),
    make('div',{class:'chips'},[
      smallSelection('extension_lag','findings'), smallSelection('effusion','findings'),
      btn('Παθητικό έλλειμμα έκτασης',{'data-q-ffd':'','aria-pressed':qualifierState.fixed_flexion_deformity}),
    ]),
    make('label',{class:'field',id:'ffdDegreesWrap'},['Έλλειμμα έκτασης σε μοίρες (αν μετρήθηκε)',make('input',{id:'ffdDegrees',type:'number',min:'1',max:'60',step:'1',inputmode:'numeric','aria-label':'Παθητικό έλλειμμα έκτασης σε μοίρες'})]),
    make('div',{class:'qualifier-block'},[
      make('p',{class:'subtle small',text:'Εστιακή ευαισθησία'}),
      make('div',{class:'chips'},Object.entries(QLABELS.tenderness).map(([id,name])=>btn(name,{'data-q-tenderness':id,'aria-pressed':qualifierState.focal_tenderness_locations.includes(id)})))
    ])
  ]);
  advanced.prepend(group);
  $('#ffdDegrees')?.addEventListener('input',event => {
    const raw = event.target.value;
    if (raw === '') {
      event.target.setCustomValidity(''); event.target.removeAttribute('aria-invalid');
      qualifierState.fixed_flexion_deformity_deg = null;
    } else {
      const value=Number.parseInt(raw,10);
      if (!Number.isInteger(value) || value < 1 || value > 60) {
        event.target.setCustomValidity('Καταχώρισε θετικό έλλειμμα 1–60° ή άφησέ το κενό.');
        event.target.setAttribute('aria-invalid','true');
        return;
      }
      event.target.setCustomValidity(''); event.target.removeAttribute('aria-invalid');
      qualifierState.fixed_flexion_deformity_deg = value;
    }
    if (!qualifierState.fixed_flexion_deformity) qualifierState.fixed_flexion_deformity = true;
    changed('fixed_flexion_deformity');
  });
}

function hideDuplicatedFindingControls() {
  for (const id of ['objective_weakness','quadriceps_weakness','extension_lag','effusion','tenderness']) {
    const node = $(`#advanced [data-select="${id}"]`); if (node) node.hidden = true;
  }
}

function appendClinicalReviewClues() {
  const clues = response?.clinical_review_clues || []; if (!clues.length) return;
  const body = $('#sheetBody');
  for (const clue of clues) {
    body.append(make('div',{class:'review-clue'},[
      make('strong',{text:clue.label}), make('p',{text:clue.detail}),
      make('p',{class:'scope-caption',text:clue.source_label+' · Κλινική ανασκόπηση: '+clue.reviewed_on}),
      make('a',{href:clue.source_url,target:'_blank',rel:'noopener noreferrer',text:'Πηγή ↗'})
    ]));
  }
}

function parentCleanup(button) {
  if (button.dataset.finding === 'pain' && !qHasPain()) qResetPain();
  if (button.dataset.phenotype === 'stiffness_symptom' && !qHasStiffness()) qResetStiffness();
  if (button.dataset.phenotype === 'weakness_symptom_or_context' && !qHasWeakness()) qResetWeakness();
  renderQualifierUI();
}

// Flatten suggestion chrome while preserving explicit add, dismissal, evidence
// access and stale-candidate handling.
suggestionCard = function(candidate, inSheet=false) {
  const item=candidate.item_id;
  const add=btn('Προσθήκη',{class:'suggestion-add','data-add':item,'aria-label':'Προσθήκη: '+label(item),'data-key':'add:'+item}); add.disabled=!fresh();
  const controls=[make('strong',{class:'suggestion-title',text:label(item)}),add];
  const existing=$$('[data-evidence]').find(n=>n.dataset.evidence===item && n.getClientRects().length && !n.closest('#suggestions') && !n.closest('#sheet'));
  if(inSheet || !existing) controls.push(evidenceButton(item));
  controls.push(btn('×',{class:'suggestion-dismiss','data-dismiss-suggestion':item,'aria-label':'Παράλειψη πρότασης: '+label(item)}));
  return make('div',{class:'suggestion suggestion-compact'},[
    make('div',{class:'suggestion-top'},[make('span',{class:'suggestion-prefix',text:'Πρόταση'}),...controls]),
    make('p',{class:'suggestion-caption',text:candidate.source_caption}),
  ]);
};
renderSuggestions = function() {
  const box=$('#suggestions'); const candidates=response?.suggestions||[];
  if(!candidates.length){box.replaceChildren();return;}
  box.replaceChildren(suggestionCard(candidates[0]));
  if(candidates.length>1) box.append(btn('Προτάσεις · '+candidates.length,{class:'suggestions-more','data-all-suggestions':''}));
};

function openJurisdictionNotice() { notice(ACTIVE_JURISDICTION.note); }

installAmendmentUI();

document.addEventListener('click', event => {
  const b = event.target.closest('button'); if (!b || !state) return;
  if (b.hasAttribute('data-jurisdiction-info')) { openJurisdictionNotice(); return; }
  if (b.dataset.finding === 'pain' || ['stiffness_symptom','weakness_symptom_or_context'].includes(b.dataset.phenotype)) {
    parentCleanup(b); return;
  }
  if (b.dataset.smartOpen) {
    const root = b.closest('[data-smart-root]'); const panel = root?.querySelector('[data-smart-panel]');
    if (panel) {
      const willOpen=panel.hidden;
      if(willOpen) $$('[data-smart-root]').forEach(other=>{if(other!==root){const p=other.querySelector('[data-smart-panel]');const s=other.querySelector('[data-smart-summary]');if(p)p.hidden=true;if(s)s.setAttribute('aria-expanded','false');}});
      panel.hidden = !willOpen; b.setAttribute('aria-expanded', String(willOpen));
    }
    return;
  }
  if (b.dataset.qPain) {
    const value=b.dataset.qPain;
    if (value==='diffuse') qualifierState.pain_locations = qualifierState.pain_locations.includes('diffuse') ? [] : ['diffuse'];
    else { qualifierState.pain_locations = qualifierState.pain_locations.filter(v=>v!=='diffuse'); toggleArray('pain_locations',value); }
    changed('pain_location'); return;
  }
  if (b.dataset.qStiffness) {
    toggleArray('stiffness_patterns',b.dataset.qStiffness);
    if (!qualifierState.stiffness_patterns.includes('morning')) qualifierState.morning_stiffness_duration=null;
    changed('stiffness_pattern'); return;
  }
  if (b.dataset.qDuration) { qualifierState.morning_stiffness_duration=b.dataset.qDuration; changed('stiffness_duration'); return; }
  if (b.dataset.qWeakness) {
    qualifierState.weakness_detail = qualifierState.weakness_detail === b.dataset.qWeakness ? null : b.dataset.qWeakness;
    changed('weakness_detail'); return;
  }
  if (b.hasAttribute('data-q-atrophy')) {
    qualifierState.visible_atrophy=!qualifierState.visible_atrophy;
    if (!qualifierState.visible_atrophy) qualifierState.atrophy_location=null;
    changed('visible_atrophy'); return;
  }
  if (b.dataset.qAtrophyLocation) { qualifierState.atrophy_location=b.dataset.qAtrophyLocation; qualifierState.visible_atrophy=true; changed('atrophy_location'); return; }
  if (b.hasAttribute('data-q-ffd')) {
    qualifierState.fixed_flexion_deformity=!qualifierState.fixed_flexion_deformity;
    if (!qualifierState.fixed_flexion_deformity) qualifierState.fixed_flexion_deformity_deg=null;
    changed('fixed_flexion_deformity'); return;
  }
  if (b.dataset.qTenderness) { toggleArray('focal_tenderness_locations',b.dataset.qTenderness); changed('focal_tenderness'); return; }
});

window.addEventListener('pagehide',()=>{qualifierState=qualifierDefaults();});

// Usability refinement v2: discoverable secondary suggestions, direct referral
// editing and ephemeral doctor-facing favorites. None of these controls mutate
// clinical meaning unless the clinician explicitly activates the underlying
// selection control.
const V2_FAVORITE_CATEGORIES = new Set(['findings','functional_impairments','goals','rehab_directions','adjunct_options']);
let v2Favorites = [];

function v2FavoriteKey(category,item) { return category+':'+item; }
function v2IsFavorite(category,item) { return v2Favorites.includes(v2FavoriteKey(category,item)); }
function v2FavoriteToggle(category,item) {
  const active=v2IsFavorite(category,item);
  return btn(active?'★':'☆',{class:'favorite-toggle','data-favorite-v2':'','data-favorite-category':category,'data-favorite-item':item,
    'aria-pressed':String(active),'aria-label':(active?'Αφαίρεση από τα Συχνά: ':'Προσθήκη στα Συχνά: ')+label(item)});
}
function v2SyncFavoriteToggles() {
  $$('[data-favorite-v2]').forEach(b=>{
    const active=v2IsFavorite(b.dataset.favoriteCategory,b.dataset.favoriteItem);
    b.textContent=active?'★':'☆'; b.setAttribute('aria-pressed',String(active));
    b.setAttribute('aria-label',(active?'Αφαίρεση από τα Συχνά: ':'Προσθήκη στα Συχνά: ')+label(b.dataset.favoriteItem));
  });
}
function v2RenderFavorites() {
  const host=$('#advancedFavoritesV2'); if(!host || !state) return;
  const items=v2Favorites.map(key=>{
    const split=key.indexOf(':'); if(split<1)return null;
    const category=key.slice(0,split),item=key.slice(split+1);
    return V2_FAVORITE_CATEGORIES.has(category)&&Array.isArray(state[category])?{category,item}:null;
  }).filter(Boolean);
  if(!items.length){host.hidden=true;host.replaceChildren();return;}
  host.hidden=false;
  host.replaceChildren(
    make('div',{class:'favorites-heading'},[
      make('strong',{text:'★ Συχνά'}),
      make('span',{class:'small subtle',text:'Συντομεύσεις · δεν επιλέγουν τίποτα από μόνες τους'}),
    ]),
    make('div',{class:'favorite-items'},items.map(({category,item})=>make('span',{class:'favorite-item'},[
      btn(label(item),{'data-select':item,'data-category':category,'aria-pressed':String(state[category].includes(item))}),
      v2FavoriteToggle(category,item),
    ])))
  );
}
function v2InstallFavoriteControls() {
  const advanced=$('#advanced'); if(!advanced)return;
  let host=$('#advancedFavoritesV2');
  if(!host){host=make('div',{id:'advancedFavoritesV2',class:'advanced-favorites',hidden:''});advanced.prepend(host);}
  const controls=[...advanced.querySelectorAll('[data-select][data-category]')];
  for(const select of controls){
    const category=select.dataset.category,item=select.dataset.select;
    if(!V2_FAVORITE_CATEGORIES.has(category)||select.closest('#advancedFavoritesV2')||select.dataset.favoriteDecorated==='1'||select.hidden)continue;
    select.dataset.favoriteDecorated='1';
    const star=v2FavoriteToggle(category,item);
    const rowMain=select.closest('.row-main');
    if(rowMain){rowMain.append(star);continue;}
    const wrapper=make('span',{class:'advanced-choice'});
    select.replaceWith(wrapper); wrapper.append(select,star);
  }
  v2RenderFavorites(); v2SyncFavoriteToggles();
}
function v2InstallDirectEdit() {
  const heading=$('.preview-heading'); const menu=heading?.querySelector('[data-menu]');
  if(!heading||!menu||$('#directEditV2'))return;
  const actions=make('div',{class:'preview-quick-actions'});
  menu.replaceWith(actions);
  const edit=btn('✎ Επεξεργασία',{id:'directEditV2',class:'quiet direct-edit','data-edit':'','aria-label':'Επεξεργασία κειμένου παραπομπής'});
  edit.disabled=true; actions.append(edit,menu);
}
function v2SyncDirectEdit() {
  const enabled=!!(fresh()&&response?.gate?.allowed);
  $$('[data-edit].direct-edit').forEach(b=>b.disabled=!enabled);
}

// Preserve one compact primary suggestion, but make the remaining candidates
// visually obvious without turning the routine surface into a recommendation feed.
renderSuggestions = function() {
  const box=$('#suggestions'); const candidates=response?.suggestions||[];
  if(!candidates.length){box.replaceChildren();return;}
  box.replaceChildren(suggestionCard(candidates[0]));
  const additional=candidates.slice(1);
  if(additional.length){
    box.append(make('button',{type:'button',class:'suggestions-more-panel','data-all-suggestions':'',
      'aria-label':'Άλλες '+additional.length+' προτάσεις. Άνοιγμα λίστας.'},[
      make('span',{class:'suggestions-more-heading',text:'Άλλες '+additional.length+' προτάσεις ›'}),
      make('span',{class:'suggestions-more-titles',text:additional.map(c=>label(c.item_id)).join(' · ')}),
    ]));
  }
};

const v2BaseBuildAdvanced=buildAdvanced;
buildAdvanced=function(){v2BaseBuildAdvanced();v2InstallFavoriteControls();};
const v2BasePaint=paint;
paint=function(){v2BasePaint();v2RenderFavorites();v2SyncFavoriteToggles();v2SyncDirectEdit();};
const v2BasePaintStatus=paintStatus;
paintStatus=function(){v2BasePaintStatus();v2SyncDirectEdit();};
const v2BaseNewDraft=newDraft;
newDraft=function(){v2Favorites=[];v2BaseNewDraft();queueMicrotask(()=>{v2RenderFavorites();v2SyncFavoriteToggles();v2SyncDirectEdit();});};
const v2BaseOpenSheet=openSheet;
openSheet=function(type,item=null,back=null){
  v2BaseOpenSheet(type,item,back);
  if(type==='preview'&&$('#sheet')?.open){
    const body=$('#sheetBody');
    if(body&&!body.querySelector('[data-edit].direct-edit')){
      const edit=btn('✎ Επεξεργασία',{class:'quiet direct-edit','data-edit':'','aria-label':'Επεξεργασία κειμένου παραπομπής'});
      edit.disabled=!(fresh()&&response?.gate?.allowed);
      body.prepend(make('div',{class:'sheet-preview-actions'},[edit]));
    }
  }
};

v2InstallDirectEdit();
document.addEventListener('click',event=>{
  const b=event.target.closest('[data-favorite-v2]'); if(!b||!state)return;
  const key=v2FavoriteKey(b.dataset.favoriteCategory,b.dataset.favoriteItem);
  v2Favorites=v2Favorites.includes(key)?v2Favorites.filter(v=>v!==key):[...v2Favorites,key];
  v2RenderFavorites();v2SyncFavoriteToggles();
});
window.addEventListener('pagehide',()=>{v2Favorites=[];});
