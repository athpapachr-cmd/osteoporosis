'use strict';
// Step 6A product-owner refinement. This classic script intentionally shares the
// global lexical environment with app.js; it does not create persistence or a
// second clinical engine.

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
  weakness: {objective: 'Αντικειμενική', quadriceps: 'Τετρακέφαλος'},
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

const qBaseNewDraft = newDraft;
newDraft = function() {
  qualifierState = qualifierDefaults();
  qBaseNewDraft();
  queueMicrotask(renderQualifierUI);
};
const qBasePaint = paint;
paint = function() { qBasePaint(); renderQualifierUI(); };
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
const qBaseOpenSheet = openSheet;
openSheet = function(type, item = null, back = null) {
  qBaseOpenSheet(type, item, back);
  if (type === 'review') appendClinicalReviewClues();
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
  if (qualifierState.weakness_detail) values.push(QLABELS.weakness[qualifierState.weakness_detail]);
  if (qualifierState.visible_atrophy) values.push('Ατροφία' + (qualifierState.atrophy_location ? ' · ' + QLABELS.atrophy[qualifierState.atrophy_location] : ''));
  return values.length ? qJoin(values) : 'Προσδιορισμός ›';
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
    if (!active) root.querySelector('[data-smart-panel]').hidden = true;
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
      btn('Fixed flexion deformity',{'data-q-ffd':'','aria-pressed':qualifierState.fixed_flexion_deformity}),
    ]),
    make('label',{class:'field',id:'ffdDegreesWrap'},['Έλλειμμα έκτασης σε μοίρες',make('input',{id:'ffdDegrees',type:'number',min:'0',max:'60',step:'1',inputmode:'numeric','aria-label':'Fixed flexion deformity σε μοίρες'})]),
    make('div',{class:'qualifier-block'},[
      make('p',{class:'subtle small',text:'Εστιακή ευαισθησία'}),
      make('div',{class:'chips'},Object.entries(QLABELS.tenderness).map(([id,name])=>btn(name,{'data-q-tenderness':id,'aria-pressed':qualifierState.focal_tenderness_locations.includes(id)})))
    ])
  ]);
  advanced.prepend(group);
  $('#ffdDegrees')?.addEventListener('input',event => {
    const raw = event.target.value;
    qualifierState.fixed_flexion_deformity_deg = raw === '' ? null : Math.max(0, Math.min(60, Number.parseInt(raw,10) || 0));
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

document.addEventListener('click', event => {
  const b = event.target.closest('button'); if (!b || !state) return;
  if (b.dataset.finding === 'pain' || ['stiffness_symptom','weakness_symptom_or_context'].includes(b.dataset.phenotype)) {
    parentCleanup(b); return;
  }
  if (b.dataset.smartOpen) {
    const root = b.closest('[data-smart-root]'); const panel = root?.querySelector('[data-smart-panel]');
    if (panel) { panel.hidden = !panel.hidden; b.setAttribute('aria-expanded', String(!panel.hidden)); }
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
