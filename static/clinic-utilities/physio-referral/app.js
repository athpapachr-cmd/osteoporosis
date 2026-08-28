(() => {
  'use strict';

  const API_BASE = '/clinical/clinic-utilities/physio-referral/api';
  const state = {
    contract: null,
    mode: 'short',
    extraContext: {},
    restrictions: [],
    acknowledgedRules: new Set(),
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    contractStatus: $('contractStatus'),
    profile: $('profileSelect'), route: $('routeSelect'), wording: $('wordingSelect'), laterality: $('lateralitySelect'),
    wordingWrap: $('wordingWrap'), lateralityWrap: $('lateralityWrap'), subtypeWrap: $('subtypeWrap'), subtype: $('subtypeSelect'), assertionWrap: $('assertionWrap'), assertion: $('assertionSelect'),
    contextCard: $('contextCard'), contextFields: $('contextFields'),
    findingsCard: $('findingsCard'), findings: $('findingsOptions'), functions: $('functionOptions'), findingsBlock: $('findingsBlock'), functionBlock: $('functionBlock'),
    rehabCard: $('rehabCard'), goals: $('goalOptions'), rehab: $('rehabOptions'), adjuncts: $('adjunctOptions'), goalsBlock: $('goalsBlock'), rehabBlock: $('rehabBlock'), adjunctBlock: $('adjunctBlock'),
    notesCard: $('notesCard'), sessions: $('sessionsInput'), freeText: $('clinicianFreeText'),
    advancedCard: $('advancedCard'), restrictionId: $('restrictionIdSelect'), restrictionValue: $('restrictionValue'), restrictionSource: $('restrictionSource'), restrictionList: $('restrictionList'), addRestrictionBtn: $('addRestrictionBtn'),
    safetyFlags: $('safetyFlagOptions'), disposition: $('dispositionSelect'), dispositionWrap: $('dispositionWrap'),
    extraContextKey: $('extraContextKey'), extraContextValue: $('extraContextValue'), extraContextList: $('extraContextList'), addContextBtn: $('addContextBtn'),
    validateBtn: $('validateBtn'), generateBtn: $('generateBtn'), validationPanel: $('validationPanel'), validationSummary: $('validationSummary'),
    output: $('outputText'), copyBtn: $('copyBtn'), printBtn: $('printBtn'), clearBtn: $('clearBtn'),
  };

  const greekStatic = {
    presentation: 'Συμπτωματολογία / κλινική παρουσίαση',
    formal_diagnosis: 'Ρητή κλινική διάγνωση',
    established_structural_diagnosis: 'Τεκμηριωμένη δομική διάγνωση',
    postoperative: 'Μετεγχειρητική αποκατάσταση',
    shared_structural: 'Δομική αποκατάσταση',
    left: 'Αριστερά', right: 'Δεξιά', bilateral: 'Αμφοτερόπλευρα', midline: 'Μέση γραμμή', not_applicable: 'Δεν εφαρμόζεται', not_stated: 'Δεν δηλώνεται',
    normal: 'Φυσιολογικό', abnormal: 'Παθολογικό', not_assessed: 'Δεν αξιολογήθηκε',
  };

  const errorLabels = {
    required_field_missing: 'Λείπει απαραίτητο πεδίο για τη συγκεκριμένη επιλογή.',
    formal_diagnosis_assertion_required: 'Χρειάζεται να επιβεβαιώσεις ότι η διάγνωση έχει τεθεί κλινικά.',
    established_structural_diagnosis_source_required: 'Χρειάζεται η πηγή της τεκμηριωμένης διάγνωσης.',
    established_nonoperative_management_context_required: 'Χρειάζεται να δηλωθεί ότι έχει επιλεγεί συντηρητική / μη χειρουργική αντιμετώπιση.',
    subtype_required: 'Χρειάζεται να επιλέξεις υποτύπο.',
    subtype_required_for_asserted_nonarthritic_intraarticular_diagnosis: 'Χρειάζεται να επιλέξεις συγκεκριμένο υποτύπο.',
    postoperative_protocol_source_required: 'Χρειάζεται η πηγή του διαθέσιμου μετεγχειρητικού πρωτοκόλλου.',
    postoperative_protocol_status_unresolved: 'Η κατάσταση του μετεγχειρητικού πρωτοκόλλου πρέπει να διευκρινιστεί.',
    lower_limb_weight_bearing_required_missing: 'Χρειάζεται να δηλωθεί η επιτρεπόμενη φόρτιση του κάτω άκρου.',
    upper_limb_use_status_required_missing: 'Χρειάζεται να δηλωθεί η επιτρεπόμενη χρήση του άνω άκρου.',
    axial_fracture_loading_or_rom_status_required_missing: 'Χρειάζεται να δηλωθούν οι περιορισμοί κίνησης/φόρτισης.',
    frailty_must_be_clinician_established_for_formal_wording: 'Η ευπάθεια (frailty) πρέπει να έχει τεθεί ρητά από τον κλινικό.',
    invalid_route_or_subtype: 'Η επιλογή δεν είναι έγκυρη για το συγκεκριμένο pathway.',
  };

  const fieldLabels = {
    'primary_problem.context.established_diagnosis_source': 'Πηγή διάγνωσης',
    'primary_problem.context.management_context': 'Πλάνο αντιμετώπισης',
    'primary_problem.context.procedure': 'Επέμβαση',
    'primary_problem.context.procedure_date_or_phase': 'Ημερομηνία / φάση επέμβασης',
    'primary_problem.context.protocol_status': 'Μετεγχειρητικό πρωτόκολλο',
    'primary_problem.context.restrictions_review_status': 'Περιορισμοί μετά την επέμβαση',
    'primary_problem.context.protocol_source': 'Πηγή πρωτοκόλλου',
    'primary_problem.context.fracture_site': 'Θέση κατάγματος',
    'primary_problem.context.fracture_phase': 'Φάση κατάγματος',
    'primary_problem.context.treatment': 'Αντιμετώπιση κατάγματος',
    'primary_problem.context.healing_stability_status': 'Πώρωση / σταθερότητα',
    'primary_problem.context.immobilization_status': 'Ακινητοποίηση',
    'primary_problem.context.weight_bearing_status': 'Επιτρεπόμενη φόρτιση',
    'primary_problem.context.upper_limb_use_status': 'Επιτρεπόμενη χρήση άνω άκρου',
    'primary_problem.context.rom_status': 'Επιτρεπόμενο εύρος κίνησης',
    'primary_problem.context.loading_strengthening_status': 'Επιτρεπόμενη ενδυνάμωση / φόρτιση',
    'primary_problem.context.muscle_group': 'Μυϊκή ομάδα',
    'primary_problem.context.injury_phase': 'Φάση κάκωσης',
    'primary_problem.context.injury_type': 'Τύπος κάκωσης',
    'primary_problem.context.functional_route_id': 'Λειτουργικό πρόβλημα',
  };

  const escapeHtml = (text) => String(text ?? '').replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

  function label(id, preferredSection = null) {
    if (!id) return '';
    const labels = state.contract?.display_labels || {};
    if (preferredSection && labels[preferredSection]?.[id]) return labels[preferredSection][id];
    for (const section of ['laterality','findings','functional_impairments','goals','rehab_directions','adjuncts','measurements','restrictions','context_values','route_detail_labels']) {
      if (labels[section]?.[id]) return labels[section][id];
    }
    return greekStatic[id] || String(id).replaceAll('_', ' ');
  }

  const option = (value, text = label(value)) => `<option value="${escapeHtml(value)}">${escapeHtml(text)}</option>`;

  async function api(path, init = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      credentials: 'same-origin',
      headers: {'Content-Type':'application/json', ...(init.headers || {})},
      ...init,
    });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try { const body = await response.json(); detail = body.detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    return response.json();
  }

  async function loadContract() {
    try {
      state.contract = await api('/contract');
      els.contractStatus.textContent = 'έτοιμο';
      els.contractStatus.classList.add('ok');
      populateProfiles();
      populateLaterality();
      resetDownstream();
    } catch (error) {
      els.contractStatus.textContent = 'σφάλμα φόρτωσης';
      showRequestError(error.message);
    }
  }

  function populateProfiles() {
    els.profile.innerHTML = '<option value="">Επίλεξε…</option>';
    Object.entries(state.contract.profiles || {}).forEach(([id, profile]) => {
      els.profile.insertAdjacentHTML('beforeend', option(id, profile.display || id));
    });
  }

  function populateLaterality() {
    els.laterality.innerHTML = (state.contract.laterality_values || []).map((id) => option(id)).join('');
    els.laterality.value = 'not_stated';
  }

  const visibilityRank = (value) => ({routine:0, visible_less_frequent:1, rare_advanced:2, shared_gateway:3, context_only:4}[value] ?? 9);

  function onProfileChange() {
    clearSelectionsAndState();
    const profileId = els.profile.value;
    const profile = state.contract?.profiles?.[profileId];
    els.route.disabled = !profile;
    els.route.innerHTML = profile ? '<option value="">Επίλεξε…</option>' : '<option value="">Επίλεξε περιοχή πρώτα</option>';
    if (profile) {
      Object.entries(profile.routes)
        .sort((a,b) => visibilityRank(a[1].visibility) - visibilityRank(b[1].visibility))
        .forEach(([id, route]) => {
          const suffix = route.visibility === 'rare_advanced' ? ' · σπάνιο/advanced' : route.visibility === 'visible_less_frequent' ? ' · λιγότερο συχνό' : '';
          els.route.insertAdjacentHTML('beforeend', option(id, `${route.display}${suffix}`));
        });
    }
    resetDownstream();
  }

  function onRouteChange() {
    clearSelectionsAndState({keepRoute:true});
    const route = currentRoute();
    if (!route) {
      resetDownstream();
      return;
    }
    configureWording(route);
    configureLaterality();
    renderSubtype();
    renderAssertion();
    renderRelevantOptions();
    renderContextFields();
    renderAdvancedControls();
    els.findingsCard.hidden = false;
    els.rehabCard.hidden = false;
    els.notesCard.hidden = false;
    els.advancedCard.hidden = false;
    els.validationSummary.textContent = 'Μπορείς ήδη να δημιουργήσεις βασικό παραπεμπτικό ή να προσθέσεις προαιρετικές λεπτομέρειες.';
  }

  function currentRoute() {
    return state.contract?.profiles?.[els.profile.value]?.routes?.[els.route.value] || null;
  }

  function configureWording(route) {
    const modes = route.wording_modes || [];
    els.wording.innerHTML = modes.map((id) => option(id)).join('');
    const preferred = ['presentation','established_structural_diagnosis','postoperative','shared_structural'].find((id) => modes.includes(id));
    els.wording.value = preferred || (modes.length === 1 ? modes[0] : '');
    els.wordingWrap.hidden = modes.length <= 1 && els.wording.value !== 'formal_diagnosis';
  }

  function configureLaterality() {
    els.lateralityWrap.hidden = false;
    els.laterality.value = 'not_stated';
  }

  function renderSubtype() {
    const subtypes = state.contract.subtypes?.[els.route.value] || [];
    els.subtypeWrap.hidden = subtypes.length === 0;
    els.subtype.innerHTML = '<option value="">Επίλεξε…</option>' + subtypes.map((id) => option(id, label(id, 'route_detail_labels'))).join('');
    if (subtypes.length === 1) els.subtype.value = subtypes[0];
  }

  function renderAssertion() {
    const formal = els.wording.value === 'formal_diagnosis';
    els.assertionWrap.hidden = !formal;
    if (!formal) els.assertion.value = 'not_stated';
  }

  function onWordingChange() {
    renderAssertion();
    renderContextFields();
  }

  function scopeForCurrentProfile() {
    return state.contract?.ui_relevance_scope?.profiles?.[els.profile.value] || {};
  }

  function prioritized(ids, preferred = []) {
    const set = new Set(ids || []);
    return [...preferred.filter((id) => set.has(id)), ...(ids || []).filter((id) => !preferred.includes(id))];
  }

  function renderRelevantOptions() {
    const scope = scopeForCurrentProfile();
    const adjustment = state.contract?.ui_relevance_scope?.route_adjustments?.[els.profile.value]?.[els.route.value] || {};
    renderChecks(els.findings, scope.findings || [], 'finding', 'findings');
    renderChecks(els.functions, scope.functional_impairments || [], 'functional', 'functional_impairments');
    renderChecks(els.goals, scope.goals || [], 'goal', 'goals');
    renderChecks(els.rehab, scope.rehab_directions || [], 'rehab', 'rehab_directions');
    renderChecks(els.adjuncts, prioritized(scope.adjuncts || [], adjustment.prioritize_adjuncts || []), 'adjunct', 'adjuncts');
    els.findingsBlock.hidden = !(scope.findings || []).length;
    els.functionBlock.hidden = !(scope.functional_impairments || []).length;
    els.goalsBlock.hidden = !(scope.goals || []).length;
    els.rehabBlock.hidden = !(scope.rehab_directions || []).length;
    els.adjunctBlock.hidden = !(scope.adjuncts || []).length;
  }

  function renderChecks(container, ids, group, section) {
    container.innerHTML = (ids || []).map((id) => `
      <label class="check-item"><input type="checkbox" data-group="${group}" value="${escapeHtml(id)}" /> <span>${escapeHtml(label(id, section))}</span></label>
    `).join('') || '<span class="empty-note">Δεν υπάρχουν σχετικές επιλογές.</span>';
  }

  function routeOverride() {
    return state.contract?.ui_route_requirements?.route_overrides?.[els.route.value] || {};
  }

  function hasPostoperativePolicy() {
    const override = routeOverride();
    return els.wording.value === 'postoperative' || override.apply_policy === 'postoperative_context' || els.route.value.startsWith('postoperative_');
  }

  function directRequiredContextKeys() {
    const paths = routeOverride().require || [];
    return paths.filter((path) => path.startsWith('primary_problem.context.')).map((path) => path.split('.').pop());
  }

  function conditionalRequiredContextKeys() {
    const conditional = routeOverride().conditional_requirements || [];
    const keys = [];
    conditional.forEach((item) => {
      const mode = item?.when?.eq?.path === 'primary_problem.wording_mode' ? item.when.eq.value : null;
      if (mode && mode !== els.wording.value) return;
      (item.require || []).forEach((path) => {
        if (path.startsWith('primary_problem.context.')) keys.push(path.split('.').pop());
      });
    });
    return keys;
  }

  function renderContextFields() {
    const profile = els.profile.value;
    const route = els.route.value;
    const fields = [];
    const required = new Set([...directRequiredContextKeys(), ...conditionalRequiredContextKeys()]);

    if (required.has('established_diagnosis_source')) {
      fields.push(selectField('established_diagnosis_source', 'Πηγή τεκμηριωμένης διάγνωσης', ['clinician_entered','imaging_confirmed','specialist_documented','prior_documented_diagnosis','other_documented'], true));
    }
    if (required.has('management_context')) {
      fields.push(selectField('management_context', 'Πλάνο αντιμετώπισης', ['nonoperative_confirmed','conservative_rehabilitation'], true));
    }

    if (hasPostoperativePolicy()) {
      const postop = state.contract.ui_route_requirements?.postoperative_context || {};
      const values = state.contract.context_value_sets?.postoperative_common || {};
      fields.push(textField('procedure', 'Επέμβαση', '', true));
      fields.push(textField('procedure_date_or_phase', 'Ημερομηνία ή φάση επέμβασης', '', true));
      fields.push(selectField('protocol_status', 'Υπάρχει μετεγχειρητικό πρωτόκολλο;', values.protocol_status || [], true));
      fields.push(selectField('restrictions_review_status', 'Έχουν ελεγχθεί οι περιορισμοί;', values.restrictions_review_status || [], true));
      fields.push(textField('protocol_source', 'Πηγή πρωτοκόλλου (μόνο αν υπάρχει)', '', false));
    }

    if (profile === 'shared_fracture') {
      const req = state.contract.ui_route_requirements?.shared_context_requirements?.shared_fracture || {};
      const v = state.contract.context_value_sets.shared_fracture || {};
      const sites = [...new Set(Object.values(req.site_groups || {}).flat())];
      fields.push(selectField('fracture_site', 'Θέση κατάγματος', sites, true, 'route_detail_labels'));
      fields.push(selectField('fracture_phase', 'Φάση κατάγματος', v.fracture_phase || [], true));
      fields.push(selectField('treatment', 'Αντιμετώπιση', v.treatment || [], true));
      fields.push(selectField('healing_stability_status', 'Πώρωση / σταθερότητα', v.healing_stability_status || [], true));
      fields.push(selectField('immobilization_status', 'Ακινητοποίηση', v.immobilization_status || [], true));
      fields.push(selectField('rom_status', 'Επιτρεπόμενο εύρος κίνησης', v.rom_status || [], true));
      fields.push(selectField('loading_strengthening_status', 'Επιτρεπόμενη ενδυνάμωση / φόρτιση', v.loading_strengthening_status || [], true));
      fields.push('<div id="fractureSiteDependent"></div>');
    }

    if (profile === 'shared_muscle_myotendinous') {
      const v = state.contract.context_value_sets.shared_muscle_myotendinous || {};
      fields.push(textField('muscle_group', 'Μυϊκή ομάδα', 'π.χ. hamstring_muscle_injury', true));
      fields.push(selectField('injury_phase', 'Φάση κάκωσης', v.injury_phase || [], true));
      fields.push(selectField('injury_type', 'Τύπος κάκωσης', v.injury_type || [], true));
      fields.push(selectField('injury_location_optional', 'Εντόπιση (προαιρετικό)', v.injury_location || [], false));
      fields.push(selectField('management_context', 'Πλάνο αντιμετώπισης', v.management_context || [], true));
    }

    if (profile === 'shared_deconditioning_balance_gait') {
      fields.push(selectField('functional_route_id', 'Κύριο λειτουργικό πρόβλημα', ['generalized_deconditioning_functional_decline','frailty_associated_functional_decline','balance_impairment_context','gait_mobility_impairment_context','post_illness_or_post_hospital_deconditioning_context'], true, 'route_detail_labels'));
      if (els.wording.value === 'formal_diagnosis') {
        fields.push(selectField('frailty_established', 'Έχει τεθεί ρητά frailty;', state.contract.context_value_sets.shared_deconditioning_balance_gait?.frailty_established || [], true));
      }
    }

    if (['neck_pain_with_radiating_upper_limb_symptoms','low_back_pain_with_radiating_leg_symptoms'].includes(route)) {
      fields.push(`<div class="context-group"><h3>Νευρολογικός έλεγχος</h3><div class="grid three">${selectField('neurological_screen.motor','Κινητικότητα',['not_assessed','normal','abnormal'],false)}${selectField('neurological_screen.sensory','Αισθητικότητα',['not_assessed','normal','abnormal'],false)}${selectField('neurological_screen.reflexes','Αντανακλαστικά',['not_assessed','normal','abnormal'],false)}</div></div>`);
    }

    els.contextFields.innerHTML = fields.join('');
    els.contextCard.hidden = fields.length === 0;
    renderFractureSiteDependent();
  }

  function renderFractureSiteDependent() {
    const holder = $('fractureSiteDependent');
    if (!holder) return;
    const siteNode = document.querySelector('[data-context-key="fracture_site"]');
    const site = siteNode?.value || '';
    const req = state.contract.ui_route_requirements?.shared_context_requirements?.shared_fracture || {};
    const groups = req.site_groups || {};
    const v = state.contract.context_value_sets.shared_fracture || {};
    if ((groups.upper_limb || []).includes(site)) {
      holder.innerHTML = selectField('upper_limb_use_status', 'Επιτρεπόμενη χρήση άνω άκρου', v.upper_limb_use_status || [], true);
    } else if ((groups.lower_limb_or_pelvis || []).includes(site)) {
      holder.innerHTML = selectField('weight_bearing_status', 'Επιτρεπόμενη φόρτιση κάτω άκρου', v.weight_bearing_status || [], true);
    } else {
      holder.innerHTML = '';
    }
  }

  function textField(key, title, placeholder, required = false) {
    return `<label>${escapeHtml(title)}${required ? ' <span class="required-mark">*</span>' : ''}<input type="text" data-context-key="${escapeHtml(key)}" placeholder="${escapeHtml(placeholder)}" /></label>`;
  }

  function selectField(key, title, values, required = false, section = 'context_values') {
    return `<label>${escapeHtml(title)}${required ? ' <span class="required-mark">*</span>' : ''}<select data-context-key="${escapeHtml(key)}"><option value="">Επίλεξε…</option>${(values || []).map((id) => option(id, label(id, section))).join('')}</select></label>`;
  }

  function renderAdvancedControls() {
    renderChecks(els.safetyFlags, state.contract.safety_input_flags || [], 'safety', null);
    els.restrictionId.innerHTML = '<option value="">Επίλεξε…</option>' + (state.contract.restrictions || []).map((id) => option(id, label(id, 'restrictions'))).join('');
    const structural = ['shared_fracture','shared_muscle_myotendinous'].includes(els.profile.value) || hasPostoperativePolicy() || els.wording.value === 'established_structural_diagnosis';
    els.dispositionWrap.hidden = !structural;
  }

  function addExtraContext() {
    const key = els.extraContextKey.value.trim();
    const value = els.extraContextValue.value.trim();
    if (!key || !value) return;
    state.extraContext[key] = value;
    els.extraContextKey.value = '';
    els.extraContextValue.value = '';
    renderExtraContext();
  }

  function renderExtraContext() {
    els.extraContextList.innerHTML = Object.entries(state.extraContext).map(([key,value]) => `<span class="chip">${escapeHtml(key)} = ${escapeHtml(value)} <button type="button" data-remove-context="${escapeHtml(key)}">×</button></span>`).join('');
  }

  function addRestriction() {
    const restriction_id = els.restrictionId.value;
    const state_or_value = els.restrictionValue.value.trim();
    const source = els.restrictionSource.value;
    if (!restriction_id || !state_or_value) return;
    state.restrictions.push({restriction_id, state_or_value, source, notes_optional:null});
    els.restrictionId.value = '';
    els.restrictionValue.value = '';
    renderRestrictions();
  }

  function renderRestrictions() {
    els.restrictionList.innerHTML = state.restrictions.map((item,index) => `<span class="chip">${escapeHtml(label(item.restriction_id, 'restrictions'))}: ${escapeHtml(item.state_or_value)} <button type="button" data-remove-restriction="${index}">×</button></span>`).join('');
  }

  function checkedValues(group) {
    return [...document.querySelectorAll(`input[data-group="${group}"]:checked`)].map((node) => node.value);
  }

  function collectContext() {
    const context = {...state.extraContext};
    document.querySelectorAll('[data-context-key]').forEach((node) => {
      const value = node.value;
      if (!value) return;
      const key = node.dataset.contextKey;
      if (key.startsWith('neurological_screen.')) {
        context.neurological_screen ||= {};
        context.neurological_screen[key.split('.')[1]] = value;
      } else {
        context[key] = value;
      }
    });
    return context;
  }

  function collectDraft() {
    return {
      contract_version: state.contract?.contract_version || 'cu1_referral_draft_v1',
      patient_context: {age_years_optional:null, skeletal_maturity_optional:null, sport_or_work_demand_optional:null, relevant_medical_context_ids:[], free_text_optional:null},
      body_region: els.profile.value || null,
      primary_problem: {
        problem_id: crypto.randomUUID ? crypto.randomUUID() : `draft-${Date.now()}`,
        profile_id: els.profile.value || null,
        route_id: els.route.value || null,
        wording_mode: els.wording.value || null,
        formal_assertion_state_optional: els.wording.value === 'formal_diagnosis' ? els.assertion.value : null,
        subtype_id_optional: els.subtypeWrap.hidden ? null : (els.subtype.value || null),
        laterality: els.laterality.value || 'not_stated',
        chronicity_or_phase_optional: null,
        context: collectContext(),
        shared_target_optional: null,
        source_route_optional: null,
      },
      secondary_problems: [],
      findings: checkedValues('finding').map((id) => ({finding_id:id, state_optional:null, laterality_optional:null, value_optional:null, unit_optional:null, free_text_optional:null})),
      functional_impairments: checkedValues('functional').map((id) => ({id, selected:true, notes_optional:null})),
      precautions: [],
      explicit_restrictions: state.restrictions,
      goals: checkedValues('goal').map((id) => ({id, selected:true, notes_optional:null})),
      rehab_directions: checkedValues('rehab').map((id) => ({id, selected:true, notes_optional:null})),
      adjunct_options: checkedValues('adjunct').map((id) => ({adjunct_id:id, selected:true, provenance:'clinician_selected'})),
      measurements: [],
      safety: {
        input_flags: checkedValues('safety'),
        acknowledged_rule_ids: [...state.acknowledgedRules],
        clinician_disposition: els.disposition.value || 'none_recorded',
      },
      sessions_optional: els.sessions.value ? Number(els.sessions.value) : null,
      clinician_free_text_optional: els.freeText.value.trim() || null,
    };
  }

  async function validateOnly() {
    if (!els.profile.value || !els.route.value) {
      showLocalMissingPrimary();
      return;
    }
    try {
      const result = await api('/validate', {method:'POST', body:JSON.stringify({draft:collectDraft()})});
      showValidation(result);
    } catch (error) {
      showRequestError(error.message);
    }
  }

  async function generate() {
    if (!els.profile.value || !els.route.value) {
      showLocalMissingPrimary();
      return;
    }
    try {
      const result = await api('/generate', {method:'POST', body:JSON.stringify({draft:collectDraft(), mode:state.mode})});
      showValidation(result);
      els.output.value = result.text || '';
      const hasText = Boolean(result.text);
      els.copyBtn.disabled = !hasText;
      els.printBtn.disabled = !hasText;
    } catch (error) {
      showRequestError(error.message);
      els.output.value = '';
      els.copyBtn.disabled = true;
      els.printBtn.disabled = true;
    }
  }

  function showLocalMissingPrimary() {
    els.validationPanel.hidden = false;
    els.validationPanel.className = 'validation-panel error';
    els.validationPanel.innerHTML = '<strong>Χρειάζεται μόνο να επιλέξεις περιοχή και πάθηση για να ξεκινήσεις.</strong>';
    els.validationSummary.textContent = 'Επίλεξε περιοχή και πάθηση.';
  }

  function friendlyError(item) {
    const path = item.metadata?.path;
    const base = errorLabels[item.error_id] || 'Χρειάζεται μία διόρθωση πριν δημιουργηθεί το παραπεμπτικό.';
    return path && fieldLabels[path] ? `${base} <strong>${escapeHtml(fieldLabels[path])}</strong>` : base;
  }

  function showValidation(result) {
    const errors = result.validation_errors || [];
    const safety = result.safety_results || [];
    els.validationPanel.hidden = false;
    els.validationPanel.className = 'validation-panel';
    clearInvalidMarkers();

    if (!errors.length && !safety.length && !result.formatter_blocked) {
      els.validationPanel.classList.add('ok');
      els.validationPanel.innerHTML = '<strong>Έτοιμο.</strong> Το παραπεμπτικό μπορεί να δημιουργηθεί.';
      els.validationSummary.textContent = 'Έτοιμο.';
      return;
    }

    if (errors.length || result.formatter_blocked) els.validationPanel.classList.add('error');
    const chunks = [];
    if (errors.length) {
      chunks.push(`<strong>Χρειάζεται να συμπληρωθεί:</strong><ul>${errors.map((item) => `<li>${friendlyError(item)}</li>`).join('')}</ul>`);
      markInvalidFromErrors(errors);
    }
    if (safety.length) {
      chunks.push(`<strong>Ασφάλεια / συνέπεια</strong><ul>${safety.map((item) => {
        const ack = item.acknowledgement_required ? `<label class="check-item"><input type="checkbox" data-ack-rule="${escapeHtml(item.rule_id)}" ${state.acknowledgedRules.has(item.rule_id) ? 'checked' : ''}/> Το έχω ελέγξει</label>` : '';
        return `<li>${escapeHtml(item.rule_id)}${item.formatter_blocked ? ' · απαιτεί ενέργεια' : ''}${ack}</li>`;
      }).join('')}</ul>`);
    }
    els.validationPanel.innerHTML = chunks.join('');
    els.validationSummary.textContent = result.formatter_blocked ? 'Συμπλήρωσε μόνο τα επισημασμένα απαραίτητα πεδία.' : 'Υπάρχουν μη blocking παρατηρήσεις.';
  }

  function clearInvalidMarkers() {
    document.querySelectorAll('.field-invalid').forEach((node) => node.classList.remove('field-invalid'));
  }

  function markInvalidFromErrors(errors) {
    errors.forEach((item) => {
      const path = item.metadata?.path || '';
      const key = path.startsWith('primary_problem.context.') ? path.replace('primary_problem.context.','') : null;
      if (key) document.querySelector(`[data-context-key="${CSS.escape(key)}"]`)?.closest('label')?.classList.add('field-invalid');
    });
  }

  function showRequestError(message) {
    els.validationPanel.hidden = false;
    els.validationPanel.className = 'validation-panel error';
    els.validationPanel.innerHTML = `<strong>Δεν ολοκληρώθηκε το αίτημα.</strong> ${escapeHtml(message)}`;
    els.validationSummary.textContent = 'Σφάλμα αιτήματος.';
  }

  async function copyOutput() {
    if (!els.output.value) return;
    await navigator.clipboard.writeText(els.output.value);
    const original = els.copyBtn.textContent;
    els.copyBtn.textContent = 'Αντιγράφηκε';
    setTimeout(() => { els.copyBtn.textContent = original; }, 1200);
  }

  function printOutput() {
    if (!els.output.value) return;
    const w = window.open('', '_blank', 'noopener,noreferrer,width=800,height=900');
    if (!w) return;
    w.document.write(`<!doctype html><html lang="el"><head><meta charset="utf-8"><title>Παραπεμπτικό Φυσιοθεραπείας</title><style>body{font-family:Arial,sans-serif;margin:36px;white-space:pre-wrap;line-height:1.5;color:#111}</style></head><body>${escapeHtml(els.output.value)}</body></html>`);
    w.document.close();
    w.focus();
    w.print();
  }

  function clearSelectionsAndState({keepRoute = false} = {}) {
    document.querySelectorAll('input[type="checkbox"]').forEach((node) => { node.checked = false; });
    state.extraContext = {};
    state.restrictions = [];
    state.acknowledgedRules.clear();
    els.disposition.value = 'none_recorded';
    els.sessions.value = '';
    els.freeText.value = '';
    if (!keepRoute) els.route.value = '';
    renderExtraContext();
    renderRestrictions();
    els.output.value = '';
    els.copyBtn.disabled = true;
    els.printBtn.disabled = true;
    els.validationPanel.hidden = true;
  }

  function resetDownstream() {
    els.wordingWrap.hidden = true;
    els.lateralityWrap.hidden = true;
    els.subtypeWrap.hidden = true;
    els.assertionWrap.hidden = true;
    els.contextCard.hidden = true;
    els.findingsCard.hidden = true;
    els.rehabCard.hidden = true;
    els.notesCard.hidden = true;
    els.advancedCard.hidden = true;
    els.wording.innerHTML = '';
    els.contextFields.innerHTML = '';
    els.findings.innerHTML = '';
    els.functions.innerHTML = '';
    els.goals.innerHTML = '';
    els.rehab.innerHTML = '';
    els.adjuncts.innerHTML = '';
    els.validationSummary.textContent = els.profile.value ? 'Επίλεξε πάθηση.' : 'Επίλεξε περιοχή και πάθηση.';
  }

  function clearDraft() {
    clearSelectionsAndState();
    els.profile.value = '';
    els.route.disabled = true;
    els.route.innerHTML = '<option value="">Επίλεξε περιοχή πρώτα</option>';
    els.laterality.value = 'not_stated';
    resetDownstream();
  }

  els.profile.addEventListener('change', onProfileChange);
  els.route.addEventListener('change', onRouteChange);
  els.wording.addEventListener('change', onWordingChange);
  els.contextFields.addEventListener('change', (event) => {
    if (event.target?.dataset?.contextKey === 'fracture_site') renderFractureSiteDependent();
  });
  els.addContextBtn.addEventListener('click', addExtraContext);
  els.extraContextList.addEventListener('click', (event) => {
    const button = event.target.closest('[data-remove-context]');
    if (!button) return;
    delete state.extraContext[button.dataset.removeContext];
    renderExtraContext();
  });
  els.addRestrictionBtn.addEventListener('click', addRestriction);
  els.restrictionList.addEventListener('click', (event) => {
    const button = event.target.closest('[data-remove-restriction]');
    if (!button) return;
    state.restrictions.splice(Number(button.dataset.removeRestriction), 1);
    renderRestrictions();
  });
  els.validationPanel.addEventListener('change', (event) => {
    const node = event.target.closest('[data-ack-rule]');
    if (!node) return;
    if (node.checked) state.acknowledgedRules.add(node.dataset.ackRule);
    else state.acknowledgedRules.delete(node.dataset.ackRule);
  });
  document.querySelectorAll('.mode').forEach((button) => button.addEventListener('click', () => {
    document.querySelectorAll('.mode').forEach((node) => node.classList.remove('active'));
    button.classList.add('active');
    state.mode = button.dataset.mode;
  }));
  els.validateBtn.addEventListener('click', validateOnly);
  els.generateBtn.addEventListener('click', generate);
  els.copyBtn.addEventListener('click', copyOutput);
  els.printBtn.addEventListener('click', printOutput);
  els.clearBtn.addEventListener('click', clearDraft);

  loadContract();
})();
