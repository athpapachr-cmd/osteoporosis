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
    subtypeWrap: $('subtypeWrap'), subtype: $('subtypeSelect'), assertionWrap: $('assertionWrap'), assertion: $('assertionSelect'),
    contextFields: $('contextFields'), extraContextKey: $('extraContextKey'), extraContextValue: $('extraContextValue'),
    extraContextList: $('extraContextList'), addContextBtn: $('addContextBtn'),
    findings: $('findingsOptions'), functions: $('functionOptions'), goals: $('goalOptions'), rehab: $('rehabOptions'), adjuncts: $('adjunctOptions'),
    restrictionId: $('restrictionIdSelect'), restrictionValue: $('restrictionValue'), restrictionSource: $('restrictionSource'), restrictionList: $('restrictionList'), addRestrictionBtn: $('addRestrictionBtn'),
    safetyFlags: $('safetyFlagOptions'), disposition: $('dispositionSelect'), sessions: $('sessionsInput'), freeText: $('clinicianFreeText'),
    validateBtn: $('validateBtn'), generateBtn: $('generateBtn'), validationPanel: $('validationPanel'), validationSummary: $('validationSummary'),
    output: $('outputText'), copyBtn: $('copyBtn'), printBtn: $('printBtn'), clearBtn: $('clearBtn'),
  };

  const greek = {
    left:'Αριστερά', right:'Δεξιά', bilateral:'Αμφοτερόπλευρα', midline:'Μέση γραμμή', not_applicable:'Δεν εφαρμόζεται', not_stated:'Δεν δηλώνεται',
    presentation:'Παρουσίαση / συμπτωματολογία', formal_diagnosis:'Ρητή κλινική διάγνωση', established_structural_diagnosis:'Εγκατεστημένη δομική διάγνωση', postoperative:'Μετεγχειρητικό', shared_structural:'Κοινό δομικό pathway',
    pain:'Πόνος', swelling:'Οίδημα', tenderness:'Ευαισθησία', bruising:'Εκχύμωση', active_rom_restricted:'Περιορισμένο ενεργητικό ROM', passive_rom_restricted:'Περιορισμένο παθητικό ROM', painful_active_rom:'Επώδυνο ενεργητικό ROM', painful_passive_rom:'Επώδυνο παθητικό ROM', objective_weakness:'Αντικειμενική μυϊκή αδυναμία', load_intolerance_without_measured_weakness:'Δυσανεξία φόρτισης χωρίς μετρημένη αδυναμία', paresthesia:'Παραισθησίες', numbness:'Αιμωδία', night_or_sleep_disturbance:'Νυχτερινή ενόχληση / διαταραχή ύπνου', walking_limitation:'Περιορισμός βάδισης', stairs_limitation:'Δυσκολία στις σκάλες', sit_to_stand_limitation:'Δυσκολία sit-to-stand', work_limitation:'Περιορισμός εργασίας', sport_or_exercise_limitation:'Περιορισμός άθλησης/άσκησης', adl_self_care_limitation:'Περιορισμός ADL/αυτοεξυπηρέτησης', balance_deficit:'Έλλειμμα ισορροπίας', poor_coordination:'Μη καλός συντονισμός', fear_or_concern_about_falling:'Φόβος/ανησυχία για πτώση',
    walking_tolerance:'Ανοχή βάδισης', standing_tolerance:'Ανοχή ορθοστασίας', sitting_tolerance:'Ανοχή καθιστής θέσης', stairs:'Σκάλες', sit_to_stand:'Sit-to-stand', transfers:'Μεταφορές', lifting_carrying:'Άρση/μεταφορά φορτίου', pushing_pulling:'Ώθηση/έλξη', driving:'Οδήγηση', desk_or_computer_work:'Εργασία γραφείου/ΗΥ', overhead_activity:'Δραστηριότητες πάνω από το ύψος του ώμου', gripping:'Λαβή', pinch:'Pinch', dexterity:'Επιδεξιότητα', squat:'Κάθισμα/squat', kneeling:'Γονάτισμα', running:'Τρέξιμο', jumping_landing:'Άλμα/προσγείωση', pivot_change_of_direction:'Pivot/αλλαγή κατεύθυνσης', sport_gym:'Άθληση/γυμναστήριο', manual_work:'Χειρωνακτική εργασία', school_pe_youth_sport:'Σχολική ΦΑ/νεανικός αθλητισμός', community_mobility:'Κινητικότητα στην κοινότητα', patient_priority_activity:'Δραστηριότητα προτεραιότητας ασθενούς',
    reduce_symptom_irritability:'Μείωση ερεθιστικότητας συμπτωμάτων', restore_safe_functional_rom:'Αποκατάσταση ασφαλούς λειτουργικού ROM', improve_strength:'Βελτίωση δύναμης', improve_endurance:'Βελτίωση αντοχής', improve_motor_control:'Βελτίωση κινητικού ελέγχου', improve_neuromuscular_control:'Βελτίωση νευρομυϊκού ελέγχου', improve_balance_postural_control:'Βελτίωση ισορροπίας/στασικού ελέγχου', improve_coordination:'Βελτίωση συντονισμού', improve_walking_tolerance:'Βελτίωση ανοχής βάδισης', improve_stair_function:'Βελτίωση λειτουργίας στις σκάλες', improve_sit_to_stand_transfer_function:'Βελτίωση sit-to-stand/μεταφορών', improve_grip_pinch_or_dexterity:'Βελτίωση λαβής/pinch/επιδεξιότητας', improve_load_tolerance:'Βελτίωση ανοχής φόρτισης', graded_return_to_activity:'Σταδιακή επιστροφή σε δραστηριότητα', graded_return_to_work:'Σταδιακή επιστροφή στην εργασία', graded_return_to_sport:'Σταδιακή επιστροφή στον αθλητισμό', improve_self_management:'Βελτίωση αυτοδιαχείρισης', improve_mobility_confidence:'Βελτίωση εμπιστοσύνης στην κινητικότητα', optimize_safe_walking_aid_use:'Βελτιστοποίηση ασφαλούς χρήσης βοηθήματος βάδισης', maintain_or_regain_adl_independence:'Διατήρηση/ανάκτηση ανεξαρτησίας ADL', reduce_falls_risk_through_modifiable_physical_factors:'Μείωση κινδύνου πτώσεων μέσω τροποποιήσιμων φυσικών παραγόντων',
    physiotherapy_assessment_and_individualized_active_rehabilitation:'Φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση', therapeutic_exercise:'Θεραπευτική άσκηση', progressive_strengthening:'Προοδευτική ενδυνάμωση', progressive_endurance_or_capacity_work:'Προοδευτική βελτίωση αντοχής/ικανότητας', mobility_exercise_when_restricted:'Ασκήσεις κινητικότητας όταν υπάρχει περιορισμός', graded_activity_exposure:'Σταδιακή έκθεση σε δραστηριότητα', graded_loading:'Σταδιακή φόρτιση', education_and_self_management:'Εκπαίδευση και αυτοδιαχείριση', home_exercise_programme:'Πρόγραμμα ασκήσεων στο σπίτι', work_activity_load_adaptation:'Προσαρμογή φορτίου εργασίας/δραστηριότητας', neuromuscular_proprioceptive_training:'Νευρομυϊκή/ιδιοδεκτική εκπαίδευση', balance_stepping_recovery_training:'Εκπαίδευση ισορροπίας/βηματισμού/ανάκτησης', gait_walking_practice:'Εκπαίδευση βάδισης', walking_aid_assessment_and_training:'Αξιολόγηση και εκπαίδευση βοηθήματος βάδισης', functional_task_retraining:'Επανεκπαίδευση λειτουργικών δραστηριοτήτων', progressive_running:'Προοδευτικό τρέξιμο', progressive_sprinting:'Προοδευτικό sprint', progressive_kicking:'Προοδευτικές λακτίσεις', progressive_jump_landing:'Προοδευτικό άλμα/προσγείωση', progressive_change_of_direction:'Προοδευτική αλλαγή κατεύθυνσης', criterion_based_return_to_training_sport_or_work:'Criterion-based επιστροφή σε προπόνηση/άθληση/εργασία', edema_management:'Διαχείριση οιδήματος', scar_desensitization_management:'Διαχείριση ουλής/απευαισθητοποίηση', protected_rom_within_restrictions:'Προστατευμένο ROM εντός περιορισμών', progressive_weight_bearing_within_restrictions:'Προοδευτική φόρτιση βάρους εντός περιορισμών', progressive_upper_limb_use_within_restrictions:'Προοδευτική χρήση άνω άκρου εντός περιορισμών',
    manual_therapy:'Manual therapy', soft_tissue_techniques:'Τεχνικές μαλακών μορίων', neurodynamic_techniques:'Νευροδυναμικές τεχνικές', selected_cervical_traction:'Επιλεγμένη αυχενική έλξη', dry_needling:'Dry needling', acupuncture:'Βελονισμός', eswt:'ESWT', taping:'Taping', heel_lift:'Heel lift', night_splint:'Night splint', orthosis_or_brace_context:'Ορθωτικό/brace ως context',
  };

  const label = (id) => greek[id] || String(id || '').replaceAll('_', ' ');
  const option = (value, text = label(value)) => `<option value="${escapeHtml(value)}">${escapeHtml(text)}</option>`;
  const escapeHtml = (text) => String(text ?? '').replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

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
      const contract = await api('/contract');
      state.contract = contract;
      els.contractStatus.textContent = `${contract.contract_version} · loaded`;
      els.contractStatus.classList.add('ok');
      renderProfiles();
      renderLaterality();
      renderStaticOptionSets();
      els.validationSummary.textContent = 'Επίλεξε κύριο πρόβλημα και συμπλήρωσε το referral draft.';
    } catch (error) {
      els.contractStatus.textContent = 'contract error';
      els.contractStatus.classList.add('error');
      showValidation({validation_errors:[{error_id:'contract_load_error', error_class:'validation_error', metadata:{detail:error.message}}], safety_results:[], formatter_blocked:true});
    }
  }

  function renderProfiles() {
    els.profile.innerHTML = '<option value="">Επίλεξε…</option>';
    Object.entries(state.contract.profiles).forEach(([id, profile]) => {
      els.profile.insertAdjacentHTML('beforeend', option(id, profile.display));
    });
  }

  function renderLaterality() {
    els.laterality.innerHTML = '';
    state.contract.laterality_values.forEach((id) => els.laterality.insertAdjacentHTML('beforeend', option(id)));
    els.laterality.value = 'not_stated';
  }

  function renderStaticOptionSets() {
    renderChecks(els.functions, state.contract.functional_impairments, 'functional');
    renderChecks(els.goals, state.contract.goals, 'goal');
    renderChecks(els.rehab, state.contract.rehab_directions, 'rehab');
    renderChecks(els.adjuncts, state.contract.adjuncts, 'adjunct');
    renderChecks(els.safetyFlags, state.contract.safety_input_flags, 'safety');
    els.restrictionId.innerHTML = '<option value="">Επίλεξε…</option>' + state.contract.restrictions.map((id) => option(id)).join('');
  }

  function renderChecks(container, ids, group) {
    container.innerHTML = (ids || []).map((id) => `
      <label class="check-item"><input type="checkbox" data-group="${group}" value="${escapeHtml(id)}" /> <span>${escapeHtml(label(id))}</span></label>
    `).join('') || '<span class="empty-note">Δεν υπάρχουν επιλογές.</span>';
  }

  function onProfileChange() {
    const profileId = els.profile.value;
    const profile = state.contract.profiles[profileId];
    els.route.disabled = !profile;
    els.route.innerHTML = profile ? '<option value="">Επίλεξε…</option>' : '<option value="">Επίλεξε περιοχή πρώτα</option>';
    if (profile) {
      const routeEntries = Object.entries(profile.routes).sort((a,b) => visibilityRank(a[1].visibility) - visibilityRank(b[1].visibility));
      routeEntries.forEach(([id, route]) => {
        const suffix = route.visibility === 'rare_advanced' ? ' · advanced' : route.visibility === 'visible_less_frequent' ? ' · less frequent' : '';
        els.route.insertAdjacentHTML('beforeend', option(id, `${route.display}${suffix}`));
      });
    }
    renderFindings();
    resetRouteDependent();
  }

  const visibilityRank = (value) => ({routine:0, visible_less_frequent:1, rare_advanced:2, shared_gateway:3, context_only:4}[value] ?? 9);

  function onRouteChange() {
    const route = currentRoute();
    els.wording.disabled = !route;
    els.wording.innerHTML = route ? '<option value="">Επίλεξε…</option>' + (route.wording_modes || []).map((id) => option(id)).join('') : '<option value="">Επίλεξε pathway πρώτα</option>';
    renderSubtype();
    renderContextFields();
    renderAssertion();
  }

  function onWordingChange() {
    renderAssertion();
    renderContextFields();
  }

  function currentRoute() {
    return state.contract?.profiles?.[els.profile.value]?.routes?.[els.route.value] || null;
  }

  function resetRouteDependent() {
    els.wording.disabled = true;
    els.wording.innerHTML = '<option value="">Επίλεξε pathway πρώτα</option>';
    els.subtypeWrap.hidden = true;
    els.assertionWrap.hidden = true;
    els.contextFields.innerHTML = '<span class="empty-note">Επίλεξε pathway.</span>';
    state.extraContext = {};
    renderExtraContext();
  }

  function renderSubtype() {
    const subtypes = state.contract.subtypes?.[els.route.value] || [];
    els.subtypeWrap.hidden = subtypes.length === 0;
    els.subtype.innerHTML = '<option value="">Δεν δηλώνεται</option>' + subtypes.map((id) => option(id)).join('');
  }

  function renderAssertion() {
    const mode = els.wording.value;
    els.assertionWrap.hidden = mode !== 'formal_diagnosis';
    if (mode !== 'formal_diagnosis') els.assertion.value = 'not_stated';
  }

  function renderFindings() {
    if (!state.contract) return;
    const common = state.contract.common_findings || [];
    const specific = state.contract.profile_findings?.[els.profile.value] || [];
    const ids = [...new Set([...common, ...specific])];
    renderChecks(els.findings, ids, 'finding');
  }

  function renderContextFields() {
    const profile = els.profile.value;
    const route = els.route.value;
    const mode = els.wording.value;
    const fields = [];

    if (mode === 'established_structural_diagnosis') {
      fields.push(textField('established_diagnosis_source', 'Πηγή εγκατεστημένης διάγνωσης', 'clinician_entered'));
      fields.push(textField('management_context', 'Management context', 'conservative_rehabilitation'));
    }

    if (mode === 'postoperative' || route.startsWith('postoperative_')) {
      fields.push(textField('procedure', 'Επέμβαση / procedure', ''));
      fields.push(textField('procedure_date_or_phase', 'Ημερομηνία ή φάση επέμβασης', ''));
      fields.push(selectField('protocol_status', 'Protocol status', state.contract.context_value_sets.postoperative_common.protocol_status || []));
      fields.push(selectField('restrictions_review_status', 'Restrictions review status', state.contract.context_value_sets.postoperative_common.restrictions_review_status || []));
      fields.push(textField('protocol_source', 'Protocol source (αν διαθέσιμο)', ''));
    }

    if (profile === 'shared_fracture') {
      const v = state.contract.context_value_sets.shared_fracture;
      fields.push(textField('fracture_site', 'Canonical fracture site', 'π.χ. distal_radius_fracture'));
      fields.push(selectField('fracture_phase', 'Φάση κατάγματος', v.fracture_phase || []));
      fields.push(selectField('fracture_context', 'Context κατάγματος', v.fracture_context || []));
      fields.push(selectField('treatment', 'Αντιμετώπιση', v.treatment || []));
      fields.push(selectField('healing_stability_status', 'Healing / stability status', v.healing_stability_status || []));
      fields.push(selectField('immobilization_status', 'Immobilization status', v.immobilization_status || []));
      fields.push(selectField('weight_bearing_status', 'Weight-bearing status', v.weight_bearing_status || []));
      fields.push(selectField('upper_limb_use_status', 'Upper-limb use status', v.upper_limb_use_status || []));
      fields.push(selectField('rom_status', 'ROM status', v.rom_status || []));
      fields.push(selectField('loading_strengthening_status', 'Loading / strengthening status', v.loading_strengthening_status || []));
      fields.push(selectField('orthopaedic_instructions_source', 'Πηγή ορθοπαιδικών οδηγιών', v.orthopaedic_instructions_source || []));
    }

    if (profile === 'shared_muscle_myotendinous') {
      const v = state.contract.context_value_sets.shared_muscle_myotendinous;
      fields.push(textField('muscle_group', 'Canonical muscle group', 'π.χ. hamstring_muscle_injury'));
      fields.push(textField('specific_muscle_optional', 'Specific muscle (optional)', ''));
      fields.push(selectField('injury_phase', 'Φάση κάκωσης', v.injury_phase || []));
      fields.push(selectField('injury_type', 'Τύπος κάκωσης', v.injury_type || []));
      fields.push(selectField('injury_location_optional', 'Εντόπιση', v.injury_location || []));
      fields.push(selectField('mri_or_ultrasound_confirmed', 'MRI/US confirmation state', v.mri_or_ultrasound_confirmed || []));
      fields.push(selectField('management_context', 'Management context', v.management_context || []));
    }

    if (profile === 'shared_deconditioning_balance_gait') {
      fields.push(selectField('functional_route_id', 'Functional route', ['generalized_deconditioning_functional_decline','frailty_associated_functional_decline','balance_impairment_context','gait_mobility_impairment_context','post_illness_or_post_hospital_deconditioning_context']));
      fields.push(selectField('frailty_established', 'Frailty established', state.contract.context_value_sets.shared_deconditioning_balance_gait.frailty_established || []));
    }

    if (['neck_pain_with_radiating_upper_limb_symptoms','low_back_pain_with_radiating_leg_symptoms'].includes(route)) {
      fields.push(`<div class="context-group"><h3>Neurological screen</h3><div class="grid three">${selectField('neurological_screen.motor','Motor',['not_assessed','normal','abnormal'])}${selectField('neurological_screen.sensory','Sensory',['not_assessed','normal','abnormal'])}${selectField('neurological_screen.reflexes','Reflexes',['not_assessed','normal','abnormal'])}</div></div>`);
    }

    els.contextFields.innerHTML = fields.join('') || '<span class="empty-note">Δεν απαιτείται ειδικό structured context για το επιλεγμένο pathway. Μπορείς να προσθέσεις canonical context από το advanced πεδίο.</span>';
  }

  function textField(key, title, placeholder) {
    return `<label>${escapeHtml(title)}<input type="text" data-context-key="${escapeHtml(key)}" placeholder="${escapeHtml(placeholder)}" /></label>`;
  }

  function selectField(key, title, values) {
    return `<label>${escapeHtml(title)}<select data-context-key="${escapeHtml(key)}"><option value="">Δεν δηλώνεται</option>${(values || []).map((id) => option(id)).join('')}</select></label>`;
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
    els.restrictionList.innerHTML = state.restrictions.map((item,index) => `<span class="chip">${escapeHtml(label(item.restriction_id))}: ${escapeHtml(item.state_or_value)} <button type="button" data-remove-restriction="${index}">×</button></span>`).join('');
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
    const findings = checkedValues('finding').map((id) => ({finding_id:id, state_optional:null, laterality_optional:null, value_optional:null, unit_optional:null, free_text_optional:null}));
    const functional = checkedValues('functional').map((id) => ({id, selected:true, notes_optional:null}));
    const goals = checkedValues('goal').map((id) => ({id, selected:true, notes_optional:null}));
    const rehab = checkedValues('rehab').map((id) => ({id, selected:true, notes_optional:null}));
    const adjuncts = checkedValues('adjunct').map((id) => ({adjunct_id:id, selected:true, provenance:'clinician_selected'}));
    const safetyFlags = checkedValues('safety');
    const sessions = els.sessions.value ? Number(els.sessions.value) : null;
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
      findings,
      functional_impairments: functional,
      precautions: [],
      explicit_restrictions: state.restrictions,
      goals,
      rehab_directions: rehab,
      adjunct_options: adjuncts,
      measurements: [],
      safety: {
        input_flags: safetyFlags,
        acknowledged_rule_ids: [...state.acknowledgedRules],
        clinician_disposition: els.disposition.value || 'none_recorded',
      },
      sessions_optional: sessions,
      clinician_free_text_optional: els.freeText.value.trim() || null,
    };
  }

  async function validateOnly() {
    try {
      const result = await api('/validate', {method:'POST', body:JSON.stringify({draft:collectDraft()})});
      showValidation(result);
    } catch (error) {
      showValidation({validation_errors:[{error_id:'request_failed', error_class:'validation_error', metadata:{detail:error.message}}], safety_results:[], formatter_blocked:true});
    }
  }

  async function generate() {
    try {
      const result = await api('/generate', {method:'POST', body:JSON.stringify({draft:collectDraft(), mode:state.mode})});
      showValidation(result);
      els.output.value = result.text || '';
      const hasText = Boolean(result.text);
      els.copyBtn.disabled = !hasText;
      els.printBtn.disabled = !hasText;
    } catch (error) {
      showValidation({validation_errors:[{error_id:'request_failed', error_class:'validation_error', metadata:{detail:error.message}}], safety_results:[], formatter_blocked:true});
      els.output.value = '';
      els.copyBtn.disabled = true;
      els.printBtn.disabled = true;
    }
  }

  function showValidation(result) {
    const errors = result.validation_errors || [];
    const safety = result.safety_results || [];
    els.validationPanel.hidden = false;
    els.validationPanel.className = 'validation-panel';
    if (!errors.length && !safety.length && !result.formatter_blocked) {
      els.validationPanel.classList.add('ok');
      els.validationPanel.innerHTML = '<strong>Validation PASS.</strong> Δεν υπάρχει ενεργό blocking αποτέλεσμα.';
      els.validationSummary.textContent = 'Έτοιμο για generation.';
      return;
    }
    const hardErrors = errors.filter((item) => item.error_class === 'validation_error');
    if (hardErrors.length || result.formatter_blocked) els.validationPanel.classList.add('error');
    const chunks = [];
    if (errors.length) {
      chunks.push(`<strong>Validation</strong><ul>${errors.map((item) => `<li><b>${escapeHtml(item.error_id)}</b> · ${escapeHtml(JSON.stringify(item.metadata || {}))}</li>`).join('')}</ul>`);
    }
    if (safety.length) {
      chunks.push(`<strong>Safety / consistency</strong><ul>${safety.map((item) => {
        const ack = item.acknowledgement_required ? `<label class="check-item"><input type="checkbox" data-ack-rule="${escapeHtml(item.rule_id)}" ${state.acknowledgedRules.has(item.rule_id) ? 'checked' : ''}/> Αναγνώριση ${escapeHtml(item.rule_id)}</label>` : '';
        return `<li><b>${escapeHtml(item.severity)}</b> · ${escapeHtml(item.rule_id)} ${item.formatter_blocked ? '· BLOCKED' : ''}${ack}</li>`;
      }).join('')}</ul>`);
    }
    els.validationPanel.innerHTML = chunks.join('');
    els.validationSummary.textContent = result.formatter_blocked ? 'Απαιτείται διόρθωση/acknowledgement/disposition πριν από generation.' : 'Υπάρχουν μη blocking παρατηρήσεις.';
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

  function clearDraft() {
    document.querySelectorAll('input[type="checkbox"]').forEach((node) => { node.checked = false; });
    document.querySelectorAll('input[type="text"],input[type="number"],textarea').forEach((node) => { node.value = ''; });
    els.profile.value = '';
    els.laterality.value = 'not_stated';
    els.disposition.value = 'none_recorded';
    state.extraContext = {};
    state.restrictions = [];
    state.acknowledgedRules.clear();
    renderExtraContext();
    renderRestrictions();
    resetRouteDependent();
    renderFindings();
    els.output.value = '';
    els.copyBtn.disabled = true;
    els.printBtn.disabled = true;
    els.validationPanel.hidden = true;
    els.validationSummary.textContent = 'Συμπλήρωσε το κύριο πρόβλημα.';
  }

  els.profile.addEventListener('change', onProfileChange);
  els.route.addEventListener('change', onRouteChange);
  els.wording.addEventListener('change', onWordingChange);
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
