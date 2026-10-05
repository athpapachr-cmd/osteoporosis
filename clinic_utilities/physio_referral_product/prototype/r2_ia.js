'use strict';
// Product-local R2 presentation. The existing projection, evidence and CU-1
// safety engine remain the authority; this layer gives each input one UI home.
const R2_RETIRED_FINDINGS = new Set([
  'pain','swelling','tenderness','effusion','joint_line_pain','anterior_peripatellar_pain',
  'walking_limitation','stairs_limitation','sit_to_stand_limitation','sport_or_exercise_limitation',
  'objective_weakness','quadriceps_weakness','active_rom_restricted','passive_rom_restricted',
  'subjective_giving_way','recurrent_instability_episode',
]);
const R2_REVIEW_LABELS = Object.freeze({
  recent_trauma:'Πρόσφατο τραύμα',
  rapid_worsening_or_deformity:'Ταχεία επιδείνωση ή παραμόρφωση',
  hot_swollen_joint:'Θερμή και διογκωμένη άρθρωση',
  acute_new_severe_pain:'Οξύς / νέος έντονος πόνος',
  major_weight_bearing_or_movement_difficulty:'Μεγάλη δυσκολία φόρτισης ή κίνησης',
  acute_or_rapid_deterioration:'Οξεία / ταχεία επιδείνωση',
  sudden_new_without_adequate_trauma:'Αιφνίδια έναρξη χωρίς επαρκές τραύμα',
  severe_weight_bearing_pain:'Έντονος πόνος στη φόρτιση',
  major_loading_difficulty:'Μεγάλη δυσκολία φόρτισης',
});
const r2Choice=(text,kind,id,extra={})=>btn(text,{class:'r2-choice','data-r2-kind':kind,'data-r2-id':id,'aria-pressed':'false',...extra});
const r2Group=(title,nodes,note='')=>make('div',{class:'r2-group'},[
  make('h3',{text:title}),...(note?[make('p',{class:'subtle small',text:note})]:[]),make('div',{class:'r2-options'},nodes),
]);
const r2Section=(title,id)=>make('section',{id,class:'r2-section'},[make('h2',{text:title})]);

function r2Install(){
  if(!meta||$('#r2ClinicalPicture'))return;
  const flow=$('.flow'),oldPicture=$('#pictureTitle')?.closest('section'),oldPlan=$('#planTitle')?.closest('section');
  if(!flow||!oldPicture||!oldPlan)return;
  const picture=r2Section('Κλινική εικόνα','r2ClinicalPicture');
  picture.append(
    r2Group('Συμπτώματα',[
      r2Choice('Πόνος','finding','pain'),
      r2Choice('Δυσκαμψία','phenotype','stiffness_symptom'),
      r2Choice('Αδυναμία που αναφέρεται','phenotype','weakness_symptom_or_context'),
      r2Choice('Οίδημα','finding','swelling'),
      r2Choice('Αίσθημα αστάθειας','finding','subjective_giving_way'),
      r2Choice('Επαναλαμβανόμενα επεισόδια αστάθειας','finding','recurrent_instability_episode'),
    ],'Το μη επιλεγμένο Οίδημα σημαίνει ότι δεν καταγράφηκε· δεν δηλώνει απουσία.'),
    make('div',{id:'r2StiffnessDetail',class:'r2-inline-detail',hidden:''},[
      r2Group('Δυσκαμψία · προαιρετικός προσδιορισμός',[
        r2Choice('Πρωινή','stiffness','morning'),r2Choice('Μετά από ακινησία','stiffness','after_inactivity'),
      ]),
      make('div',{id:'r2MorningDetail',hidden:''},[r2Group('Πρωινή διάρκεια',[
        r2Choice('≤30′','duration','le_30'),r2Choice('>30′','duration','gt_30'),
      ])]),
    ]),
    make('div',{class:'r2-duration'},[
      make('label',{class:'field'},['Διάρκεια συμπτωμάτων (προαιρετικά)',make('input',{id:'r2DurationValue',type:'number',min:'1',max:'99',inputmode:'numeric','aria-label':'Διάρκεια συμπτωμάτων'})]),
      make('label',{class:'field'},['Μονάδα',make('select',{id:'r2DurationUnit','aria-label':'Μονάδα διάρκειας συμπτωμάτων'},[
        make('option',{value:'weeks',text:'εβδομάδες'}),make('option',{value:'months',text:'μήνες'}),make('option',{value:'years',text:'έτη'}),
      ])]),
    ]),
    r2Group('Παρατηρήσεις που μπορεί να απαιτούν επανεκτίμηση',Object.entries(R2_REVIEW_LABELS).map(([id,name])=>r2Choice(name,'observation',id)),
      'Επίλεξε μόνο ρητά διαπιστωμένες παρατηρήσεις. Μία μεμονωμένη επιλογή δεν τεκμηριώνει διάγνωση.'),
    make('div',{id:'r2ReviewCues',class:'r2-review-cues',hidden:''}),
    r2Group('Ανεπίλυτες ανησυχίες ασφάλειας',Object.entries(meta.safety_labels).map(([id,name])=>r2Choice(name,'safety',id)),
      'Η ρητή επιλογή παραμένει ανεξάρτητο CU-1 block· η μη επιλογή δεν είναι αρνητικός έλεγχος.'),
  );

  const functionSection=r2Section('Λειτουργικότητα','r2Functionality');
  const frequent=['walking_tolerance','stairs','sit_to_stand','sport_gym'];
  const functionIds=Object.keys(meta.labels.functional_impairments);
  functionSection.append(
    r2Group('Συχνές δυσκολίες',frequent.map(id=>r2Choice(label(id),'function',id))),
    make('details',{class:'r2-details'},[
      make('summary',{text:'Άλλες λειτουργικές δυσκολίες'}),
      make('div',{class:'r2-options'},functionIds.filter(id=>!frequent.includes(id)).map(id=>r2Choice(label(id),'function',id))),
    ]),
  );

  const exam=r2Section('Εξέταση','r2Examination');
  exam.append(
    r2Group('Αντικειμενική μυϊκή αδυναμία',[
      r2Choice('Αδυναμία έκτασης / τετρακεφάλου','weakness','knee_extension_exam'),
      r2Choice('Αδυναμία κάμψης / ισχιοκνημιαίων','weakness','knee_flexion_exam'),
      r2Choice('Αδυναμία κάμψης και έκτασης','weakness','knee_extension_flexion_exam'),
      r2Choice('Ατροφία τετρακεφάλου','atrophy','quadriceps'),
    ]),
    r2Group('Εύρος κίνησης',[
      r2Choice('Περιορισμός ενεργητικής κάμψης','qualifier','active_flexion_restricted'),
      r2Choice('Περιορισμός παθητικής κάμψης','qualifier','passive_flexion_restricted'),
      r2Choice('Υστέρηση ενεργητικής έκτασης','finding','extension_lag'),
      r2Choice('Παθητικό έλλειμμα έκτασης','qualifier','fixed_flexion_deformity'),
      r2Choice('Επώδυνη ενεργητική κίνηση','finding','painful_active_rom'),
      r2Choice('Επώδυνη παθητική κίνηση','finding','painful_passive_rom'),
    ]),
    r2Group('Σταθερότητα και ισορροπία',[
      r2Choice('Διαταραχή ισορροπίας','finding','balance_deficit'),
      ...Object.entries(V51_STABILITY_LABELS).map(([id,name])=>r2Choice(name,'stability',id)),
    ]),
  );

  const plan=r2Section('Προτεινόμενο πλάνο','r2ProposedPlan');
  plan.append(make('p',{class:'subtle small',text:'Η ενεργητική αποκατάσταση παραμένει ο προεπιλεγμένος πυρήνας. Εσύ επιλέγεις κάθε πρόσθετη κατεύθυνση.'}));
  plan.append($('#plan'),$('#suggestions'));
  const additional=make('details',{id:'r2PlanAdditional',class:'r2-details'},[
    make('summary',{text:'Πρόσθετες / Περισσότερες επιλογές πλάνου'}),
    r2Group('Πρόσθετες κατευθύνσεις',Object.keys(meta.labels.rehab_directions)
      .filter(id=>!meta.defaults.includes(id)).map(id=>make('div',{'data-r2-extra-id':id},[row(id,'rehab_directions')])),
      'Το βοήθημα βάδισης είναι προαιρετικό και δεν προτείνεται ή επιλέγεται αυτόματα.'),
    r2Group('Λειτουργικοί στόχοι',Object.keys(meta.labels.goals).map(id=>r2Choice(label(id),'goal',id))),
    r2Group('Συμπληρωματικές επιλογές',Object.keys(meta.labels.adjuncts)
      .map(id=>make('div',{'data-r2-extra-id':id},[row(id,'adjunct_options')]))),
    make('div',{class:'r2-restriction'},[
      make('label',{class:'field'},['Ρητός περιορισμός (προαιρετικά)',make('select',{id:'r2RestrictionId','aria-label':'Είδος περιορισμού'},[
        make('option',{value:'',text:'Χωρίς καταγεγραμμένο περιορισμό'}),
        ...Object.entries(meta.restrictions).map(([id,name])=>make('option',{value:id,text:name})),
      ])]),
      make('label',{class:'field'},['Οδηγία',make('textarea',{id:'r2RestrictionText',maxlength:'300',rows:'2','aria-label':'Ρητή οδηγία περιορισμού',placeholder:'Μόνο η ρητή κλινική οδηγία'})]),
    ]),
  ]);
  plan.append(additional);
  const note=make('textarea',{id:'r2ClinicalNote',maxlength:'800',rows:'3',autocomplete:'off',spellcheck:'false',
    'aria-label':'Πρόσθετη κλινική σημείωση',placeholder:'Προαιρετική πληροφορία προς τον φυσιοθεραπευτή'});
  plan.append(make('label',{class:'field r2-note'},['Πρόσθετη κλινική σημείωση',note]));

  oldPicture.before(picture,functionSection,exam,plan);
  for(const old of [oldPicture,oldPlan,$('#advancedToggle')?.closest('section')]){
    if(old){old.hidden=true;old.inert=true;old.style.display='none';}
  }
  note.addEventListener('input',()=>{state.clinician_free_text_optional=note.value;changed();});
  const duration=$('#r2DurationValue'),unit=$('#r2DurationUnit');
  const durationChange=()=>{
    const number=Number(duration.value);
    if(duration.value&&(!Number.isInteger(number)||number<1||number>99)){
      qualifierState.symptom_duration_value=null;qualifierState.symptom_duration_unit=null;
      duration.setCustomValidity('Δώσε ακέραιο αριθμό από 1 έως 99.');duration.reportValidity();changed('r2_duration');return;
    }
    duration.setCustomValidity('');
    qualifierState.symptom_duration_value=duration.value?number:null;
    qualifierState.symptom_duration_unit=duration.value?unit.value:null;
    changed('r2_duration');
  };
  duration.addEventListener('change',durationChange);unit.addEventListener('change',()=>{if(duration.value)durationChange();});
  const restrictionId=$('#r2RestrictionId'),restrictionText=$('#r2RestrictionText');
  const restrictionChange=()=>{
    state.explicit_restrictions=restrictionId.value?[{restriction_id:restrictionId.value,state_or_value:restrictionText.value,source:'clinician_entered'}]:[];
    changed('r2_restriction');
  };
  restrictionId.addEventListener('change',restrictionChange);restrictionText.addEventListener('change',restrictionChange);
  r2Sync();
}

function r2Sync(){
  if(!state||!$('#r2ClinicalPicture'))return;
  for(const b of $$('[data-r2-kind]')){
    const kind=b.dataset.r2Kind,id=b.dataset.r2Id;
    let active=false;
    if(kind==='finding')active=state.findings.includes(id);
    else if(kind==='function')active=state.functional_impairments.includes(id);
    else if(kind==='goal')active=state.goals.includes(id);
    else if(kind==='safety')active=state.safety_flags.includes(id);
    else if(kind==='phenotype')active=!!state.phenotype[id];
    else if(kind==='stiffness')active=qualifierState.stiffness_patterns.includes(id);
    else if(kind==='duration')active=qualifierState.morning_stiffness_duration===id;
    else if(kind==='weakness')active=qualifierState.weakness_detail===id;
    else if(kind==='atrophy')active=qualifierState.visible_atrophy&&qualifierState.atrophy_location===id;
    else if(kind==='qualifier'||kind==='observation')active=qualifierState[id]===true;
    else if(kind==='stability')active=(qualifierState.stability_findings||[]).includes(id);
    b.setAttribute('aria-pressed',String(active));
  }
  $('#r2StiffnessDetail').hidden=!state.phenotype.stiffness_symptom;
  $('#r2MorningDetail').hidden=!qualifierState.stiffness_patterns.includes('morning');
  for(const extra of $$('[data-r2-extra-id]'))extra.hidden=chosen().includes(extra.dataset.r2ExtraId);
  const note=$('#r2ClinicalNote');if(note&&document.activeElement!==note)note.value=state.clinician_free_text_optional;
  const duration=$('#r2DurationValue');if(duration&&document.activeElement!==duration)duration.value=qualifierState.symptom_duration_value??'';
  const unit=$('#r2DurationUnit');if(unit&&document.activeElement!==unit)unit.value=qualifierState.symptom_duration_unit||'months';
  const restriction=state.explicit_restrictions[0];
  const restrictionId=$('#r2RestrictionId');if(restrictionId&&document.activeElement!==restrictionId)restrictionId.value=restriction?.restriction_id||'';
  const restrictionText=$('#r2RestrictionText');if(restrictionText&&document.activeElement!==restrictionText)restrictionText.value=restriction?.state_or_value||'';
  r2RenderCues();
}

function r2RenderCues(){
  const host=$('#r2ReviewCues');if(!host)return;
  const cues=fresh()?response.review_cues||[]:[];
  if(!cues.length){host.hidden=true;host.replaceChildren();return;}
  const choice=response?.gate?.review_choice;
  const rows=cues.map(cue=>make('div',{class:'r2-cue'},[
    make('strong',{text:'Σήμα κλινικής επανεκτίμησης'}),
    make('p',{text:cue.message}),
    make('p',{class:'small',text:'Καταγράφηκαν: '+cue.positive_observations.map(id=>R2_REVIEW_LABELS[id]||id).join(' · ')}),
    make('p',{class:'small subtle'},[make('a',{href:cue.source_url,target:'_blank',rel:'noopener noreferrer',text:cue.source_label}),' · '+cue.reviewed_on]),
  ]));
  host.replaceChildren(...rows,make('p',{class:'small',text:'Η παραπομπή απαιτεί ρητή κλινική απόφαση πριν από την αντιγραφή.'}),
    make('div',{class:'r2-options'},[
      btn('Συνέχιση συνήθους παραπομπής μετά από κλινική επανεκτίμηση',{'data-r2-decision':'continue','aria-pressed':String(choice==='continue')}),
      btn('Αναβολή παραπομπής / πρώτα επανεκτίμηση',{'data-r2-decision':'defer','aria-pressed':String(choice==='defer')}),
    ]));
  host.hidden=false;
}

const r2BasePaint=paint;
paint=function(){r2BasePaint();r2Install();r2Sync();};
const r2BaseStatusText=statusText;
statusText=function(){
  if(fresh()&&response?.gate?.review_required&&!response?.gate?.blocked){
    if(response.gate.review_choice==='defer')return 'Παραπομπή σε αναβολή · πρώτα κλινική επανεκτίμηση';
    if(!response.gate.review_choice)return 'Απαιτείται απόφαση κλινικής επανεκτίμησης';
  }
  return r2BaseStatusText();
};

document.addEventListener('click',event=>{
  const b=event.target.closest('[data-r2-kind],[data-r2-decision]');if(!b||!state)return;
  event.preventDefault();event.stopImmediatePropagation();
  if(b.dataset.r2Decision){
    const cues=response?.review_cues||[];
    if(!fresh()||!cues.length)return;
    window.physioR2ReviewDecision={revision,cue_ids:cues.map(c=>c.rule_id).sort(),choice:b.dataset.r2Decision};
    refresh();return;
  }
  const kind=b.dataset.r2Kind,id=b.dataset.r2Id;
  if(kind==='finding')state.findings=state.findings.includes(id)?state.findings.filter(v=>v!==id):[...state.findings,id];
  else if(kind==='function'||kind==='goal'||kind==='safety'){
    const key={function:'functional_impairments',goal:'goals',safety:'safety_flags'}[kind];
    state[key]=state[key].includes(id)?state[key].filter(v=>v!==id):[...state[key],id];
  }else if(kind==='phenotype'){
    state.phenotype[id]=!state.phenotype[id];
    if(id==='stiffness_symptom'&&!state.phenotype[id])qResetStiffness();
  }else if(kind==='stiffness'){
    qualifierState.stiffness_patterns=qualifierState.stiffness_patterns.includes(id)
      ?qualifierState.stiffness_patterns.filter(v=>v!==id):[...qualifierState.stiffness_patterns,id];
    if(id==='morning'&&!qualifierState.stiffness_patterns.includes(id))qualifierState.morning_stiffness_duration=null;
  }else if(kind==='duration')qualifierState.morning_stiffness_duration=qualifierState.morning_stiffness_duration===id?null:id;
  else if(kind==='weakness')qualifierState.weakness_detail=qualifierState.weakness_detail===id?null:id;
  else if(kind==='atrophy'){
    const active=qualifierState.visible_atrophy&&qualifierState.atrophy_location===id;
    qualifierState.visible_atrophy=!active;qualifierState.atrophy_location=active?null:id;
  }else if(kind==='qualifier'||kind==='observation')qualifierState[id]=!qualifierState[id];
  else if(kind==='stability'){
    const values=qualifierState.stability_findings||[];
    qualifierState.stability_findings=values.includes(id)?values.filter(v=>v!==id):[...values,id];
  }
  changed('r2_'+id);
},true);
