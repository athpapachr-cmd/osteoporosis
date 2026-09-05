(() => {
  const $ = id => document.getElementById(id);
  const state = { contract: null, pathway: 'A1', selectedHistoryId: '', nsaids: [], others: [], physio: null };

  async function api(path, options = {}) {
    const response = await fetch(`/clinical/clinic-utilities/rf${path}`, { credentials: 'same-origin', ...options });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try { const body = await response.json(); detail = body.detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    return response;
  }

  async function loadContract() {
    try {
      const response = await api('/api/contract');
      state.contract = await response.json();
      populateContract();
      const allConfigured = state.contract.doctor_configured && Object.values(state.contract.products).every(v => v.configured);
      $('configStatus').textContent = allConfigured ? 'RF v2 έτοιμο' : 'Απαιτείται server config';
      $('configStatus').className = `status-pill ${allConfigured ? 'ok' : 'warn'}`;
    } catch (error) {
      $('configStatus').textContent = 'σφάλμα φόρτωσης'; $('configStatus').className = 'status-pill error'; showError(error.message);
    }
  }

  function populateContract() {
    const c = state.contract;
    Object.entries(c.products).forEach(([key, item]) => {
      const option = new Option(item.label + (item.configured ? '' : ' — μη ρυθμισμένο'), key); option.disabled = !item.configured; $('productSelect').add(option);
    });
    Object.entries(c.indications).forEach(([key, item]) => $('indicationSelect').add(new Option(item.label, key)));
    Object.entries(c.laterality).forEach(([key, label]) => $('lateralitySelect').add(new Option(label, key)));
    $('lateralitySelect').value = 'none';
    $('reasonOptions').replaceChildren(...Object.entries(c.rf_reasons).map(([key, label]) => checkboxItem(key, label)));
  }

  function checkboxItem(value, label) {
    const wrap = document.createElement('label'); wrap.className = 'check-item';
    const input = document.createElement('input'); input.type = 'checkbox'; input.value = value; input.className = 'rf-reason';
    wrap.append(input, document.createTextNode(label)); return wrap;
  }

  function setPathway(pathway) {
    state.pathway = pathway;
    document.querySelectorAll('.segment').forEach(btn => btn.classList.toggle('active', btn.dataset.pathway === pathway));
    $('a1Flow').hidden = pathway !== 'A1'; $('a2Flow').hidden = pathway !== 'A2'; indicationChanged(); updateSummary();
  }

  function indicationChanged() {
    const code = $('indicationSelect').value; const item = state.contract?.indications?.[code]; const isOther = code.startsWith('OTHER_');
    $('otherAreaWrap').hidden = !isOther; $('otherDiagnosisWrap').hidden = !isOther;
    if (code === 'OTHER_LATERAL_EPICONDYLITIS') { $('otherArea').value = 'Αγκώνας'; $('otherDiagnosis').value = 'Έξω επικονδυλίτιδα'; }
    if (code === 'OTHER_DEQUERVAIN') { $('otherArea').value = 'Καρπός'; $('otherDiagnosis').value = 'De Quervain'; }
    if (code === 'OTHER_CUSTOM') { $('otherArea').value = ''; $('otherDiagnosis').value = ''; }
    $('interventionCard').hidden = !item?.requires_intervention || state.pathway !== 'A1';
    if (item?.site_key === 'si') { $('interventionTitle').textContent = 'Ιερολαγόνια — προηγούμενη έγχυση'; $('interventionHelp').textContent = 'Σημείο εφαρμογής, ημερομηνία και VAS πριν/μετά την έγχυση κορτιζόνης/τοπικού αναισθητικού.'; }
    else if (item?.site_key === 'hip') { $('interventionTitle').textContent = 'Ισχίο — διαγνωστικό block'; $('interventionHelp').textContent = 'Το νέο επίσημο έντυπο απαιτεί στοιχεία διαγνωστικού block για RF στο ισχίο.'; }
    state.selectedHistoryId = ''; $('historyChoices').replaceChildren(); updateSummary();
  }

  async function parseMedication() {
    const text = $('medicationText').value.trim();
    if (!text) { state.nsaids = []; state.others = []; renderTrials(); return; }
    try {
      $('medicationStatus').textContent = 'Ανάλυση…';
      const response = await api('/api/parse-medications', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({text}) });
      const result = await response.json(); state.nsaids = result.nsaid_candidates; state.others = result.other_candidates; renderTrials();
      $('medicationStatus').textContent = `Αυτόματη επιλογή ${result.auto_selected_nsaids.length + result.auto_selected_others.length} γραμμών · έλεγξε μόνο αν χρειάζεται.`;
    } catch (error) { showError(error.message); }
  }

  function renderTrials() { renderTrialGroup('nsaidRows', state.nsaids); renderTrialGroup('otherRows', state.others); }
  function renderTrialGroup(containerId, items) {
    const container = $(containerId); container.replaceChildren();
    if (!items.length) { const note = document.createElement('div'); note.className='muted'; note.textContent='Δεν αναγνωρίστηκε σχετικό φάρμακο.'; container.append(note); return; }
    items.forEach(item => {
      const row = document.createElement('div'); row.className='trial';
      const check=document.createElement('input'); check.type='checkbox'; check.checked=!!item.auto_selected;
      const drug=document.createElement('input'); drug.value=item.drug_name||'';
      const dose=document.createElement('input'); dose.value=item.dose||''; dose.placeholder='Δόση'; dose.className='dose';
      const duration=document.createElement('input'); duration.value=item.duration||''; duration.placeholder='Διάρκεια'; duration.className='duration';
      check.addEventListener('change',()=>item.auto_selected=check.checked); drug.addEventListener('input',()=>item.drug_name=drug.value); dose.addEventListener('input',()=>item.dose=dose.value); duration.addEventListener('input',()=>item.duration=duration.value);
      row.title=item.source_text||''; row.append(check,drug,dose,duration); container.append(row);
    });
  }
  function selectedTrials(items) { return items.filter(x=>x.auto_selected).slice(0,3).map(x=>({source_text:x.source_text||'',drug_name:x.drug_name||'',dose:x.dose||'',duration:x.duration||''})); }

  async function parsePhysio() {
    try {
      const response = await api('/api/parse-physio', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('physioDates').value})}); state.physio = await response.json();
      $('physioStart').textContent=state.physio.start_date||'—'; $('physioEnd').textContent=state.physio.end_date||'—'; $('physioCount').textContent=state.physio.treatment_count||0;
      const bad=state.physio.invalid_or_ambiguous_tokens||[]; $('physioWarning').hidden=!bad.length; $('physioWarning').textContent=bad.length?`Χρειάζονται διόρθωση: ${bad.join(', ')}`:'';
    } catch(error){ showError(error.message); }
  }

  async function lookupHistory() {
    const item=state.contract?.indications?.[$('indicationSelect').value];
    if (!$('identityNumber').value.trim() || !item) { showError('Συμπλήρωσε ταυτότητα και ένδειξη πριν την αναζήτηση.'); return; }
    try {
      $('historyStatus').textContent='Αναζήτηση…';
      const response=await api('/api/history',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({identity_number:$('identityNumber').value.trim(),site_key:item.site_key,laterality:$('lateralitySelect').value})});
      const result=await response.json(); renderHistory(result.procedures||[]); $('historyStatus').textContent=result.found?`${result.procedures.length} εφαρμογή/ές`:'Δεν βρέθηκε ιστορικό — χρησιμοποίησε legacy καταχώρηση.';
    } catch(error){ showError(error.message); }
  }

  function renderHistory(rows) {
    const container=$('historyChoices'); container.replaceChildren(); state.selectedHistoryId='';
    rows.forEach(row=>{
      const label=document.createElement('label'); label.className='history-option'; const radio=document.createElement('input'); radio.type='radio'; radio.name='history'; radio.value=row.procedure_history_id;
      radio.addEventListener('change',()=>{state.selectedHistoryId=row.procedure_history_id;$('legacyDetails').open=false;});
      const text=document.createElement('div'); text.innerHTML=`<strong>${escapeHtml(row.actual_procedure_date)}</strong> · ${escapeHtml(row.exact_location)}<br><span class="muted">VAS ${row.vas_before} → ${row.vas_after} · follow-up ${escapeHtml(row.last_followup_date)}: ${row.last_followup_vas}/10 · ${escapeHtml(row.provenance)}</span>`; label.append(radio,text); container.append(label);
    });
  }

  function escapeHtml(value){ return String(value||'').replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]||c)); }
  function adverseValues(){ return [...$('adverseRows').querySelectorAll('.adverse')].map(row=>({treatment:row.querySelector('[data-kind=treatment]').value.trim(),effect:row.querySelector('[data-kind=effect]').value.trim()})).filter(x=>x.treatment&&x.effect); }
  function addAdverse(){ const row=document.createElement('div'); row.className='adverse'; row.innerHTML='<input data-kind="treatment" placeholder="Φάρμακο / θεραπεία"><input data-kind="effect" placeholder="Ανεπιθύμητη ενέργεια"><button type="button" class="btn ghost small">×</button>'; row.querySelector('button').onclick=()=>row.remove(); $('adverseRows').append(row); }
  function checkedReasons(){ return [...document.querySelectorAll('.rf-reason:checked')].map(x=>x.value); }

  function buildDraft(){
    const code=$('indicationSelect').value, indication=state.contract?.indications?.[code];
    const draft={pathway:state.pathway,patient_name:$('patientName').value.trim(),identity_number:$('identityNumber').value.trim(),gesy_number:$('gesyNumber').value.trim(),age:Number($('age').value),product_key:$('productSelect').value,indication_code:code,laterality:$('lateralitySelect').value,exact_location:$('exactLocation').value.trim(),other_area:$('otherArea').value.trim(),other_diagnosis:$('otherDiagnosis').value.trim(),additional_notes:$('additionalNotes').value.trim()};
    if(state.pathway==='A1'){
      Object.assign(draft,{rf_reason_codes:checkedReasons(),rf_reason_other:$('reasonOther').value.trim(),pain_onset_date:$('painOnsetDate').value,pain_onset_vas:Number($('painOnsetVas').value),last_assessment_date:$('lastAssessmentDate').value,last_assessment_vas:Number($('lastAssessmentVas').value),full_medication_text:$('medicationText').value,nsaid_trials:selectedTrials(state.nsaids),other_analgesic_trials:selectedTrials(state.others),adverse_effects:adverseValues(),physio_dates_text:$('physioDates').value});
      if(indication?.requires_intervention) draft.intervention={site:$('interventionSite').value.trim(),date:$('interventionDate').value,vas_before:Number($('interventionVasBefore').value),vas_after:Number($('interventionVasAfter').value)};
    } else if(state.selectedHistoryId) draft.procedure_history_id=state.selectedHistoryId;
    else draft.legacy_history={actual_procedure_date:$('legacyProcedureDate').value,vas_before:Number($('legacyVasBefore').value),vas_after:Number($('legacyVasAfter').value),last_followup_date:$('legacyFollowupDate').value,last_followup_vas:Number($('legacyFollowupVas').value)};
    return draft;
  }

  async function createPdf(){
    const file=$('imagingReport').files[0]; if(!file){showError('Απαιτείται η απεικονιστική έκθεση PDF.');return;}
    try{
      $('createPdfBtn').disabled=true;$('createPdfBtn').textContent='Δημιουργία…'; const form=new FormData(); form.append('draft_json',JSON.stringify(buildDraft()));form.append('imaging_report',file,file.name);
      const response=await api('/api/create',{method:'POST',body:form}); const blob=await response.blob(); const url=URL.createObjectURL(blob); window.open(url,'_blank','noopener'); setTimeout(()=>URL.revokeObjectURL(url),120000); showOk('Το επίσημο PDF δημιουργήθηκε.');
    }catch(error){showError(error.message);}finally{$('createPdfBtn').disabled=false;$('createPdfBtn').textContent='Δημιουργία επίσημου PDF';}
  }
  function showError(message){$('validationPanel').hidden=false;$('validationPanel').className='validation-panel error';$('validationPanel').textContent=message;}
  function showOk(message){$('validationPanel').hidden=false;$('validationPanel').className='validation-panel ok';$('validationPanel').textContent=message;}
  function updateSummary(){const code=$('indicationSelect')?.value||'',label=state.contract?.indications?.[code]?.label||'—';$('summaryText').textContent=`${state.pathway==='A1'?'Νέα θεραπεία':'Συνέχιση'} · ${label}`;const items=[`Ασθενής: ${$('patientName')?.value||'—'}`,`Προϊόν: ${state.contract?.products?.[$('productSelect')?.value]?.label||'—'}`,`Εντόπιση: ${$('exactLocation')?.value||'—'}`];$('reviewSummary').innerHTML=items.map(x=>`<div class="summary-item">${escapeHtml(x)}</div>`).join('');}

  document.addEventListener('DOMContentLoaded',()=>{
    loadContract(); document.querySelectorAll('.segment').forEach(btn=>btn.addEventListener('click',()=>setPathway(btn.dataset.pathway))); $('indicationSelect').addEventListener('change',indicationChanged); $('parseMedicationBtn').addEventListener('click',parseMedication);
    let medTimer;$('medicationText').addEventListener('input',()=>{clearTimeout(medTimer);medTimer=setTimeout(parseMedication,500);}); let physioTimer;$('physioDates').addEventListener('input',()=>{clearTimeout(physioTimer);physioTimer=setTimeout(parsePhysio,350);});
    $('usualReasonBtn').addEventListener('click',()=>document.querySelectorAll('.rf-reason').forEach(x=>x.checked=['FAILED_PHARMACOLOGIC','FAILED_CONSERVATIVE'].includes(x.value))); $('addAdverseBtn').addEventListener('click',addAdverse); $('historyLookupBtn').addEventListener('click',lookupHistory); $('createPdfBtn').addEventListener('click',createPdf); $('clearBtn').addEventListener('click',()=>location.reload()); ['patientName','productSelect','exactLocation'].forEach(id=>$(id).addEventListener('input',updateSummary)); setPathway('A1'); addAdverse();
  });
})();
