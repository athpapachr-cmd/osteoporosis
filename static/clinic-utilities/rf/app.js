(() => {
  const $ = id => document.getElementById(id);
  const state = { contract: null, pathway: 'A1', selectedHistoryId: '', nsaids: [], others: [], unknowns: [], learnedMedications: [], physio: null, imagingValidation: '', horosImages: [], horosNextId: 0, doctors: [], editingDoctor: null, draftId: '', draftRevision: null, sourceKind: '', sourceId: '', sourceOrigin: '', sourcePatientName: '', sourceNameConfirmed: false, sourcePage: 1, draftPage: 1, requestToken: 0, correctionOfId: '', correctionExpectedRevision: null, selectedOutcome: null, selectedOutcomeId: '', outcomePage: 1, outcomeToken: 0, outcomeSelectionToken: 0, outcomeContext: '' };

  async function api(path, options = {}) {
    const response = await fetch(`/clinical/clinic-utilities/rf${path}`, { credentials: 'same-origin', ...options });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try { const body = await response.json(); detail = body.detail || detail; } catch (_) {}
      const error = new Error(detail);
      error.status = response.status;
      throw error;
    }
    return response;
  }

  async function loadContract() {
    // A directory change must not rebuild the patient's RF draft or choose
    // the physician that happened to be edited/added.
    const initialLoad = !state.contract;
    const previousIssuer = initialLoad ? "papachristou" : $("doctorSelect").value;
    try {
      const response = await api('/api/contract');
      state.contract = await response.json();
      if (initialLoad) populateContract();
      else populateDoctorOptions(previousIssuer);
      refreshDoctorReadiness();
      updateSummary();
      // Losing an active issuer requires an explicit user selection, not fallback.
      return !initialLoad && !!previousIssuer && $("doctorSelect").value !== previousIssuer;
    } catch (error) {
      $('configStatus').textContent = 'σφάλμα φόρτωσης';
      $('configStatus').className = 'status-pill error';
      showError(error.message);
      return null;
    }
  }

  function refreshDoctorReadiness() {
    const configured = Boolean(state.contract?.doctors?.[$("doctorSelect").value]?.configured);
    const productsReady = Object.values(state.contract?.products || {}).every(item => item.configured);
    const ready = configured && productsReady;
    $('configStatus').textContent = ready ? 'RF v2 έτοιμο' :
      ($("doctorSelect").value ? 'Απαιτείται server config' : 'Επίλεξε ιατρό αίτησης');
    $('configStatus').className = 'status-pill ' + (ready ? 'ok' : 'warn');
  }

  function populateDoctorOptions(chosen) {
    const doctorSelect = $("doctorSelect");
    doctorSelect.replaceChildren();
    doctorSelect.add(new Option("Επίλεξε ιατρό…", ""));
    Object.entries(state.contract?.doctors || {}).forEach(([key, item]) => {
      if (item.configured) doctorSelect.add(new Option(item.label, key));
    });
    // Never substitute the default when the previously chosen issuer was removed.
    doctorSelect.value = chosen && state.contract?.doctors?.[chosen]?.configured ? chosen : "";
    doctorSelect.dataset.currentDoctor = doctorSelect.value;
  }

  function populateContract() {
    const c = state.contract;
    populateDoctorOptions(c.doctors?.papachristou?.configured ? "papachristou" : "");
    Object.entries(c.products).forEach(([key, item]) => {
      const option = new Option(item.label + (item.configured ? '' : ' — μη ρυθμισμένο'), key);
      option.disabled = !item.configured;
      $('productSelect').add(option);
    });
    Object.entries(c.indications).forEach(([key, item]) => $('indicationSelect').add(new Option(item.label, key)));
    const sideSelect = $('lateralitySelect');
    sideSelect.replaceChildren(new Option('Επιλέξτε…', ''));
    Object.entries(c.laterality).forEach(([key, label]) => sideSelect.add(new Option(label, key)));
    sideSelect.value = '';
    $('reasonOptions').replaceChildren(...Object.entries(c.rf_reasons).map(([key, label]) => checkboxItem(key, label)));
  }


  function doctorStatus(message, error = false) {
    $("doctorDirectoryStatus").textContent = message;
    $("doctorDirectoryStatus").className = error ? "validation-panel error" : "muted";
  }

  function resetDoctorForm() {
    $("doctorForm").reset();
    $("doctorEditingKey").value = "";
    state.editingDoctor = null;
    $("doctorSave").textContent = "Προσθήκη ιατρού";
  }

  function editDoctor(row) {
    state.editingDoctor = row;
    $("doctorEditingKey").value = row.key;
    for (const field of ["name", "gesy_code", "specialty", "medical_center", "phone", "email"]) {
      $("doctorForm").elements.namedItem(field).value = row[field];
    }
    $("doctorSave").textContent = "Αποθήκευση αλλαγών";
  }

  function renderDoctorDirectory() {
    const list = $("doctorDirectoryList");
    list.replaceChildren();
    for (const row of state.doctors) {
      const item = document.createElement("div");
      item.className = "doctor-directory-row" + (row.active ? "" : " is-inactive");
      const desc = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = row.name;
      const details = document.createElement("div");
      details.className = "muted";
      details.textContent = row.gesy_code + (row.active ? "" : " · Εκτός επιλογών");
      desc.append(title, details);
      const actions = document.createElement("div");
      actions.className = "doctor-directory-actions";
      const edit = document.createElement("button");
      edit.type = "button"; edit.className = "btn secondary small"; edit.textContent = "Επεξεργασία";
      edit.addEventListener("click", () => editDoctor(row));
      const toggle = document.createElement("button");
      toggle.type = "button"; toggle.className = "btn ghost small";
      toggle.textContent = row.active ? "Αφαίρεση" : "Επαναφορά";
      toggle.addEventListener("click", async () => {
        try {
          await api("/api/doctors/" + encodeURIComponent(row.key) + "/status", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({revision: row.revision, active: !row.active}),
          });
          resetDoctorForm();
          await loadDoctorDirectory();
          const lostIssuer = await loadContract();
          doctorStatus(lostIssuer
            ? "Ο επιλεγμένος ιατρός αφαιρέθηκε. Επίλεξε ρητά άλλον ιατρό για αυτή την αίτηση."
            : (row.active ? "Ο ιατρός αφαιρέθηκε από τις διαθέσιμες επιλογές." : "Ο ιατρός επανενεργοποιήθηκε."));
        } catch (error) {
          doctorStatus(error.message, true);
          await loadDoctorDirectory();
        }
      });
      actions.append(edit, toggle);
      item.append(desc, actions);
      list.append(item);
    }
  }

  async function loadDoctorDirectory() {
    try {
      const response = await api("/api/doctors");
      state.doctors = (await response.json()).doctors;
      renderDoctorDirectory();
    } catch (error) { doctorStatus(error.message, true); }
  }

  async function saveDoctor(event) {
    event.preventDefault();
    const fields = ["name", "gesy_code", "specialty", "medical_center", "phone", "email"];
    const form = $("doctorForm");
    const data = {};
    for (const field of fields) data[field] = form.elements.namedItem(field).value.trim();
    const previous = state.editingDoctor;
    if (previous) data.revision = previous.revision;
    const path = previous ? "/api/doctors/" + encodeURIComponent(previous.key) : "/api/doctors";
    try {
      $("doctorSave").disabled = true;
      const response = await api(path, {
        method: previous ? "PUT" : "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data),
      });
      await response.json();
      resetDoctorForm();
      await loadDoctorDirectory();
      const lostIssuer = await loadContract();
      doctorStatus(lostIssuer
        ? "Ο προηγούμενος εκδότης δεν είναι διαθέσιμος. Επίλεξε ρητά άλλον ιατρό."
        : (previous ? "Τα στοιχεία του ιατρού ενημερώθηκαν." : "Ο ιατρός προστέθηκε. Επίλεξε τον αν θέλεις να εκδώσει αυτή την αίτηση."));
    } catch (error) {
      doctorStatus(error.message, true);
      await loadDoctorDirectory();
    } finally { $("doctorSave").disabled = false; }
  }

  function checkboxItem(value, label) {
    const wrap = document.createElement('label');
    wrap.className = 'check-item';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = value;
    input.className = 'rf-reason';
    wrap.append(input, document.createTextNode(label));
    return wrap;
  }

  function setPathway(pathway) {
    if (state.pathway !== pathway) {
      state.sourceKind = '';
      state.sourceId = '';
      state.sourceOrigin = '';
      state.sourcePatientName = '';
      state.sourceNameConfirmed = false;
      $('rfSourceLoaded').hidden = true;
      $('rfNameConflict').hidden = true;
    }
    state.pathway = pathway;
    document.querySelectorAll('.segment').forEach(btn => btn.classList.toggle('active', btn.dataset.pathway === pathway));
    $('a1Flow').hidden = pathway !== 'A1';
    $('a2SourceFlow').hidden = pathway !== 'A2';
    $('a2Flow').hidden = pathway !== 'A2';
    $('rfPatientStep').textContent = pathway === 'A2' ? '3' : '2';
    $('rfIndicationStep').textContent = pathway === 'A2' ? '4' : '3';
    indicationChanged();
    updateSummary();
  }

  function indicationChanged() {
    const code = $('indicationSelect').value;
    const item = state.contract?.indications?.[code];
    const isOther = code.startsWith('OTHER_');
    $('otherAreaWrap').hidden = !isOther;
    $('otherDiagnosisWrap').hidden = !isOther;
    if (code === 'OTHER_LATERAL_EPICONDYLITIS') {
      $('otherArea').value = 'Αγκώνας';
      $('otherDiagnosis').value = 'Έξω επικονδυλίτιδα';
    }
    if (code === 'OTHER_DEQUERVAIN') {
      $('otherArea').value = 'Καρπός';
      $('otherDiagnosis').value = 'De Quervain';
    }
    if (code === 'OTHER_CUSTOM') {
      $('otherArea').value = '';
      $('otherDiagnosis').value = '';
    }
    $('interventionCard').hidden = !item?.requires_intervention || state.pathway !== 'A1';
    if (item?.site_key === 'si') {
      $('interventionTitle').textContent = 'Ιερολαγόνια — προηγούμενη έγχυση';
      $('interventionHelp').textContent = 'Σημείο εφαρμογής, ημερομηνία και VAS πριν/μετά την έγχυση κορτιζόνης/τοπικού αναισθητικού.';
    } else if (item?.site_key === 'hip') {
      $('interventionTitle').textContent = 'Ισχίο — διαγνωστικό block';
      $('interventionHelp').textContent = 'Το νέο επίσημο έντυπο απαιτεί στοιχεία διαγνωστικού block για RF στο ισχίο.';
    }
    applySuggestedLocation();
    state.selectedHistoryId = '';
    $('historyChoices').replaceChildren();
    updateSummary();
  }

  function applySuggestedLocation() {
    const code = $('indicationSelect').value;
    const side = $('lateralitySelect').value;
    const item = state.contract?.indications?.[code];
    let suggested = '';
    if (item && (side === 'left' || side === 'right')) {
      suggested = item.location_labels?.[side] || '';
      if (!suggested && code === 'OTHER_CUSTOM') {
        const area = $('otherArea').value.trim();
        if (area) suggested = `${side === 'left' ? 'Αριστερά' : 'Δεξιά'} — ${area}`;
      }
    }
    const exact = $('exactLocation');
    exact.value = suggested;
    exact.readOnly = item?.location_mode === 'derived';
    exact.title = exact.readOnly ? 'Η εντόπιση προκύπτει αυτόματα από ένδειξη + πλευρά.' : '';
    updateSummary();
  }

  async function parseMedication() {
    const text = $('medicationText').value.trim();
    if (!text) {
      state.nsaids = [];
      state.others = [];
      state.unknowns = [];
      renderTrials();
      $('medicationStatus').textContent = 'Δεν έχει γίνει ανάλυση.';
      return;
    }
    try {
      $('medicationStatus').textContent = 'Ανάλυση…';
      const response = await api('/api/parse-medications', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({text})
      });
      const result = await response.json();
      state.nsaids = result.nsaid_candidates || [];
      state.others = result.other_candidates || [];
      state.unknowns = result.unrecognized_candidates || [];
      renderTrials();
      const recognized = state.nsaids.length + state.others.length;
      const unknown = state.unknowns.length;
      $('medicationStatus').textContent = unknown
        ? `Αναγνωρίστηκαν ${recognized} · ${unknown} χρειάζονται ταξινόμηση.`
        : `Αναγνωρίστηκαν ${recognized} γραμμές · επιλέγονται έως 3 ΜΣΑΦ + έως 3 άλλα αναλγητικά.`;
    } catch (error) {
      showError(error.message);
    }
  }

  async function saveMedicationAlias(alias, category) {
    const name = String(alias || '').trim();
    if (!name) throw new Error('Συμπλήρωσε πρώτα το όνομα/alias του φαρμάκου.');
    const response = await api('/api/medication-dictionary', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({alias: name, display_name: name, category, active_ingredient: ''})
    });
    const result = await response.json();
    return result.entry;
  }

  function addManualTrial(category) {
    const items = category === 'nsaid' ? state.nsaids : state.others;
    const selectedCount = items.filter(item => item.auto_selected).length;
    items.push({
      source_text: '',
      category,
      canonical_key: '',
      drug_name: '',
      dose: '',
      duration: '',
      warning: '',
      auto_selected: selectedCount < 3,
      learned: false,
      manual: true
    });
    renderTrials();
  }

  async function learnManualTrial(item, category) {
    try {
      const entry = await saveMedicationAlias(item.drug_name, category);
      item.learned = true;
      item.manual = false;
      item.canonical_key = `learned:${entry.normalized_alias}`;
      showOk(`${entry.display_name} αποθηκεύτηκε ως ${category === 'nsaid' ? 'ΜΣΑΦ' : 'άλλο αναλγητικό'}.`);
      renderTrials();
      await loadMedicationDictionary();
    } catch (error) {
      showError(error.message);
    }
  }

  async function classifyUnknown(item, category, aliasInput) {
    try {
      const entry = await saveMedicationAlias(aliasInput.value, category);
      showOk(`${entry.display_name} μαθεύτηκε ως ${category === 'nsaid' ? 'ΜΣΑΦ' : 'άλλο αναλγητικό'}.`);
      await parseMedication();
      await loadMedicationDictionary();
    } catch (error) {
      showError(error.message);
    }
  }

  function renderTrials() {
    renderTrialGroup('nsaidRows', state.nsaids, 'nsaid');
    renderTrialGroup('otherRows', state.others, 'other');
    renderUnknownMedications();
  }

  function renderTrialGroup(containerId, items, category) {
    const container = $(containerId);
    container.replaceChildren();
    if (!items.length) {
      const note = document.createElement('div');
      note.className = 'muted';
      note.textContent = 'Δεν αναγνωρίστηκε σχετικό φάρμακο.';
      container.append(note);
      return;
    }
    items.forEach(item => {
      const row = document.createElement('div');
      row.className = 'trial';
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.checked = !!item.auto_selected;
      const drug = document.createElement('input');
      drug.value = item.drug_name || '';
      drug.placeholder = 'Φάρμακο';
      const dose = document.createElement('input');
      dose.value = item.dose || '';
      dose.placeholder = 'Δόση';
      dose.className = 'dose';
      const duration = document.createElement('input');
      duration.value = item.duration || '';
      duration.placeholder = 'Διάρκεια';
      duration.className = 'duration';
      check.addEventListener('change', () => item.auto_selected = check.checked);
      drug.addEventListener('input', () => item.drug_name = drug.value);
      dose.addEventListener('input', () => item.dose = dose.value);
      duration.addEventListener('input', () => item.duration = duration.value);
      row.title = item.source_text || '';
      row.append(check, drug, dose, duration);
      container.append(row);

      if (item.learned) {
        const badge = document.createElement('div');
        badge.className = 'learned-badge';
        badge.textContent = '✓ learned · clinician-confirmed';
        container.append(badge);
      }
      if (item.manual && !item.learned) {
        const actions = document.createElement('div');
        actions.className = 'trial-actions';
        const learn = document.createElement('button');
        learn.type = 'button';
        learn.className = 'btn ghost small';
        learn.textContent = 'Αποθήκευση στο λεξικό';
        learn.addEventListener('click', () => learnManualTrial(item, category));
        actions.append(learn);
        container.append(actions);
      }
      if (item.warning) {
        const warning = document.createElement('div');
        warning.className = 'trial-warning';
        warning.textContent = item.warning;
        container.append(warning);
      }
    });
  }

  function renderUnknownMedications() {
    const panel = $('unknownMedicationPanel');
    const container = $('unknownMedicationRows');
    container.replaceChildren();
    panel.hidden = !state.unknowns.length;
    state.unknowns.forEach(item => {
      const row = document.createElement('div');
      row.className = 'unknown-medication';
      const source = document.createElement('div');
      source.className = 'unknown-source';
      source.textContent = item.source_text || '';
      const alias = document.createElement('input');
      alias.value = item.suggested_name || '';
      alias.placeholder = 'Όνομα / εμπορικό alias';
      const actions = document.createElement('div');
      actions.className = 'unknown-actions';
      const nsaid = document.createElement('button');
      nsaid.type = 'button';
      nsaid.className = 'btn secondary small';
      nsaid.textContent = 'ΜΣΑΦ + μάθηση';
      nsaid.addEventListener('click', () => classifyUnknown(item, 'nsaid', alias));
      const other = document.createElement('button');
      other.type = 'button';
      other.className = 'btn secondary small';
      other.textContent = 'Άλλο + μάθηση';
      other.addEventListener('click', () => classifyUnknown(item, 'other', alias));
      actions.append(nsaid, other);
      row.append(source, alias, actions);
      container.append(row);
    });
  }

  async function loadMedicationDictionary() {
    try {
      const response = await api('/api/medication-dictionary');
      const result = await response.json();
      state.learnedMedications = result.entries || [];
      renderMedicationDictionary();
    } catch (error) {
      showError(error.message);
    }
  }

  async function reclassifyMedication(entry, category) {
    try {
      await saveMedicationAlias(entry.alias, category);
      await loadMedicationDictionary();
      await parseMedication();
    } catch (error) {
      showError(error.message);
    }
  }

  async function deleteMedicationAlias(entry) {
    try {
      await api(`/api/medication-dictionary/${encodeURIComponent(entry.id)}`, {method:'DELETE'});
      await loadMedicationDictionary();
      await parseMedication();
    } catch (error) {
      showError(error.message);
    }
  }

  function renderMedicationDictionary() {
    const container = $('learnedMedicationRows');
    container.replaceChildren();
    if (!state.learnedMedications.length) {
      const empty = document.createElement('div');
      empty.className = 'muted';
      empty.textContent = 'Δεν υπάρχουν ακόμη εκμαθημένα φάρμακα.';
      container.append(empty);
      return;
    }
    state.learnedMedications.forEach(entry => {
      const row = document.createElement('div');
      row.className = 'dictionary-row';
      const label = document.createElement('div');
      const strong = document.createElement('strong');
      strong.textContent = entry.display_name || entry.alias;
      const meta = document.createElement('span');
      meta.className = 'muted';
      meta.textContent = entry.category === 'nsaid' ? 'ΜΣΑΦ' : 'Άλλο';
      label.append(strong, document.createTextNode(' · '), meta);
      const actions = document.createElement('div');
      actions.className = 'dictionary-actions';
      const asNsaid = document.createElement('button');
      asNsaid.type = 'button';
      asNsaid.className = 'btn ghost small';
      asNsaid.textContent = 'ΜΣΑΦ';
      asNsaid.disabled = entry.category === 'nsaid';
      asNsaid.addEventListener('click', () => reclassifyMedication(entry, 'nsaid'));
      const asOther = document.createElement('button');
      asOther.type = 'button';
      asOther.className = 'btn ghost small';
      asOther.textContent = 'Άλλο';
      asOther.disabled = entry.category === 'other';
      asOther.addEventListener('click', () => reclassifyMedication(entry, 'other'));
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'btn ghost small';
      remove.textContent = 'Διαγραφή';
      remove.addEventListener('click', () => deleteMedicationAlias(entry));
      actions.append(asNsaid, asOther, remove);
      row.append(label, actions);
      container.append(row);
    });
  }

  function selectedTrials(items) {
    return items
      .filter(x => x.auto_selected)
      .slice(0, 3)
      .map(x => ({
        source_text: x.source_text || '',
        drug_name: x.drug_name || '',
        dose: x.dose || '',
        duration: x.duration || ''
      }));
  }

  async function parsePhysio() {
    try {
      const response = await api('/api/parse-physio', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({text: $('physioDates').value})
      });
      state.physio = await response.json();
      $('physioStart').textContent = state.physio.start_date || '—';
      $('physioEnd').textContent = state.physio.end_date || '—';
      $('physioCount').textContent = state.physio.treatment_count || 0;
      const bad = state.physio.invalid_or_ambiguous_tokens || [];
      $('physioWarning').hidden = !bad.length;
      $('physioWarning').textContent = bad.length ? `Χρειάζονται διόρθωση: ${bad.join(', ')}` : '';
    } catch (error) {
      showError(error.message);
    }
  }

  async function lookupHistory(preferId = '') {
    const item = state.contract?.indications?.[$('indicationSelect').value];
    if (!$('identityNumber').value.trim()) {
      showError('Συμπλήρωσε μόνο την ταυτότητα ή αναζήτησε πρώτα παλιά RF αίτηση.');
      return;
    }
    try {
      $('historyStatus').textContent = 'Αναζήτηση…';
      const response = await api('/api/history', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          identity_number: $('identityNumber').value.trim(),
          site_key: item?.site_key || '',
          laterality: ['left','right'].includes($('lateralitySelect').value) ? $('lateralitySelect').value : null
        })
      });
      const result = await response.json();
      renderHistory(result.procedures || [], preferId);
      $('historyStatus').textContent = result.found ? `${result.procedures.length} εφαρμογή/ές` : 'Δεν βρέθηκε ιστορικό — χρησιμοποίησε legacy καταχώρηση.';
    } catch (error) {
      showError(error.message);
    }
  }

  function renderHistory(rows, preferId = '') {
    const container = $('historyChoices');
    container.replaceChildren();
    state.selectedHistoryId = '';
    rows.forEach(row => {
      const label = document.createElement('label');
      label.className = 'history-option';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'history';
      radio.value = row.procedure_history_id;
      if (preferId && row.procedure_history_id === preferId) {
        radio.checked = true;
        state.selectedHistoryId = preferId;
      }
      radio.addEventListener('change', () => {
        state.selectedHistoryId = row.procedure_history_id;
        $('legacyDetails').open = false;
      });
      const text = document.createElement('div');
      text.innerHTML = `<strong>${escapeHtml(row.actual_procedure_date)}</strong> · ${escapeHtml(row.exact_location)}<br><span class="muted">VAS ${row.vas_before} → ${row.vas_after} · follow-up ${escapeHtml(row.last_followup_date)}: ${row.last_followup_vas}/10 · ${escapeHtml(row.provenance)}</span>`;
      label.append(radio, text);
      container.append(label);
    });
  }

  function escapeHtml(value) {
    return String(value || '').replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c] || c));
  }

  function adverseValues() {
    return [...$('adverseRows').querySelectorAll('.adverse')]
      .map(row => ({
        treatment: row.querySelector('[data-kind=treatment]').value.trim(),
        effect: row.querySelector('[data-kind=effect]').value.trim()
      }))
      .filter(x => x.treatment && x.effect);
  }

  function addAdverse() {
    const row = document.createElement('div');
    row.className = 'adverse';
    row.innerHTML = '<input data-kind="treatment" placeholder="Φάρμακο / θεραπεία"><input data-kind="effect" placeholder="Ανεπιθύμητη ενέργεια"><button type="button" class="btn ghost small">×</button>';
    row.querySelector('button').onclick = () => row.remove();
    $('adverseRows').append(row);
  }

  function checkedReasons() {
    return [...document.querySelectorAll('.rf-reason:checked')].map(x => x.value);
  }

  function numberOrNull(id) {
    const raw = $(id).value.trim();
    if (raw === '') return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  }

  function buildDraft() {
    const code = $('indicationSelect').value;
    const indication = state.contract?.indications?.[code];
    const draft = {
      pathway: state.pathway,
      doctor_key: $('doctorSelect').value,
      patient_name: $('patientName').value.trim(),
      identity_number: $('identityNumber').value.trim(),
      gesy_number: $('gesyNumber').value.trim(),
      age: numberOrNull('age'),
      product_key: $('productSelect').value,
      indication_code: code,
      laterality: $('lateralitySelect').value,
      exact_location: $('exactLocation').value.trim(),
      other_area: $('otherArea').value.trim(),
      other_diagnosis: $('otherDiagnosis').value.trim(),
      additional_notes: $('additionalNotes').value.trim(),
      imaging_review_confirmed: $('imagingConfirm').checked,
      correction_of_application_id: state.correctionOfId,
      correction_source_expected_revision: state.correctionOfId ? state.correctionExpectedRevision : null
    };

    if (state.pathway === 'A1') {
      Object.assign(draft, {
        rf_reason_codes: checkedReasons(),
        rf_reason_other: $('reasonOther').value.trim(),
        pain_onset_date: $('painOnsetDate').value,
        pain_onset_vas: numberOrNull('painOnsetVas'),
        last_assessment_date: $('lastAssessmentDate').value,
        last_assessment_vas: numberOrNull('lastAssessmentVas'),
        full_medication_text: $('medicationText').value,
        nsaid_trials: selectedTrials(state.nsaids),
        other_analgesic_trials: selectedTrials(state.others),
        adverse_effects: adverseValues(),
        physio_dates_text: $('physioDates').value
      });
      if (indication?.requires_intervention) {
        draft.intervention = {
          site: $('interventionSite').value.trim(),
          date: $('interventionDate').value,
          vas_before: numberOrNull('interventionVasBefore'),
          vas_after: numberOrNull('interventionVasAfter')
        };
      }
    } else if (state.selectedHistoryId) {
      draft.procedure_history_id = state.selectedHistoryId;
    } else {
      draft.legacy_history = {
        actual_procedure_date: $('legacyProcedureDate').value,
        vas_before: numberOrNull('legacyVasBefore'),
        vas_after: numberOrNull('legacyVasAfter'),
        last_followup_date: $('legacyFollowupDate').value,
        last_followup_vas: numberOrNull('legacyFollowupVas')
      };
    }
    if (state.pathway === 'A2') {
      draft.a2_current_followup_date = $('rfCurrentFollowupDate').value;
      draft.a2_current_followup_vas = numberOrNull('rfCurrentFollowupVas');
      draft.legacy_confirmed = $('rfActualConfirm').checked;
      draft.source_origin = state.sourceOrigin;
      draft.source_kind = state.sourceKind;
      draft.source_id = state.sourceId;
      draft.source_name_confirmed = state.sourceNameConfirmed;
    }
    return draft;
  }

  async function validateImaging() {
    const file = $('imagingReport').files[0];
    const status = $('imagingStatus');
    const confirmWrap = $('imagingConfirmWrap');
    const confirm = $('imagingConfirm');
    state.imagingValidation = '';
    confirm.checked = false;
    confirmWrap.hidden = true;
    if (!file) {
      status.hidden = true;
      status.textContent = '';
      return;
    }
    status.hidden = false;
    status.className = 'validation-panel';
    status.textContent = 'Έλεγχος τύπου εγγράφου…';
    const form = new FormData();
    form.append('imaging_report', file, file.name);
    try {
      const response = await api('/api/validate-imaging', {method:'POST', body:form});
      const result = await response.json();
      state.imagingValidation = result.status;
      status.textContent = result.message;
      status.className = `validation-panel ${result.status === 'imaging_supported' ? 'ok' : result.status === 'clearly_non_imaging' ? 'error' : ''}`;
      confirmWrap.hidden = !result.requires_confirmation;
    } catch (error) {
      state.imagingValidation = 'validation_error';
      status.className = 'validation-panel error';
      status.textContent = error.message;
    }
  }

  const MAX_HOROS_IMAGES = 10;
  const MAX_HOROS_IMAGE_BYTES = 8 * 1024 * 1024;
  const MAX_HOROS_TOTAL_BYTES = 40 * 1024 * 1024;

  function renderHorosImages(message = "") {
    const grid = $("horosImageGrid");
    grid.replaceChildren();
    state.horosImages.forEach(item => {
      const figure = document.createElement("figure");
      figure.className = "horos-item";
      const image = document.createElement("img");
      image.className = "horos-thumb";
      image.src = item.url;
      image.alt = "Προεπισκόπηση " + item.file.name;
      const caption = document.createElement("figcaption");
      caption.textContent = item.file.name;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "btn ghost small";
      remove.textContent = "Αφαίρεση";
      remove.setAttribute("aria-label", "Αφαίρεση " + item.file.name);
      remove.addEventListener("click", () => {
        URL.revokeObjectURL(item.url);
        state.horosImages = state.horosImages.filter(x => x.id !== item.id);
        renderHorosImages();
      });
      figure.append(image, caption, remove);
      grid.append(figure);
    });
    const status = $("horosImageStatus");
    status.className = message ? "validation-panel error" : "muted";
    const count = state.horosImages.length;
    status.textContent = message || (count
      ? `${count} ${count === 1 ? "εικόνα έτοιμη" : "εικόνες έτοιμες"} για επισύναψη.`
      : "Δεν έχουν προστεθεί εικόνες.");
  }

  function addHorosImages() {
    const picker = $("horosJpegs");
    const selected = Array.from(picker.files || []);
    picker.value = "";  // allow reselection of the same file
    let message = "";
    for (const file of selected) {
      const total = state.horosImages.reduce((sum, item) => sum + item.file.size, 0);
      if (state.horosImages.length >= MAX_HOROS_IMAGES) { message = "Επιτρέπονται έως 10 εικόνες Horos."; break; }
      if (!/\.jpe?g$/i.test(file.name) || (file.type && !["image/jpeg", "image/pjpeg"].includes(file.type))) {
        message = "Επιτρέπονται μόνο αρχεία JPEG (.jpg/.jpeg)."; continue;
      }
      if (file.size > MAX_HOROS_IMAGE_BYTES || total + file.size > MAX_HOROS_TOTAL_BYTES) {
        message = "Υπέρβαση ορίου 8 MB ανά εικόνα / 40 MB συνολικά."; continue;
      }
      state.horosImages.push({file, url: URL.createObjectURL(file), id: ++state.horosNextId});
    }
    renderHorosImages(message);
  }

  function releaseHorosPreviews() {
    state.horosImages.forEach(item => URL.revokeObjectURL(item.url));
    state.horosImages = [];
  }


  // The same RF protected endpoint owns all retrieval. Never cache PHI in
  // browser storage, and never treat a previous application as a performed RF.
  const rfFields = {
    patient_name: 'patientName', identity_number: 'identityNumber',
    gesy_number: 'gesyNumber', age: 'age', product_key: 'productSelect',
    indication_code: 'indicationSelect', laterality: 'lateralitySelect',
    exact_location: 'exactLocation', other_area: 'otherArea',
    other_diagnosis: 'otherDiagnosis', additional_notes: 'additionalNotes',
    rf_reason_other: 'reasonOther', pain_onset_date: 'painOnsetDate',
    pain_onset_vas: 'painOnsetVas', last_assessment_date: 'lastAssessmentDate',
    last_assessment_vas: 'lastAssessmentVas',
    full_medication_text: 'medicationText', physio_dates_text: 'physioDates',
    a2_current_followup_date: 'rfCurrentFollowupDate',
    a2_current_followup_vas: 'rfCurrentFollowupVas'
  };


  function sameRFPersonName(a, b) {
    const norm = value => String(value || '').normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '').toLocaleLowerCase('el')
      .replace(/ς/g, 'σ').trim().replace(/\s+/g, ' ');
    return norm(a) === norm(b);
  }

  function sourceNameMismatch() {
    return Boolean((state.sourceKind || state.sourceOrigin === 'pdf') &&
      state.sourcePatientName && $('patientName').value.trim() &&
      !sameRFPersonName(state.sourcePatientName, $('patientName').value));
  }

  function unresolvedSourceNameConflict() {
    return sourceNameMismatch() && !state.sourceNameConfirmed;
  }

  function renderSourceNameConflict() {
    const mismatch = sourceNameMismatch();
    $('rfNameConflict').hidden = !mismatch;
    if (!mismatch) return;
    $('rfNameConflictText').textContent = state.sourceNameConfirmed
      ? 'Η διαφορά ονόματος επιβεβαιώθηκε ρητά για την επιλεγμένη ταυτότητα.'
      : 'Η προηγούμενη RF αναφέρει «' + state.sourcePatientName +
        '», ενώ η τρέχουσα «' + $('patientName').value.trim() +
        '». Επίλεξε ποιο όνομα είναι σωστό πριν την έκδοση.';
    $('rfConfirmCurrentNameBtn').disabled = state.sourceNameConfirmed;
  }

  function resolveSourceName(choice) {
    if (!sourceNameMismatch()) return;
    if (choice === 'previous') {
      $('patientName').value = state.sourcePatientName;
      state.sourceNameConfirmed = false;
    } else if (choice === 'current') {
      state.sourceNameConfirmed = true;
    }
    renderSourceNameConflict();
    updateSummary();
  }

  function invalidateSourceNameConfirmation() {
    state.sourceNameConfirmed = false;
    renderSourceNameConflict();
  }

  function populateRfFields(fields, overwrite = false) {
    const conflicts = [];
    if (!fields || !state.contract) return conflicts;
    const desiredIdentity = String(fields.identity_number || '').trim();
    const existingIdentity = $('identityNumber').value.trim();
    if (!overwrite && existingIdentity && desiredIdentity &&
        existingIdentity.replace(/[\W_]/g,'').toUpperCase() !== desiredIdentity.replace(/[\W_]/g,'').toUpperCase()) {
      throw new Error('Η παλιά αίτηση αφορά άλλη ταυτότητα. Καθάρισε ή ολοκλήρωσε την τρέχουσα αίτηση πρώτα.');
    }
    // Order matters: the indication and side update the native derived-location
    // controls; only then can the historical exact location be used.
    if (fields.indication_code && state.contract.indications[fields.indication_code]) {
      const el = $('indicationSelect');
      if (overwrite || !el.value) { el.value = fields.indication_code; indicationChanged(); }
      else if (el.value !== fields.indication_code) conflicts.push('ένδειξη');
    }
    if (fields.laterality && ['left','right'].includes(fields.laterality)) {
      const el = $('lateralitySelect');
      if (overwrite || !el.value) { el.value = fields.laterality; applySuggestedLocation(); }
      else if (el.value !== fields.laterality) conflicts.push('πλευρά');
    }
    for (const [field, id] of Object.entries(rfFields)) {
      if (!(field in fields) || ['indication_code','laterality'].includes(field)) continue;
      const val = fields[field];
      if (val === null || val === undefined || val === '') continue;
      if ((field === 'product_key' && !state.contract.products?.[val]) ||
          (field === 'age' && (Number(val) < 1 || Number(val) > 130))) continue;
      const el = $(id);
      if (!el) continue;
      const value = String(val);
      if (overwrite || !el.value.trim() || (field === 'exact_location' && !el.value.trim())) {
        el.value = value;
      } else if (el.value.trim() !== value) conflicts.push(field);
    }
    if (fields.rf_reason_codes && Array.isArray(fields.rf_reason_codes)) {
      document.querySelectorAll('.rf-reason').forEach(c => {
        if (overwrite || !c.checked) c.checked = fields.rf_reason_codes.includes(c.value);
      });
    }
    updateSummary();
    return conflicts;
  }

  function sourceNotice(text) {
    $('rfSourceStatus').textContent = text;
  }

  function renderReuseCard(container, card, onSelect, action) {
    const item = document.createElement('div');
    item.className = 'rf-source-item';
    const head = document.createElement('strong');
    head.textContent = card.patient_name || 'Χωρίς καταχωρισμένο όνομα';
    const sub = document.createElement('div');
    sub.className = 'muted';
    sub.textContent = [card.date || card.updated_at || '',
      card.identity_number || '', card.gesy_number || '',
      card.indication_code || '', card.exact_location || '',
      card.kind === 'procedure' ? 'Πραγματική εφαρμογή' :
      card.kind === 'application' ? 'Προηγούμενη αίτηση — όχι εφαρμογή' :
      card.pathway || 'Πρόχειρο'].filter(Boolean).join(' · ');
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'btn secondary small';
    btn.textContent = action;
    btn.addEventListener('click', () => onSelect(card));
    item.append(head, sub, btn); container.append(item);
    return item;
  }

  async function searchRfSources(page = 1) {
    const query = $('rfSourceSearch').value.trim();
    if (query.length < 2) { sourceNotice('Γράψε μόνο ένα όνομα, ταυτότητα ή ΓεΣΥ (2+ χαρακτήρες).'); return; }
    const token = ++state.requestToken;
    sourceNotice('Αναζήτηση…');
    try {
      const response = await api('/api/reuse/search?query=' +
        encodeURIComponent(query) + '&page=' + page);
      const result = await response.json();
      if (token !== state.requestToken || query !== $('rfSourceSearch').value.trim()) return;
      const list = $('rfSourceList');
      if (page === 1) list.replaceChildren();
      for (const source of result.results || []) {
        renderReuseCard(list, source, loadRfSource, 'Φόρτωση στοιχείων');
      }
      state.sourcePage = page;
      $('rfSourceNextBtn').hidden = !result.has_more;
      sourceNotice(result.results?.length
        ? 'Επίλεξε ρητά την κατάλληλη παλιά αίτηση ή πραγματική εφαρμογή.'
        : 'Δεν βρέθηκαν διαθέσιμα στοιχεία. Μπορείς να ανεβάσεις παλιό RF PDF.');
    } catch (error) { showError(error.message); }
  }

  async function loadRfSource(source) {
    try {
      const result = await (await api('/api/reuse/' + encodeURIComponent(source.kind) +
        '/' + encodeURIComponent(source.id))).json();
      if (state.pathway !== 'A2') setPathway('A2');
      // Historical age is not the patient's current age.
      const previousFields = {...(result.fields || {})};
      delete previousFields.age;
      const conflicts = populateRfFields(previousFields);
      state.sourceKind = source.kind;
      state.sourceId = source.id;
      state.sourceOrigin = 'database';
      state.sourcePatientName = String(result.fields?.patient_name || '').trim();
      state.sourceNameConfirmed = false;
      renderSourceNameConflict();
      if (result.actual_procedure) {
        renderHistory([result.actual_procedure], result.actual_procedure.procedure_history_id);
        $('legacyDetails').open = false;
      } else if ($('identityNumber').value && $('indicationSelect').value &&
                 $('lateralitySelect').value) {
        await lookupHistory();
      }
      const text = (source.kind === 'application'
        ? 'Προηγούμενη αίτηση (δεν επιβεβαιώνει πραγματική εφαρμογή).'
        : 'Καταγεγραμμένη πραγματική εφαρμογή.') +
        (conflicts.length ? ' Δεν αντικαταστάθηκαν ήδη συμπληρωμένα πεδία: ' + conflicts.join(', ') + '.' : '');
      $('rfSourceLoaded').textContent = text;
      $('rfSourceLoaded').hidden = false;
      sourceNotice('Τα διαθέσιμα στοιχεία μεταφέρθηκαν. Έλεγξε όσα λείπουν ή έχουν αλλάξει.');
      showOk('Φορτώθηκαν τα διαθέσιμα RF στοιχεία· χωρίς αλλαγή ιατρού εκδότη.');
    } catch (error) { showError(error.message); }
  }

  async function importRfPdf() {
    const file = $('rfPreviousPdf').files[0];
    if (!file) return;
    const form = new FormData();
    form.append('previous_rf_pdf', file, file.name);
    sourceNotice('Ανάγνωση παλιού RF PDF…');
    try {
      const result = await (await api('/api/reuse/pdf', {method:'POST', body:form})).json();
      if (result.status === 'unrecognized') { sourceNotice(result.message); return; }
      if (state.pathway !== 'A2') setPathway('A2');
      // Historical age in a prior RF PDF is not automatically current.
      const previousFields = {...(result.fields || {})};
      delete previousFields.age;
      const conflicts = populateRfFields(previousFields);
      state.sourceKind = ''; state.sourceId = ''; state.sourceOrigin = 'pdf';
      state.sourcePatientName = String(result.fields?.patient_name || '').trim();
      state.sourceNameConfirmed = false;
      renderSourceNameConflict();
      state.selectedHistoryId = '';
      $('historyChoices').replaceChildren();
      $('rfActualConfirm').checked = false;
      const previous = result.fields?.previous_procedure_candidate;
      if (previous) {
        for (const [key, id] of Object.entries({
          actual_procedure_date: 'legacyProcedureDate', vas_before: 'legacyVasBefore',
          vas_after: 'legacyVasAfter', last_followup_date: 'legacyFollowupDate',
          last_followup_vas: 'legacyFollowupVas'
        })) {
          if (previous[key] !== undefined && !$(''+id).value) $(id).value = previous[key];
        }
        $('legacyDetails').open = true;
      }
      $('rfSourceLoaded').textContent = 'Πηγή: παλιό PDF (προσωρινά στοιχεία, όχι τεκμήριο εφαρμογής).' +
        (conflicts.length ? ' Έλεγξε τις διαφορές στα πεδία: ' + conflicts.join(', ') : '');
      $('rfSourceLoaded').hidden = false;
      sourceNotice(result.message + ' Επιβεβαίωσε την πραγματική εφαρμογή αν την καταχωρήσεις.');
    } catch (error) { showError(error.message); }
    finally { $('rfPreviousPdf').value = ''; }
  }

  function cleanTrial(item) {
    const allowed = ['drug_name','dose','duration','auto_selected','source_text',
      'manual','learned','warning','canonical_key'];
    return Object.fromEntries(allowed.filter(key => item[key] !== undefined)
      .map(key => [key, item[key]]));
  }


  // Issued RF and GeSY submission are independent; every state is
  // clinician-entered with explicit event date/provenance.
  const outcomeActionLabels = {
    submit: "Σημειώθηκε ως υποβληθείσα",
    approve: "Καταχώριση έγκρισης",
    reject: "Καταχώριση απόρριψης",
    schedule: "Προγραμματίστηκε",
    perform: "Πραγματοποιήθηκε",
    correct_decision: "Διόρθωση ημερομηνίας / λόγου απόφασης",
    correct_submission: "Διόρθωση ημερομηνίας υποβολής",
    claim: "Αναλαμβάνω παρακολούθηση",
    release: "Σταματώ την προσωπική παρακολούθηση",
  };

  const outcomeStages = {
    issued: "Έκδοση PDF — υποβολή άγνωστη",
    submitted: "Υποβλήθηκε — αναμονή απάντησης",
    approved: "Εγκρίθηκε",
    rejected: "Απορρίφθηκε",
    scheduled: "Προγραμματίστηκε",
    performed: "Πραγματοποιήθηκε",
  };

  function todayLocalRf() {
    const d = new Date();
    const pad = n => String(n).padStart(2, '0');
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
  }

  function normalizeRfIdentity(value) {
    return String(value || '').replace(/[^0-9A-Za-z]/g, '').toUpperCase();
  }
  function resetRfBuilderForCorrection() {
    // Call only after protected rejection + revision read, selection/auth checks,
    // patient identity check and explicit consent to discard unrelated draft data.
    for (const id of Object.values(rfFields)) $(id).value = '';
    for (const id of ['legacyProcedureDate','legacyVasBefore','legacyVasAfter',
      'legacyFollowupDate','legacyFollowupVas','rfCurrentFollowupDate',
      'rfCurrentFollowupVas','interventionSite','interventionDate',
      'interventionVasBefore','interventionVasAfter','medicationText','physioDates']) {
      const field = $(id); if (field) field.value = '';
    }
    document.querySelectorAll('.rf-reason').forEach(c => { c.checked = false; });
    state.nsaids = []; state.others = []; state.unknowns = []; state.physio = null;
    renderTrials();
    $('adverseRows').replaceChildren(); addAdverse();
    state.selectedHistoryId = '';
    $('historyChoices').replaceChildren();
    state.sourceKind = ''; state.sourceId = ''; state.sourceOrigin = '';
    state.sourcePatientName = ''; state.sourceNameConfirmed = false;
    $('rfSourceLoaded').hidden = true; $('rfNameConflict').hidden = true;
    state.draftId = ''; state.draftRevision = null;
    $('rfActiveDraftLabel').textContent = '';
    $('doctorSelect').value = '';
    refreshDoctorReadiness();
    $('rfActualConfirm').checked = false;
    $('imagingReport').value = ''; state.imagingValidation = '';
    $('imagingStatus').hidden = true; $('imagingConfirmWrap').hidden = true;
    $('imagingConfirm').checked = false;
    releaseHorosPreviews(); renderHorosImages();
  }

  function renderCorrectionContext() {
    const el = $('rfCorrectionContext');
    el.hidden = !state.correctionOfId;
    el.textContent = state.correctionOfId
      ? 'Διορθωτική αίτηση από καταχωρισμένη απόρριψη. Πρέπει να επιβεβαιώσεις ιατρό και στοιχεία, και να επισυνάψεις ξανά την απεικόνιση.'
      : '';
  }

  function outcomeAvailableActions(item) {
    const actions = [];
    if (item.stage === 'issued') actions.push('submit', 'approve', 'reject');
    if (item.stage === 'submitted') actions.push('approve', 'reject');
    if (item.stage === 'approved') actions.push('schedule', 'perform', 'correct_decision');
    if (item.stage === 'scheduled') actions.push('schedule', 'perform', 'correct_decision');
    if (item.stage === 'performed') actions.push('correct_decision');
    if (item.stage === 'rejected') actions.push('correct_decision');
    if (item.submitted_date) actions.push('correct_submission');
    actions.push(item.mine_to_follow ? 'release' : 'claim');
    return actions;
  }

  function rfOutcomeContext() {
    return JSON.stringify([$('rfOutcomeSearch').value.trim(), $('rfOutcomeScope').value]);
  }

  function invalidateRfOutcomeSelection() {
    ++state.outcomeSelectionToken;
    state.selectedOutcome = null;
    state.selectedOutcomeId = '';
    $('rfOutcomeSelected').hidden = true;
  }

  function outcomeActionChanged() {
    const action = $('rfOutcomeAction').value;
    const noDate = action === 'claim' || action === 'release';
    $('rfOutcomeDate').closest('label').hidden = noDate;
    $('rfOutcomeCorrectDecisionWrap').hidden = action !== 'correct_decision';
    if (action === 'correct_decision') {
      $('rfOutcomeCorrectDecision').value = state.selectedOutcome?.decision || 'approved';
    }
    $('rfOutcomeReasonWrap').hidden = action !== 'reject' &&
      !(action === 'correct_decision' && $('rfOutcomeCorrectDecision').value === 'rejected');
    $('rfOutcomePerformWrap').hidden = action !== 'perform';
    $('rfOutcomePerformConfirm').checked = false;
    if (!noDate) {
      $('rfOutcomeDate').value = action === 'schedule' ? ''
        : action === 'correct_decision' ? (state.selectedOutcome?.decision_date || todayLocalRf())
        : action === 'correct_submission' ? (state.selectedOutcome?.submitted_date || todayLocalRf())
        : todayLocalRf();
    } else $('rfOutcomeDate').value = '';
    $('rfOutcomeReason').value = action === 'correct_decision'
      ? (state.selectedOutcome?.rejection_reason || '') : '';
  }

  function renderSelectedOutcome(item) {
    state.selectedOutcome = item;
    state.selectedOutcomeId = item.id;
    $('rfOutcomeSelected').hidden = false;
    $('rfOutcomeTitle').textContent = (item.patient_name || 'Αίτηση RF') + ' · ' +
      (outcomeStages[item.stage] || item.stage);
    const reported = item.last_decision_report;
    const sourceLabels = {
      clinician: 'καταχώριση ιατρού', colleague: 'ενημέρωση συναδέλφου',
      gesy_notice: 'έγγραφο ΓεΣΥ', other: 'άλλη δηλωμένη πηγή',
    };
    const source = reported && sourceLabels[reported.reported_source] || 'άγνωστη';
    const recorded = reported?.recorded_at && !Number.isNaN(Date.parse(reported.recorded_at))
      ? new Date(reported.recorded_at).toLocaleString('el-GR') : 'άγνωστος';
    // Only existing audit data is shown; missing historical source/time remains unknown.
    // textContent prevents rejection reasons and any reported values becoming HTML.
    $('rfOutcomeFacts').textContent = [
      'Έκδοση: ' + item.issued_date,
      'Ιατρός: ' + (item.issuer_name || 'Μη τεκμηριωμένος'),
      'Υποβολή: ' + (item.submitted_date || 'μη καταχωρισμένη'),
      'Απόφαση: ' + (item.decision_date || 'άγνωστη'),
      item.decision ? 'Προέλευση απόφασης: ' + source : '',
      item.decision ? 'Ημερομηνία γεγονότος: ' + (reported?.event_date || item.decision_date || 'άγνωστη') : '',
      item.decision ? 'Καταχώριση στο Cockpit: ' + recorded : '',
      item.rejection_reason ? 'Απόρριψη: ' + item.rejection_reason : '',
      'Παρακολούθηση: ' + (item.mine_to_follow ? 'έχω αναλάβει' : 'χωρίς προσωπική ανάληψη'),
    ].filter(Boolean).join(' · ');
    const actions = $('rfOutcomeAction');
    actions.replaceChildren();
    for (const action of outcomeAvailableActions(item)) {
      actions.add(new Option(outcomeActionLabels[action], action));
    }
    $('rfOutcomeCorrection').hidden = item.stage !== 'rejected';
    outcomeActionChanged();
  }

  async function selectRfOutcome(card) {
    const token = ++state.outcomeSelectionToken;
    const authAtStart = tracking.authEpoch;
    const context = rfOutcomeContext();
    const selectedId = card.id;
    // A is no longer actionable while B's detail request is pending.
    state.selectedOutcome = null;
    state.selectedOutcomeId = selectedId;
    $('rfOutcomeSelected').hidden = true;
    try {
      const item = await (await api('/api/outcomes/' +
        encodeURIComponent(selectedId))).json();
      if (token !== state.outcomeSelectionToken ||
          state.selectedOutcomeId !== selectedId ||
          context !== rfOutcomeContext() || item.id !== selectedId) return;
      renderSelectedOutcome(item);
    } catch (error) {
      if (token !== state.outcomeSelectionToken || authAtStart !== tracking.authEpoch ||
          state.selectedOutcomeId !== selectedId || context !== rfOutcomeContext()) return;
      if ([401,403].includes(error.status)) protectedTrackingFailure(error,true);
      else showError(error.message);
    }
  }

  function outcomeCorrectionKindChanged() {
    const correction = $('rfOutcomeAction').value === 'correct_decision';
    $('rfOutcomeReasonWrap').hidden = !correction ||
      $('rfOutcomeCorrectDecision').value !== 'rejected';
    if ($('rfOutcomeReasonWrap').hidden) $('rfOutcomeReason').value = '';
  }

  async function loadRfOutcomes(page = 1) {
    const query = $('rfOutcomeSearch').value.trim();
    const scope = $('rfOutcomeScope').value;
    const context = rfOutcomeContext();
    if (context !== state.outcomeContext) {
      state.outcomeContext = context;
      invalidateRfOutcomeSelection();
    }
    const token = ++state.outcomeToken;
    try {
      const response = await api('/api/outcomes/search', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({query, scope, page}),
      });
      const data = await response.json();
      if (token !== state.outcomeToken || query !== $('rfOutcomeSearch').value.trim()
          || scope !== $('rfOutcomeScope').value) return;
      const list = $('rfOutcomeResults');
      if (page === 1) list.replaceChildren();
      for (const item of data.applications || []) {
        const row = document.createElement('div');
        row.className = 'rf-outcome-card';
        const info = document.createElement('div');
        const title = document.createElement('strong');
        title.textContent = item.patient_name || 'Αίτηση RF';
        const details = document.createElement('div');
        details.className = 'muted';
        details.textContent = [
          item.issued_date, item.issuer_name || 'Άγνωστος εκδότης',
          outcomeStages[item.stage] || item.stage,
          item.mine_to_follow ? 'Δική μου παρακολούθηση' : '',
        ].filter(Boolean).join(' · ');
        info.append(title, details);
        const open = document.createElement('button');
        open.type = 'button'; open.className = 'btn secondary small';
        open.textContent = 'Άνοιγμα έκβασης';
        open.addEventListener('click', () => {
          // Cards from a prior search/scope must never open an unintended RF.
          if (rfOutcomeContext() === context && state.outcomeContext === context)
            selectRfOutcome(item);
        });
        row.append(info, open);
        list.append(row);
      }
      state.outcomePage = page;
      $('rfOutcomeNext').hidden = !data.has_more;
      const c = data.counts || {};
      $('rfOutcomeCounts').textContent =
        'Σύνολο ' + data.denominator + ' αιτήσεις · Αναμονή ' + (c.pending || 0) +
        ' · Εγκρίσεις ' + (c.approved || 0) + ' · Απορρίψεις ' + (c.rejected || 0) +
        ' · Προς δική μου ενέργεια ' + (c.mine_actionable || 0) +
        (data.truncated ? ' · Η αναζήτηση καλύπτει περιορισμένο αριθμό πρόσφατων εγγραφών.' : '');
    } catch (error) { showError(error.message); }
  }

  async function applyRfOutcome() {
    const item = state.selectedOutcome;
    const token = state.outcomeSelectionToken;
    const authAtStart = tracking.authEpoch;
    const context = rfOutcomeContext();
    if (!item || item.id !== state.selectedOutcomeId ||
        $('rfOutcomeSelected').hidden || context !== state.outcomeContext) return;
    const action = $('rfOutcomeAction').value;
    try {
      const response = await api('/api/outcomes/' + encodeURIComponent(item.id) +
        '/events', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({
            action, revision: item.revision,
            event_date: ['claim', 'release'].includes(action) ? '' : $('rfOutcomeDate').value,
            rejection_reason: $('rfOutcomeReasonWrap').hidden ? '' : $('rfOutcomeReason').value.trim(),
            reported_source: $('rfOutcomeSource').value,
            confirmed_performed: $('rfOutcomePerformConfirm').checked,
            corrected_decision: action === 'correct_decision' ?
              $('rfOutcomeCorrectDecision').value : '',
          }),
        });
      const updated = await response.json();
      // The posted ID was explicitly selected when the request began.
      // Never overwrite a newer selection with its late response.
      if (token !== state.outcomeSelectionToken ||
          state.selectedOutcomeId !== item.id ||
          context !== rfOutcomeContext() || updated.id !== item.id) return;
      renderSelectedOutcome(updated);
      await loadRfOutcomes();
      if ($('rfTrackingView').hidden === false) loadRfTracking(true);
      showOk('Η έκβαση αποθηκεύτηκε. Η έκδοση του PDF παραμένει αμετάβλητη.');
    } catch (error) {
      if (token !== state.outcomeSelectionToken || authAtStart !== tracking.authEpoch ||
          state.selectedOutcomeId !== item.id || context !== rfOutcomeContext()) return;
      if ([401,403].includes(error.status)) protectedTrackingFailure(error,true);
      else {
        if (error.status === 409) invalidateRfOutcomeSelection();
        showError(error.message);
      }
    }
  }

  async function beginRfCorrection() {
    const item = state.selectedOutcome;
    const token = state.outcomeSelectionToken;
    const context = rfOutcomeContext();
    const authAtStart = tracking.authEpoch;
    const dirtyAtStart = tracking.dirtyGeneration;
    const creatorAtStart = state.contract ? JSON.stringify(buildDraft()) : '';
    const imagingAtStart = $('imagingReport').files?.[0];
    if (!item || item.stage !== 'rejected' || item.decision !== 'rejected' ||
        item.id !== state.selectedOutcomeId || context !== state.outcomeContext ||
        !state.contract || !Number.isInteger(item.revision) || item.revision < 1) return;
    if (!window.confirm('Η διορθωτική RF θα αντικαταστήσει τα πεδία της τρέχουσας φόρμας. ' +
        'Αποθήκευσε πρώτα τυχόν εργασία που θέλεις να διατηρήσεις. Συνέχεια;')) return;
    try {
      const response = await api('/api/reuse/rejected-correction', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({application_id: item.id, expected_revision: item.revision}),
      });
      const result = await response.json();
      if (token !== state.outcomeSelectionToken ||
          state.selectedOutcomeId !== item.id || context !== rfOutcomeContext() ||
          authAtStart !== tracking.authEpoch || dirtyAtStart !== tracking.dirtyGeneration ||
          creatorAtStart !== JSON.stringify(buildDraft()) ||
          imagingAtStart !== $('imagingReport').files?.[0] ||
          state.selectedOutcome?.revision !== item.revision ||
          result.application_id !== item.id || result.revision !== item.revision ||
          !result.fields || typeof result.fields !== 'object') return;
      const fields = result.fields;
      const oldID = normalizeRfIdentity($('identityNumber').value);
      const newID = normalizeRfIdentity(fields.identity_number);
      if (!newID || (oldID && oldID !== newID))
        throw new Error('Υπάρχει άλλη ταυτότητα στην τρέχουσα αίτηση. Αποθήκευσέ την και καθάρισέ την πρώτα.');
      if (!state.contract.indications?.[fields.indication_code] ||
          !['left','right'].includes(fields.laterality) ||
          !state.contract.products?.[fields.product_key] ||
          !['A1','A2'].includes(result.source?.pathway))
        throw new Error('Τα πεδία της απορριφθείσας RF δεν είναι έγκυρα για επαναχρησιμοποίηση.');
      // Deliberate single client commit after every async guard has passed.
      resetRfBuilderForCorrection();
      setPathway(result.source.pathway);
      populateRfFields(fields, true);
      state.nsaids = Array.isArray(fields.nsaid_trials) ? fields.nsaid_trials.map(cleanTrial) : [];
      state.others = Array.isArray(fields.other_analgesic_trials) ?
        fields.other_analgesic_trials.map(cleanTrial) : [];
      renderTrials();
      if (Array.isArray(fields.adverse_effects) && fields.adverse_effects.length) {
        $('adverseRows').replaceChildren();
        for (const adverse of fields.adverse_effects) {
          addAdverse();
          const rows = $('adverseRows').querySelectorAll('.adverse');
          const last = rows[rows.length - 1];
          last.querySelector('[data-kind=treatment]').value = adverse.treatment || '';
          last.querySelector('[data-kind=effect]').value = adverse.effect || '';
        }
      }
      state.correctionOfId = item.id;
      state.correctionExpectedRevision = item.revision;
      showRfView('builder');
      renderCorrectionContext();
      showOk('Άνοιξε διορθωτική RF. Επίλεξε ξανά τον ιατρό και επισύναψε νέα απεικόνιση.');
      updateSummary();
    } catch (error) {
      if (token !== state.outcomeSelectionToken || authAtStart !== tracking.authEpoch ||
          state.selectedOutcomeId !== item.id || context !== rfOutcomeContext()) return;
      if ([401,403].includes(error.status)) protectedTrackingFailure(error,true);
      else showError(error.message);
    }
  }

  function savedDraftFields() {
    const draft = buildDraft();
    delete draft.correction_of_application_id;
    delete draft.correction_source_expected_revision;
    delete draft.imaging_review_confirmed; // old confirmation never carries over
    delete draft.source_name_confirmed; // a name discrepancy must be reconfirmed on resume
    draft.source_patient_name = state.sourcePatientName;
    draft.draft_nsaids = state.nsaids.map(cleanTrial);
    draft.draft_others = state.others.map(cleanTrial);
    return draft;
  }

  async function saveRfDraft() {
    try {
      const stateFields = savedDraftFields();
      const active = !!state.draftId;
      const url = active ? '/api/drafts/' + encodeURIComponent(state.draftId) : '/api/drafts';
      const response = await api(url, {
        method: active ? 'PUT' : 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({fields: stateFields, ...(active ? {revision: state.draftRevision} : {})})
      });
      const result = await response.json();
      state.draftId = result.id;
      state.draftRevision = result.revision;
      $('rfActiveDraftLabel').textContent = 'Πρόχειρο αποθηκευμένο · έκδοση ' + result.revision;
      showOk('Το πρόχειρο αποθηκεύτηκε χωρίς επισυναπτόμενα.');
      await listRfDrafts();
    } catch (error) { showError(error.message); }
  }

  async function listRfDrafts(page = 1) {
    try {
      const query = $('rfDraftSearch').value.trim();
      const result = await (await api('/api/drafts?query=' + encodeURIComponent(query) +
        '&page=' + page)).json();
      const list = $('rfDraftList');
      if (page === 1) list.replaceChildren();
      for (const item of result.drafts || []) {
        const row = renderReuseCard(list, item, openRfDraft, 'Συνέχεια');
        const discard = document.createElement('button');
        discard.type = 'button'; discard.className = 'btn ghost small';
        discard.textContent = 'Διαγραφή πρόχειρου';
        discard.addEventListener('click', () => discardRfDraft(item));
        row.append(discard);
      }
      state.draftPage = page;
      $('rfDraftNextBtn').hidden = !result.has_more;
    } catch (error) { showError(error.message); }
  }

  async function discardRfDraft(item) {
    if (!window.confirm('Να διαγραφεί οριστικά αυτό το πρόχειρο;')) return;
    try {
      await api('/api/drafts/' + encodeURIComponent(item.id), {
        method:'DELETE', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({revision:item.revision})
      });
      if (state.draftId === item.id) {
        state.draftId = ''; state.draftRevision = null;
        $('rfActiveDraftLabel').textContent = '';
      }
      await listRfDrafts();
    } catch (error) { showError(error.message); }
  }

  async function restoreRfDraft(fields) {
    setPathway(fields.pathway === 'A2' ? 'A2' : 'A1');
    // A durable draft is never durable permission to reissue a rejected attempt.
    state.correctionOfId = '';
    state.correctionExpectedRevision = null;
    renderCorrectionContext();
    state.sourceKind = fields.source_kind || '';
    state.sourceId = fields.source_id || '';
    state.sourceOrigin = fields.source_origin || '';
    state.sourcePatientName = fields.source_patient_name || '';
    state.sourceNameConfirmed = false;
    for (const id of Object.values(rfFields)) $(id).value = '';
    // Never let a suspended/deactivated physician revert silently to default.
    const doctor = fields.doctor_key || '';
    $('doctorSelect').value = state.contract?.doctors?.[doctor]?.configured ? doctor : '';
    refreshDoctorReadiness();
    $('indicationSelect').value = fields.indication_code || '';
    indicationChanged();
    $('lateralitySelect').value = fields.laterality || '';
    applySuggestedLocation();
    populateRfFields(fields, true);
    document.querySelectorAll('.rf-reason').forEach(c =>
      c.checked = (fields.rf_reason_codes || []).includes(c.value));
    $('imagingReport').value = '';
    $('imagingStatus').hidden = true;
    state.imagingValidation = '';
    $('imagingConfirm').checked = false;
    $('imagingConfirmWrap').hidden = true;
    releaseHorosPreviews();
    renderHorosImages();
    state.nsaids = fields.draft_nsaids || [];
    state.others = fields.draft_others || [];
    state.unknowns = [];
    renderTrials();
    $('adverseRows').replaceChildren();
    for (const item of fields.adverse_effects || []) {
      addAdverse();
      const rows = $('adverseRows').querySelectorAll('.adverse');
      const last = rows[rows.length - 1];
      last.querySelector('[data-kind=treatment]').value = item.treatment || '';
      last.querySelector('[data-kind=effect]').value = item.effect || '';
    }
    if (!(fields.adverse_effects || []).length) addAdverse();
    const intervention = fields.intervention || {};
    for (const [key,id] of Object.entries({
      site:'interventionSite',date:'interventionDate',
      vas_before:'interventionVasBefore',vas_after:'interventionVasAfter'
    })) $(id).value = intervention[key] ?? '';
    const old = fields.legacy_history || {};
    for (const [key,id] of Object.entries({
      actual_procedure_date:'legacyProcedureDate',vas_before:'legacyVasBefore',
      vas_after:'legacyVasAfter',last_followup_date:'legacyFollowupDate',
      last_followup_vas:'legacyFollowupVas'
    })) $(id).value = old[key] ?? '';
    $('rfActualConfirm').checked = !!fields.legacy_confirmed;
    const prior = fields.procedure_history_id || '';
    state.selectedHistoryId = '';
    $('historyChoices').replaceChildren();
    if (prior && fields.identity_number) await lookupHistory(prior);
    if (fields.physio_dates_text) await parsePhysio();
    $('rfSourceLoaded').hidden = !state.sourceOrigin;
    $('rfSourceLoaded').textContent = state.sourceOrigin
      ? 'Προηγούμενη RF πηγή: ' + (state.sourceOrigin === 'pdf'
        ? 'αρχείο PDF (δεν αποθηκεύτηκε)' : 'RF ιστορικό') : '';
    if (state.sourceKind && state.sourceId) {
      // Reconcile saved form text with the current protected source, not an
      // untrusted source-name snapshot preserved in the draft.
      const previous = await (await api('/api/reuse/' +
        encodeURIComponent(state.sourceKind) + '/' +
        encodeURIComponent(state.sourceId))).json();
      state.sourcePatientName = String(previous.fields?.patient_name || '').trim();
    }
    renderSourceNameConflict();
    updateSummary();
  }

  async function openRfDraft(card) {
    try {
      const item = await (await api('/api/drafts/' + encodeURIComponent(card.id))).json();
      await restoreRfDraft(item.fields || {});
      state.draftId = item.id;
      state.draftRevision = item.revision;
      $('rfActiveDraftLabel').textContent = 'Πρόχειρο ανοιχτό · έκδοση ' + item.revision;
      showOk('Το πρόχειρο επανήλθε. Επισύναψε ξανά PDF/εικόνες πριν την έκδοση.');
    } catch (error) { showError(error.message); }
  }

  async function createPdf() {
    if (unresolvedSourceNameConflict()) {
      showError('Επίλυσε ρητά τη διαφορά ονόματος και ταυτότητας με την παλιά αίτηση πριν την έκδοση.');
      return;
    }
    if (!state.contract?.doctors?.[$('doctorSelect').value]?.configured) {
      showError('Επίλεξε ρητά τον ιατρό της αίτησης.');
      return;
    }
    if (!['left','right'].includes($('lateralitySelect').value)) {
      showError('Επίλεξε μία πλευρά: δεξιά ή αριστερά.');
      return;
    }
    if (!$('exactLocation').value.trim()) {
      showError('Απαιτείται μία ακριβής εντόπιση RF.');
      return;
    }
    const file = $('imagingReport').files[0];
    if (!file) {
      showError('Απαιτείται η απεικονιστική έκθεση PDF.');
      return;
    }
    if (state.imagingValidation === 'clearly_non_imaging') {
      showError('Το επιλεγμένο PDF έχει αναγνωριστεί ως μη απεικονιστικό.');
      return;
    }
    if (state.imagingValidation === 'ambiguous_or_unreadable' && !$('imagingConfirm').checked) {
      showError('Επιβεβαίωσε ότι το PDF είναι η απαιτούμενη απεικονιστική έκθεση.');
      return;
    }
    if (state.pathway === 'A2' && state.sourceOrigin === 'pdf' &&
        !state.selectedHistoryId && !$('rfActualConfirm').checked) {
      showError('Επιβεβαίωσε πρώτα ότι έγινε η πραγματική προηγούμενη εφαρμογή RF.');
      return;
    }
    try {
      $('createPdfBtn').disabled = true;
      $('createPdfBtn').textContent = 'Δημιουργία…';
      const form = new FormData();
      const requestDraft = buildDraft();
      if (state.draftId) {
        requestDraft.draft_id = state.draftId;
        requestDraft.draft_revision = state.draftRevision;
      }
      form.append('draft_json', JSON.stringify(requestDraft));
      form.append('imaging_report', file, file.name);
      state.horosImages.forEach(item => form.append('horos_jpegs', item.file, item.file.name));
      const response = await api('/api/create', {method:'POST', body:form});
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank', 'noopener');
      setTimeout(() => URL.revokeObjectURL(url), 120000);
      state.draftId = ''; state.draftRevision = null;
      state.correctionOfId = '';
      state.correctionExpectedRevision = null;
      renderCorrectionContext();
      $('rfActiveDraftLabel').textContent = '';
      await listRfDrafts();
      await loadRfOutcomes();
      showOk('Το επίσημο PDF δημιουργήθηκε. Η υποβολή στο ΓεΣΥ καταχωρίζεται ξεχωριστά.');

    } catch (error) {
      showError(error.message);
    } finally {
      $('createPdfBtn').disabled = false;
      $('createPdfBtn').textContent = 'Δημιουργία επίσημου PDF';
    }
  }

  function rfFeedbackTarget() {
    return $('rfTrackingView').hidden !== false ? $('validationPanel') :
      tracking.tab === 'week' ? $('rfWeekStatus') : $('rfTrackStatus');
  }
  function showError(message) {
    const target = rfFeedbackTarget();
    target.hidden = false; target.className = 'validation-panel error';
    target.textContent = message;
  }
  function showOk(message) {
    const target = rfFeedbackTarget();
    target.hidden = false; target.className = 'validation-panel ok';
    target.textContent = message;
  }

  function updateSummary() {
    const code = $('indicationSelect')?.value || '';
    const label = state.contract?.indications?.[code]?.label || '—';
    $('summaryText').textContent = `${state.pathway === 'A1' ? 'Νέα θεραπεία' : 'Συνέχιση'} · ${label}`;
    const items = [
      `Ασθενής: ${$('patientName')?.value || '—'}`,
      `Προϊόν: ${state.contract?.products?.[$('productSelect')?.value]?.label || '—'}`,
      `Ιατρός: ${state.contract?.doctors?.[$('doctorSelect').value]?.label || '—'}`,
      `Εντόπιση: ${$('exactLocation')?.value || '—'}`
    ];
    $('reviewSummary').innerHTML = items.map(x => `<div class="summary-item">${escapeHtml(x)}</div>`).join('');
  }


  // Both protected routes are views of the same document and RF controller.
  const tracking = {listGeneration: 0, listContextGeneration: 0, weekGeneration: 0, authEpoch: 0, dirtyGeneration: 0,
    cursor: '', searchContext: '', weekStart: '', tab: 'list', initialized: false};
  const nicosiaDay = () => {
    const f = new Intl.DateTimeFormat('en-CA', {timeZone:'Asia/Nicosia',
      year:'numeric', month:'2-digit', day:'2-digit'});
    const part = Object.fromEntries(f.formatToParts(new Date()).map(x => [x.type,x.value]));
    return part.year + '-' + part.month + '-' + part.day;
  };
  const addDate = (day, offset) => {
    const d = new Date(day + 'T12:00:00Z');
    d.setUTCDate(d.getUTCDate() + offset);
    return d.toISOString().slice(0,10);
  };
  const mondayFor = day => {
    const weekday = new Date(day + 'T12:00:00Z').getUTCDay();
    return addDate(day, -(weekday + 6) % 7);
  };
  function showRfView(view, push = true) {
    const isTracking = view === 'tracking';
    ++tracking.listGeneration;
    ++tracking.listContextGeneration;
    ++tracking.weekGeneration;
    invalidateRfOutcomeSelection();
    $('rfTrackingView').hidden = !isTracking;
    $('rfBuilderView').hidden = isTracking;
    $('rfOutcomesWorkspace').hidden = true; // Old embedded ledger is superseded.
    if (isTracking) {
      // Returning to Tracking cannot silently preserve a previous correction authorization.
      state.correctionOfId = '';
      state.correctionExpectedRevision = null;
      renderCorrectionContext();
      $('rfTrackingDetailHost').append($('rfOutcomeSelected'));
      $('rfOutcomeSelected').hidden = true;
      tracking.initialized = true;
      if (tracking.tab === 'week') loadRfWeek();
      else loadRfTracking(true);
    } else {
      $('rfOutcomesWorkspace').append($('rfOutcomeSelected'));
    }
    if (push) history.pushState({}, '', isTracking ?
      '/clinical/clinic-utilities/rf/tracking' : '/clinical/clinic-utilities/rf');
  }
  function trackingContext() {
    return JSON.stringify([$('rfTrackQuery').value.trim(), $('rfTrackScope').value,
      $('rfTrackStage').value]);
  }
  function protectedTrackingFailure(error, list) {
    const message = String(error?.message || '');
    if ([401,403].includes(error?.status) || /401|403/.test(message)) {
      ++tracking.authEpoch;
      ++tracking.listGeneration;
      ++tracking.listContextGeneration;
      ++tracking.weekGeneration;
      state.correctionOfId = ''; state.correctionExpectedRevision = null;
      renderCorrectionContext();
      invalidateRfOutcomeSelection();
      $('rfTrackResults').replaceChildren();
      $('rfWeekGrid').replaceChildren();
      $('rfTrackMore').hidden = true;
      $('rfTrackStatus').textContent = 'Απαιτείται νέα εξουσιοδότηση.';
      $('rfWeekStatus').textContent = 'Απαιτείται νέα εξουσιοδότηση.';
    } else if (list) {
      ++tracking.listGeneration;
      ++tracking.listContextGeneration;
      tracking.cursor = '';
      invalidateRfOutcomeSelection();
      $('rfTrackResults').replaceChildren();
      $('rfTrackMore').hidden = true;
      $('rfTrackStatus').textContent = 'Τα αποτελέσματα δεν είναι διαθέσιμα. Δοκίμασε ανανέωση.';
    } else {
      invalidateRfOutcomeSelection();
      $('rfWeekGrid').replaceChildren();
      $('rfWeekStatus').textContent = 'Το εβδομαδιαίο ημερολόγιο δεν είναι διαθέσιμο.';
    }
  }
  async function loadRfTracking(reset = true) {
    const context = trackingContext(), epoch = tracking.authEpoch;
    if (reset || context !== tracking.searchContext) {
      tracking.cursor = '';
      tracking.searchContext = context;
      ++tracking.listContextGeneration;
      $('rfTrackResults').replaceChildren();
      $('rfTrackMore').hidden = true;
      invalidateRfOutcomeSelection();
    }
    const cursor = tracking.cursor, generation = ++tracking.listGeneration;
    const contextGeneration = tracking.listContextGeneration;
    $('rfTrackMore').disabled = true;
    try {
      const response = await api('/api/outcomes/search', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({mode:'tracking', query:$('rfTrackQuery').value.trim(),
          scope:$('rfTrackScope').value, stage:$('rfTrackStage').value, cursor})});
      const result = await response.json();
      if (generation !== tracking.listGeneration || epoch !== tracking.authEpoch ||
          context !== trackingContext() || $('rfTrackingView').hidden) return;
      if (!Array.isArray(result.applications) || typeof result.scan_complete !== 'boolean' ||
          typeof result.has_more !== 'boolean' || result.has_more && !result.next_cursor)
        throw new Error('Malformed RF tracking source');
      for (const item of result.applications) {
        const row = document.createElement('div');
        row.className = 'rf-outcome-card';
        const label = document.createElement('div');
        label.textContent = [item.patient_name || 'Αίτηση RF', item.issued_date || '',
          item.pathway || 'A1',
          [item.site_key || '', item.laterality === 'left' ? 'αριστερά' :
            item.laterality === 'right' ? 'δεξιά' : ''].filter(Boolean).join(' / '),
          item.issuer_name || 'Μη τεκμηριωμένος εκδότης',
          outcomeStages[item.stage] || item.stage].filter(Boolean).join(' · ');
        const open = document.createElement('button');
        open.type = 'button'; open.className = 'btn secondary small';
        open.textContent = 'Άνοιγμα';
        open.addEventListener('click', () => {
          if (contextGeneration === tracking.listContextGeneration &&
              epoch === tracking.authEpoch && context === trackingContext() &&
              tracking.tab === 'list' && !$('rfTrackingView').hidden) selectRfOutcome(item);
        });
        row.append(label,open);
        $('rfTrackResults').append(row);
      }
      tracking.cursor = result.next_cursor || '';
      $('rfTrackMore').hidden = !result.has_more;
      $('rfTrackStatus').textContent = result.has_more
        ? 'Μερική προβολή · συνέχισε για παλαιότερες αιτήσεις.'
        : $('rfTrackResults').children.length
          ? 'Ολοκληρώθηκε η ανάγνωση του διαθέσιμου μητρώου. Οι καταστάσεις ενδέχεται να έχουν αλλάξει.'
          : 'Δεν βρέθηκαν αιτήσεις στο πλήρως ελεγμένο εύρος.';
    } catch (error) {
      if (generation === tracking.listGeneration && epoch === tracking.authEpoch &&
          context === trackingContext() && tracking.tab === 'list' &&
          !$('rfTrackingView').hidden) {
        $('rfTrackMore').hidden = true;
        protectedTrackingFailure(error,true);
      }
    } finally {
      if (generation === tracking.listGeneration) $('rfTrackMore').disabled = false;
    }
  }
  async function loadRfWeek() {
    const week = tracking.weekStart, generation = ++tracking.weekGeneration;
    const epoch = tracking.authEpoch;
    $('rfWeekTitle').textContent = week + ' — ' + addDate(week,6);
    $('rfWeekGrid').replaceChildren();
    $('rfWeekStatus').textContent = 'Ανάγνωση προστατευμένων ημερομηνιών…';
    try {
      const result = await (await api('/api/outcomes/calendar', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({week_start:week})})).json();
      if (generation !== tracking.weekGeneration || epoch !== tracking.authEpoch ||
          week !== tracking.weekStart || $('rfTrackingView').hidden) return;
      if (!Array.isArray(result.facts) || typeof result.complete !== 'boolean')
        throw new Error('Malformed RF week source');
      for (let d=0; d<7; d++) {
        const date = addDate(week,d);
        const day = document.createElement('section');
        day.className = 'card rf-week-day';
        const title = document.createElement('h3');
        title.textContent = ['Δευ','Τρι','Τετ','Πεμ','Παρ','Σαβ','Κυρ'][d] + ' · ' + date;
        day.append(title);
        for (const fact of result.facts.filter(f=>f.date===date)) {
          const line = document.createElement('button');
          line.type = 'button'; line.className = 'btn ghost small';
          line.textContent = (fact.kind === 'performed' ? 'Πραγματοποιήθηκε' :
            'Προγραμματισμένη') + ' · ' + (fact.patient_name || 'RF') +
            ' · ώρα μη επιβεβαιωμένη';
          line.addEventListener('click', () => selectRfOutcome({id:fact.application_id}));
          day.append(line);
        }
        $('rfWeekGrid').append(day);
      }
      $('rfWeekStatus').textContent = result.complete
        ? 'Εμφανίζονται μόνο καταχωρισμένες ημερομηνίες RF, όχι ραντεβού Γραμματείας.'
        : 'Μερική προβολή εβδομάδας · δεν επιβεβαιώνεται η πληρότητα.';
    } catch(error) {
      if (generation === tracking.weekGeneration && epoch === tracking.authEpoch &&
          week === tracking.weekStart && tracking.tab === 'week' &&
          !$('rfTrackingView').hidden) protectedTrackingFailure(error,false);
    }
  }
  function setTrackingTab(tab) {
    ++tracking.listGeneration;
    ++tracking.listContextGeneration;
    ++tracking.weekGeneration;
    invalidateRfOutcomeSelection();
    tracking.tab = tab;
    $('rfTrackListView').hidden = tab !== 'list';
    $('rfTrackWeekView').hidden = tab !== 'week';
    if (tab === 'week') loadRfWeek();
    else loadRfTracking(true);
  }

  document.addEventListener('DOMContentLoaded', () => {
    tracking.weekStart = mondayFor(nicosiaDay());
    $('rfBuilderView').addEventListener('input', () => ++tracking.dirtyGeneration, true);
    $('rfBuilderView').addEventListener('change', () => ++tracking.dirtyGeneration, true);
    window.addEventListener('clinical:authenticated', () => {
      ++tracking.authEpoch;
      ++tracking.listGeneration;
      ++tracking.listContextGeneration;
      ++tracking.weekGeneration;
      state.correctionOfId = '';
      state.correctionExpectedRevision = null;
      renderCorrectionContext();
      invalidateRfOutcomeSelection();
      $('rfTrackResults').replaceChildren();
      $('rfWeekGrid').replaceChildren();
      $('rfTrackMore').hidden = true;
      if (!$('rfTrackingView').hidden) {
        if (tracking.tab === 'week') loadRfWeek();
        else loadRfTracking(true);
      }
    });
    $('rfNavTracking').addEventListener('click', e => {
      e.preventDefault(); showRfView('tracking');
    });
    $('rfNavCreate').addEventListener('click', e => {
      e.preventDefault(); showRfView('builder');
    });
    window.addEventListener('popstate', () => {
      showRfView(location.pathname.endsWith('/tracking') ? 'tracking' : 'builder', false);
    });
    $('rfWeekNoDate').addEventListener('click', () => {
      $('rfTrackScope').value = 'all';
      $('rfTrackStage').value = 'approved';
      setTrackingTab('list');
    });
    $('rfTrackRefresh').addEventListener('click', () => tracking.tab === 'week'
      ? loadRfWeek() : loadRfTracking(true));
    $('rfTrackMore').addEventListener('click', () => loadRfTracking(false));
    $('rfTrackListTab').addEventListener('click', () => setTrackingTab('list'));
    $('rfTrackWeekTab').addEventListener('click', () => setTrackingTab('week'));
    $('rfWeekPrev').addEventListener('click', () => {
      tracking.weekStart = addDate(tracking.weekStart,-7); loadRfWeek();
    });
    $('rfWeekNext').addEventListener('click', () => {
      tracking.weekStart = addDate(tracking.weekStart,7); loadRfWeek();
    });
    $('rfWeekToday').addEventListener('click', () => {
      tracking.weekStart = mondayFor(nicosiaDay()); loadRfWeek();
    });
    for (const id of ['rfTrackQuery','rfTrackScope','rfTrackStage'])
      $(id).addEventListener('change', () => loadRfTracking(true));
    $('rfTrackQuery').addEventListener('keydown', e => {
      if (e.key === 'Enter') {e.preventDefault();loadRfTracking(true);}
    });
    showRfView(location.pathname.endsWith('/tracking') ? 'tracking' : 'builder', false);
    document.querySelectorAll('.segment').forEach(btn => btn.addEventListener('click', () => setPathway(btn.dataset.pathway)));
    $('indicationSelect').addEventListener('change', indicationChanged);
    $('lateralitySelect').addEventListener('change', applySuggestedLocation);
    $('otherArea').addEventListener('input', () => {
      if ($('indicationSelect').value === 'OTHER_CUSTOM') applySuggestedLocation();
    });
    $('parseMedicationBtn').addEventListener('click', parseMedication);
    $('addNsaidBtn').addEventListener('click', () => addManualTrial('nsaid'));
    $('addOtherBtn').addEventListener('click', () => addManualTrial('other'));
    $('refreshMedicationDictionary').addEventListener('click', loadMedicationDictionary);
    $('learnedMedicationDetails').addEventListener('toggle', event => {
      if (event.currentTarget.open) loadMedicationDictionary();
    });
    let medTimer;
    $('medicationText').addEventListener('input', () => {
      clearTimeout(medTimer);
      medTimer = setTimeout(parseMedication, 500);
    });
    let physioTimer;
    $('physioDates').addEventListener('input', () => {
      clearTimeout(physioTimer);
      physioTimer = setTimeout(parsePhysio, 350);
    });
    $('usualReasonBtn').addEventListener('click', () => document.querySelectorAll('.rf-reason').forEach(x => x.checked = ['FAILED_PHARMACOLOGIC','FAILED_CONSERVATIVE'].includes(x.value)));
    $('addAdverseBtn').addEventListener('click', addAdverse);
    $('historyLookupBtn').addEventListener('click', () => lookupHistory());
    $('rfSourceSearchBtn').addEventListener('click', () => searchRfSources());
    $('rfSourceNextBtn').addEventListener('click', () => searchRfSources(state.sourcePage + 1));
    $('rfSourceSearch').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); searchRfSources(); }});
    $('rfPreviousPdf').addEventListener('change', importRfPdf);
    $('rfUsePreviousNameBtn').addEventListener('click', () => resolveSourceName('previous'));
    $('rfConfirmCurrentNameBtn').addEventListener('click', () => resolveSourceName('current'));
    ['patientName', 'identityNumber'].forEach(id =>
      $(id).addEventListener('input', invalidateSourceNameConfirmation));
    $('rfDraftSaveBtn').addEventListener('click', saveRfDraft);
    $('rfDraftReloadBtn').addEventListener('click', () => listRfDrafts());
    $('rfDraftSearchBtn').addEventListener('click', () => listRfDrafts());
    $('rfDraftNextBtn').addEventListener('click', () => listRfDrafts(state.draftPage + 1));
    $('rfOutcomeRefresh').addEventListener('click', () => loadRfOutcomes());
    $('rfOutcomeNext').addEventListener('click', () => loadRfOutcomes(state.outcomePage + 1));
    $('rfOutcomeScope').addEventListener('change', () => {
      invalidateRfOutcomeSelection();
      loadRfOutcomes();
    });
    $('rfOutcomeSearch').addEventListener('input', () => {
      ++state.outcomeToken;
      invalidateRfOutcomeSelection();
      $('rfOutcomeResults').replaceChildren();
      $('rfOutcomeCounts').textContent = '';
      $('rfOutcomeNext').hidden = true;
    });
    $('rfOutcomeSearch').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); loadRfOutcomes(); }});
    $('rfOutcomeAction').addEventListener('change', outcomeActionChanged);
    $('rfOutcomeCorrectDecision').addEventListener('change', outcomeCorrectionKindChanged);
    $('rfOutcomeApply').addEventListener('click', applyRfOutcome);
    $('rfOutcomeCorrection').addEventListener('click', beginRfCorrection);
    $('rfDraftSearch').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); listRfDrafts(); }});
    loadContract().then(() => { listRfDrafts(); loadRfOutcomes(); });
    $('imagingReport').addEventListener('change', validateImaging);
    $('horosJpegs').addEventListener('change', addHorosImages);
    $('doctorManagement').addEventListener('toggle', e => {
      if (e.currentTarget.open) loadDoctorDirectory();
    });
    $('doctorForm').addEventListener('submit', saveDoctor);
    $('doctorCancel').addEventListener('click', resetDoctorForm);
    window.addEventListener('pagehide', releaseHorosPreviews);
    $('doctorSelect').addEventListener('change', () => {
      $('doctorSelect').dataset.currentDoctor = $('doctorSelect').value;
      refreshDoctorReadiness();
      updateSummary();
    });
    $('createPdfBtn').addEventListener('click', createPdf);
    $('clearBtn').addEventListener('click', () => location.reload());
    ['patientName','productSelect','exactLocation'].forEach(id => $(id).addEventListener('input', updateSummary));
    setPathway('A1');
    addAdverse();
  });
})();
