(() => {
  const $ = id => document.getElementById(id);
  const state = { contract: null, pathway: 'A1', selectedHistoryId: '', nsaids: [], others: [], unknowns: [], learnedMedications: [], physio: null, imagingValidation: '', horosImages: [], horosNextId: 0, doctors: [], editingDoctor: null };

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
    state.pathway = pathway;
    document.querySelectorAll('.segment').forEach(btn => btn.classList.toggle('active', btn.dataset.pathway === pathway));
    $('a1Flow').hidden = pathway !== 'A1';
    $('a2Flow').hidden = pathway !== 'A2';
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

  async function lookupHistory() {
    const item = state.contract?.indications?.[$('indicationSelect').value];
    if (!$('identityNumber').value.trim() || !item || !['left','right'].includes($('lateralitySelect').value)) {
      showError('Συμπλήρωσε ταυτότητα, ένδειξη και μία πλευρά (δεξιά ή αριστερά) πριν την αναζήτηση.');
      return;
    }
    try {
      $('historyStatus').textContent = 'Αναζήτηση…';
      const response = await api('/api/history', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          identity_number: $('identityNumber').value.trim(),
          site_key: item.site_key,
          laterality: $('lateralitySelect').value
        })
      });
      const result = await response.json();
      renderHistory(result.procedures || []);
      $('historyStatus').textContent = result.found ? `${result.procedures.length} εφαρμογή/ές` : 'Δεν βρέθηκε ιστορικό — χρησιμοποίησε legacy καταχώρηση.';
    } catch (error) {
      showError(error.message);
    }
  }

  function renderHistory(rows) {
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
      imaging_review_confirmed: $('imagingConfirm').checked
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

  async function createPdf() {
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
    try {
      $('createPdfBtn').disabled = true;
      $('createPdfBtn').textContent = 'Δημιουργία…';
      const form = new FormData();
      form.append('draft_json', JSON.stringify(buildDraft()));
      form.append('imaging_report', file, file.name);
      state.horosImages.forEach(item => form.append('horos_jpegs', item.file, item.file.name));
      const response = await api('/api/create', {method:'POST', body:form});
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank', 'noopener');
      setTimeout(() => URL.revokeObjectURL(url), 120000);
      showOk('Το επίσημο PDF δημιουργήθηκε.');
    } catch (error) {
      showError(error.message);
    } finally {
      $('createPdfBtn').disabled = false;
      $('createPdfBtn').textContent = 'Δημιουργία επίσημου PDF';
    }
  }

  function showError(message) {
    $('validationPanel').hidden = false;
    $('validationPanel').className = 'validation-panel error';
    $('validationPanel').textContent = message;
  }

  function showOk(message) {
    $('validationPanel').hidden = false;
    $('validationPanel').className = 'validation-panel ok';
    $('validationPanel').textContent = message;
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

  document.addEventListener('DOMContentLoaded', () => {
    loadContract();
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
    $('historyLookupBtn').addEventListener('click', lookupHistory);
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
