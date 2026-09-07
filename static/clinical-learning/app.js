(() => {
  'use strict';

  const state = {
    preview: null,
    detail: null,
    foundationNode: null,
  };

  const $ = (id) => document.getElementById(id);

  const esc = (value) => String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');

  function setStatus(text) {
    $('globalStatus').textContent = text;
  }

  async function api(path, options = {}) {
    const response = await fetch(`/clinical/learning${path}`, {
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    let body = null;
    try { body = await response.json(); } catch (_) { body = {}; }
    if (!response.ok) {
      const detail = body?.detail || body || {};
      const err = new Error(detail.code || `HTTP ${response.status}`);
      err.detail = detail;
      err.status = response.status;
      throw err;
    }
    return body;
  }

  function issueHtml(issue, cls = 'issue') {
    const code = esc(issue?.code || 'validation_error');
    const path = esc(issue?.path || '');
    return `<div class="${cls}"><b>${code}</b>${path ? ` · ${path}` : ''}</div>`;
  }

  function showError(error) {
    setStatus('Χρειάζεται διόρθωση');
    const detail = error?.detail || {};
    const issues = detail.issues || [];
    const message = issues.length
      ? issues.map((x) => `${x.code}${x.path ? ` @ ${x.path}` : ''}`).join('\n')
      : `${detail.code || error.message}${detail.path ? ` @ ${detail.path}` : ''}`;
    window.alert(message);
  }

  document.querySelectorAll('.tab').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((x) => x.classList.remove('active'));
      document.querySelectorAll('.view').forEach((x) => x.classList.remove('active'));
      button.classList.add('active');
      $(`view-${button.dataset.view}`).classList.add('active');
      if (button.dataset.view === 'history') loadHistory();
      if (button.dataset.view === 'foundation') loadFoundation();
      if (button.dataset.view === 'due') loadDue();
    });
  });

  function parseChallengeTextarea() {
    const parsed = JSON.parse($('challengeJson').value);
    return parsed && typeof parsed === 'object' && parsed.challenge ? parsed.challenge : parsed;
  }

  $('clearChallenge').addEventListener('click', () => {
    $('challengeJson').value = '';
    state.preview = null;
    $('previewContent').hidden = true;
    $('previewEmpty').hidden = false;
    setStatus('Έτοιμο');
  });

  $('previewChallenge').addEventListener('click', async () => {
    try {
      setStatus('Validation…');
      const challenge = parseChallengeTextarea();
      const result = await api('/api/challenges/preview', {
        method: 'POST',
        body: JSON.stringify({ challenge }),
      });
      state.preview = result;
      renderPreview(result);
      setStatus(result.valid ? 'Preview valid' : 'Preview blocked');
    } catch (error) {
      if (error instanceof SyntaxError) {
        window.alert('Το JSON δεν είναι έγκυρο.');
        setStatus('Invalid JSON');
        return;
      }
      showError(error);
    }
  });

  function scopeClass(scope) {
    if (scope === 'real_deidentified_case_fact') return 'scope-real';
    if (scope === 'synthetic_case_fact') return 'scope-synthetic';
    return 'scope-inference';
  }

  function renderPreview(result) {
    $('previewEmpty').hidden = true;
    $('previewContent').hidden = false;
    const p = result.normalized_summary || {};

    $('challengeMeta').innerHTML = [
      ['Title', p.title],
      ['Mode', p.challenge_mode],
      ['Duplicate state', result.duplicate_state],
      ['Topics', (p.topics || []).join(' · ')],
      ['Foundation', (p.foundation_node_ids || []).join(' · ') || '—'],
      ['Review state', p.record_review_state],
    ].map(([k, v]) => `<div class="meta"><b>${esc(k)}</b>${esc(v || '—')}</div>`).join('');

    const issues = result.issues || [];
    const warnings = result.warnings || [];
    $('previewWarnings').innerHTML = [
      ...issues.map((x) => issueHtml(x, 'issue')),
      ...warnings.map((x) => issueHtml(x, 'warning')),
      ...(result.valid ? ['<div class="success">Schema / privacy / reference integrity preview: PASS</div>'] : []),
    ].join('');

    $('factLedger').innerHTML = (p.fact_ledger || []).map((fact) => `
      <div class="item">
        <div class="item-head"><span class="item-title">${esc(fact.statement)}</span><span class="badge ${scopeClass(fact.fact_scope)}">${esc(fact.fact_scope)}</span></div>
        <div class="muted">source: ${esc(fact.source)} · certainty: ${esc(fact.certainty || '—')} · authoritative_for_patient: false</div>
      </div>`).join('') || '<div class="empty">No facts</div>';

    $('observations').innerHTML = (p.observations || []).map((obs, index) => `
      <div class="item observation" data-index="${index}">
        <div class="item-head"><span class="item-title">${esc(obs.statement)}</span><span><span class="badge">${esc(obs.category)}</span><span class="badge">${esc(obs.importance)}</span></span></div>
        <div class="observation-controls">
          <select class="obs-disposition" data-index="${index}">
            <option value="pending">pending</option>
            <option value="accepted">accepted</option>
            <option value="modified">modified</option>
            <option value="dismissed">dismissed</option>
          </select>
          <input class="obs-modified" data-index="${index}" placeholder="Modified statement — required only for modified" disabled>
        </div>
      </div>`).join('') || '<div class="empty">No observations</div>';

    document.querySelectorAll('.obs-disposition').forEach((select) => {
      select.addEventListener('change', () => {
        const input = document.querySelector(`.obs-modified[data-index="${select.dataset.index}"]`);
        input.disabled = select.value !== 'modified';
      });
    });

    $('references').innerHTML = (p.references || []).map((ref) => `
      <div class="item"><div class="item-title">${esc(ref.title)}</div><div class="muted">${esc(ref.evidence_type)} · ${esc(ref.pmid ? `PMID ${ref.pmid}` : ref.doi ? `DOI ${ref.doi}` : ref.url || 'no locator')} · imported verification → unverified</div></div>`).join('') || '<div class="empty">No references</div>';

    $('confirmReview').checked = false;
    $('saveChallenge').disabled = !result.valid;
  }

  function reviewedChallengeFromPreview() {
    if (!state.preview?.normalized_summary) throw new Error('No valid preview');
    const challenge = structuredClone(state.preview.normalized_summary);
    (challenge.observations || []).forEach((obs, index) => {
      const disposition = document.querySelector(`.obs-disposition[data-index="${index}"]`)?.value || 'pending';
      const modified = document.querySelector(`.obs-modified[data-index="${index}"]`)?.value.trim() || null;
      obs.clinician_disposition = disposition;
      obs.clinician_modified_statement = disposition === 'modified' ? modified : null;
      obs.disposition_note = null;
    });
    challenge.privacy = challenge.privacy || {};
    challenge.privacy.contains_direct_identifiers = false;
    challenge.privacy.deidentification_attested = true;
    return challenge;
  }

  $('saveChallenge').addEventListener('click', async () => {
    if (!$('confirmReview').checked) {
      window.alert('Απαιτείται η clinician review / de-identification επιβεβαίωση.');
      return;
    }
    try {
      setStatus('Saving…');
      const challenge = reviewedChallengeFromPreview();
      if ((challenge.observations || []).some((x) => x.clinician_disposition === 'pending')) {
        window.alert('Κάθε observation πρέπει να είναι accepted, modified ή dismissed.');
        return;
      }
      const body = await api('/api/challenges', {
        method: 'POST',
        body: JSON.stringify({ challenge, confirm_save: true }),
      });
      setStatus(`Saved revision ${body.revision}`);
      window.alert(`Αποθηκεύτηκε το Challenge revision ${body.revision}.`);
      $('challengeJson').value = JSON.stringify(body.payload, null, 2);
      await loadDue();
    } catch (error) { showError(error); }
  });

  async function loadHistory() {
    try {
      setStatus('Loading history…');
      const params = new URLSearchParams();
      if ($('filterTopic').value.trim()) params.set('topic', $('filterTopic').value.trim());
      if ($('filterFoundation').value.trim()) params.set('foundation_node', $('filterFoundation').value.trim());
      if ($('filterMode').value) params.set('challenge_mode', $('filterMode').value);
      const body = await api(`/api/challenges?${params.toString()}`);
      const items = body.items || [];
      $('historyList').innerHTML = items.map((item) => `
        <button class="item history-item" data-id="${esc(item.challenge_id)}">
          <div class="item-head"><span class="item-title">${esc(item.title)}</span><span class="badge">rev ${esc(item.revision)}</span></div>
          <div class="muted">${esc(String(item.created_at || '').slice(0, 10))} · ${esc(item.challenge_mode)} · ${esc((item.topics || []).join(' · '))}</div>
        </button>`).join('') || '<div class="empty">Δεν υπάρχουν Challenges με αυτά τα φίλτρα.</div>';
      document.querySelectorAll('.history-item').forEach((button) => button.addEventListener('click', () => loadChallengeDetail(button.dataset.id)));
      setStatus('History loaded');
    } catch (error) { showError(error); }
  }

  $('refreshHistory').addEventListener('click', loadHistory);
  $('applyHistoryFilters').addEventListener('click', loadHistory);

  async function loadChallengeDetail(challengeId) {
    try {
      const body = await api(`/api/challenges/${encodeURIComponent(challengeId)}`);
      state.detail = body;
      $('historyDetailCard').hidden = false;
      renderChallengeDetail(body);
    } catch (error) { showError(error); }
  }

  function latestRevision(detail) {
    const revisions = detail?.revisions || [];
    return revisions.length ? revisions[revisions.length - 1] : null;
  }

  function renderChallengeDetail(detail) {
    if (detail.deleted) {
      $('historyDetail').innerHTML = `<div class="warning">Deleted content. Tombstone only · max revision ${esc(detail.max_deleted_revision)}</div>`;
      return;
    }
    const latest = latestRevision(detail);
    const p = latest?.payload || {};
    const overlays = new Map((latest?.reference_verification || []).map((x) => [String(x.reference_id), x]));
    const refs = (p.references || []).map((ref) => {
      const overlay = overlays.get(String(ref.reference_id)) || { verification_state: 'unverified', verification_note: '' };
      return `<div class="item reference-row">
        <div><b>${esc(ref.title)}</b><div class="muted">${esc(ref.reference_id)}</div></div>
        <select class="ref-state" data-ref="${esc(ref.reference_id)}"><option ${overlay.verification_state === 'unverified' ? 'selected' : ''}>unverified</option><option ${overlay.verification_state === 'verified_locator' ? 'selected' : ''}>verified_locator</option><option ${overlay.verification_state === 'verified_content' ? 'selected' : ''}>verified_content</option><option ${overlay.verification_state === 'invalid_or_unresolved' ? 'selected' : ''}>invalid_or_unresolved</option></select>
        <button class="secondary verify-ref" data-ref="${esc(ref.reference_id)}">Update</button>
      </div>`;
    }).join('');
    $('historyDetail').innerHTML = `
      <div class="meta-grid"><div class="meta"><b>Challenge</b>${esc(detail.challenge_id)}</div><div class="meta"><b>Latest revision</b>${esc(latest?.revision)}</div><div class="meta"><b>Mode</b>${esc(p.challenge_mode)}</div><div class="meta"><b>Topics</b>${esc((p.topics || []).join(' · '))}</div></div>
      <h3>Reference verification overlay</h3>${refs || '<div class="empty">No references</div>'}
      <h3>Immutable revision payload</h3><div class="json-detail">${esc(JSON.stringify(p, null, 2))}</div>`;
    document.querySelectorAll('.verify-ref').forEach((button) => button.addEventListener('click', async () => {
      const ref = button.dataset.ref;
      const verification_state = document.querySelector(`.ref-state[data-ref="${CSS.escape(ref)}"]`).value;
      try {
        await api(`/api/challenges/${encodeURIComponent(detail.challenge_id)}/revisions/${latest.revision}/references/${encodeURIComponent(ref)}/verification`, {
          method: 'POST', body: JSON.stringify({ verification_state }),
        });
        await loadChallengeDetail(detail.challenge_id);
      } catch (error) { showError(error); }
    }));
  }

  function downloadText(filename, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = filename; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  $('exportJson').addEventListener('click', () => {
    const latest = latestRevision(state.detail);
    if (!latest) return;
    downloadText(`challenge-${state.detail.challenge_id}-r${latest.revision}.json`, JSON.stringify(latest.payload, null, 2), 'application/json');
  });

  $('exportMarkdown').addEventListener('click', () => {
    const latest = latestRevision(state.detail);
    if (!latest) return;
    const p = latest.payload;
    const facts = (p.fact_ledger || []).map((x) => `- [${x.fact_scope}] ${x.statement}`).join('\n');
    const observations = (p.observations || []).map((x) => `- **${x.category} / ${x.importance}** — ${x.clinician_modified_statement || x.statement} (${x.clinician_disposition})`).join('\n');
    const refs = (p.references || []).map((x) => `- ${x.title}${x.pmid ? ` — PMID ${x.pmid}` : x.doi ? ` — DOI ${x.doi}` : ''}`).join('\n');
    const md = `# ${p.title}\n\nMode: ${p.challenge_mode}\n\nTopics: ${(p.topics || []).join(', ')}\n\n## Initial case\n\n${p.initial_case}\n\n## Fact Ledger\n${facts}\n\n## Observations\n${observations}\n\n## References\n${refs}\n`;
    downloadText(`challenge-${state.detail.challenge_id}-r${latest.revision}.md`, md, 'text/markdown');
  });

  $('deleteChallenge').addEventListener('click', async () => {
    if (!state.detail?.challenge_id || !window.confirm('Οριστική διαγραφή όλων των Challenge revisions/content; θα παραμείνει μόνο non-content tombstone.')) return;
    try {
      await api(`/api/challenges/${encodeURIComponent(state.detail.challenge_id)}`, {
        method: 'DELETE', body: JSON.stringify({ confirm_delete: true }),
      });
      await loadChallengeDetail(state.detail.challenge_id);
      await loadHistory();
      await loadDue();
    } catch (error) { showError(error); }
  });

  async function loadFoundation() {
    try {
      setStatus('Loading Foundation Map…');
      const body = await api('/api/foundation');
      $('foundationGrid').innerHTML = (body.items || []).map((item) => {
        const node = item.node || {};
        const st = item.state || {};
        return `<button class="foundation-card" data-node="${esc(node.node_id)}">
          <h3>${esc(node.title || node.label || node.node_id)}</h3>
          <div class="state-row"><span class="badge state">${esc(st.state)}</span><span class="muted">${esc(st.retention_state)}</span></div>
          <div class="muted">evidence attempts: ${esc(item.attempt_count)}${st.next_review_due ? ` · next ${esc(st.next_review_due)}` : ''}</div>
        </button>`;
      }).join('');
      document.querySelectorAll('.foundation-card').forEach((button) => button.addEventListener('click', () => openAssessment(button.dataset.node)));
      setStatus('Foundation Map loaded');
    } catch (error) { showError(error); }
  }

  $('refreshFoundation').addEventListener('click', loadFoundation);

  function openAssessment(nodeId) {
    state.foundationNode = nodeId;
    $('assessmentCard').hidden = false;
    $('assessmentTitle').textContent = `Foundation assessment · ${nodeId}`;
    const attempt = {
      attempt_id: crypto.randomUUID(),
      schema_version: 'foundation_assessment_attempt_v1',
      module: 'osteoporosis',
      foundation_node_id: nodeId,
      assessed_at: new Date().toISOString(),
      evidence: [{
        evidence_id: crypto.randomUUID(),
        method: 'self_rating_only',
        result: 'not_assessed',
        clinician_reviewed: true,
        note: null,
        source_artifact_type: 'foundation_assessment',
        source_artifact_id: null,
      }],
      proposed_state: 'UNKNOWN_UNTESTED',
      clinician_final_state: 'UNKNOWN_UNTESTED',
      clinician_note: null,
    };
    $('assessmentJson').value = JSON.stringify(attempt, null, 2);
    $('assessmentResult').textContent = '';
    $('assessmentCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function assessmentRequest(save) {
    if (!state.foundationNode) return;
    try {
      const attempt = JSON.parse($('assessmentJson').value);
      const next_review_due = $('assessmentDue').value || null;
      const path = `/api/foundation/${encodeURIComponent(state.foundationNode)}/assessments${save ? '' : '/preview'}`;
      const body = await api(path, {
        method: 'POST',
        body: JSON.stringify({ attempt, next_review_due, ...(save ? { confirm_save: true } : {}) }),
      });
      $('assessmentResult').textContent = JSON.stringify(body, null, 2);
      if (save) { await loadFoundation(); await loadDue(); setStatus('Foundation assessment saved'); }
      else setStatus(body.valid ? 'Assessment preview valid' : 'Assessment preview blocked');
    } catch (error) {
      if (error instanceof SyntaxError) { window.alert('Το assessment JSON δεν είναι έγκυρο.'); return; }
      showError(error);
    }
  }

  $('previewAssessment').addEventListener('click', () => assessmentRequest(false));
  $('saveAssessment').addEventListener('click', () => assessmentRequest(true));

  async function loadDue() {
    try {
      const body = await api('/api/due');
      const items = body.items || [];
      $('dueList').innerHTML = items.map((item) => `
        <div class="item due-${esc(item.due_status)}">
          <div class="item-head"><span class="item-title">${esc(item.item_type)}</span><span class="badge">${esc(item.due_status)}</span></div>
          <div>${esc(item.target_id)}</div>
          <div class="muted">due ${esc(item.due_on || '—')} · occurrence ${esc(item.occurrence)} · ${esc(item.reason_code)} · source ${esc(item.source_artifact_type)}:${esc(item.source_artifact_id)}</div>
        </div>`).join('') || '<div class="empty">Δεν υπάρχουν materialized learning due items.</div>';
      setStatus('Due state loaded');
    } catch (error) { showError(error); }
  }

  $('refreshDue').addEventListener('click', loadDue);

  setStatus('Έτοιμο');
})();
