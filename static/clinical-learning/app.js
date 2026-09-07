(() => {
  'use strict';

  const state = {
    preview: null,
    detail: null,
    selectedRevision: null,
    revisionChallengeId: null,
    foundationNode: null,
    foundationAttemptId: null,
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

  function activateView(name) {
    document.querySelectorAll('.tab').forEach((button) => {
      button.classList.toggle('active', button.dataset.view === name);
    });
    document.querySelectorAll('.view').forEach((view) => view.classList.remove('active'));
    $(`view-${name}`).classList.add('active');
  }

  document.querySelectorAll('.tab').forEach((button) => {
    button.addEventListener('click', () => {
      activateView(button.dataset.view);
      if (button.dataset.view === 'history') loadHistory();
      if (button.dataset.view === 'foundation') loadFoundation();
      if (button.dataset.view === 'due') loadDue();
    });
  });

  function resetRevisionMode() {
    state.revisionChallengeId = null;
    $('revisionModeLabel').textContent = 'Νέο Challenge';
    $('saveChallenge').textContent = 'Save reviewed Challenge';
  }

  function parseChallengeTextarea() {
    const parsed = JSON.parse($('challengeJson').value);
    return parsed && typeof parsed === 'object' && parsed.challenge ? parsed.challenge : parsed;
  }

  $('clearChallenge').addEventListener('click', () => {
    $('challengeJson').value = '';
    state.preview = null;
    $('previewContent').hidden = true;
    $('previewEmpty').hidden = false;
    resetRevisionMode();
    setStatus('Έτοιμο');
  });

  $('previewChallenge').addEventListener('click', async () => {
    try {
      setStatus('Validation…');
      const challenge = parseChallengeTextarea();
      if (state.revisionChallengeId && String(challenge?.challenge_id || '') !== state.revisionChallengeId) {
        window.alert('Η νέα revision πρέπει να διατηρεί το ίδιο challenge_id.');
        setStatus('Revision identity mismatch');
        return;
      }
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
        <div class="muted">source: ${esc(fact.source)} · introduced via: ${esc(fact.introduced_via)} · stage: ${esc(fact.introduced_at_stage)} · certainty: ${esc(fact.certainty || '—')} · authoritative_for_patient: false</div>
      </div>`).join('') || '<div class="empty">No facts</div>';

    $('observations').innerHTML = (p.observations || []).map((obs, index) => `
      <div class="item observation" data-index="${index}">
        <div class="item-head"><span class="item-title">${esc(obs.statement)}</span><span><span class="badge">${esc(obs.category)}</span><span class="badge">${esc(obs.importance)}</span></span></div>
        <div class="muted">facts: ${esc((obs.linked_fact_ids || []).join(' · ') || '—')} · references: ${esc((obs.linked_reference_ids || []).join(' · ') || '—')}</div>
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
        if (input.disabled) input.value = '';
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
      if ((challenge.observations || []).some((x) => x.clinician_disposition === 'modified' && !String(x.clinician_modified_statement || '').trim())) {
        window.alert('Κάθε modified observation χρειάζεται modified statement.');
        return;
      }
      const revising = Boolean(state.revisionChallengeId);
      const path = revising
        ? `/api/challenges/${encodeURIComponent(state.revisionChallengeId)}`
        : '/api/challenges';
      const body = await api(path, {
        method: revising ? 'PUT' : 'POST',
        body: JSON.stringify({ challenge, confirm_save: true }),
      });
      setStatus(`Saved revision ${body.revision}`);
      window.alert(`Αποθηκεύτηκε το Challenge revision ${body.revision}.`);
      $('challengeJson').value = JSON.stringify(body.payload, null, 2);
      state.preview = null;
      resetRevisionMode();
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
      const revisions = body?.revisions || [];
      state.selectedRevision = revisions.length ? revisions[revisions.length - 1].revision : null;
      $('historyDetailCard').hidden = false;
      renderChallengeDetail(body);
    } catch (error) { showError(error); }
  }

  function latestRevision(detail) {
    const revisions = detail?.revisions || [];
    return revisions.length ? revisions[revisions.length - 1] : null;
  }

  function selectedRevision(detail) {
    const revisions = detail?.revisions || [];
    if (!revisions.length) return null;
    return revisions.find((item) => Number(item.revision) === Number(state.selectedRevision)) || revisions[revisions.length - 1];
  }

  function renderChallengeDetail(detail) {
    if (detail.deleted) {
      $('revisionSelector').innerHTML = '';
      $('historyDetail').innerHTML = `<div class="warning">Deleted content. Tombstone only · max revision ${esc(detail.max_deleted_revision)}</div>`;
      $('newRevision').disabled = true;
      return;
    }
    $('newRevision').disabled = false;
    const revisions = detail.revisions || [];
    const latest = latestRevision(detail);
    const viewed = selectedRevision(detail);
    const p = viewed?.payload || {};

    $('revisionSelector').innerHTML = revisions.map((item) => `
      <button class="${Number(item.revision) === Number(viewed?.revision) ? 'primary' : 'secondary'} revision-choice" data-revision="${esc(item.revision)}">rev ${esc(item.revision)}</button>`
    ).join('');
    document.querySelectorAll('.revision-choice').forEach((button) => button.addEventListener('click', () => {
      state.selectedRevision = Number(button.dataset.revision);
      renderChallengeDetail(detail);
    }));

    const overlays = new Map((viewed?.reference_verification || []).map((x) => [String(x.reference_id), x]));
    const refs = (p.references || []).map((ref) => {
      const overlay = overlays.get(String(ref.reference_id)) || { verification_state: 'unverified' };
      return `<div class="item reference-row">
        <div><b>${esc(ref.title)}</b><div class="muted">${esc(ref.reference_id)}</div></div>
        <select class="ref-state" data-ref="${esc(ref.reference_id)}"><option ${overlay.verification_state === 'unverified' ? 'selected' : ''}>unverified</option><option ${overlay.verification_state === 'verified_locator' ? 'selected' : ''}>verified_locator</option><option ${overlay.verification_state === 'verified_content' ? 'selected' : ''}>verified_content</option><option ${overlay.verification_state === 'invalid_or_unresolved' ? 'selected' : ''}>invalid_or_unresolved</option></select>
        <button class="secondary verify-ref" data-ref="${esc(ref.reference_id)}">Update</button>
      </div>`;
    }).join('');
    $('historyDetail').innerHTML = `
      <div class="meta-grid"><div class="meta"><b>Challenge</b>${esc(detail.challenge_id)}</div><div class="meta"><b>Viewing revision</b>${esc(viewed?.revision)}</div><div class="meta"><b>Latest revision</b>${esc(latest?.revision)}</div><div class="meta"><b>Mode</b>${esc(p.challenge_mode)}</div><div class="meta"><b>Topics</b>${esc((p.topics || []).join(' · '))}</div><div class="meta"><b>Review state</b>${esc(p.record_review_state)}</div></div>
      <h3>Reference verification overlay · revision ${esc(viewed?.revision)}</h3>${refs || '<div class="empty">No references</div>'}
      <h3>Immutable revision payload</h3><div class="json-detail">${esc(JSON.stringify(p, null, 2))}</div>`;
    document.querySelectorAll('.verify-ref').forEach((button) => button.addEventListener('click', async () => {
      const ref = button.dataset.ref;
      const verification_state = document.querySelector(`.ref-state[data-ref="${CSS.escape(ref)}"]`).value;
      try {
        await api(`/api/challenges/${encodeURIComponent(detail.challenge_id)}/revisions/${viewed.revision}/references/${encodeURIComponent(ref)}/verification`, {
          method: 'POST',
          body: JSON.stringify({ verification_state }),
        });
        const preserveRevision = viewed.revision;
        await loadChallengeDetail(detail.challenge_id);
        state.selectedRevision = preserveRevision;
        renderChallengeDetail(state.detail);
      } catch (error) { showError(error); }
    }));
  }

  $('newRevision').addEventListener('click', () => {
    const latest = latestRevision(state.detail);
    if (!latest || state.detail?.deleted) return;
    const candidate = structuredClone(latest.payload || {});
    candidate.revision = Number(latest.revision) + 1;
    candidate.supersedes_revision = Number(latest.revision);
    candidate.record_review_state = 'imported_pending_review';
    candidate.reviewed_at = null;
    candidate.linked_signal_ids = [];
    (candidate.references || []).forEach((ref) => {
      ref.verification_state = 'unverified';
      ref.verification_note = null;
    });
    (candidate.observations || []).forEach((obs) => {
      obs.clinician_disposition = 'pending';
      obs.clinician_modified_statement = null;
      obs.disposition_note = null;
    });
    state.revisionChallengeId = state.detail.challenge_id;
    state.preview = null;
    $('challengeJson').value = JSON.stringify(candidate, null, 2);
    $('previewContent').hidden = true;
    $('previewEmpty').hidden = false;
    $('revisionModeLabel').textContent = `Νέα immutable revision · source rev ${latest.revision}`;
    $('saveChallenge').textContent = 'Save new reviewed revision';
    activateView('import');
    setStatus('Revision candidate ready · Validate & Preview');
    $('challengeJson').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  function downloadText(filename, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  $('exportJson').addEventListener('click', () => {
    const viewed = selectedRevision(state.detail);
    if (!viewed) return;
    downloadText(`challenge-${state.detail.challenge_id}-r${viewed.revision}.json`, JSON.stringify(viewed.payload, null, 2), 'application/json');
  });

  $('exportMarkdown').addEventListener('click', () => {
    const viewed = selectedRevision(state.detail);
    if (!viewed) return;
    const p = viewed.payload;
    const facts = (p.fact_ledger || []).map((x) => `- [${x.fact_scope}] ${x.statement}`).join('\n');
    const observations = (p.observations || []).map((x) => `- **${x.category} / ${x.importance}** — ${x.clinician_modified_statement || x.statement} (${x.clinician_disposition})`).join('\n');
    const refs = (p.references || []).map((x) => `- ${x.title}${x.pmid ? ` — PMID ${x.pmid}` : x.doi ? ` — DOI ${x.doi}` : ''}`).join('\n');
    const md = `# ${p.title}\n\nRevision: ${viewed.revision}\n\nMode: ${p.challenge_mode}\n\nTopics: ${(p.topics || []).join(', ')}\n\n## Initial case\n\n${p.initial_case}\n\n## Fact Ledger\n${facts}\n\n## Observations\n${observations}\n\n## References\n${refs}\n`;
    downloadText(`challenge-${state.detail.challenge_id}-r${viewed.revision}.md`, md, 'text/markdown');
  });

  $('deleteChallenge').addEventListener('click', async () => {
    if (!state.detail?.challenge_id || !window.confirm('Οριστική διαγραφή όλων των Challenge revisions/content; θα παραμείνει μόνο non-content tombstone.')) return;
    try {
      await api(`/api/challenges/${encodeURIComponent(state.detail.challenge_id)}`, {
        method: 'DELETE',
        body: JSON.stringify({ confirm_delete: true }),
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
          <div class="muted">evidence attempts: ${esc(item.attempt_count)}${st.last_assessed_at ? ` · last ${esc(String(st.last_assessed_at).slice(0, 10))}` : ''}${st.next_review_due ? ` · next ${esc(st.next_review_due)}` : ''}</div>
        </button>`;
      }).join('');
      document.querySelectorAll('.foundation-card').forEach((button) => button.addEventListener('click', () => openAssessment(button.dataset.node)));
      setStatus('Foundation Map loaded');
    } catch (error) { showError(error); }
  }

  $('refreshFoundation').addEventListener('click', loadFoundation);

  function syncEvidenceRow(checkbox) {
    const method = checkbox.value;
    const result = document.querySelector(`.evidence-result[data-method="${CSS.escape(method)}"]`);
    const note = document.querySelector(`.evidence-note[data-method="${CSS.escape(method)}"]`);
    if (result) result.disabled = !checkbox.checked;
    if (note) note.disabled = !checkbox.checked;
  }

  document.querySelectorAll('.evidence-enabled').forEach((checkbox) => {
    checkbox.addEventListener('change', () => syncEvidenceRow(checkbox));
    syncEvidenceRow(checkbox);
  });

  function openAssessment(nodeId) {
    state.foundationNode = nodeId;
    state.foundationAttemptId = crypto.randomUUID();
    $('assessmentCard').hidden = false;
    $('assessmentTitle').textContent = `Foundation assessment · ${nodeId}`;
    $('assessmentProposedState').value = 'UNKNOWN_UNTESTED';
    $('assessmentState').value = 'UNKNOWN_UNTESTED';
    $('assessmentDue').value = '';
    $('assessmentNote').value = '';
    document.querySelectorAll('.evidence-enabled').forEach((checkbox) => {
      checkbox.checked = false;
      const result = document.querySelector(`.evidence-result[data-method="${CSS.escape(checkbox.value)}"]`);
      const note = document.querySelector(`.evidence-note[data-method="${CSS.escape(checkbox.value)}"]`);
      if (result) result.value = checkbox.value === 'self_rating_only' ? 'not_assessed' : 'demonstrated';
      if (note) note.value = '';
      syncEvidenceRow(checkbox);
    });
    $('assessmentResult').textContent = '';
    $('assessmentCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function buildAssessmentAttempt() {
    const checked = [...document.querySelectorAll('.evidence-enabled:checked')];
    if (!checked.length) {
      const error = new Error('foundation_assessment_evidence_required');
      error.detail = { code: 'foundation_assessment_evidence_required', path: 'attempt.evidence' };
      throw error;
    }
    const evidence = checked.map((checkbox) => {
      const method = checkbox.value;
      const result = document.querySelector(`.evidence-result[data-method="${CSS.escape(method)}"]`)?.value || 'not_assessed';
      const note = document.querySelector(`.evidence-note[data-method="${CSS.escape(method)}"]`)?.value.trim() || null;
      return {
        evidence_id: crypto.randomUUID(),
        method,
        result,
        clinician_reviewed: true,
        note,
        source_artifact_type: 'foundation_assessment',
        source_artifact_id: null,
      };
    });
    return {
      attempt_id: state.foundationAttemptId,
      schema_version: 'foundation_assessment_attempt_v1',
      module: 'osteoporosis',
      foundation_node_id: state.foundationNode,
      assessed_at: new Date().toISOString(),
      evidence,
      proposed_state: $('assessmentProposedState').value,
      clinician_final_state: $('assessmentState').value,
      clinician_note: $('assessmentNote').value.trim() || null,
    };
  }

  async function assessmentRequest(save) {
    if (!state.foundationNode || !state.foundationAttemptId) return;
    try {
      const attempt = buildAssessmentAttempt();
      const next_review_due = $('assessmentDue').value || null;
      const path = `/api/foundation/${encodeURIComponent(state.foundationNode)}/assessments${save ? '' : '/preview'}`;
      const body = await api(path, {
        method: 'POST',
        body: JSON.stringify({ attempt, next_review_due, ...(save ? { confirm_save: true } : {}) }),
      });
      if (save) {
        $('assessmentResult').textContent = JSON.stringify({ saved: true, state: body.state, due: body.due }, null, 2);
        setStatus('Foundation assessment saved');
        await loadFoundation();
        await loadDue();
        state.foundationNode = null;
        state.foundationAttemptId = null;
        $('assessmentCard').hidden = true;
      } else {
        $('assessmentResult').textContent = JSON.stringify({
          valid: body.valid,
          issues: body.issues || [],
          clinician_final_state: body.clinician_final_state,
          retention_state: body.retention_state,
          next_review_due: body.next_review_due,
        }, null, 2);
        setStatus(body.valid ? 'Assessment preview valid' : 'Assessment preview blocked');
      }
    } catch (error) { showError(error); }
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