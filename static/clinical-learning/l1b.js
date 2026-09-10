(() => {
  'use strict';

  const state = {
    inbox: [],
    selectedImport: null,
    pendingSaveImportId: null,
    loops: [],
    selectedLoop: null,
  };

  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');

  function setStatus(text) {
    if ($('globalStatus')) $('globalStatus').textContent = text;
  }

  const nativeFetch = window.fetch.bind(window);

  function requestPath(input) {
    const raw = typeof input === 'string' ? input : input?.url || '';
    try { return new URL(raw, window.location.origin).pathname; } catch (_) { return String(raw); }
  }

  function requestMethod(input, options) {
    return String(options?.method || input?.method || 'GET').toUpperCase();
  }

  function isChallengePersistenceRequest(input, options) {
    const path = requestPath(input);
    const method = requestMethod(input, options);
    if (method === 'POST' && path === '/clinical/learning/api/challenges') return true;
    return method === 'PUT'
      && /^\/clinical\/learning\/api\/challenges\/[^/]+$/.test(path);
  }

  async function linkSavedChallengeToPending(response) {
    if (!state.pendingSaveImportId || !response.ok) return;
    let body = null;
    try { body = await response.clone().json(); } catch (_) { return; }
    if (!body?.challenge_id || !body?.revision) return;

    const importId = state.pendingSaveImportId;
    const linkResponse = await nativeFetch(
      `/clinical/learning/api/imports/${encodeURIComponent(importId)}/accepted`,
      {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          challenge_id: body.challenge_id,
          revision: body.revision,
          confirm_link: true,
        }),
      },
    );
    if (!linkResponse.ok) {
      let detail = {};
      try { detail = (await linkResponse.json())?.detail || {}; } catch (_) { detail = {}; }
      setStatus('Challenge saved · Learning Loop link needs retry');
      window.alert(`Το Challenge αποθηκεύτηκε, αλλά η σύνδεση με το Learning Loop δεν ολοκληρώθηκε (${detail.code || linkResponse.status}). Το pending import παραμένει διαθέσιμο για ασφαλές retry.`);
      return;
    }

    state.pendingSaveImportId = null;
    setStatus(`Saved revision ${body.revision} · Learning Loop activated`);
    setTimeout(() => {
      loadInbox();
      loadLoops();
    }, 0);
  }

  // app.js was loaded first and resolves the global fetch at request time. This
  // narrow wrapper observes only successful Challenge POST/PUT writes that began
  // from an Inbox review. It never alters the Challenge response or bypasses the
  // existing clinician-review save path.
  window.fetch = async (input, options = {}) => {
    const response = await nativeFetch(input, options);
    if (state.pendingSaveImportId && isChallengePersistenceRequest(input, options)) {
      try { await linkSavedChallengeToPending(response); } catch (error) {
        setStatus('Challenge saved · Learning Loop link needs retry');
        window.alert(`Το Challenge αποθηκεύτηκε, αλλά η σύνδεση με το Learning Loop χρειάζεται retry (${error?.message || 'link_error'}).`);
      }
    }
    return response;
  };

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
      const error = new Error(detail.code || `HTTP ${response.status}`);
      error.detail = detail;
      error.status = response.status;
      throw error;
    }
    return body;
  }

  function showError(error) {
    const detail = error?.detail || {};
    const issues = detail.issues || [];
    const message = issues.length
      ? issues.map((x) => `${x.code}${x.path ? ` @ ${x.path}` : ''}`).join('\n')
      : `${detail.code || error.message}${detail.path ? ` @ ${detail.path}` : ''}`;
    setStatus('Χρειάζεται διόρθωση');
    window.alert(message);
  }

  function clickView(name) {
    document.querySelector(`.tab[data-view="${CSS.escape(name)}"]`)?.click();
  }

  function extractClipboardJson(rawText) {
    let text = String(rawText ?? '').trim();
    if (!text) throw new SyntaxError('clipboard_empty');

    const fences = [...text.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)];
    if (fences.length) {
      const candidate = fences.find((match) => String(match[1] || '').trim().startsWith('{')) || fences[0];
      text = String(candidate[1] || '').trim();
    }

    if (!text.startsWith('{')) {
      const firstBrace = text.indexOf('{');
      const lastBrace = text.lastIndexOf('}');
      if (firstBrace >= 0 && lastBrace > firstBrace) text = text.slice(firstBrace, lastBrace + 1);
    }
    return text;
  }

  function clipboardImportEnvelope(rawText) {
    const parsed = JSON.parse(extractClipboardJson(rawText));
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new SyntaxError('clipboard_object_required');
    }

    if (parsed.episode !== undefined) {
      if (!parsed.episode || typeof parsed.episode !== 'object' || Array.isArray(parsed.episode)) {
        throw new SyntaxError('clipboard_episode_object_required');
      }
      const envelope = { episode: parsed.episode };
      if (parsed.source_event_id !== undefined) envelope.source_event_id = parsed.source_event_id;
      if (parsed.loop_plan !== undefined) envelope.loop_plan = parsed.loop_plan;
      if (parsed.resources !== undefined) envelope.resources = parsed.resources;
      return envelope;
    }

    const episode = structuredClone(parsed);
    const envelope = { episode };
    if (typeof episode.source_event_id === 'string' && episode.source_event_id.trim()) {
      envelope.source_event_id = episode.source_event_id.trim();
      delete episode.source_event_id;
    }
    return envelope;
  }

  function showClipboardFallback(message) {
    const fallback = $('clipboardFallback');
    if (fallback) fallback.open = true;
    if ($('clipboardImportResult')) $('clipboardImportResult').textContent = message || '';
    $('clipboardManualJson')?.focus();
  }

  function clipboardReceipt(body) {
    return {
      state: body.state,
      import_id: body.import_id,
      source_event_id: body.source_event_id,
      source_format: body.source_format,
      idempotent: Boolean(body.idempotent),
    };
  }

  async function submitClipboardArtifact(rawText, { clearManual = false } = {}) {
    const envelope = clipboardImportEnvelope(rawText);
    setStatus('Sending Challenge to Inbox…');
    const body = await api('/api/imports', {
      method: 'POST',
      body: JSON.stringify(envelope),
    });
    if ($('clipboardImportResult')) {
      $('clipboardImportResult').textContent = JSON.stringify(clipboardReceipt(body), null, 2);
    }
    if (clearManual && $('clipboardManualJson')) $('clipboardManualJson').value = '';
    await loadInbox();
    clickView('inbox');
    openInboxItem(body.import_id);
    setStatus(body.state === 'pending_review' ? 'Challenge imported · pending review' : `Challenge import · ${body.state}`);
    return body;
  }

  async function importFromClipboard() {
    const button = $('clipboardImport');
    if (button) button.disabled = true;
    let text = '';
    try {
      if (!navigator.clipboard?.readText) {
        showClipboardFallback('Το browser δεν επιτρέπει άμεση ανάγνωση clipboard. Κάνε Paste παρακάτω και πάτησε Send pasted Challenge.');
        setStatus('Clipboard read unavailable · manual paste ready');
        return;
      }
      try {
        text = await navigator.clipboard.readText();
      } catch (_) {
        showClipboardFallback('Η πρόσβαση στο clipboard δεν επιτράπηκε. Κάνε Paste παρακάτω και πάτησε Send pasted Challenge.');
        setStatus('Clipboard permission unavailable · manual paste ready');
        return;
      }
      if (!String(text).trim()) {
        showClipboardFallback('Το clipboard είναι κενό. Κάνε Paste το Challenge artifact παρακάτω.');
        setStatus('Clipboard empty · manual paste ready');
        return;
      }
      try {
        await submitClipboardArtifact(text);
      } catch (error) {
        if ($('clipboardManualJson')) $('clipboardManualJson').value = text;
        showClipboardFallback('Το artifact δεν εισήχθη. Το κείμενο διατηρήθηκε προσωρινά στο πεδίο για διόρθωση/retry.');
        if (error instanceof SyntaxError) {
          setStatus('Clipboard JSON needs correction');
          window.alert('Το copied Challenge artifact δεν περιέχει έγκυρο JSON object.');
        } else {
          showError(error);
        }
      }
    } finally {
      if (button) button.disabled = false;
    }
  }

  async function importFromManualPaste() {
    const text = $('clipboardManualJson')?.value || '';
    if (!String(text).trim()) {
      window.alert('Κάνε πρώτα Paste το Challenge artifact.');
      return;
    }
    try {
      await submitClipboardArtifact(text, { clearManual: true });
    } catch (error) {
      if (error instanceof SyntaxError) {
        setStatus('Pasted JSON needs correction');
        window.alert('Το pasted Challenge artifact δεν περιέχει έγκυρο JSON object.');
        return;
      }
      showError(error);
    }
  }

  function installClipboardHandoffSurface() {
    const inboxView = $('view-inbox');
    if (!inboxView || $('clipboardHandoffCard')) return;
    const card = document.createElement('article');
    card.className = 'card';
    card.id = 'clipboardHandoffCard';
    card.innerHTML = `
      <div class="section-head">
        <div>
          <h2>Quick Challenge Handoff</h2>
          <p>Αντέγραψε το structured Challenge artifact από τη συζήτηση και στείλ' το κατευθείαν στο Inbox.</p>
        </div>
        <button class="primary" id="clipboardImport">Paste & Send Challenge</button>
      </div>
      <div class="notice">Η ανάγνωση του clipboard γίνεται μόνο μετά από δικό σου tap. Δημιουργείται μόνο <b>pending_review</b> candidate· δεν γίνεται αυτόματα accepted learning record.</div>
      <pre id="clipboardImportResult" class="result"></pre>
      <details id="clipboardFallback">
        <summary>Χειροκίνητη επικόλληση</summary>
        <p class="muted">Αν το iPhone/browser δεν επιτρέψει clipboard read, κάνε Paste εδώ. Το πεδίο δεν αποθηκεύεται σε localStorage/sessionStorage.</p>
        <textarea id="clipboardManualJson" class="short-textarea" spellcheck="false" placeholder='Paste Challenge JSON or the copied json code block here'></textarea>
        <div class="row"><button class="secondary" id="clipboardManualSend">Send pasted Challenge</button></div>
      </details>`;
    inboxView.prepend(card);
    $('clipboardImport')?.addEventListener('click', importFromClipboard);
    $('clipboardManualSend')?.addEventListener('click', importFromManualPaste);
  }

  function observationGroups(challenge) {
    const groups = new Map();
    (challenge?.observations || []).forEach((item) => {
      const category = item.category || 'other';
      if (!groups.has(category)) groups.set(category, []);
      groups.get(category).push(item);
    });
    return groups;
  }

  function renderEpisodeTimeline(challenge) {
    const responses = challenge?.reasoning_responses || [];
    const disclosures = [...(challenge?.progressive_disclosures || [])].sort((a, b) => Number(a.sequence) - Number(b.sequence));
    let html = `<div class="timeline-step"><span class="badge">Case</span><div class="timeline-text">${esc(challenge?.initial_case || '—')}</div></div>`;
    responses.forEach((response, index) => {
      html += `<div class="timeline-step clinician-step"><span class="badge">Δική σου απάντηση · ${esc(response.stage)}</span><div class="timeline-text">${esc(response.text)}</div></div>`;
      const disclosure = disclosures[index];
      if (disclosure) {
        html += `<div class="timeline-step disclosure-step"><span class="badge">Νέο στοιχείο ${esc(disclosure.sequence)}</span><div class="timeline-text">${esc(disclosure.narrative || disclosure.label || '—')}</div></div>`;
      }
    });
    if (challenge?.final_clinician_decision) {
      html += `<div class="timeline-step clinician-step"><span class="badge">Final decision</span><div class="timeline-text">${esc(challenge.final_clinician_decision)}</div></div>`;
    }
    return html;
  }

  function renderDebrief(challenge) {
    const groups = observationGroups(challenge);
    const order = [
      ['strength', 'Strengths'],
      ['missed_opportunity', 'Needs reinforcement'],
      ['clear_error', 'Clear errors'],
      ['defensible_disagreement', 'Defensible disagreements'],
      ['evidence_gap', 'Evidence gaps'],
      ['blind_spot', 'Blind spots'],
      ['reasoning_pattern', 'Reasoning patterns'],
      ['clinical_insight', 'Clinical insights'],
      ['uncertainty', 'Uncertainty'],
    ];
    return order.map(([key, title]) => {
      const items = groups.get(key) || [];
      if (!items.length) return '';
      return `<div class="debrief-group debrief-${esc(key)}"><h4>${esc(title)} <span class="badge">${items.length}</span></h4>${items.map((item) => `<div class="item"><div>${esc(item.statement)}</div><div class="muted">${esc(item.importance)}${(item.gap_classes || []).length ? ` · ${esc(item.gap_classes.join(' · '))}` : ''}</div></div>`).join('')}</div>`;
    }).join('');
  }

  function renderPlanSummary(plan) {
    const objectives = plan?.objectives || [];
    const bridges = plan?.bridge_targets || [];
    const occurrences = plan?.consolidation_occurrences || [];
    return `
      <div class="learning-summary-grid">
        <div class="metric"><b>${esc(objectives.length)}</b><span>learning targets</span></div>
        <div class="metric"><b>${esc(bridges.length)}</b><span>bridge targets</span></div>
        <div class="metric"><b>${esc(occurrences.length)}</b><span>planned repeats</span></div>
      </div>
      ${objectives.length ? `<h4>Τι θέλει ενίσχυση</h4><div class="stack">${objectives.map((item) => `<div class="item"><b>${esc(item.title)}</b><div class="muted">${esc(item.rationale)} · ${esc((item.foundation_node_ids || []).join(' · '))}</div></div>`).join('')}</div>` : ''}
      ${bridges.length ? `<h4>Γεφύρωση νησίδων</h4><div class="stack">${bridges.map((item) => `<div class="item bridge-item"><div class="item-head"><b>${esc((item.foundation_node_ids || []).join(' ↔ '))}</b><span class="badge">${esc(item.state)}</span></div><div>${esc(item.rationale)}</div></div>`).join('')}</div>` : ''}
      ${occurrences.length ? `<h4>Επαναλήψεις</h4><div class="repeat-strip">${occurrences.map((item) => `<div class="repeat-chip"><b>${esc(item.sequence)} · ${esc(item.kind)}</b><span>${esc(item.due_on)}</span></div>`).join('')}</div>` : ''}`;
  }

  async function loadInbox() {
    try {
      setStatus('Loading Inbox…');
      const body = await api('/api/imports');
      state.inbox = body.items || [];
      $('inboxList').innerHTML = state.inbox.map((item) => {
        const p = item.normalized_challenge || {};
        const counts = observationGroups(p);
        return `<button class="item history-item inbox-item" data-import="${esc(item.import_id)}">
          <div class="item-head"><span class="item-title">${esc(p.title || 'Untitled learning episode')}</span><span class="badge">${esc(item.state)}</span></div>
          <div class="muted">${esc(String(p.created_at || '').slice(0, 10))} · ${esc((p.topics || []).join(' · '))}</div>
          <div class="mini-stats">strengths ${esc((counts.get('strength') || []).length)} · reinforce ${esc((counts.get('missed_opportunity') || []).length)} · errors ${esc((counts.get('clear_error') || []).length)} · bridges ${esc((item.loop_plan?.bridge_targets || []).length)}</div>
        </button>`;
      }).join('') || '<div class="empty">Δεν υπάρχουν pending imports. Αντέγραψε το structured Challenge artifact και χρησιμοποίησε το Quick Challenge Handoff παραπάνω.</div>';
      document.querySelectorAll('.inbox-item').forEach((button) => button.addEventListener('click', () => openInboxItem(button.dataset.import)));
      setStatus('Inbox loaded');
    } catch (error) { showError(error); }
  }

  function openInboxItem(importId) {
    const item = state.inbox.find((candidate) => candidate.import_id === importId);
    if (!item) return;
    state.selectedImport = item;
    $('inboxDetailCard').hidden = false;
    const p = item.normalized_challenge || {};
    const warnings = item.adapter_warnings || [];
    const resources = item.resources || [];
    $('inboxDetail').innerHTML = `
      <div class="meta-grid">
        <div class="meta"><b>Mode</b>${esc(p.challenge_mode)}</div>
        <div class="meta"><b>Topics</b>${esc((p.topics || []).join(' · '))}</div>
        <div class="meta"><b>Foundation</b>${esc((p.foundation_node_ids || []).join(' · ') || '—')}</div>
        <div class="meta"><b>Source format</b>${esc(item.source_format)}</div>
      </div>
      ${warnings.length ? `<div class="warning"><b>Adapter warnings</b><br>${warnings.map(esc).join('<br>')}</div>` : ''}
      <h3>Case & reasoning trajectory</h3>
      <div class="learning-timeline">${renderEpisodeTimeline(p)}</div>
      <h3>Performance debrief</h3>
      ${renderDebrief(p) || '<div class="empty">No debrief observations</div>'}
      <h3>Learning plan</h3>
      ${renderPlanSummary(item.loop_plan || {})}
      <h3>Learning resources <span class="badge">${esc(resources.length)}</span></h3>
      <div class="stack">${resources.map((resource) => `<div class="item"><b>${esc(resource.title)}</b><div>${esc(resource.rationale)}</div><div class="muted">${esc(resource.provider)} · ${esc(resource.kind)} · checked ${esc(String(resource.checked_at || '').slice(0, 10))}</div></div>`).join('') || '<div class="empty">Fresh resources will appear when supplied by the learning source.</div>'}</div>`;
    $('reviewPendingImport').disabled = item.state !== 'pending_review';
    $('rejectPendingImport').disabled = item.state !== 'pending_review';
    $('inboxDetailCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  $('refreshInbox')?.addEventListener('click', loadInbox);

  $('reviewPendingImport')?.addEventListener('click', () => {
    const item = state.selectedImport;
    if (!item || item.state !== 'pending_review') return;
    state.pendingSaveImportId = item.import_id;
    const candidate = structuredClone(item.normalized_challenge || {});
    candidate.privacy = candidate.privacy || {};
    // Preview is no-write. Final persistence still requires the explicit checkbox
    // in the canonical review surface below.
    candidate.privacy.contains_direct_identifiers = false;
    candidate.privacy.deidentification_attested = true;
    $('challengeJson').value = JSON.stringify(candidate, null, 2);
    $('revisionModeLabel').textContent = `Inbox review · ${item.import_id}`;
    clickView('import');
    $('previewChallenge').click();
  });

  $('rejectPendingImport')?.addEventListener('click', async () => {
    const item = state.selectedImport;
    if (!item || item.state !== 'pending_review') return;
    if (!window.confirm('Reject this pending learning import? No accepted Challenge content will be created.')) return;
    try {
      await api(`/api/imports/${encodeURIComponent(item.import_id)}/reject`, {
        method: 'POST',
        body: JSON.stringify({ confirm_reject: true }),
      });
      if (state.pendingSaveImportId === item.import_id) state.pendingSaveImportId = null;
      $('inboxDetailCard').hidden = true;
      state.selectedImport = null;
      await loadInbox();
    } catch (error) { showError(error); }
  });

  $('importEpisode')?.addEventListener('click', async () => {
    try {
      const episode = JSON.parse($('episodeJson').value);
      setStatus('Converting learning episode…');
      const body = await api('/api/imports', {
        method: 'POST',
        body: JSON.stringify({ episode }),
      });
      $('episodeImportResult').textContent = JSON.stringify({
        import_id: body.import_id,
        state: body.state,
        source_format: body.source_format,
        warnings: body.adapter_warnings,
      }, null, 2);
      await loadInbox();
      clickView('inbox');
      openInboxItem(body.import_id);
    } catch (error) {
      if (error instanceof SyntaxError) {
        window.alert('Το learning episode JSON δεν είναι έγκυρο.');
        return;
      }
      showError(error);
    }
  });

  $('clearEpisode')?.addEventListener('click', () => {
    $('episodeJson').value = '';
    $('episodeImportResult').textContent = '';
  });

  $('clearChallenge')?.addEventListener('click', () => {
    state.pendingSaveImportId = null;
  });
  $('newRevision')?.addEventListener('click', () => {
    state.pendingSaveImportId = null;
  });

  function nextOccurrence(loop) {
    const occurrences = loop?.plan?.consolidation_occurrences || [];
    return occurrences.find((item) => item?.due_state?.due_status !== 'completed') || null;
  }

  async function loadLoops() {
    try {
      setStatus('Loading Learning Loop…');
      const body = await api('/api/learning-loops');
      state.loops = body.items || [];
      $('loopList').innerHTML = state.loops.map((loop) => {
        const next = nextOccurrence(loop);
        const bridges = loop.plan?.bridge_targets || [];
        const objectives = loop.plan?.objectives || [];
        return `<button class="item history-item loop-item" data-cycle="${esc(loop.cycle_id)}">
          <div class="item-head"><span class="item-title">${esc(loop.challenge_title || loop.challenge_id)}</span><span class="badge">rev ${esc(loop.source_revision)}</span></div>
          <div class="muted">targets ${esc(objectives.length)} · bridges ${esc(bridges.length)} · attempts ${esc((loop.attempts || []).length)}</div>
          <div>${next ? `Next: <b>${esc(next.kind)}</b> · ${esc(next.due_state?.due_on || next.due_on)} · ${esc(next.due_state?.due_status || 'planned')}` : 'Consolidation cycle complete'}</div>
        </button>`;
      }).join('') || '<div class="empty">Δεν υπάρχει ακόμη accepted Challenge με ενεργό Learning Loop.</div>';
      document.querySelectorAll('.loop-item').forEach((button) => button.addEventListener('click', () => openLoop(button.dataset.cycle)));
      setStatus('Learning Loop loaded');
    } catch (error) { showError(error); }
  }

  function attemptForOccurrence(loop, occurrenceId) {
    return (loop.attempts || []).filter((item) => String(item.occurrence_id) === String(occurrenceId));
  }

  function renderLoop(loop) {
    const plan = loop.plan || {};
    const objectives = plan.objectives || [];
    const bridges = plan.bridge_targets || [];
    const resources = loop.resources || [];
    const occurrences = plan.consolidation_occurrences || [];
    const occurrenceHtml = occurrences.map((occurrence) => {
      const attempts = attemptForOccurrence(loop, occurrence.occurrence_id);
      const latest = attempts.length ? attempts[attempts.length - 1] : null;
      const completed = occurrence?.due_state?.due_status === 'completed' || occurrence.status === 'completed';
      return `<div class="item consolidation-card" data-occurrence="${esc(occurrence.occurrence_id)}">
        <div class="item-head"><span class="item-title">${esc(occurrence.sequence)} · ${esc(occurrence.kind)}</span><span class="badge">${esc(occurrence?.due_state?.due_status || occurrence.status)}</span></div>
        <div class="muted">due ${esc(occurrence?.due_state?.due_on || occurrence.due_on)} · Foundation ${esc((occurrence.target_foundation_node_ids || []).join(' · ') || '—')}</div>
        <p class="test-prompt">${esc(occurrence.prompt)}</p>
        ${latest ? `<div class="result-block"><b>Τελευταία απάντηση</b><div>${esc(latest.response_text)}</div><div class="muted">result ${esc(latest.result)} · ${esc(String(latest.answered_at || '').slice(0, 16))}</div>${latest.evaluator_note ? `<div>${esc(latest.evaluator_note)}</div>` : ''}</div>` : ''}
        ${completed ? `<details><summary>Expected points / rubric</summary><div class="muted">${esc((occurrence.expected_points || []).join(' · ') || 'No fixed rubric')}</div></details>` : `
          <label class="field full"><span>Η απάντησή σου</span><textarea class="short-textarea consolidation-response" data-occurrence="${esc(occurrence.occurrence_id)}" placeholder="Απάντησε χωρίς να ανοίξεις το προηγούμενο Challenge."></textarea></label>
          <div class="row">
            <select class="consolidation-result" data-occurrence="${esc(occurrence.occurrence_id)}"><option value="not_assessed">not_assessed</option><option value="retained">retained</option><option value="partially_retained">partially_retained</option><option value="not_retained">not_retained</option><option value="improved_beyond_original">improved_beyond_original</option></select>
            <label class="check"><input class="consolidation-reviewed" data-occurrence="${esc(occurrence.occurrence_id)}" type="checkbox"> Clinician-reviewed result</label>
            <button class="primary submit-consolidation" data-occurrence="${esc(occurrence.occurrence_id)}">Save attempt</button>
          </div>`}
      </div>`;
    }).join('');

    $('loopDetail').innerHTML = `
      <div class="meta-grid"><div class="meta"><b>Challenge</b>${esc(loop.challenge_title || loop.challenge_id)}</div><div class="meta"><b>Revision</b>${esc(loop.source_revision)}</div><div class="meta"><b>Attempts</b>${esc((loop.attempts || []).length)}</div><div class="meta"><b>Resources</b>${esc(resources.length)}</div></div>
      <h3>Learning objectives</h3><div class="stack">${objectives.map((item) => `<div class="item"><b>${esc(item.title)}</b><div>${esc(item.rationale)}</div><div class="muted">${esc((item.foundation_node_ids || []).join(' · '))} · ${esc((item.gap_classes || []).join(' · '))}</div></div>`).join('') || '<div class="empty">No objectives</div>'}</div>
      <h3>Γεφύρωση νησίδων</h3><div class="stack">${bridges.map((item) => `<div class="item bridge-item"><div class="item-head"><b>${esc((item.foundation_node_ids || []).join(' ↔ '))}</b><span class="badge">${esc(item.state)}</span></div><div>${esc(item.rationale)}</div></div>`).join('') || '<div class="empty">No bridge targets</div>'}</div>
      <h3>Fresh / targeted resources</h3><div class="stack">${resources.map((resource) => `<div class="item resource-item"><div class="item-head"><b>${esc(resource.title)}</b><span class="badge">${esc(resource.status)}</span></div><div>${esc(resource.rationale)}</div><div class="muted">${esc(resource.provider)} · ${esc(resource.kind)} · access ${esc(resource.access_state)} · checked ${esc(String(resource.checked_at || '').slice(0, 10))}</div><a href="${esc(resource.url)}" target="_blank" rel="noopener noreferrer">Open resource</a></div>`).join('') || '<div class="empty">No fresh resources yet</div>'}</div>
      <h3>Repeated consolidation</h3><div class="stack">${occurrenceHtml}</div>`;

    document.querySelectorAll('.submit-consolidation').forEach((button) => button.addEventListener('click', () => submitAttempt(loop.cycle_id, button.dataset.occurrence)));
  }

  function openLoop(cycleId) {
    const loop = state.loops.find((candidate) => candidate.cycle_id === cycleId);
    if (!loop) return;
    state.selectedLoop = loop;
    $('loopDetailCard').hidden = false;
    renderLoop(loop);
    $('loopDetailCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function submitAttempt(cycleId, occurrenceId) {
    const response = document.querySelector(`.consolidation-response[data-occurrence="${CSS.escape(occurrenceId)}"]`)?.value.trim() || '';
    if (!response) {
      window.alert('Γράψε πρώτα την απάντησή σου.');
      return;
    }
    const result = document.querySelector(`.consolidation-result[data-occurrence="${CSS.escape(occurrenceId)}"]`)?.value || 'not_assessed';
    const clinician_reviewed = Boolean(document.querySelector(`.consolidation-reviewed[data-occurrence="${CSS.escape(occurrenceId)}"]`)?.checked);
    try {
      setStatus('Saving consolidation attempt…');
      await api(`/api/learning-loops/${encodeURIComponent(cycleId)}/occurrences/${encodeURIComponent(occurrenceId)}/attempts`, {
        method: 'POST',
        body: JSON.stringify({ response_text: response, result, clinician_reviewed }),
      });
      await loadLoops();
      const updated = state.loops.find((candidate) => candidate.cycle_id === cycleId);
      if (updated) openLoop(cycleId);
      setStatus('Consolidation saved · later repetitions preserved');
    } catch (error) { showError(error); }
  }

  $('refreshLoops')?.addEventListener('click', loadLoops);

  document.querySelectorAll('.tab').forEach((button) => {
    button.addEventListener('click', () => {
      if (button.dataset.view === 'inbox') loadInbox();
      if (button.dataset.view === 'loop') loadLoops();
    });
  });

  // The default surface is the Inbox. Existing L-1 app.js owns the generic tab
  // activation and Challenge/Foundation/History/Due workflows.
  installClipboardHandoffSurface();
  loadInbox();
})();
