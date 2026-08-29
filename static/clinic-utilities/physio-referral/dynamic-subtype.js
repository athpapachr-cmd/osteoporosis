(() => {
  'use strict';

  const postoperativeSubtypes = new Set([
    'extensor_tendon_repair_postoperative',
    'flexor_tendon_repair_postoperative',
  ]);

  const directionLabels = {
    recommend: 'Σύσταση',
    consider: 'Να εξεταστεί',
    may_consider: 'Μπορεί να εξεταστεί',
    do_not_offer: 'Να μην προσφέρεται',
    insufficient_evidence: 'Ανεπαρκής τεκμηρίωση',
    context_only: 'Πλαίσιο / περιορισμός',
  };
  const domainLabels = {
    diagnostic_definition: 'Διαγνωστικό πλαίσιο',
    history: 'Ιστορικό',
    examination: 'Αξιολόγηση',
    core_rehabilitation: 'Κύρια αποκατάσταση',
    rehab_phase: 'Στάδιο αποκατάστασης',
    progression_criteria: 'Πρόοδος / επανένταξη',
    adjunct: 'Συμπληρωματικά μέσα',
    safety: 'Ασφάλεια / επανεκτίμηση',
    differential: 'Διαφορική',
    execution_detail: 'Λεπτομέρεια εκτέλεσης',
  };
  const scopeLabels = {
    referral_core: 'Υποστηρίζει το παραπεμπτικό',
    clinician_ui_only: 'Μόνο για κλινική ενημέρωση',
    therapist_execution_detail: 'Λεπτομέρεια εκτέλεσης φυσιοθεραπευτή',
  };
  const freshnessLabels = {
    current: 'τρέχουσα',
    review_due: 'χρειάζεται επανέλεγχο',
    stale: 'παρωχημένη / προς επανέλεγχο',
    superseded: 'αντικαταστάθηκε',
  };

  const escapeHtml = (text) => String(text ?? '').replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const humanize = (text) => String(text || '').replaceAll('_', ' ');

  function syncSubtypeDependentWording() {
    const subtype = document.getElementById('subtypeSelect');
    const wording = document.getElementById('wordingSelect');
    if (!subtype || !wording) return;

    if (postoperativeSubtypes.has(subtype.value)) {
      const postoperativeOption = [...wording.options].some((option) => option.value === 'postoperative');
      if (postoperativeOption && wording.value !== 'postoperative') {
        wording.value = 'postoperative';
        wording.dispatchEvent(new Event('change', {bubbles: true}));
      }
    }
  }

  function ensureEvidencePanel() {
    let panel = document.getElementById('clinicianEvidencePanel');
    if (panel) return panel;
    const outputSticky = document.querySelector('.output-sticky');
    if (!outputSticky) return null;

    panel = document.createElement('details');
    panel.id = 'clinicianEvidencePanel';
    panel.className = 'clinician-evidence-panel';
    panel.innerHTML = `
      <summary><strong>Τεκμηρίωση / Παραπομπές</strong><span id="evidenceBadge" class="evidence-badge">—</span></summary>
      <div id="evidenceContent" class="evidence-content"><p class="evidence-muted">Επίλεξε πάθηση για να εμφανιστεί η σχετική τεκμηρίωση.</p></div>
    `;
    outputSticky.appendChild(panel);

    const style = document.createElement('style');
    style.textContent = `
      .clinician-evidence-panel{margin-top:14px;border:1px solid rgba(80,100,120,.22);border-radius:12px;background:rgba(255,255,255,.72);overflow:hidden}
      .clinician-evidence-panel>summary{display:flex;justify-content:space-between;gap:10px;align-items:center;cursor:pointer;padding:12px 14px;list-style:none}
      .clinician-evidence-panel>summary::-webkit-details-marker{display:none}
      .evidence-badge{font-size:.76rem;font-weight:700;padding:3px 8px;border-radius:999px;background:rgba(50,90,120,.08);white-space:nowrap}
      .evidence-content{padding:0 14px 14px;font-size:.88rem;line-height:1.45}
      .evidence-muted{opacity:.7;margin:.45rem 0}
      .evidence-alert{padding:9px 10px;border-left:3px solid currentColor;background:rgba(120,90,40,.06);margin:8px 0}
      .evidence-section{margin-top:13px}
      .evidence-section h4{margin:0 0 7px;font-size:.9rem}
      .evidence-source{padding:8px 0;border-top:1px solid rgba(80,100,120,.13)}
      .evidence-source:first-of-type{border-top:0}
      .evidence-source-title{font-weight:700}
      .evidence-meta{font-size:.78rem;opacity:.76;margin-top:2px}
      .evidence-source a{font-size:.78rem}
      .evidence-claim{padding:8px 0;border-top:1px solid rgba(80,100,120,.13)}
      .evidence-claim:first-of-type{border-top:0}
      .evidence-tags{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px}
      .evidence-tag{font-size:.7rem;padding:2px 6px;border-radius:999px;background:rgba(50,90,120,.08)}
      .evidence-gap-list{margin:5px 0 0;padding-left:18px}
    `;
    document.head.appendChild(style);
    return panel;
  }

  function sourceLink(source) {
    const url = source.url || (source.doi ? `https://doi.org/${encodeURIComponent(source.doi)}` : '');
    if (!url) return '';
    return `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">Άνοιγμα πηγής ↗</a>`;
  }

  function renderEvidence(data) {
    ensureEvidencePanel();
    const content = document.getElementById('evidenceContent');
    const badge = document.getElementById('evidenceBadge');
    if (!content || !badge) return;

    const sources = data.sources || [];
    const claims = data.claims || [];
    const gaps = data.evidence_gaps || [];
    const status = data.sequence_status || data.coverage_status || (data.has_applicable_profile ? 'profile' : 'χωρίς profile');
    const subtypeRequired = data.selection_state === 'subtype_required_for_evidence';
    badge.textContent = subtypeRequired
      ? 'χρειάζεται υπότυπος'
      : sources.length
        ? `${sources.length} πηγ${sources.length === 1 ? 'ή' : 'ές'}`
        : 'χωρίς πηγές';

    const warning = subtypeRequired
      ? '<div class="evidence-alert"><strong>Επίλεξε υπότυπο για την τεκμηρίωση.</strong> Οι βιβλιογραφικές πηγές δεν αναμειγνύονται μεταξύ κλινικά διαφορετικών υποτύπων.</div>'
      : !data.has_applicable_profile
        ? '<div class="evidence-alert"><strong>Δεν υπάρχει εφαρμοστέο structured evidence profile.</strong> Δεν πρέπει να χρησιμοποιηθεί generic evidence fallback.</div>'
        : (String(status).includes('blocked') || String(status).includes('incomplete'))
          ? `<div class="evidence-alert"><strong>Περιορισμένη route coverage:</strong> ${escapeHtml(humanize(status))}. Το κενό τεκμηρίωσης δεν πρέπει να συμπληρώνεται με δανεισμένο protocol.</div>`
          : '';

    const sourceHtml = sources.length ? sources.map((source) => {
      const meta = [source.authors_or_organization, source.year_or_version, source.reference].filter(Boolean).map(escapeHtml).join(' · ');
      const freshness = source.freshness_state ? ` · ${escapeHtml(freshnessLabels[source.freshness_state] || source.freshness_state)}` : '';
      return `<div class="evidence-source">
        <div class="evidence-source-title">${escapeHtml(source.title || 'Πηγή')}</div>
        <div class="evidence-meta">${meta}${freshness}</div>
        ${source.population_scope ? `<div class="evidence-meta">Πληθυσμός: ${escapeHtml(source.population_scope)}</div>` : ''}
        ${sourceLink(source)}
      </div>`;
    }).join('') : subtypeRequired
      ? '<p class="evidence-muted">Η βιβλιογραφία θα εμφανιστεί αφού επιλεγεί ο κατάλληλος υπότυπος.</p>'
      : '<p class="evidence-muted">Δεν έχουν επιλυθεί ανθρώπινα αναγνώσιμες πηγές για αυτή την επιλογή.</p>';

    const claimHtml = claims.length ? claims.map((claim) => {
      const tags = [
        domainLabels[claim.domain] || humanize(claim.domain),
        directionLabels[claim.recommendation_direction] || humanize(claim.recommendation_direction),
        scopeLabels[claim.output_scope] || humanize(claim.output_scope),
        claim.strength ? `Strength: ${claim.strength}` : '',
        claim.certainty ? `Certainty: ${claim.certainty}` : '',
      ].filter(Boolean).map((value) => `<span class="evidence-tag">${escapeHtml(value)}</span>`).join('');
      const conditions = (claim.applicability_conditions || []).length
        ? `<div class="evidence-meta">Ισχύει υπό: ${(claim.applicability_conditions || []).map(humanize).map(escapeHtml).join(' · ')}</div>`
        : '';
      return `<div class="evidence-claim"><div class="evidence-tags">${tags}</div><div>${escapeHtml(claim.claim_summary || '')}</div>${conditions}</div>`;
    }).join('') : '<p class="evidence-muted">Δεν υπάρχουν resolved claims για την τρέχουσα επιλογή.</p>';

    const gapsHtml = gaps.length
      ? `<div class="evidence-section"><h4>Κενά / περιορισμοί τεκμηρίωσης</h4><ul class="evidence-gap-list">${gaps.map((gap) => `<li>${escapeHtml(humanize(gap))}</li>`).join('')}</ul></div>`
      : '';

    content.innerHTML = `${warning}
      <div class="evidence-section"><h4>Βασικές πηγές</h4>${sourceHtml}</div>
      <details class="evidence-section"><summary><strong>Τι υποστηρίζουν οι πηγές</strong></summary>${claimHtml}</details>
      ${gapsHtml}`;
  }

  function renderEvidenceError(message) {
    ensureEvidencePanel();
    const content = document.getElementById('evidenceContent');
    const badge = document.getElementById('evidenceBadge');
    if (badge) badge.textContent = 'σφάλμα';
    if (content) content.innerHTML = `<div class="evidence-alert">Δεν φορτώθηκε η τεκμηρίωση: ${escapeHtml(message)}</div>`;
  }

  async function refreshEvidence() {
    ensureEvidencePanel();
    const profile = document.getElementById('profileSelect');
    const route = document.getElementById('routeSelect');
    const subtype = document.getElementById('subtypeSelect');
    const content = document.getElementById('evidenceContent');
    const badge = document.getElementById('evidenceBadge');
    if (!profile?.value || !route?.value) {
      if (badge) badge.textContent = '—';
      if (content) content.innerHTML = '<p class="evidence-muted">Επίλεξε πάθηση για να εμφανιστεί η σχετική τεκμηρίωση.</p>';
      return;
    }
    if (badge) badge.textContent = 'φόρτωση…';
    if (content) content.innerHTML = '<p class="evidence-muted">Φόρτωση route-specific τεκμηρίωσης…</p>';
    try {
      const response = await fetch('/clinical/clinic-utilities/physio-referral/api/evidence', {
        method: 'POST',
        credentials: 'same-origin',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          profile_id: profile.value,
          route_id: route.value,
          subtype_id_optional: subtype?.value || null,
        }),
      });
      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try { detail = (await response.json()).detail || detail; } catch (_) {}
        throw new Error(detail);
      }
      renderEvidence(await response.json());
    } catch (error) {
      renderEvidenceError(error.message || String(error));
    }
  }

  document.addEventListener('change', (event) => {
    if (event.target?.id === 'subtypeSelect') syncSubtypeDependentWording();
    if (['profileSelect', 'routeSelect', 'subtypeSelect', 'wordingSelect'].includes(event.target?.id)) {
      setTimeout(refreshEvidence, 0);
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      ensureEvidencePanel();
      refreshEvidence();
    });
  } else {
    ensureEvidencePanel();
    refreshEvidence();
  }
})();
