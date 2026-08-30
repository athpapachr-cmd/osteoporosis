(() => {
  'use strict';

  const API_CONTRACT = '/clinical/clinic-utilities/physio-referral/api/contract';
  const SECTIONS = ['findings', 'functional_impairments', 'goals', 'rehab_directions', 'adjuncts'];
  const SECTION_UI = {
    findings: {container: 'findingsOptions', block: 'findingsBlock', group: 'finding', labelSection: 'findings'},
    functional_impairments: {container: 'functionOptions', block: 'functionBlock', group: 'functional', labelSection: 'functional_impairments'},
    goals: {container: 'goalOptions', block: 'goalsBlock', group: 'goal', labelSection: 'goals'},
    rehab_directions: {container: 'rehabOptions', block: 'rehabBlock', group: 'rehab', labelSection: 'rehab_directions'},
    adjuncts: {container: 'adjunctOptions', block: 'adjunctBlock', group: 'adjunct', labelSection: 'adjuncts'},
  };

  const profileSelect = document.getElementById('profileSelect');
  const routeSelect = document.getElementById('routeSelect');
  const subtypeSelect = document.getElementById('subtypeSelect');
  const wordingSelect = document.getElementById('wordingSelect');
  const contextFields = document.getElementById('contextFields');
  const findingsCard = document.getElementById('findingsCard');
  const rehabCard = document.getElementById('rehabCard');

  if (!profileSelect || !routeSelect || !subtypeSelect || !wordingSelect || !contextFields) return;

  let contract = null;

  const escapeHtml = (text) => String(text ?? '').replace(/[&<>\'\"]/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
  }[c]));

  function label(id, section) {
    const value = contract?.display_labels?.[section]?.[id];
    return typeof value === 'string' && value.trim() ? value : String(id).replaceAll('_', ' ');
  }

  function cloneScope(base) {
    const result = {};
    SECTIONS.forEach((section) => {
      result[section] = Array.isArray(base?.[section]) ? [...base[section]] : [];
    });
    return result;
  }

  function unique(items) {
    return [...new Set((items || []).filter((item) => typeof item === 'string' && item))];
  }

  function prioritized(ids, preferred) {
    const all = unique(ids);
    const first = unique(preferred).filter((id) => all.includes(id));
    return [...first, ...all.filter((id) => !first.includes(id))];
  }

  function applyLayer(scope, layer) {
    if (!layer || typeof layer !== 'object') return scope;
    const replace = layer.replace || {};
    const include = layer.include || {};
    const exclude = layer.exclude || {};
    const prioritize = layer.prioritize || {};

    SECTIONS.forEach((section) => {
      if (Array.isArray(replace[section])) scope[section] = unique(replace[section]);
      if (Array.isArray(include[section])) scope[section] = unique([...scope[section], ...include[section]]);
      if (Array.isArray(exclude[section])) {
        const blocked = new Set(exclude[section]);
        scope[section] = scope[section].filter((id) => !blocked.has(id));
      }
      if (Array.isArray(prioritize[section])) scope[section] = prioritized(scope[section], prioritize[section]);
    });
    return scope;
  }

  function collectContext() {
    const context = {};
    contextFields.querySelectorAll('[data-context-key]').forEach((node) => {
      if (!node.value) return;
      const key = node.dataset.contextKey;
      if (!key) return;
      context[key] = node.value;
    });
    return context;
  }

  function matches(rule, context) {
    if (!rule || typeof rule !== 'object') return true;
    const modes = rule.wording_modes;
    if (Array.isArray(modes) && modes.length && !modes.includes(wordingSelect.value)) return false;
    const equals = rule.context_equals || {};
    return Object.entries(equals).every(([key, value]) => context[key] === value);
  }

  function resolvedScope() {
    const base = contract?.ui_relevance_scope?.profiles?.[profileSelect.value] || {};
    const scope = cloneScope(base);

    const legacy = contract?.ui_relevance_scope?.route_adjustments?.[profileSelect.value]?.[routeSelect.value] || {};
    if (Array.isArray(legacy.prioritize_adjuncts)) {
      scope.adjuncts = prioritized(scope.adjuncts, legacy.prioritize_adjuncts);
    }

    const route = contract?.ui_relevance_hierarchy?.routes?.[profileSelect.value]?.[routeSelect.value];
    if (!route) return scope;

    applyLayer(scope, route);

    const subtype = subtypeSelect.value;
    if (subtype && route.subtypes?.[subtype]) applyLayer(scope, route.subtypes[subtype]);

    const context = collectContext();
    (route.context_variants || []).forEach((variant) => {
      if (matches(variant.match, context)) applyLayer(scope, variant);
    });
    return scope;
  }

  function checkedValues(group) {
    return new Set(
      [...document.querySelectorAll(`input[data-group="${group}"]:checked`)].map((node) => node.value),
    );
  }

  function renderSection(section, ids) {
    const ui = SECTION_UI[section];
    const container = document.getElementById(ui.container);
    const block = document.getElementById(ui.block);
    if (!container || !block) return;

    const previouslyChecked = checkedValues(ui.group);
    container.innerHTML = (ids || []).map((id) => `
      <label class="check-item"><input type="checkbox" data-group="${ui.group}" value="${escapeHtml(id)}" ${previouslyChecked.has(id) ? 'checked' : ''}/> <span>${escapeHtml(label(id, ui.labelSection))}</span></label>
    `).join('') || '<span class="empty-note">Δεν υπάρχουν σχετικές επιλογές.</span>';
    block.hidden = !(ids || []).length;
  }

  function render() {
    if (!contract || !profileSelect.value || !routeSelect.value) return;
    const scope = resolvedScope();
    SECTIONS.forEach((section) => renderSection(section, scope[section] || []));

    const hasFindings = scope.findings.length || scope.functional_impairments.length;
    const hasRehab = scope.goals.length || scope.rehab_directions.length || scope.adjuncts.length;
    if (findingsCard) findingsCard.hidden = !hasFindings;
    if (rehabCard) rehabCard.hidden = !hasRehab;
  }

  async function load() {
    try {
      const response = await fetch(API_CONTRACT, {credentials: 'same-origin'});
      if (!response.ok) return;
      contract = await response.json();
      render();
    } catch (_) {
      // Presentation enhancement only; primary app owns request-error handling.
    }
  }

  profileSelect.addEventListener('change', () => queueMicrotask(render));
  routeSelect.addEventListener('change', () => queueMicrotask(render));
  subtypeSelect.addEventListener('change', () => queueMicrotask(render));
  wordingSelect.addEventListener('change', () => queueMicrotask(render));
  contextFields.addEventListener('change', () => queueMicrotask(render));

  load();
})();
