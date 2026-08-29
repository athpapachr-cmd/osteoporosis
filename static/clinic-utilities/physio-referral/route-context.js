(() => {
  'use strict';

  const API_CONTRACT = '/clinical/clinic-utilities/physio-referral/api/contract';
  let contract = null;

  const routeSelect = document.getElementById('routeSelect');
  const wordingSelect = document.getElementById('wordingSelect');
  const contextCard = document.getElementById('contextCard');
  const contextFields = document.getElementById('contextFields');

  if (!routeSelect || !wordingSelect || !contextCard || !contextFields) return;

  const escapeHtml = (text) => String(text ?? '').replace(/[&<>'"]/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
  }[c]));

  function shouldShow(field) {
    const showWhen = field?.show_when;
    if (!showWhen) return true;
    const modes = showWhen.wording_modes;
    if (Array.isArray(modes) && modes.length && !modes.includes(wordingSelect.value)) return false;
    return true;
  }

  function renderEnumField(key, field) {
    const values = Array.isArray(field.values) ? field.values : [];
    const labels = field.value_labels_el || {};
    const requiredMark = field.required === true ? ' <span class="required-mark">*</span>' : '';
    const options = values.map((value) => {
      const display = labels[value];
      if (typeof display !== 'string' || !display.trim()) return '';
      return `<option value="${escapeHtml(value)}">${escapeHtml(display)}</option>`;
    }).join('');
    return `<label data-route-context-generated="true">${escapeHtml(field.label_el || key)}${requiredMark}<select data-context-key="${escapeHtml(key)}"><option value="">Επίλεξε…</option>${options}</select></label>`;
  }

  function render() {
    contextFields.querySelectorAll('[data-route-context-generated="true"]').forEach((node) => node.remove());
    if (!contract || !routeSelect.value) return;

    const fields = contract.route_context_intake?.routes?.[routeSelect.value]?.fields || {};
    const rendered = [];
    Object.entries(fields).forEach(([key, field]) => {
      if (!field || !shouldShow(field)) return;
      if (field.type === 'enum') rendered.push(renderEnumField(key, field));
    });

    if (rendered.length) {
      contextFields.insertAdjacentHTML('beforeend', rendered.join(''));
      contextCard.hidden = false;
    }
  }

  async function load() {
    try {
      const response = await fetch(API_CONTRACT, {credentials: 'same-origin'});
      if (!response.ok) return;
      contract = await response.json();
      render();
    } catch (_) {
      // The primary app owns request-error presentation; this enhancement fails closed/silent.
    }
  }

  routeSelect.addEventListener('change', () => queueMicrotask(render));
  wordingSelect.addEventListener('change', () => queueMicrotask(render));
  load();
})();
