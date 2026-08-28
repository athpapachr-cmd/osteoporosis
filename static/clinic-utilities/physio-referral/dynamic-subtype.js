(() => {
  'use strict';

  const postoperativeSubtypes = new Set([
    'extensor_tendon_repair_postoperative',
    'flexor_tendon_repair_postoperative',
  ]);

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

  document.addEventListener('change', (event) => {
    if (event.target?.id === 'subtypeSelect') syncSubtypeDependentWording();
  });
})();
