'use strict';

(() => {
  if (/^#local_visit_audio_v1=[A-Za-z0-9_-]{43}$/.test(location.hash)) return;

  const source = new URL(location.href);
  const target = new URL('/static/cockpit/index.html', location.origin);
  target.searchParams.set('capture', '1');
  const id = source.searchParams.get('patient_id');
  if (id && id.length <= 120) target.searchParams.set('entry_patient', id);
  location.replace(target.href);
})();
