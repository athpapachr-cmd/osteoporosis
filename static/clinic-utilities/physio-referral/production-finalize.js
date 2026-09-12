'use strict';
// Production-only export behavior. The tested prototype intentionally stamps
// exported text as synthetic; the protected Cockpit surface must not. Clinical
// projection/state remains owned by the same deterministic server path.
const productionOpenSheet = openSheet;
openSheet = function(type, item = null, back = null) {
  const result = productionOpenSheet(type, item, back);
  if (type === 'copyFallback') {
    const area = $('#sheetBody textarea');
    if (area) area.value = effectiveText();
  }
  return result;
};

copyReferral = async function() {
  if (!canExport()) return notice('Η παραπομπή δεν είναι έτοιμη για εξαγωγή.');
  const atRevision = revision;
  try {
    await navigator.clipboard.writeText(effectiveText());
    notice(atRevision === revision ? 'Αντιγράφηκε το κείμενο.' : 'Αντιγράφηκε προηγούμενη εκδοχή. Οι επιλογές άλλαξαν.');
  } catch (_error) {
    if (canExport()) openSheet('copyFallback');
  }
};

preparePrint = function() {
  $('#printArea').textContent = canExport() ? effectiveText() : 'Η παραπομπή δεν είναι διαθέσιμη για εξαγωγή.';
};
// app.js registered its prototype beforeprint handler first; this later handler
// deliberately overwrites the print area with production text at the same event.
window.addEventListener('beforeprint', preparePrint);
