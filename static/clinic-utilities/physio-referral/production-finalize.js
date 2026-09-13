'use strict';
// Production-only export behavior. Clinical projection/state remains owned by
// the shared deterministic server path. The protected page now loads the
// clinical-sheet V4 and V5.1 presentation layers explicitly in deterministic
// order before this final production-only export layer.

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
