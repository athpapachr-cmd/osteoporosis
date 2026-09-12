'use strict';
// Production-only export behavior. The tested prototype intentionally stamps
// exported text as synthetic; the protected Cockpit surface must not. Clinical
// projection/state remains owned by the same deterministic server path.

// Load the bounded v5 clinical-picture interaction after the inherited product
// scripts. The v4 layout CSS and legacy qualifier DOM remain compatibility
// backing surfaces; v5 only changes how the same structured fields are reached.
const v4ClinicalHost = $('#phenotype');
if (v4ClinicalHost) v4ClinicalHost.style.visibility = 'hidden';
const v4Style = document.createElement('link');
v4Style.rel = 'stylesheet';
v4Style.href = '/static/clinic-utilities/physio-referral/product-clinical-sheet-v4.css';
document.head.append(v4Style);
const v5Script = document.createElement('script');
v5Script.src = '/static/clinic-utilities/physio-referral/product-clinical-sheet-v5.js';
v5Script.async = false;
v5Script.onload = () => { if (v4ClinicalHost) v4ClinicalHost.style.visibility = ''; };
v5Script.onerror = () => {
  if (v4ClinicalHost) v4ClinicalHost.style.visibility = '';
  notice('Η συμπαγής κλινική εικόνα δεν φορτώθηκε. Ανανέωσε τη σελίδα.');
};
document.head.append(v5Script);

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
window.addEventListener('beforeprint', preparePrint);
