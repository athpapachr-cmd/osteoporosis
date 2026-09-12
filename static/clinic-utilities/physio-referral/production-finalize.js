'use strict';
// Production-only export behavior. The tested prototype intentionally stamps
// exported text as synthetic; the protected Cockpit surface must not. Clinical
// projection/state remains owned by the same deterministic server path.

// Load the bounded v4 clinical-picture presentation after the inherited product
// scripts. Keep the old qualifier DOM as a hidden compatibility backing surface;
// v4 only changes how those same fields are reached and displayed.
const v4ClinicalHost = $('#phenotype');
if (v4ClinicalHost) v4ClinicalHost.style.visibility = 'hidden';
const v4Style = document.createElement('link');
v4Style.rel = 'stylesheet';
v4Style.href = '/static/clinic-utilities/physio-referral/product-clinical-sheet-v4.css';
document.head.append(v4Style);
const v4Script = document.createElement('script');
v4Script.src = '/static/clinic-utilities/physio-referral/product-clinical-sheet-v4.js';
v4Script.async = false;
v4Script.onload = () => { if (v4ClinicalHost) v4ClinicalHost.style.visibility = ''; };
v4Script.onerror = () => {
  if (v4ClinicalHost) v4ClinicalHost.style.visibility = '';
  notice('Η συμπαγής κλινική εικόνα δεν φορτώθηκε. Ανανέωσε τη σελίδα.');
};
document.head.append(v4Script);

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
