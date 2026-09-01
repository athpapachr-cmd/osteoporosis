(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function lastTextSpan(button) {
    const spans = $$('span', button);
    return spans.length ? spans[spans.length - 1] : null;
  }

  function replaceFollowingText(strong, text) {
    if (!strong) return;
    let node = strong.nextSibling;
    while (node && node.nodeType !== 3) node = node.nextSibling;
    if (node) node.textContent = ` ${text} `;
  }

  function rewriteLegacyDynamicText() {
    const pill = $("#pilotPill");
    if (pill && /pilot/i.test(pill.textContent || "")) pill.textContent = "ΕΠΙΣΚΕΨΗ · DRAFT";

    const identity = $("#caseIdDisplay");
    if (identity && /^PILOT-/i.test(identity.textContent || "")) identity.textContent = "Νέα επίσκεψη";

    const status = $("#draftStatus");
    if (status && /pilot case/i.test(status.textContent || "")) {
      status.textContent = status.textContent
        .replace(/Νέο pilot case/ig, "Νέα επίσκεψη")
        .replace(/pilot case/ig, "επίσκεψη");
    }
  }

  function applyClinicalWorkspaceShell() {
    document.title = "Clinical Excellence — Osteoporosis";
    const description = $('meta[name="description"]');
    if (description) description.content = "Protected patient-centric osteoporosis clinical workspace.";

    const title = $(".title-block h1");
    if (title) title.textContent = "Clinical Excellence — Osteoporosis";
    const subtitle = $(".title-block > p");
    if (subtitle) subtitle.textContent = "Κλινική επίσκεψη · protected server-backed workspace";

    $$('[data-nav-action="new-case"]').forEach(button => {
      const text = lastTextSpan(button); if (text) text.textContent = "Νέα επίσκεψη";
    });
    $$('[data-nav-action="cases"]').forEach(button => {
      const text = lastTextSpan(button); if (text) text.textContent = "Επισκέψεις";
    });

    const banner = $(".baseline-banner");
    if (banner) {
      const strong = $("strong", banner); if (strong) strong.textContent = "Clinical Guidance ενεργή";
      const text = $("div > span", banner);
      if (text) text.textContent = "Η επίσκεψη καταγράφεται στον protected patient record. Routine performance feedback παραμένει κρυφό μέχρι το κατάλληλο measurement phase.";
    }

    const privacy = $("#privacyStrip");
    if (privacy) {
      const strong = $("strong", privacy);
      if (strong) {
        strong.textContent = "Protected clinical mode:";
        replaceFollowingText(strong, "Οι επισκέψεις συγχρονίζονται στον protected server μετά από authentication. Απόφυγε περιττά αναγνωριστικά σε free-text πεδία.");
      }
    }

    const caseCode = $("#caseIdDisplay")?.closest(".case-code");
    if (caseCode) {
      const label = $(".meta-label", caseCode); if (label) label.textContent = "Επίσκεψη";
    }

    const sampleBox = $(".sampling-box");
    if (sampleBox) sampleBox.hidden = true;

    const cancel = $("#cancelCaseBtn");
    if (cancel) cancel.hidden = true;

    const quickNotesHelp = $(".notes-card .card-heading p");
    if (quickNotesHelp) quickNotesHelp.textContent = "Μόνο κάτι που αξίζει να θυμάσαι από τη σημερινή επίσκεψη.";

    const derived = $(".derived-note");
    if (derived) derived.textContent = "BMI, απώλεια ύψους, recency κατάγματος, recurrent falls και dose/duration bands χρησιμοποιούνται στο background για longitudinal applicability· δεν εμφανίζονται ως performance score.";

    const casesDialog = $("#casesDialog");
    if (casesDialog) {
      const heading = $(".modal-head h2", casesDialog); if (heading) heading.textContent = "Legacy browser cache";
      const help = $(".modal-head p", casesDialog); if (help) help.textContent = "Δεν είναι η πηγή αλήθειας. Οι κλινικές επισκέψεις ανοίγουν από τον protected patient record.";
    }

    const privacyDialog = $("#privacyDialog");
    if (privacyDialog) {
      const heading = $(".modal-head h2", privacyDialog); if (heading) heading.textContent = "Privacy — clinical workspace";
      const help = $(".modal-head p", privacyDialog); if (help) help.textContent = "Το public GitHub repository δεν είναι clinical data store. Η κλινική ροή χρησιμοποιεί protected server storage μετά από authentication.";
      const items = $$(".privacy-list li", privacyDialog);
      const copy = [
        "Χρησιμοποίησε το protected Patient ID και απόφυγε περιττά αναγνωριστικά σε free-text πεδία.",
        "Μην επικολλάς raw Heidi transcript σε αυτή τη φόρμα· transcript-assisted capture έχει ξεχωριστό privacy boundary.",
        "Το browser cache είναι προσωρινό working cache και όχι η authoritative patient record.",
        "Η ύπαρξη protected clinical routes δεν αποτελεί από μόνη της πλήρη GDPR/privacy certification του συνολικού συστήματος."
      ];
      items.forEach((item, index) => { if (copy[index]) item.textContent = copy[index]; });
    }

    rewriteLegacyDynamicText();
    [$("#pilotPill"), $("#caseIdDisplay"), $("#draftStatus")].filter(Boolean).forEach(node => {
      const observer = new MutationObserver(rewriteLegacyDynamicText);
      observer.observe(node, { childList: true, characterData: true, subtree: true });
    });
  }

  applyClinicalWorkspaceShell();
})();
