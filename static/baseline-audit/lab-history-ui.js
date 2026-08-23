(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);

  const LAB_VALUE_IDS = [
    "#s3Ca", "#s3Phos", "#s3VitD", "#s3Pth", "#s3Creat", "#s3Egfr",
    "#s3Urea", "#s3Alp", "#s3Mg", "#s3Ctx", "#s3P1np", "#s3BoneAlp",
    "#s3Osteocalcin", "#s3Glucose", "#s3Hba1c", "#s3Tsh", "#s3Ft4",
    "#s3Esr", "#s3Crp", "#s3Testosterone", "#s3Fsh", "#s3Estradiol",
    "#s3Cortisol", "#s3UrineCa"
  ];

  const LAB_SELECT_IDS = ["#s3BtmContext"];
  const LAB_CHECK_IDS = ["#s3Cbc", "#s3Liver", "#s3Celiac", "#s3Spep"];

  function dispatchValueChange(node) {
    if (!node) return;
    node.dispatchEvent(new Event("input", { bubbles: true }));
    node.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function clearCurrentLabEntry() {
    const date = $("#s3LabsDate");
    if (!date) return;

    date.value = "";
    dispatchValueChange(date);

    [...LAB_VALUE_IDS, ...LAB_SELECT_IDS].forEach((selector) => {
      const node = $(selector);
      if (!node) return;
      node.value = "";
      dispatchValueChange(node);
    });

    LAB_CHECK_IDS.forEach((selector) => {
      const node = $(selector);
      if (!node) return;
      node.checked = false;
      dispatchValueChange(node);
    });

    date.focus();
  }

  function removeRegistryLabTable() {
    // The longitudinal laboratory table belongs only in Step 3. The patient
    // registry keeps counts/timeline, but no duplicate persistent table.
    $("#clinicalLabHistory")?.remove();
  }

  function injectNewLabsButton() {
    if ($("#clinicalNewLabsBtn")) return;
    const date = $("#s3LabsDate");
    const card = date?.closest("article");
    if (!date || !card) return;

    const dateGrid = date.closest(".step3-top-grid") || date.parentElement;
    if (!dateGrid) return;

    const wrap = document.createElement("div");
    wrap.className = "clinical-new-labs-action";
    wrap.style.display = "flex";
    wrap.style.alignItems = "end";
    wrap.style.gap = "8px";

    const button = document.createElement("button");
    button.id = "clinicalNewLabsBtn";
    button.type = "button";
    button.className = "btn secondary";
    button.textContent = "＋ Νέες αναλύσεις";
    button.title = "Καθαρίζει μόνο την τρέχουσα φόρμα εργαστηριακών για νέα ημερομηνία. Το ιστορικό παραμένει αποθηκευμένο.";
    button.addEventListener("click", clearCurrentLabEntry);

    wrap.appendChild(button);
    dateGrid.appendChild(wrap);
  }

  function sync() {
    removeRegistryLabTable();
    injectNewLabsButton();
  }

  sync();
  setTimeout(sync, 0);
  setTimeout(sync, 250);

  // Patient Registry and Step 3 are dynamically hydrated/re-rendered. Keep the
  // UI invariant without touching the underlying laboratory persistence model.
  const observer = new MutationObserver(() => sync());
  observer.observe(document.body, { childList: true, subtree: true });
})();
