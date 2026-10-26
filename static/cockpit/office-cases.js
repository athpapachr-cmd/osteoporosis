(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const money = new Intl.NumberFormat("el-CY", {style:"currency", currency:"EUR"});
  const recordedTime = new Intl.DateTimeFormat("el-CY", {timeZone:"Asia/Nicosia",
    year:"numeric", month:"2-digit", day:"2-digit", hour:"2-digit", minute:"2-digit", hourCycle:"h23"});
  const stageNames = {issued:"Εκδόθηκε", submitted:"Υποβλήθηκε", approved:"Εγκρίθηκε",
    rejected:"Απορρίφθηκε", scheduled:"Προγραμματίστηκε", performed:"Πραγματοποιήθηκε"};
  let activeTab = "surgery", generation = 0, detailGeneration = 0,
    caseRows = [], caseOffset = 0, selectedCase = null;
  let rfRows = [], rfCursor = "", pendingPaymentId = "", pendingPaymentCaseId = "";
  let needsPaymentReconciliation = false;

  const show = (id, value) => { $(id).textContent = value; };
  function euro(cents) { return cents === null || cents === undefined ? "Άγνωστο" : money.format(cents / 100); }
  function cyprusTimestamp(value) {
    if (!value || typeof value !== "string") return "Άγνωστη ώρα";
    // API timestamps are naive UTC, because PostgreSQL columns are timestamp without time zone.
    const normalized = /(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? value : value + "Z";
    const parsed = new Date(normalized);
    return Number.isNaN(parsed.getTime()) ? "Άγνωστη ώρα" : recordedTime.format(parsed);
  }
  function cents(value) {
    const match = String(value || "").trim().replace(",", ".").match(/^(\d+)(?:\.(\d{1,2}))?$/);
    if (!match) throw new Error("Συμπλήρωσε ποσό με έως δύο δεκαδικά.");
    const amount = Number(match[1]) * 100 + Number((match[2] || "").padEnd(2, "0"));
    if (!Number.isSafeInteger(amount) || amount > 2147483647) throw new Error("Το ποσό είναι πολύ μεγάλο.");
    return amount;
  }
  function node(tag, value, cls="") {
    const element = document.createElement(tag);
    element.textContent = value;
    if (cls) element.className = cls;
    return element;
  }
  function button(label, handler) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "button secondary";
    item.textContent = label;
    item.addEventListener("click", handler);
    return item;
  }
  async function api(path, options={}) {
    const response = await fetch(path, {
      credentials:"same-origin", cache:"no-store",
      headers:{"Content-Type":"application/json"}, ...options
    });
    if (!response.ok) {
      const error = new Error("HTTP " + response.status);
      error.status = response.status;
      try { error.detail = (await response.json()).detail || ""; } catch (_) { error.detail = ""; }
      throw error;
    }
    return response.json();
  }
  function errorText(error) {
    if ([401,403].includes(error.status)) {
      clear();
      window.dispatchEvent(new Event("clinical:unauthenticated"));
      return "Απαιτείται Clinical Key. Πάτησε «Πρόσβαση» πάνω δεξιά.";
    }
    return error.status === 409 ? (error.detail || "Τα στοιχεία άλλαξαν. Ανανέωσε και δοκίμασε ξανά.")
      : error.message && !error.status ? error.message : "Η προστατευμένη καταχώριση δεν ολοκληρώθηκε.";
  }
  function setAccidentWritesEnabled(caseEnabled, paymentEnabled=false) {
    for (const control of $("accidentForm").elements) control.disabled = !caseEnabled;
    for (const control of $("accidentPaymentForm").elements) control.disabled = !paymentEnabled;
  }
  function clear() {
    if (pendingPaymentId) needsPaymentReconciliation = true;
    ++generation;
    ++detailGeneration;
    caseRows = []; caseOffset = 0; selectedCase = null; rfRows = []; rfCursor = "";
    pendingPaymentId = ""; pendingPaymentCaseId = "";
    resetAccidentForm();
    $("accidentPaymentForm").reset();
    $("officeRfSearch").reset();
    $("paymentReviewConfirm").hidden = true;
    setAccidentWritesEnabled(false);
    $("accidentList").replaceChildren(); $("accidentDetail").hidden = true;
    $("accidentNext").hidden = true; $("officeRfList").replaceChildren(); $("officeRfNext").hidden = true;
    show("accidentState", "Σύνδεση απαιτείται"); show("officeRfState", "Σύνδεση απαιτείται");
    show("officeRfNote", "");
    show("accidentNote", needsPaymentReconciliation
      ? "Η προηγούμενη πληρωμή ίσως καταχωρίστηκε. Μετά τη σύνδεση, άνοιξε το περιστατικό και έλεγξε το ιστορικό πριν από νέα καταχώριση."
      : "");
  }
  function selectTab(tab, conflict=false) {
    if (!["surgery","accidents","rf"].includes(tab)) return;
    if (tab !== activeTab) { ++generation; ++detailGeneration; }
    activeTab = tab;
    document.querySelectorAll("[data-office-tab]").forEach(item => {
      const current = item.dataset.officeTab === tab;
      item.classList.toggle("selected", current);
      item.setAttribute("aria-pressed", String(current));
    });
    document.querySelectorAll("[data-office-tab-panel]").forEach(item => {
      item.hidden = item.dataset.officeTabPanel !== tab;
    });
    if (tab === "accidents") return loadAccidents();
    if (tab === "rf") return loadRF(true, conflict);
  }
  function open() { selectTab(activeTab); }

  function resetAccidentForm() {
    $("accidentForm").reset(); $("accidentCaseId").value = "";
    $("accidentRevision").value = ""; $("accidentEditorDetails").open = false;
  }
  function fillAccidentForm(row) {
    $("accidentCaseId").value = row.case_id;
    $("accidentRevision").value = String(row.revision);
    $("accidentPatientName").value = row.patient_name;
    $("accidentPhone").value = row.phone;
    $("accidentDate").value = row.accident_date;
    $("accidentVisitDate").value = row.visit_date || "";
    $("accidentWritingDate").value = row.writing_started_date || "";
    $("accidentSentDate").value = row.report_sent_date || "";
    $("accidentAmountDue").value = row.amount_due_cents === null ? "" : (row.amount_due_cents / 100).toFixed(2);
    $("accidentEditorDetails").open = true;
    $("accidentPatientName").focus();
  }
  function drawCases() {
    const list = $("accidentList"); list.replaceChildren();
    if (!caseRows.length) { list.append(node("p", "Δεν υπάρχουν καταχωρισμένα τροχαία.")); return; }
    for (const row of caseRows) {
      const card = document.createElement("div"); card.className = "office-item";
      card.append(node("strong", row.patient_name),
        node("span", "Ατύχημα: " + row.accident_date + " · Τηλ.: " + row.phone),
        node("span", "Απαιτούμενο: " + euro(row.amount_due_cents) +
          " · Πληρωμένο: " + euro(row.received_cents) + " · Υπόλοιπο: " + euro(row.balance_cents)));
      const actions = document.createElement("div"); actions.className = "office-item-actions";
      actions.append(button("Άνοιγμα", () => selectAccident(row.case_id)),
        button("Επεξεργασία", () => { fillAccidentForm(row); selectAccident(row.case_id); }));
      card.append(actions); list.append(card);
    }
  }
  async function loadAccidents(reset=true) {
    const token = ++generation;
    setAccidentWritesEnabled(false);
    if (reset) { caseRows = []; caseOffset = 0; $("accidentList").replaceChildren(); }
    show("accidentState", "Φόρτωση…");
    try {
      const result = await api("/clinical/office-cases/accidents?offset=" + caseOffset);
      if (token !== generation) return;
      if (!Array.isArray(result.cases)) throw new Error("Invalid protected source");
      caseRows.push(...result.cases);
      caseOffset = result.next_offset || 0;
      $("accidentNext").hidden = !result.has_more;
      setAccidentWritesEnabled(true);
      show("accidentState", String(caseRows.length) + (result.has_more ? "+" : "") + " περιστατικά");
      show("accidentNote", needsPaymentReconciliation
        ? "Έλεγξε το ιστορικό πληρωμών στο περιστατικό πριν από νέα καταχώριση."
        : result.has_more ? "Εμφανίζονται τα 50 πιο πρόσφατα. Τα υπόλοιπα απαιτούν επόμενη σελίδα." : "");
      drawCases();
      if (selectedCase) await selectAccident(selectedCase.case_id);
    } catch (error) {
      if (token !== generation) return;
      show("accidentNote", errorText(error));
      show("accidentState", "Μη διαθέσιμο");
      if (reset) { caseRows = []; $("accidentList").replaceChildren(); $("accidentDetail").hidden = true; }
    }
  }
  async function selectAccident(caseId) {
    const token = generation, detailToken = ++detailGeneration;
    try {
      const row = await api("/clinical/office-cases/accidents/" + encodeURIComponent(caseId));
      if (token !== generation || detailToken !== detailGeneration) return;
      if (pendingPaymentCaseId && pendingPaymentCaseId !== caseId) {
        pendingPaymentId = ""; pendingPaymentCaseId = "";
        $("accidentPaymentForm").reset();
      }
      selectedCase = row;
      $("accidentDetail").hidden = false;
      $("paymentReviewConfirm").hidden = !needsPaymentReconciliation;
      setAccidentWritesEnabled(true, !needsPaymentReconciliation);
      show("accidentDetailTitle", row.patient_name);
      show("accidentDetailSummary", "Ατύχημα " + row.accident_date +
        " · Επίσκεψη " + (row.visit_date || "—") + " · Έναρξη συγγραφής " +
        (row.writing_started_date || "—") + " · Αποστολή " + (row.report_sent_date || "—") +
        " · Υπόλοιπο " + euro(row.balance_cents));
      const list = $("accidentPaymentList"); list.replaceChildren();
      for (const payment of row.payments) {
        const card = document.createElement("div"); card.className = "office-item";
        card.append(node("strong", euro(payment.amount_cents)),
          node("span", payment.paid_on + " · " + payment.payer +
            (payment.voided_at ? " · Ακυρώθηκε: " + payment.void_reason : "")),
          node("span", "Καταχώριση (Κύπρος): " + cyprusTimestamp(payment.created_at) +
            (payment.voided_at ? " · Ακύρωση (Κύπρος): " + cyprusTimestamp(payment.voided_at) : "")));
        if (!payment.voided_at) card.append(button("Ακύρωση λανθασμένης πληρωμής", () => voidPayment(payment)));
        list.append(card);
      }
      if (!row.payments.length) list.append(node("p", "Δεν καταχωρίστηκε πληρωμή."));
      if (needsPaymentReconciliation) show("accidentNote",
        "Έλεγξε τις καταχωρισμένες πληρωμές και επιβεβαίωσε παρακάτω πριν ξεκινήσεις νέα προσπάθεια.");
    } catch (error) {
      if (token === generation && detailToken === detailGeneration) show("accidentNote", errorText(error));
    }
  }
  async function saveAccident(event) {
    event.preventDefault();
    const id = $("accidentCaseId").value, token = generation;
    let due;
    try { due = $("accidentAmountDue").value.trim() ? cents($("accidentAmountDue").value) : null; }
    catch (error) { show("accidentNote", error.message); return; }
    const payload = {
      patient_name:$("accidentPatientName").value.trim(), phone:$("accidentPhone").value.trim(),
      accident_date:$("accidentDate").value, visit_date:$("accidentVisitDate").value || null,
      writing_started_date:$("accidentWritingDate").value || null,
      report_sent_date:$("accidentSentDate").value || null, amount_due_cents:due,
    };
    if (id) payload.expected_revision = Number($("accidentRevision").value);
    try {
      const row = await api("/clinical/office-cases/accidents" + (id ? "/" + encodeURIComponent(id) : ""),
        {method:id ? "PUT" : "POST", body:JSON.stringify(payload)});
      if (token !== generation) return;
      resetAccidentForm(); selectedCase = row;
      show("accidentNote", "Το περιστατικό αποθηκεύτηκε.");
      await loadAccidents();
    } catch (error) { if (token === generation) show("accidentNote", errorText(error)); }
  }
  async function savePayment(event) {
    event.preventDefault();
    if (!selectedCase) return;
    const caseId = selectedCase.case_id, token = generation;
    let amount;
    try { amount = cents($("paymentAmount").value); if (amount <= 0) throw new Error("Το ποσό πρέπει να είναι θετικό."); }
    catch (error) { show("accidentNote", error.message); return; }
    if (!pendingPaymentId) { pendingPaymentId = crypto.randomUUID(); pendingPaymentCaseId = caseId; }
    try {
      const row = await api("/clinical/office-cases/accidents/" + encodeURIComponent(caseId) + "/payments", {
        method:"POST", body:JSON.stringify({payment_request_id:pendingPaymentId, amount_cents:amount,
          paid_on:$("paymentDate").value, payer:$("paymentPayer").value.trim()})
      });
      if (token !== generation) return;
      pendingPaymentId = ""; pendingPaymentCaseId = "";
      $("accidentPaymentForm").reset(); selectedCase = row;
      show("accidentNote", "Η επιμέρους πληρωμή καταχωρίστηκε.");
      await loadAccidents();
    } catch (error) {
      if (token === generation) show("accidentNote", errorText(error) +
        " Αν η σύνδεση διακόπηκε, έλεγξε τις πληρωμές πριν ξεκινήσεις νέα καταχώριση.");
    }
  }
  async function voidPayment(payment) {
    if (!selectedCase) return;
    const reason = window.prompt("Λόγος ακύρωσης της λανθασμένης πληρωμής:");
    if (!reason || reason.trim().length < 3) return;
    const caseId = selectedCase.case_id, token = generation;
    try {
      const row = await api("/clinical/office-cases/accidents/" + encodeURIComponent(caseId) +
        "/payments/" + encodeURIComponent(payment.payment_id) + "/void", {
          method:"POST", body:JSON.stringify({expected_revision:selectedCase.revision, reason:reason.trim()})
        });
      if (token !== generation) return;
      selectedCase = row; show("accidentNote", "Η πληρωμή ακυρώθηκε με διατηρημένο ιστορικό.");
      await loadAccidents();
    } catch (error) { if (token === generation) show("accidentNote", errorText(error)); }
  }

  function drawRF() {
    const list = $("officeRfList"); list.replaceChildren();
    if (!rfRows.length) { list.append(node("p", "Δεν βρέθηκαν RF σε αυτή την αναζήτηση.")); return; }
    for (const card of rfRows) {
      const item = document.createElement("div"); item.className = "office-item";
      item.append(node("strong", card.patient_name || "RF"),
        node("span", "ΑΔΤ: " + card.identity_number + " · " + (stageNames[card.stage] || card.stage)),
        node("span", "Ένδειξη: " + card.indication_code + " · Πλευρά: " + card.laterality +
          (card.scheduled_date ? " · Ημ. προγραμματισμού: " + card.scheduled_date : "")));
      const actions = document.createElement("div"); actions.className = "office-item-actions";
      if (card.surgery_id) {
        const surgery = card.linked_surgery;
        actions.append(node("span", surgery
          ? "Συνδεδεμένο χειρουργείο: " + surgery.surgery_id + " · " +
            surgery.full_name + " · " + surgery.procedure_type + " · " +
            (surgery.surgery_date || "Άγνωστη ημερομηνία") + " · " +
            ({pending:"Σε αναμονή", completed:"Ολοκληρώθηκε", deleted:"Αφαιρέθηκε από τα pending"}[surgery.status]
              || surgery.status) +
            (surgery.room_entry_time ? " · Ώρα αίθουσας: " + surgery.room_entry_time : "")
          : "Συνδεδεμένο χειρουργείο: " + card.surgery_id));
      }
      else if (["approved","scheduled"].includes(card.stage))
        actions.append(button("Προσθήκη στη λίστα χειρουργείων", () => window.CockpitSurgeryQueue?.prefillFromRF?.(card)));
      item.append(actions); list.append(item);
    }
  }
  async function loadRF(reset, conflict=false) {
    const token = ++generation;
    if (reset) { rfRows = []; rfCursor = ""; $("officeRfList").replaceChildren(); }
    show("officeRfState", "Φόρτωση…");
    try {
      const params = new URLSearchParams({query:$("officeRfQuery").value.trim(),
        stage:$("officeRfStage").value, cursor:rfCursor});
      const result = await api("/clinical/office-cases/rf?" + params.toString());
      if (token !== generation) return;
      if (!Array.isArray(result.applications)) throw new Error("Invalid protected RF source");
      rfRows.push(...result.applications);
      rfCursor = result.next_cursor || "";
      $("officeRfNext").hidden = !rfCursor;
      show("officeRfState", String(rfRows.length) + (result.has_more ? "+" : "") + " RF");
      show("officeRfNote", (conflict ? "Η RF άλλαξε ή συνδέθηκε. Έλεγξε την τρέχουσα κατάστασή της. " : "") +
        (result.has_more ? "Υπάρχουν περισσότερες RF. Η τρέχουσα σελίδα δεν είναι πλήρες μητρώο." : ""));
      drawRF();
    } catch (error) {
      if (token !== generation) return;
      show("officeRfNote", errorText(error)); show("officeRfState", "Μη διαθέσιμο");
      rfRows = []; $("officeRfList").replaceChildren(); $("officeRfNext").hidden = true;
    }
  }

  document.querySelectorAll("[data-office-tab]").forEach(item => item.addEventListener("click", () => selectTab(item.dataset.officeTab)));
  $("accidentForm").addEventListener("submit", saveAccident);
  $("accidentCancel").addEventListener("click", resetAccidentForm);
  $("paymentReviewConfirm").addEventListener("click", () => {
    if (!selectedCase) return;
    needsPaymentReconciliation = false;
    $("paymentReviewConfirm").hidden = true;
    setAccidentWritesEnabled(true, true);
    show("accidentNote", "Το ιστορικό ελέγχθηκε. Μπορείς να καταχωρίσεις νέα πληρωμή.");
  });
  $("accidentNext").addEventListener("click", () => loadAccidents(false));
  $("accidentPaymentForm").addEventListener("submit", savePayment);
  $("officeRfSearch").addEventListener("submit", event => { event.preventDefault(); loadRF(true); });
  $("officeRfNext").addEventListener("click", () => loadRF(false));
  window.addEventListener("clinical:authenticated", () => {
    clear(); if (activeTab === "accidents") loadAccidents(); if (activeTab === "rf") loadRF(true);
  });
  window.addEventListener("clinical:unauthenticated", clear);
  window.CockpitOfficeCases = {open, selectTab,
    refreshRF() { if (activeTab === "rf") loadRF(true); },
    async showRFConflict(applicationId) {
      const refresh = selectTab("rf", true);
      const token = generation;
      if (refresh) await refresh;
      if (token !== generation || activeTab !== "rf") return;
      try {
        const card = await api("/clinical/office-cases/rf/" + encodeURIComponent(applicationId));
        if (token !== generation || activeTab !== "rf") return;
        rfRows = [card, ...rfRows.filter(row => row.id !== card.id)];
        drawRF();
        show("officeRfNote", "Η RF άλλαξε ή συνδέθηκε. Εμφανίζεται η τρέχουσα εγγραφή.");
      } catch (error) {
        if (token === generation && activeTab === "rf") show("officeRfNote", errorText(error));
      }
    }};
  setAccidentWritesEnabled(false);
})();
