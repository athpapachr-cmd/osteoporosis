(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const lateralityLabels = {
    left: "Αριστερά",
    right: "Δεξιά",
    bilateral: "Αμφοτερόπλευρα",
    not_applicable: "Δεν εφαρμόζεται",
    unspecified: "Δεν έχει καθοριστεί"
  };

  const appointmentCategoryLabels = {
    osteoporosis_first: "Πρώτη επίσκεψη",
    osteoporosis_review: "Επανέλεγχος",
    osteoporosis_unspecified: "Οστεοπόρωση",
    prolia: "Prolia",
    aclasta: "Aclasta",
    other: "Ραντεβού"
  };

  const appointmentTimeFormatter = new Intl.DateTimeFormat("el-CY", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Asia/Nicosia"
  });

  const appointmentDateFormatter = new Intl.DateTimeFormat("el-CY", {
    day: "numeric", month: "numeric", year: "numeric", timeZone: "Asia/Nicosia"
  });

  let surgeryRows = [];
  let surgerySortKey = "queue_position";
  let surgerySortDirection = "asc";

  function formatToday(now) {
    return new Intl.DateTimeFormat("el-GR", {
      weekday: "long",
      day: "numeric",
      month: "long",
      timeZone: "Asia/Nicosia"
    }).format(now);
  }

  function formatIsoDate(value) {
    if (!value) return "—";
    const parts = String(value).split("-");
    if (parts.length !== 3) return value;
    return `${parts[2]}/${parts[1]}/${parts[0]}`;
  }

  function parseClinicalAppointmentDate(value) {
    if (!value) return null;
    const text = String(value);
    const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(text);
    const parsed = new Date(hasZone ? text : `${text}Z`);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  function setAppointmentSlot(prefix, row, emptyText, now) {
    const timeNode = $(`${prefix}AppointmentTime`);
    const patientNode = $(`${prefix}AppointmentPatient`);
    const typeNode = $(`${prefix}AppointmentType`);
    const slotButton = $(`visitSidebar${prefix[0].toUpperCase()}${prefix.slice(1)}`);
    if (slotButton) slotButton.disabled = !row;

    if (!row) {
      timeNode.textContent = "—";
      patientNode.textContent = emptyText;
      delete patientNode.dataset.appointmentId;
      typeNode.textContent = "";
      return;
    }

    const start = parseClinicalAppointmentDate(row.start_at);
    const laterDay = start && now
      && appointmentDateFormatter.format(start) !== appointmentDateFormatter.format(now);
    timeNode.textContent = start
      ? `${laterDay ? `${appointmentDateFormatter.format(start)} · ` : ""}${appointmentTimeFormatter.format(start)}`
      : "—";
    patientNode.textContent = row.patient_display_name || "Χωρίς καταχωρημένο όνομα";
    patientNode.dataset.appointmentId = row.appointment_id;
    typeNode.textContent = [row.clinic, row.reason || appointmentCategoryLabels[row.category] || "Ραντεβού"]
      .filter(Boolean).join(" · ");
  }

  function renderTodayContext(context, now) {
    setAppointmentSlot("previous", context.previous, "Δεν υπάρχει προηγούμενο σήμερα", now);
    setAppointmentSlot("next", context.next, "Δεν υπάρχει επόμενο στο πρόγραμμα", now);

    if (context.current_conflict_count > 1) {
      setAppointmentSlot("current", null, "", now);
      $("currentAppointmentTime").textContent = "—";
      $("currentAppointmentPatient").textContent = `${context.current_conflict_count} ταυτόχρονα ραντεβού`;
      $("currentAppointmentType").textContent = "Έλεγξε το πρόγραμμα στο Reception";
    } else {
      setAppointmentSlot("current", context.current, "Χωρίς ενεργό ραντεβού", now);
    }
  }

  function clearTodayContext(message) {
    setAppointmentSlot("previous", null, message);
    setAppointmentSlot("current", null, message);
    setAppointmentSlot("next", null, message);
  }

  async function loadCockpitContext() {
    const now = new Date();
    $("todayHeading").textContent = formatToday(now);
    try {
      const response = await fetch("/clinical/calendar/cockpit-context", { credentials: "same-origin" });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        const failure = new Error(`HTTP ${response.status}`);
        failure.lastFetchedAt = error.detail?.last_fetched_at || null;
        throw failure;
      }
      const context = await response.json();
      const generated = parseClinicalAppointmentDate(context.generated_at);
      if (!generated || !Number.isInteger(context.today_total)
          || !Number.isInteger(context.current_conflict_count)) throw new Error("Invalid context");
      renderTodayContext(context, generated);
      const updated = parseClinicalAppointmentDate(context.source_updated_at);
      $("calendarState").textContent = `${context.today_total} σήμερα`;
      const fresh = updated && generated.getTime() - updated.getTime() <= 5 * 60 * 1000;
      $("calendarState").classList.toggle("ready", Boolean(fresh));
      $("calendarNote").textContent = updated
        ? `${fresh ? "" : "Το πρόγραμμα μπορεί να έχει παλιώσει. "}Τελευταία ανάγνωση κρατήσεων: ${appointmentDateFormatter.format(updated)} · ${appointmentTimeFormatter.format(updated)}.`
        : "Δεν υπάρχει διαθέσιμη πρόσφατη ανάγνωση κρατήσεων. Έλεγξε το πρόγραμμα στο Reception.";
    } catch (error) {
      clearTodayContext("Μη διαθέσιμο");
      $("calendarState").textContent = "Μη διαθέσιμο";
      $("calendarState").classList.remove("ready");
      const last = parseClinicalAppointmentDate(error.lastFetchedAt);
      $("calendarNote").textContent = last
        ? `Το πρόγραμμα δεν είναι διαθέσιμο. Τελευταία επιτυχής ανάγνωση: ${appointmentDateFormatter.format(last)} · ${appointmentTimeFormatter.format(last)}. Έλεγξε το Reception.`
        : "Δεν ήταν δυνατή η φόρτωση του προγράμματος. Έλεγξε τη σύνδεση ή άνοιξε το Reception.";
    }
  }

  async function apiJson(url, options = {}) {
    const response = await fetch(url, {
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {})
      },
      ...options
    });

    if (!response.ok) {
      const error = new Error(`HTTP ${response.status}`);
      error.status = response.status;
      try {
        const payload = await response.json();
        error.detail = payload.detail || "";
      } catch (_) {
        error.detail = "";
      }
      throw error;
    }
    return response.json();
  }

  function surgerySortValue(row, key) {
    if (key === "laterality") return lateralityLabels[row.laterality] || row.laterality || "";
    if (key === "surgery_date") return row.surgery_date || "9999-12-31";
    if (key === "queue_position") return Number(row.queue_position || 0);
    return String(row[key] || "").toLocaleLowerCase("el-GR");
  }

  function sortedSurgeryRows() {
    const rows = [...surgeryRows];
    const direction = surgerySortDirection === "asc" ? 1 : -1;
    rows.sort((left, right) => {
      const a = surgerySortValue(left, surgerySortKey);
      const b = surgerySortValue(right, surgerySortKey);
      if (typeof a === "number" && typeof b === "number") return (a - b) * direction;
      return String(a).localeCompare(String(b), "el", { numeric: true, sensitivity: "base" }) * direction;
    });
    return rows;
  }

  function cell(text, className = "") {
    const td = document.createElement("td");
    if (className) td.className = className;
    td.textContent = text;
    return td;
  }

  function actionButton(label, action, surgeryId, title) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "surgery-row-action";
    button.dataset.action = action;
    button.dataset.surgeryId = surgeryId;
    button.textContent = label;
    button.title = title || label;
    return button;
  }

  function publishSurgeryCounts() {
    // Only aggregate counts; do not pass identities or surgery details to
    // the Home attention preview, and do not create a second surgery reader.
    const detail = {total: surgeryRows.length,
      undated: surgeryRows.filter(row => !row.surgery_date).length};
    window.CockpitSurgerySummary = detail;
    if (typeof window.dispatchEvent === "function" && typeof window.CustomEvent === "function") {
      window.dispatchEvent(new window.CustomEvent("cockpit:surgery-counts", {detail}));
    }
  }

  function renderSurgeryRows() {
    publishSurgeryCounts();
    const tbody = $("surgeryTableBody");
    tbody.replaceChildren();

    const rows = sortedSurgeryRows();
    if (!rows.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 9;
      td.className = "surgery-empty";
      td.textContent = "Δεν υπάρχουν pending χειρουργεία.";
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    for (const row of rows) {
      const tr = document.createElement("tr");
      tr.dataset.surgeryId = row.surgery_id;

      tr.appendChild(cell(String(row.queue_position), "surgery-rank"));
      tr.appendChild(cell(row.full_name));
      tr.appendChild(cell(row.identity_number));
      tr.appendChild(cell(formatIsoDate(row.date_of_birth)));
      tr.appendChild(cell(row.procedure_type));
      tr.appendChild(cell(lateralityLabels[row.laterality] || row.laterality || "—"));
      tr.appendChild(cell(row.phone));

      const dateCell = document.createElement("td");
      const dateInput = document.createElement("input");
      dateInput.type = "date";
      dateInput.className = "surgery-date-input";
      dateInput.dataset.surgeryId = row.surgery_id;
      dateInput.value = row.surgery_date || "";
      dateInput.setAttribute("aria-label", `Ημερομηνία χειρουργείου για ${row.full_name}`);
      dateCell.appendChild(dateInput);
      tr.appendChild(dateCell);

      const actions = document.createElement("td");
      actions.className = "surgery-row-actions";
      actions.append(
        actionButton("↑", "up", row.surgery_id, "Μετακίνηση πάνω"),
        actionButton("↓", "down", row.surgery_id, "Μετακίνηση κάτω"),
        actionButton("✎", "edit", row.surgery_id, "Επεξεργασία"),
        actionButton("✓", "complete", row.surgery_id, "Ολοκληρώθηκε"),
        actionButton("🗑", "delete", row.surgery_id, "Διαγραφή pending χειρουργείου")
      );
      tr.appendChild(actions);
      tbody.appendChild(tr);
    }
  }

  function setSurgeryUnavailable(message) {
    $("surgeryQueueState").textContent = message;
    $("surgeryQueueState").classList.remove("ready");
    const tbody = $("surgeryTableBody");
    tbody.replaceChildren();
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 9;
    td.className = "surgery-empty";
    td.textContent = message;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }

  async function loadSurgeryQueue() {
    try {
      surgeryRows = await apiJson("/clinical/surgeries");
      $("surgeryQueueState").textContent = `${surgeryRows.length} pending`;
      $("surgeryQueueState").classList.add("ready");
      $("surgeryQueueNote").textContent = surgeryRows.length
        ? "Η σειρά # είναι η αποθηκευμένη manual σειρά. Τα sortable headers αλλάζουν μόνο την προβολή."
        : "Δεν υπάρχουν pending χειρουργεία. Πρόσθεσε το πρώτο όταν χρειαστεί.";
      renderSurgeryRows();
    } catch (error) {
      surgeryRows = [];
      window.CockpitSurgerySummary = null;
      if (typeof window.dispatchEvent === "function" && typeof window.CustomEvent === "function") {
        window.dispatchEvent(new window.CustomEvent("cockpit:surgery-counts", {
          detail: {unavailable: true}
        }));
      }
      if (error.status === 401) {
        setSurgeryUnavailable("Σύνδεση απαιτείται");
        $("surgeryQueueNote").textContent = "Άνοιξε το Clinical Calendar ή άλλο protected clinical εργαλείο και συνδέσου με το Clinical Key.";
      } else {
        setSurgeryUnavailable("Μη διαθέσιμο");
        $("surgeryQueueNote").textContent = "Η protected λίστα χειρουργείων δεν είναι διαθέσιμη αυτή τη στιγμή.";
      }
    }
  }

  function resetSurgeryForm() {
    $("surgeryForm").reset();
    $("surgeryId").value = "";
    $("surgeryLaterality").value = "unspecified";
    $("surgerySubmit").textContent = "Προσθήκη";
    $("surgeryCancelEdit").hidden = true;
  }

  function startSurgeryEdit(row) {
    $("surgeryId").value = row.surgery_id;
    $("surgeryFullName").value = row.full_name || "";
    $("surgeryIdentityNumber").value = row.identity_number || "";
    $("surgeryDateOfBirth").value = row.date_of_birth || "";
    $("surgeryPhone").value = row.phone || "";
    $("surgeryProcedureType").value = row.procedure_type || "";
    $("surgeryLaterality").value = row.laterality || "unspecified";
    $("surgeryDate").value = row.surgery_date || "";
    $("surgerySubmit").textContent = "Αποθήκευση";
    $("surgeryCancelEdit").hidden = false;
    $("surgeryEditorDetails").open = true;
    $("surgeryFullName").focus();
  }

  async function submitSurgeryForm(event) {
    event.preventDefault();
    const surgeryId = $("surgeryId").value;
    const payload = {
      full_name: $("surgeryFullName").value.trim(),
      identity_number: $("surgeryIdentityNumber").value.trim(),
      date_of_birth: $("surgeryDateOfBirth").value,
      phone: $("surgeryPhone").value.trim(),
      procedure_type: $("surgeryProcedureType").value.trim(),
      laterality: $("surgeryLaterality").value,
      surgery_date: $("surgeryDate").value || null
    };

    try {
      if (surgeryId) {
        await apiJson(`/clinical/surgeries/${encodeURIComponent(surgeryId)}`, {
          method: "PUT",
          body: JSON.stringify(payload)
        });
      } else {
        await apiJson("/clinical/surgeries", {
          method: "POST",
          body: JSON.stringify(payload)
        });
      }
      resetSurgeryForm();
      $("surgeryEditorDetails").open = false;
      surgerySortKey = "queue_position";
      surgerySortDirection = "asc";
      await loadSurgeryQueue();
    } catch (error) {
      $("surgeryQueueNote").textContent = error.detail || "Δεν ήταν δυνατή η αποθήκευση της εγγραφής.";
    }
  }

  async function handleSurgeryAction(event) {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const row = surgeryRows.find((item) => item.surgery_id === button.dataset.surgeryId);
    if (!row) return;

    const action = button.dataset.action;
    if (action === "edit") {
      startSurgeryEdit(row);
      return;
    }

    if (action === "complete") {
      if (!window.confirm(`Να σημειωθεί ως ολοκληρωμένο το χειρουργείο για ${row.full_name};`)) return;
      try {
        await apiJson(`/clinical/surgeries/${encodeURIComponent(row.surgery_id)}/complete`, {
          method: "POST",
          body: "{}"
        });
        await loadSurgeryQueue();
      } catch (error) {
        $("surgeryQueueNote").textContent = error.detail || "Δεν ολοκληρώθηκε η ενημέρωση.";
      }
      return;
    }

    if (action === "delete") {
      if (!window.confirm(`Να διαγραφεί από τα pending το χειρουργείο για ${row.full_name};`)) return;
      try {
        await apiJson(`/clinical/surgeries/${encodeURIComponent(row.surgery_id)}`, {
          method: "DELETE"
        });
        await loadSurgeryQueue();
      } catch (error) {
        $("surgeryQueueNote").textContent = error.detail || "Δεν ήταν δυνατή η διαγραφή.";
      }
      return;
    }

    if (action === "up" || action === "down") {
      try {
        surgeryRows = await apiJson(`/clinical/surgeries/${encodeURIComponent(row.surgery_id)}/move`, {
          method: "POST",
          body: JSON.stringify({ direction: action })
        });
        surgerySortKey = "queue_position";
        surgerySortDirection = "asc";
        renderSurgeryRows();
      } catch (error) {
        $("surgeryQueueNote").textContent = error.detail || "Δεν άλλαξε η σειρά.";
      }
    }
  }

  async function handleSurgeryDateChange(event) {
    const input = event.target.closest("input.surgery-date-input");
    if (!input) return;
    try {
      const updated = await apiJson(`/clinical/surgeries/${encodeURIComponent(input.dataset.surgeryId)}`, {
        method: "PUT",
        body: JSON.stringify({ surgery_date: input.value || null })
      });
      const index = surgeryRows.findIndex((row) => row.surgery_id === updated.surgery_id);
      if (index >= 0) surgeryRows[index] = updated;
      renderSurgeryRows();
    } catch (error) {
      $("surgeryQueueNote").textContent = error.detail || "Δεν αποθηκεύτηκε η ημερομηνία.";
      await loadSurgeryQueue();
    }
  }

  function bindSurgerySorting() {
    document.querySelectorAll(".surgery-sort").forEach((button) => {
      button.addEventListener("click", () => {
        const key = button.dataset.sort;
        if (surgerySortKey === key) {
          surgerySortDirection = surgerySortDirection === "asc" ? "desc" : "asc";
        } else {
          surgerySortKey = key;
          surgerySortDirection = "asc";
        }
        renderSurgeryRows();
      });
    });
  }

  $("surgeryForm").addEventListener("submit", submitSurgeryForm);
  $("surgeryCancelEdit").addEventListener("click", () => {
    resetSurgeryForm();
    $("surgeryEditorDetails").open = false;
  });
  $("surgeryTableBody").addEventListener("click", handleSurgeryAction);
  $("surgeryTableBody").addEventListener("change", handleSurgeryDateChange);
  bindSurgerySorting();
  document.querySelectorAll("[data-expand-secondary]").forEach(link=>{
    link.addEventListener("click",()=>{const panel=$("cockpitOtherFunctions");if(panel)panel.open=true;});
  });
  resetSurgeryForm();
  loadCockpitContext();
  window.setInterval(loadCockpitContext, 60000);
  loadSurgeryQueue();
})();
