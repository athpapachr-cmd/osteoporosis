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
    aclasta: "Aclasta"
  };

  const appointmentTimeFormatter = new Intl.DateTimeFormat("el-CY", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  });

  let surgeryRows = [];
  let surgerySortKey = "queue_position";
  let surgerySortDirection = "asc";

  function localDayBounds() {
    const now = new Date();
    const start = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0);
    const end = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1, 0, 0, 0, 0);
    return { now, start, end };
  }

  function formatToday(now) {
    return new Intl.DateTimeFormat("el-GR", {
      weekday: "long",
      day: "numeric",
      month: "long"
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
    const hasZone = /(?:Z|[+-]\\d{2}:?\\d{2})$/.test(text);
    const parsed = new Date(hasZone ? text : `${text}Z`);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  function appointmentContext(rows, now) {
    const nowMs = now.getTime();
    const items = (rows || [])
      .map((row) => {
        const start = parseClinicalAppointmentDate(row.start_at);
        const end = parseClinicalAppointmentDate(row.end_at);
        if (!start || !end || end <= start) return null;
        return { row, start: start.getTime(), end: end.getTime() };
      })
      .filter(Boolean)
      .sort((left, right) => left.start - right.start);

    const previousItems = items.filter((item) => item.end <= nowMs);
    const activeItems = items.filter((item) => item.start <= nowMs && nowMs < item.end);
    const nextItem = items.find((item) => item.start > nowMs) || null;

    return {
      previous: previousItems.length ? previousItems[previousItems.length - 1].row : null,
      current: activeItems.length === 1 ? activeItems[0].row : null,
      currentCount: activeItems.length,
      next: nextItem ? nextItem.row : null
    };
  }

  function setAppointmentSlot(prefix, row, emptyText) {
    const timeNode = $(`${prefix}AppointmentTime`);
    const patientNode = $(`${prefix}AppointmentPatient`);
    const typeNode = $(`${prefix}AppointmentType`);

    if (!row) {
      timeNode.textContent = "—";
      patientNode.textContent = emptyText;
      typeNode.textContent = "";
      return;
    }

    const start = parseClinicalAppointmentDate(row.start_at);
    timeNode.textContent = start ? appointmentTimeFormatter.format(start) : "—";
    patientNode.textContent = row.patient_display_name || "Χωρίς καταχωρημένο όνομα";
    typeNode.textContent = appointmentCategoryLabels[row.category] || "Οστεοπόρωση";
  }

  function renderTodayContext(rows, now) {
    const context = appointmentContext(rows, now);
    setAppointmentSlot("previous", context.previous, "Δεν υπάρχει προηγούμενο σήμερα");
    setAppointmentSlot("next", context.next, "Δεν υπάρχει επόμενο σήμερα");

    if (context.currentCount > 1) {
      $("currentAppointmentTime").textContent = "—";
      $("currentAppointmentPatient").textContent = `${context.currentCount} ταυτόχρονα ραντεβού`;
      $("currentAppointmentType").textContent = "Δες το εβδομαδιαίο ημερολόγιο";
    } else {
      setAppointmentSlot("current", context.current, "Χωρίς ενεργό ραντεβού");
    }
  }

  function clearTodayContext(message) {
    setAppointmentSlot("previous", null, message);
    setAppointmentSlot("current", null, message);
    setAppointmentSlot("next", null, message);
  }

  async function loadClinicalCalendarSummary() {
    const { now, start, end } = localDayBounds();
    $("todayHeading").textContent = formatToday(now);
    try {
      const url = `/clinical/calendar/appointments?start=${encodeURIComponent(start.toISOString())}&end=${encodeURIComponent(end.toISOString())}`;
      const response = await fetch(url, { credentials: "same-origin" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const rows = await response.json();
      const consultations = rows.filter((row) =>
        ["osteoporosis_first", "osteoporosis_review", "osteoporosis_unspecified"].includes(row.category)
      ).length;
      const treatments = rows.filter((row) => ["prolia", "aclasta"].includes(row.category)).length;

      renderTodayContext(rows, now);
      $("todayOsteoporosisCount").textContent = String(consultations);
      $("todayTreatmentCount").textContent = String(treatments);
      $("calendarState").textContent = rows.length ? `${rows.length} σήμερα` : "0 σήμερα";
      $("calendarState").classList.add("ready");
      $("calendarNote").textContent = rows.length
        ? "Προβολή προγράμματος από το Clinical Calendar. Το εβδομαδιαίο ημερολόγιο παραμένει η πλήρης προβολή."
        : "Δεν υπάρχουν ραντεβού οστεοπόρωσης για σήμερα.";
    } catch (_) {
      clearTodayContext("Μη διαθέσιμο");
      $("todayOsteoporosisCount").textContent = "—";
      $("todayTreatmentCount").textContent = "—";
      $("calendarState").textContent = "Μη διαθέσιμο";
      $("calendarState").classList.remove("ready");
      $("calendarNote").textContent = "Δεν ήταν δυνατή η φόρτωση της σημερινής κλινικής εικόνας. Άνοιξε το εβδομαδιαίο ημερολόγιο για σύνδεση ή έλεγχο.";
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

  function renderSurgeryRows() {
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

  resetSurgeryForm();
  loadClinicalCalendarSummary();
  loadSurgeryQueue();
})();
