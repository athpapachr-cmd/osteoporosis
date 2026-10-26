(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const clinicalURL=path=>((typeof window!=="undefined"&&window.CockpitClinicalBase)||"/clinical")+path.slice(9);

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
  // The existing protected surgery loader owns the ONLY source of Home detail.
  let surgeryGeneration = 0;
  let surgeryReady = false;
  let surgeryAuthEpoch = 0;
  const surgeryDateParts = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Nicosia", year: "numeric", month: "2-digit", day: "2-digit"
  });

  function surgeryToday() {
    const parts = Object.fromEntries(surgeryDateParts.formatToParts(new Date())
      .filter(part => part.type !== "literal").map(part => [part.type, part.value]));
    return [parts.year, parts.month, parts.day].join("-");
  }

  function validSurgeryDate(value) {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    const [year, month, day] = value.split("-").map(Number);
    if (year < 100 || month < 1 || month > 12 || day < 1 || day > 31) return false;
    const parsed = new Date(Date.UTC(year, month - 1, day));
    return parsed.getUTCFullYear() === year && parsed.getUTCMonth() + 1 === month
      && parsed.getUTCDate() === day;
  }

  function validSurgeryRows(rows) {
    return Array.isArray(rows) && rows.every(row =>
      row && typeof row === "object" && !Array.isArray(row)
      && typeof row.surgery_id === "string" && row.surgery_id.trim()
      && typeof row.full_name === "string" && row.full_name.trim()
      && typeof row.procedure_type === "string" && row.procedure_type.trim()
      && Number.isInteger(row.queue_position) && row.queue_position > 0);
  }

  function surgicalAttentionSummary(rows) {
    const today = surgeryToday();
    const dates = new Map();
    let undated = 0, invalid = 0, overdue = 0;
    for (const row of rows) {
      const date = row.surgery_date;
      if (date === null || date === undefined || date === "") { undated++; continue; }
      if (!validSurgeryDate(date)) { invalid++; continue; }
      if (date < today) { overdue++; continue; } // Pending remains pending, never "performed".
      if (!dates.has(date)) dates.set(date, []);
      dates.get(date).push(row);
    }
    const upcoming = [...dates.keys()].sort().slice(0, 3).map(date => {
      const sameDay = dates.get(date);
      const group = { date, count: sameDay.length };
      if (sameDay.length === 1) {
        group.name = sameDay[0].full_name;
        group.procedure = sameDay[0].procedure_type;
      }
      return group; // 2+ rows disclose a count, never identities.
    });
    return { total: rows.length, undated, invalid, overdue, upcoming };
  }

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
      delete patientNode.dataset.appointmentDate;
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
    patientNode.dataset.appointmentDate=start?new Intl.DateTimeFormat("en-CA",{timeZone:"Asia/Nicosia",year:"numeric",month:"2-digit",day:"2-digit"}).format(start):"";
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
      const response = await fetch(clinicalURL("/clinical/calendar/cockpit-context"), { credentials: "same-origin" });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        const failure = new Error(`HTTP ${response.status}`);
        failure.status = response.status;
        failure.lastFetchedAt = error.detail?.last_fetched_at || null;
        throw failure;
      }
      const context = await response.json();
      const generated = parseClinicalAppointmentDate(context.generated_at);
      if (!generated || !Number.isInteger(context.today_total)
          || !Number.isInteger(context.current_conflict_count)) throw new Error("Invalid context");
      renderTodayContext(context, generated);
      if(context.synthetic_scenario) $("todayHeading").textContent="Σενάριο 17/10/2026";
      const updated = parseClinicalAppointmentDate(context.source_updated_at);
      $("calendarState").textContent = `${context.today_total} σήμερα`;
      const retry = $("cockpitCalendarRetry");
      if (retry) retry.hidden = true;
      const fresh = updated && generated.getTime() - updated.getTime() <= 5 * 60 * 1000;
      $("calendarState").classList.toggle("ready", Boolean(fresh));
      $("calendarNote").textContent = updated
        ? `${fresh ? "" : "Το πρόγραμμα μπορεί να έχει παλιώσει. "}Τελευταία ανάγνωση κρατήσεων: ${appointmentDateFormatter.format(updated)} · ${appointmentTimeFormatter.format(updated)}.`
        : "Δεν υπάρχει διαθέσιμη πρόσφατη ανάγνωση κρατήσεων. Έλεγξε το πρόγραμμα στο Reception.";
      if(context.synthetic_scenario) $("calendarNote").textContent="Αποθηκευμένα δοκιμαστικά ραντεβού της 17/10/2026 · χωρίς εξωτερικό πάροχο.";
    } catch (error) {
      clearTodayContext("Μη διαθέσιμο");
      $("calendarState").textContent = "Μη διαθέσιμο";
      $("calendarState").classList.remove("ready");
      const last = parseClinicalAppointmentDate(error.lastFetchedAt);
      $("calendarNote").textContent = error.status === 401 || error.status === 403
        ? "Απαιτείται Clinical Key. Πάτησε «Πρόσβαση» πάνω δεξιά για σύνδεση."
        : last
          ? `Το πρόγραμμα της Γραμματείας δεν είναι διαθέσιμο ή δεν καλύπτει σήμερα. Τελευταία επιτυχής ανάγνωση: ${appointmentDateFormatter.format(last)} · ${appointmentTimeFormatter.format(last)}.`
          : "Δεν υπάρχει πρόσφατο πλήρες πρόγραμμα από τη Γραμματεία. Δεν σημαίνει ότι το Clinical Key είναι λάθος.";
      const retry = $("cockpitCalendarRetry");
      if (retry) retry.hidden = false;
    }
  }

  async function apiJson(url, options = {}, current = () => true) {
    const authAtStart = surgeryAuthEpoch;
    const response = await fetch(clinicalURL(url), {
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {})
      },
      ...options
    });

    if (!response.ok) {
      // A current protected failure revokes every Cockpit patient/financial
      // projection. A late response from an earlier auth epoch cannot do so.
      if ((response.status === 401 || response.status === 403)
          && authAtStart === surgeryAuthEpoch && current()) {
        window.dispatchEvent(new window.CustomEvent("clinical:unauthenticated"));
      }
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
    if (!surgeryReady) return;
    // Only the existing successfully loaded protected queue may publish Home
    // date groups; single-day detail is in memory only, never stored/logged.
    const detail = surgicalAttentionSummary(surgeryRows);
    window.CockpitSurgerySummary = detail;
    if (typeof window.dispatchEvent === "function" && typeof window.CustomEvent === "function") {
      window.dispatchEvent(new window.CustomEvent("cockpit:surgery-counts", {detail}));
    }
  }

    function renderSurgeryRows() {
    if (!surgeryReady) return; // Sort/UI events cannot republish revoked data.
    publishSurgeryCounts();
    const tbody = $("surgeryTableBody");
    tbody.replaceChildren();

    const rows = sortedSurgeryRows();
    if (!rows.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 10;
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
      dateInput.disabled = Boolean(row.room_entry_time);
      dateInput.setAttribute("aria-label", row.room_entry_time
        ? `Άλλαξε ημερομηνία και ώρα από την επεξεργασία για ${row.full_name}`
        : `Ημερομηνία χειρουργείου για ${row.full_name}`);
      if (row.room_entry_time) dateInput.title = "Άλλαξε ημερομηνία και ώρα από την επεξεργασία ✎";
      dateCell.appendChild(dateInput);
      tr.appendChild(dateCell);
      tr.appendChild(cell(row.room_entry_time || "—"));

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
    td.colSpan = 10;
    td.className = "surgery-empty";
    td.textContent = message;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }

  function setSurgeryWritesEnabled(enabled) {
    for (const control of $("surgeryForm").elements) control.disabled = !enabled;
  }

  function clearSurgeryProjection(message, needsKey = false) {
    surgeryReady = false;
    surgeryRows = [];
    window.CockpitSurgerySummary = null;
    if (typeof window.dispatchEvent === "function" && typeof window.CustomEvent === "function") {
      window.dispatchEvent(new window.CustomEvent("cockpit:surgery-counts", {
        detail: { unavailable: true }
      }));
    }
    setSurgeryUnavailable(message);
    setSurgeryWritesEnabled(false);
    $("surgeryQueueNote").textContent = needsKey
      ? "Πάτησε «Πρόσβαση» πάνω δεξιά για σύνδεση με το Clinical Key."
      : "Η προστατευμένη λίστα χειρουργείων δεν είναι διαθέσιμη αυτή τη στιγμή.";
    resetSurgeryForm();
    $("surgeryEditorDetails").open = false;
  }

  async function loadSurgeryQueue() {
    const generation = ++surgeryGeneration;
    // Invalidate immediately, including Clinical Key re-auth and post-write
    // reloads. An older success or failure can never restore/clear a newer view.
    clearSurgeryProjection("Φόρτωση…");
    try {
      const rows = await apiJson("/clinical/surgeries", {}, () => generation === surgeryGeneration);
      if (generation !== surgeryGeneration) return;
      if (!validSurgeryRows(rows)) throw new Error("Invalid protected surgery source");
      surgeryRows = rows;
      surgeryReady = true;
      setSurgeryWritesEnabled(true);
      $("surgeryQueueState").textContent = String(rows.length) + " pending";
      $("surgeryQueueState").classList.add("ready");
      $("surgeryQueueNote").textContent = rows.length
        ? "Η σειρά # είναι η αποθηκευμένη manual σειρά. Τα sortable headers αλλάζουν μόνο την προβολή."
        : "Δεν υπάρχουν pending χειρουργεία. Πρόσθεσε το πρώτο όταν χρειαστεί.";
      renderSurgeryRows();
    } catch (error) {
      if (generation !== surgeryGeneration) return;
      const needsKey = error.status === 401 || error.status === 403;
      clearSurgeryProjection(needsKey ? "Σύνδεση απαιτείται" : "Μη διαθέσιμο", needsKey);
    }
  }

    function resetSurgeryForm() {
    $("surgeryForm").reset();
    $("surgeryId").value = "";
    $("surgeryRfApplicationId").value = "";
    $("surgeryRfRevision").value = "";
    $("surgeryRfSourceNote").hidden = true;
    $("surgeryLaterality").value = "unspecified";
    $("surgerySubmit").textContent = "Προσθήκη";
    $("surgeryCancelEdit").hidden = true;
  }

  function startSurgeryEdit(row) {
    resetSurgeryForm();
    $("surgeryId").value = row.surgery_id;
    $("surgeryFullName").value = row.full_name || "";
    $("surgeryIdentityNumber").value = row.identity_number || "";
    $("surgeryDateOfBirth").value = row.date_of_birth || "";
    $("surgeryPhone").value = row.phone || "";
    $("surgeryProcedureType").value = row.procedure_type || "";
    $("surgeryLaterality").value = row.laterality || "unspecified";
    $("surgeryDate").value = row.surgery_date || "";
    $("surgeryRoomEntryTime").value = row.room_entry_time || "";
    if (row.rf_application_id) {
      $("surgeryRfSourceNote").textContent = "Συνδεδεμένη με RF: " + row.rf_application_id;
      $("surgeryRfSourceNote").hidden = false;
    }
    $("surgerySubmit").textContent = "Αποθήκευση";
    $("surgeryCancelEdit").hidden = false;
    $("surgeryEditorDetails").open = true;
    $("surgeryFullName").focus();
  }

  async function submitSurgeryForm(event) {
    event.preventDefault();
    const writeEpoch = surgeryAuthEpoch;
    const surgeryId = $("surgeryId").value;
    const rfSourceId = $("surgeryRfApplicationId").value;
    const payload = {
      full_name: $("surgeryFullName").value.trim(),
      identity_number: $("surgeryIdentityNumber").value.trim(),
      date_of_birth: $("surgeryDateOfBirth").value,
      phone: $("surgeryPhone").value.trim(),
      procedure_type: $("surgeryProcedureType").value.trim(),
      laterality: $("surgeryLaterality").value,
      surgery_date: $("surgeryDate").value || null,
      room_entry_time: $("surgeryRoomEntryTime").value || null
    };

    try {
      if (surgeryId) {
        await apiJson(`/clinical/surgeries/${encodeURIComponent(surgeryId)}`, {
          method: "PUT",
          body: JSON.stringify(payload)
        });
      } else if (rfSourceId) {
        await apiJson("/clinical/office-cases/surgeries/from-rf/" +
          encodeURIComponent(rfSourceId), {
          method: "POST",
          body: JSON.stringify({
            expected_rf_revision: Number($("surgeryRfRevision").value),
            date_of_birth: payload.date_of_birth,
            phone: payload.phone,
            procedure_type: payload.procedure_type,
            surgery_date: payload.surgery_date,
            room_entry_time: payload.room_entry_time
          })
        });
      } else {
        await apiJson("/clinical/surgeries", {
          method: "POST",
          body: JSON.stringify(payload)
        });
      }
      if (writeEpoch !== surgeryAuthEpoch) return;
      resetSurgeryForm();
      $("surgeryEditorDetails").open = false;
      surgerySortKey = "queue_position";
      surgerySortDirection = "asc";
      await loadSurgeryQueue();
      window.CockpitOfficeCases?.refreshRF?.();
    } catch (error) {
      if (writeEpoch !== surgeryAuthEpoch) return;
      if (rfSourceId && error.status === 409) {
        resetSurgeryForm();
        window.CockpitOfficeCases?.showRFConflict?.(rfSourceId);
        return;
      }
      $("surgeryQueueNote").textContent = error.detail || "Δεν ήταν δυνατή η αποθήκευση της εγγραφής.";
      if ([401, 403, 503].includes(error.status)) await loadSurgeryQueue();
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
      const writeEpoch = surgeryAuthEpoch;
      try {
        await apiJson(`/clinical/surgeries/${encodeURIComponent(row.surgery_id)}/complete`, {
          method: "POST",
          body: "{}"
        });
        if (writeEpoch !== surgeryAuthEpoch) return;
        await loadSurgeryQueue();
      } catch (error) {
        if (writeEpoch !== surgeryAuthEpoch) return;
        $("surgeryQueueNote").textContent = error.detail || "Δεν ολοκληρώθηκε η ενημέρωση.";
        if ([401, 403, 503].includes(error.status)) await loadSurgeryQueue();
      }
      return;
    }

    if (action === "delete") {
      if (!window.confirm(`Να διαγραφεί από τα pending το χειρουργείο για ${row.full_name};`)) return;
      const writeEpoch = surgeryAuthEpoch;
      try {
        await apiJson(`/clinical/surgeries/${encodeURIComponent(row.surgery_id)}`, {
          method: "DELETE"
        });
        if (writeEpoch !== surgeryAuthEpoch) return;
        await loadSurgeryQueue();
      } catch (error) {
        if (writeEpoch !== surgeryAuthEpoch) return;
        $("surgeryQueueNote").textContent = error.detail || "Δεν ήταν δυνατή η διαγραφή.";
        if ([401, 403, 503].includes(error.status)) await loadSurgeryQueue();
      }
      return;
    }

    if (action === "up" || action === "down") {
      const writeEpoch = surgeryAuthEpoch;
      try {
        await apiJson("/clinical/surgeries/" + encodeURIComponent(row.surgery_id) + "/move", {
          method: "POST",
          body: JSON.stringify({ direction: action })
        });
        if (writeEpoch !== surgeryAuthEpoch) return;
        surgerySortKey = "queue_position";
        surgerySortDirection = "asc";
        await loadSurgeryQueue(); // Never publish unguarded mutation responses.
      } catch (error) {
        if (writeEpoch !== surgeryAuthEpoch) return;
        $("surgeryQueueNote").textContent = error.detail || "Δεν άλλαξε η σειρά.";
        if ([401, 403, 503].includes(error.status)) await loadSurgeryQueue();
      }
    }
  }

    async function handleSurgeryDateChange(event) {
    const input = event.target.closest("input.surgery-date-input");
    if (!input) return;
    const writeEpoch = surgeryAuthEpoch;
    try {
      await apiJson("/clinical/surgeries/" + encodeURIComponent(input.dataset.surgeryId), {
        method: "PUT",
        body: JSON.stringify({ surgery_date: input.value || null })
      });
      if (writeEpoch !== surgeryAuthEpoch) return;
      await loadSurgeryQueue(); // Protected read is always generation-guarded.
    } catch (error) {
      if (writeEpoch !== surgeryAuthEpoch) return;
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
  const calendarRetry = $("cockpitCalendarRetry");
  if (calendarRetry) calendarRetry.addEventListener("click", loadCockpitContext);
  if (typeof window !== "undefined" && typeof window.addEventListener === "function") window.addEventListener("clinical:authenticated", () => {
    ++surgeryAuthEpoch;
    loadCockpitContext();
    loadSurgeryQueue();
  });
  if (typeof window !== "undefined" && typeof window.addEventListener === "function") window.addEventListener("clinical:unauthenticated", () => {
    ++surgeryAuthEpoch;
    ++surgeryGeneration; // Fence any protected read already in flight.
    clearSurgeryProjection("Σύνδεση απαιτείται", true);
  });
  resetSurgeryForm();
  window.CockpitSurgeryQueue = {
    prefillFromRF(card) {
      if (!surgeryReady || !card || !card.id || !Number.isInteger(card.revision) ||
          !["approved", "scheduled"].includes(card.stage)) return;
      resetSurgeryForm();
      $("surgeryRfApplicationId").value = card.id;
      $("surgeryRfRevision").value = String(card.revision);
      $("surgeryFullName").value = card.patient_name || "";
      $("surgeryIdentityNumber").value = card.identity_number || "";
      $("surgeryLaterality").value = ["left", "right"].includes(card.laterality) ? card.laterality : "unspecified";
      $("surgeryDate").value = card.stage === "scheduled" ? card.scheduled_date || "" : "";
      $("surgeryRfSourceNote").textContent = "Από RF: έλεγξε τα στοιχεία και συμπλήρωσε ημερομηνία γέννησης, τηλέφωνο και επέμβαση πριν την προσθήκη.";
      $("surgeryRfSourceNote").hidden = false;
      window.CockpitOfficeCases?.selectTab?.("surgery");
      $("surgeryEditorDetails").open = true;
      $("surgeryDateOfBirth").focus();
    }
  };
  loadCockpitContext();
  window.setInterval(loadCockpitContext, 60000);
  loadSurgeryQueue();
})();
