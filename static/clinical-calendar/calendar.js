(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  let weekOffset = 0;
  let calendarMode = "all";
  let weekRequest = 0;
  let authGeneration = 0;
  let appointmentRows = new Map();
  let weekRows = null;
  let displayedMonday = null;
  let weekReadAt = null;
  let weekState = "loading";
  const DAY_COUNT_KEY = "clinical-excellence.calendar-day-count.v1";
  let dayCount = 7;
  try { if (window.localStorage.getItem(DAY_COUNT_KEY) === "5") dayCount = 5; } catch (_) { /* Session-only fallback. */ }
  let classificationPending = false;
  let classificationNeedsRefresh = false;
  let classificationController = null;
  const RELEVANT_CATEGORIES = new Set(["osteoporosis_first", "osteoporosis_review",
    "osteoporosis_unspecified", "prolia", "aclasta"]);

  const CATEGORY_LABELS = {
    osteoporosis_first: "1ο ραντεβού οστεοπόρωσης",
    osteoporosis_review: "Review οστεοπόρωσης",
    osteoporosis_unspecified: "Οστεοπόρωση · ταξινόμηση",
    prolia: "Prolia",
    aclasta: "Aclasta",
    other: "Ραντεβού",
  };

  const MANUAL_CLASSIFICATION_OPTIONS = [
    ["osteoporosis_first", "Πρώτη επίσκεψη"],
    ["osteoporosis_review", "Επανέλεγχος"],
    ["prolia", "Prolia"],
    ["aclasta", "Aclasta"],
  ];

  async function api(path, options = {}) {
    const res = await fetch(path, {
      credentials: "same-origin",
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    let body = null;
    try { body = await res.json(); } catch { body = null; }
    if (!res.ok) {
      const detail = body?.detail;
      const message = typeof detail === "string"
        ? detail
        : (detail?.message || detail?.status || `HTTP ${res.status}`);
      const err = new Error(message);
      err.status = res.status;
      err.detail = detail;
      throw err;
    }
    return body;
  }

  function setStatus(text, kind = "") {
    const n = $("#connectionStatus");
    if (!n) return;
    n.textContent = text;
    n.className = `status ${kind}`;
  }

  async function checkAuth() {
    const ticket = ++authGeneration;
    try {
      const status = await api("/clinical/status", { method: "GET", headers: {} });
      if (ticket !== authGeneration) return;
      $("#authCard").hidden = true;
      $("#calendarApp").hidden = false;
      setStatus(`Protected DB · ${status.database_dialect}`, "ok");
      await loadWeek();
    } catch (err) {
      if (ticket !== authGeneration) return;
      clearProtectedCalendar();
      const msg = $("#authMessage");
      msg.textContent = err.status === 503 ? "Το clinical access δεν έχει ρυθμιστεί." : "Απαιτείται σύνδεση.";
      msg.className = "status err";
    }
  }

  function clearProtectedCalendar() {
    ++authGeneration;
    ++weekRequest;
    $("#calendarApp").hidden = true;
    $("#authCard").hidden = false;
    $("#weekGrid").replaceChildren();
    closeAppointment();
    appointmentRows.clear();
    weekRows = null;
    weekReadAt = null;
    $("#calendarSearch").value = "";
    $("#weekendNotice").replaceChildren();
    $("#weekendNotice").hidden = true;
    for (const id of ["#countOsteoporosis", "#countProlia", "#countAclasta", "#countUnspecified"])
      $(id).textContent = "—";
    setStatus("");
    $("#authMessage").textContent = "Απαιτείται σύνδεση.";
  }

  async function login() {
    const key = $("#keyInput").value;
    if (!key) return;
    try {
      await api("/clinical/login", { method: "POST", body: JSON.stringify({ key }) });
      $("#keyInput").value = "";
      await checkAuth();
    } catch (err) {
      const msg = $("#authMessage");
      msg.textContent = err.message;
      msg.className = "status err";
    }
  }

  function mondayForOffset(offset) {
    const parts = Object.fromEntries(new Intl.DateTimeFormat("en-GB", {
      timeZone: "Asia/Nicosia", year: "numeric", month: "2-digit", day: "2-digit"
    }).formatToParts(new Date()).filter(p => p.type !== "literal").map(p => [p.type, Number(p.value)]));
    const d = new Date(Date.UTC(parts.year, parts.month - 1, parts.day, 12));
    const weekday = d.getUTCDay();
    const diff = weekday === 0 ? -6 : 1 - weekday;
    d.setUTCDate(d.getUTCDate() + diff + offset * 7);
    return d;
  }

  function addDays(date, days) {
    const d = new Date(date);
    d.setUTCDate(d.getUTCDate() + days);
    return d;
  }

  function dateKey(date) {
    const parts = Object.fromEntries(new Intl.DateTimeFormat("en-GB", {
      timeZone: "Asia/Nicosia", year: "numeric", month: "2-digit", day: "2-digit"
    }).formatToParts(date).filter(p => p.type !== "literal").map(p => [p.type, p.value]));
    return `${parts.year}-${parts.month}-${parts.day}`;
  }

  function cyprusMidnight(date) {
    const [year, month, day] = dateKey(date).split("-").map(Number);
    const probe = new Date(Date.UTC(year, month - 1, day));
    const parts = Object.fromEntries(new Intl.DateTimeFormat("en-GB", {
      timeZone: "Asia/Nicosia", year:"numeric", month:"2-digit", day:"2-digit",
      hour:"2-digit", minute:"2-digit", second:"2-digit", hourCycle:"h23"
    }).formatToParts(probe).filter(p => p.type !== "literal").map(p => [p.type, Number(p.value)]));
    const offset = Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute, parts.second) - probe.getTime();
    return new Date(Date.UTC(year, month - 1, day) - offset);
  }

  function parseServerDate(value) {
    if (!value) return null;
    const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(value);
    return new Date(hasZone ? value : `${value}Z`);
  }

  const fmtDay = new Intl.DateTimeFormat("el-CY", { timeZone: "Asia/Nicosia", weekday: "long" });
  const fmtDate = new Intl.DateTimeFormat("el-CY", { timeZone: "Asia/Nicosia", day: "2-digit", month: "2-digit" });
  const fmtLong = new Intl.DateTimeFormat("el-CY", { timeZone: "Asia/Nicosia", day: "2-digit", month: "short", year: "numeric" });
  const fmtTime = new Intl.DateTimeFormat("el-CY", { timeZone: "Asia/Nicosia", hour: "2-digit", minute: "2-digit", hour12: false });

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
  }

  function classificationControl(row, category) {
    if (category !== "osteoporosis_unspecified" && !row.manual_category) return "";
    const placeholder = row.manual_category ? "Αυτόματο" : "Ταξινόμηση…";
    const options = MANUAL_CLASSIFICATION_OPTIONS.map(([value, label]) =>
      `<option value="${esc(value)}" ${row.manual_category === value ? "selected" : ""}>${esc(label)}</option>`
    ).join("");
    return `<label class="classification-control">
      <span>${row.manual_category ? "Χειροκίνητη ταξινόμηση" : "Χρειάζεται ταξινόμηση"}</span>
      <select class="classification-select" data-appointment-id="${esc(row.appointment_id)}">
        <option value="">${esc(placeholder)}</option>
        ${options}
      </select>
    </label>`;
  }

  function appointmentHtml(row, key) {
    const start = parseServerDate(row.start_at);
    const category = row.category || "other";
    const label = CATEGORY_LABELS[category] || "Ραντεβού";
    const patient = row.patient_display_name || row.linked_patient_id || "Χωρίς αντιστοίχιση ασθενούς";
    const reason = typeof row.reason === "string" ? row.reason.replace(/\s+/g, " ").trim() : "";
    const time = start ? fmtTime.format(start) : "—";
    const accessible = time + " · " + patient + " · " + label + (reason ? " · " + reason : "");
    const marker = category === "other" ? "" : `<span class="appt-indicator ${esc(category)}" aria-hidden="true"></span>`;
    return `<button type="button" class="appt-row" data-appointment-key="${esc(key)}" data-base-label="${esc(accessible)}" aria-label="${esc(accessible)}" title="${esc(patient + " · " + label + (reason ? " · " + reason : ""))}">
      <time>${esc(time)}</time><span class="appt-copy"><span class="appt-name">${esc(patient)}</span>${reason ? `<span class="appt-reason">${esc(reason)}</span>` : ""}</span>${marker}<span class="appt-now" hidden aria-hidden="true">Τώρα</span>
    </button>`;
  }

  function closeAppointment() {
    const dialog = $("#appointmentDialog");
    if (classificationPending) {
      classificationNeedsRefresh = true;
      classificationController?.abort();
      classificationPending = false;
    }
    if (dialog.open) dialog.close();
    $("#appointmentDialogContent").replaceChildren();
  }

  function showUnconfirmedClassification() {
    const content = $("#appointmentDialogContent");
    const select = content.querySelector(".classification-select");
    const status = content.querySelector("[role='status']");
    if (select) select.disabled = true;
    if (!status) return;
    status.textContent = "Η αποθήκευση δεν επιβεβαιώθηκε και μπορεί να έχει ολοκληρωθεί ή να ολοκληρωθεί αργότερα. Ανανέωσε το πρόγραμμα και έλεγξε την ταξινόμηση πριν ξαναπροσπαθήσεις. ";
    const refresh = document.createElement("button");
    refresh.type = "button";
    refresh.className = "button ghost";
    refresh.textContent = "Ανανέωση προγράμματος";
    refresh.addEventListener("click", () => void loadWeek());
    status.append(refresh);
  }

  function openAppointment(entry) {
    const {row, ambiguous} = entry;
    const start = parseServerDate(row.start_at);
    const end = parseServerDate(row.end_at);
    const category = row.category || "other";
    const label = CATEGORY_LABELS[category] || "Ραντεβού";
    const name = row.patient_display_name || row.linked_patient_id || "Χωρίς αντιστοίχιση ασθενούς";
    const meta = [row.status, row.clinic].filter(Boolean).join(" · ");
    const duration = row.duration_minutes ?? (start && end ? Math.round((end - start) / 60000) : null);
    const time = start ? fmtTime.format(start) : "—";
    const dialog = $("#appointmentDialog");
    $("#appointmentDialogContent").innerHTML = `<div class="appointment-dialog-header">
      <div><p class="appointment-dialog-date">${start ? esc(fmtLong.format(start)) : ""} · ${esc(time)}${Number.isFinite(duration) && duration > 0 ? " · " + esc(duration) + "′" : ""}</p>
      <h2 id="appointmentDialogTitle">${esc(name)}</h2></div>
      <button type="button" class="appointment-dialog-close" aria-label="Κλείσιμο">×</button></div>
      <p><span class="badge ${esc(category)}">${esc(label)}</span></p>
      ${row.reason ? `<p class="appointment-dialog-reason">${esc(row.reason)}</p>` : '<p class="appointment-dialog-reason">Δεν έχει καταχωριστεί λόγος επίσκεψης.</p>'}
      ${meta ? `<p class="appointment-dialog-meta">${esc(meta)}</p>` : ""}
      ${ambiguous ? '<p class="appointment-dialog-status">Το αναγνωριστικό αυτής της κράτησης εμφανίζεται περισσότερες από μία φορές. Έλεγξε την κράτηση στη Γραμματεία πριν ανοίξεις καρτέλα ή αλλάξεις ταξινόμηση.</p>' : classificationControl(row, category)}
      <p class="appointment-dialog-status" role="status"></p>
      ${ambiguous ? "" : `<a class="button primary appointment-chart" href="/static/cockpit/?booking=${encodeURIComponent(row.appointment_id)}&amp;booking_date=${start ? dateKey(start) : ''}">Καρτέλα · Τελευταία επίσκεψη</a>`}`;
    if (classificationNeedsRefresh) showUnconfirmedClassification();
    dialog.showModal();
  }

  function normalizeSearch(value) {
    return String(value || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "")
      .toLocaleLowerCase("el").replace(/ς/g, "σ").replace(/\s+/g, " ").trim();
  }

  function clinicKey(value) {
    const key = normalizeSearch(value);
    if (["limassol", "λεμεσοσ", "λεμεσσοσ"].includes(key)) return "limassol";
    if (["evrychou", "enrychou", "ευρυχου"].includes(key)) return "evrychou";
    return "unknown";
  }

  function refreshTimeIndicators() {
    const now = new Date();
    const today = dateKey(now);
    $("#weekGrid").querySelectorAll(".day-column[data-date]").forEach(col => {
      const current = col.dataset.date === today;
      col.classList.toggle("is-today", current);
      const label = col.querySelector(".today-label");
      if (label) label.hidden = !current;
    });
    $("#weekGrid").querySelectorAll(".appt-row").forEach(button => {
      const entry = appointmentRows.get(button.dataset.appointmentKey);
      if (!entry) return;
      const start = parseServerDate(entry.row.start_at);
      const end = parseServerDate(entry.row.end_at);
      const active = start && end && Number.isFinite(start.getTime()) && Number.isFinite(end.getTime())
        && start.getTime() <= now.getTime() && now.getTime() < end.getTime();
      button.classList.toggle("is-current", !!active);
      const badge = button.querySelector(".appt-now");
      if (badge) badge.hidden = !active;
      if (active) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
      button.setAttribute("aria-label", button.dataset.baseLabel + (active ? " · Προγραμματισμένο τώρα" : ""));
    });
  }

  function renderWeek(rows, monday, mode) {
    const counts = new Map();
    (rows || []).forEach(row => counts.set(String(row.appointment_id), (counts.get(String(row.appointment_id)) || 0) + 1));
    appointmentRows = new Map((rows || []).map((row, index) =>
      [String(index), {row, ambiguous: counts.get(String(row.appointment_id)) > 1}]));
    const grouped = new Map();
    const query = normalizeSearch($("#calendarSearch").value);
    const clinic = $("#calendarClinic").value;
    let weekendCount = 0;
    let visibleCount = 0;
    const visibleRows = [];
    const saturday = dateKey(addDays(monday, 5));
    const sunday = dateKey(addDays(monday, 6));
    for (let i = 0; i < 7; i += 1) grouped.set(dateKey(addDays(monday, i)), []);
    appointmentRows.forEach((entry, rowKey) => {
      const row = entry.row;
      const start = parseServerDate(row.start_at);
      if (!start) return;
      const dayKey = dateKey(start);
      if (dayKey === saturday || dayKey === sunday) weekendCount++;
      if (mode === "osteoporosis" && !RELEVANT_CATEGORIES.has(row.category)) return;
      if (mode !== "all" && mode !== "osteoporosis" && row.category !== mode) return;
      if (clinic !== "all" && clinicKey(row.clinic) !== clinic) return;
      const haystack = normalizeSearch((row.patient_display_name || "") + " " + (row.reason || "") + " " + (row.clinic || "") + " " + (CATEGORY_LABELS[row.category] || ""));
      if (query && !haystack.includes(query)) return;
      if (dayCount === 5 && (dayKey === saturday || dayKey === sunday)) return;
      if (grouped.has(dayKey)) {
        grouped.get(dayKey).push({row, key: rowKey});
        visibleRows.push(row);
      }
    });

    const grid = $("#weekGrid");
    grid.innerHTML = "";
    grid.dataset.days = String(dayCount);
    for (let i = 0; i < dayCount; i += 1) {
      const day = addDays(monday, i);
      const key = dateKey(day);
      const items = grouped.get(key) || [];
      visibleCount += items.length;
      const col = document.createElement("section");
      col.className = "day-column";
      col.dataset.date = key;
      col.innerHTML = `<div class="day-head"><strong>${esc(fmtDay.format(day))} <span class="today-label" hidden>· Σήμερα</span></strong><span>${esc(fmtDate.format(day))} · ${items.length}</span></div><div class="day-body">${items.length ? items.map(item => appointmentHtml(item.row, item.key)).join("") : '<div class="empty-day">Δεν υπάρχουν ραντεβού σε αυτή την προβολή.</div>'}</div>`;
      grid.appendChild(col);
    }
    const notice = $("#weekendNotice");
    notice.replaceChildren();
    notice.hidden = dayCount !== 5;
    if (dayCount === 5) {
      notice.append(document.createTextNode("Σαββατοκύριακο: " + weekendCount + " ραντεβού συνολικά, ανεξάρτητα από φίλτρα · "));
      const show = document.createElement("button");
      show.type = "button";
      show.textContent = "Προβολή 7 ημερών";
      show.addEventListener("click", () => setDayCount(7));
      notice.append(show);
    }
    refreshTimeIndicators();
    const modeLabel = mode === "all" ? "όλες οι κατηγορίες" : mode === "osteoporosis" ? "οστεοπόρωση" : (CATEGORY_LABELS[mode] || "επιλεγμένη κατηγορία");
    setStatus(`${visibleCount} ραντεβού · ${modeLabel}${query ? " · αναζήτηση" : ""}${weekReadAt ? " · ανάγνωση " + fmtTime.format(weekReadAt) : ""}`, "ok");

    const osteoporosis = visibleRows.filter(x => ["osteoporosis_first", "osteoporosis_review", "osteoporosis_unspecified"].includes(x.category)).length;
    $("#countOsteoporosis").textContent = osteoporosis;
    $("#countProlia").textContent = visibleRows.filter(x => x.category === "prolia").length;
    $("#countAclasta").textContent = visibleRows.filter(x => x.category === "aclasta").length;
    $("#countUnspecified").textContent = visibleRows.filter(x => x.category === "osteoporosis_unspecified").length;
  }

  function renderWeekUnavailable(monday, loading = false) {
    appointmentRows.clear();
    const grid = $("#weekGrid");
    grid.innerHTML = "";
    grid.dataset.days = String(dayCount);
    $("#weekendNotice").hidden = true;
    $("#weekendNotice").replaceChildren();
    for (let i = 0; i < dayCount; i += 1) {
      const day = addDays(monday, i);
      const col = document.createElement("section");
      col.className = "day-column";
      col.innerHTML = `<div class="day-head"><strong>${esc(fmtDay.format(day))}</strong><span>${esc(fmtDate.format(day))}</span></div><div class="day-body"><div class="empty-day">${loading ? "Φόρτωση…" : "Το ημερολόγιο δεν είναι διαθέσιμο αυτή τη στιγμή."}</div></div>`;
      grid.appendChild(col);
    }
    $("#countOsteoporosis").textContent = "—";
    $("#countProlia").textContent = "—";
    $("#countAclasta").textContent = "—";
    $("#countUnspecified").textContent = "—";
  }

  async function loadWeek() {
    closeAppointment();
    const ticket = ++weekRequest;
    const monday = mondayForOffset(weekOffset);
    displayedMonday = monday;
    weekRows = null;
    weekReadAt = null;
    weekState = "loading";
    const end = addDays(monday, 7);
    $("#weekLabel").textContent = `${fmtLong.format(monday)} — ${fmtLong.format(addDays(monday, 6))}`;
    setStatus("Φόρτωση…");
    renderWeekUnavailable(monday, true);
    try {
      const rows = await api(`/clinical/calendar/agenda?start=${encodeURIComponent(cyprusMidnight(monday).toISOString())}&end=${encodeURIComponent(cyprusMidnight(end).toISOString())}`);
      if (ticket !== weekRequest) return;
      if (!Array.isArray(rows)) throw new Error("Invalid schedule");
      weekRows = rows;
      weekReadAt = new Date();
      weekState = "ready";
      renderWeek(rows, monday, calendarMode);
      classificationNeedsRefresh = false;
    } catch (err) {
      if (ticket !== weekRequest) return;
      if (err.status === 401 || err.status === 403) { clearProtectedCalendar(); return; }
      const last = parseServerDate(err.detail?.last_fetched_at);
      const freshness = last ? ` · τελευταία ανάγνωση ${fmtLong.format(last)} ${fmtTime.format(last)}` : "";
      setStatus(`Μη διαθέσιμο${freshness}`, "err");
      weekState = "unavailable";
      renderWeekUnavailable(monday);
    }
  }

  $("#loginBtn").addEventListener("click", login);
  $("#keyInput").addEventListener("keydown", e => { if (e.key === "Enter") login(); });
  $("#prevWeek").addEventListener("click", () => { weekOffset -= 1; loadWeek(); });
  $("#thisWeek").addEventListener("click", () => { weekOffset = 0; loadWeek(); });
  $("#nextWeek").addEventListener("click", () => { weekOffset += 1; loadWeek(); });
  $("#refreshWeek").addEventListener("click", () => loadWeek());
  function changeMode() {
    calendarMode = $("#calendarCategory").value;
    if (weekRows) renderWeek(weekRows, displayedMonday, calendarMode);
  }
  $("#calendarCategory").addEventListener("change", changeMode);
  $("#calendarClinic").addEventListener("change", () => {
    if (weekRows) renderWeek(weekRows, displayedMonday, calendarMode);
  });
  function setDayCount(value) {
    dayCount = value === 5 ? 5 : 7;
    $("#showFiveDays").setAttribute("aria-pressed", String(dayCount === 5));
    $("#showSevenDays").setAttribute("aria-pressed", String(dayCount === 7));
    try { window.localStorage.setItem(DAY_COUNT_KEY, String(dayCount)); } catch (_) { /* Session-only fallback. */ }
    if (weekRows) renderWeek(weekRows, displayedMonday, calendarMode);
    else if (displayedMonday) renderWeekUnavailable(displayedMonday, weekState === "loading");
  }
  $("#showFiveDays").addEventListener("click", () => setDayCount(5));
  $("#showSevenDays").addEventListener("click", () => setDayCount(7));
  $("#calendarSearch").addEventListener("input", () => {
    if (weekRows) renderWeek(weekRows, displayedMonday, calendarMode);
  });
  $("#showFiveDays").setAttribute("aria-pressed", String(dayCount === 5));
  $("#showSevenDays").setAttribute("aria-pressed", String(dayCount === 7));
  if (typeof window.setInterval === "function") window.setInterval(refreshTimeIndicators, 30000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshTimeIndicators(); });

  $("#weekGrid").addEventListener("click", event => {
    const button = event.target.closest(".appt-row");
    const row = button && appointmentRows.get(button.dataset.appointmentKey);
    if (row) openAppointment(row);
  });
  $("#appointmentDialog").addEventListener("click", event => {
    if (event.target.closest(".appointment-dialog-close") && !classificationPending) closeAppointment();
  });
  $("#appointmentDialog").addEventListener("cancel", event => {
    if (classificationPending) event.preventDefault();
  });
  $("#appointmentDialog").addEventListener("close", () => {
    if (!$("#appointmentDialog").open) $("#appointmentDialogContent").replaceChildren();
  });
  $("#appointmentDialog").addEventListener("change", async (event) => {
    const select = event.target.closest(".classification-select");
    if (!select || classificationPending || classificationNeedsRefresh) return;
    const ticket = weekRequest;
    const closeButton = $("#appointmentDialogContent .appointment-dialog-close");
    const status = $("#appointmentDialogContent [role='status']");
    const controller = new AbortController();
    classificationController = controller;
    classificationPending = true;
    if (closeButton) closeButton.disabled = true;
    if (status) status.textContent = "Αποθήκευση ταξινόμησης…";
    select.disabled = true;
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      await api(
        `/clinical/calendar/appointments/${encodeURIComponent(select.dataset.appointmentId)}/classification`,
        {
          method: "PUT",
          signal: controller.signal,
          body: JSON.stringify({ category: select.value || null }),
        }
      );
      if (classificationController === controller) {
        classificationPending = false;
        classificationController = null;
      }
      if (ticket === weekRequest) await loadWeek();
    } catch (err) {
      if (classificationController === controller) {
        classificationPending = false;
        classificationController = null;
      }
      if (ticket !== weekRequest) return;
      if (err.status === 401 || err.status === 403) { clearProtectedCalendar(); return; }
      classificationNeedsRefresh = true;
      if (closeButton) closeButton.disabled = false;
      showUnconfirmedClassification();
    } finally {
      clearTimeout(timeout);
    }
  });

  if (typeof window !== "undefined" && typeof window.addEventListener === "function") window.addEventListener("clinical:authenticated", checkAuth);
  if (typeof window !== "undefined" && typeof window.addEventListener === "function") window.addEventListener("clinical:unauthenticated", clearProtectedCalendar);
  checkAuth();
})();
