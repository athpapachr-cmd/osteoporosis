(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  let weekOffset = 0;
  let calendarMode = "all";
  let weekRequest = 0;
  let authGeneration = 0;

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

  function appointmentHtml(row) {
    const start = parseServerDate(row.start_at);
    const category = row.category || "other";
    const label = CATEGORY_LABELS[category] || "Οστεοπόρωση";
    const patient = row.patient_display_name || row.linked_patient_id || "Χωρίς αντιστοίχιση ασθενούς";
    const reason = row.reason || "";
    const meta = [row.status, row.clinic].filter(Boolean).join(" · ");
    const end = parseServerDate(row.end_at);
    const duration = row.duration_minutes ?? (start && end ? Math.round((end - start) / 60000) : null);
    return `<article class="appt">
      <div class="appt-time"><strong>${start ? fmtTime.format(start) : "—"}</strong><span class="appt-duration">${Number.isFinite(duration) && duration > 0 ? esc(duration) + "′" : ""}</span></div>
      <span class="badge ${esc(category)}">${esc(label)}</span>
      <div class="appt-name">${esc(patient)}</div>
      <div class="appt-meta">${esc(meta)}</div>
      ${reason ? `<div class="appt-label">${esc(reason)}</div>` : ""}
      <a class="button ghost" href="/static/cockpit/?booking=${encodeURIComponent(row.appointment_id)}&amp;booking_date=${start ? dateKey(start) : ''}">Καρτέλα · Τελευταία επίσκεψη</a>
      ${classificationControl(row, category)}
    </article>`;
  }

  function renderWeek(rows, monday) {
    const grouped = new Map();
    for (let i = 0; i < 7; i += 1) grouped.set(dateKey(addDays(monday, i)), []);
    (rows || []).forEach(row => {
      const start = parseServerDate(row.start_at);
      if (!start) return;
      const key = dateKey(start);
      if (grouped.has(key)) grouped.get(key).push(row);
    });

    const grid = $("#weekGrid");
    grid.innerHTML = "";
    for (let i = 0; i < 7; i += 1) {
      const day = addDays(monday, i);
      const key = dateKey(day);
      const items = grouped.get(key) || [];
      const col = document.createElement("section");
      col.className = "day-column";
      col.innerHTML = `<div class="day-head"><strong>${esc(fmtDay.format(day))}</strong><span>${esc(fmtDate.format(day))}</span></div><div class="day-body">${items.length ? items.map(appointmentHtml).join("") : '<div class="empty-day">Δεν υπάρχουν ραντεβού σε αυτή την προβολή.</div>'}</div>`;
      grid.appendChild(col);
    }

    const osteoporosis = (rows || []).filter(x => ["osteoporosis_first", "osteoporosis_review", "osteoporosis_unspecified"].includes(x.category)).length;
    $("#countOsteoporosis").textContent = osteoporosis;
    $("#countProlia").textContent = (rows || []).filter(x => x.category === "prolia").length;
    $("#countAclasta").textContent = (rows || []).filter(x => x.category === "aclasta").length;
    $("#countUnspecified").textContent = (rows || []).filter(x => x.category === "osteoporosis_unspecified").length;
  }

  function renderWeekUnavailable(monday, loading = false) {
    const grid = $("#weekGrid");
    grid.innerHTML = "";
    for (let i = 0; i < 7; i += 1) {
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
    const ticket = ++weekRequest;
    const requestedMode = calendarMode;
    const monday = mondayForOffset(weekOffset);
    const end = addDays(monday, 7);
    $("#weekLabel").textContent = `${fmtLong.format(monday)} — ${fmtLong.format(addDays(monday, 6))}`;
    setStatus("Φόρτωση…");
    renderWeekUnavailable(monday, true);
    try {
      const endpoint = requestedMode === "all" ? "agenda" : "appointments";
      const rows = await api(`/clinical/calendar/${endpoint}?start=${encodeURIComponent(cyprusMidnight(monday).toISOString())}&end=${encodeURIComponent(cyprusMidnight(end).toISOString())}`);
      if (ticket !== weekRequest) return;
      if (!Array.isArray(rows)) throw new Error("Invalid schedule");
      renderWeek(rows, monday);
      setStatus(`${rows.length} ραντεβού · ${requestedMode === "all" ? "όλες οι κατηγορίες" : "οστεοπόρωση"}`, "ok");
    } catch (err) {
      if (ticket !== weekRequest) return;
      if (err.status === 401 || err.status === 403) { clearProtectedCalendar(); return; }
      const last = parseServerDate(err.detail?.last_fetched_at);
      const freshness = last ? ` · τελευταία ανάγνωση ${fmtLong.format(last)} ${fmtTime.format(last)}` : "";
      setStatus(`Μη διαθέσιμο${freshness}`, "err");
      renderWeekUnavailable(monday);
    }
  }

  $("#loginBtn").addEventListener("click", login);
  $("#keyInput").addEventListener("keydown", e => { if (e.key === "Enter") login(); });
  $("#prevWeek").addEventListener("click", () => { weekOffset -= 1; loadWeek(); });
  $("#thisWeek").addEventListener("click", () => { weekOffset = 0; loadWeek(); });
  $("#nextWeek").addEventListener("click", () => { weekOffset += 1; loadWeek(); });
  function changeMode(mode) {
    if (calendarMode === mode) return;
    calendarMode = mode;
    $("#showAllAppointments").setAttribute("aria-pressed", String(mode === "all"));
    $("#showOsteoporosisAppointments").setAttribute("aria-pressed", String(mode === "osteoporosis"));
    loadWeek();
  }
  $("#showAllAppointments").addEventListener("click", () => changeMode("all"));
  $("#showOsteoporosisAppointments").addEventListener("click", () => changeMode("osteoporosis"));

  $("#weekGrid").addEventListener("change", async (event) => {
    const select = event.target.closest(".classification-select");
    if (!select) return;
    select.disabled = true;
    try {
      await api(
        `/clinical/calendar/appointments/${encodeURIComponent(select.dataset.appointmentId)}/classification`,
        {
          method: "PUT",
          body: JSON.stringify({ category: select.value || null }),
        }
      );
      await loadWeek();
    } catch (err) {
      if (err.status === 401 || err.status === 403) { clearProtectedCalendar(); return; }
      setStatus(err.message, "err");
      select.disabled = false;
    }
  });

  if (typeof window !== "undefined" && typeof window.addEventListener === "function") window.addEventListener("clinical:authenticated", checkAuth);
  if (typeof window !== "undefined" && typeof window.addEventListener === "function") window.addEventListener("clinical:unauthenticated", clearProtectedCalendar);
  checkAuth();
})();
