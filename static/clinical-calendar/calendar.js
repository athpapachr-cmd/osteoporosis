(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  let weekOffset = 0;

  const CATEGORY_LABELS = {
    osteoporosis_first: "1ο ραντεβού οστεοπόρωσης",
    osteoporosis_review: "Review οστεοπόρωσης",
    osteoporosis_unspecified: "Οστεοπόρωση · ταξινόμηση",
    prolia: "Prolia",
    aclasta: "Aclasta",
    other: "Άλλο",
  };

  async function api(path, options = {}) {
    const res = await fetch(path, {
      credentials: "same-origin",
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    let body = null;
    try { body = await res.json(); } catch { body = null; }
    if (!res.ok) {
      const err = new Error(body?.detail || `HTTP ${res.status}`);
      err.status = res.status;
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
    try {
      const status = await api("/clinical/status", { method: "GET", headers: {} });
      $("#authCard").hidden = true;
      $("#calendarApp").hidden = false;
      setStatus(`Protected DB · ${status.database_dialect}`, "ok");
      await loadWeek();
    } catch (err) {
      $("#authCard").hidden = false;
      $("#calendarApp").hidden = true;
      const msg = $("#authMessage");
      msg.textContent = err.status === 503 ? "Το clinical access δεν έχει ρυθμιστεί." : "Απαιτείται σύνδεση.";
      msg.className = "status err";
    }
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
    const now = new Date();
    const d = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const weekday = d.getDay();
    const diff = weekday === 0 ? -6 : 1 - weekday;
    d.setDate(d.getDate() + diff + offset * 7);
    d.setHours(0, 0, 0, 0);
    return d;
  }

  function addDays(date, days) {
    const d = new Date(date);
    d.setDate(d.getDate() + days);
    return d;
  }

  function dateKey(date) {
    const y = date.getFullYear();
    const m = String(date.getMonth() + 1).padStart(2, "0");
    const d = String(date.getDate()).padStart(2, "0");
    return `${y}-${m}-${d}`;
  }

  function parseServerDate(value) {
    if (!value) return null;
    const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(value);
    return new Date(hasZone ? value : `${value}Z`);
  }

  const fmtDay = new Intl.DateTimeFormat("el-CY", { weekday: "long" });
  const fmtDate = new Intl.DateTimeFormat("el-CY", { day: "2-digit", month: "2-digit" });
  const fmtLong = new Intl.DateTimeFormat("el-CY", { day: "2-digit", month: "short", year: "numeric" });
  const fmtTime = new Intl.DateTimeFormat("el-CY", { hour: "2-digit", minute: "2-digit", hour12: false });

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
  }

  function appointmentHtml(row) {
    const start = parseServerDate(row.start_at);
    const category = row.category || "other";
    const label = CATEGORY_LABELS[category] || category;
    const patient = row.patient_display_name || row.linked_patient_id || "Χωρίς αντιστοίχιση ασθενούς";
    const detail = row.label || row.comment || "";
    const clinic = row.clinic ? ` · ${row.clinic}` : "";
    return `<article class="appt">
      <div class="appt-time"><strong>${start ? fmtTime.format(start) : "—"}</strong><span class="appt-duration">${row.duration_minutes || 0}'</span></div>
      <span class="badge ${esc(category)}">${esc(label)}</span>
      <div class="appt-name">${esc(patient)}</div>
      <div class="appt-meta">${esc(row.status || "scheduled")}${esc(clinic)}</div>
      ${detail ? `<div class="appt-label">${esc(detail)}</div>` : ""}
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
      col.innerHTML = `<div class="day-head"><strong>${esc(fmtDay.format(day))}</strong><span>${esc(fmtDate.format(day))}</span></div><div class="day-body">${items.length ? items.map(appointmentHtml).join("") : '<div class="empty-day">Δεν υπάρχουν σχετικά ραντεβού.</div>'}</div>`;
      grid.appendChild(col);
    }

    const osteoporosis = (rows || []).filter(x => ["osteoporosis_first", "osteoporosis_review", "osteoporosis_unspecified"].includes(x.category)).length;
    $("#countOsteoporosis").textContent = osteoporosis;
    $("#countProlia").textContent = (rows || []).filter(x => x.category === "prolia").length;
    $("#countAclasta").textContent = (rows || []).filter(x => x.category === "aclasta").length;
    $("#countUnspecified").textContent = (rows || []).filter(x => x.category === "osteoporosis_unspecified").length;
  }

  async function loadWeek() {
    const monday = mondayForOffset(weekOffset);
    const end = addDays(monday, 7);
    $("#weekLabel").textContent = `${fmtLong.format(monday)} — ${fmtLong.format(addDays(monday, 6))}`;
    setStatus("Φόρτωση…");
    try {
      const rows = await api(`/clinical/calendar/appointments?start=${encodeURIComponent(monday.toISOString())}&end=${encodeURIComponent(end.toISOString())}`);
      renderWeek(rows, monday);
      setStatus(`${rows.length} σχετικά ραντεβού`, "ok");
    } catch (err) {
      setStatus(err.message, "err");
      renderWeek([], monday);
    }
  }

  $("#loginBtn").addEventListener("click", login);
  $("#keyInput").addEventListener("keydown", e => { if (e.key === "Enter") login(); });
  $("#prevWeek").addEventListener("click", () => { weekOffset -= 1; loadWeek(); });
  $("#thisWeek").addEventListener("click", () => { weekOffset = 0; loadWeek(); });
  $("#nextWeek").addEventListener("click", () => { weekOffset += 1; loadWeek(); });

  checkAuth();
})();
