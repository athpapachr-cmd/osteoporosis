(() => {
  "use strict";
  // A bounded, read-only projection of the existing protected owners.
  const root = document.getElementById("doctorAttentionList");
  if (!root) return;
  const url = path => (window.CockpitClinicalBase || "/clinical") + path.slice(9);
  const dayParts = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Nicosia", year: "numeric", month: "2-digit", day: "2-digit"
  });
  const dateLabel = new Intl.DateTimeFormat("el-CY", {
    timeZone: "UTC", weekday: "short", day: "numeric", month: "short"
  });
  const state = {authorized: false, epoch: 0, load: 0, hidden: false,
    surgeries: null, rf: null, reports: null, tasks: null, calls: null, updated: 0, day: null};
  const nav = () => window.CockpitHomeAttentionNavigation || {};
  const today = () => {
    const p = Object.fromEntries(dayParts.formatToParts(new Date())
      .filter(x => x.type !== "literal").map(x => [x.type, x.value]));
    return p.year + "-" + p.month + "-" + p.day;
  };
  function validDate(value) {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    const [y, m, d] = value.split("-").map(Number);
    const t = new Date(Date.UTC(y, m - 1, d));
    return y >= 100 && t.getUTCFullYear() === y && t.getUTCMonth() + 1 === m && t.getUTCDate() === d;
  }
  const pretty = value => dateLabel.format(new Date(value + "T12:00:00Z"));
  const name = value => state.hidden ? "Απόκρυψη ονόματος" : value;
  const node = (tag, cls, value) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (value !== undefined) el.textContent = value;
    return el;
  };
  function row(title, detail, open, hot = false) {
    const b = node("button", "doctor-action-row" + (hot ? " doctor-attention-urgent" : ""));
    b.type = "button";
    const body = node("span"); body.append(node("strong", "", title));
    if (detail) body.append(node("small", "", detail));
    b.append(body, node("span", "", "›"));
    b.addEventListener("click", open); return b;
  }
  function section(title, value, entries, open, empty, coverage) {
    const box = node("section", "doctor-attention-section");
    box.append(node("h3", "doctor-attention-heading", title));
    if (value === undefined) box.append(node("p", "doctor-help", "Φόρτωση…"));
    else if (value === null) box.append(row("Μη διαθέσιμο", "Δεν γνωρίζουμε αν υπάρχουν εκκρεμότητες.", open));
    else {
      entries.slice(0, 4).forEach(entry => box.append(entry));
      if (entries.length > 4) box.append(row("Και " + (entries.length - 4) + " ακόμη", "Άνοιγμα λίστας", open));
      if (!entries.length) box.append(node("p", "doctor-help", empty));
      if (coverage) box.append(node("p", "doctor-attention-coverage", coverage));
    }
    root.append(box);
  }
  function draw() {
    root.replaceChildren();
    if (!state.authorized) {
      root.append(node("p", "doctor-help", "Συνδέσου για να δεις τις εκκρεμότητες.")); return;
    }
    const toggle = node("button", "doctor-attention-privacy", state.hidden ? "Εμφάνιση ονομάτων εδώ" : "Απόκρυψη ονομάτων εδώ");
    toggle.type = "button"; toggle.setAttribute("aria-pressed", String(state.hidden));
    toggle.addEventListener("click", () => { state.hidden = !state.hidden; draw(); });
    root.append(toggle);
    const now = today(), go = tab => () => nav().cases?.(tab);
    const s = state.surgeries;
    const surgeries = s ? [...s.attention].sort((a,b) => {
      const aPast = a.date < now, bPast = b.date < now;
      if (aPast !== bPast) return aPast ? 1 : -1;
      return aPast ? b.date.localeCompare(a.date) : a.date.localeCompare(b.date) || a.time.localeCompare(b.time);
    }).map(x => row(
      (x.date < now ? "Περασμένη ημερομηνία" : x.date === now ? "Σήμερα" : pretty(x.date)) + " · " + name(x.name),
      state.hidden ? "Χειρουργείο" : x.procedure + (x.time ? " · " + x.time : "") + (x.date < now ? " · Παραμένει εκκρεμές" : ""),
      go("surgery"), x.date <= now)) : [];
    if (s && (s.undated || s.invalid)) surgeries.push(row("Χειρουργεία χωρίς έγκυρη ημερομηνία · " + (s.undated + s.invalid), "Έλεγχος λίστας", go("surgery")));
    section("Χειρουργεία", s, surgeries, go("surgery"), "Δεν υπάρχουν εκκρεμή χειρουργεία στο επόμενο επταήμερο.");
    const rf = state.rf;
    const rfRows = rf ? rf.items.map(x => row("Εγκεκριμένη RF · " + name(x.patient_name), "Χωρίς σύνδεση με χειρουργείο", () => nav().cases?.("rf", x.id))) : [];
    section("RF", rf, rfRows, go("rf"), "Δεν βρέθηκαν ασύνδετες εγκεκριμένες RF στην προβολή.",
      rf && (rf.partial ? "Μερική κάλυψη· άνοιξε τη λίστα για περισσότερα." : "Στιγμιότυπο των διαθέσιμων εγγραφών."));
    const rep = state.reports;
    const reports = rep ? rep.items.map(x => row(
      (x.writing ? "Έκθεση σε σύνταξη · " : "Σταλμένη έκθεση με υπόλοιπο · ") + name(x.patient_name),
      x.writing ? "Έχει καταχωριστεί έναρξη σύνταξης" : "Υπάρχει καταχωρισμένο ανεξόφλητο υπόλοιπο",
      go("accidents"))) : [];
    section("Εκθέσεις · Τροχαία", rep, reports, go("accidents"), "Δεν βρέθηκαν τέτοιες καταχωρίσεις στην προβολή.",
      rep && rep.partial ? "Μερική κάλυψη· εμφανίζονται τα πρόσφατα περιστατικά." : "");
    const tasks = state.tasks;
    const taskRows = tasks ? tasks.items.map(x => row(
      (x.review_date < now ? "Εκπρόθεσμη · " : "Υποχρέωση · ") + name(x.patient_name),
      (state.hidden ? "Υποχρέωση ασθενούς" : x.description) + " · " + pretty(x.review_date),
      () => nav().obligations?.(x.pending_id), x.review_date < now)) : [];
    section("Υποχρεώσεις ασθενών", tasks, taskRows, () => nav().obligations?.(),
      "Δεν βρέθηκαν υποχρεώσεις με ημερομηνία στην προβολή.",
      tasks ? "Προβολή έως 100 πρόσφατων ανοικτών υποχρεώσεων." : "");
    const calls = state.calls;
    const callRows = calls && calls.count !== null && calls.count > 0
      ? [row("Μη σημειωμένες ως διεκπεραιωμένες · " + calls.count,
        "Στη διαθέσιμη προβολή της Γραμματείας", () => nav().calls?.())] : [];
    section("Κλήσεις γραμματείας", calls, callRows, () => nav().calls?.(),
      calls && calls.count === null ? "Ο αριθμός δεν είναι διαθέσιμος." : "Δεν βρέθηκαν μη σημειωμένες κλήσεις στην διαθέσιμη προβολή.",
      calls ? "Κάλυψη: έως 200 καταγεγραμμένες κλήσεις." : "");
  }
  function revoke() {
    state.authorized = false; ++state.epoch; ++state.load;
    state.surgeries = state.rf = state.reports = state.tasks = state.calls = null;
    state.updated = 0; state.day = null; draw();
  }
  async function request(path, epoch, load) {
    const response = await fetch(url(path), {credentials: "same-origin", cache: "no-store", headers: {Accept: "application/json"}});
    if (epoch !== state.epoch || load !== state.load || !state.authorized) throw Error("stale");
    if (response.status === 401 || response.status === 403) {
      window.dispatchEvent(new Event("clinical:unauthenticated")); throw Error("revoked");
    }
    if (!response.ok) throw Error("unavailable");
    const result = await response.json();
    if (epoch !== state.epoch || load !== state.load || !state.authorized) throw Error("stale");
    return result;
  }
  async function loadSource(key, path, parse, epoch, load) {
    try {
      const data = await request(path, epoch, load);
      const parsed = parse(data);
      if (epoch === state.epoch && load === state.load && state.authorized) state[key] = parsed;
    } catch (_) {
      if (epoch === state.epoch && load === state.load && state.authorized) state[key] = null;
    }
    if (epoch === state.epoch && load === state.load && state.authorized) draw();
  }
  function refresh(reloadSurgery = true) {
    if (!state.authorized) return;
    const epoch = state.epoch, load = ++state.load;
    state.day = today();
    state.rf = state.reports = state.tasks = state.calls = undefined;
    state.updated = Date.now(); draw();
    if (reloadSurgery !== false) window.CockpitSurgeryQueue?.refreshAttention?.();
    void loadSource("rf", "/clinical/office-cases/rf?query=&stage=approved&cursor=", d => {
      if (!d || !Array.isArray(d.applications) || typeof d.has_more !== "boolean" ||
          d.applications.some(x => !x || typeof x.id !== "string" || !x.id || x.stage !== "approved" || typeof x.patient_name !== "string" ||
            !(x.surgery_id === null || typeof x.surgery_id === "string"))) throw Error("invalid");
      return {items: d.applications.filter(x => x.surgery_id === null), partial: d.has_more};
    }, epoch, load);
    void loadSource("reports", "/clinical/office-cases/accidents?offset=0", d => {
      if (!d || !Array.isArray(d.cases) || typeof d.has_more !== "boolean" ||
          d.cases.some(x => !x || typeof x.patient_name !== "string" ||
            !(x.report_sent_date == null || validDate(x.report_sent_date)) ||
            !(x.writing_started_date == null || validDate(x.writing_started_date)))) throw Error("invalid");
      return {items: d.cases.flatMap(x => {
        if (validDate(x.writing_started_date) && !x.report_sent_date) return [{patient_name: x.patient_name, writing: true}];
        if (validDate(x.report_sent_date) && Number.isInteger(x.balance_cents) && x.balance_cents > 0)
          return [{patient_name: x.patient_name, writing: false}];
        return [];
      }), partial: d.has_more};
    }, epoch, load);
    void loadSource("tasks", "/clinical/clinical-updates?scope=real&active_only=true", d => {
      if (!d || !Array.isArray(d.items) || d.items.length > 100 || d.items.some(x => !x ||
          typeof x.patient_name !== "string" || typeof x.description !== "string" ||
          typeof x.pending_id !== "string" || typeof x.status !== "string")) throw Error("invalid");
      const n = new Date(today() + "T12:00:00Z"); n.setUTCDate(n.getUTCDate() + 7);
      const end = n.toISOString().slice(0, 10);
      return {items: d.items.filter(x => ["open", "awaiting_patient", "awaiting_result"].includes(x.status) &&
        validDate(x.review_date) && x.review_date <= end).sort((a,b) => a.review_date.localeCompare(b.review_date))};
    }, epoch, load);
    void loadSource("calls", "/clinical/reception-calls/peek", d => {
      if (!d || d.source !== "reception_dashboard" || d.is_unread_count !== false ||
          !["available_call_log_max_200", "ambiguous_call_identity"].includes(d.coverage) ||
          !(d.unhandled_count === null || (Number.isInteger(d.unhandled_count) && d.unhandled_count >= 0 && d.unhandled_count <= 200))) throw Error("invalid");
      return {count: d.unhandled_count};
    }, epoch, load);
  }
  function receiveSurgery(detail) {
    if (!state.authorized) return;
    if (!detail || detail.unavailable || !Array.isArray(detail.attention) ||
        !Number.isInteger(detail.undated) || !Number.isInteger(detail.invalid) ||
        detail.attention.some(x => !x || !validDate(x.date) || typeof x.name !== "string" ||
          typeof x.procedure !== "string" || typeof x.time !== "string")) state.surgeries = null;
    else state.surgeries = {attention: detail.attention, undated: detail.undated, invalid: detail.invalid};
    draw();
  }
  function authenticated() {
    revoke(); state.authorized = true; state.surgeries = undefined; draw(); refresh(false);
  }
  window.addEventListener("clinical:authenticated", authenticated);
  window.addEventListener("clinical:unauthenticated", revoke);
  window.addEventListener("cockpit:surgery-counts", event => receiveSurgery(event.detail));
  const maybeRefresh = () => {
    if (state.authorized && !document.hidden && !document.getElementById("doctorMainView")?.hidden &&
        (state.day !== today() || Date.now() - state.updated > 300000)) refresh();
  };
  document.addEventListener("visibilitychange", maybeRefresh);
  window.addEventListener("focus", maybeRefresh);
  if (typeof window.setInterval === "function") window.setInterval(maybeRefresh, 60000);
  window.CockpitHomeAttention = {draw, refresh};
  draw();
  const bootstrap = state.epoch;
  fetch("/clinical/status", {credentials: "same-origin", cache: "no-store"}).then(response => {
    if (bootstrap !== state.epoch) return;
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) window.dispatchEvent(new Event("clinical:unauthenticated"));
      else revoke();
      return;
    }
    state.authorized = true; state.surgeries = undefined;
    if (window.CockpitSurgerySummary) receiveSurgery(window.CockpitSurgerySummary);
    refresh(false);
  }).catch(() => { if (bootstrap === state.epoch) revoke(); });
})();
