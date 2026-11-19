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
  const shortDateLabel = new Intl.DateTimeFormat("el-CY", {
    timeZone: "UTC", day: "numeric", month: "short"
  });
  const state = {authorized: false, epoch: 0, load: 0, hidden: false,
    surgeries: null, rf: null, reports: null, tasks: null, calls: null, updated: 0, day: null,
    showAll: false, showAllSurgery: false, filter: "all", showOld: false};
  const focusKeys = new WeakMap();
  let focusTargets = new Map();
  function trackFocus(element, key) {
    if (key) { focusKeys.set(element, key); focusTargets.set(key, element); }
    return element;
  }
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
  const shortDate = value => shortDateLabel.format(new Date(value + "T12:00:00Z"));
  function tidy(value) {
    const s = String(value || "");
    return s.toLocaleLowerCase("el")
      .replace(/(^|[\s-])(\p{L})/gu, (_m,a,b) => a + b.toUpperCase());
  }
  const name = value => state.hidden ? "Απόκρυψη ονόματος" : tidy(value);
  const node = (tag, cls, value) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (value !== undefined) el.textContent = value;
    return el;
  };
  const dayValue = value => Date.parse(value + "T00:00:00Z") / 86400000;
  function nextDay(value) {
    const d = new Date(value + "T12:00:00Z"); d.setUTCDate(d.getUTCDate() + 1);
    return d.toISOString().slice(0, 10);
  }
  function sourceDate(value) { return validDate(value) ? value : null; }
  function dueLabel(item) {
    if (validDate(item.review_date)) return "Προθεσμία " + shortDate(item.review_date);
    return item.review_date ? "Μη έγκυρη ημερομηνία" : "Χωρίς ημερομηνία";
  }
  function actionButton(label, cls, handler, focusKey = null) {
    const button = node("button", cls, label); button.type = "button";
    button.addEventListener("click", handler); return trackFocus(button, focusKey);
  }
  function basicRow(tag, title, detail, meta, handler, opts = {}) {
    const b = node("button", "ha-row" + (opts.focus ? " ha-focus" : "") + (opts.footer ? " ha-counter" : "")); b.type = "button";
    if (opts.aria) b.setAttribute("aria-label", opts.aria);
    if (!opts.footer) b.append(node("span", "ha-tag" + (opts.focus ? " ha-tag-fill" : ""), tag));
    const body = node("span", "ha-txt");
    body.append(node("strong", "ha-line", title));
    if (detail) body.append(node("span", "ha-sub", detail));
    b.append(body, node("span", "ha-meta", meta || ""));
    b.addEventListener("click", handler); return trackFocus(b, opts.focusKey);
  }
  function taskRow(group, now, actionsAvailable) {
    const outer = node("div", "ha-row ha-task-row");
    outer.append(node("span", "ha-tag", "Υποχρέωση"));
    const body = node("span", "ha-txt");
    const overdue = group.items.filter(x => validDate(x.review_date) && x.review_date < now).length;
    const count = group.items.length;
    const status = count + (count === 1 ? " υποχρέωση" : " υποχρεώσεις") +
      (overdue ? " · " + overdue + (overdue === 1 ? " εκπρόθεσμη" : " εκπρόθεσμες") : "");
    const label = name(group.name) + " · " + status;
    if (actionsAvailable) {
      const main = actionButton(label, "ha-task-name", () => nav().obligations?.(
        group.items.length === 1 ? group.items[0].pending_id : undefined), "task-group:" + group.key);
      main.setAttribute("aria-label", "Υποχρέωση · " + label); body.append(main);
    } else body.append(node("strong", "ha-line", label));
    for (const item of group.items.slice(0, 2)) {
      const description = state.hidden ? "Υποχρέωση ασθενούς" : item.description;
      const line = actionsAvailable
        ? actionButton("", "ha-sub ha-task-detail", () => nav().obligations?.(item.pending_id), "task:" + group.key + ":" + item.pending_id)
        : node("span", "ha-sub");
      line.append(node("span", "ha-sub-desc", description), node("span", "ha-sub-date", " · " + dueLabel(item)));
      body.append(line);
    }
    if (group.items.length > 2) {
      const more = "+" + (group.items.length - 2) + " ακόμη";
      if (actionsAvailable) body.append(actionButton(more, "ha-more-tasks", () => nav().obligations?.(), "task-more:" + group.key));
      else body.append(node("span", "ha-sub", more));
    }
    outer.append(body);
    const meta = node("span", "ha-meta");
    if (overdue && group.date) {
      const dot = node("span", "ha-dot"); dot.setAttribute("aria-hidden", "true");
      const days = dayValue(now) - dayValue(group.date);
      meta.setAttribute("aria-label", "Εκπρόθεσμη κατά " + days + " ημέρες");
      meta.append(dot, node("span", "", days + " ημ."));
    } else if (group.date) meta.append(node("span", "", pretty(group.date)));
    outer.append(meta); return outer;
  }
  function groupTasks(items) {
    const groups = new Map();
    items.forEach((item, index) => {
      const id = typeof item.patient_id === "string" && item.patient_id.trim() ? item.patient_id : null;
      const key = id ? "patient:" + id : "unknown:" + index;
      if (!groups.has(key)) groups.set(key, {kind: "task", key, name: item.patient_name, items: [], date: null});
      groups.get(key).items.push(item);
    });
    for (const group of groups.values()) {
      group.items.sort((a,b) => (sourceDate(a.review_date) || "9999-99-99").localeCompare(sourceDate(b.review_date) || "9999-99-99"));
      group.date = sourceDate(group.items[0].review_date);
    }
    return [...groups.values()];
  }
  function oldItem(item, now) {
    if (item.kind === "task") return item.items.every(x => validDate(x.review_date) && dayValue(now) - dayValue(x.review_date) > 90);
    return !!item.date && dayValue(now) - dayValue(item.date) > 90;
  }
  function age(date, now, label) {
    if (!date) return "";
    const days=dayValue(now)-dayValue(date);
    return days < 0 ? label + " " + pretty(date) : label + " · " + days + " ημ.";
  }
  function draw() {
    const active = document.activeElement, previousFocus = active ? focusKeys.get(active) : null;
    const wasInside = Boolean(active && typeof root.contains === "function" && root.contains(active));
    focusTargets = new Map();
    const restoreFocus = () => {
      const target = (previousFocus && focusTargets.get(previousFocus)) || (wasInside && focusTargets.get("heading"));
      if (target && typeof target.focus === "function") target.focus({preventScroll:true});
    };
    root.replaceChildren(); root.className = "doctor-attention-list ha-root";
    const header = node("div", "ha-header");
    const heading = node("h2", "ha-title", "Σήμερα και αύριο"); heading.setAttribute("tabindex", "-1");
    header.append(trackFocus(heading, "heading"));
    if (state.authorized) {
      const toggle = actionButton(state.hidden ? "◉ Ονόματα κρυφά" : "◉ Ονόματα", "ha-toggle", () => { state.hidden = !state.hidden; draw(); }, "privacy");
      toggle.setAttribute("aria-label", state.hidden ? "Εμφάνιση ονομάτων εδώ" : "Απόκρυψη ονομάτων εδώ");
      toggle.setAttribute("aria-pressed", String(state.hidden)); header.append(toggle);
    }
    root.append(header);
    if (!state.authorized) { root.append(node("p", "ha-note", "Συνδέσου για να δεις τις εκκρεμότητες.")); restoreFocus(); return; }
    const now = today(), tomorrow = nextDay(now), go = tab => () => nav().cases?.(tab);
    const s = state.surgeries, focusSurgeries = s ? s.attention.filter(x => x.date === now || x.date === tomorrow) : [];
    focusSurgeries.sort((a,b) => a.date.localeCompare(b.date) || a.time.localeCompare(b.time));
    if (focusSurgeries.length) {
      const section = node("section", "ha-section");
      const shown = state.showAllSurgery ? focusSurgeries : focusSurgeries.slice(0, 3);
      let current = "";
      for (const item of shown) {
        if (item.date !== current) {
          current = item.date;
          section.append(node("h3", "ha-label", (current === now ? "Σήμερα · " : "Αύριο · ") + pretty(current)));
        }
        section.append(basicRow("Χειρουργείο", (item.time ? item.time + " · " : "") + name(item.name),
          state.hidden ? "" : item.procedure, item.time ? "Αίθουσα " + item.time : "", go("surgery"), {focus:true}));
      }
      if (focusSurgeries.length > 3) {
        const more = actionButton(state.showAllSurgery ? "Λιγότερα χειρουργεία" : "Δες όλα τα " + focusSurgeries.length + " χειρουργεία", "ha-ghost", () => { state.showAllSurgery = !state.showAllSurgery; draw(); }, "more-surgeries");
        more.setAttribute("aria-expanded", String(state.showAllSurgery)); section.append(more);
      }
      root.append(section);
    }
    const items = [];
    if (s) for (const x of s.attention) if (x.date !== now && x.date !== tomorrow)
      items.push({kind:"surgery", date:x.date, value:x});
    if (state.tasks) items.push(...groupTasks(state.tasks.items));
    if (state.reports) items.push(...state.reports.items.map(x => ({kind:"report", date:x.date, value:x})));
    if (state.rf) items.push(...state.rf.items.map(x => ({kind:"rf", date:sourceDate(x.decision_date), value:x})));
    items.sort((a,b) => (a.date || "9999-99-99").localeCompare(b.date || "9999-99-99"));
    const list = node("section", "ha-section");
    const selected = state.showAll ? items : items.slice(0, 4);
    list.append(node("h3", "ha-label", "Να γίνουν · " + (items.length > 4 ? selected.length + " από " : "") + items.length));
    if (!items.length) list.append(node("p", "ha-note", "Δεν βρέθηκαν στοιχεία στην τρέχουσα προβολή."));
    const kinds = [["all","Όλα"],["task","Υποχρεώσεις"],["report","Εκθέσεις"],["rf","RF"],["surgery","Χειρουργεία"]];
    if (state.showAll) {
      const filters=node("div","ha-filters");
      for (const [kind,label] of kinds) {
        const count=kind === "all" ? items.length : items.filter(x=>x.kind===kind).length;
        if (kind !== "all" && !count) continue;
        const b=actionButton(label + " " + count,"ha-filter",()=>{state.filter=kind;draw();},"filter:"+kind);
        b.setAttribute("aria-pressed",String(state.filter===kind));filters.append(b);
      }
      list.append(filters);
      if (state.tasks) list.append(node("p","ha-note","Υποχρεώσεις: " + state.tasks.items.length + " ενεργές εγγραφές στην προβολή έως 100 · οι αριθμοί φίλτρων είναι γραμμές."));
    }
    function addItem(item) {
      const x=item.value;
      if (item.kind === "task") list.append(taskRow(item, now, state.tasks.actionsAvailable));
      else if (item.kind === "rf") {
        const b=basicRow("RF",name(x.patient_name),"Χωρίς σύνδεση με χειρουργείο",age(item.date,now,"Έγκριση") || "Εγκεκριμένη",()=>nav().cases?.("rf",x.id),
          {aria:"Εγκεκριμένη RF · " + name(x.patient_name),focusKey:"rf:"+x.id});
        if(item.date && dayValue(now)-dayValue(item.date)>=30)b.className += " ha-aged";list.append(b);
      } else if (item.kind === "report") {
        const b=basicRow("Έκθεση",name(x.patient_name),x.writing?"Σε σύνταξη":"Σταλμένη με υπόλοιπο",age(item.date,now,x.writing?"Έναρξη":"Αποστολή"),go("accidents"));
        if(item.date && dayValue(now)-dayValue(item.date)>=30)b.className += " ha-aged";list.append(b);
      } else list.append(basicRow("Χειρουργείο",name(x.name),
        (state.hidden ? "" : x.procedure + (x.date < now ? " · " : "")) + (x.date < now ? "Περασμένη ημερομηνία" : ""),pretty(x.date),go("surgery")));
    }
    if (!state.showAll) selected.forEach(addItem);
    else {
      const filtered=items.filter(x=>state.filter==="all" || x.kind===state.filter);
      const recent=filtered.filter(x=>!oldItem(x,now)),old=filtered.filter(x=>oldItem(x,now));
      recent.forEach(addItem);
      if (old.length) {
        const oldToggle=actionButton("Παλιότερη ημερομηνία · πάνω από 90 ημέρες ("+old.length+")" + (state.showOld ? " · Απόκρυψη" : ""),"ha-ghost",()=>{state.showOld=!state.showOld;draw();},"old-items");
        oldToggle.setAttribute("aria-expanded",String(state.showOld));list.append(oldToggle);
        if(state.showOld)old.forEach(addItem);
      }
    }
    if (items.length > 4) {
      const more = actionButton(state.showAll ? "Δες λιγότερα" : "Δες όλα τα " + items.length, "ha-ghost", () => { state.showAll = !state.showAll; if(state.showAll)state.showOld=items.slice(0,4).some(x=>oldItem(x,now)); else {state.filter="all";state.showOld=false;} draw(); }, "more-items");
      more.setAttribute("aria-expanded", String(state.showAll)); list.append(more);
    }
    root.append(list);
    const footer = node("section", "ha-section ha-footer");
    if (s && s.undated) footer.append(basicRow("", s.undated + (s.undated === 1 ? " χειρουργείο" : " χειρουργεία") + " χωρίς ημερομηνία", "", "Άνοιγμα", go("surgery"), {footer:true,focusKey:"surgery-undated"}));
    if (s && s.invalid) footer.append(basicRow("", s.invalid + (s.invalid === 1 ? " χειρουργείο" : " χειρουργεία") + " με μη έγκυρη ημερομηνία", "", "Άνοιγμα", go("surgery"), {footer:true,focusKey:"surgery-invalid"}));
    if (s && s.later) footer.append(basicRow("", s.later + (s.later === 1 ? " χειρουργείο" : " χειρουργεία") + " μετά το επταήμερο", "", "Άνοιγμα", go("surgery"), {footer:true,focusKey:"surgery-later"}));
    if (state.calls && state.calls.count > 0) footer.append(basicRow("", state.calls.count + (state.calls.count === 1 ? " κλήση" : " κλήσεις") + " χωρίς σήμανση διεκπεραίωσης", "", "Άνοιγμα", () => nav().calls?.(), {footer:true,focusKey:"calls"}));
    const sources = [["Χειρουργεία", s], ["RF", state.rf], ["Εκθέσεις", state.reports], ["Υποχρεώσεις", state.tasks], ["Κλήσεις", state.calls]];
    const loading = sources.filter(([,value]) => value === undefined).map(([label]) => label);
    const missing = sources.filter(([,value]) => value === null).map(([label]) => label);
    if (state.calls && state.calls.count === null) missing.push("Αριθμός κλήσεων");
    if (loading.length) footer.append(node("p", "ha-note", "Φόρτωση: " + loading.join(", ") + "."));
    if (missing.length) footer.append(node("p", "ha-warning", "Μη διαθέσιμο: " + missing.join(", ") + ". Δεν σημαίνει ότι δεν υπάρχουν εκκρεμότητες."));
    if (state.tasks && !state.tasks.actionsAvailable) footer.append(node("p", "ha-warning", "Η ταυτότητα ορισμένων υποχρεώσεων δεν επιβεβαιώθηκε· το άνοιγμα λεπτομερειών δεν είναι διαθέσιμο."));
    const partial = [];
    if (state.rf?.partial) partial.push("RF");
    if (state.reports?.partial) partial.push("Εκθέσεις");
    if (partial.length) footer.append(node("p", "ha-warning", "Μερική κάλυψη: " + partial.join(", ") + ". Άνοιξε τις αντίστοιχες λίστες."));
    const coverage = [];
    if (state.tasks) coverage.push("έως 100 πρόσφατες υποχρεώσεις");
    if (state.calls) coverage.push("έως 200 καταγεγραμμένες κλήσεις");
    if (coverage.length) {
      const details=node("details","ha-coverage");details.append(node("summary","", "ⓘ Κάλυψη"),node("p","ha-note","Η προβολή περιλαμβάνει " + coverage.join(" και ") + "."));footer.append(details);
    }
    root.append(footer);
    restoreFocus();
  }
  function revoke() {
    state.authorized = false; ++state.epoch; ++state.load;
    state.surgeries = state.rf = state.reports = state.tasks = state.calls = null;
    state.updated = 0; state.day = null; state.showAll = state.showAllSurgery = state.showOld = false; state.filter = "all"; draw();
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
        if (validDate(x.writing_started_date) && !x.report_sent_date) return [{patient_name: x.patient_name, writing: true, date: x.writing_started_date}];
        if (validDate(x.report_sent_date) && Number.isInteger(x.balance_cents) && x.balance_cents > 0)
          return [{patient_name: x.patient_name, writing: false, date: x.report_sent_date}];
        return [];
      }), partial: d.has_more};
    }, epoch, load);
    void loadSource("tasks", "/clinical/clinical-updates?scope=real&active_only=true", d => {
      if (!d || !Array.isArray(d.items) || d.items.length > 100 || d.items.some(x => !x ||
          typeof x.patient_name !== "string" || typeof x.description !== "string" ||
          typeof x.pending_id !== "string" || typeof x.status !== "string")) throw Error("invalid");
      const active = d.items.filter(x => ["open", "awaiting_patient", "awaiting_result"].includes(x.status));
      return {items: active, actionsAvailable: active.every(x => typeof x.patient_id === "string" && !!x.patient_id.trim() && !!x.pending_id.trim())};
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
        !Number.isInteger(detail.total) || !Number.isInteger(detail.undated) || !Number.isInteger(detail.invalid) ||
        detail.undated < 0 || detail.invalid < 0 || detail.total < detail.undated + detail.invalid + detail.attention.length ||
        detail.attention.some(x => !x || !validDate(x.date) || typeof x.name !== "string" ||
          typeof x.procedure !== "string" || typeof x.time !== "string")) state.surgeries = null;
    else state.surgeries = {attention: detail.attention.map(x => ({...x, time:/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(x.time) ? x.time : ""})),
      undated: detail.undated, invalid: detail.invalid,
      later: detail.total - detail.undated - detail.invalid - detail.attention.length};
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
