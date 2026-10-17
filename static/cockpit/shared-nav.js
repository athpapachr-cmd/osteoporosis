(() => {
  "use strict";
  // Protected /clinical/login already sets the same-origin HttpOnly cookie.
  // A floating key entry is a convenience, not a new login authority.
  const ROOT = "/static/cockpit/";
  const shortcuts = [
    ["Εβδομαδιαίο ημερολόγιο", "/static/clinical-calendar/"],
    ["Καταγραφή επίσκεψης", "/static/cockpit/visit-capture/"],
    ["RF · Ραδιοσυχνότητες", "/clinical/clinic-utilities/rf"],
    ["Παραπεμπτικό φυσιοθεραπείας", "/clinical/clinic-utilities/physio-referral"],
    ["Αναρρωτική άδεια", "/clinical/clinic-utilities/sick-leave"],
    ["Ιατρική έκθεση", "/clinical/clinic-utilities/medical-report"],
    ["Οστεοπόρωση", "/static/baseline-audit/"],
    ["Learning Hub", "/static/clinical-learning/"],
  ];

  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  };

  function boot() {
    if (document.getElementById("ceGlobalNav")) return;
    const nav = element("nav", "ce-global-nav");
    nav.id = "ceGlobalNav";
    nav.setAttribute("aria-label", "Κοινή πλοήγηση Cockpit");
    const actions = element("div", "ce-global-actions");
    const home = element("a", "ce-global-button");
    home.href = ROOT;
    home.title = "Επιστροφή στο Cockpit Home";
    home.append(element("span", "ce-global-glyph", "⌂"), document.createTextNode("Αρχική"));

    const actionsButton = element("button", "ce-global-button");
    actionsButton.type = "button";
    actionsButton.id = "ceGlobalShortcuts";
    actionsButton.setAttribute("aria-expanded", "false");
    actionsButton.append(element("span", "ce-global-glyph", "⌘"),
      document.createTextNode("Συντομεύσεις"));

    const keyButton = element("button", "ce-global-button");
    keyButton.id = "ceGlobalKey";
    keyButton.type = "button";
    keyButton.title = "Προστατευμένη πρόσβαση — σύνδεση ή επανασύνδεση";
    keyButton.setAttribute("aria-expanded", "false");
    keyButton.append(element("span", "ce-global-glyph", "🔑"),
      document.createTextNode("Πρόσβαση"));

    const shortcutPanel = element("section", "ce-global-popover");
    shortcutPanel.id = "ceGlobalShortcutsPanel";
    shortcutPanel.hidden = true;
    shortcutPanel.setAttribute("aria-label", "Συντομεύσεις");
    shortcutPanel.append(element("h2", "", "Άνοιγμα λειτουργίας"));
    shortcuts.forEach(([label, url]) => {
      const a = element("a", "ce-global-shortcut");
      a.href = url;
      a.append(document.createTextNode(label), element("span", "", "›"));
      shortcutPanel.append(a);
    });

    // Stale bootstrap/probe results cannot overwrite a new staff principal.
    let authRevision = 0;
    const accessPanel = element("section", "ce-global-popover");
    accessPanel.id = "ceGlobalAccessPanel";
    accessPanel.hidden = true;
    accessPanel.setAttribute("aria-label", "Clinical Key");
    accessPanel.append(element("h2", "", "Πρόσβαση στο Cockpit"));
    const explanation = element("p", "",
      "Μία σύνδεση για τις προστατευμένες οθόνες. Αν λείπει το πρόγραμμα της Γραμματείας, το κλειδί δεν διορθώνει απαραίτητα την πηγή.");
    accessPanel.append(explanation);
    const form = element("form");
    form.id = "ceGlobalLoginForm";
    const label = element("label", "", "Κλειδί πρόσβασης");
    const input = element("input");
    input.id = "ceGlobalKeyInput";
    input.type = "password";
    input.autocomplete = "current-password";
    input.required = true;
    label.append(input);
    const submit = element("button", "ce-global-submit", "Σύνδεση");
    submit.type = "submit";
    const status = element("p", "ce-global-status");
    status.id = "ceGlobalAuthStatus";
    status.setAttribute("role", "status");
    form.append(label, submit, status);
    accessPanel.append(form);
    actions.append(home, actionsButton, keyButton, shortcutPanel, accessPanel);
    nav.append(actions);
    document.body.insertBefore(nav, document.body.firstChild);

    const close = () => {
      shortcutPanel.hidden = true;
      accessPanel.hidden = true;
      actionsButton.setAttribute("aria-expanded", "false");
      keyButton.setAttribute("aria-expanded", "false");
      input.value = "";
    };
    const openPanel = panel => {
      const wasOpen = !panel.hidden;
      close();
      panel.hidden = wasOpen;
      if (panel.hidden) return;
      (panel === accessPanel ? keyButton : actionsButton)
        .setAttribute("aria-expanded", "true");
      if (panel === accessPanel) {
        status.textContent = "Έλεγχος σύνδεσης…";
        status.dataset.kind = "";
        // Auth probe is read-only; no provider, patient or schedule API.
        const probeRevision=authRevision;
        fetch("/clinical/status", {credentials:"same-origin", cache:"no-store"})
          .then(r => {
            if (panel.hidden||probeRevision!==authRevision) return;
            status.textContent = r.ok ? "Συνδεδεμένο. Μπορείς να αλλάξεις κλειδί." :
              r.status === 401 || r.status === 403 ? "Απαιτείται σύνδεση." :
              "Δεν ήταν δυνατός ο έλεγχος. Μπορείς να δοκιμάσεις σύνδεση.";
            if (!r.ok && (r.status === 401 || r.status === 403)) {
              fetch("/clinical/staff-session", {credentials:"same-origin", cache:"no-store"})
                .then(async staff => {
                  if (staff.ok&&!panel.hidden&&probeRevision===authRevision) {
                    const identity=await staff.json();
                    if(panel.hidden||probeRevision!==authRevision)return;
                    window.CockpitStaffRole=identity.staff_role;
                    status.textContent="Συνδεδεμένο προσωπικό: "+
                      (identity.staff_role==="secretary"?"γραμματεία":"γιατρός")+".";
                  }
                }).catch(()=>{});
            }
          })
          .catch(() => {
            if (!panel.hidden&&probeRevision===authRevision)
              status.textContent = "Δεν ήταν δυνατός ο έλεγχος σύνδεσης.";
          });
        input.focus();
      }
    };

    actionsButton.addEventListener("click", () => openPanel(shortcutPanel));
    keyButton.addEventListener("click", () => openPanel(accessPanel));
    document.addEventListener("pointerdown", e => {
      if (!nav.contains(e.target)) close();
    });
    document.addEventListener("keydown", e => {
      if (e.key === "Escape") close();
    });
    form.addEventListener("submit", async e => {
      e.preventDefault();
      const key = input.value;
      input.value = "";
      if (!key || submit.disabled) return;
      const loginRevision=++authRevision;
      submit.disabled = true;
      status.dataset.kind = "";
      status.textContent = "Σύνδεση…";
      try {
        const response = await fetch("/clinical/login", {
          method: "POST",
          credentials: "same-origin",
          cache: "no-store",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({key}),
        });
        if(loginRevision!==authRevision)return;
        if (!response.ok) {
          status.dataset.kind = "error";
          status.textContent = response.status === 401 ? "Λανθασμένο Clinical Key." :
            response.status === 503 ? "Η κλινική σύνδεση δεν είναι ρυθμισμένη." :
            "Δεν ολοκληρώθηκε η σύνδεση. Δοκίμασε ξανά.";
          return;
        }
        const result=await response.json();
        if(loginRevision!==authRevision)return;
        window.CockpitStaffRole=result.staff_role||null;
        status.dataset.kind = "success";
        status.textContent = result.staff_role==="secretary"?
          "Συνδέθηκες ως γραμματεία. Πρόσβαση μόνο σε επιτρεπόμενες προβολές.":
          "Συνδέθηκες. Οι κλινικές προβολές ανανεώνονται.";
        window.dispatchEvent(new Event("clinical:authenticated"));
      } catch (_) {
        if(loginRevision!==authRevision)return;
        status.dataset.kind = "error";
        status.textContent = "Αποτυχία σύνδεσης. Έλεγξε το δίκτυο και δοκίμασε ξανά.";
      } finally {
        submit.disabled = false;
      }
    });
    // A protected consumer can request the common key panel, without embedding
    // a second password form or keeping any secret in local/session storage.
    window.addEventListener("clinical:request-login", () => openPanel(accessPanel));
    window.addEventListener("clinical:unauthenticated",()=>{
      authRevision++;
      window.CockpitStaffRole=null;
    });
    // Role is only a presentation hint. Server validates every protected request.
    const bootstrapRevision=authRevision;
    fetch("/clinical/staff-session",{credentials:"same-origin",cache:"no-store"})
      .then(r=>r.ok?r.json():null)
      .then(v=>{
        if(bootstrapRevision===authRevision)window.CockpitStaffRole=v?.staff_role||null;
      })
      .catch(()=>{
        if(bootstrapRevision===authRevision)window.CockpitStaffRole=null;
      });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
