(() => {
  "use strict";

  function makeNavLink({ id, href, title, iconText, labelText }) {
    const link = document.createElement("a");
    link.id = id;
    link.href = href;
    link.className = "side-item";
    link.title = title;

    const icon = document.createElement("span");
    icon.className = "side-icon";
    icon.textContent = iconText;

    const label = document.createElement("span");
    label.textContent = labelText;

    link.append(icon, label);
    return link;
  }

  function injectWorkspaceLinks() {
    const nav = document.querySelector(".side-nav");
    if (!nav) return;

    let calendar = document.querySelector("#clinicalCalendarNavLink");
    if (!calendar) {
      calendar = makeNavLink({
        id: "clinicalCalendarNavLink",
        href: "../clinical-calendar/",
        title: "Clinical Calendar",
        iconText: "▦",
        labelText: "Ημερολόγιο",
      });
      const cases = nav.querySelector("[data-nav-action='cases']");
      if (cases) cases.insertAdjacentElement("afterend", calendar);
      else nav.appendChild(calendar);
    }

    if (!document.querySelector("#physioReferralNavLink")) {
      const physio = makeNavLink({
        id: "physioReferralNavLink",
        href: "/clinical/clinic-utilities/physio-referral",
        title: "Physiotherapy Referral v2",
        iconText: "↗",
        labelText: "Παραπεμπτικό Φ/Θ",
      });
      calendar.insertAdjacentElement("afterend", physio);
    }
  }

  injectWorkspaceLinks();
  const observer = new MutationObserver(injectWorkspaceLinks);
  observer.observe(document.body, { childList: true, subtree: true });
})();
