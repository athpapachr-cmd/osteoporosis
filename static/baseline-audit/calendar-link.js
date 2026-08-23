(() => {
  "use strict";

  function injectCalendarLink() {
    const nav = document.querySelector(".side-nav");
    if (!nav || document.querySelector("#clinicalCalendarNavLink")) return;

    const link = document.createElement("a");
    link.id = "clinicalCalendarNavLink";
    link.href = "../clinical-calendar/";
    link.className = "side-item";
    link.title = "Clinical Calendar";

    const icon = document.createElement("span");
    icon.className = "side-icon";
    icon.textContent = "▦";

    const label = document.createElement("span");
    label.textContent = "Ημερολόγιο";

    link.append(icon, label);

    const cases = nav.querySelector("[data-nav-action='cases']");
    if (cases) cases.insertAdjacentElement("afterend", link);
    else nav.appendChild(link);
  }

  injectCalendarLink();
  const observer = new MutationObserver(injectCalendarLink);
  observer.observe(document.body, { childList: true, subtree: true });
})();
