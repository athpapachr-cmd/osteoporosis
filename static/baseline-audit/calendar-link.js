(() => {
  "use strict";

  function injectCalendarLink() {
    const head = document.querySelector(".clinical-registry-head");
    if (!head || document.querySelector("#clinicalCalendarLink")) return;
    const link = document.createElement("a");
    link.id = "clinicalCalendarLink";
    link.href = "../clinical-calendar/";
    link.className = "btn secondary";
    link.textContent = "Clinical Calendar";
    const status = document.querySelector("#clinicalStatus");
    if (status) head.insertBefore(link, status);
    else head.appendChild(link);
  }

  injectCalendarLink();
  const observer = new MutationObserver(injectCalendarLink);
  observer.observe(document.body, { childList: true, subtree: true });
})();
