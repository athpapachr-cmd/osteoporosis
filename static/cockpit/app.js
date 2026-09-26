(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  function localDayBounds() {
    const now = new Date();
    const start = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0);
    const end = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1, 0, 0, 0, 0);
    return { now, start, end };
  }

  function formatToday(now) {
    return new Intl.DateTimeFormat("el-GR", {
      weekday: "long",
      day: "numeric",
      month: "long"
    }).format(now);
  }

  async function loadClinicalCalendarSummary() {
    const { now, start, end } = localDayBounds();
    $("todayHeading").textContent = formatToday(now);
    try {
      const url = `/clinical/calendar/appointments?start=${encodeURIComponent(start.toISOString())}&end=${encodeURIComponent(end.toISOString())}`;
      const response = await fetch(url, { credentials: "same-origin" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const rows = await response.json();
      const consultations = rows.filter((row) =>
        ["osteoporosis_first", "osteoporosis_review", "osteoporosis_unspecified"].includes(row.category)
      ).length;
      const treatments = rows.filter((row) => ["prolia", "aclasta"].includes(row.category)).length;
      $("todayOsteoporosisCount").textContent = String(consultations);
      $("todayTreatmentCount").textContent = String(treatments);
      $("calendarState").textContent = rows.length ? `${rows.length} σήμερα` : "0 σήμερα";
      $("calendarState").classList.add("ready");
      $("calendarNote").textContent = rows.length
        ? "Σύνοψη χωρίς στοιχεία ταυτότητας ασθενών. Άνοιξε το ημερολόγιο για τις λεπτομέρειες."
        : "Δεν υπάρχουν εισαγμένα κλινικά ραντεβού για σήμερα. Το Cal.com reason feed θα συνδεθεί στο επόμενο integration slice.";
    } catch (_) {
      $("todayOsteoporosisCount").textContent = "—";
      $("todayTreatmentCount").textContent = "—";
      $("calendarState").textContent = "Μη διαθέσιμο";
      $("calendarNote").textContent = "Άνοιξε το Clinical Calendar για σύνδεση ή έλεγχο της τρέχουσας τροφοδοσίας.";
    }
  }

  loadClinicalCalendarSummary();
})();
