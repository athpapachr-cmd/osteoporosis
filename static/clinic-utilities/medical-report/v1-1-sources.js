(() => {
  "use strict";

  const EXTRA_TYPES = [
    ["prescription", "Συνταγή"],
    ["imaging_referral", "Παραπεμπτικό για απεικονιστική εξέταση"],
    ["specialist_referral", "Παραπεμπτικό προς ειδικό / άλλο ιατρό"],
    ["lab_or_service_referral", "Παραπεμπτικό εργαστηριακών / άλλης υπηρεσίας"],
    ["heidi_transcript", "Heidi transcript"],
  ];
  const $ = (id) => document.getElementById(id);
  let timer = null;
  let startedAt = 0;
  let baseText = "";

  function currentTypes() {
    return Array.from(document.querySelectorAll("#fileList .source-type-select")).map((node) => node.value);
  }

  function removeFile(index) {
    const input = $("sourceFiles");
    if (!input || typeof DataTransfer === "undefined") return;
    const files = Array.from(input.files || []);
    const types = currentTypes().filter((_, i) => i !== index);
    const transfer = new DataTransfer();
    files.forEach((file, i) => { if (i !== index) transfer.items.add(file); });
    input.files = transfer.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
    setTimeout(() => {
      enhanceRows();
      Array.from(document.querySelectorAll("#fileList .source-type-select")).forEach((select, i) => {
        if (!types[i]) return;
        select.value = types[i];
        select.dispatchEvent(new Event("change", { bubbles: true }));
      });
    }, 0);
  }

  function enhanceRows() {
    Array.from(document.querySelectorAll("#fileList li")).forEach((row, index) => {
      const select = row.querySelector(".source-type-select");
      if (select) {
        EXTRA_TYPES.forEach(([value, label]) => {
          if (select.querySelector(`option[value="${value}"]`)) return;
          const option = document.createElement("option");
          option.value = value;
          option.textContent = label;
          select.append(option);
        });
      }
      if (row.querySelector(".v11-remove-file")) return;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "text-button v11-remove-file";
      button.textContent = "Αφαίρεση";
      button.setAttribute("aria-label", "Αφαίρεση αρχείου");
      button.addEventListener("click", () => removeFile(index));
      row.append(button);
    });
  }

  function stopTimer() {
    clearInterval(timer);
    timer = null;
  }

  function startTimer(box) {
    stopTimer();
    startedAt = Date.now();
    baseText = box.textContent.replace(/ · \d\d:\d\d.*$/, "");
    timer = setInterval(() => {
      if (box.hidden) { stopTimer(); return; }
      const total = Math.floor((Date.now() - startedAt) / 1000);
      const mm = String(Math.floor(total / 60)).padStart(2, "0");
      const ss = String(total % 60).padStart(2, "0");
      box.textContent = `${baseText} · ${mm}:${ss} · συνεχίζει να εκτελείται — μην το υποβάλετε ξανά`;
    }, 1000);
  }

  function monitorWorking() {
    const box = $("workingBox");
    if (!box) return;
    new MutationObserver(() => {
      if (box.hidden) stopTimer(); else startTimer(box);
    }).observe(box, { attributes: true, attributeFilter: ["hidden"] });
  }

  function injectStyle() {
    if ($("v11SourceStyle")) return;
    const style = document.createElement("style");
    style.id = "v11SourceStyle";
    style.textContent = `
      #fileList li{gap:10px;align-items:center;flex-wrap:wrap}.v11-remove-file{margin-left:auto}
      .v11-vision-badge{margin-top:8px}.v11-chat-log{max-height:250px;overflow:auto;display:grid;gap:8px;margin:12px 0}
      .v11-chat-message{padding:10px 12px;border:1px solid #d8e1ea;border-radius:10px;background:#f8fafc}
      .v11-chat-message.clinician{background:#eef6ff}.v11-chat-message p{margin:5px 0 0}
      .v11-resolved-warning{opacity:.72}.v11-resolutions{margin-top:16px;padding-top:12px;border-top:1px solid #dde5ed}
    `;
    document.head.append(style);
  }

  const list = $("fileList");
  if (list) new MutationObserver(enhanceRows).observe(list, { childList: true, subtree: true });
  injectStyle();
  enhanceRows();
  monitorWorking();

  window.MedicalReportV11Sources = Object.freeze({ enhanceRows });
})();
