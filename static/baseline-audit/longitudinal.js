(() => {
  "use strict";

  const STORAGE_KEY = "osteoporosis.baselineAuditPilot.v1_1";
  const ACTIVE_KEY = "osteoporosis.baselineAuditPilot.activeCase.v1_1";
  const $ = (s, r=document) => r.querySelector(s);
  const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));
  const num = (v) => v === "" || v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v);
  const fmt = (v, d=1) => v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d);

  const MACHINE_OPTIONS = [
    ["", "—"],
    ["hologic_horizon", "Hologic Horizon"],
    ["hologic_discovery", "Hologic Discovery"],
    ["ge_lunar_idxa", "GE Lunar iDXA"],
    ["ge_lunar_prodigy", "GE Lunar Prodigy"],
    ["norland", "Norland"],
    ["other_unknown", "Other / unknown"]
  ];
  const RISK_OPTIONS = [
    ["", "—"], ["low","Low"], ["intermediate","Intermediate"], ["high","High"], ["very_high","Very high"], ["uncertain","Uncertain"], ["not_applicable","N/A"]
  ];
  const PLUS_ADJUSTMENTS = [
    ["fracture_recency","Recency of fracture"],
    ["oral_glucocorticoid_exposure","Higher oral glucocorticoid exposure"],
    ["trabecular_bone_score","Trabecular bone score (TBS)"],
    ["falls_previous_year","Falls in previous year"],
    ["type2_diabetes_duration","Duration of type 2 diabetes"],
    ["lumbar_spine_bmd","Lumbar spine BMD"],
    ["hip_axis_length","Hip axis length"],
    ["primary_hyperparathyroidism","Primary hyperparathyroidism"],
    ["number_of_prior_fractures","Number of prior fractures"],
    ["other","Other"]
  ];

  const optionHtml = (items) => items.map(([v,l]) => `<option value="${v}">${l}</option>`).join("");

  function getCases(){ try{return JSON.parse(localStorage.getItem(STORAGE_KEY)||"[]");}catch{return [];} }
  function setCases(cases){ localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
  function activeUuid(){ return localStorage.getItem(ACTIVE_KEY)||""; }
  function activeCase(){ const id=activeUuid(); return getCases().find(c=>c.internal_uuid===id)||null; }

  function stateDefaults(){
    return {
      risk_categories:{mof:"",hip:"",overall:""},
      fraxplus:{used:"",adjusted_mof:null,adjusted_hip:null,dominant_adjustment:"",adjustments:[],note:""},
      frax_history:[],
      dxa_history:[],
      updated_at:null
    };
  }
  function normalize(raw){
    const b=stateDefaults();
    if(!raw||typeof raw!=="object") return b;
    return {...b,...raw,risk_categories:{...b.risk_categories,...(raw.risk_categories||{})},fraxplus:{...b.fraxplus,...(raw.fraxplus||{})},frax_history:Array.isArray(raw.frax_history)?raw.frax_history:[],dxa_history:Array.isArray(raw.dxa_history)?raw.dxa_history:[]};
  }
  let state=stateDefaults();
  let loaded="";

  function persist(){
    if(!loaded) loaded=activeUuid();
    if(!loaded) return;
    collectCurrentEnhancements();
    state.updated_at=new Date().toISOString();
    const cases=getCases(); const i=cases.findIndex(c=>c.internal_uuid===loaded); if(i<0) return;
    cases[i]={...cases[i], longitudinal_review:state}; setCases(cases);
  }

  function replaceMachineInput(){
    const old=$("#s3DxaMachine");
    if(!old || old.tagName === "SELECT") return;
    const select=document.createElement("select"); select.id="s3DxaMachine"; select.innerHTML=optionHtml(MACHINE_OPTIONS);
    select.value=old.value || "";
    old.replaceWith(select);
    const parent=select.closest("label");
    if(parent && !$("#s3MachineLocalLabel")){ const extra=document.createElement("input"); extra.id="s3MachineLocalLabel"; extra.type="text"; extra.maxLength=80; extra.placeholder="Local machine label / ID (optional)"; parent.appendChild(extra); }
  }

  function injectStep2(){
    const grid=$('[data-step-panel="2"] .step2-grid'); if(!grid || $("#longitudinalRiskCard")) return;
    const card=document.createElement("article"); card.className="card step2-card longitudinal-card"; card.id="longitudinalRiskCard";
    card.innerHTML=`
      <div class="card-heading"><div><h2>FRAX / FRAXplus & longitudinal risk</h2><p>MOF και hip παραμένουν ξεχωριστές πιθανότητες και ξεχωριστές κατηγορίες. Τα FRAXplus values καταγράφονται ως externally calculated outputs.</p></div></div>
      <div class="longitudinal-grid three">
        <label><span>MOF risk category</span><select id="lrMofCategory">${optionHtml(RISK_OPTIONS)}</select></label>
        <label><span>Hip risk category</span><select id="lrHipCategory">${optionHtml(RISK_OPTIONS)}</select></label>
        <label><span>Overall management category</span><select id="lrOverallCategory">${optionHtml(RISK_OPTIONS)}</select></label>
      </div>
      <div class="fraxplus-box">
        <div class="longitudinal-grid three">
          <label><span>FRAXplus χρησιμοποιήθηκε;</span><select id="lrPlusUsed"><option value="">—</option><option value="yes">Ναι</option><option value="no">Όχι</option><option value="not_applicable">N/A</option></select></label>
          <label><span>Adjusted MOF %</span><input id="lrPlusMof" type="number" min="0" max="100" step="0.1" /></label>
          <label><span>Adjusted hip %</span><input id="lrPlusHip" type="number" min="0" max="100" step="0.1" /></label>
        </div>
        <label><span>Dominant adjustment</span><select id="lrDominantAdjustment"><option value="">—</option>${optionHtml(PLUS_ADJUSTMENTS)}</select></label>
        <div class="chip-checks compact" id="lrPlusAdjustmentChecks">${PLUS_ADJUSTMENTS.map(([v,l])=>`<label><input type="checkbox" value="${v}" />${l}</label>`).join("")}</div>
        <label class="full-field"><span>FRAXplus note <small>(optional)</small></span><input id="lrPlusNote" maxlength="300" /></label>
        <div class="mini-muted">Δεν γίνεται local calculation ή stacking FRAXplus modifiers. Καταγράφεται το αποτέλεσμα που επέστρεψε το FRAXplus.</div>
      </div>
      <div class="longitudinal-actions"><div><strong>FRAX history</strong><div class="mini-muted">Για σύγκριση διαδοχικών formal risk assessments.</div></div><button type="button" class="btn secondary" id="lrAddFrax">＋ FRAX snapshot</button></div>
      <div id="lrFraxTable"></div>
      <div class="longitudinal-summary" id="lrFraxSummary"></div>
      <div class="longitudinal-grid"><div class="longitudinal-chart"><h4>MOF trend — raw vs adjusted</h4><div id="lrMofChart"></div></div><div class="longitudinal-chart"><h4>Hip trend — raw vs adjusted</h4><div id="lrHipChart"></div></div></div>
      <div class="longitudinal-actions"><div><strong>DXA longitudinal overview</strong><div class="mini-muted">Η καταχώρηση ιστορικών DXA γίνεται στο Step 3. Εδώ εμφανίζεται read-only overview.</div></div><button type="button" class="btn secondary" id="lrGoStep3">Άνοιγμα Step 3</button></div>
      <div id="lrDxaOverview"></div>`;
    grid.appendChild(card);

    const overall=$("#resultingRiskCategory");
    if(overall){ const label=overall.closest("label")?.querySelector("span"); if(label) label.textContent="Overall management risk category"; }
  }

  function injectStep3(){
    const grid=$('[data-step-panel="3"] .step3-grid'); if(!grid || $("#longitudinalDxaCard")) return;
    const card=document.createElement("article"); card.className="card step3-card longitudinal-card"; card.id="longitudinalDxaCard";
    card.innerHTML=`
      <div class="card-heading"><div><h2>DXA longitudinal history</h2><p>Ιστορικές μετρήσεις για πίνακα, BMD/T-score charts και descriptive change analysis.</p></div><button type="button" class="btn secondary" id="lrAddDxa">＋ Prior DXA</button></div>
      <div id="lrDxaTable"></div>
      <div class="longitudinal-summary" id="lrDxaSummary"></div>
      <div class="longitudinal-grid"><div class="longitudinal-chart"><h4>BMD trend by site</h4><div id="lrBmdChart"></div></div><div class="longitudinal-chart"><h4>T-score trend by site</h4><div id="lrTChart"></div></div></div>
      <div class="analysis-caution">Η % μεταβολή BMD είναι περιγραφική. Δεν χαρακτηρίζεται ως σημαντική/μη σημαντική χωρίς κατάλληλη συγκρισιμότητα μηχανήματος και facility LSC.</div>`;
    grid.appendChild(card);
  }

  function currentFraxPoint(){
    const c=activeCase(); const date=c?.encounter_date||"";
    return {date, raw_mof:num($("#fraxMof")?.value ?? c?.step2?.formal_risk?.mof), raw_hip:num($("#fraxHip")?.value ?? c?.step2?.formal_risk?.hip), adjusted_mof:num($("#lrPlusMof")?.value), adjusted_hip:num($("#lrPlusHip")?.value), current:true};
  }
  function currentDxaPoint(){
    const c=activeCase();
    return {date:$("#s3DxaDate")?.value||c?.step3?.dxa?.date||"", machine:$("#s3DxaMachine")?.value||c?.step3?.dxa?.machine||"", machine_label:$("#s3MachineLocalLabel")?.value||"", spine_bmd:num($("#s3SpineBmd")?.value), spine_t:num($("#s3SpineT")?.value), total_hip_bmd:num($("#s3TotalHipBmd")?.value), total_hip_t:num($("#s3TotalHipT")?.value), fn_bmd:num($("#s3FnBmd")?.value), fn_t:num($("#s3FnT")?.value), current:true};
  }

  function table(rows, columns, removeKind){
    if(!rows.length) return '<div class="no-data-note">Δεν υπάρχουν ιστορικά δεδομένα.</div>';
    return `<div class="longitudinal-table-wrap"><table class="longitudinal-table"><thead><tr>${columns.map(c=>`<th>${c[1]}</th>`).join("")}<th></th></tr></thead><tbody>${rows.map((r,i)=>`<tr>${columns.map(c=>`<td>${c[2]?c[2](r[c[0]],r):r[c[0]]||"—"}</td>`).join("")}<td class="history-row-actions">${r.current?'<span class="mini-muted">Current</span>':`<button type="button" class="btn secondary" data-remove-${removeKind}="${i}">×</button>`}</td></tr>`).join("")}</tbody></table></div>`;
  }

  function svgLineChart(containerId, rows, series){
    const host=$(containerId); if(!host) return;
    const validRows=rows.filter(r=>r.date && series.some(s=>num(r[s.key])!==null)).sort((a,b)=>a.date.localeCompare(b.date));
    if(validRows.length<2){host.innerHTML='<div class="no-data-note">Χρειάζονται τουλάχιστον 2 χρονολογικά σημεία.</div>';return;}
    const values=[]; validRows.forEach(r=>series.forEach(s=>{const v=num(r[s.key]); if(v!==null) values.push(v);}));
    let min=Math.min(...values), max=Math.max(...values); if(min===max){min-=1;max+=1;} const pad=(max-min)*0.12; min-=pad; max+=pad;
    const W=520,H=170,L=42,R=10,T=12,B=28, pw=W-L-R, ph=H-T-B;
    const x=i=>L+(validRows.length===1?pw/2:(i*pw/(validRows.length-1))); const y=v=>T+(max-v)*ph/(max-min);
    const palette=["#2563eb","#0f766e","#7c3aed","#d97706"];
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img"><line x1="${L}" y1="${T}" x2="${L}" y2="${H-B}" stroke="#cbd5e1"/><line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="#cbd5e1"/>`;
    for(let g=0;g<4;g++){const val=min+(max-min)*g/3, yy=y(val); svg+=`<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#eef2f7"/><text x="${L-5}" y="${yy+4}" text-anchor="end" font-size="10" fill="#64748b">${val.toFixed(1)}</text>`;}
    series.forEach((s,si)=>{const pts=[];validRows.forEach((r,i)=>{const v=num(r[s.key]);if(v!==null)pts.push([x(i),y(v),v]);}); if(pts.length>1) svg+=`<polyline fill="none" stroke="${palette[si%palette.length]}" stroke-width="2.2" points="${pts.map(p=>`${p[0]},${p[1]}`).join(" ")}"/>`; pts.forEach(p=>svg+=`<circle cx="${p[0]}" cy="${p[1]}" r="3.2" fill="${palette[si%palette.length]}"><title>${s.label}: ${p[2]}</title></circle>`);});
    validRows.forEach((r,i)=>{const label=r.date.length>=7?r.date.slice(0,7):r.date;svg+=`<text x="${x(i)}" y="${H-8}" text-anchor="middle" font-size="9" fill="#64748b">${label}</text>`;}); svg+='</svg>';
    const legend=`<div class="trend-legend">${series.map((s,i)=>`<span style="color:${palette[i%palette.length]}"><i class="trend-dot"></i>${s.label}</span>`).join("")}</div>`;
    host.innerHTML=legend+svg;
  }

  function allFraxRows(){const rows=[...state.frax_history];const cur=currentFraxPoint();if(cur.date && [cur.raw_mof,cur.raw_hip,cur.adjusted_mof,cur.adjusted_hip].some(v=>v!==null)) rows.push(cur);return rows.sort((a,b)=>(a.date||"").localeCompare(b.date||""));}
  function allDxaRows(){const rows=[...state.dxa_history];const cur=currentDxaPoint();if(cur.date && [cur.spine_bmd,cur.total_hip_bmd,cur.fn_bmd,cur.spine_t,cur.total_hip_t,cur.fn_t].some(v=>v!==null)) rows.push(cur);return rows.sort((a,b)=>(a.date||"").localeCompare(b.date||""));}

  function renderFrax(){
    const rows=allFraxRows(); const historical=state.frax_history;
    const host=$("#lrFraxTable"); if(host) host.innerHTML=table(rows,[["date","Date"],["raw_mof","MOF %",v=>fmt(v)],["raw_hip","Hip %",v=>fmt(v)],["adjusted_mof","Adj MOF %",v=>fmt(v)],["adjusted_hip","Adj Hip %",v=>fmt(v)]],"frax");
    const s=$("#lrFraxSummary"); if(s){const e=rows[0],l=rows[rows.length-1];s.innerHTML=rows.length>=2?`<div><span>MOF change</span><strong>${fmt((l.raw_mof??0)-(e.raw_mof??0))} pp</strong></div><div><span>Hip change</span><strong>${fmt((l.raw_hip??0)-(e.raw_hip??0))} pp</strong></div><div><span>Assessments</span><strong>${rows.length}</strong></div>`:'<div><span>Trend</span><strong>Χρειάζεται δεύτερο σημείο</strong></div>';}
    svgLineChart("#lrMofChart",rows,[{key:"raw_mof",label:"Raw MOF"},{key:"adjusted_mof",label:"FRAXplus MOF"}]);
    svgLineChart("#lrHipChart",rows,[{key:"raw_hip",label:"Raw hip"},{key:"adjusted_hip",label:"FRAXplus hip"}]);
    $$("[data-remove-frax]").forEach(btn=>btn.onclick=()=>{state.frax_history.splice(Number(btn.dataset.removeFrax),1);persist();renderAll();});
  }

  function pctChange(a,b){ if(a===null||b===null||a===0) return null; return (b-a)/a*100; }
  function renderDxa(){
    const rows=allDxaRows();
    const cols=[["date","Date"],["machine","Machine",v=>MACHINE_OPTIONS.find(x=>x[0]===v)?.[1]||v||"—"],["spine_bmd","Spine BMD",v=>fmt(v,3)],["spine_t","Spine T",v=>fmt(v)],["total_hip_bmd","Hip BMD",v=>fmt(v,3)],["total_hip_t","Hip T",v=>fmt(v)],["fn_bmd","FN BMD",v=>fmt(v,3)],["fn_t","FN T",v=>fmt(v)]];
    const host=$("#lrDxaTable"); if(host) host.innerHTML=table(rows,cols,"dxa");
    const overview=$("#lrDxaOverview"); if(overview) overview.innerHTML=table(rows,cols.slice(0,6),"noop");
    const s=$("#lrDxaSummary"); if(s){if(rows.length>=2){const e=rows[0],l=rows[rows.length-1];s.innerHTML=`<div><span>Spine BMD Δ</span><strong>${fmt(pctChange(e.spine_bmd,l.spine_bmd))}%</strong></div><div><span>Total hip BMD Δ</span><strong>${fmt(pctChange(e.total_hip_bmd,l.total_hip_bmd))}%</strong></div><div><span>FN BMD Δ</span><strong>${fmt(pctChange(e.fn_bmd,l.fn_bmd))}%</strong></div><div><span>Scans</span><strong>${rows.length}</strong></div>`;}else{s.innerHTML='<div><span>Trend</span><strong>Χρειάζεται δεύτερο DXA</strong></div>';}}
    svgLineChart("#lrBmdChart",rows,[{key:"spine_bmd",label:"Spine BMD"},{key:"total_hip_bmd",label:"Total hip BMD"},{key:"fn_bmd",label:"FN BMD"}]);
    svgLineChart("#lrTChart",rows,[{key:"spine_t",label:"Spine T"},{key:"total_hip_t",label:"Total hip T"},{key:"fn_t",label:"FN T"}]);
    $$("[data-remove-dxa]").forEach(btn=>btn.onclick=()=>{state.dxa_history.splice(Number(btn.dataset.removeDxa),1);persist();renderAll();});
  }

  function collectCurrentEnhancements(){
    if($("#lrMofCategory")) state.risk_categories={mof:$("#lrMofCategory").value,hip:$("#lrHipCategory").value,overall:$("#lrOverallCategory").value};
    if($("#lrPlusUsed")) state.fraxplus={used:$("#lrPlusUsed").value,adjusted_mof:num($("#lrPlusMof").value),adjusted_hip:num($("#lrPlusHip").value),dominant_adjustment:$("#lrDominantAdjustment").value,adjustments:$$('#lrPlusAdjustmentChecks input:checked').map(x=>x.value),note:$("#lrPlusNote").value.trim()};
  }

  function hydrate(){
    if($("#lrMofCategory")){ $("#lrMofCategory").value=state.risk_categories.mof||"";$("#lrHipCategory").value=state.risk_categories.hip||"";$("#lrOverallCategory").value=state.risk_categories.overall||$("#resultingRiskCategory")?.value||"";$("#lrPlusUsed").value=state.fraxplus.used||"";$("#lrPlusMof").value=state.fraxplus.adjusted_mof??"";$("#lrPlusHip").value=state.fraxplus.adjusted_hip??"";$("#lrDominantAdjustment").value=state.fraxplus.dominant_adjustment||"";$("#lrPlusNote").value=state.fraxplus.note||"";$$('#lrPlusAdjustmentChecks input').forEach(x=>x.checked=state.fraxplus.adjustments.includes(x.value)); }
  }

  function addFrax(){
    collectCurrentEnhancements(); const c=activeCase();
    state.frax_history.push({date:c?.encounter_date||new Date().toISOString().slice(0,10),raw_mof:num($("#fraxMof")?.value),raw_hip:num($("#fraxHip")?.value),adjusted_mof:num($("#lrPlusMof")?.value),adjusted_hip:num($("#lrPlusHip")?.value),country_model:$("#fraxCountryModel")?.value||"",framework:$("#declaredRiskFramework")?.value||""});
    persist();renderAll();
  }
  function addDxa(){
    const row={date:"",machine:"",machine_label:"",spine_bmd:null,spine_t:null,total_hip_bmd:null,total_hip_t:null,fn_bmd:null,fn_t:null};
    const date=prompt("Ημερομηνία prior DXA (YYYY-MM-DD):",""); if(!date) return; row.date=date;
    const spineBmd=prompt("Spine BMD g/cm² (optional):",""); row.spine_bmd=num(spineBmd);
    const spineT=prompt("Spine T-score (optional):",""); row.spine_t=num(spineT);
    const hipBmd=prompt("Total hip BMD g/cm² (optional):",""); row.total_hip_bmd=num(hipBmd);
    const hipT=prompt("Total hip T-score (optional):",""); row.total_hip_t=num(hipT);
    const fnBmd=prompt("Femoral neck BMD g/cm² (optional):",""); row.fn_bmd=num(fnBmd);
    const fnT=prompt("Femoral neck T-score (optional):",""); row.fn_t=num(fnT);
    state.dxa_history.push(row);persist();renderAll();
  }

  function renderAll(){hydrate();renderFrax();renderDxa();}
  function load(){
    replaceMachineInput();injectStep2();injectStep3();
    const c=activeCase();loaded=c?.internal_uuid||"";state=normalize(c?.longitudinal_review);renderAll();
  }
  function bind(){
    document.addEventListener("input",e=>{if(e.target.closest("#longitudinalRiskCard,#longitudinalDxaCard") || e.target.matches("#s3DxaMachine,#s3MachineLocalLabel,#s3DxaDate,#s3SpineBmd,#s3SpineT,#s3TotalHipBmd,#s3TotalHipT,#s3FnBmd,#s3FnT,#fraxMof,#fraxHip")){persist();renderFrax();renderDxa();}});
    document.addEventListener("change",e=>{if(e.target.closest("#longitudinalRiskCard,#longitudinalDxaCard") || e.target.matches("#resultingRiskCategory,#declaredRiskFramework,#fraxCountryModel")){persist();renderFrax();renderDxa();}});
    document.addEventListener("click",e=>{
      if(e.target.closest("#lrAddFrax")) addFrax();
      if(e.target.closest("#lrAddDxa")) addDxa();
      if(e.target.closest("#lrGoStep3")){const b=$('.step-tab[data-step="3"]');if(b)b.click();}
      if(e.target.closest("[data-load-case]")||e.target.closest('[data-nav-action="new-case"]')) setTimeout(load,20);
    });
    $$(".step-tab").forEach(b=>b.addEventListener("click",()=>{if(["2","3"].includes(b.dataset.step))setTimeout(()=>{replaceMachineInput();injectStep2();injectStep3();load();},0);}));
  }

  if(!document.querySelector('link[data-longitudinal-style]')){const l=document.createElement("link");l.rel="stylesheet";l.href="./longitudinal.css";l.dataset.longitudinalStyle="true";document.head.appendChild(l);}
  bind(); setTimeout(load,0);
})();