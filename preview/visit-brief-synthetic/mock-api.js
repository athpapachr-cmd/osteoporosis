/* SYNTHETIC Cockpit Visit Brief demo. Every /clinical/* request is mocked in memory.
   No production backend, Aiven, provider, cookies, or patient records are touched. */
(() => {
  "use strict";
  const SYN = Object.freeze({ demo: true, noPHI: true });
  window.CockpitSyntheticDemo = SYN;
  window.CockpitStaffRole = "clinician";
  const uid = n => "cal.com:synthetic-visit-" + n;
  const appointments = [
    {appointment_id:uid(1),patient_display_name:"Δοκιμαστική Ασθενής Α",start_at:"2026-10-17T06:00:00Z",end_at:"2026-10-17T06:30:00Z",clinic:"Λεμεσός",reason:"Επανέλεγχος γόνατος",category:"other"},
    {appointment_id:uid(2),patient_display_name:"Δοκιμαστικός Ασθενής Β",start_at:"2026-10-17T08:00:00Z",end_at:"2026-10-17T08:30:00Z",clinic:"Λεμεσός",reason:"Παρακολούθηση οστεοπόρωσης",category:"osteoporosis_review"},
    {appointment_id:uid(3),patient_display_name:"Δοκιμαστική Ασθενής Γ",start_at:"2026-10-17T10:00:00Z",end_at:"2026-10-17T10:30:00Z",clinic:"Λεμεσός",reason:"Έλεγχος μαγνητικής",category:"other"},
  ];
  const people = [
    {patient_id:"SYN-A",demographics:{full_name:"Δοκιμαστική Ασθενής Α",date_of_birth:"1975-01-01"}},
    {patient_id:"SYN-B",demographics:{full_name:"Δοκιμαστικός Ασθενής Β",date_of_birth:"1952-01-01"}},
    {patient_id:"SYN-C",demographics:{full_name:"Δοκιμαστική Ασθενής Γ",date_of_birth:"1986-01-01"}},
  ];
  const details = [
    {visit:"Επανεξέταση άλγους γόνατος",finding:"Βελτίωση βάδισης και εύρους κίνησης",decision:"Συνέχιση σταδιακής ενδυνάμωσης",drug:null,pending:"CD μαγνητικής προς προσκόμιση",review:"Επισκόπηση εικόνων CD, εφόσον παραδόθηκαν"},
    {visit:"Επανέλεγχος οστεοπόρωσης",finding:"Επανεκτίμηση ιστορικού και παρακολούθησης",decision:"Έλεγχος αναλύσεων πριν από την επόμενη απόφαση",drug:{name:"Συνθετική αγωγή (demo)",action:"continued"},pending:"Εκκρεμεί εργαστηριακός έλεγχος",review:"Επιβεβαίωση παραλαβής εργαστηριακών αποτελεσμάτων"},
    {visit:"Προηγούμενη ορθοπαιδική αξιολόγηση",finding:"Αναφέρθηκε έλεγχος με MRI",decision:"Επανεκτίμηση μετά τις απεικονιστικές εξετάσεις",drug:null,pending:"Γνωμάτευση ακτινολόγου",review:"Έλεγχος γνωμάτευσης και απεικονιστικού υλικού"},
  ];
  const revisions = new Map();
  let loggedIn = true;
  const observation = "a".repeat(64);
  const json = (data, status = 200) => new Response(JSON.stringify(data), {status,headers:{"Content-Type":"application/json","Cache-Control":"no-store"}});
  function booking(n) {
    const a = appointments[n];
    return {booking_uid:a.appointment_id,patient_display_name:a.patient_display_name,reason:a.reason,start_at:a.start_at,clinic:a.clinic,observation};
  }
  function match(uidValue) { return appointments.findIndex(a=>a.appointment_id===uidValue); }
  const pick = (i) => {
    const a = appointments[i], p=people[i], d=details[i];
    return {
      patient_id:p.patient_id,check_heading:"Για το δοκιμαστικό ραντεβού",
      previous_visit:{date:"2026-10-10",reason:d.visit,findings:[d.finding],source_encounter_id:"SYN-ENCOUNTER"},
      prior_plan:{decisions:[{text:d.decision,certainty:"certain",verification:"clinician_confirmed",source_roles:["clinician_edit"]}],
        medications:d.drug?[{...d.drug,certainty:"certain",verification:"source_stated",source_roles:["clinician_edit"]}]:[],
        next_contact:{when:"2026-10-17",purpose:a.reason}},
      historical_plan:null,
      pending:[{description:d.pending,status:"awaiting_result",review_date:"2026-10-17",truth:"last_recorded_at_read"}],
      appointment_checks:[{text:d.review,kind:"pending_status_check"}],partial:false
    };
  };
  const canonicalName = s => String(s||"").normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase();
  // Do not let any clinical fetch accidentally reach a real service.
  const originalFetch = window.fetch.bind(window);
  window.fetch = async (resource, init = {}) => {
    const url = new URL(typeof resource === "string" ? resource : resource.url, location.href);
    const path = url.pathname;
    if (!path.startsWith("/clinical/")) {
      if (url.origin !== location.origin) return json({detail:"Demo: external requests disabled"},403);
      return originalFetch(resource,init);
    }
    const body = (()=>{try{return JSON.parse(init.body||"{}")}catch{return {}}})();
    if (path === "/clinical/login") {
      if (body.key !== "demo") return json({detail:"Δοκιμαστικό κλειδί: demo"},401);
      loggedIn=true; window.CockpitStaffRole="clinician";
      return json({authenticated:true,session:"synthetic",staff_role:"clinician"});
    }
    if (path === "/clinical/logout") { loggedIn=false;window.CockpitStaffRole=null;window.dispatchEvent(new Event("clinical:unauthenticated"));return json({authenticated:false}); }
    if (path === "/clinical/staff-session")return loggedIn?json({staff_role:"clinician",principal_id:"synthetic-doctor"}):json({detail:"Unauthorized"},401);
    if (path === "/clinical/status")return loggedIn?json({authenticated:true}):json({detail:"Unauthorized"},401);
    if(!loggedIn)return json({detail:"Unauthorized"},401);
    if (path === "/clinical/calendar/cockpit-context")return json({
      generated_at:"2026-10-17T08:00:00Z",source_updated_at:"2026-10-17T07:59:00Z",
      today_total:3,previous:appointments[0],current:appointments[1],next:appointments[2],
      current_conflict_count:0,upcoming_today:appointments
    });
    if (path === "/clinical/calendar/appointments")return json(appointments);
    if (path === "/clinical/recent-encounters")return json(people.map((p,i)=>({patient_id:p.patient_id,patient_display_name:p.demographics.full_name,encounter_date:"2026-10-10",visit_type:details[i].visit})));
    if (path === "/clinical/calendar/visit-brief/read") {
      const n=match(body.booking_uid); if(n<0)return json({detail:"Booking unavailable"},409);
      const rev=revisions.get(body.booking_uid)||0;
      return json({mode:"booking_only",reason:rev?"needs_reconfirmation":"unlinked",booking:booking(n),revision:rev,linked:rev>0});
    }
    if (path === "/clinical/calendar/visit-brief/patient-search") {
      const n=match(body.booking_uid);
      if(n<0 || body.observed_booking!==observation)return json({detail:"Changed booking"},409);
      const query=canonicalName(body.query);
      // Only the matching synthetic booking attendee appears in this preview.
      const p=people[n];
      return json(canonicalName(p.demographics.full_name).includes(query)||query.includes("δοκιμασ")?[p]:[]);
    }
    if (path === "/clinical/patients") {
      const query=canonicalName(url.searchParams.get("query"));
      return json(people.filter(p=>canonicalName(p.demographics.full_name).includes(query)));
    }
    if (path === "/clinical/calendar/visit-brief/confirm") {
      const n=match(body.booking_uid);if(n<0)return json({detail:"Booking unavailable"},409);
      const prior=revisions.get(body.booking_uid)||0;
      if(body.observed_booking!==observation || !body.independently_verified
         || body.patient_id!==people[n].patient_id || body.expected_revision!==prior)
        return json({detail:"Independent identity/revision check failed"},409);
      revisions.set(body.booking_uid,prior+1);
      return json({mode:"confirmed_once",revision:prior+1,booking:booking(n),brief:pick(n)});
    }
    if (path === "/clinical/calendar/visit-brief/unlink") {
      const n=match(body.booking_uid),prior=revisions.get(body.booking_uid)||0;
      if(n<0||prior!==body.expected_revision)return json({detail:"Revision conflict"},409);
      revisions.delete(body.booking_uid);
      return json({mode:"booking_only",reason:"unlinked",revision:prior+1});
    }
    if (path === "/clinical/surgeries")return json([]);
    if (path === "/clinical/clinic-utilities/rf/api/outcomes/search")return json({
      scope:"all",page:1,truncated:false,denominator:0,applications:[],
      counts:{issued_unknown:0,pending:0,approved:0,rejected:0,scheduled:0,performed:0,mine_actionable:0}
    });
    return json({detail:"Αυτή η λειτουργία δεν διατίθεται στην απομονωμένη προεπισκόπηση."},404);
  };
  // No navigation into production links or the real secretary/dashboard.
  document.addEventListener("click", event => {
    const link=event.target.closest("a[href]");
    if(!link)return;
    const href=link.getAttribute("href")||"";
    if(href==="/" || href.startsWith("/static/cockpit/") && (href==="/static/cockpit/"||href==="/static/cockpit/index.html"))return;
    event.preventDefault();
    alert("Συνθετική προεπισκόπηση: οι υπόλοιπες λειτουργίες δεν συνδέονται με το πραγματικό Cockpit.");
  },true);
})();
