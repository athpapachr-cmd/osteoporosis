/* Isolated synthetic preview: does not call backend, localStorage, external APIs or auth. */
(() => {
  'use strict';
  const Core = window.StrategyCore;
  const fixtures = window.StrategyFixtures;
  const chosen = window.__STRATEGY_CASE__ || new URLSearchParams(location.search).get('case') || 'J';
  const fixture = fixtures[chosen] || fixtures.J;
  let session = Core.createSession(fixture);
  let mode = 'rest';
  let whyOpen = false;
  let provenanceOpen = false;
  const root = document.getElementById('root');
  const enc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const btn = (action, label, type='button', css='button') => `<button type="${type}" class="${css}" data-action="${action}">${label}</button>`;
  const timelineLabel = () => {
    const context=Core.getTreatmentContext(session.patient,session.discussion);
    const actual=context.lastActualDose || session.discussion.confirmed_last_dose;
    if(context.actualDoseConflict) return 'Οι καταγραφές χορήγησης είναι αντικρουόμενες.';
    if(actual) return 'Τελευταία πραγματική χορήγηση: '+ actual.split('-').reverse().join('/');
    return 'Τελευταία πραγματική χορήγηση: δεν έχει επιβεβαιωθεί.';
  };
  const contextLabel = () => Core.contextLine(session.patient,session.discussion);
  const typeLabel = key => ({bone_building:'Σχηματισμός οστού',bone_loss_reduction:'Περιορισμός οστικής απώλειας',nonpharm:'Άσκηση και πρόληψη πτώσεων',continuity_review:'Συνέχεια μετά την Prolia',current_treatment_review:'Σημερινή αγωγή'})[key]||key;

  function riskHeader(){
    const p=session.patient;
    const summary=p.risk?.label==='Πολύ υψηλός'?'Πολύ υψηλός καταγματικός κίνδυνος':'Εκτίμηση καταγματικού κινδύνου';
    return `<section class="overview"><div class="breadcrumb">Κλινική εικόνα / Κίνδυνος　·　Θεραπευτική στρατηγική</div>
    <div class="kicker">Σήμερα μας απασχολεί</div><div class="risk-chip">${enc(summary)}</div>
    ${p.fractures?.length?`<p class="supporting">${enc(p.fractures[0].relative)}</p>`:''}</section>`;
  }
  function whyPane(){
    if(!whyOpen)return '';
    return `<section class="soft-panel"><h3>Γιατί βλέπουμε αυτό το ερώτημα;</h3>
    <p class="note">Η σημερινή εστίαση βασίζεται στην καταγεγραμμένη εικόνα κινδύνου, στα πρόσφατα συμβάντα και στο πραγματικά γνωστό ιστορικό θεραπείας.</p>
    <div class="lightrow"><span>Εκτίμηση κινδύνου</span><strong>${enc(session.patient.risk?.label||'Μη διαθέσιμη')}</strong></div>
    <div class="lightrow"><span>Θεραπευτικό ιστορικό</span><strong>${enc(Core.getTreatmentContext(session.patient,session.discussion).status.replaceAll('_',' '))}</strong></div>
    <p class="fineprint">Ανασκοπημένη βάση: NOGG 2024 §6. Η αριθμητική ερμηνεία FRAX/NOGG δεν επανυπολογίζεται εδώ.</p>
    <button class="quiet-link" data-action="provenance">${provenanceOpen?'Απόκρυψη':'Πηγή και όρια'}</button>
    ${provenanceOpen?`<p class="note">Η ένδειξη προέρχεται από συνθετικό fixture. Δεν τεκμηριώνεται εδώ αυτόματη εφαρμογή των βρετανικών αριθμητικών ορίων NOGG στην Κύπρο. Καμία επίδραση σε πραγματικά ιατρικά δεδομένα.</p>`:''}
    </section>`;
  }
  function sessionHeader(){
    return `${riskHeader()}<section class="surface" data-fixture="${enc(session.patient.id)}">
      <div class="micro">Θεραπευτική στρατηγική</div><h1>${enc(Core.currentQuestion(session.patient,session.discussion))}</h1>
      <p class="context">${enc(contextLabel())}</p>${mainContent()}
      ${mode!=='rest'?'<div class="actionbar"><button class="quiet-button" data-action="close-detail">Κλείσιμο λεπτομερειών</button></div>':''}
      <div class="panel" style="margin-top:26px">
        <button class="quiet-button" data-action="why">${whyOpen?'−':'+'} Γιατί αυτό το ερώτημα;</button>${whyPane()}
      </div>
    </section>`;
  }
  function fallbackAction(){return `<button class="quiet-button" data-action="direct-decision">Έχω ήδη καταλήξει σε απόφαση</button>`;}
  function rest(){
    const context=Core.getTreatmentContext(session.patient,session.discussion);
    let content='';
    if(context.priorDenosumab){
      content=`<div class="signal"><strong>Προηγούμενη Prolia</strong>Η προηγούμενη έκθεση έχει σημασία. Δεν υποθέτουμε ότι η θεραπεία συνεχίζεται ή έχει διακοπεί.</div>
      ${btn('begin','Δες τη θεραπευτική πορεία →')}`;
    } else if(session.patient.new_fracture_during_treatment){
      content=`<p class="note">Το νέο κάταγμα χρειάζεται επανεκτίμηση. Δεν σημαίνει από μόνο του αποτυχία της σημερινής αγωγής.</p>${btn('begin','Τι χρειάζεται να ελέγξουμε; →')}`;
    } else if(context.status==='not_recorded' || context.status==='known_incomplete'){
      content=`${btn('begin','Να δούμε τι ταιριάζει →')}`;
    } else {
      content=`${btn('begin','Να εξετάσουμε τις επιλογές →')}`;
    }
    return `<div class="actionbar">${content}</div><div class="actionbar">${fallbackAction()}</div>`;
  }
  function verifyHistory(){
    return `<div class="panel"><div class="micro">Μία πληροφορία που έχει σημασία</div>
    <h2>Έχει προηγηθεί θεραπεία οστεοπόρωσης;</h2>
    <p class="note">Δεν βρήκα καταγεγραμμένη αγωγή. Αυτό δεν σημαίνει ότι δεν έχει λάβει ποτέ θεραπεία.</p>
    <div class="choices">
      <button class="choice" data-history="confirmed_none">Επιβεβαιωμένα, όχι</button>
      <button class="choice" data-history="previous_therapy">Ναι, έχει λάβει</button>
    </div>
    <button class="quiet-link" data-history="unknown">Δεν είναι γνωστό ακόμη</button>
    </div>`;
  }
  function uncertainHistory(){
    const val=session.discussion.history_confirmation;
    return `<div class="panel"><h2>${val==='previous_therapy'?'Υπάρχει προηγούμενη θεραπεία':'Το προηγούμενο ιστορικό παραμένει αβέβαιο'}</h2>
    <p class="note">${val==='previous_therapy'?'Χρειάζεται να καταγραφεί ποια αγωγή έχει δοθεί και πότε, πριν παρουσιαστεί εξατομικευμένη καταλληλότητα.':'Μπορείς να συνεχίσεις την κλινική συζήτηση, αλλά δεν θα δηλώσουμε ότι ο ασθενής είναι treatment-naïve.'}</p>
    <div class="actionbar">${btn('direct-decision','Καταγραφή σημερινής κατάληξης')}
    ${btn('verify-again','Διευκρίνιση ιστορικού','button','button secondary')}</div></div>`;
  }
  function journey(){
    const c=Core.getTreatmentContext(session.patient,session.discussion);
    const known=c.lastActualDose||session.discussion.confirmed_last_dose;
    const resolved=c.intent!=='not_established';
    return `<div class="panel"><div class="micro">Από την πραγματική θεραπευτική πορεία</div>
      <h2>Τι γνωρίζουμε για την Prolia;</h2>
      <div class="lightrow"><div><strong>Προηγούμενη χορήγηση</strong><div class="secondary-copy">Επιβεβαιωμένη έκθεση, όχι απόδειξη συνέχισης σήμερα</div></div><span>✓</span></div>
      <div class="lightrow"><div><strong>Τελευταία πραγματική χορήγηση</strong><div class="secondary-copy">${enc(timelineLabel())}</div></div></div>
      ${!known&&!c.actualDoseConflict?`<div class="soft-panel"><label class="micro" for="doseDate">Αν γνωρίζεις την πραγματική ημερομηνία</label><input class="field narrow" id="doseDate" type="date" aria-label="Ημερομηνία πραγματικής χορήγησης">
      <div class="actionbar">${btn('save-dose','Χρησιμοποίησε αυτή την ημερομηνία','button','button secondary smallbtn')}<button class="quiet-link" data-action="dose-unknown">Δεν έχει επιβεβαιωθεί</button></div></div>`:''}
      ${c.actualDoseConflict?'<div class="signal"><strong>Αντικρουόμενες καταγραφές</strong>Χρειάζεται διευκρίνιση από την πραγματική πηγή. Δεν επιλέγεται αυθαίρετα μία ημερομηνία.</div>':''}
      <div class="step"><h3>${resolved?'Καταγεγραμμένη σημερινή πορεία':'Τι ακολούθησε την Prolia;'}</h3>
      ${resolved?`<p>${enc(({ongoing:'Η θεραπεία συνεχίζεται.',transition_considered:'Εξετάζεται αλλαγή ή διακοπή.',subsequent_treatment:'Έχει καταγραφεί μεταγενέστερη θεραπεία.'})[c.intent]||'Δεν έχει επιβεβαιωθεί.')}</p>`:''}</div>
      ${!resolved?`<div class="choices">
        <button class="choice" data-course="ongoing">Συνεχίζεται</button>
        <button class="choice" data-course="transition_considered">Εξετάζεται αλλαγή</button>
        <button class="choice" data-course="subsequent_treatment">Ακολούθησε άλλη αγωγή</button>
      </div><button class="quiet-link" data-course="not_established">Δεν έχει διευκρινιστεί</button>`:''}
      <div class="actionbar">${btn('open-continuity','Τι σημαίνει για την απόφαση; →')}${btn('direct-decision','Καταγραφή κατάληξης','button','button secondary')}</div>
    </div>`;
  }
  function strategyContext(){
    const p=session.patient;
    if(p.new_fracture_during_treatment) return `<div class="signal"><strong>Κάταγμα παρά τη θεραπεία</strong>Χρειάζεται να κατανοήσουμε τη λήψη, τη διάρκεια και το πλαίσιο. Δεν χαρακτηρίζουμε αυτόματα τη θεραπεία αποτυχημένη.</div>`;
    if(p.completed_anabolic) return `<div class="signal"><strong>Μετά από θεραπεία αναδόμησης</strong>Στη σημερινή συζήτηση έχει σημασία πώς θα διατηρηθεί το όφελος.</div>`;
    if(p.access?.status==='unknown') return `<div class="signal"><strong>Πρόσβαση σε θεραπεία</strong>Δεν έχει επιβεβαιωθεί ακόμη η οδός διάθεσης. Η κλινική σύσταση παραμένει ξεχωριστή.</div>`;
    if(p.renal?.status==='unknown') return `<div class="signal"><strong>Νεφρική πληροφορία</strong>Χρειάζεται διευκρίνιση αν επηρεάζει τη θεραπεία που συζητείται.</div>`;
    if(p.patient_declined) return `<div class="signal"><strong>Προτίμηση ασθενούς</strong>Η μη αποδοχή φαρμάκου δεν σημαίνει απουσία θεραπευτικής ένδειξης.</div>`;
    return '';
  }
  function chooseApproaches(){
    const p=session.patient,c=Core.getTreatmentContext(p,session.discussion);
    if(['not_recorded','known_incomplete','conflicting'].includes(c.status))return uncertainHistory();
    const options=p.lower_risk?['nonpharm'] : (c.priorDenosumab && c.intent==='subsequent_treatment' && p.current_treatment)?['current_treatment_review','nonpharm']:c.priorDenosumab?['continuity_review','nonpharm']:['bone_building','bone_loss_reduction','nonpharm'];
    const current=options.includes(session.discussion.selected_focus)?session.discussion.selected_focus:options[0];
    const a=Core.approachInfo(current,p);
    return `<div class="panel"><div class="micro">Συζήτηση — όχι αυτόματη επιλογή</div>
    <h2>${c.priorDenosumab?'Τι χρειάζεται να προσέξουμε στη συνέχεια;':'Ποιες προσεγγίσεις αξίζει να συζητήσουμε;'}</h2>
    ${strategyContext()}
    ${c.priorDenosumab?`<p class="note">${enc(timelineLabel())} ${c.intent==='not_established'?'Δεν έχει επιβεβαιωθεί αν η αγωγή συνεχίζεται ή έχει αλλάξει.':''}</p>`:''}
    <div class="approach-switch" role="group" aria-label="Προσέγγιση προς διερεύνηση">
    ${options.map(k=>`<button class="switchbtn ${k===current?'active':''}" data-focus="${k}">${enc(typeLabel(k))}</button>`).join('')}
    </div>
    <div class="approach-card active"><h3 class="approach-head">${enc(a.name)}</h3>
      <div class="step"><h3>Γιατί να το εξετάσουμε;</h3><p>${enc(a.why)}</p></div>
      <div class="step"><h3>Για τον συγκεκριμένο ασθενή</h3><p>${enc(a.matters)}</p></div>
      <div class="step"><h3>Τι σημαίνει για τη συνέχεια;</h3><p>${enc(a.later)}</p></div>
      <details class="step"><summary class="quiet-link">Γιατί; · Επιστημονική βάση</summary><p class="note">${enc(a.source)}. Η καταλληλότητα δεν έχει υπολογιστεί από το Cockpit.</p></details>
      <div class="actionbar">${btn('consider-'+current,session.discussion.considered.includes(current)?'✓ Κρατήθηκε στη συζήτηση':'Κράτησέ το στη συζήτηση','button','button secondary')}</div>
    </div>
    <p class="fineprint">Οι προσεγγίσεις είναι αφετηρία συζήτησης, όχι ισοδύναμες συστάσεις ούτε κατάλογος φαρμάκων.</p>
    <div class="actionbar">${btn('direct-decision','Πού καταλήξαμε σήμερα; →')}</div>
    </div>`;
  }
  function decisionForm(){
    const d=session.discussion;
    return `<div class="panel"><div class="micro">Καταγραφή συζήτησης — μόνο για αυτό το demo</div><h2>Πού καταλήξαμε σήμερα;</h2>
    <p class="note">Η σύσταση, η θέση του ασθενούς και η τελική κατάληξη είναι διαφορετικά πράγματα. Τίποτε εδώ δεν αποτελεί συνταγή ή χορήγηση.</p>
    <form id="decisionForm" class="decision">
    <label>Τι συστήνεις ως γιατρός;<textarea maxlength="800" name="recommendation" placeholder="Σύντομη σύσταση ή τι χρειάζεται πρώτα να αποσαφηνιστεί">${enc(d.recommendation)}</textarea></label>
    <div class="twocol"><label>Θέση ασθενούς<select name="patient_preference">
    ${[['not_discussed','Δεν έχει συζητηθεί'],['accepts','Συμφωνεί'],['declines','Δεν επιθυμεί'],['alternative','Προτιμά κάτι άλλο'],['undecided','Δεν έχει αποφασίσει']].map(([v,l])=>`<option value="${v}" ${v===d.patient_preference?'selected':''}>${l}</option>`).join('')}</select></label>
    <label>Σημερινή κατάληξη<select name="decision">
    ${[['pending','Παραμένει ανοιχτή'],['agreed','Συμφωνημένη επιλογή'],['deferred','Αναβολή απόφασης'],['declined','Άρνηση θεραπείας']].map(([v,l])=>`<option value="${v}" ${v===d.decision?'selected':''}>${l}</option>`).join('')}</select></label></div>
    <label>Γιατί; <span class="muted">(προαιρετικό)</span><textarea maxlength="800" name="rationale" placeholder="Συνδέεται με όσα γνωρίζουμε, χωρίς να επινοούνται δεδομένα">${enc(d.rationale)}</textarea></label>
    <div class="actionbar"><button type="submit" class="button">Δες την καταγεγραμμένη κατάληξη →</button></div>
    </form></div>`;
  }
  function handoff(){
    const h=Core.makeHandoff(session);
    if(!h)return decisionForm();
    const dt={agreed:'Συμφωνημένη επιλογή',deferred:'Αναβολή',declined:'Άρνηση',pending:'Ανοιχτή απόφαση'};
    const pref={not_discussed:'Δεν συζητήθηκε',accepts:'Συμφωνεί',declines:'Δεν επιθυμεί',alternative:'Άλλη προτίμηση',undecided:'Δεν έχει αποφασίσει'};
    return `<div class="panel"><div class="micro">Μόνο προσωρινή συνθετική καταγραφή</div><h2>Η σημερινή κατάληξη</h2>
    <div class="summary"><dl><dt>Σύσταση</dt><dd>${enc(h.recommendation||'Δεν καταγράφηκε σύσταση')}</dd><dt>Θέση ασθενούς</dt><dd>${enc(pref[h.patient_preference])}</dd><dt>Απόφαση</dt><dd>${enc(dt[h.decision])}</dd>
    ${h.rationale?`<dt>Αιτιολόγηση</dt><dd>${enc(h.rationale)}</dd>`:''}</dl></div>
    ${h.unresolved.length?`<p class="note">Ανοιχτά ζητήματα: ${h.unresolved.map(x=>({'unverified_treatment_history':'μη επιβεβαιωμένο ιστορικό','last_actual_denosumab_dose_not_established':'μη επιβεβαιωμένη τελευταία πραγματική χορήγηση','conflicting_administration_records':'αντικρουόμενες χορηγήσεις'})[x]||x).join(' · ')}</p>`:''}
    <p class="fineprint">Δεν δημιουργήθηκε συνταγή, χορήγηση, πραγματικό task ή μόνιμη κλινική απόφαση.</p>
    <div class="actionbar">${btn('edit-decision','Διόρθωση κατάληξης','button','button secondary')}<button class="quiet-button" data-action="show-handoff">${session.opened==='handoff-json'?'Απόκρυψη':'Τι θα μεταφερθεί στο Σχέδιο;'}</button></div>
    ${session.opened==='handoff-json'?`<div class="soft-panel"><p class="note">Το μελλοντικό Σχέδιο θα λάβει μόνο την πραγματική απόφαση και τις εκκρεμότητες — δεν θα θεωρήσει καμία ενέργεια εκτελεσμένη.</p><pre style="font-size:.73rem;white-space:pre-wrap;overflow-wrap:anywhere">${enc(JSON.stringify(h,null,2))}</pre></div>`:''}
    </div>`;
  }
  function mainContent(){
    if(mode==='verify')return verifyHistory();
    if(mode==='uncertain')return uncertainHistory();
    if(mode==='journey')return journey();
    if(mode==='approaches')return chooseApproaches();
    if(mode==='decision')return decisionForm();
    if(mode==='handoff')return handoff();
    return rest();
  }
  function render(){root.innerHTML=sessionHeader();document.title='Θεραπευτική στρατηγική · '+session.patient.id+' · Συνθετικό';}
  root.addEventListener('click',event=>{
    const node=event.target.closest('button[data-action],button[data-history],button[data-course],button[data-focus]');
    if(!node)return;
    if(node.dataset.history){Core.setHistory(session,node.dataset.history);mode=node.dataset.history==='confirmed_none'?'approaches':'uncertain';render();return;}
    if(node.dataset.course){Core.setCourse(session,node.dataset.course);render();return;}
    if(node.dataset.focus){session.discussion.selected_focus=node.dataset.focus;render();return;}
    const action=node.dataset.action;
    if(action==='close-detail'){mode='rest';render();return;}
    if(action==='why'){whyOpen=!whyOpen;render();return;}
    if(action==='provenance'){provenanceOpen=!provenanceOpen;render();return;}
    if(action==='begin'){
      const c=Core.getTreatmentContext(session.patient,session.discussion);
      mode=c.priorDenosumab?'journey':c.status==='not_recorded'?'verify':'approaches';render();return;
    }
    if(action==='verify-again'){mode='verify';render();return;}
    if(action==='save-dose'){
      const val=document.getElementById('doseDate')?.value;
      if(val){Core.setDose(session,val);render();}else document.getElementById('doseDate')?.focus();return;
    }
    if(action==='dose-unknown'){Core.setDose(session,null);render();return;}
    if(action==='open-continuity'){mode='approaches';render();return;}
    if(action==='direct-decision'){mode='decision';render();return;}
    if(action.startsWith('consider-')){Core.markConsidered(session,action.slice(9));render();return;}
    if(action==='edit-decision'){mode='decision';render();return;}
    if(action==='show-handoff'){session.opened=session.opened==='handoff-json'?'':'handoff-json';render();return;}
  });
  root.addEventListener('submit',event=>{
    if(event.target.id!=='decisionForm')return;
    event.preventDefault();const data=new FormData(event.target);
    Core.setDecision(session,{recommendation:data.get('recommendation'),patient_preference:data.get('patient_preference'),decision:data.get('decision'),rationale:data.get('rationale')});
    mode='handoff';render();
  });
  render();
})();