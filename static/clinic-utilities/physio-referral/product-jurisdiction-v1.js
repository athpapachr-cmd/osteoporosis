'use strict';
// Jurisdiction overlay v1 presentation. The server owns profile validation and
// local/core separation. This layer only renders already-reviewed display data
// inside an evidence sheet the clinician explicitly opened.

const JURISDICTION_DIRECTIONS = Object.freeze({
  for:'Υπέρ',
  conditional_for:'Υπέρ υπό προϋποθέσεις',
  against:'Δεν συνιστάται',
  against_routine_use:'Κατά της συνήθους χρήσης',
  neutral_or_insufficient:'Ανεπαρκή / ουδέτερα δεδομένα',
  informational:'Πληροφοριακό',
  not_applicable:'Δεν εφαρμόζεται',
});

function jurisdictionStatusText(local) {
  const status=local?.operational_status||{};
  const parts=[];
  if(status.guideline_status==='published_active') parts.push('Τοπική οδηγία: δημοσιευμένη / ενεργή');
  else if(status.guideline_status==='published_status_unclear') parts.push('Κατάσταση τοπικής οδηγίας: ασαφής');
  if(status.information_system_status==='planned') parts.push('Ενσωμάτωση στο Σύστημα Πληροφορικής ΓεΣΥ: προγραμματισμένη');
  else if(status.information_system_status==='active_verified') parts.push('Ενσωμάτωση στο Σύστημα Πληροφορικής: επιβεβαιωμένα ενεργή');
  if(status.public_artifact_finality==='metadata_conflict') parts.push('Το δημόσιο κείμενο διατηρεί ασυμφωνία μεταδεδομένων έκδοσης');
  return parts.join(' · ');
}

function jurisdictionRelationText(local) {
  if(local.relationship_to_core==='local_position_within_international_conflict') {
    return 'Η τοπική θέση ακολουθεί μία πλευρά της διεθνούς διαφωνίας· η διεθνής κατάσταση δεν αλλάζει.';
  }
  if(local.relationship_to_core==='local_difference') {
    return 'Η τοπική θέση διαφέρει από τη διεθνή θέση που εμφανίζεται παραπάνω.';
  }
  if(local.relationship_to_core==='agreement') {
    return 'Η τοπική θέση είναι συμβατή με τη διεθνή θέση που εμφανίζεται παραπάνω.';
  }
  return 'Τοπικό πλαίσιο που διατηρείται ξεχωριστά από τη διεθνή τεκμηρίωση.';
}

function safeJurisdictionSourceLink(local) {
  const href=local?.source_provenance?.source_url;
  if(!href) return null;
  try {
    const url=new URL(href);
    if(url.protocol!=='https:' || url.username || url.password) return null;
    return make('a',{href:url.href,target:'_blank',rel:'noopener noreferrer',text:'Τοπική πηγή ↗'});
  } catch(_error) { return null; }
}

function jurisdictionEvidenceNode(item) {
  const local=response?.evidence?.[item]?.jurisdiction;
  const profile=response?.jurisdiction_profile;
  if(!local || !profile || local.jurisdiction_profile_id!==profile.profile_id) return null;
  if(local.policy_class!=='clinical_guidance') return null;

  const policy=local.display_policy||{};
  const headline=policy.local_label || profile.label || 'Τοπική θέση';
  const node=make('section',{class:'source-position jurisdiction-position-v1','data-jurisdiction-position':local.local_position_id},[
    make('div',{class:'source-heading'},[
      make('span',{text:headline}),
      make('span',{class:'source-direction',text:JURISDICTION_DIRECTIONS[local.local_direction]||local.local_direction}),
    ]),
    make('p',{class:'source-summary',text:local.normalized_local_position}),
    make('p',{class:'scope-caption',text:jurisdictionRelationText(local)}),
  ]);

  const detail=make('details',{class:'evidence-deep jurisdiction-deep-v1'},[
    make('summary',{text:'Τοπική πηγή και κατάσταση εφαρμογής'}),
    make('p',{class:'scope-caption',text:jurisdictionStatusText(local)||'Η κατάσταση εφαρμογής δεν έχει τεκμηριωθεί περαιτέρω.'}),
  ]);
  const provenance=local.source_provenance||{};
  if(provenance.source_title) detail.append(make('p',{text:provenance.source_title}));
  if(provenance.recommendation_page_or_section) detail.append(make('p',{class:'scope-caption',text:'Εντοπισμός: '+provenance.recommendation_page_or_section}));
  if(provenance.reviewed_on) detail.append(make('p',{class:'scope-caption',text:'Τοπική ανασκόπηση: '+provenance.reviewed_on.split('-').reverse().join('/')}));
  const link=safeJurisdictionSourceLink(local); if(link) detail.append(link);
  node.append(detail);
  return node;
}

function appendJurisdictionEvidence(item) {
  const body=$('#sheetBody'); if(!body) return;
  if(body.querySelector('[data-jurisdiction-position]')) return;
  const node=jurisdictionEvidenceNode(item); if(!node) return;
  const firstSource=body.querySelector('.source-position');
  if(firstSource) body.insertBefore(node,firstSource); else body.append(node);
}

const jurisdictionBaseOpenSheet=openSheet;
openSheet=function(type,item=null,back=null){
  const result=jurisdictionBaseOpenSheet(type,item,back);
  if(type==='evidence') appendJurisdictionEvidence(item);
  return result;
};
