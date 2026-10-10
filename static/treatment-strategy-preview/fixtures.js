/* Entirely fictional synthetic cases. No patient data or runtime imports. */
(function(root){
 'use strict';
 const base={revision:'fixture-1',source_refs:['synthetic:fracture','synthetic:risk'],
  risk:{label:'Πολύ υψηλός',source_ref:'synthetic:risk',jurisdiction:'UNRESOLVED_CY_NOGG'},
  fractures:[{site:'vertebral',relative:'πρόσφατο σπονδυλικό κάταγμα'}],
  episodes:[],administrations:[],therapy_course:'not_established',
  confirmed_no_prior_therapy:false,history_conflict:false,administration_conflict:false,
  new_fracture_during_treatment:false,completed_anabolic:false,current_treatment:null,lower_risk:false};
 const clone=o=>JSON.parse(JSON.stringify(o));
 function make(id,updates){return Object.assign(clone(base),{id},updates);}
 const samples={
   J:make('J',{title:'Πρώτη θεραπευτική συζήτηση',description:'Δεν έχει καταγραφεί προηγούμενη θεραπεία.'}),
   J0:make('J0',{title:'Επιβεβαιωμένο χωρίς προηγούμενη θεραπεία',confirmed_no_prior_therapy:true}),
   JD:make('JD',{title:'Προηγούμενη Prolia',episodes:[{agent:'denosumab',exposure:'confirmed',label:'Prolia'}],source_refs:['synthetic:denosumab','synthetic:fracture','synthetic:risk']}),
   JDknown:make('JDknown',{title:'Prolia — γνωστή πραγματική χορήγηση',episodes:[{agent:'denosumab',exposure:'confirmed'}],administrations:[{agent:'denosumab',date:'2026-04-11',status:'administered'}],therapy_course:'ongoing'}),
   JDconflict:make('JDconflict',{title:'Prolia — αντικρουόμενες χορηγήσεις',episodes:[{agent:'denosumab',exposure:'confirmed'}],administration_conflict:true}),
   S3:make('S3',{title:'Νέο κάταγμα υπό διφωσφονικό',episodes:[{agent:'alendronate',exposure:'confirmed'}],current_treatment:{agent:'alendronate',label:'αλενδρονάτη'},new_fracture_during_treatment:true}),
   S4:make('S4',{title:'Μετά από τεριπαρατίδη',episodes:[{agent:'teriparatide',exposure:'confirmed'}],completed_anabolic:true}),
   S5:make('S5',{title:'Εκκρεμής πρόσβαση',confirmed_no_prior_therapy:true,access:{status:'unknown',channels:['ΓεΣΥ','ιδιωτική']}}),
   S6:make('S6',{title:'Ανοικτή νεφρική πληροφορία',confirmed_no_prior_therapy:true,renal:{status:'unknown'}}),
   S7:make('S7',{title:'Ο ασθενής δεν επιθυμεί φάρμακο',confirmed_no_prior_therapy:true,patient_declined:true}),
   S8:make('S8',{title:'Χαμηλότερος κίνδυνος',lower_risk:true,risk:{label:'Χαμηλότερος',source_ref:'synthetic:risk',jurisdiction:'UNRESOLVED_CY_NOGG'},fractures:[],confirmed_no_prior_therapy:true})
 };
 root.StrategyFixtures=Object.freeze(samples);
 if(typeof module!=='undefined'&&module.exports)module.exports=samples;
})(typeof globalThis!=='undefined'?globalThis:this);