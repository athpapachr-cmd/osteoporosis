from __future__ import annotations


def provider_profile() -> str:
    return """MODULE: osteoporosis

Emit semantic candidates using ONLY these concept keys when applicable:
encounter.archetype
anthropometrics.weight
anthropometrics.current_height
fracture.site
fracture.date
fracture.low_trauma
fracture.occurred_on_treatment
fracture.vertebral_level
risk.current_smoking
risk.high_alcohol
risk.rheumatoid_arthritis
risk.glucocorticoids
risk.glucocorticoid_prednisolone_mg_day
risk.glucocorticoid_duration_months
risk.falls_last_12_months
risk.cfs_score
frax.tool_name
frax.country_or_surrogate_model
frax.femoral_neck_bmd_used
frax.mof_percent
frax.hip_percent
frax.adjusted_mof_percent
frax.adjusted_hip_percent
risk.resulting_category
dxa.date
dxa.spine_bmd
dxa.spine_t_score
dxa.total_hip_bmd
dxa.total_hip_t_score
dxa.femoral_neck_bmd
dxa.femoral_neck_t_score
vfa.indicated
vfa.action
vfa.modality
vfa.vertebral_fracture_found
labs.date
labs.calcium
labs.phosphate
labs.vitamin_d
labs.pth
labs.ctx
labs.p1np
treatment.agent
treatment.status
treatment.start_date
treatment.end_date
treatment.duration_years
administration.agent
administration.scheduled_date
administration.actual_date
administration.next_due_date
administration.status
decision.type
decision.selected_agent
patient.preference
patient.acceptance
followup.task_type
followup.due_date
followup.timeframe_text
clinical.unmapped_narrative

GENERAL SEMANTIC RULES
- Do not emit application target paths, storage keys or database field names.
- Keep option discussed, clinician recommendation, patient preference, patient acceptance and final decision as separate semantic assertions.
- A referral/request is not a completed test or result. A prescription/recommendation is not medication taken or administered.
- Do not invent exact dates from relative/vague timing. Objective results must remain distinct from patient-reported history and clinician interpretation.
- Keep components belonging to one real-world event together in one candidate, and represent distinct repeated events as separate candidates. Never repeat the same concept_key within one candidate.
- When a speaker explicitly corrects a fact, preserve the corrected assertion and do not also return the superseded value as current truth.
- Facts about a relative or another person must use speaker=third_party and must not be attributed to the patient.
- Preserve explicit negative treatment exposure with polarity=negative; do not turn it into positive treatment or administration history.
- A future planned administration is not a completed administration; use planned/future semantics and never invent an actual administration date.
- Preserve explicitly stated numeric values even when they appear clinically implausible; downstream deterministic validation, not the model, decides runtime compatibility.
- An explicit affirmative occurrence such as "I had a fracture" uses polarity=positive unless the source explicitly negates it or leaves occurrence unclear.
- Do not create osteoporosis follow-up/task truth from unrelated non-osteoporosis narrative. Use clinical.unmapped_narrative for clinically meaningful unrelated content unless an osteoporosis-related task is explicitly stated.

EXACT VALUE-KIND AND CODE CONTRACTS
Use the exact kind and exact case-sensitive code strings below. Do not substitute synonyms, labels or free text when a code contract exists.

fracture.site
  kind=code
  allowed: vertebral | hip | distal_radius | proximal_humerus | pelvis | other
  Examples: hip fracture -> hip; distal radius/wrist fracture -> distal_radius.
fracture.date
  kind=date
  Exact month uses YYYY-MM + precision=month. Relative/vague timing uses normalized=null + precision=relative.
fracture.low_trauma / fracture.occurred_on_treatment
  kind=boolean

anthropometrics.weight
  kind=quantity, unit=kg
anthropometrics.current_height
  kind=quantity, unit=cm

risk.current_smoking / risk.high_alcohol / risk.rheumatoid_arthritis / risk.glucocorticoids
  kind=boolean
risk.falls_last_12_months / risk.cfs_score
  kind=integer
risk.glucocorticoid_prednisolone_mg_day
  kind=quantity, unit=mg/day
risk.glucocorticoid_duration_months
  kind=quantity, unit=months

frax.tool_name
  kind=code
  allowed: frax | fraxplus | other
  Explicit "FRAX" -> code=frax.
frax.femoral_neck_bmd_used
  kind=boolean
frax.mof_percent / frax.hip_percent / frax.adjusted_mof_percent / frax.adjusted_hip_percent
  kind=number

DXA
  dxa.date kind=date
  dxa.*_bmd kind=quantity, unit=g/cm²
  dxa.*_t_score kind=number

VFA
  vfa.indicated kind=code allowed yes | no | uncertain, or kind=boolean when the source is explicitly binary
  vfa.action kind=code allowed performed | already_available_reviewed | arranged | reasoned_not_done | missed | not_applicable
  vfa.modality kind=code allowed VFA | spine_xray | CT | MRI | other
  vfa.vertebral_fracture_found kind=boolean

LABS
  labs.date kind=date
  labs.calcium / labs.phosphate / labs.vitamin_d / labs.pth / labs.ctx / labs.p1np kind=quantity with the stated source unit preserved

AGENT CODES for treatment.agent / administration.agent / decision.selected_agent
  kind=code
  allowed: none | alendronate | risedronate | ibandronate_oral | zoledronate | ibandronate_iv | denosumab | teriparatide | romosozumab | raloxifene | hormone_therapy | other

treatment.status
  kind=code
  allowed: planned | active | completed | stopped | holiday | unknown
administration.status
  kind=code
  allowed: done | due | overdue | missed | planned | not_applicable

decision.type
  kind=code
  allowed: start | continue | stop | switch | defer | no_drug_treatment | complete_course | consolidate | refer | uncertain
patient.acceptance
  kind=code
  allowed: accepted | declined | undecided

followup.task_type
  kind=code
  allowed: lab | DXA | administration | followup_visit | referral | VFA_or_imaging | adherence_check | exercise_or_falls | nutrition | other
  Use code=followup_visit for an explicit future return/review visit.
  Use exact code=DXA for an explicit DXA referral/order task.
followup.due_date
  kind=date and only when an exact source-supported day exists
followup.timeframe_text
  kind=text and preserve vague/relative wording without inventing an exact date
clinical.unmapped_narrative / patient.preference
  kind=text

SEMANTIC OWNERSHIP FOR THERAPY / DECISIONS
- treatment.* describes actual historical/current treatment exposure or a true treatment episode. Do NOT use treatment.agent or treatment.status merely because an agent was discussed, recommended, prescribed or selected for a future plan.
- For an agent discussed as an option, emit decision.selected_agent with semantic_type=option_discussed. It will intentionally remain non-authoritative/ambiguous downstream.
- For a clinician-recommended agent, emit decision.selected_agent with semantic_type=clinician_recommendation. Do not add treatment.status from recommendation wording.
- For an explicit final selected agent, emit decision.selected_agent with semantic_type=final_decision. If the clinician explicitly says the final decision is to start it, also emit decision.type code=start in that final_decision assertion/candidate.
- Patient agreement such as "I agree" after a recommendation/decision must be a separate patient_accepted assertion with patient.acceptance code=accepted and speaker=patient.
- A prescription/recommendation alone may use a clinician_recommendation treatment.agent only when the transcript frames it as the prescribed/recommended treatment entity, but it must NEVER create administration.agent, administration.status, administration.actual_date or an inferred treatment.status.

ADMINISTRATION SAFETY
- administration.* is reserved for an actual or explicitly scheduled administration event, never for a treatment option/prescription alone.
- If the source explicitly says administration occurrence is uncertain, use semantic_type=uncertain_needs_review and do not emit administration.status or administration.actual_date as truth. The named drug may be retained as treatment.agent for review with the source/certainty preserved.
- For an explicitly scheduled future denosumab administration, use one followup_task candidate with administration.agent code=denosumab, administration.scheduled_date exact date, administration.status code=planned, speaker=clinician and temporality=future. Never emit status=done or an actual_date unless explicitly stated.

SOURCE / NEGATION SAFETY
- If the patient states that a relative receives an agent, represent that relative fact with speaker=third_party and polarity=positive. Do not infer treatment.status unless the source explicitly states a supported status.
- If the patient separately states "I have never received" an agent, represent the patient fact separately with speaker=patient, polarity=negative and treatment.agent code for the named agent. Do not emit positive treatment.status or administration truth.
- A standalone explicit "I have never received denosumab" must be patient_history_fact + speaker=patient + polarity=negative + treatment.agent code=denosumab, with no treatment.status or administration assertion.
"""
