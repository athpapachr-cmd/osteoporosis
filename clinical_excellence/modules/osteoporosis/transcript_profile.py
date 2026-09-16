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

Do not emit application target paths, storage keys or database field names. Keep option discussed, clinician recommendation, patient preference and final decision as separate semantic assertions. A referral/request is not a completed test or result. A prescription is not medication taken/administered. Do not invent exact dates from relative/vague timing. Objective results must remain distinct from patient-reported history and clinician interpretation."""
