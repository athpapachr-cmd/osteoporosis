from __future__ import annotations

from typing import Any

from clinical_excellence.core.transcript_contracts import (
    BooleanValueV1, CandidateComponentV1, CodeValueV1, DateValueV1, IntegerValueV1,
    NumberValueV1, ProviderCandidateV1, QuantityValueV1, TargetMappingV1, TextValueV1,
)

AGENTS = {"none","alendronate","risedronate","ibandronate_oral","zoledronate","ibandronate_iv","denosumab","teriparatide","romosozumab","raloxifene","hormone_therapy","other"}
ARCHETYPES = {"initial_assessment_new_or_uncertain_diagnosis","initial_assessment_known_osteoporosis_or_osteopenia","routine_followup_stable","treatment_start","treatment_continuation_or_due_monitoring","treatment_change_or_transition","post_fragility_fracture","fracture_on_treatment","adverse_effect_or_intolerance","treatment_completion_or_consolidation","other"}
DECISIONS = {"start","continue","stop","switch","defer","no_drug_treatment","complete_course","consolidate","refer","uncertain"}
TASK_TYPES = {"lab","DXA","administration","followup_visit","referral","VFA_or_imaging","adherence_check","exercise_or_falls","nutrition","other"}


def _value(component: CandidateComponentV1) -> Any:
    value = component.value
    if isinstance(value, TextValueV1): return value.text
    if isinstance(value, CodeValueV1): return value.code
    if isinstance(value, (NumberValueV1, IntegerValueV1, BooleanValueV1)): return value.value
    if isinstance(value, QuantityValueV1): return {"value": value.value, "unit": value.unit}
    if isinstance(value, DateValueV1): return {"normalized": value.normalized, "precision": value.precision, "date_text": value.date_text}
    return None


def _mapped(c: CandidateComponentV1, path: str, proposed: Any | None = None, reason: str = "DIRECT_RUNTIME_TARGET") -> TargetMappingV1:
    return TargetMappingV1(component_keys=[c.concept_key], target_path=path, status="mapped", reason_code=reason, proposed_value=_value(c) if proposed is None else proposed)


def _ambiguous(c: CandidateComponentV1, reason: str, path: str | None = None) -> TargetMappingV1:
    return TargetMappingV1(component_keys=[c.concept_key], target_path=path, status="ambiguous", reason_code=reason, proposed_value=_value(c))


def _unmapped(c: CandidateComponentV1, reason: str) -> TargetMappingV1:
    return TargetMappingV1(component_keys=[c.concept_key], target_path=None, status="unmapped", reason_code=reason, proposed_value=_value(c))


def _code(c: CandidateComponentV1) -> str | None:
    return c.value.code if isinstance(c.value, CodeValueV1) else None


def _bool(c: CandidateComponentV1) -> bool | None:
    return c.value.value if isinstance(c.value, BooleanValueV1) else None


def _number(c: CandidateComponentV1) -> float | int | None:
    if isinstance(c.value, (NumberValueV1, IntegerValueV1)): return c.value.value
    return None


def _quantity(c: CandidateComponentV1, allowed_units: set[str]) -> float | None:
    if not isinstance(c.value, QuantityValueV1): return None
    unit = c.value.unit.strip().lower().replace("²", "2")
    if unit not in {x.lower().replace("²", "2") for x in allowed_units}: return None
    return c.value.value


def _date(c: CandidateComponentV1, precision: str = "day") -> str | None:
    if not isinstance(c.value, DateValueV1): return None
    return c.value.normalized if c.value.precision == precision else None


def map_candidate(candidate: ProviderCandidateV1) -> list[TargetMappingV1]:
    out: list[TargetMappingV1] = []
    for c in candidate.components:
        key = c.concept_key
        if key == "encounter.archetype":
            code = _code(c); out.append(_mapped(c,"encounter_archetype",code) if code in ARCHETYPES else _ambiguous(c,"UNSUPPORTED_CODE","encounter_archetype")); continue
        if key == "anthropometrics.weight":
            val = _quantity(c,{"kg"}); out.append(_mapped(c,"anthropometrics.weight_kg",val) if val is not None else _ambiguous(c,"UNSUPPORTED_UNIT","anthropometrics.weight_kg")); continue
        if key == "anthropometrics.current_height":
            val = _quantity(c,{"cm"}); out.append(_mapped(c,"anthropometrics.current_height_cm",val) if val is not None else _ambiguous(c,"UNSUPPORTED_UNIT","anthropometrics.current_height_cm")); continue
        if key == "fracture.site": out.append(_mapped(c,"fracture_history.events[].site")); continue
        if key == "fracture.date":
            val = _date(c,"month"); out.append(_mapped(c,"fracture_history.events[].month",val) if val else _ambiguous(c,"TARGET_SUPPORTS_MONTH_ONLY","fracture_history.events[].month")); continue
        if key in {"fracture.low_trauma","fracture.occurred_on_treatment"}:
            val = _bool(c); path = "fracture_history.events[].low_trauma" if key.endswith("low_trauma") else "fracture_history.events[].occurred_on_treatment"
            out.append(_mapped(c,path,"yes" if val else "no") if val is not None else _ambiguous(c,"TYPE_MISMATCH",path)); continue
        if key == "fracture.vertebral_level": out.append(_mapped(c,"fracture_history.events[].vertebral_level")); continue
        if key in {"risk.current_smoking","risk.high_alcohol","risk.rheumatoid_arthritis"}:
            path={"risk.current_smoking":"risk_assessment.current_smoking","risk.high_alcohol":"risk_assessment.high_alcohol_3_units_day","risk.rheumatoid_arthritis":"risk_assessment.rheumatoid_arthritis"}[key]
            val=_bool(c); out.append(_mapped(c,path,"yes") if val is True else _ambiguous(c,"NEGATIVE_DEFAULT_SEMANTICS",path) if val is False else _ambiguous(c,"TYPE_MISMATCH",path)); continue
        if key == "risk.glucocorticoids":
            val=_bool(c); out.append(_mapped(c,"risk_context.glucocorticoids",True) if val is True else _ambiguous(c,"NEGATIVE_DEFAULT_SEMANTICS","risk_context.glucocorticoids") if val is False else _ambiguous(c,"TYPE_MISMATCH","risk_context.glucocorticoids")); continue
        if key == "risk.glucocorticoid_prednisolone_mg_day":
            val=_quantity(c,{"mg/day","mg/d","mg per day"}); out.append(_mapped(c,"risk_context.glucocorticoid_prednisolone_mg_day",val) if val is not None else _ambiguous(c,"UNSUPPORTED_UNIT","risk_context.glucocorticoid_prednisolone_mg_day")); continue
        if key == "risk.glucocorticoid_duration_months":
            val=_quantity(c,{"months","month","mo"}); out.append(_mapped(c,"risk_context.glucocorticoid_duration_months",val) if val is not None else _ambiguous(c,"UNSUPPORTED_UNIT","risk_context.glucocorticoid_duration_months")); continue
        if key == "risk.falls_last_12_months": out.append(_mapped(c,"risk_context.falls_last_12_months",_number(c)) if _number(c) is not None else _ambiguous(c,"TYPE_MISMATCH","risk_context.falls_last_12_months")); continue
        if key == "risk.cfs_score": out.append(_mapped(c,"risk_context.cfs_score",_number(c)) if _number(c) is not None else _ambiguous(c,"TYPE_MISMATCH","risk_context.cfs_score")); continue
        if key == "frax.tool_name": out.append(_mapped(c,"risk_assessment.tool_name")); continue
        if key == "frax.country_or_surrogate_model": out.append(_mapped(c,"risk_assessment.country_or_surrogate_model")); continue
        if key == "frax.femoral_neck_bmd_used":
            val=_bool(c); out.append(_mapped(c,"risk_assessment.femoral_neck_bmd_used","yes" if val else "no") if val is not None else _ambiguous(c,"TYPE_MISMATCH","risk_assessment.femoral_neck_bmd_used")); continue
        if key in {"frax.mof_percent","frax.hip_percent"}:
            path="risk_assessment.frax_mof_percent" if key.endswith("mof_percent") else "risk_assessment.frax_hip_percent"; val=_number(c); out.append(_mapped(c,path,val) if val is not None else _ambiguous(c,"TYPE_MISMATCH",path)); continue
        if key in {"frax.adjusted_mof_percent","frax.adjusted_hip_percent"}: out.append(_unmapped(c,"ADJUSTED_RISK_MUST_NOT_OVERWRITE_ORIGINAL_FRAX")); continue
        if key == "risk.resulting_category": out.append(_mapped(c,"risk_assessment.resulting_risk_category")); continue
        if key == "dxa.date":
            val=_date(c); out.append(_mapped(c,"step3.dxa.date",val) if val else _ambiguous(c,"TARGET_REQUIRES_EXACT_DATE","step3.dxa.date")); continue
        if key in {"dxa.spine_bmd","dxa.total_hip_bmd","dxa.femoral_neck_bmd"}:
            path={"dxa.spine_bmd":"step3.dxa.spine_bmd","dxa.total_hip_bmd":"step3.dxa.total_hip_bmd","dxa.femoral_neck_bmd":"step3.dxa.femoral_neck_bmd"}[key]; val=_quantity(c,{"g/cm2","g/cm²"}); out.append(_mapped(c,path,val) if val is not None and candidate.semantic_type=="objective_result" else _ambiguous(c,"OBJECTIVE_RESULT_OR_UNIT_REQUIRED",path)); continue
        if key in {"dxa.spine_t_score","dxa.total_hip_t_score","dxa.femoral_neck_t_score"}:
            path={"dxa.spine_t_score":"step3.dxa.spine_t","dxa.total_hip_t_score":"step3.dxa.total_hip_t","dxa.femoral_neck_t_score":"step3.dxa.femoral_neck_t"}[key]; val=_number(c); out.append(_mapped(c,path,val) if val is not None and candidate.semantic_type=="objective_result" else _ambiguous(c,"OBJECTIVE_RESULT_REQUIRED",path)); continue
        if key in {"vfa.indicated","vfa.action","vfa.modality"}: out.append(_mapped(c,{"vfa.indicated":"step3.vfa.indicated","vfa.action":"step3.vfa.action","vfa.modality":"step3.vfa.modality"}[key])); continue
        if key == "vfa.vertebral_fracture_found":
            val=_bool(c); out.append(_mapped(c,"step3.vfa.vertebral_found","yes" if val else "no") if val is not None and candidate.semantic_type=="objective_result" else _ambiguous(c,"OBJECTIVE_RESULT_REQUIRED","step3.vfa.vertebral_found")); continue
        if key == "labs.date":
            val=_date(c); out.append(_mapped(c,"step3.labs.labs_date",val) if val else _ambiguous(c,"TARGET_REQUIRES_EXACT_DATE","step3.labs.labs_date")); continue
        if key.startswith("labs."):
            lab=key.split(".",1)[1]; units={"calcium":{"mmol/l"},"phosphate":{"mmol/l"},"vitamin_d":{"ng/ml"},"pth":{"pg/ml"},"ctx":{"ng/ml"},"p1np":{"ng/ml"}}; field={"calcium":"ca","phosphate":"phosphate","vitamin_d":"vitamin_d","pth":"pth","ctx":"ctx","p1np":"p1np"}.get(lab)
            if field and lab in units:
                val=_quantity(c,units[lab]); path=f"step3.labs.{field}"; out.append(_mapped(c,path,val) if val is not None and candidate.semantic_type=="objective_result" else _ambiguous(c,"OBJECTIVE_RESULT_OR_UNIT_REQUIRED",path))
            else: out.append(_unmapped(c,"UNSUPPORTED_LAB_TARGET"))
            continue
        if key == "treatment.agent":
            code=_code(c); out.append(_mapped(c,"step4.treatment_episodes[].agent",code) if code in AGENTS else _ambiguous(c,"UNSUPPORTED_AGENT","step4.treatment_episodes[].agent")); continue
        if key in {"treatment.status","treatment.duration_years"}: out.append(_mapped(c,"step4.treatment_episodes[]."+key.split(".",1)[1],_number(c) if key.endswith("years") else _value(c))); continue
        if key in {"treatment.start_date","treatment.end_date"}:
            val=_date(c); path="step4.treatment_episodes[]."+key.split(".",1)[1]; out.append(_mapped(c,path,val) if val else _ambiguous(c,"TARGET_REQUIRES_EXACT_DATE",path)); continue
        if key == "administration.agent":
            code=_code(c); out.append(_mapped(c,"step4.administrations[].agent",code) if code in AGENTS else _ambiguous(c,"UNSUPPORTED_AGENT","step4.administrations[].agent")); continue
        if key in {"administration.scheduled_date","administration.actual_date","administration.next_due_date"}:
            val=_date(c); path="step4.administrations[]."+key.split(".",1)[1]; out.append(_mapped(c,path,val) if val else _ambiguous(c,"TARGET_REQUIRES_EXACT_DATE",path)); continue
        if key == "administration.status": out.append(_mapped(c,"step4.administrations[].status")); continue
        if key == "decision.type":
            code=_code(c); out.append(_mapped(c,"step4.decision.type",code) if code in DECISIONS and candidate.semantic_type=="final_decision" else _ambiguous(c,"FINAL_DECISION_REQUIRED","step4.decision.type")); continue
        if key == "decision.selected_agent":
            code=_code(c); out.append(_mapped(c,"step4.decision.selected_agent",code) if code in AGENTS and candidate.semantic_type=="final_decision" else _ambiguous(c,"FINAL_DECISION_REQUIRED","step4.decision.selected_agent")); continue
        if key == "patient.preference": out.append(_unmapped(c,"PREFERENCE_CONTENT_HAS_NO_LOSSLESS_RUNTIME_TARGET")); continue
        if key == "patient.acceptance":
            code=_code(c); mapping={"accepted":"yes","declined":"no","undecided":"undecided"}; out.append(_mapped(c,"step4.decision.patient_accepted",mapping[code]) if code in mapping and candidate.semantic_type in {"patient_accepted","patient_declined","patient_undecided"} else _ambiguous(c,"PATIENT_DISPOSITION_REQUIRED","step4.decision.patient_accepted")); continue
        if key == "followup.task_type":
            code=_code(c); out.append(_mapped(c,"step4.tasks[].type",code) if code in TASK_TYPES else _ambiguous(c,"UNSUPPORTED_TASK_TYPE","step4.tasks[].type")); continue
        if key == "followup.due_date":
            val=_date(c); out.append(_mapped(c,"step4.tasks[].due_date",val) if val else _ambiguous(c,"TARGET_REQUIRES_EXACT_DATE","step4.tasks[].due_date")); continue
        if key == "followup.timeframe_text": out.append(_mapped(c,"step4.tasks[].timeframe_text")); continue
        if key == "clinical.unmapped_narrative": out.append(_unmapped(c,"NO_CURRENT_RUNTIME_TARGET")); continue
        out.append(_unmapped(c,"UNKNOWN_CONCEPT_KEY"))
    return out
