# main.py

from enum import Enum
from typing import List, Optional, Tuple
import json
import os
import importlib.util
from html import escape
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field, conint, confloat

from typing import Dict, Any

from sqlalchemy import (
    Column,
    DateTime,
    JSON,
    String,
    create_engine,
    select,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
SOURCE_INDEX = ROOT_DIR / "index.html"
TARGET_INDEX = STATIC_DIR / "index.html"

try:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_INDEX.exists():
        shutil.copyfile(SOURCE_INDEX, TARGET_INDEX)
except OSError:
    pass

# =========================
# App & CORS
# =========================

app = FastAPI(
    title="Papachristou Ortho Osteoporosis Support",
    description=(
        "Guideline-inspired decision support for osteoporosis risk stratification, "
        "FRAX-style internal risk indexing, calcium intake estimation, lab pattern hints, "
        "and treatment context. For clinician use only."
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for now; you can restrict later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static-root")


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    if SOURCE_INDEX.exists():
        return FileResponse(SOURCE_INDEX, media_type="text/html")
    raise HTTPException(status_code=404, detail="Index file not available")

# =========================
# OpenAI client
# =========================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# Database (SQLite minimal)
# =========================


class Base(DeclarativeBase):
    pass


class AssessmentORM(Base):
    __tablename__ = "assessments"

    id = Column(String, primary_key=True)  # assessment_id
    patient_id = Column(String, index=True)
    created_at = Column(DateTime, index=True)

    input_json = Column(JSON)
    output_json = Column(JSON)

    risk_category = Column(String, index=True)
    current_therapy_type = Column(String, index=True)


DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./osteoporosis.db")

# Normalize Render/Postgres URL and choose an installed SQLAlchemy driver.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL and "+psycopg2" not in DATABASE_URL:
    has_psycopg = importlib.util.find_spec("psycopg") is not None
    has_psycopg2 = importlib.util.find_spec("psycopg2") is not None
    if has_psycopg:
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    elif has_psycopg2:
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    future=True,
)

Base.metadata.create_all(bind=engine)

# =========================
# Enums & Schemas
# =========================


class Sex(str, Enum):
    female = "female"
    male = "male"


class MenopauseStatus(str, Enum):
    pre = "pre"
    peri = "peri"
    post = "post"
    unknown = "unknown"


class FractureType(str, Enum):
    hip = "hip"
    vertebral = "vertebral"
    non_vertebral = "non_vertebral"


class RiskCategory(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"
    very_high = "very_high"


class ExerciseLevel(str, Enum):
    none = "none"
    light = "light"
    moderate = "moderate"
    vigorous = "vigorous"


class DailyWalking(str, Enum):
    none = "none"
    under_15_min = "under_15_min"
    between_15_30_min = "between_15_30_min"
    over_30_min = "over_30_min"


class MorseAmbulatoryAid(str, Enum):
    none_bedrest_assist = "none_bedrest_assist"
    cane_walker_crutches = "cane_walker_crutches"
    furniture = "furniture"


class MorseGait(str, Enum):
    normal_bedrest_wheelchair = "normal_bedrest_wheelchair"
    weak = "weak"
    impaired = "impaired"


class MorseMentalStatus(str, Enum):
    aware_of_limitations = "aware_of_limitations"
    overestimates_or_forgets_limitations = "overestimates_or_forgets_limitations"


class CurrentTherapyType(str, Enum):
    none = "none"
    oral_bisphosphonate = "oral_bisphosphonate"
    iv_bisphosphonate = "iv_bisphosphonate"
    denosumab = "denosumab"
    teriparatide = "teriparatide"
    romosozumab = "romosozumab"
    raloxifene = "raloxifene"
    other = "other"

class TherapyEpisode(BaseModel):
    therapy_type: CurrentTherapyType
    duration_years: Optional[float] = Field(
        default=None,
        description="Approximate duration of this episode in years",
    )
    is_holiday: bool = Field(
        default=False,
        description="True if this episode represents a drug holiday / no pharmacologic treatment",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional notes (e.g. reason for switch, holiday, adverse effect)",
    )



class TScoreSnapshot(BaseModel):
    date: Optional[str] = Field(
        default=None,
        description="Date of the DEXA scan (YYYY-MM-DD).",
    )
    spine_total: Optional[float] = None
    spine_l1: Optional[float] = None
    spine_l2: Optional[float] = None
    spine_l3: Optional[float] = None
    spine_l4: Optional[float] = None
    total_hip: Optional[float] = None
    femoral_neck: Optional[float] = None


class Suggestion(BaseModel):
    category: str
    text: str
    evidence_ids: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)


class VeryHighCriterionStatus(BaseModel):
    key: str
    label: str
    met: bool
    detail: Optional[str] = None


class OsteoInput(BaseModel):
    # Demographics / context
    age: conint(ge=18, le=110)
    sex: Sex
    menopause_status: MenopauseStatus = MenopauseStatus.unknown

    # BMD / fractures
    spine_t_score: Optional[confloat(le=5.0, ge=-8.0)] = Field(
        default=None, description="Lumbar spine T-score"
    )
    total_hip_t_score: Optional[confloat(le=5.0, ge=-8.0)] = Field(
        default=None, description="Total hip T-score"
    )
    femoral_neck_t_score: Optional[confloat(le=5.0, ge=-8.0)] = Field(
        default=None, description="Femoral neck T-score"
    )
    prior_fragility_fractures: List[FractureType] = Field(
        default_factory=list,
        description="List of prior low-trauma fractures",
    )
    vertebral_fracture_count: conint(ge=0, le=20) = Field(
        default=0, description="Number of vertebral fragility fractures"
    )
    hip_fracture_count: conint(ge=0, le=4) = Field(
        default=0, description="Number of hip fragility fractures"
    )
    recent_fragility_fracture: bool = Field(
        default=False, description="Any recent/imminent fragility fracture pattern"
    )

    # FRAX (if already calculated externally; optional)
    frax_major_osteoporotic: Optional[confloat(ge=0.0, le=100.0)] = None
    frax_hip: Optional[confloat(ge=0.0, le=100.0)] = None
    # Kept only for backwards compatibility with older stored records.
    dvo_3y_risk_percent: Optional[confloat(ge=0.0, le=100.0)] = None

    # FRAX-style clinical risk factors for internal risk index (NOT official FRAX)
    weight_kg: Optional[confloat(gt=0.0, le=300.0)] = None
    height_cm: Optional[confloat(gt=0.0, le=250.0)] = None
    parental_hip_fracture: bool = False
    current_smoker: bool = False
    glucocorticoids: bool = False
    rheumatoid_arthritis: bool = False
    secondary_osteoporosis: bool = False
    high_alcohol_intake: bool = False

    # Mineral metabolism & bone labs (old names kept for backwards compatibility)
    serum_calcium: Optional[float] = Field(
        default=None, description="(Deprecated) Serum calcium, use serum_calcium_mg_dl"
    )
    vitamin_d_25oh: Optional[float] = Field(
        default=None, description="(Deprecated) 25-OH Vit D, use vitamin_d_25oh_ng_ml"
    )

    serum_calcium_mg_dl: Optional[float] = Field(
        default=None, description="Serum calcium (mg/dL)"
    )
    serum_phosphorus_mg_dl: Optional[float] = Field(
        default=None, description="Serum phosphorus (mg/dL)"
    )
    vitamin_d_25oh_ng_ml: Optional[float] = Field(
        default=None, description="25-OH Vitamin D (ng/mL)"
    )
    pth_pg_ml: Optional[float] = Field(
        default=None, description="Intact PTH (pg/mL)"
    )

    # 24h urine
    urine_calcium_24h_mg: Optional[float] = Field(
        default=None, description="24-hour urine calcium (mg/24h)"
    )

    # Bone turnover markers (single timepoint)
    osteocalcin_ng_ml: Optional[float] = None
    bone_alk_phos_u_l: Optional[float] = None
    total_alk_phos_u_l: Optional[float] = None
    ctx_ng_ml: Optional[float] = None
    p1np_ng_ml: Optional[float] = None

    # Other labs
    serum_magnesium_mg_dl: Optional[float] = None
    serum_zinc_ug_dl: Optional[float] = None
    serum_glucose_mg_dl: Optional[float] = None
    tsh_u_iu_ml: Optional[float] = None
    free_t4_ng_dl: Optional[float] = None

    serum_urea_mg_dl: Optional[float] = None
    serum_creatinine_mg_dl: Optional[float] = None

    # Supplements / intake
    calcium_supplement: bool = False
    calcium_supplement_mg_per_day: Optional[float] = None

    vitamin_d_supplement: bool = False
    vitamin_d3_iu_per_day: Optional[float] = None

    magnesium_supplement: bool = False
    magnesium_supplement_mg_per_day: Optional[float] = None

    zinc_supplement: bool = False
    zinc_supplement_mg_per_day: Optional[float] = None

    boron_supplement: bool = False
    boron_supplement_mg_per_day: Optional[float] = None

    vitamin_k2_supplement: bool = False
    vitamin_k2_ug_per_day: Optional[float] = None

    fortibone_supplement: bool = False
    colabone_supplement: bool = False

    # Current pharmacologic osteoporosis therapy
    current_therapy_type: CurrentTherapyType = CurrentTherapyType.none
    current_therapy_duration_years: Optional[float] = Field(
        default=None, description="Approximate duration on current therapy (years)"
    )
    fractures_during_current_therapy: bool = Field(
        default=False,
        description="Any new fragility fracture while on current therapy?",
    )
    significant_therapy_adverse_effects: bool = Field(
        default=False,
        description="Any clinically significant adverse effects related to current therapy?",
    )

    # Past osteoporosis treatment episodes (including holidays)
    therapy_history: List[TherapyEpisode] = Field(
        default_factory=list,
        description=(
            "Chronological list of past therapy episodes for this patient, "
            "including drug holidays. The last ongoing episode should match current_therapy_type."
        ),
    )

    # Historical T-score panels
    t_score_history: List[TScoreSnapshot] = Field(
        default_factory=list,
        description="Chronological list of T-score panels with scan dates.",
    )


    # Lifestyle / functional status
    exercise_level: ExerciseLevel = ExerciseLevel.none
    daily_walking: DailyWalking = DailyWalking.none
    dietician_follow_up: bool = False
    frailty: bool = False
    cfs_score: Optional[conint(ge=1, le=9)] = Field(
        default=None,
        description="Clinical Frailty Scale (CFS), 1-9.",
    )
    high_falls_risk: bool = False
    history_of_falls_last_year: bool = False
    dementia_or_cognitive_impairment: bool = False
    significant_immobility: bool = False
    morse_history_of_falling_3m: bool = False
    morse_secondary_diagnosis: bool = False
    morse_ambulatory_aid: MorseAmbulatoryAid = MorseAmbulatoryAid.none_bedrest_assist
    morse_iv_or_heparin_lock: bool = False
    morse_gait: MorseGait = MorseGait.normal_bedrest_wheelchair
    morse_mental_status: MorseMentalStatus = MorseMentalStatus.aware_of_limitations

    # Calcium intake fields (food)
    milk_portions_per_day: conint(ge=0, le=20) = 0
    yogurt_portions_per_day: conint(ge=0, le=20) = 0
    cheese_portions_per_day: conint(ge=0, le=20) = 0
    leafy_greens_portions_per_day: conint(ge=0, le=20) = 0
    fortified_food_portions_per_day: conint(ge=0, le=20) = 0
    other_dairy_portions_per_day: conint(ge=0, le=20) = 0


class FollowUpStep(BaseModel):
    text: str
    timeframe: str


class OsteoAssessment(BaseModel):
    risk_category: RiskCategory
    risk_reasons: List[str]
    internal_frax_like_index: Optional[float]
    internal_frax_like_note: Optional[str]
    estimated_total_calcium_mg: Optional[float]
    calcium_intake_note: Optional[str]
    suggestions: List[Suggestion]
    clinical_note: str
    patient_summary: str
    follow_up_steps: List[FollowUpStep]
    very_high_criteria: List[VeryHighCriterionStatus] = Field(default_factory=list)


class OsteoEvaluationRequest(BaseModel):
    patient_id: str = Field(
        description="Your internal patient ID or EMR ID for this patient."
    )
    input_data: OsteoInput


class OsteoStoredAssessment(BaseModel):
    assessment_id: str
    patient_id: str
    created_at: datetime
    input_data: OsteoInput
    assessment: OsteoAssessment

class PatientHandoutRequest(BaseModel):
    assessment: OsteoAssessment
    input_data: Optional[OsteoInput] = None
    agreed_plan: List[str] = Field(default_factory=list)
    patient_elaboration: Optional[str] = None


class PatientHandoutResponse(BaseModel):
    handout_html: str


HANDOUT_MONTHS = [0, 3, 6, 12, 18, 24]


THERAPY_PROFILE_LABELS: Dict[str, str] = {
    "none": "No active osteoporosis drug",
    "alendronate": "Alendronate (oral bisphosphonate)",
    "risedronate": "Risedronate (oral bisphosphonate)",
    "oral_bp": "Oral bisphosphonate (class)",
    "zoledronate": "Zoledronate (IV bisphosphonate)",
    "denosumab": "Denosumab",
    "teriparatide": "Teriparatide",
    "romosozumab": "Romosozumab",
    "raloxifene": "Raloxifene (SERM)",
    "other": "Other therapy class",
}


def therapy_profile_label(profile: str) -> str:
    return THERAPY_PROFILE_LABELS.get(profile, profile.replace("_", " "))


def therapy_profile_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    if not t:
        return None
    if "denosumab" in t or "dmab" in t:
        return "denosumab"
    if "zoledronate" in t or "zol" in t:
        return "zoledronate"
    if "alendronate" in t or "aln" in t:
        return "alendronate"
    if "risedronate" in t or "ris" in t:
        return "risedronate"
    if "oral" in t and "bisphosph" in t:
        return "oral_bp"
    if "bisphosph" in t or "διφωσφ" in t:
        return "oral_bp"
    if "romosozumab" in t or "evenity" in t:
        return "romosozumab"
    if "teriparatide" in t or "αναβολ" in t:
        return "teriparatide"
    if "raloxifene" in t or "serm" in t:
        return "raloxifene"
    return None


def therapy_profile_from_enum(therapy: CurrentTherapyType) -> str:
    if therapy == CurrentTherapyType.none:
        return "none"
    if therapy == CurrentTherapyType.oral_bisphosphonate:
        return "oral_bp"
    if therapy == CurrentTherapyType.iv_bisphosphonate:
        return "zoledronate"
    if therapy == CurrentTherapyType.denosumab:
        return "denosumab"
    if therapy == CurrentTherapyType.teriparatide:
        return "teriparatide"
    if therapy == CurrentTherapyType.romosozumab:
        return "romosozumab"
    if therapy == CurrentTherapyType.raloxifene:
        return "raloxifene"
    return "other"


def infer_current_therapy_profile(input_data: OsteoInput) -> str:
    current = input_data.current_therapy_type
    default_profile = therapy_profile_from_enum(current)
    for episode in reversed(input_data.therapy_history):
        if episode.therapy_type != current:
            continue
        parsed = therapy_profile_from_text(episode.notes or "")
        if parsed:
            return parsed
    return default_profile


def extract_agreed_actions(agreed_plan: List[str]) -> set[str]:
    actions: set[str] = set()
    for item in agreed_plan:
        t = item.lower()
        if "έναρξη" in t or "start" in t:
            actions.add("start")
        if "συνέχιση" in t or "continue" in t:
            actions.add("continue")
        if "διακοπή" in t or "stop" in t or "holiday" in t:
            actions.add("stop")
        if "αλλαγή" in t or "switch" in t or "transition" in t or "->" in t:
            actions.add("switch")
    return actions


def infer_target_therapy_profile(
    assessment: OsteoAssessment,
    _input_data: OsteoInput,
    agreed_plan: List[str],
) -> str:
    for item in agreed_plan:
        if "->" in item:
            rhs = item.split("->", 1)[1]
            parsed_rhs = therapy_profile_from_text(rhs)
            if parsed_rhs is not None:
                return parsed_rhs
        if "→" in item:
            rhs = item.split("→", 1)[1]
            parsed_rhs = therapy_profile_from_text(rhs)
            if parsed_rhs is not None:
                return parsed_rhs
        parsed = therapy_profile_from_text(item)
        if parsed is not None:
            return parsed
    if assessment.risk_category in [RiskCategory.very_high, RiskCategory.high]:
        return "denosumab"
    return "oral_bp"


def _trajectory_values(
    current_profile: str,
    action: str,
    target_profile: Optional[str],
) -> Tuple[List[float], str]:
    bp_profiles = {"alendronate", "risedronate", "oral_bp", "zoledronate"}

    if action == "stop":
        if current_profile == "denosumab":
            return [100.0, 98.4, 96.6, 93.5, 91.6, 90.0], "Class-specific (denosumab): stopping without consolidation may cause rapid loss."
        if current_profile == "romosozumab":
            return [100.0, 99.1, 98.4, 97.4, 96.8, 96.2], "Class-specific (romosozumab): follow with anti-resorptive to preserve gains."
        if current_profile == "teriparatide":
            return [100.0, 99.3, 98.6, 97.5, 96.9, 96.3], "Class-specific (teriparatide): consolidation therapy helps maintain BMD gains."
        if current_profile == "zoledronate":
            return [100.0, 100.0, 99.9, 99.6, 99.2, 98.9], "Class-specific (zoledronate): slower offset, with gradual decline over time."
        if current_profile == "alendronate":
            return [100.0, 99.9, 99.6, 99.2, 98.8, 98.4], "Class-specific (alendronate): offset is gradual; monitor for drift."
        if current_profile == "risedronate":
            return [100.0, 99.8, 99.4, 98.9, 98.4, 98.0], "Class-specific (risedronate): offset may appear earlier than stronger-retention BPs."
        if current_profile == "oral_bp":
            return [100.0, 99.9, 99.5, 99.1, 98.7, 98.3], "Class-specific (oral BP): gradual decline after stop."
        if current_profile == "raloxifene":
            return [100.0, 99.8, 99.4, 98.9, 98.5, 98.1], "Class-specific (SERM): modest decline expected after discontinuation."
        return [100.0, 99.7, 99.0, 98.2, 97.6, 97.0], "No active therapy pattern shown; individual trend can differ."

    if action == "continue":
        if current_profile == "denosumab":
            return [100.0, 101.3, 102.7, 104.8, 106.4, 107.6], "Class-specific (denosumab): larger anti-resorptive gains with on-time dosing."
        if current_profile == "zoledronate":
            return [100.0, 100.5, 101.0, 101.8, 102.3, 102.8], "Class-specific (zoledronate): stabilization with modest incremental gains."
        if current_profile == "alendronate":
            return [100.0, 100.4, 100.9, 101.6, 102.1, 102.6], "Class-specific (alendronate): gradual gains over time."
        if current_profile == "risedronate":
            return [100.0, 100.3, 100.7, 101.3, 101.8, 102.2], "Class-specific (risedronate): modest but positive BMD trajectory."
        if current_profile == "oral_bp":
            return [100.0, 100.4, 100.8, 101.5, 102.0, 102.4], "Class-specific (oral BP): expected gradual improvement."
        if current_profile == "teriparatide":
            return [100.0, 101.9, 103.7, 106.0, 107.4, 108.2], "Class-specific (teriparatide): early anabolic gain then consolidation is needed."
        if current_profile == "romosozumab":
            return [100.0, 102.6, 105.1, 107.5, 108.1, 108.3], "Class-specific (romosozumab): larger early gains, then transition planning."
        if current_profile == "raloxifene":
            return [100.0, 100.2, 100.5, 100.9, 101.2, 101.4], "Class-specific (SERM): modest anti-fracture support with small BMD increase."
        if current_profile == "other":
            return [100.0, 100.2, 100.6, 101.0, 101.4, 101.8], "Class-specific (other): trajectory shown as moderate improvement pattern."
        return [100.0, 99.8, 99.5, 99.0, 98.6, 98.2], "No active drug is recorded; this line reflects background decline risk."

    if action == "start":
        chosen = target_profile or "oral_bp"
        if chosen == "denosumab":
            return [100.0, 101.4, 102.9, 105.0, 106.6, 107.8], "Class-specific start (denosumab): robust anti-resorptive trajectory."
        if chosen == "zoledronate":
            return [100.0, 100.6, 101.2, 102.0, 102.5, 103.0], "Class-specific start (zoledronate): steady gains with annual/parenteral schedule."
        if chosen == "alendronate":
            return [100.0, 100.5, 101.0, 101.8, 102.3, 102.8], "Class-specific start (alendronate): gradual oral BP response."
        if chosen == "risedronate":
            return [100.0, 100.4, 100.9, 101.5, 102.0, 102.4], "Class-specific start (risedronate): moderate expected gains."
        if chosen == "oral_bp":
            return [100.0, 100.5, 101.0, 101.7, 102.2, 102.7], "Class-specific start (oral BP class): progressive BMD improvement."
        if chosen == "teriparatide":
            return [100.0, 102.0, 104.2, 107.0, 108.8, 109.8], "Class-specific start (teriparatide): anabolic-first rapid gain profile."
        if chosen == "romosozumab":
            return [100.0, 102.8, 105.5, 108.0, 108.7, 109.0], "Class-specific start (romosozumab): strong early gain with planned follow-on."
        if chosen == "raloxifene":
            return [100.0, 100.3, 100.6, 101.0, 101.3, 101.5], "Class-specific start (raloxifene): mild BMD increase pattern."
        return [100.0, 100.5, 101.0, 101.8, 102.4, 103.0], "Projected trend after treatment start."

    # switch
    chosen = target_profile or "denosumab"
    if current_profile in bp_profiles and chosen == "denosumab":
        return [100.0, 101.4, 103.0, 105.0, 106.6, 107.8], "Class-specific switch (BP -> denosumab): typically additional BMD gain."
    if current_profile == "alendronate" and chosen == "zoledronate":
        return [100.0, 100.1, 100.3, 100.6, 100.8, 101.0], "Class-specific switch (alendronate -> zoledronate): often limited additional BMD gain."
    if current_profile == "risedronate" and chosen == "zoledronate":
        return [100.0, 100.2, 100.4, 100.8, 101.0, 101.2], "Class-specific switch (risedronate -> zoledronate): modest additional gain."
    if current_profile in bp_profiles and chosen == "romosozumab":
        return [100.0, 101.2, 102.6, 104.5, 106.0, 106.8], "Class-specific switch (BP -> romosozumab): gains occur, potentially blunted vs treatment-naive."
    if current_profile in bp_profiles and chosen == "teriparatide":
        return [100.0, 101.0, 102.2, 103.8, 105.0, 105.8], "Class-specific switch (BP -> teriparatide): gains expected, with possible blunting."
    if current_profile == "denosumab" and chosen == "zoledronate":
        return [100.0, 99.9, 99.7, 99.4, 99.2, 99.1], "Class-specific switch (denosumab -> zoledronate): helps protect against rebound bone loss."
    if current_profile == "denosumab" and chosen in {"alendronate", "risedronate", "oral_bp"}:
        return [100.0, 99.8, 99.5, 99.1, 98.9, 98.7], "Class-specific switch (denosumab -> oral BP): partial protection from rebound loss."
    if current_profile == "denosumab" and chosen in {"teriparatide", "romosozumab"}:
        return [100.0, 98.9, 97.7, 97.1, 96.7, 96.3], "Class-specific caution (denosumab -> anabolic): transient bone loss can occur."
    if current_profile == "teriparatide" and chosen == "denosumab":
        return [100.0, 101.2, 102.4, 104.0, 105.1, 106.0], "Class-specific switch (teriparatide -> denosumab): consolidates and extends gains."
    if current_profile == "teriparatide" and chosen in {"zoledronate", "alendronate", "risedronate", "oral_bp"}:
        return [100.0, 100.8, 101.7, 103.0, 104.0, 104.7], "Class-specific switch (teriparatide -> BP): consolidation strategy for maintenance."
    if current_profile == "romosozumab" and chosen == "denosumab":
        return [100.0, 101.3, 102.6, 104.3, 105.5, 106.3], "Class-specific switch (romosozumab -> denosumab): maintains gains and lowers fracture risk."
    if current_profile == "romosozumab" and chosen in {"zoledronate", "alendronate", "risedronate", "oral_bp"}:
        return [100.0, 100.9, 101.8, 103.1, 104.0, 104.7], "Class-specific switch (romosozumab -> BP): standard anti-resorptive consolidation."
    return [100.0, 100.5, 101.2, 102.1, 102.8, 103.4], "Projected trend after treatment switch."


def build_svg_trajectory_chart(months: List[int], values: List[float], color: str) -> str:
    width = 460
    height = 190
    left = 42
    right = 12
    top = 16
    bottom = 30
    x_span = width - left - right
    y_span = height - top - bottom
    y_min = min(min(values), 98.0) - 0.5
    y_max = max(max(values), 102.0) + 0.5
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_pos(month: int) -> float:
        return left + (month / 24.0) * x_span

    def y_pos(val: float) -> float:
        return top + ((y_max - val) / (y_max - y_min)) * y_span

    pts = " ".join(f"{x_pos(m):.2f},{y_pos(v):.2f}" for m, v in zip(months, values))
    baseline_y = y_pos(100.0)
    y_ticks = [round(y_min, 1), 100.0, round(y_max, 1)]
    y_grid = "".join(
        f'<line x1="{left}" y1="{y_pos(t):.2f}" x2="{width-right}" y2="{y_pos(t):.2f}" stroke="#e2e8f0" stroke-width="1" />'
        for t in y_ticks
    )
    y_labels = "".join(
        f'<text x="{left-6}" y="{y_pos(t)+4:.2f}" text-anchor="end" font-size="10" fill="#64748b">{t:.1f}</text>'
        for t in y_ticks
    )
    x_labels = "".join(
        f'<text x="{x_pos(m):.2f}" y="{height-8}" text-anchor="middle" font-size="10" fill="#64748b">{m}m</text>'
        for m in months
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="190" role="img" aria-label="Projected BMD trajectory">'
        f'{y_grid}'
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{width-right}" y2="{baseline_y:.2f}" stroke="#94a3b8" stroke-dasharray="4 3" />'
        f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{pts}" />'
        f"{y_labels}{x_labels}"
        f'<text x="{width-6}" y="{height-8}" text-anchor="end" font-size="10" fill="#64748b">Months</text>'
        f'<text x="{left}" y="{top-4}" text-anchor="start" font-size="10" fill="#64748b">Relative BMD index (baseline=100)</text>'
        "</svg>"
    )


def build_trajectory_cards(
    assessment: OsteoAssessment,
    input_data: Optional[OsteoInput],
    agreed_plan: List[str],
) -> List[Dict[str, str]]:
    if input_data is None:
        return []
    selected_actions = extract_agreed_actions(agreed_plan)
    current_profile = infer_current_therapy_profile(input_data)
    target_profile = infer_target_therapy_profile(assessment, input_data, agreed_plan)

    scenarios = [
        ("start", "#2563eb"),
        ("continue", "#0f766e"),
        ("switch", "#7c3aed"),
        ("stop", "#b91c1c"),
    ]
    cards: List[Dict[str, str]] = []
    for action, color in scenarios:
        action_target_profile: Optional[str] = None
        if action in ["start", "switch"]:
            action_target_profile = target_profile
        values, note = _trajectory_values(current_profile, action, action_target_profile)

        if action == "start":
            title = f"After starting therapy ({therapy_profile_label(action_target_profile or target_profile)})"
        elif action == "continue":
            title = f"With continuation ({therapy_profile_label(current_profile)})"
        elif action == "switch":
            title = (
                f"After switch ({therapy_profile_label(current_profile)} -> "
                f"{therapy_profile_label(action_target_profile or target_profile)})"
            )
        else:
            title = f"After stopping therapy ({therapy_profile_label(current_profile)})"

        cards.append(
            {
                "title": title,
                "note": note,
                "svg": build_svg_trajectory_chart(HANDOUT_MONTHS, values, color),
                "active": "true" if action in selected_actions else "false",
            }
        )
    return cards


def build_patient_handout_html(
    assessment: OsteoAssessment,
    input_data: Optional[OsteoInput] = None,
    agreed_plan: Optional[List[str]] = None,
    patient_elaboration: Optional[str] = None,
) -> str:
    suggestions_html = "".join(
        f"<li>{escape(s.text)}</li>" for s in assessment.suggestions[:6]
    )
    follow_up_html = "".join(
        f"<li><strong>{escape(step.timeframe)}</strong>: {escape(step.text)}</li>"
        for step in assessment.follow_up_steps
    )
    agreed_plan_items = [p.strip() for p in (agreed_plan or []) if p and p.strip()]
    agreed_plan_html = "".join(f"<li>{escape(item)}</li>" for item in agreed_plan_items)
    trajectory_cards = build_trajectory_cards(assessment, input_data, agreed_plan_items)
    trajectory_html = ""
    if trajectory_cards:
        trajectory_html = "".join(
            (
                f'<div class="trajectory-card{" active" if c["active"] == "true" else ""}">'
                f'<h3>{escape(c["title"])}</h3>'
                f'{c["svg"]}'
                f'<p class="trajectory-note">{escape(c["note"])}</p>'
                "</div>"
            )
            for c in trajectory_cards
        )

    patient_elab_html = ""
    if patient_elaboration and patient_elaboration.strip():
        patient_elab_html = f"""
      <div class="section">
        <h2>Patient explanation</h2>
        <div class="narrative">{escape(patient_elaboration.strip())}</div>
      </div>
        """
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Osteoporosis handout</title>
      <style>
        body {{
          font-family: 'Helvetica Neue', system-ui, sans-serif;
          padding: 32px;
          color: #0f172a;
          line-height: 1.5;
        }}
        h1 {{
          margin-bottom: 4px;
          font-size: 24px;
        }}
        h2 {{
          margin-top: 24px;
          font-size: 16px;
          letter-spacing: 0.05em;
          color: #475569;
        }}
        p {{
          margin: 6px 0;
        }}
        ul {{
          padding-left: 18px;
        }}
        .section {{
          margin-bottom: 18px;
        }}
        .chip {{
          display: inline-flex;
          padding: 4px 10px;
          background: #e0f2fe;
          border-radius: 999px;
          font-size: 12px;
          color: #0c4a6e;
          margin-right: 6px;
        }}
        .narrative {{
          white-space: pre-wrap;
          background: #f8fafc;
          border: 1px solid #e2e8f0;
          border-radius: 10px;
          padding: 12px;
        }}
        .trajectory-grid {{
          display: grid;
          grid-template-columns: 1fr;
          gap: 14px;
        }}
        .trajectory-card {{
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          padding: 10px 12px;
          background: #ffffff;
        }}
        .trajectory-card.active {{
          border-color: #2563eb;
          box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12);
          background: #f8fbff;
        }}
        .trajectory-card h3 {{
          margin: 0 0 8px 0;
          font-size: 14px;
          color: #0f172a;
        }}
        .trajectory-note {{
          margin: 8px 0 0 0;
          color: #475569;
          font-size: 12px;
        }}
        .trajectory-footnote {{
          margin-top: 8px;
          color: #64748b;
          font-size: 12px;
        }}
      </style>
    </head>
    <body>
      <h1>Osteoporosis handout</h1>
      <p>This summary is meant for discussion and not as a prescription.</p>
      <div class="section">
        <h2>Risk snapshot</h2>
        <div class="chip">{assessment.risk_category.value.upper()} risk</div>
        {"".join(f'<div class="chip">{escape(reason)}</div>' for reason in assessment.risk_reasons[:3])}
      </div>
      {patient_elab_html}
      <div class="section">
        <h2>Key guidance</h2>
        <ul>{suggestions_html or "<li>No suggestions recorded.</li>"}</ul>
      </div>
      <div class="section">
        <h2>Follow-up plan</h2>
        <ul>{follow_up_html or "<li>No specific follow-up steps yet.</li>"}</ul>
      </div>
      <div class="section">
        <h2>Bone mass trend (next 24 months)</h2>
        <div class="trajectory-grid">
          {trajectory_html or '<p>No treatment trajectory available yet (load/update patient treatment data first).</p>'}
        </div>
        <p class="trajectory-footnote">
          These curves are educational, not patient-specific predictions. They summarize typical directional patterns from sequential-therapy literature and should be interpreted by the treating clinician.
        </p>
      </div>
      <div class="section">
        <h2>Agreed plan</h2>
        <ul>{agreed_plan_html or "<li>No agreed plan items selected yet.</li>"}</ul>
      </div>
      <div class="section">
        <h2>What to share with your doctor</h2>
        <p>Bring lab results (calcium, vitamin D, renal function) and any notes about symptoms/falls.</p>
      </div>
    </body>
    </html>
    """
    return html.strip()

class ElaborationRequest(BaseModel):
    assessment: OsteoAssessment
    audience: str = Field(default="clinician", description="clinician or patient")


class ElaborationResponse(BaseModel):
    elaborated_text: str

class TreatmentRecommendationRequest(BaseModel):
    assessment: OsteoStoredAssessment


class TreatmentRecommendationResponse(BaseModel):
    treatment_recommendation: str

class LiteratureQuestionRequest(BaseModel):
    assessment: OsteoStoredAssessment
    question: str

class LiteratureQuestionResponse(BaseModel):
    answer: str
class OsteoHistoryEntry(BaseModel):
    assessment_id: str
    patient_id: str
    created_at: datetime
    input_data: OsteoInput
    assessment: OsteoAssessment


class TrendSummary(BaseModel):
    bmd_trend_text: Optional[str]
    ctx_trend_text: Optional[str]
    p1np_trend_text: Optional[str]
    # add p1np_trend_text, etc. as needed


class UpdateAssessmentRequest(BaseModel):
    patient_id: Optional[str] = None
    input_data: OsteoInput


class TherapyComparisonRequest(BaseModel):
    current_therapy: Optional[CurrentTherapyType] = None
    from_therapy: Optional[CurrentTherapyType] = None
    to_therapy: Optional[CurrentTherapyType] = None
    limit: int = Field(default=200, ge=10, le=2000)


class TherapyComparisonRow(BaseModel):
    patient_id: str
    assessment_id: str
    created_at: datetime
    risk_category: str
    current_therapy: str
    previous_therapy: Optional[str] = None
    transition: Optional[str] = None


class TherapyComparisonResponse(BaseModel):
    total_latest_patients: int
    matched_patients: int
    filters: Dict[str, Optional[str]]
    by_risk: Dict[str, int]
    by_current_therapy: Dict[str, int]
    transitions: Dict[str, int]
    rows: List[TherapyComparisonRow]


class DeleteAssessmentResponse(BaseModel):
    deleted: bool
    assessment_id: str
    message: str

# =========================
# Helper calculations
# =========================


def calculate_bmi(weight_kg: Optional[float], height_cm: Optional[float]) -> Optional[float]:
    if weight_kg is None or height_cm is None or height_cm <= 0:
        return None
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def compute_internal_frax_like_index(data: OsteoInput) -> Tuple[Optional[float], Optional[str]]:
    score = 0.0
    reasons: List[str] = []

    # Age
    if data.age >= 80:
        score += 4
        reasons.append("age ≥80")
    elif data.age >= 70:
        score += 3
        reasons.append("age 70–79")
    elif data.age >= 60:
        score += 2
        reasons.append("age 60–69")
    elif data.age >= 50:
        score += 1
        reasons.append("age 50–59")

    # BMI
    bmi = calculate_bmi(data.weight_kg, data.height_cm)
    if bmi is not None:
        if bmi < 18.5:
            score += 2
            reasons.append("BMI <18.5")
        elif 18.5 <= bmi < 20.0:
            score += 1
            reasons.append("BMI 18.5–19.9")

    # Clinical risk factors
    if data.parental_hip_fracture:
        score += 1
        reasons.append("parental hip fracture")
    if data.current_smoker:
        score += 1
        reasons.append("current smoking")
    if data.glucocorticoids:
        score += 1
        reasons.append("glucocorticoid use")
    if data.rheumatoid_arthritis:
        score += 1
        reasons.append("rheumatoid arthritis")
    if data.secondary_osteoporosis:
        score += 1
        reasons.append("secondary osteoporosis")
    if data.high_alcohol_intake:
        score += 1
        reasons.append("high alcohol intake")

    if score == 0.0 and not reasons:
        return None, None

    note = (
        "Internal fracture risk index (not official FRAX) based on: "
        + ", ".join(reasons)
        + ". Higher score indicates more clinical risk factors."
    )
    return score, note


def compute_morse_fall_risk(data: OsteoInput) -> Tuple[Optional[int], Optional[str], bool]:
    """
    Morse Fall Scale approximation:
      history of falling (25), secondary diagnosis (15), ambulatory aid (0/15/30),
      IV/heparin lock (20), gait (0/10/20), mental status (0/15)
    """
    has_non_default_component = (
        data.morse_history_of_falling_3m
        or data.morse_secondary_diagnosis
        or data.morse_iv_or_heparin_lock
        or data.morse_ambulatory_aid != MorseAmbulatoryAid.none_bedrest_assist
        or data.morse_gait != MorseGait.normal_bedrest_wheelchair
        or data.morse_mental_status != MorseMentalStatus.aware_of_limitations
    )
    if not has_non_default_component:
        return None, None, False

    score = 0
    if data.morse_history_of_falling_3m:
        score += 25
    if data.morse_secondary_diagnosis:
        score += 15
    if data.morse_ambulatory_aid == MorseAmbulatoryAid.cane_walker_crutches:
        score += 15
    elif data.morse_ambulatory_aid == MorseAmbulatoryAid.furniture:
        score += 30
    if data.morse_iv_or_heparin_lock:
        score += 20
    if data.morse_gait == MorseGait.weak:
        score += 10
    elif data.morse_gait == MorseGait.impaired:
        score += 20
    if data.morse_mental_status == MorseMentalStatus.overestimates_or_forgets_limitations:
        score += 15

    if score >= 45:
        band = "high"
        high_flag = True
    elif score >= 25:
        band = "moderate"
        high_flag = False
    else:
        band = "low"
        high_flag = False

    note = f"Morse fall risk score {score} ({band} risk)."
    return score, note, high_flag


def has_effective_high_falls_risk(data: OsteoInput) -> bool:
    _, _, morse_high = compute_morse_fall_risk(data)
    return data.high_falls_risk or morse_high


def has_effective_frailty(data: OsteoInput) -> bool:
    return data.frailty or ((data.cfs_score or 0) >= 5)


EVIDENCE_REGISTRY: Dict[str, str] = {
    "KANIS_2020": "Kanis et al. Osteoporos Int 2020;31:1-12 (IOF/ESCEO algorithm and risk stratification).",
    "DIAB_WATTS_2013": "Diab & Watts. Ther Adv Musculoskelet Dis 2013;5:107-111 (DOI: 10.1177/1759720X13477714).",
    "FLEX_2006": "Black et al. JAMA 2006;296:2927-2938 (FLEX alendronate extension).",
    "VERT_NA_2008": "Watts et al. Osteoporos Int 2008;19:365-372 (VERT-NA extension, risedronate off-treatment).",
    "HORIZON_EXT_2012": "Black et al. J Bone Miner Res 2012;27:243-254 (HORIZON extension, 3 vs 6 years zoledronate).",
    "FREEDOM_2017": "Bone et al. Lancet Diabetes Endocrinol 2017;5:513-523 (FREEDOM extension long-term denosumab).",
    "ARCH_2017": "Saag et al. N Engl J Med 2017;377:1417-1427 (romosozumab-to-alendronate vs alendronate).",
    "VERO_2018": "Kendler et al. Lancet 2018;391:230-240 (teriparatide vs risedronate in severe osteoporosis).",
    "FRAME_2016": "Cosman et al. N Engl J Med 2016;375:1532-1543 (romosozumab then denosumab).",
    "DATA_SWITCH_2015": "Leder et al. Lancet 2015;386:1147-1155 (teriparatide/denosumab sequencing and BMD).",
}


def attach_evidence_to_suggestions(suggestions: List[Suggestion]) -> None:
    """
    Attach evidence IDs/references to suggestions so each decision can be
    traced to major studies/guidelines in UI output.
    """
    for s in suggestions:
        ids = set(s.evidence_ids or [])
        t = s.text.lower()

        if s.category == "conference_protocol" or "iof/esceo" in t or "kanis" in t:
            ids.add("KANIS_2020")
        if "holiday" in t or "discontinu" in t:
            ids.update({"DIAB_WATTS_2013", "FLEX_2006", "VERT_NA_2008", "HORIZON_EXT_2012"})
        if "denosumab" in t and ("interruption" in t or "rebound" in t or "stop" in t):
            ids.add("FREEDOM_2017")
        if "anabolic-first" in t or "very-high-risk pathway" in t:
            ids.update({"ARCH_2017", "VERO_2018"})
        if "romosozumab" in t and "denosumab" in t:
            ids.add("FRAME_2016")
        if "transition" in t or "sequential" in t:
            ids.add("DATA_SWITCH_2015")

        ordered_ids = [eid for eid in EVIDENCE_REGISTRY.keys() if eid in ids]
        s.evidence_ids = ordered_ids
        s.evidence_refs = [EVIDENCE_REGISTRY[eid] for eid in ordered_ids]


def get_kanis_major_thresholds_by_age(age: int) -> Tuple[float, float]:
    """
    Kanis et al. Osteoporos Int 2020;31:1-12 (UK FRAX appendix):
    intervention threshold (IT) for major osteoporotic fracture by age band,
    and upper assessment threshold (UAT) ~= 1.2 * IT.
    """
    # Age band cutoffs from the appendix table in the position paper.
    if age <= 54:
        it = 7.8
    elif age <= 59:
        it = 11.0
    elif age <= 64:
        it = 14.0
    elif age <= 69:
        it = 19.0
    elif age <= 74:
        it = 22.0
    elif age <= 79:
        it = 26.0
    elif age <= 84:
        it = 31.0
    else:
        it = 33.0
    uat = round(it * 1.2, 1)
    return it, uat


def calculate_calcium_intake(data: OsteoInput) -> Tuple[Optional[float], Optional[str]]:
    MG_PER_PORTION_MILK = 300.0
    MG_PER_PORTION_YOGURT = 250.0
    MG_PER_PORTION_CHEESE = 200.0
    MG_PER_PORTION_LEAFY = 100.0
    MG_PER_PORTION_FORTIFIED = 150.0
    MG_PER_PORTION_OTHER_DAIRY = 150.0

    dietary_mg = 0.0
    dietary_mg += data.milk_portions_per_day * MG_PER_PORTION_MILK
    dietary_mg += data.yogurt_portions_per_day * MG_PER_PORTION_YOGURT
    dietary_mg += data.cheese_portions_per_day * MG_PER_PORTION_CHEESE
    dietary_mg += data.leafy_greens_portions_per_day * MG_PER_PORTION_LEAFY
    dietary_mg += data.fortified_food_portions_per_day * MG_PER_PORTION_FORTIFIED
    dietary_mg += data.other_dairy_portions_per_day * MG_PER_PORTION_OTHER_DAIRY

    supplement_mg = data.calcium_supplement_mg_per_day or 0.0
    total_mg = dietary_mg + supplement_mg

    if total_mg == 0.0:
        return None, None

    if total_mg < 800:
        detail = (
            f"Estimated total calcium intake ≈ {total_mg:.0f} mg/day, which is below "
            "common targets (often around 1000–1200 mg/day in many guidelines)."
        )
    elif 800 <= total_mg <= 1500:
        detail = (
            f"Estimated total calcium intake ≈ {total_mg:.0f} mg/day, which is within a "
            "broad range often considered acceptable for many adults, depending on age "
            "and comorbidities."
        )
    else:
        detail = (
            f"Estimated total calcium intake ≈ {total_mg:.0f} mg/day, which may be above "
            "usual targets for many patients; consider whether such intake is necessary, "
            "especially in those at risk of hypercalcemia or kidney stones."
        )

    note = detail + " This is a rough estimate based on reported portions and supplement dose."
    return total_mg, note


def bmd_trend_from_history(history: List[OsteoHistoryEntry]) -> Optional[str]:
    """
    Simple BMD trend description based on first vs last visit.
    """
    if len(history) < 2:
        return None

    first = history[0].input_data
    last = history[-1].input_data

    parts = []

    if first.spine_t_score is not None and last.spine_t_score is not None:
        delta = last.spine_t_score - first.spine_t_score
        if abs(delta) >= 0.3:
            direction = "improved" if delta > 0 else "worsened"
            parts.append(
                f"Lumbar spine T-score has {direction} from {first.spine_t_score:.2f} "
                f"to {last.spine_t_score:.2f} over the recorded interval."
            )

    if first.total_hip_t_score is not None and last.total_hip_t_score is not None:
        delta = last.total_hip_t_score - first.total_hip_t_score
        if abs(delta) >= 0.3:
            direction = "improved" if delta > 0 else "worsened"
            parts.append(
                f"Total hip T-score has {direction} from {first.total_hip_t_score:.2f} "
                f"to {last.total_hip_t_score:.2f}."
            )

    if first.femoral_neck_t_score is not None and last.femoral_neck_t_score is not None:
        delta = last.femoral_neck_t_score - first.femoral_neck_t_score
        if abs(delta) >= 0.3:
            direction = "improved" if delta > 0 else "worsened"
            parts.append(
                f"Femoral neck T-score has {direction} from {first.femoral_neck_t_score:.2f} "
                f"to {last.femoral_neck_t_score:.2f}."
            )

    if not parts:
        return None
    return " ".join(parts)


def ctx_trend_from_history(history: List[OsteoHistoryEntry]) -> Optional[str]:
    """
    CTX trend based on first vs last available CTX values.
    """
    values = [
        (h.created_at, h.input_data.ctx_ng_ml)
        for h in history
        if h.input_data.ctx_ng_ml is not None
    ]
    if len(values) < 2:
        return None
    values.sort(key=lambda x: x[0])
    t0, v0 = values[0]
    t1, v1 = values[-1]
    if v0 is None or v0 <= 0:
        return None
    change = (v1 - v0) / v0 * 100.0
    if change <= -50:
        return (
            f"CTX has fallen from ~{v0:.2f} to ~{v1:.2f} ng/mL (~{change:.0f}% change) across "
            "recorded measurements, compatible with strong anti-resorptive effect when "
            "interpreted with lab-specific reference ranges."
        )
    if change >= 50:
        return (
            f"CTX has risen from ~{v0:.2f} to ~{v1:.2f} ng/mL (~+{change:.0f}% change); this may "
            "reflect reduced effect or treatment interruption and should be interpreted in the "
            "context of dosing, timing and adherence."
        )
    return None


def p1np_trend_from_history(history: List[OsteoHistoryEntry]) -> Optional[str]:
    """
    P1NP trend for anabolic therapy monitoring.
    """
    values = [
        (h.created_at, h.input_data.p1np_ng_ml)
        for h in history
        if h.input_data.p1np_ng_ml is not None
    ]
    if len(values) < 2:
        return None
    values.sort(key=lambda x: x[0])
    t0, v0 = values[0]
    t1, v1 = values[-1]
    if v0 is None or v0 <= 0:
        return None
    change = (v1 - v0) / v0 * 100.0
    if change >= 50:
        return (
            f"P1NP has risen from ~{v0:.1f} to ~{v1:.1f} ng/mL (~+{change:.0f}% change). "
            "In the context of anabolic therapy, such an increase is often compatible "
            "with the expected pharmacodynamic response, subject to lab reference ranges "
            "and clinical judgement."
        )
    if change < 20:
        return (
            f"P1NP change from ~{v0:.1f} to ~{v1:.1f} ng/mL appears modest. If this reflects "
            "true low change and not lab variation, it may suggest a blunted anabolic response "
            "and could prompt closer review of dosing, adherence, and secondary factors."
        )
    return None

# =========================
# Hyperparathyroidism helper
# =========================


def add_hyperparathyroid_suggestions(data: OsteoInput, suggestions: List[Suggestion]) -> None:
    ca = data.serum_calcium_mg_dl or data.serum_calcium
    pth = data.pth_pg_ml
    vitd = data.vitamin_d_25oh_ng_ml or data.vitamin_d_25oh
    phos = data.serum_phosphorus_mg_dl
    uca = data.urine_calcium_24h_mg

    if ca is None or pth is None:
        return

    high_ca = ca > 10.5
    low_ca = ca < 8.5
    high_pth = pth > 65
    low_pth = pth < 15

    # Primary hyperparathyroidism pattern
    if high_ca and high_pth:
        text = (
            f"Serum calcium {ca:.2f} mg/dL with elevated PTH {pth:.1f} pg/mL. "
            "This pattern can be seen in primary hyperparathyroidism or related disorders. "
        )
        if phos is not None and phos < 2.5:
            text += f"Phosphorus is low ({phos:.2f} mg/dL), which may further support this pattern. "
        if vitd is not None and vitd < 20:
            text += (
                f"Vitamin D is low ({vitd:.1f} ng/mL); correcting vitamin D while "
                "monitoring calcium and PTH may be reasonable before final "
                "conclusions. "
            )
        suggestions.append(
            Suggestion(
                category="labs_hyperparathyroidism",
                text=text + "Correlate with repeat labs, renal function, and imaging as needed.",
            )
        )
        return

    # Secondary hyperparathyroidism (e.g. Vit D deficiency)
    if not high_ca and high_pth and vitd is not None and vitd < 20:
        text = (
            f"PTH {pth:.1f} pg/mL with calcium {ca:.2f} mg/dL and low vitamin D "
            f"{vitd:.1f} ng/mL. This combination can be compatible with secondary "
            "hyperparathyroidism (e.g. vitamin D deficiency or CKD). "
        )
        if phos is not None:
            text += f"Phosphorus {phos:.2f} mg/dL; interpret with renal function and PTH trends. "
        suggestions.append(
            Suggestion(
                category="labs_hyperparathyroidism",
                text=text + "Consider appropriate workup and vitamin D repletion according to guidelines.",
            )
        )
        return

    # Hypocalcemia with low PTH
    if low_ca and low_pth:
        suggestions.append(
            Suggestion(
                category="labs_hyperparathyroidism",
                text=(
                    f"Hypocalcemia ({ca:.2f} mg/dL) with low PTH ({pth:.1f} pg/mL). "
                    "This can be compatible with hypoparathyroidism. Correlate with "
                    "clinical picture, magnesium, phosphorus, and drug history."
                ),
            )
        )

    # 24h urine calcium commentary
    if uca is not None:
        if uca < 100:
            suggestions.append(
                Suggestion(
                    category="urine_calcium",
                    text=(
                        f"24-hour urine calcium is {uca:.0f} mg/24h, which is relatively low; "
                        "interpret in context (dietary calcium, vitamin D, and possible familial "
                        "hypocalciuric hypercalcemia if hypercalcemia is present)."
                    ),
                )
            )
        elif uca > 300:
            suggestions.append(
                Suggestion(
                    category="urine_calcium",
                    text=(
                        f"24-hour urine calcium is {uca:.0f} mg/24h, on the higher side; "
                        "this can contribute to nephrolithiasis risk and may be relevant in "
                        "the context of hypercalcemia or high-dose calcium/vitamin D."
                    ),
                )
            )

# =========================
# Current therapy helper
# =========================


def add_current_therapy_suggestions(
    data: OsteoInput,
    risk: RiskCategory,
    suggestions: List[Suggestion],
) -> None:
    ttype = data.current_therapy_type
    dur = data.current_therapy_duration_years
    fx_on_tx = data.fractures_during_current_therapy
    adr = data.significant_therapy_adverse_effects

    if ttype == CurrentTherapyType.none:
        return

    if fx_on_tx:
        suggestions.append(
            Suggestion(
                category="current_therapy",
                text=(
                    "A new fragility fracture has occurred while on the current osteoporosis "
                    "therapy. This may represent suboptimal response, adherence issues, or "
                    "evolving risk profile and generally warrants review of the treatment "
                    "strategy, including verifying adherence, secondary causes, and "
                    "considering alternative or intensified therapy according to guidelines."
                ),
            )
        )
        suggestions.append(
            Suggestion(
                category="current_therapy",
                text=(
                    "Pragmatic treatment-failure framing: >=2 fragility fractures after >=1 year "
                    "on therapy, or 1 fracture with poor BMD/marker response, is commonly treated as "
                    "clear inadequate response and usually prompts escalation/switch."
                ),
            )
        )

    if adr:
        suggestions.append(
            Suggestion(
                category="current_therapy",
                text=(
                    "Clinically significant adverse effects related to the current therapy "
                    "have been reported. Balancing fracture risk reduction against adverse "
                    "effects may justify discussion of dose adjustment, switching drug "
                    "class, or discontinuation according to guidelines and patient preference."
                ),
            )
        )

    if ttype in [CurrentTherapyType.oral_bisphosphonate, CurrentTherapyType.iv_bisphosphonate]:
        suggestions.append(
            Suggestion(
                category="current_therapy",
                text=(
                    "Drug-holiday reference: Diab & Watts, Ther Adv Musculoskelet Dis 2013 "
                    "(DOI: 10.1177/1759720X13477714). Suggested holiday strategy is risk-tailored, "
                    "with periodic reassessment using BMD/fracture events."
                ),
            )
        )
        if dur is not None:
            if dur < 3:
                suggestions.append(
                    Suggestion(
                        category="current_therapy",
                        text=(
                            f"Currently on bisphosphonate therapy for ~{dur:.1f} years. For many "
                            "patients, a 3–5 year course is typical, with longer durations considered "
                            "in very-high-risk cases. Ongoing need should be reassessed periodically."
                        ),
                    )
                )
            elif 3 <= dur <= 5:
                suggestions.append(
                    Suggestion(
                        category="current_therapy",
                        text=(
                            f"Bisphosphonate therapy duration is ~{dur:.1f} years. For patients at "
                            "lower risk, a treatment holiday may be considered after 3–5 years, "
                            "whereas in high or very-high-risk cases, continuation or a change "
                            "of agent may be appropriate. Decisions should be individualized."
                        ),
                    )
                )
                if risk in [RiskCategory.low, RiskCategory.moderate]:
                    suggestions.append(
                        Suggestion(
                            category="current_therapy",
                            text=(
                                "Practical holiday option for lower-to-moderate risk: consider pause "
                                "after ~3–5 years and monitor until significant BMD loss (beyond LSC) "
                                "or a new fragility fracture."
                            ),
                        )
                    )
            else:
                suggestions.append(
                    Suggestion(
                        category="current_therapy",
                        text=(
                            f"Bisphosphonate therapy duration exceeds 5 years (~{dur:.1f} years). "
                            "This is often a point where treatment holiday, continuation, or "
                            "alternative therapy is actively reconsidered in light of current "
                            "fracture risk, BMD trends, and any adverse events."
                        ),
                    )
                )
                if risk == RiskCategory.moderate:
                    suggestions.append(
                        Suggestion(
                            category="current_therapy",
                            text=(
                                "For moderate-risk profiles after longer exposure, Diab & Watts suggest "
                                "a potential holiday window around 3–5 years, with restart if BMD declines "
                                "significantly or fracture occurs."
                            ),
                        )
                    )
                if risk in [RiskCategory.high, RiskCategory.very_high]:
                    suggestions.append(
                        Suggestion(
                            category="current_therapy",
                            text=(
                                "For high/very-high-risk profiles, prolonged treatment (often up to ~10 years) "
                                "is commonly favored; if a holiday is considered, keep it short (about 1–2 years) "
                                "with close follow-up and consideration of non-bisphosphonate bridging."
                            ),
                        )
                    )
        else:
            suggestions.append(
                Suggestion(
                    category="current_therapy",
                    text=(
                        "Bisphosphonate therapy is ongoing; document approximate duration to "
                        "better frame decisions around continuation versus holiday."
                    ),
                )
            )

    elif ttype == CurrentTherapyType.denosumab:
        suggestions.append(
            Suggestion(
                category="current_therapy",
                text=(
                    "Patient is currently on denosumab. Abrupt discontinuation is associated "
                    "with rebound bone turnover and increased vertebral fracture risk; any "
                    "decision to stop or switch therapy should include a plan for subsequent "
                    "anti-resorptive coverage according to current guidelines."
                ),
            )
        )

    elif ttype == CurrentTherapyType.teriparatide:
        suggestions.append(
            Suggestion(
                category="current_therapy",
                text=(
                    "Patient is on or has been on anabolic therapy (e.g. teriparatide). "
                    "Course duration is typically limited (often up to 18–24 months) and is "
                    "usually followed by an anti-resorptive agent to consolidate gains in BMD."
                ),
            )
        )

    elif ttype == CurrentTherapyType.romosozumab:
        suggestions.append(
            Suggestion(
                category="current_therapy",
                text=(
                    "Patient is on or has been on romosozumab. Treatment duration is usually "
                    "limited (e.g. 12 months) and is typically followed by an anti-resorptive "
                    "agent to maintain BMD improvements."
                ),
            )
        )

    elif ttype == CurrentTherapyType.raloxifene:
        suggestions.append(
            Suggestion(
                category="current_therapy",
                text=(
                    "Patient is on raloxifene. This agent primarily reduces vertebral fracture "
                    "risk and may be best suited to specific risk profiles; reassess whether "
                    "it remains the most appropriate choice given current fracture risk and "
                    "co-morbidities."
                ),
            )
        )

    elif ttype == CurrentTherapyType.other:
        suggestions.append(
            Suggestion(
                category="current_therapy",
                text=(
                    "Current osteoporosis therapy is recorded as 'other'. Ensure that the "
                    "specific agent, dose, and duration are documented to support decisions "
                    "about continuation or modification of therapy."
                ),
            )
        )
def add_therapy_history_suggestions(
    data: OsteoInput,
    suggestions: List[Suggestion],
) -> None:
    history = data.therapy_history
    if not history:
        return

    # Summarise total exposure per therapy class
    totals: Dict[CurrentTherapyType, float] = {}
    for ep in history:
        if ep.duration_years is not None:
            totals[ep.therapy_type] = totals.get(ep.therapy_type, 0.0) + ep.duration_years

    parts = []
    for ttype, years in totals.items():
        if ttype == CurrentTherapyType.none:
            continue
        if years >= 1.0:
            parts.append(f"{ttype.value.replace('_', ' ')} ~{years:.1f} years")

    if parts:
        suggestions.append(
            Suggestion(
                category="therapy_history",
                text=(
                    "Cumulative osteoporosis pharmacotherapy history includes: "
                    + "; ".join(parts)
                    + ". This overall exposure may influence decisions about future "
                    "drug choice, duration, and treatment holidays."
                ),
            )
        )

    # Denosumab episodes: warn about rebound if any
    had_denosumab = any(ep.therapy_type == CurrentTherapyType.denosumab for ep in history)
    if had_denosumab:
        suggestions.append(
            Suggestion(
                category="therapy_history",
                text=(
                    "Patient has previously been on denosumab. Past or future interruptions "
                    "in denosumab require careful planning of subsequent anti-resorptive "
                    "therapy to mitigate rebound bone turnover and vertebral fracture risk."
                ),
            )
        )

    # Anabolic episodes (teriparatide/romosozumab): consolidation after course
    had_anabolic = any(
        ep.therapy_type in [CurrentTherapyType.teriparatide, CurrentTherapyType.romosozumab]
        for ep in history
    )
    if had_anabolic:
        suggestions.append(
            Suggestion(
                category="therapy_history",
                text=(
                    "History of anabolic therapy (e.g. teriparatide/romosozumab). "
                    "Ensuring adequate consolidation with an anti-resorptive agent "
                    "after completion of anabolic courses is important for maintaining BMD gains."
                ),
            )
        )

    # Recognise holidays
    had_holiday = any(ep.is_holiday for ep in history)
    if had_holiday:
        suggestions.append(
            Suggestion(
                category="therapy_history",
                text=(
                    "At least one treatment holiday is recorded. The timing and duration "
                    "of holidays in relation to overall fracture risk and BMD trends should "
                    "be periodically re-evaluated."
                ),
            )
        )


def determine_conference_risk_tier(data: OsteoInput) -> str:
    """
    Conference-inspired 3-tier framing used for sequential-therapy suggestions:
    low / high / very_high.
    """
    t_scores = [
        ts
        for ts in [
            data.spine_t_score,
            data.total_hip_t_score,
            data.femoral_neck_t_score,
        ]
        if ts is not None
    ]
    min_t = min(t_scores) if t_scores else None

    has_major_fx = (
        FractureType.hip in data.prior_fragility_fractures
        or FractureType.vertebral in data.prior_fragility_fractures
    )
    vertebral_count = max(
        data.vertebral_fracture_count,
        1 if FractureType.vertebral in data.prior_fragility_fractures else 0,
    )
    hip_count = max(
        data.hip_fracture_count,
        1 if FractureType.hip in data.prior_fragility_fractures else 0,
    )
    multiple_fx = (
        len(data.prior_fragility_fractures) >= 2
        or vertebral_count >= 2
        or hip_count >= 2
    )
    frax_major = data.frax_major_osteoporotic or 0.0
    frax_hip = data.frax_hip or 0.0
    very_high_frax = frax_major >= 30.0 or frax_hip >= 4.5
    kanis_it, kanis_uat = get_kanis_major_thresholds_by_age(data.age)
    kanis_very_high = data.frax_major_osteoporotic is not None and frax_major >= kanis_uat
    kanis_high = data.frax_major_osteoporotic is not None and frax_major >= kanis_it
    effective_high_falls_risk = has_effective_high_falls_risk(data)
    effective_frailty = has_effective_frailty(data)
    low_bmd_high_burden = (
        min_t is not None
        and min_t <= -2.5
        and data.age >= 75
        and (effective_frailty or data.glucocorticoids or effective_high_falls_risk)
    )

    if (
        has_major_fx
        or multiple_fx
        or (data.recent_fragility_fracture and min_t is not None and min_t <= -2.5)
        or data.fractures_during_current_therapy
        or (min_t is not None and min_t <= -3.0)
        or very_high_frax
        or kanis_very_high
        or low_bmd_high_burden
        or (effective_high_falls_risk and (data.dementia_or_cognitive_impairment or data.significant_immobility))
    ):
        return "very_high"

    if (min_t is not None and min_t <= -2.5) or frax_major >= 20.0 or frax_hip >= 3.0 or kanis_high:
        return "high"

    return "low"


def add_conference_protocol_suggestions(data: OsteoInput, suggestions: List[Suggestion]) -> None:
    """
    Add explicit conference-derived treatment/sequencing guidance.
    """
    tier = determine_conference_risk_tier(data)
    kanis_it, kanis_uat = get_kanis_major_thresholds_by_age(data.age)

    suggestions.append(
        Suggestion(
            category="conference_protocol",
            text=(
                "Conference protocol tier (sequential-therapy framework): "
                f"{tier.replace('_', ' ').upper()}."
            ),
        )
    )
    suggestions.append(
        Suggestion(
            category="conference_protocol",
            text=(
                "Algorithm anchor: Kanis et al, Osteoporos Int 2020;31:1-12 "
                "(IOF/ESCEO position paper). For age "
                f"{data.age}, FRAX major IT ~{kanis_it:.1f}% and UAT ~{kanis_uat:.1f}% "
                "(country calibration may differ)."
            ),
        )
    )

    if tier == "low":
        suggestions.append(
            Suggestion(
                category="conference_protocol",
                text=(
                    "Low-risk pathway: optimize calcium/vitamin D, prescribe risk-appropriate exercise "
                    "and falls prevention, lifestyle reassurance; SERM/MHT may be considered in selected cases."
                ),
            )
        )
    elif tier == "high":
        suggestions.append(
            Suggestion(
                category="conference_protocol",
                text=(
                    "High-risk pathway: anti-resorptive-first strategy is commonly favored "
                    "(e.g. denosumab or zoledronate, or oral bisphosphonate depending context)."
                ),
            )
        )
    else:
        suggestions.append(
            Suggestion(
                category="conference_protocol",
                text=(
                    "Very-high-risk pathway: anabolic-first strategy (e.g. teriparatide or "
                    "romosozumab/Evenity) followed by anti-resorptive consolidation."
                ),
            )
        )
    if data.recent_fragility_fracture:
        suggestions.append(
            Suggestion(
                category="conference_protocol",
                text=(
                    "Recent fracture supports an imminent-risk framing; early treatment action is emphasized "
                    "in the IOF/ESCEO algorithmic approach."
                ),
            )
        )

    # Treatment-failure framing from IOF-style definitions in the slides.
    if data.fractures_during_current_therapy:
        suggestions.append(
            Suggestion(
                category="conference_protocol",
                text=(
                    "Fracture on therapy indicates at least inadequate clinical response and requires "
                    "secondary-cause/adherence reassessment. Pragmatic failure definition often used: "
                    ">=2 fractures after >=1 year on therapy, or 1 fracture plus insufficient BMD/BTM response."
                ),
            )
        )

    # Transition rules emphasized in the slide set.
    suggestions.append(
        Suggestion(
            category="conference_protocol",
            text=(
                "Sequential transition notes for BMD trajectory: BP->denosumab usually yields further BMD gains; "
                "alendronate->zoledronate alone often gives limited additional BMD gain; "
                "BP->anabolic may show some blunting of anabolic gains."
            ),
        )
    )
    suggestions.append(
        Suggestion(
            category="conference_protocol",
            text=(
                "After teriparatide/anabolic course, transition to denosumab or bisphosphonate is associated "
                "with additional BMD gain and lower fracture risk versus stopping therapy."
            ),
        )
    )
    suggestions.append(
        Suggestion(
            category="conference_protocol",
            text=(
                "After denosumab, transition to bisphosphonate (oral or IV such as zoledronate) helps partially "
                "or fully protect against rebound bone loss; direct denosumab->teriparatide sequence may trigger "
                "accelerated turnover and transient BMD loss."
            ),
        )
    )
    suggestions.append(
        Suggestion(
            category="conference_protocol",
            text=(
                "Maintenance concept after initial sequence: intermittent bisphosphonate strategy "
                "(e.g. time-limited oral BP blocks and/or single zoledronate doses spaced over years) "
                "with periodic reassessment of risk, BMD and fracture events."
            ),
        )
    )


# =========================
# Risk and suggestions
# =========================


def determine_risk_category(
    data: OsteoInput,
    internal_index: Optional[float],
) -> Tuple[RiskCategory, List[str]]:
    reasons: List[str] = []
    effective_high_falls_risk = has_effective_high_falls_risk(data)
    effective_frailty = has_effective_frailty(data)

    has_hip = FractureType.hip in data.prior_fragility_fractures
    has_vertebral = FractureType.vertebral in data.prior_fragility_fractures
    vertebral_count = max(data.vertebral_fracture_count, 1 if has_vertebral else 0)
    hip_count = max(data.hip_fracture_count, 1 if has_hip else 0)

    t_scores = [
        ts
        for ts in [
            data.spine_t_score,
            data.total_hip_t_score,
            data.femoral_neck_t_score,
        ]
        if ts is not None
    ]
    min_t = min(t_scores) if t_scores else None

    frax_major = data.frax_major_osteoporotic or 0.0
    frax_hip = data.frax_hip or 0.0
    frax_major_high = frax_major >= 20.0
    frax_hip_high = frax_hip >= 3.0

    internal_index_high = internal_index is not None and internal_index >= 6.0
    very_high_frax = frax_major >= 30.0 or frax_hip >= 4.5
    kanis_it, kanis_uat = get_kanis_major_thresholds_by_age(data.age)
    kanis_very_high_frax = data.frax_major_osteoporotic is not None and frax_major >= kanis_uat
    kanis_high_frax = data.frax_major_osteoporotic is not None and frax_major >= kanis_it
    low_bmd_plus_clinical_burden = (
        min_t is not None
        and min_t <= -2.5
        and data.age >= 75
        and (effective_frailty or data.glucocorticoids or effective_high_falls_risk)
    )
    imminent_clinical_pattern = (
        data.fractures_during_current_therapy
        or (data.history_of_falls_last_year and effective_high_falls_risk)
    )

    if has_hip or has_vertebral:
        reasons.append("History of hip or vertebral fragility fracture.")
        return RiskCategory.very_high, reasons

    if vertebral_count >= 2:
        reasons.append(f"Multiple vertebral fragility fractures reported (n={vertebral_count}).")
        return RiskCategory.very_high, reasons

    if hip_count >= 2:
        reasons.append(f"Multiple hip fragility fractures reported (n={hip_count}).")
        return RiskCategory.very_high, reasons

    if len(data.prior_fragility_fractures) >= 2:
        reasons.append("Multiple prior fragility fractures.")
        return RiskCategory.very_high, reasons

    if data.recent_fragility_fracture and min_t is not None and min_t <= -2.5:
        reasons.append(
            f"Recent fragility fracture pattern with low BMD (minimum T-score {min_t:.2f})."
        )
        return RiskCategory.very_high, reasons

    if effective_high_falls_risk and (data.dementia_or_cognitive_impairment or data.significant_immobility):
        reasons.append("High falls risk combined with dementia or significant immobility.")
        return RiskCategory.very_high, reasons

    if min_t is not None and min_t <= -3.0:
        reasons.append(f"Very low T-score (≤ -3.0); minimum T-score {min_t:.2f}.")
        return RiskCategory.very_high, reasons

    if very_high_frax:
        reasons.append(
            f"Very high external FRAX pattern: major={frax_major:.1f}%, hip={frax_hip:.1f}%."
        )
        return RiskCategory.very_high, reasons

    if kanis_very_high_frax:
        reasons.append(
            f"FRAX major probability exceeds age-specific upper assessment threshold "
            f"(major={frax_major:.1f}% >= UAT {kanis_uat:.1f}% at age {data.age})."
        )
        return RiskCategory.very_high, reasons

    if low_bmd_plus_clinical_burden:
        reasons.append(
            "Low BMD with advanced age and frailty/GC/falls burden (very-high-risk pattern)."
        )
        return RiskCategory.very_high, reasons

    if imminent_clinical_pattern:
        reasons.append(
            "Imminent-risk clinical pattern (fracture on therapy and/or recurrent falls risk)."
        )
        return RiskCategory.very_high, reasons

    if min_t is not None and min_t <= -2.5:
        reasons.append(f"Osteoporosis-range T-score (≤ -2.5); minimum T-score {min_t:.2f}.")
        return RiskCategory.high, reasons

    if frax_major_high or frax_hip_high:
        reasons.append(
            f"Elevated external FRAX risk: major={frax_major:.1f}%, hip={frax_hip:.1f}%."
        )
        return RiskCategory.high, reasons

    if kanis_high_frax:
        reasons.append(
            f"FRAX major probability exceeds age-specific intervention threshold "
            f"(major={frax_major:.1f}% >= IT {kanis_it:.1f}% at age {data.age})."
        )
        return RiskCategory.high, reasons

    if internal_index_high:
        reasons.append(
            f"Cluster of FRAX-style clinical risk factors (internal index {internal_index:.1f})."
        )
        return RiskCategory.high, reasons

    internal_index_moderate = internal_index is not None and 3.0 <= internal_index < 6.0

    if min_t is not None and -2.5 < min_t <= -1.0:
        reasons.append(
            f"Osteopenic T-score range (between -1.0 and -2.5); minimum T-score {min_t:.2f}."
        )
        return RiskCategory.moderate, reasons

    if frax_major >= 10.0 or frax_hip >= 1.0:
        reasons.append(
            f"Intermediate external FRAX risk: major={frax_major:.1f}%, hip={frax_hip:.1f}%."
        )
        return RiskCategory.moderate, reasons

    if internal_index_moderate:
        reasons.append(
            f"Some FRAX-style clinical risk factors (internal index {internal_index:.1f})."
        )
        return RiskCategory.moderate, reasons

    reasons.append(
        "No major risk features identified (no prior major fragility fracture, no "
        "osteoporosis-range T-score, and no strong accumulation of FRAX-style clinical "
        "risk factors)."
    )
    return RiskCategory.low, reasons


def build_very_high_criteria_status(data: OsteoInput) -> List[VeryHighCriterionStatus]:
    t_scores = [
        ts
        for ts in [
            data.spine_t_score,
            data.total_hip_t_score,
            data.femoral_neck_t_score,
        ]
        if ts is not None
    ]
    min_t = min(t_scores) if t_scores else None
    effective_high_falls_risk = has_effective_high_falls_risk(data)
    effective_frailty = has_effective_frailty(data)
    frax_major = data.frax_major_osteoporotic or 0.0
    frax_hip = data.frax_hip or 0.0
    vertebral_count = max(
        data.vertebral_fracture_count,
        1 if FractureType.vertebral in data.prior_fragility_fractures else 0,
    )
    hip_count = max(
        data.hip_fracture_count,
        1 if FractureType.hip in data.prior_fragility_fractures else 0,
    )

    status: List[VeryHighCriterionStatus] = []
    status.append(
        VeryHighCriterionStatus(
            key="multiple_vertebral_fractures",
            label="Multiple vertebral fractures",
            met=vertebral_count >= 2,
            detail=f"vertebral fractures: n={vertebral_count}",
        )
    )
    status.append(
        VeryHighCriterionStatus(
            key="multiple_hip_fractures",
            label="Multiple hip fractures (including bilateral pattern)",
            met=hip_count >= 2,
            detail=f"hip fractures: n={hip_count}",
        )
    )
    status.append(
        VeryHighCriterionStatus(
            key="recent_fracture_with_low_bmd",
            label="Recent fragility fracture + low BMD (T-score <= -2.5)",
            met=data.recent_fragility_fracture and min_t is not None and min_t <= -2.5,
            detail=f"recent fracture={data.recent_fragility_fracture}, min T-score={min_t if min_t is not None else 'n/a'}",
        )
    )
    status.append(
        VeryHighCriterionStatus(
            key="very_low_bmd",
            label="Very low BMD (T-score <= -3.0)",
            met=min_t is not None and min_t <= -3.0,
            detail=f"min T-score={min_t if min_t is not None else 'n/a'}",
        )
    )
    status.append(
        VeryHighCriterionStatus(
            key="frax_above_very_high_threshold",
            label="FRAX 10-year above very-high thresholds",
            met=frax_major >= 30.0 or frax_hip >= 4.5,
            detail=f"major={frax_major:.1f}% (>=30), hip={frax_hip:.1f}% (>=4.5)",
        )
    )
    status.append(
        VeryHighCriterionStatus(
            key="low_bmd_age_frailty_gc_falls",
            label="Low BMD + advanced age + frailty/GC/falls risk",
            met=(
                min_t is not None
                and min_t <= -2.5
                and data.age >= 75
                and (effective_frailty or data.glucocorticoids or effective_high_falls_risk)
            ),
            detail=(
                f"min T-score={min_t if min_t is not None else 'n/a'}, age={data.age}, "
                f"frailty={effective_frailty}, glucocorticoids={data.glucocorticoids}, "
                f"high falls risk={effective_high_falls_risk}"
            ),
        )
    )
    status.append(
        VeryHighCriterionStatus(
            key="fracture_on_therapy",
            label="Fracture while on active therapy",
            met=data.fractures_during_current_therapy,
            detail=f"fracture_on_therapy={data.fractures_during_current_therapy}",
        )
    )
    return status


def build_suggestions(
    data: OsteoInput,
    risk: RiskCategory,
    calcium_total_mg: Optional[float],
    calcium_note: Optional[str],
) -> List[Suggestion]:
    suggestions: List[Suggestion] = []
    _, morse_note, morse_high = compute_morse_fall_risk(data)
    effective_high_falls_risk = data.high_falls_risk or morse_high
    creat = data.serum_creatinine_mg_dl
    urea = data.serum_urea_mg_dl

    # Pharmacologic / fracture risk
    if risk == RiskCategory.very_high:
        suggestions.append(
            Suggestion(
                category="pharmacologic",
                text=(
                    "Very-high fracture risk profile. In many modern guideline pathways "
                    "(IOF/ESCEO-aligned, Endocrine practice patterns), an anabolic-first "
                    "strategy is often considered (e.g. teriparatide or romosozumab when "
                    "appropriate), followed by anti-resorptive consolidation to preserve gains."
                ),
            )
        )
        suggestions.append(
            Suggestion(
                category="pharmacologic",
                text=(
                    "If fracture occurred during current therapy, reassess for treatment failure: "
                    "check adherence/administration, secondary causes (vitamin D, calcium/PTH, renal "
                    "function, malabsorption), and consider escalation/switch rather than simple continuation."
                ),
            )
        )
    elif risk == RiskCategory.high:
        suggestions.append(
            Suggestion(
                category="pharmacologic",
                text=(
                    "High fracture risk profile. Guideline-based anti-fracture therapy is usually "
                    "indicated, commonly with anti-resorptive options (oral/IV bisphosphonate or "
                    "denosumab depending renal profile, adherence feasibility, and patient factors)."
                ),
            )
        )
    elif risk == RiskCategory.moderate:
        suggestions.append(
            Suggestion(
                category="pharmacologic",
                text=(
                    "Risk category is moderate. Consider additional risk modifiers, "
                    "repeating DEXA at an appropriate interval, and discussing the "
                    "risk–benefit profile of pharmacologic therapy."
                ),
            )
        )
    else:
        suggestions.append(
            Suggestion(
                category="pharmacologic",
                text=(
                    "Risk category is low. Pharmacologic therapy may not be necessary, "
                    "but lifestyle optimization and periodic reassessment remain important."
                ),
            )
        )

    # Vitamin D
    vitd = data.vitamin_d_25oh_ng_ml or data.vitamin_d_25oh
    if vitd is not None:
        if vitd < 20:
            suggestions.append(
                Suggestion(
                    category="vitamin_d",
                    text=(
                        f"25-OH Vitamin D is low ({vitd:.1f} ng/mL). Consider correcting vitamin D "
                        "to at least the lower end of your target range before or alongside "
                        "pharmacologic osteoporosis therapy."
                    ),
                )
            )
        elif 20 <= vitd < 30:
            suggestions.append(
                Suggestion(
                    category="vitamin_d",
                    text=(
                        f"25-OH Vitamin D is borderline ({vitd:.1f} ng/mL). Mild optimization may "
                        "be beneficial in the context of bone health."
                    ),
                )
            )
        else:
            suggestions.append(
                Suggestion(
                    category="vitamin_d",
                    text=(
                        f"25-OH Vitamin D ({vitd:.1f} ng/mL) appears adequate for most patients; "
                        "continue current approach unless other factors suggest otherwise."
                    ),
                )
            )
    else:
        suggestions.append(
            Suggestion(
                category="vitamin_d",
                text=(
                    "No 25-OH Vitamin D value provided. Consider checking vitamin D status "
                    "if clinically indicated."
                ),
            )
        )

    # Calcium – intake-based if available
    ca_lab = data.serum_calcium_mg_dl or data.serum_calcium
    if calcium_total_mg is not None and calcium_note is not None:
        suggestions.append(
            Suggestion(
                category="calcium",
                text=calcium_note,
            )
        )
    else:
        if ca_lab is not None:
            if ca_lab < 8.5:
                suggestions.append(
                    Suggestion(
                        category="calcium",
                        text=(
                            f"Serum calcium appears low ({ca_lab:.2f} mg/dL). Investigate and correct "
                            "hypocalcemia before initiating or continuing certain osteoporosis "
                            "therapies."
                        ),
                    )
                )
            elif ca_lab > 10.5:
                suggestions.append(
                    Suggestion(
                        category="calcium",
                        text=(
                            f"Serum calcium appears elevated ({ca_lab:.2f} mg/dL). Consider evaluating for "
                            "hypercalcemia and adjusting calcium/vitamin D intake accordingly."
                        ),
                    )
                )
            else:
                suggestions.append(
                    Suggestion(
                        category="calcium",
                        text=(
                            f"Serum calcium ({ca_lab:.2f} mg/dL) is within a typical reference range. "
                            "Ensure total calcium intake (diet + supplementation) is appropriate "
                            "for age and risk profile."
                        ),
                    )
                )
        else:
            suggestions.append(
                Suggestion(
                    category="calcium",
                    text=(
                        "No calcium intake estimate or serum calcium value provided. "
                        "Consider reviewing dietary/supplemental calcium and checking serum "
                        "calcium if clinically indicated."
                    ),
                )
            )

    # Micronutrients & collagen supplements
    if data.magnesium_supplement or data.zinc_supplement or data.boron_supplement or data.vitamin_k2_supplement:
        parts = []
        if data.magnesium_supplement:
            dose = (
                f" (~{data.magnesium_supplement_mg_per_day:.0f} mg/day)"
                if data.magnesium_supplement_mg_per_day
                else ""
            )
            parts.append(f"magnesium{dose}")
        if data.zinc_supplement:
            dose = (
                f" (~{data.zinc_supplement_mg_per_day:.0f} mg/day)"
                if data.zinc_supplement_mg_per_day
                else ""
            )
            parts.append(f"zinc{dose}")
        if data.boron_supplement:
            dose = (
                f" (~{data.boron_supplement_mg_per_day:.1f} mg/day)"
                if data.boron_supplement_mg_per_day
                else ""
            )
            parts.append(f"boron{dose}")
        if data.vitamin_k2_supplement:
            dose = (
                f" (~{data.vitamin_k2_ug_per_day:.0f} μg/day)"
                if data.vitamin_k2_ug_per_day
                else ""
            )
            parts.append(f"vitamin K2{dose}")
        combo = ", ".join(parts)
        suggestions.append(
            Suggestion(
                category="micronutrients",
                text=(
                    f"Supportive micronutrient supplementation reported ({combo}). These may support "
                    "general bone and metabolic health but do not replace guideline-directed "
                    "osteoporosis pharmacotherapy when indicated."
                ),
            )
        )
    else:
        suggestions.append(
            Suggestion(
                category="micronutrients",
                text=(
                    "No specific magnesium/zinc/boron/vitamin K2 supplementation reported. "
                    "Consider overall dietary quality and micronutrient sufficiency, ideally "
                    "guided by a dietician when needed."
                ),
            )
        )

    if data.fortibone_supplement or data.colabone_supplement:
        which = []
        if data.fortibone_supplement:
            which.append("Fortibone")
        if data.colabone_supplement:
            which.append("Colabone")
        label = ", ".join(which)
        suggestions.append(
            Suggestion(
                category="collagen",
                text=(
                    f"Collagen-based supplementation reported ({label}). Frame this as an adjunct to, "
                    "not a substitute for, established anti-fracture pharmacotherapy and lifestyle measures."
                ),
            )
        )

    # Lifestyle / exercise
    if data.exercise_level in [ExerciseLevel.none, ExerciseLevel.light]:
        suggestions.append(
            Suggestion(
                category="lifestyle",
                text=(
                    "Encourage structured, safe exercise including resistance training and "
                    "balance work, tailored to the patient's capacity and comorbidities."
                ),
            )
        )
    else:
        suggestions.append(
            Suggestion(
                category="lifestyle",
                text=(
                    "Current exercise level is at least moderate. Maintain or refine the "
                    "program with emphasis on resistance and balance training for "
                    "fracture prevention."
                ),
            )
        )

    # Daily walking
    if data.daily_walking in [DailyWalking.none, DailyWalking.under_15_min]:
        suggestions.append(
            Suggestion(
                category="walking",
                text=(
                    "Daily walking is limited. If feasible, gradually increase walking time "
                    "towards a regular, tolerable routine to support bone and muscle health."
                ),
            )
        )
    else:
        suggestions.append(
            Suggestion(
                category="walking",
                text=(
                    "Daily walking time appears reasonable. Continue encouraging regular "
                    "weight-bearing activity within safety limits."
                ),
            )
        )

    # Falls risk / cognition / immobility
    if effective_high_falls_risk or data.history_of_falls_last_year:
        suggestions.append(
            Suggestion(
                category="falls_risk",
                text=(
                    "Falls risk is elevated. Consider a multifactorial falls assessment "
                    "including medication review, vision, footwear, and home safety "
                    "measures; physiotherapy for balance and strength can be helpful."
                ),
            )
        )
        if morse_note is not None:
            suggestions.append(
                Suggestion(
                    category="falls_risk",
                    text=f"{morse_note} Use this together with clinical judgment and observed mobility.",
                )
            )

    if data.dementia_or_cognitive_impairment:
        suggestions.append(
            Suggestion(
                category="cognition",
                text=(
                    "Cognitive impairment present. Involve caregivers and consider a "
                    "simplified regimen with clear instructions to reduce medication "
                    "errors and support adherence and falls prevention."
                ),
            )
        )

    if data.significant_immobility:
        suggestions.append(
            Suggestion(
                category="mobility",
                text=(
                    "Significant immobility noted. Consider physiotherapy, occupational "
                    "therapy, and safe mobilization strategies to minimize deconditioning "
                    "and further fracture risk."
                ),
            )
        )
    if data.cfs_score is not None:
        if data.cfs_score >= 5:
            suggestions.append(
                Suggestion(
                    category="frailty",
                    text=(
                        f"CFS score is {data.cfs_score}/9 (frailty range). Consider geriatric-oriented "
                        "falls prevention, strength/balance rehabilitation, and simplified medication pathways."
                    ),
                )
            )
        else:
            suggestions.append(
                Suggestion(
                    category="frailty",
                    text=(
                        f"CFS score is {data.cfs_score}/9 (not in frailty range). Continue prevention measures "
                        "and monitor function over time."
                    ),
                )
            )

    # Nutrition / dietician
    if data.dietician_follow_up:
        suggestions.append(
            Suggestion(
                category="nutrition",
                text=(
                    "Dietician follow-up already in place. Coordinate with nutrition "
                    "support to align calcium, protein, and micronutrient intake with "
                    "bone health goals."
                ),
            )
        )
    else:
        suggestions.append(
            Suggestion(
                category="nutrition",
                text=(
                    "No dietician follow-up reported. Consider dietician referral for "
                    "calcium/protein optimization, weight management, and broader "
                    "cardiometabolic support, especially in higher-risk patients."
                ),
            )
        )

    # Hyperparathyroidism / 24h urine patterns
    add_hyperparathyroid_suggestions(data, suggestions)

    # Renal status framing for treatment selection
    if creat is None:
        suggestions.append(
            Suggestion(
                category="renal_precheck",
                text=(
                    "Renal function data are incomplete (no serum creatinine provided). Add "
                    "creatinine/eGFR before finalizing anti-osteoporotic drug selection, especially "
                    "when considering IV bisphosphonate pathways."
                ),
            )
        )
    else:
        kidney_bits = [f"Serum creatinine {creat:.3f} mg/dL"]
        if urea is not None:
            kidney_bits.append(f"urea {urea:.1f} mg/dL")
        suggestions.append(
            Suggestion(
                category="renal_precheck",
                text=(
                    ", ".join(kidney_bits)
                    + ". Interpret with eGFR and comorbidity profile when choosing anti-resorptive regimen."
                ),
            )
        )

    # Current pharmacologic therapy continuation/change framing
    add_current_therapy_suggestions(data, risk, suggestions)

    add_therapy_history_suggestions(data, suggestions)
    add_conference_protocol_suggestions(data, suggestions)

    return suggestions


def determine_follow_up_steps(data: OsteoInput, risk: RiskCategory) -> List[FollowUpStep]:
    steps: List[FollowUpStep] = []

    def add_step(text: str, timeframe: str):
        steps.append(FollowUpStep(text=text, timeframe=timeframe))

    if risk in [RiskCategory.high, RiskCategory.very_high]:
        add_step(
            "Repeat DEXA (lumbar spine + hip) at ~12 months after treatment start/change, then interval by risk trajectory.",
            "6-12 months",
        )
    elif risk == RiskCategory.moderate:
        add_step(
            "Consider follow-up DEXA in 18–36 months unless therapy changes.",
            "12+ months",
        )
    else:
        add_step(
            "Reassess BMD in 2–3 years unless events occur.",
            "12+ months",
        )

    baseline_labs = ["25-OH Vitamin D", "serum calcium", "phosphorus", "creatinine/eGFR"]
    if data.pth_pg_ml is None:
        baseline_labs.append("PTH")
    add_step(
        f"Before initiating/changing therapy, complete baseline labs ({', '.join(baseline_labs)}).",
        "0-1 month",
    )

    labs = ["25-OH Vitamin D", "serum calcium", "creatinine/eGFR"]
    if data.current_therapy_type in [CurrentTherapyType.denosumab, CurrentTherapyType.romosozumab, CurrentTherapyType.iv_bisphosphonate]:
        labs.append("phosphorus/magnesium as needed")
    add_step(
        f"Monitor labs ({', '.join(labs)}) every ~6 months in active treatment phases.",
        "0-6 months",
    )

    if data.current_therapy_type in [CurrentTherapyType.teriparatide, CurrentTherapyType.romosozumab]:
        add_step(
            "Consider bone-turnover marker follow-up (P1NP +/- CTX) at ~3–6 months to support pharmacodynamic monitoring.",
            "3-6 months",
        )

    if data.current_therapy_type == CurrentTherapyType.denosumab:
        add_step(
            "Ensure on-time denosumab interval and document a transition plan before any interruption to reduce rebound vertebral fracture risk.",
            "0-6 months",
        )

    if data.fractures_during_current_therapy:
        add_step(
            "If fracture on therapy, reassess rapidly (adherence, secondary causes, vertebral imaging when indicated) and consider treatment escalation/switch.",
            "0-6 months",
        )

    if data.significant_therapy_adverse_effects:
        add_step(
            "Track adverse effect symptoms and reassess tolerance within 3 months.",
            "0-6 months",
        )

    return steps


def build_clinical_note(
    data: OsteoInput,
    risk: RiskCategory,
    reasons: List[str],
    suggestions: List[Suggestion],
    internal_index: Optional[float],
    internal_index_note: Optional[str],
    calcium_total_mg: Optional[float],
    calcium_note: Optional[str],
) -> str:
    lines: List[str] = []
    morse_score, morse_note, morse_high = compute_morse_fall_risk(data)
    effective_high_falls_risk = data.high_falls_risk or morse_high
    effective_frailty = has_effective_frailty(data)
    kanis_it, kanis_uat = get_kanis_major_thresholds_by_age(data.age)

    lines.append("Osteoporosis decision-support summary (for clinician use only).")
    lines.append("")
    lines.append(
        f"Patient profile: age {data.age}, sex {data.sex.value}, menopause status {data.menopause_status.value}."
    )
    lines.append(
        "Algorithmic reference framework: IOF/ESCEO position paper (Kanis et al, Osteoporos Int 2020;31:1-12). "
        f"Age-specific FRAX major IT/UAT at this age: ~{kanis_it:.1f}% / ~{kanis_uat:.1f}%."
    )

    # BMD summary
    t_parts = []
    if data.spine_t_score is not None:
        t_parts.append(f"spine T-score {data.spine_t_score:.2f}")
    if data.total_hip_t_score is not None:
        t_parts.append(f"total hip T-score {data.total_hip_t_score:.2f}")
    if data.femoral_neck_t_score is not None:
        t_parts.append(f"femoral neck T-score {data.femoral_neck_t_score:.2f}")
    if t_parts:
        lines.append("BMD: " + ", ".join(t_parts) + ".")
    else:
        lines.append("BMD: no T-scores provided.")

    if data.prior_fragility_fractures:
        frx_types = ", ".join(sorted({f.value for f in data.prior_fragility_fractures}))
        lines.append(f"Prior fragility fractures: {frx_types}.")
    else:
        lines.append("Prior fragility fractures: none reported.")
    if data.vertebral_fracture_count:
        lines.append(f"Reported vertebral fracture count: {data.vertebral_fracture_count}.")
    if data.hip_fracture_count:
        lines.append(f"Reported hip fracture count: {data.hip_fracture_count}.")
    if data.recent_fragility_fracture:
        lines.append("Recent fragility fracture pattern: yes (imminent-risk marker).")

    if data.frax_major_osteoporotic is not None or data.frax_hip is not None:
        lines.append(
            "FRAX (10-year): "
            f"major osteoporotic {data.frax_major_osteoporotic or 0:.1f}%, "
            f"hip {data.frax_hip or 0:.1f}%."
        )
        if (data.frax_major_osteoporotic or 0.0) >= 30.0 or (data.frax_hip or 0.0) >= 4.5:
            lines.append("FRAX is above the very-high-risk thresholds (major >=30% and/or hip >=4.5%).")

    # Internal index
    if internal_index is not None and internal_index_note is not None:
        lines.append(
            f"Internal FRAX-style clinical risk index (not official FRAX): {internal_index:.1f}."
        )
        lines.append(internal_index_note)

    # Labs summary (new fields + backward compat)
    lab_bits = []

    vitd = data.vitamin_d_25oh_ng_ml or data.vitamin_d_25oh
    if vitd is not None:
        lab_bits.append(f"25-OH Vit D {vitd:.1f} ng/mL")

    ca_lab = data.serum_calcium_mg_dl or data.serum_calcium
    if ca_lab is not None:
        lab_bits.append(f"serum Ca {ca_lab:.2f} mg/dL")

    if data.serum_phosphorus_mg_dl is not None:
        lab_bits.append(f"serum P {data.serum_phosphorus_mg_dl:.2f} mg/dL")
    if data.pth_pg_ml is not None:
        lab_bits.append(f"PTH {data.pth_pg_ml:.1f} pg/mL")
    if data.serum_magnesium_mg_dl is not None:
        lab_bits.append(f"serum Mg {data.serum_magnesium_mg_dl:.2f} mg/dL")
    if data.serum_zinc_ug_dl is not None:
        lab_bits.append(f"serum Zn {data.serum_zinc_ug_dl:.0f} μg/dL")
    if data.serum_urea_mg_dl is not None:
        lab_bits.append(f"serum urea {data.serum_urea_mg_dl:.2f} mg/dL")
    if data.serum_creatinine_mg_dl is not None:
        lab_bits.append(f"serum creatinine {data.serum_creatinine_mg_dl:.3f} mg/dL")

    if lab_bits:
        lines.append("Labs: " + ", ".join(lab_bits) + ".")
    if data.t_score_history:
        latest = data.t_score_history[-1]
        parts = []
        if latest.spine_total is not None:
            parts.append(f"spine total {latest.spine_total:.2f}")
        if latest.total_hip is not None:
            parts.append(f"hip total {latest.total_hip:.2f}")
        if latest.femoral_neck is not None:
            parts.append(f"femoral neck {latest.femoral_neck:.2f}")
        if parts:
            date_label = latest.date or "latest"
            lines.append(f"T-score ({date_label}): " + ", ".join(parts) + ".")

    # Calcium intake
    if calcium_total_mg is not None and calcium_note is not None:
        lines.append(calcium_note)

    # Lifestyle / functional
    lines.append(
        f"Exercise level: {data.exercise_level.value}, daily walking: {data.daily_walking.value}."
    )
    fragility_modifiers = []
    if effective_high_falls_risk:
        fragility_modifiers.append("high falls risk")
    if data.history_of_falls_last_year:
        fragility_modifiers.append("falls in last year")
    if data.frailty:
        fragility_modifiers.append("frailty")
    if data.cfs_score is not None:
        fragility_modifiers.append(f"CFS {data.cfs_score}/9")
    if data.glucocorticoids:
        fragility_modifiers.append("glucocorticoid exposure")
    if data.dementia_or_cognitive_impairment:
        fragility_modifiers.append("cognitive impairment")
    if data.significant_immobility:
        fragility_modifiers.append("immobility")
    if fragility_modifiers:
        lines.append("Functional/fragility modifiers: " + ", ".join(fragility_modifiers) + ".")
    if effective_frailty and not data.frailty and data.cfs_score is not None:
        lines.append("Frailty signal derived from CFS score (>=5).")
    if morse_score is not None and morse_note is not None:
        lines.append(morse_note)

    # Risk category
    lines.append("")
    lines.append(f"Risk stratification: {risk.value.upper()} fracture risk.")
    if reasons:
        lines.append("Key drivers: " + "; ".join(reasons) + ".")

    # Suggestions grouped by category
    lines.append("")
    lines.append("Decision-support suggestions (non-binding, for reflection and documentation):")
    by_cat = {}
    for s in suggestions:
        by_cat.setdefault(s.category, []).append(s.text)

    for cat, texts in sorted(by_cat.items()):
        lines.append(f"- {cat}:")
        for t in texts:
            lines.append(f"  • {t}")

    lines.append("")
    lines.append(
        "This summary is generated by a rules-based support tool and does not replace "
        "clinical judgment or formal guideline consultation."
    )

    return "\n".join(lines)


def build_patient_summary(
    risk: RiskCategory,
    suggestions: List[Suggestion],
) -> str:
    risk_text_map = {
        RiskCategory.low: "Your current risk of bone fracture appears low.",
        RiskCategory.moderate: "Your current risk of bone fracture appears moderate.",
        RiskCategory.high: "Your current risk of bone fracture appears higher than average.",
        RiskCategory.very_high: "Your current risk of bone fracture appears clearly high.",
    }

    lines: List[str] = []
    lines.append(risk_text_map[risk])
    lines.append(
        "This is based on information like bone density, any past fractures, and other "
        "health and lifestyle factors."
    )

    important_categories = {
        "vitamin_d",
        "calcium",
        "lifestyle",
        "walking",
        "falls_risk",
        "nutrition",
    }
    key_points = [s.text for s in suggestions if s.category in important_categories]

    if key_points:
        lines.append("")
        lines.append("Key areas we may focus on together include:")
        for kp in key_points:
            lines.append(f"- {kp}")

    lines.append("")
    lines.append(
        "These points are for discussion with your doctor; they are not recommendations "
        "on their own."
    )

    return "\n".join(lines)


def compute_assessment_from_input(input_data: OsteoInput) -> OsteoAssessment:
    internal_index, internal_index_note = compute_internal_frax_like_index(input_data)
    calcium_total_mg, calcium_note = calculate_calcium_intake(input_data)
    risk, reasons = determine_risk_category(input_data, internal_index)
    suggestions = build_suggestions(input_data, risk, calcium_total_mg, calcium_note)
    attach_evidence_to_suggestions(suggestions)
    follow_up_steps = determine_follow_up_steps(input_data, risk)
    very_high_criteria = build_very_high_criteria_status(input_data)

    clinical_note = build_clinical_note(
        input_data,
        risk,
        reasons,
        suggestions,
        internal_index,
        internal_index_note,
        calcium_total_mg,
        calcium_note,
    )
    patient_summary = build_patient_summary(risk, suggestions)

    return OsteoAssessment(
        risk_category=risk,
        risk_reasons=reasons,
        internal_frax_like_index=internal_index,
        internal_frax_like_note=internal_index_note,
        estimated_total_calcium_mg=calcium_total_mg,
        calcium_intake_note=calcium_note,
        suggestions=suggestions,
        clinical_note=clinical_note,
        patient_summary=patient_summary,
        follow_up_steps=follow_up_steps,
        very_high_criteria=very_high_criteria,
    )

# =========================
# API endpoints
# =========================


@app.post("/osteoporosis/evaluate", response_model=OsteoStoredAssessment)
def evaluate_osteoporosis(req: OsteoEvaluationRequest) -> OsteoStoredAssessment:
    """
    Evaluate osteoporosis fracture risk and generate written suggestions.
    Also persist the assessment per patient in SQLite.
    """
    input_data = req.input_data
    assessment = compute_assessment_from_input(input_data)

    stored = OsteoStoredAssessment(
        assessment_id=str(uuid4()),
        patient_id=req.patient_id,
        created_at=datetime.utcnow(),
        input_data=input_data,
        assessment=assessment,
    )

    with Session(engine) as session:
        row = AssessmentORM(
            id=stored.assessment_id,
            patient_id=stored.patient_id,
            created_at=stored.created_at,
            input_json=json.loads(stored.input_data.model_dump_json()),
            output_json=json.loads(stored.assessment.model_dump_json()),
            risk_category=stored.assessment.risk_category.value,
            current_therapy_type=stored.input_data.current_therapy_type.value,
        )
        session.add(row)
        session.commit()

    return stored


@app.put(
    "/osteoporosis/assessment/{assessment_id}",
    response_model=OsteoStoredAssessment,
)
def update_assessment(assessment_id: str, req: UpdateAssessmentRequest) -> OsteoStoredAssessment:
    """
    Update an existing assessment in place (same assessment_id), useful when
    latest visit data become available later and user does not want a new visit row.
    """
    with Session(engine) as session:
        row = session.get(AssessmentORM, assessment_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Assessment not found")

        input_data = req.input_data
        assessment = compute_assessment_from_input(input_data)

        row.patient_id = req.patient_id or row.patient_id
        row.input_json = json.loads(input_data.model_dump_json())
        row.output_json = json.loads(assessment.model_dump_json())
        row.risk_category = assessment.risk_category.value
        row.current_therapy_type = input_data.current_therapy_type.value
        session.add(row)
        session.commit()
        session.refresh(row)

    return OsteoStoredAssessment(
        assessment_id=row.id,
        patient_id=row.patient_id,
        created_at=row.created_at,
        input_data=OsteoInput.model_validate(row.input_json),
        assessment=OsteoAssessment.model_validate(row.output_json),
    )


@app.delete(
    "/osteoporosis/assessment/{assessment_id}",
    response_model=DeleteAssessmentResponse,
)
def delete_assessment(assessment_id: str) -> DeleteAssessmentResponse:
    """
    Delete a stored assessment (useful for duplicate/invalid entries).
    """
    with Session(engine) as session:
        row = session.get(AssessmentORM, assessment_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Assessment not found")
        session.delete(row)
        session.commit()
    return DeleteAssessmentResponse(
        deleted=True,
        assessment_id=assessment_id,
        message="Assessment deleted.",
    )


@app.get(
    "/osteoporosis/patient/{patient_id}/latest",
    response_model=OsteoStoredAssessment,
)
def get_latest_assessment(patient_id: str) -> OsteoStoredAssessment:
    """
    Fetch the most recent stored assessment for a given patient.
    """
    with Session(engine) as session:
        stmt = (
            select(AssessmentORM)
            .where(AssessmentORM.patient_id == patient_id)
            .order_by(AssessmentORM.created_at.desc())
        )
        row = session.execute(stmt).scalars().first()

        if row is None:
            raise HTTPException(status_code=404, detail="No assessments for this patient")

        input_data = OsteoInput.model_validate(row.input_json)
        assessment = OsteoAssessment.model_validate(row.output_json)

        return OsteoStoredAssessment(
            assessment_id=row.id,
            patient_id=row.patient_id,
            created_at=row.created_at,
            input_data=input_data,
            assessment=assessment,
        )

@app.get(
    "/osteoporosis/patient/{patient_id}/history",
    response_model=List[OsteoHistoryEntry],
)
def get_history(patient_id: str) -> List[OsteoHistoryEntry]:
    """
    Return all stored assessments for a given patient, oldest first.
    Useful for reviewing past visits and building trends.
    """
    with Session(engine) as session:
        stmt = (
            select(AssessmentORM)
            .where(AssessmentORM.patient_id == patient_id)
            .order_by(AssessmentORM.created_at.asc())
        )
        rows = session.execute(stmt).scalars().all()

    history: List[OsteoHistoryEntry] = []
    for row in rows:
        input_data = OsteoInput.model_validate(row.input_json)
        assessment = OsteoAssessment.model_validate(row.output_json)
        history.append(
            OsteoHistoryEntry(
                assessment_id=row.id,
                patient_id=row.patient_id,
                created_at=row.created_at,
                input_data=input_data,
                assessment=assessment,
            )
        )
    return history


@app.post(
    "/osteoporosis/analytics/therapy-comparison",
    response_model=TherapyComparisonResponse,
)
def compare_therapy_patterns(req: TherapyComparisonRequest) -> TherapyComparisonResponse:
    """
    Cohort comparison on latest assessment per patient:
    - filter by current therapy
    - filter by transition (from_therapy -> to_therapy/current)
    """
    with Session(engine) as session:
        subq = (
            select(
                AssessmentORM.patient_id.label("pid"),
                func.max(AssessmentORM.created_at).label("max_created"),
            )
            .group_by(AssessmentORM.patient_id)
            .subquery()
        )
        stmt = (
            select(AssessmentORM)
            .join(
                subq,
                (AssessmentORM.patient_id == subq.c.pid)
                & (AssessmentORM.created_at == subq.c.max_created),
            )
            .limit(req.limit)
        )
        rows = session.execute(stmt).scalars().all()

    parsed_rows: List[TherapyComparisonRow] = []
    by_risk: Dict[str, int] = {}
    by_current_therapy: Dict[str, int] = {}
    transitions: Dict[str, int] = {}

    for row in rows:
        input_data = OsteoInput.model_validate(row.input_json)
        assessment = OsteoAssessment.model_validate(row.output_json)

        previous = None
        if input_data.therapy_history and len(input_data.therapy_history) >= 2:
            previous = input_data.therapy_history[-2].therapy_type.value

        current = input_data.current_therapy_type.value
        transition = f"{previous}->{current}" if previous else None

        if req.current_therapy and current != req.current_therapy.value:
            continue
        if req.to_therapy and current != req.to_therapy.value:
            continue
        if req.from_therapy and previous != req.from_therapy.value:
            continue

        parsed_rows.append(
            TherapyComparisonRow(
                patient_id=row.patient_id,
                assessment_id=row.id,
                created_at=row.created_at,
                risk_category=assessment.risk_category.value,
                current_therapy=current,
                previous_therapy=previous,
                transition=transition,
            )
        )

        by_risk[assessment.risk_category.value] = by_risk.get(assessment.risk_category.value, 0) + 1
        by_current_therapy[current] = by_current_therapy.get(current, 0) + 1
        if transition:
            transitions[transition] = transitions.get(transition, 0) + 1

    parsed_rows.sort(key=lambda r: r.created_at, reverse=True)

    return TherapyComparisonResponse(
        total_latest_patients=len(rows),
        matched_patients=len(parsed_rows),
        filters={
            "current_therapy": req.current_therapy.value if req.current_therapy else None,
            "from_therapy": req.from_therapy.value if req.from_therapy else None,
            "to_therapy": req.to_therapy.value if req.to_therapy else None,
        },
        by_risk=by_risk,
        by_current_therapy=by_current_therapy,
        transitions=transitions,
        rows=parsed_rows,
    )

def bmd_trend_from_history(history: List[OsteoHistoryEntry]) -> Optional[str]:
    """
    Simple BMD trend description based on first vs last visit.
    """
    if len(history) < 2:
        return None

    first = history[0].input_data
    last = history[-1].input_data

    parts = []

    if first.spine_t_score is not None and last.spine_t_score is not None:
        delta = last.spine_t_score - first.spine_t_score
        if abs(delta) >= 0.3:
            direction = "improved" if delta > 0 else "worsened"
            parts.append(
                f"Lumbar spine T-score has {direction} from {first.spine_t_score:.2f} "
                f"to {last.spine_t_score:.2f} over the recorded interval."
            )

    if first.total_hip_t_score is not None and last.total_hip_t_score is not None:
        delta = last.total_hip_t_score - first.total_hip_t_score
        if abs(delta) >= 0.3:
            direction = "improved" if delta > 0 else "worsened"
            parts.append(
                f"Total hip T-score has {direction} from {first.total_hip_t_score:.2f} "
                f"to {last.total_hip_t_score:.2f}."
            )

    if first.femoral_neck_t_score is not None and last.femoral_neck_t_score is not None:
        delta = last.femoral_neck_t_score - first.femoral_neck_t_score
        if abs(delta) >= 0.3:
            direction = "improved" if delta > 0 else "worsened"
            parts.append(
                f"Femoral neck T-score has {direction} from {first.femoral_neck_t_score:.2f} "
                f"to {last.femoral_neck_t_score:.2f}."
            )

    if not parts:
        return None
    return " ".join(parts)


def ctx_trend_from_history(history: List[OsteoHistoryEntry]) -> Optional[str]:
    """
    CTX trend based on first vs last available CTX values.
    """
    values = [
        (h.created_at, h.input_data.ctx_ng_ml)
        for h in history
        if h.input_data.ctx_ng_ml is not None
    ]
    if len(values) < 2:
        return None
    values.sort(key=lambda x: x[0])
    t0, v0 = values[0]
    t1, v1 = values[-1]
    if v0 is None or v0 <= 0:
        return None
    change = (v1 - v0) / v0 * 100.0
    if change <= -50:
        return (
            f"CTX has fallen from ~{v0:.2f} to ~{v1:.2f} ng/mL (~{change:.0f}% change) across "
            "recorded measurements, compatible with strong anti-resorptive effect when "
            "interpreted with lab-specific reference ranges."
        )
    if change >= 50:
        return (
            f"CTX has risen from ~{v0:.2f} to ~{v1:.2f} ng/mL (~+{change:.0f}% change); this may "
            "reflect reduced effect or treatment interruption and should be interpreted in the "
            "context of dosing, timing and adherence."
        )
    return None


def p1np_trend_from_history(history: List[OsteoHistoryEntry]) -> Optional[str]:
    """
    P1NP trend for anabolic therapy monitoring.
    """
    values = [
        (h.created_at, h.input_data.p1np_ng_ml)
        for h in history
        if h.input_data.p1np_ng_ml is not None
    ]
    if len(values) < 2:
        return None
    values.sort(key=lambda x: x[0])
    t0, v0 = values[0]
    t1, v1 = values[-1]
    if v0 is None or v0 <= 0:
        return None
    change = (v1 - v0) / v0 * 100.0
    if change >= 50:
        return (
            f"P1NP has risen from ~{v0:.1f} to ~{v1:.1f} ng/mL (~+{change:.0f}% change). "
            "In the context of anabolic therapy, such an increase is often compatible "
            "with the expected pharmacodynamic response, subject to lab reference ranges "
            "and clinical judgement."
        )
    if change < 20:
        return (
            f"P1NP change from ~{v0:.1f} to ~{v1:.1f} ng/mL appears modest. If this reflects "
            "true low change and not lab variation, it may suggest a blunted anabolic response "
            "and could prompt closer review of dosing, adherence, and secondary factors."
        )
    return None

def bmd_trend_from_history(history: List[OsteoHistoryEntry]) -> Optional[str]:
    """
    Simple BMD trend description based on first vs last visit.
    """
    if len(history) < 2:
        return None

    first = history[0].input_data
    last = history[-1].input_data

    parts = []

    if first.spine_t_score is not None and last.spine_t_score is not None:
        delta = last.spine_t_score - first.spine_t_score
        if abs(delta) >= 0.3:
            direction = "improved" if delta > 0 else "worsened"
            parts.append(
                f"Lumbar spine T-score has {direction} from {first.spine_t_score:.2f} "
                f"to {last.spine_t_score:.2f} over the recorded interval."
            )

    if first.total_hip_t_score is not None and last.total_hip_t_score is not None:
        delta = last.total_hip_t_score - first.total_hip_t_score
        if abs(delta) >= 0.3:
            direction = "improved" if delta > 0 else "worsened"
            parts.append(
                f"Total hip T-score has {direction} from {first.total_hip_t_score:.2f} "
                f"to {last.total_hip_t_score:.2f}."
            )

    if first.femoral_neck_t_score is not None and last.femoral_neck_t_score is not None:
        delta = last.femoral_neck_t_score - first.femoral_neck_t_score
        if abs(delta) >= 0.3:
            direction = "improved" if delta > 0 else "worsened"
            parts.append(
                f"Femoral neck T-score has {direction} from {first.femoral_neck_t_score:.2f} "
                f"to {last.femoral_neck_t_score:.2f}."
            )

    if not parts:
        return None
    return " ".join(parts)


def ctx_trend_from_history(history: List[OsteoHistoryEntry]) -> Optional[str]:
    """
    CTX trend based on first vs last available CTX values.
    """
    values = [
        (h.created_at, h.input_data.ctx_ng_ml)
        for h in history
        if h.input_data.ctx_ng_ml is not None
    ]
    if len(values) < 2:
        return None
    values.sort(key=lambda x: x[0])
    t0, v0 = values[0]
    t1, v1 = values[-1]
    if v0 is None or v0 <= 0:
        return None
    change = (v1 - v0) / v0 * 100.0
    if change <= -50:
        return (
            f"CTX has fallen from ~{v0:.2f} to ~{v1:.2f} ng/mL (~{change:.0f}% change) across "
            "recorded measurements, compatible with strong anti-resorptive effect when "
            "interpreted with lab-specific reference ranges."
        )
    if change >= 50:
        return (
            f"CTX has risen from ~{v0:.2f} to ~{v1:.2f} ng/mL (~+{change:.0f}% change); this may "
            "reflect reduced effect or treatment interruption and should be interpreted in the "
            "context of dosing, timing and adherence."
        )
    return None


def p1np_trend_from_history(history: List[OsteoHistoryEntry]) -> Optional[str]:
    """
    P1NP trend for anabolic therapy monitoring.
    """
    values = [
        (h.created_at, h.input_data.p1np_ng_ml)
        for h in history
        if h.input_data.p1np_ng_ml is not None
    ]
    if len(values) < 2:
        return None
    values.sort(key=lambda x: x[0])
    t0, v0 = values[0]
    t1, v1 = values[-1]
    if v0 is None or v0 <= 0:
        return None
    change = (v1 - v0) / v0 * 100.0
    if change >= 50:
        return (
            f"P1NP has risen from ~{v0:.1f} to ~{v1:.1f} ng/mL (~+{change:.0f}% change). "
            "In the context of anabolic therapy, such an increase is often compatible "
            "with the expected pharmacodynamic response, subject to lab reference ranges "
            "and clinical judgement."
        )
    if change < 20:
        return (
            f"P1NP change from ~{v0:.1f} to ~{v1:.1f} ng/mL appears modest. If this reflects "
            "true low change and not lab variation, it may suggest a blunted anabolic response "
            "and could prompt closer review of dosing, adherence, and secondary factors."
        )
    return None


@app.get(
    "/osteoporosis/patient/{patient_id}/trend",
    response_model=TrendSummary,
)
def get_trend_summary(patient_id: str) -> TrendSummary:
    """
    Compute simple BMD and marker trends across all stored assessments
    for a patient.
    """
    with Session(engine) as session:
        stmt = (
            select(AssessmentORM)
            .where(AssessmentORM.patient_id == patient_id)
            .order_by(AssessmentORM.created_at.asc())
        )
        rows = session.execute(stmt).scalars().all()

    history = [
        OsteoHistoryEntry(
            assessment_id=r.id,
            patient_id=r.patient_id,
            created_at=r.created_at,
            input_data=OsteoInput.model_validate(r.input_json),
            assessment=OsteoAssessment.model_validate(r.output_json),
        )
        for r in rows
    ]

    bmd_text = bmd_trend_from_history(history) if history else None
    ctx_text = ctx_trend_from_history(history) if history else None
    p1np_text = p1np_trend_from_history(history) if history else None

    return TrendSummary(
        bmd_trend_text=bmd_text,
        ctx_trend_text=ctx_text,
        p1np_trend_text=p1np_text,
    )

@app.post("/osteoporosis/elaborate", response_model=ElaborationResponse)
def elaborate_osteoporosis(req: ElaborationRequest) -> ElaborationResponse:
    """
    Use an LLM (OpenAI) to elaborate on an existing osteoporosis assessment
    WITHOUT changing its medical content.
    """
    if openai_client is None:
        return ElaborationResponse(
            elaborated_text=(
                "LLM elaboration is not available because OPENAI_API_KEY is not configured "
                "on the server."
            )
        )

    a = req.assessment

    if req.audience == "clinician":
        style_instruction = (
            "Γράψε στα ελληνικά, με καθαρή δομή για κλινικό: "
            "1) σύντομη περίληψη κινδύνου, "
            "2) κρίσιμα ευρήματα (T-scores, κατάγματα, labs), "
            "3) ενότητα «Προτεινόμενες ενέργειες» με 3-5 bullets. "
            "Οι προτεινόμενες ενέργειες να βασίζονται στα δεδομένα εισόδου/suggestions "
            "και να είναι συγκεκριμένες για παρακολούθηση 6-12 μηνών, χωρίς δοσολογίες."
        )
    else:
        style_instruction = (
            "Γράψε στα ελληνικά για ασθενή, απλά και ευανάγνωστα: "
            "1) σύντομη εξήγηση της κατάστασης σε 3-4 προτάσεις, "
            "2) ενότητα «Τι να κάνω τώρα» με 4-6 bullets, "
            "3) ενότητα «Τι να συζητήσω στο επόμενο ραντεβού» με 2-3 bullets. "
            "Μην δίνεις δοσολογίες ή νέα φάρμακα."
        )
    style_instruction += " Να ολοκληρώνεις πλήρως την τελευταία πρόταση (όχι κομμένη κατάληξη)."

    system_prompt = (
        "Είσαι ένας ιδιαίτερα προσεκτικός βοηθός ιατρικής τεκμηρίωσης. ΔΕΝ εισάγεις ποτέ νέες "
        "διαγνώσεις, θεραπείες, αλλαγές δοσολογίας ή ερμηνείες εργαστηριακών εξετάσεων πέρα από "
        "ό,τι σου δίνεται ρητά. Απλώς αναδιατάσσεις και διατυπώνεις με σαφήνεια το υπάρχον κείμενο. "
        "Αν κάτι δεν αναφέρεται στα δεδομένα εισόδου, δεν το εφευρίσκεις."
    )

    user_payload = {
        "risk_category": a.risk_category,
        "risk_reasons": a.risk_reasons,
        "very_high_criteria_met": [c.label for c in a.very_high_criteria if c.met],
        "internal_frax_like_index": a.internal_frax_like_index,
        "calcium_intake_note": a.calcium_intake_note,
        "suggestions": [s.text for s in a.suggestions],
        "clinical_note": a.clinical_note,
        "patient_summary": a.patient_summary,
    }

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": style_instruction + "\n\n" + str(user_payload),
                },
            ],
            temperature=0.2,
            max_tokens=1400,
        )
        text = completion.choices[0].message.content.strip()
    except Exception as e:
        text = (
            "LLM elaboration is temporarily unavailable. "
            f"(Technical error: {e})"
        )

    return ElaborationResponse(elaborated_text=text)


@app.post("/osteoporosis/patient-handout", response_model=PatientHandoutResponse)
def create_patient_handout(req: PatientHandoutRequest) -> PatientHandoutResponse:
    html = build_patient_handout_html(
        req.assessment,
        input_data=req.input_data,
        agreed_plan=req.agreed_plan,
        patient_elaboration=req.patient_elaboration,
    )
    return PatientHandoutResponse(handout_html=html)


def build_treatment_recommendation_context(stored: OsteoStoredAssessment) -> str:
    lines: List[str] = []
    input_data = stored.input_data
    assessment = stored.assessment
    kanis_it, kanis_uat = get_kanis_major_thresholds_by_age(input_data.age)

    lines.append(f"Risk category: {assessment.risk_category.value.upper()} fracture risk.")
    if assessment.risk_reasons:
        lines.append("Key risk drivers: " + "; ".join(assessment.risk_reasons) + ".")
    if assessment.internal_frax_like_index is not None:
        lines.append(
            f"Internal FRAX-style index {assessment.internal_frax_like_index:.1f}."
        )

    lines.append(f"Current therapy: {input_data.current_therapy_type.value}.")
    if input_data.current_therapy_duration_years is not None:
        lines.append(
            f"Duration on current therapy: {input_data.current_therapy_duration_years:.1f} years."
        )
    if input_data.fractures_during_current_therapy:
        lines.append("New fragility fracture occurred while on therapy.")
    if input_data.significant_therapy_adverse_effects:
        lines.append("Patient reported significant adverse effects during therapy.")

    if input_data.therapy_history:
        recent = input_data.therapy_history[-3:]
        episodes = []
        for ep in recent:
            duration = (
                f"{ep.duration_years:.1f} yrs" if ep.duration_years is not None else "unknown duration"
            )
            episodes.append(f"{ep.therapy_type.value} ({duration}, holiday={ep.is_holiday})")
        lines.append("Recent therapy episodes: " + "; ".join(episodes) + ".")

    if assessment.suggestions:
        snippets = [s.text for s in assessment.suggestions[:4]]
        lines.append("Top suggestions: " + "; ".join(snippets) + ".")

    if assessment.clinical_note:
        first_line = assessment.clinical_note.splitlines()[0]
        lines.append("Clinical note begins: " + first_line)

    lab_notes = []
    vitd = input_data.vitamin_d_25oh_ng_ml or input_data.vitamin_d_25oh
    ca = input_data.serum_calcium_mg_dl or input_data.serum_calcium
    phos = input_data.serum_phosphorus_mg_dl
    if vitd is not None:
        lab_notes.append(f"25-OH Vit D {vitd:.1f} ng/mL")
    if ca is not None:
        lab_notes.append(f"calcium {ca:.2f} mg/dL")
    if phos is not None:
        lab_notes.append(f"phosphorus {phos:.2f} mg/dL")
    if input_data.pth_pg_ml is not None:
        lab_notes.append(f"PTH {input_data.pth_pg_ml:.1f} pg/mL")
    if input_data.serum_urea_mg_dl is not None:
        lab_notes.append(f"urea {input_data.serum_urea_mg_dl:.2f} mg/dL")
    if input_data.serum_creatinine_mg_dl is not None:
        lab_notes.append(f"creatinine {input_data.serum_creatinine_mg_dl:.3f} mg/dL")
    if laboratory := ", ".join(lab_notes):
        lines.append("Recent labs: " + laboratory + ".")
    if input_data.t_score_history:
        latest = input_data.t_score_history[-1]
        if latest.spine_total is not None or latest.total_hip is not None:
            spine_text = (
                f"spine {latest.spine_total:.2f}" if latest.spine_total is not None else ""
            )
            hip_text = (
                f"hip {latest.total_hip:.2f}" if latest.total_hip is not None else ""
            )
            entries = "; ".join(filter(None, [spine_text, hip_text]))
            date_info = f" ({latest.date})" if latest.date else ""
            lines.append(f"Latest T-score{date_info}: {entries}.")

    conference_tier = determine_conference_risk_tier(input_data)
    lines.append(
        "Conference-derived sequential-therapy tier: "
        f"{conference_tier.replace('_', ' ').upper()}."
    )
    lines.append(
        "Evidence anchor: Kanis et al (Osteoporos Int 2020;31:1-12). "
        f"At age {input_data.age}, FRAX major IT ~{kanis_it:.1f}% and UAT ~{kanis_uat:.1f}% "
        "(country-calibrated FRAX thresholds may vary)."
    )
    lines.append(
        "Discontinuation/holiday anchor for bisphosphonates: Diab & Watts "
        "(Ther Adv Musculoskelet Dis 2013, DOI: 10.1177/1759720X13477714): "
        "lower risk often reassess holiday after 3-5 years exposure; moderate risk often after 5-10 years "
        "with holiday ~3-5 years; high risk usually continue longer (~10 years) and, if holiday is needed, "
        "keep short (~1-2 years) with close monitoring."
    )
    lines.append(
        "Conference transition rules to preserve BMD: "
        "BP->denosumab tends to increase BMD; alendronate->zoledronate often has limited extra BMD gain; "
        "BP->anabolic may blunt gains; teriparatide/anabolic->denosumab or BP improves consolidation; "
        "denosumab->BP helps protect from rebound loss; avoid denosumab->teriparatide direct sequence when possible."
    )

    return "\n".join(lines)


def build_question_context(stored: OsteoStoredAssessment) -> str:
    ctx_lines = [build_treatment_recommendation_context(stored)]
    t_scores = stored.input_data.t_score_history
    if t_scores:
        latest = t_scores[-1]
        parts = []
        if latest.spine_total is not None:
            parts.append(f"spine total {latest.spine_total:.2f}")
        if latest.total_hip is not None:
            parts.append(f"hip total {latest.total_hip:.2f}")
        if latest.femoral_neck is not None:
            parts.append(f"femoral neck {latest.femoral_neck:.2f}")
        if parts:
            ctx_lines.append(f"Latest T-score ({latest.date or 'latest'}): " + ", ".join(parts))
    if t_scores:
        ctx_lines.append(f"T-score history entries: {len(t_scores)}.")
    return "\n".join(ctx_lines)


@app.post(
    "/osteoporosis/treatment-recommendation",
    response_model=TreatmentRecommendationResponse,
)
def recommend_treatment_change(
    req: TreatmentRecommendationRequest,
) -> TreatmentRecommendationResponse:
    if openai_client is None:
        return TreatmentRecommendationResponse(
            treatment_recommendation=(
                "AI treatment guidance is unavailable because OPENAI_API_KEY is not configured on the server."
            )
        )

    system_prompt = (
        "Είσαι σύμβουλος οστεοπόρωσης για κλινικούς. Γράφεις στα ελληνικά, συγκεκριμένα και "
        "πρακτικά, ευθυγραμμισμένα με guideline λογική (NOGG/ESCEO/Endocrine). "
        "Επιτρέπεται να αναφέρεις κατηγορίες φαρμάκων και λογική ακολουθίας (anabolic-first σε very high risk, "
        "έπειτα anti-resorptive consolidation), χωρίς δοσολογίες και χωρίς να αντικαθιστάς την κλινική κρίση. "
        "Δώσε προτεραιότητα στους conference-derived κανόνες ακολουθιών όταν υπάρχουν."
    )

    context = build_treatment_recommendation_context(req.assessment)
    user_prompt = (
        "Patient context:\n"
        f"{context}\n\n"
        "Με βάση τα παραπάνω, δώσε θεραπευτική καθοδήγηση σε 4 ενότητες:\n"
        "1) «Προτεινόμενη στρατηγική τώρα» (2-3 bullets με συγκεκριμένες classes: διφωσφονικά/denosumab/οστεοαναβολικό όπου ταιριάζει).\n"
        "2) «Τι λείπει πριν την τελική επιλογή» (εργαστηριακά/κλινικά δεδομένα που απαιτούνται).\n"
        "3) «Παρακολούθηση 6-12 μηνών» (συγκεκριμένα χρονικά checkpoints για labs και DEXA).\n"
        "4) «Safety net» (2 σύντομα bullets για fracture-on-therapy, denosumab interruption ή adverse effects).\n"
        "Μην δίνεις δοσολογίες. Να είσαι συγκεκριμένος και όχι γενικός."
    )

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1400,
        )
        text = completion.choices[0].message.content.strip()
    except Exception as exc:
        text = (
            "AI treatment guidance is temporarily unavailable. "
            f"(Technical error: {exc})"
        )

    return TreatmentRecommendationResponse(treatment_recommendation=text)


@app.post("/osteoporosis/question", response_model=LiteratureQuestionResponse)
def ask_literature_question(req: LiteratureQuestionRequest) -> LiteratureQuestionResponse:
    if openai_client is None:
        return LiteratureQuestionResponse(
            answer="LLM assistance is unavailable because OPENAI_API_KEY is not configured."
        )

    system_prompt = (
        "Είσαι ένας ενημερωμένος σύμβουλος οστεοπόρωσης. Απαντάς στα ελληνικά, "
        "αναφερόμενος σε ευρέως αποδεκτές κατευθυντήριες οδηγίες (π.χ. NOGG, Endocrine Society, "
        "ACP, IOF). Δεν εισάγεις νέες διαγνώσεις, αλλά ερμηνεύεις δεδομένα και παρέχεις αξιόπιστη επεξήγηση."
    )

    context = build_question_context(req.assessment)
    user_prompt = (
        "Κλινικό πλαίσιο:\n"
        f"{context}\n\n"
        "Ερώτηση:\n"
        f"{req.question}\n\n"
            "Απάντησε οργανομετρικά με 2-3 παραγράφους: (1) σύντομη πηγή/αιτιολόγηση από τη βιβλιογραφία, "
            "(2) πρακτική απάντηση, (3) αν χρειάζονται επιπλέον εξετάσεις ή δεδομένα, σημείωσε τα."
            " Να ολοκληρώνεις πλήρως την τελευταία πρόταση (όχι κομμένη κατάληξη)."
    )

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as exc:
        answer = f"Η έρευνα είναι προσωρινά ανέφικτη. (Σφάλμα: {exc})"

    return LiteratureQuestionResponse(answer=answer)


@app.get("/health")
def read_root():
    return {"status": "ok", "app": "osteoporosis backend is running"}
