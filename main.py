# main.py# main.py

from enum import Enum
from typing import List, Optional, Tuple

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field, conint, confloat

# =========================
# App & CORS
# =========================

app = FastAPI(
    title="Papachristou Ortho Osteoporosis Support",
    description=(
        "Guideline-inspired decision support for osteoporosis risk stratification, "
        "FRAX-style internal risk indexing, calcium intake estimation, and lab pattern hints. "
        "For clinician use only."
    ),
    version="0.3.0",
)

# Allow the local cockpit (localhost) and other origins (for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later if you want
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client for elaboration endpoint
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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
    light = "light"        # occasional walking
    moderate = "moderate"  # regular walking ± some resistance work
    vigorous = "vigorous"  # structured resistance, etc.


class DailyWalking(str, Enum):
    none = "none"
    under_15_min = "under_15_min"
    between_15_30_min = "between_15_30_min"
    over_30_min = "over_30_min"


class Suggestion(BaseModel):
    category: str  # e.g. "pharmacologic", "vitamin_d", "calcium", "labs_hyperparathyroidism"
    text: str


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

    # FRAX (if already calculated externally; optional)
    frax_major_osteoporotic: Optional[confloat(ge=0.0, le=100.0)] = None
    frax_hip: Optional[confloat(ge=0.0, le=100.0)] = None

    # FRAX-style clinical risk factors for internal risk index (NOT official FRAX)
    weight_kg: Optional[confloat(gt=0.0, le=300.0)] = None
    height_cm: Optional[confloat(gt=0.0, le=250.0)] = None
    parental_hip_fracture: bool = False
    current_smoker: bool = False
    glucocorticoids: bool = False
    rheumatoid_arthritis: bool = False
    secondary_osteoporosis: bool = False
    high_alcohol_intake: bool = False  # ≥3 units/day

    # Mineral metabolism & bone labs
    # Keep old names for backwards compatibility, but prefer the *_mg_dl / *_ng_ml fields.
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

    # Bone turnover markers
    osteocalcin_ng_ml: Optional[float] = Field(
        default=None, description="Osteocalcin (ng/mL)"
    )
    bone_alk_phos_u_l: Optional[float] = Field(
        default=None, description="Bone alkaline phosphatase (U/L)"
    )
    total_alk_phos_u_l: Optional[float] = Field(
        default=None, description="Total alkaline phosphatase (U/L)"
    )
    ctx_ng_ml: Optional[float] = Field(
        default=None, description="CTX (β-CrossLaps, ng/mL)"
    )
    p1np_ng_ml: Optional[float] = Field(
        default=None, description="P1NP (ng/mL)"
    )

    # Other labs
    serum_magnesium_mg_dl: Optional[float] = Field(
        default=None, description="Serum magnesium (mg/dL)"
    )
    serum_zinc_ug_dl: Optional[float] = Field(
        default=None, description="Serum zinc (μg/dL)"
    )
    serum_glucose_mg_dl: Optional[float] = Field(
        default=None, description="Fasting glucose (mg/dL)"
    )
    tsh_u_iu_ml: Optional[float] = Field(
        default=None, description="TSH (μIU/mL)"
    )
    free_t4_ng_dl: Optional[float] = Field(
        default=None, description="Free T4 (ng/dL)"
    )

    # Supplements / intake
    calcium_supplement: bool = False
    calcium_supplement_mg_per_day: Optional[float] = Field(
        default=None, description="Calcium supplement (mg/day)"
    )

    vitamin_d_supplement: bool = False
    vitamin_d3_iu_per_day: Optional[float] = Field(
        default=None, description="Vitamin D3 supplement (IU/day)"
    )

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

    # Lifestyle / functional status
    exercise_level: ExerciseLevel = ExerciseLevel.none
    daily_walking: DailyWalking = DailyWalking.none
    dietician_follow_up: bool = False
    high_falls_risk: bool = False
    history_of_falls_last_year: bool = False
    dementia_or_cognitive_impairment: bool = False
    significant_immobility: bool = False

    # Calcium intake fields (IOF-style paradigm, "portions per day")
    milk_portions_per_day: conint(ge=0, le=20) = 0
    yogurt_portions_per_day: conint(ge=0, le=20) = 0
    cheese_portions_per_day: conint(ge=0, le=20) = 0
    leafy_greens_portions_per_day: conint(ge=0, le=20) = 0
    fortified_food_portions_per_day: conint(ge=0, le=20) = 0
    other_dairy_portions_per_day: conint(ge=0, le=20) = 0


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


class ElaborationRequest(BaseModel):
    assessment: OsteoAssessment
    audience: str = Field(default="clinician", description="clinician or patient")


class ElaborationResponse(BaseModel):
    elaborated_text: str


# =========================
# Helper calculations
# =========================


def calculate_bmi(weight_kg: Optional[float], height_cm: Optional[float]) -> Optional[float]:
    if weight_kg is None or height_cm is None or height_cm <= 0:
        return None
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def compute_internal_frax_like_index(data: OsteoInput) -> Tuple[Optional[float], Optional[str]]:
    """
    Simple internal fracture risk index based on FRAX-style variables.
    This is *not* the official FRAX algorithm.
    """
    score = 0.0
    reasons: List[str] = []

    # Age contribution
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

    # BMI contribution
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


def calculate_calcium_intake(data: OsteoInput) -> Tuple[Optional[float], Optional[str]]:
    """
    Estimate daily calcium intake based on rough portion equivalents and supplement dose.
    Approximate only; for decision support, not precise nutrition prescription.
    """
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


# =========================
# Hyperparathyroidism pattern helper
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
                "monitoring calcium and PTH may be reasonable before final conclusions. "
            )
        suggestions.append(
            Suggestion(
                category="labs_hyperparathyroidism",
                text=text + "Correlate with repeat labs, renal function, and imaging as needed.",
            )
        )
    # Secondary hyperparathyroidism (vitamin D deficiency–type pattern)
    elif not high_ca and high_pth and vitd is not None and vitd < 20:
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
    # Hypocalcemia with low PTH (hypoparathyroidism-type pattern)
    elif low_ca and low_pth:
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
# Risk and suggestion logic
# =========================


def determine_risk_category(
    data: OsteoInput,
    internal_index: Optional[float],
) -> Tuple[RiskCategory, List[str]]:
    reasons: List[str] = []

    has_hip = FractureType.hip in data.prior_fragility_fractures
    has_vertebral = FractureType.vertebral in data.prior_fragility_fractures

    if has_hip or has_vertebral:
        reasons.append("History of hip or vertebral fragility fracture.")
        return RiskCategory.very_high, reasons

    if len(data.prior_fragility_fractures) >= 2:
        reasons.append("Multiple prior fragility fractures.")
        return RiskCategory.very_high, reasons

    if data.high_falls_risk and (data.dementia_or_cognitive_impairment or data.significant_immobility):
        reasons.append("High falls risk combined with dementia or significant immobility.")
        return RiskCategory.very_high, reasons

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

    if min_t is not None and min_t <= -2.5:
        reasons.append(f"Osteoporosis-range T-score (≤ -2.5); minimum T-score {min_t:.2f}.")
        return RiskCategory.high, reasons

    if frax_major_high or frax_hip_high:
        reasons.append(
            f"Elevated external FRAX risk: major={frax_major:.1f}%, hip={frax_hip:.1f}%."
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


def build_suggestions(
    data: OsteoInput,
    risk: RiskCategory,
    calcium_total_mg: Optional[float],
    calcium_note: Optional[str],
) -> List[Suggestion]:
    suggestions: List[Suggestion] = []

    # Pharmacologic / fracture risk
    if risk in [RiskCategory.high, RiskCategory.very_high]:
        suggestions.append(
            Suggestion(
                category="pharmacologic",
                text=(
                    "Risk category is high/very high. According to typical osteoporosis "
                    "guidelines, pharmacologic anti-fracture therapy is generally indicated. "
                    "Choice of specific agent and regimen should follow formal guidelines "
                    "and your clinical judgment."
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

    # Vitamin D (lab-based, with backward compatibility)
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

    # Calcium: intake-focused if we have calculator output
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
            dose = f" (~{data.magnesium_supplement_mg_per_day:.0f} mg/day)" if data.magnesium_supplement_mg_per_day else ""
            parts.append(f"magnesium{dose}")
        if data.zinc_supplement:
            dose = f" (~{data.zinc_supplement_mg_per_day:.0f} mg/day)" if data.zinc_supplement_mg_per_day else ""
            parts.append(f"zinc{dose}")
        if data.boron_supplement:
            dose = f" (~{data.boron_supplement_mg_per_day:.1f} mg/day)" if data.boron_supplement_mg_per_day else ""
            parts.append(f"boron{dose}")
        if data.vitamin_k2_supplement:
            dose = f" (~{data.vitamin_k2_ug_per_day:.0f} μg/day)" if data.vitamin_k2_ug_per_day else ""
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

        # Phosphorus
    if data.serum_phosphorus_mg_dl is not None:
        p = data.serum_phosphorus_mg_dl
        if p < 2.5:
            suggestions.append(Suggestion(
                category="labs_phosphorus",
                text=(
                    f"Serum phosphorus ({p:.2f} mg/dL) appears low. Consider integrating this "
                    "with calcium, PTH, and vitamin D to evaluate for possible secondary "
                    "mineral metabolism disturbances."
                ),
            ))
        elif p > 4.5:
            suggestions.append(Suggestion(
                category="labs_phosphorus",
                text=(
                    f"Serum phosphorus ({p:.2f} mg/dL) appears elevated. Correlate with renal "
                    "function, PTH, and other labs."
                ),
            ))

    # Bone turnover markers (very high-level commentary only)
    if data.ctx_ng_ml is not None:
        ctx = data.ctx_ng_ml
        suggestions.append(Suggestion(
            category="bone_turnover",
            text=(
                f"CTX measured at {ctx:.2f} ng/mL. Interpret in context of lab reference "
                "ranges, time since last dose of any anti-resorptive, and circadian variation."
            ),
        ))

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
    if data.high_falls_risk or data.history_of_falls_last_year:
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

def add_hyperparathyroid_suggestions(data: OsteoInput, suggestions: List[Suggestion]) -> None:
    ca = data.serum_calcium_mg_dl
    pth = data.pth_pg_ml
    vitd = data.vitamin_d_25oh_ng_ml
    phos = data.serum_phosphorus_mg_dl
    uca = data.urine_calcium_24h_mg

    # Only proceed if we have at least calcium and PTH
    if ca is None or pth is None:
        return

    # Very rough working ranges; you will interpret them clinically
    high_ca = ca > 10.5
    low_ca = ca < 8.5
    high_pth = pth > 65     # depends on lab
    low_pth = pth < 15

    # Primary hyperparathyroidism pattern (classic)
    if high_ca and high_pth:
        text = (
            f"Serum calcium {ca:.2f} mg/dL with elevated PTH {pth:.1f} pg/mL. "
            "This pattern can be seen in primary hyperparathyroidism or related disorders. "
        )
        if phos is not None and phos < 2.5:
            text += (
                f"Phosphorus is low ({phos:.2f} mg/dL), which may further support this pattern. "
            )
        if vitd is not None and vitd < 20:
            text += (
                f"Vitamin D is low ({vitd:.1f} ng/mL); correcting vitamin D while monitoring "
                "calcium and PTH may be reasonable before final conclusions."
            )
        suggestions.append(Suggestion(
            category="labs_hyperparathyroidism",
            text=text + "Correlate with repeat labs, renal function, and imaging as needed."
        ))
        return

    # Secondary hyperparathyroidism pattern (vitamin D deficiency / CKD, etc.)
    if not high_ca and high_pth and vitd is not None and vitd < 20:
        text = (
            f"PTH {pth:.1f} pg/mL with calcium {ca:.2f} mg/dL and low vitamin D "
            f"{vitd:.1f} ng/mL. This combination can be compatible with secondary "
            "hyperparathyroidism (e.g. vitamin D deficiency or CKD). "
        )
        if phos is not None:
            text += f"Phosphorus {phos:.2f} mg/dL; interpret with renal function and PTH trends. "
        suggestions.append(Suggestion(
            category="labs_hyperparathyroidism",
            text=text + "Consider appropriate workup and vitamin D repletion according to guidelines."
        ))
        return

    # Hypocalcemia with low PTH (hypoparathyroidism pattern)
    if low_ca and low_pth:
        suggestions.append(Suggestion(
            category="labs_hyperparathyroidism",
            text=(
                f"Hypocalcemia ({ca:.2f} mg/dL) with low PTH ({pth:.1f} pg/mL). "
                "This can be compatible with hypoparathyroidism. Correlate with "
                "clinical picture, magnesium, phosphorus, and drug history."
            ),
        ))

    # 24h urine calcium comments (e.g., low vs high)
    if uca is not None:
        if uca < 100:
            suggestions.append(Suggestion(
                category="urine_calcium",
                text=(
                    f"24-hour urine calcium is {uca:.0f} mg/24h, which is relatively low; "
                    "interpret in context (dietary calcium, vitamin D, and possible familial "
                    "hypocalciuric hypercalcemia if hypercalcemia is present)."
                ),
            ))
        elif uca > 300:
            suggestions.append(Suggestion(
                category="urine_calcium",
                text=(
                    f"24-hour urine calcium is {uca:.0f} mg/24h, on the higher side; "
                    "this can contribute to nephrolithiasis risk and may be relevant in "
                    "the context of hypercalcemia or high-dose calcium/vitamin D."
                ),
            ))   


    return suggestions


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

    lines.append("Osteoporosis decision-support summary (for clinician use only).")
    lines.append("")
    lines.append(
        f"Patient profile: age {data.age}, sex {data.sex.value}, menopause status {data.menopause_status.value}."
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

    if data.frax_major_osteoporotic is not None or data.frax_hip is not None:
        lines.append(
            "FRAX (10-year): "
            f"major osteoporotic {data.frax_major_osteoporotic or 0:.1f}%, "
            f"hip {data.frax_hip or 0:.1f}%."
        )

    # Internal index
    if internal_index is not None and internal_index_note is not None:
        lines.append(
            f"Internal FRAX-style clinical risk index (not official FRAX): {internal_index:.1f}."
        )
        lines.append(internal_index_note)

    # Labs
    lab_bits = []
    if data.vitamin_d_25oh is not None:
        lab_bits.append(f"25-OH Vit D {data.vitamin_d_25oh:.1f}")
    if data.serum_calcium is not None:
        lab_bits.append(f"serum Ca {data.serum_calcium:.2f}")
    if data.magnesium is not None:
        lab_bits.append(f"Mg {data.magnesium:.2f}")
    if data.zinc is not None:
        lab_bits.append(f"Zn {data.zinc:.2f}")
    if lab_bits:
        lines.append("Labs: " + ", ".join(lab_bits) + ".")

    # Calcium intake
    if calcium_total_mg is not None and calcium_note is not None:
        lines.append(calcium_note)

    # Lifestyle / functional
    lines.append(
        f"Exercise level: {data.exercise_level.value}, daily walking: {data.daily_walking.value}."
    )
    fragility_modifiers = []
    if data.high_falls_risk:
        fragility_modifiers.append("high falls risk")
    if data.history_of_falls_last_year:
        fragility_modifiers.append("falls in last year")
    if data.dementia_or_cognitive_impairment:
        fragility_modifiers.append("cognitive impairment")
    if data.significant_immobility:
        fragility_modifiers.append("immobility")

    if fragility_modifiers:
        lines.append("Functional/fragility modifiers: " + ", ".join(fragility_modifiers) + ".")

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
    # Short, gentle, non-prescriptive patient-style text
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

    # Pull a few key themes from suggestions
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


# =========================
# API Endpoint
# =========================


@app.post("/osteoporosis/evaluate", response_model=OsteoAssessment)
def evaluate_osteoporosis(input_data: OsteoInput) -> OsteoAssessment:
    """
    Evaluate osteoporosis fracture risk and generate written suggestions.
    This is a clinician-facing decision-support tool, not an autonomous prescriber.
    """

    # Internal FRAX-style index
    internal_index, internal_index_note = compute_internal_frax_like_index(input_data)

    # Calcium intake
    calcium_total_mg, calcium_note = calculate_calcium_intake(input_data)

    # Risk stratification
    risk, reasons = determine_risk_category(input_data, internal_index)

    # Suggestions
    suggestions = build_suggestions(input_data, risk, calcium_total_mg, calcium_note)

    # Notes
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
    )

# =========================
# API LLM
# =========================

@app.post("/osteoporosis/elaborate", response_model=ElaborationResponse)
def elaborate_osteoporosis(req: ElaborationRequest) -> ElaborationResponse:
    """
    Use an LLM to elaborate on an existing osteoporosis assessment
    WITHOUT changing its medical content.
    """
    a = req.assessment

    if req.audience == "clinician":
        style_instruction = (
            "Write 1–2 concise paragraphs as a short clinical impression for an "
            "orthopaedic specialist. Summarise the risk category, key drivers, relevant "
            "labs and calcium intake, and the main suggestions. DO NOT introduce new "
            "diagnoses, new treatments, or drug brand names. Stay neutral and guideline-aligned."
        )
    else:
        style_instruction = (
            "Write 1–2 short, simple paragraphs addressed to a patient. Explain their "
            "bone health situation, their approximate fracture risk, and the main areas "
            "you and their doctor may focus on (e.g. vitamin D, calcium, exercise, "
            "falls prevention). Avoid specific drug names, numbers, or lab ranges. "
            "Encourage them to discuss details with their doctor."
        )

    system_prompt = (
        "You are a careful medical documentation assistant. You NEVER introduce new "
        "diagnoses, treatments, dose changes, or lab interpretations beyond what you "
        "are explicitly given. You only rephrase and organise the input content. "
        "If something is not mentioned in the input, you do not speculate about it."
    )

    # Feed the structured assessment, not raw patient identifiers
    user_payload = {
        "risk_category": a.risk_category,
        "risk_reasons": a.risk_reasons,
        "internal_frax_like_index": a.internal_frax_like_index,
        "calcium_intake_note": a.calcium_intake_note,
        "suggestions": [s.text for s in a.suggestions],
        "clinical_note": a.clinical_note,
        "patient_summary": a.patient_summary,
    }

    # Call OpenAI Chat Completions (modern API)
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",  # good balance of cost/quality for note generation
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": style_instruction + "\n\n" + str(user_payload),
                },
            ],
            temperature=0.2,
            max_tokens=400,
        )
        text = completion.choices[0].message.content.strip()
    except Exception as e:
        # Fail safely: return a clear message instead of crashing
        text = (
            "LLM elaboration is temporarily unavailable. "
            f"(Technical error: {e})"
        )

    return ElaborationResponse(elaborated_text=text)


