# main.py

from datetime import date, datetime, timedelta, time, timezone
from typing import List, Optional, Dict, Any
import time as _time_module
from time import perf_counter
import asyncio
import sys
import base64
import hmac
import hashlib
from urllib.parse import urlencode
from fastapi import FastAPI, HTTPException, Request, Query, APIRouter, Header, Depends
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from zoneinfo import ZoneInfo

from cal_limassol import (
    LimassolBookRequest,
    LimassolSuggestionRequest,
    limassol_book,
    limassol_suggestions,
    router as limassol_router,
)
from cal_evrychou import (
    EvrychouBookRequest,
    EvrychouSuggestionRequest,
    evrychou_book,
    evrychou_suggestions,
    router as evrychou_router,
)
try:
    from cal_check_appointment import router as check_appointment_router
except ModuleNotFoundError:
    # Legacy check flow; the active production path is cal_check_appointment_by_phone.
    check_appointment_router = APIRouter()
from cal_check_appointment_by_phone import router as check_by_phone_router
from cal_client import CalClient

import logging
import json
import httpx
import os
import requests
import re
import unicodedata


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Cypriot BERT imports (ΥΠΟ ΣΥΝΘΗΚΗ — βλ. σχόλιο) ---
# ΣΗΜΑΝΤΙΚΟ: torch + transformers καταναλώνουν ~1.2-1.5GB RAM μόνο από το import,
# ΑΣΧΕΤΑ από το αν φορτωθεί μοντέλο. Σε Render 2GB plan αυτό μόνο φτάνει να
# προκαλέσει OOM kill. Τα παρακάτω flags είναι False by default, οπότε τα heavy
# imports γίνονται ΜΟΝΟ αν κάποιος τα ενεργοποιήσει ρητά στο Render env.
RECEPTION_STABILITY_MODE = _env_bool("RECEPTION_STABILITY_MODE", default=True)
ENABLE_CY_MLM = _env_bool("ENABLE_CY_MLM", default=False) and not RECEPTION_STABILITY_MODE
ENABLE_CY_BERT = _env_bool("ENABLE_CY_BERT", default=False) and not RECEPTION_STABILITY_MODE
ENABLE_CY_INTENT = _env_bool("ENABLE_CY_INTENT", default=False) and not RECEPTION_STABILITY_MODE

# ΣΗΜΑΝΤΙΚΟ: ENABLE_CY_MLM/ENABLE_CY_BERT παραμένουν στη βαριά διαδρομή
# (torch + transformers, ~1.2-1.5GB) γιατί δεν έχουν ακόμα μετατραπεί σε
# ONNX. Το ENABLE_CY_INTENT ΔΕΝ χρειάζεται πια torch/transformers — το
# μοντέλο σερβίρεται μέσω onnxruntime (~50-80MB) + tokenizers (standalone,
# Rust-based, μερικά MB — ΟΧΙ η βαριά transformers βιβλιοθήκη). Έτσι το
# ENABLE_CY_INTENT μπορεί να ενεργοποιηθεί με ασφάλεια σε Render Starter
# (2GB) χωρίς OOM ρίσκο.
if ENABLE_CY_MLM or ENABLE_CY_BERT:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForMaskedLM,
    )
    import torch
    import torch.nn.functional as F
else:
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoModelForMaskedLM = None  # type: ignore
    torch = None  # type: ignore
    F = None  # type: ignore

if ENABLE_CY_INTENT:
    import onnxruntime as _cy_intent_ort
    from tokenizers import Tokenizer as _CyIntentTokenizer
    import numpy as _cy_intent_np
else:
    _cy_intent_ort = None  # type: ignore
    _CyIntentTokenizer = None  # type: ignore
    _cy_intent_np = None  # type: ignore

from appointment_utils import (
    appointment_reason_present,
    appointment_triage_needed as should_triage_appointment,
    claim_action_once,
    classify_slot_selection_utterance,
    clear_conversation_state,
    clinic_operates_on_date,
    clinic_schedule_reason,
    clinic_supports_time_preference,
    DAY_WORDS_TO_INT,
    doctor_location_for_date,
    doctor_weekly_schedule_summary,
    get_clinic_info,
    infer_appointment_duration_minutes,
    infer_call_stage,
    infer_next_action,
    is_cyprus_public_holiday,
    is_non_working_day,
    is_weekend_day,
    make_friendly_day_gr,
    make_friendly_time_gr,
    mark_cancellation_confirmed,
    message_guard_key,
    name_collection_required_for_route,
    extract_patient_name_from_text,
    get_conversation_state,
    normalize_booking_phone,
    normalize_guard_text,
    release_action_claim,
    resolve_reference_date,
    resolve_target_date,
    rule_based_intent,
    classify_priority_level,
    should_say_latency_notice,
    update_conversation_state,
)
from cal_setmore_sync import sync_cal_to_setmore
from setmore_client import (
    create_setmore_shadow_appointment,
    list_setmore_synced_appointments,
    update_setmore_appointment_customer,
)
try:
    from setmore_patient_cache import (
        fetch_setmore_patient_rows,
        lookup_patient_cache_by_phone,
        lookup_patient_cache_by_name,
        refresh_setmore_patient_cache,
    )
except ModuleNotFoundError as exc:
    print(f"WARNING: setmore_patient_cache unavailable ({exc}); patient cache features disabled.")

    async def fetch_setmore_patient_rows(*, lookback_days=None, lookahead_days=None, page_limit=None):
        return [], 0

    def lookup_patient_cache_by_phone(phone_number: str | None, path=None):
        return None

    def lookup_patient_cache_by_name(first_name: str | None, last_name: str | None, path=None):
        return None

    async def refresh_setmore_patient_cache(path=None, *, lookback_days=None, lookahead_days=None, page_limit=None):
        return {
            "success": False,
            "path": str(path) if path else "",
            "appointments_scanned": 0,
            "patients_written": 0,
            "error": "setmore_patient_cache module not available",
        }

try:
    from patient_directory_db import (
        init_patient_directory_db,
        lookup_patient_directory_by_name,
        lookup_patient_directory_by_phone,
        patient_directory_db_enabled,
        upsert_patient_rows,
    )
    try:
        from patient_directory_db import patient_directory_database_url_configured
    except ImportError:
        def patient_directory_database_url_configured() -> bool:
            return bool(
                os.getenv("PATIENT_DIRECTORY_DATABASE_URL")
                or os.getenv("RECEPTION_DATABASE_URL")
                or os.getenv("DATABASE_URL")
            )

    try:
        from patient_directory_db import patient_directory_driver_available
    except ImportError:
        def patient_directory_driver_available() -> bool:
            try:
                import psycopg  # noqa: F401
                return True
            except Exception:
                return False
except (ModuleNotFoundError, ImportError) as exc:
    print(f"WARNING: patient_directory_db unavailable ({exc}); DB cache features disabled.")

    def patient_directory_database_url_configured() -> bool:
        return False

    def patient_directory_db_enabled() -> bool:
        return False

    def patient_directory_driver_available() -> bool:
        return False

    def init_patient_directory_db() -> bool:
        return False

    def upsert_patient_rows(rows):
        return {"success": False, "inserted": 0, "updated": 0, "error": "patient_directory_db module not available"}

    def lookup_patient_directory_by_phone(phone_number: str | None):
        return None

    def lookup_patient_directory_by_name(first_name: str | None, last_name: str | None):
        return None

try:
    from appointment_utils import (
        normalize_clinic_name,
        normalize_day_preference,
        normalize_time_preference,
    )
except Exception:
    print("WARNING: normalize_* helpers missing in appointment_utils; using local fallback.")

    def _strip_accents(value: str) -> str:
        normalized = unicodedata.normalize("NFD", value)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    def _normalize_token(value: str) -> str:
        value = _strip_accents((value or "").strip().lower())
        value = re.sub(r"[^a-z0-9α-ω\s]", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def normalize_day_preference(value: str | None, default: str = "any") -> str:
        token = _normalize_token(value or "")
        mapping = {
            "today": {"today", "σημερα", "simera"},
            "tomorrow": {"tomorrow", "αυριο", "avr"},
            "monday": {"monday", "δευτερα"},
            "tuesday": {"tuesday", "τριτη"},
            "wednesday": {"wednesday", "τεταρτη"},
            "thursday": {"thursday", "πεμπτη"},
            "friday": {"friday", "παρασκευη"},
        }
        for key, vals in mapping.items():
            if token in vals:
                return key
        return default

    def normalize_time_preference(value: str | None, default: str = "any") -> str:
        token = _normalize_token(value or "")
        if token in {"morning", "πρωι", "πρωινο", "πρωις"}:
            return "morning"
        if token in {"noon", "μεσημερι"}:
            return "noon"
        if token in {
            "after_16",
            "afternoon",
            "απογευμα",
            "μετα τις 4",
            "μετα τις 5",
            "μετα τις πεντε",
            "after 5",
            "after five",
        }:
            return "after_16"
        return default

    def normalize_clinic_name(value: str | None, default: str = "") -> str:
        token = _normalize_token(value or "")
        if token in {
            "limassol",
            "λεμεσος",
            "λεμεσο",
            "λεμεσου",
            "βασιλειος κωνσταντινου",
            "βασιλειου κωνσταντινου",
            "βασιλειος παυλου",
            "βασιλειου παυλου",
            "βασιλιου κωνσταντινου",
            "βασιλιου παυλου",
            "βασιλεως κωνσταντινου",
            "βασιλεωςκωνσταντινου",
        }:
            return "limassol"
        if token in {"evrychou", "eurychou", "ευρυχου", "ευριχου", "ευρυχωρο", "ευρυχωρου"}:
            return "evrychou"
        return default

PRE_ROUTE_PATIENT_LOOKUP = _env_bool("PRE_ROUTE_PATIENT_LOOKUP", default=False) and not RECEPTION_STABILITY_MODE
PRE_ROUTE_PATIENT_LOOKUP_TIMEOUT_SECONDS = float(
    os.getenv("PRE_ROUTE_PATIENT_LOOKUP_TIMEOUT_SECONDS", "0.35")
)

CY_MLM_MODEL_NAME = os.getenv("CY_MLM_MODEL_NAME", "petros/bert-base-cypriot-uncased-v1")
if ENABLE_CY_MLM:
    print("Loading Cypriot masked LM model:", CY_MLM_MODEL_NAME)
    try:
        cy_mlm_tokenizer = AutoTokenizer.from_pretrained(CY_MLM_MODEL_NAME)
        cy_mlm_model = AutoModelForMaskedLM.from_pretrained(CY_MLM_MODEL_NAME)
        cy_mlm_model.eval()
        CY_MLM_READY = True
        print("Cypriot masked LM loaded successfully.")
    except Exception as e:
        print("ERROR loading Cypriot masked LM:", e)
        cy_mlm_tokenizer = None
        cy_mlm_model = None
        CY_MLM_READY = False
else:
    print("Cypriot masked LM disabled (ENABLE_CY_MLM=false).")
    cy_mlm_tokenizer = None
    cy_mlm_model = None
    CY_MLM_READY = False


app = FastAPI(
    title="Ortho Reception Backend - Synthflow Only",
    description="Backend για Natasa (Synthflow actions: get_availability, book, message)",
    version="3.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# #8 Action authentication
# ============================================================
# Προαιρετικό shared-secret. Αν οριστεί το env var ACTIONS_SHARED_SECRET,
# κάθε /actions/* και /dograh/webhook κλήση πρέπει να φέρει το ίδιο token
# στο header X-Reception-Token. Αν ΔΕΝ οριστεί (π.χ. τοπικά), ο έλεγχος
# παρακάμπτεται ώστε να μη σπάσει τίποτα — βάλε το secret στο Render
# και στα Dograh custom_headers για να ενεργοποιηθεί.

ACTIONS_SHARED_SECRET = (os.getenv("ACTIONS_SHARED_SECRET") or "").strip()
ACTION_TOKEN_HEADER = "X-Reception-Token"


async def require_action_token(request: Request) -> None:
    if not ACTIONS_SHARED_SECRET:
        return  # auth disabled (no secret configured)
    provided = (request.headers.get(ACTION_TOKEN_HEADER) or "").strip()
    # Επιτρέπουμε και Authorization: Bearer <token> ως εναλλακτική
    if not provided:
        auth = (request.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    if not hmac.compare_digest(provided, ACTIONS_SHARED_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")


async def _setmore_patient_cache_refresh_loop() -> None:
    initial_delay = int(os.getenv("SETMORE_PATIENT_CACHE_INITIAL_DELAY_SECONDS", "10"))
    refresh_interval = int(os.getenv("SETMORE_PATIENT_CACHE_REFRESH_SECONDS", "86400"))
    await asyncio.sleep(max(initial_delay, 0))
    while True:
        try:
            if patient_directory_db_enabled():
                init_patient_directory_db()
                rows, appointments_scanned = await fetch_setmore_patient_rows()
                db_result = upsert_patient_rows(rows)
                result = {
                    "success": True,
                    "storage": "postgres",
                    "appointments_scanned": appointments_scanned,
                    "patients_upserted": db_result.get("upserted", 0),
                }
                print("SETMORE PATIENT DIRECTORY REFRESH:", json.dumps(result, ensure_ascii=False))
            else:
                result = await refresh_setmore_patient_cache()
                result["storage"] = "csv"
                print("SETMORE PATIENT CACHE REFRESH:", json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            print("SETMORE PATIENT DIRECTORY REFRESH ERROR:", str(exc))
        await asyncio.sleep(max(refresh_interval, 300))


@app.on_event("startup")
async def _start_setmore_patient_cache_refresh() -> None:
    enabled = os.getenv("SETMORE_PATIENT_CACHE_AUTO_REFRESH", "true").strip().lower()
    print(
        "PATIENT DIRECTORY STARTUP:",
        json.dumps(
            {
                "auto_refresh": enabled not in {"0", "false", "no", "off"},
                "database_url_configured": patient_directory_database_url_configured(),
                "driver_available": patient_directory_driver_available(),
                "postgres_enabled": patient_directory_db_enabled(),
                "pre_route_lookup": PRE_ROUTE_PATIENT_LOOKUP,
            },
            ensure_ascii=False,
        ),
    )
    if enabled in {"0", "false", "no", "off"}:
        print("SETMORE PATIENT DIRECTORY REFRESH: disabled by SETMORE_PATIENT_CACHE_AUTO_REFRESH")
        return
    asyncio.create_task(_setmore_patient_cache_refresh_loop())


@app.on_event("startup")
async def _init_conversation_state_store() -> None:
    try:
        from state_store import init_state_store
        init_state_store()
    except Exception as exc:
        print(f"STATE STORE init skipped ({exc}); using in-memory state.")
    try:
        from call_metrics import init_call_metrics
        init_call_metrics()
    except Exception as exc:
        print(f"CALL METRICS init skipped ({exc}); CSV only.")


async def _conversation_state_cleanup_loop() -> None:
    """
    Περιοδικό καθάρισμα ληγμένων conversation states (memory + DB).

    ΣΗΜΑΝΤΙΚΟ: το state_store.cleanup_expired() δεν καλείται ποτέ πουθενά
    αλλού — χωρίς αυτό το background loop, το in-memory _MEMORY_STATE dict
    μεγαλώνει επ' άπειρον (κάθε νέος καλών προσθέτει entry που έμενε για
    πάντα στη μνήμη μέχρι restart), προσθέτοντας στην πίεση μνήμης που
    συνέβαλε στα OOM kills.
    """
    interval = int(os.getenv("CONVERSATION_STATE_CLEANUP_INTERVAL_SECONDS", "1800"))  # 30 λεπτά
    await asyncio.sleep(60)  # μικρή αρχική αναμονή μετά το startup
    while True:
        try:
            from state_store import cleanup_expired
            deleted = cleanup_expired()
            if deleted:
                print(f"STATE STORE CLEANUP: removed {deleted} expired entries.")
        except Exception as exc:
            print(f"STATE STORE CLEANUP error: {exc}")
        await asyncio.sleep(max(interval, 300))


@app.on_event("startup")
async def _start_conversation_state_cleanup() -> None:
    asyncio.create_task(_conversation_state_cleanup_loop())


# ── Health/uptime tracking για Cal.com + Setmore (για το dashboard insights) ──
# ΣΗΜΑΝΤΙΚΟ: ΞΕΧΩΡΙΣΤΟ από το live call path. Καλεί τα εξωτερικά APIs ΑΡΑΙΑ
# (κάθε 15 λεπτά, όχι σε κάθε κλήση ασθενούς) ώστε να ΜΗΝ ξαναεισάγει το ίδιο
# πρόβλημα σταθερότητας που είχαμε με το /health/detailed (ζωντανά εξωτερικά
# calls μέσα στο live request path). Αποθηκεύει αποτελέσματα σε bounded
# in-memory ιστορικό για να υπολογίζεται % uptime τελευταίων 24 ωρών.
from collections import deque as _deque_for_health
import threading as _threading_for_health

_HEALTH_HISTORY: _deque_for_health = _deque_for_health(maxlen=200)  # ~50 ώρες με interval 15 λεπτών
_HEALTH_HISTORY_LOCK = _threading_for_health.Lock()


async def _check_external_health_once() -> dict:
    """Ένας γύρος ελέγχου Cal.com + Setmore, με σύντομο timeout. Δεν πετάει exception."""
    import time as _time
    result = {"checked_at": datetime.now(tz=timezone.utc).isoformat()}
    try:
        t0 = _time.monotonic()
        from cal_config import CAL_BASE_URL
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(
                f"{CAL_BASE_URL}/event-types",
                headers={
                    "Authorization": f"Bearer {os.getenv('CAL_API_KEY', '')}",
                    "cal-api-version": "2024-06-14",
                },
            )
        result["cal_com_ok"] = (resp.status_code == 200)
        result["cal_com_latency_ms"] = int((_time.monotonic() - t0) * 1000)
    except Exception:
        result["cal_com_ok"] = False
        result["cal_com_latency_ms"] = None

    try:
        t0 = _time.monotonic()
        from setmore_client import get_setmore_headers
        headers = await asyncio.wait_for(get_setmore_headers(), timeout=4.0)
        result["setmore_ok"] = bool(headers)
        result["setmore_latency_ms"] = int((_time.monotonic() - t0) * 1000)
    except Exception:
        result["setmore_ok"] = False
        result["setmore_latency_ms"] = None

    return result


async def _external_health_polling_loop() -> None:
    interval = int(os.getenv("EXTERNAL_HEALTH_POLL_INTERVAL_SECONDS", "900"))  # 15 λεπτά
    await asyncio.sleep(30)  # μικρή αρχική αναμονή μετά το startup
    while True:
        try:
            result = await _check_external_health_once()
            with _HEALTH_HISTORY_LOCK:
                _HEALTH_HISTORY.append(result)
        except Exception as exc:
            print(f"EXTERNAL HEALTH POLL error: {exc}")
        await asyncio.sleep(max(interval, 300))


@app.on_event("startup")
async def _start_external_health_polling() -> None:
    enabled = os.getenv("EXTERNAL_HEALTH_POLL_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return
    asyncio.create_task(_external_health_polling_loop())

# ============================================================
# Config: Google Calendar
# ============================================================

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
CALENDAR_ID = os.getenv("CALENDAR_ID")  # π.χ. "athpapachr@gmail.com"

if not GOOGLE_SERVICE_ACCOUNT_JSON:
    print("WARNING: GOOGLE_SERVICE_ACCOUNT_JSON is not set")
if not CALENDAR_ID:
    print("WARNING: CALENDAR_ID is not set")
else:
    print("Using CALENDAR_ID:", CALENDAR_ID)


def get_gcal_session() -> AuthorizedSession:
    """
    Δημιουργεί AuthorizedSession για Google APIs μέσω service account.
    Το JSON key περνάει από env var GOOGLE_SERVICE_ACCOUNT_JSON.
    """
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON env var")

    service_account_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    authed_session = AuthorizedSession(credentials)
    return authed_session


def create_gcal_event(
    calendar_id: str,
    start_iso: str,
    end_iso: str,
    summary: str,
    description: str,
    timezone_str: str = "Asia/Nicosia",
) -> str:
    """
    Δημιουργεί event στο Google Calendar και επιστρέφει το event id.
    """
    if not calendar_id:
        raise RuntimeError("Missing CALENDAR_ID env var")

    session = get_gcal_session()

    event_body = {
        "summary": summary,
        "description": description,
        "start": {
            "dateTime": start_iso,
            "timeZone": timezone_str,
        },
        "end": {
            "dateTime": end_iso,
            "timeZone": timezone_str,
        },
    }

    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
    print("GCAL REQUEST URL:", url)
    print("GCAL EVENT BODY:", event_body)

    resp = session.post(url, json=event_body)

    if resp.status_code not in (200, 201):
        print("GCAL ERROR BODY:", resp.text)
        raise RuntimeError(
            f"Google Calendar API error {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    print("GCAL RESPONSE:", data)
    return data.get("id", "")


def to_google_time(dt: datetime) -> str:
    """
    Μετατρέπει datetime σε ISO8601 UTC με 'Z', όπως το θέλει πιο σίγουρα το Google.
    """
    # Αν το dt δεν έχει tzinfo, το θεωρούμε local και το κάνουμε aware
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(tz=None).isoformat().replace("+00:00", "Z")


def get_gcal_events(
    calendar_id: str,
    time_min: datetime,
    time_max: datetime,
) -> List[dict]:
    """
    Φέρνει όλα τα events από το Google Calendar στο διάστημα [time_min, time_max].
    """
    if not calendar_id:
        raise RuntimeError("Missing CALENDAR_ID env var")

    session = get_gcal_session()

    time_min_str = to_google_time(time_min)
    time_max_str = to_google_time(time_max)

    url = (
        f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
        f"?timeMin={time_min_str}&timeMax={time_max_str}&singleEvents=true&orderBy=startTime"
    )

    print("GCAL LIST EVENTS URL:", url)

    resp = session.get(url)
    if resp.status_code != 200:
        print("GCAL LIST EVENTS ERROR:", resp.text)
        raise RuntimeError(
            f"Google Calendar API (list events) error {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    items = data.get("items", [])
    print(f"GCAL LIST EVENTS: fetched {len(items)} events")
    return items

# ============================================================
# Config: Brevo (SMTP via API)
# ============================================================

BREVO_API_KEY = os.getenv("BREVO_API_KEY")
BREVO_SENDER_EMAIL = os.getenv("BREVO_SENDER_EMAIL")
BREVO_SENDER_NAME = os.getenv("BREVO_SENDER_NAME", "Ortho Papachristou")
BREVO_RECIPIENT_EMAIL = os.getenv("BREVO_RECIPIENT_EMAIL")
PRIORITY_ALERT_RECIPIENT_EMAIL = os.getenv("PRIORITY_ALERT_RECIPIENT_EMAIL") or BREVO_RECIPIENT_EMAIL
PRIORITY_ALERT_WEBHOOK_URL = os.getenv("PRIORITY_ALERT_WEBHOOK_URL", "").strip()
CALL_WATCHDOG_WEBHOOK_URL = os.getenv("CALL_WATCHDOG_WEBHOOK_URL", "").strip()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
DASHBOARD_KEY = os.getenv("DASHBOARD_KEY", "").strip()
SYNC_ADMIN_TOKEN = os.getenv("SYNC_ADMIN_TOKEN", "").strip()
DOGRAH_API_KEY = os.getenv("DOGRAH_API_KEY", "").strip()
DOGRAH_CALL_DETAIL_URL_TEMPLATE = os.getenv(
    "DOGRAH_CALL_DETAIL_URL_TEMPLATE",
    "https://app.dograh.com/api/v1/workflow/{workflow_id}/runs/{run_id}",
).strip()


def _run_setmore_to_cal_sync_safely() -> tuple[bool, int | None, str | None]:
    """
    ⚠️ ΔΕΝ χρειάζεται πια να καλείται από τα endpoints — σκάει σε async context
    (asyncio.run nested error, επιβεβαιώθηκε στο /sync/both). Δες
    _run_setmore_to_cal_sync_async() παρακάτω.

    Καλεί in-process το setmore_to_cal_sync.main(), προστατεύοντας από δύο
    προβλήματα που προκαλούσαν HTTP 500 στο /admin/trigger-sync και
    /sync/both:

    1. Το main() καλεί argparse.parse_args() χωρίς ρητά arguments, που
       διαβάζει by default το sys.argv[1:]. Επειδή αυτή η κλήση γίνεται
       in-process μέσα στο ΙΔΙΟ process που ξεκίνησε με
       `uvicorn main:app --host 0.0.0.0 --port 10000`, το sys.argv εδώ
       περιέχει ΑΥΤΑ τα uvicorn arguments — το argparse τα βλέπει σαν δικά
       του, αποτυγχάνει με «unrecognized arguments», και καλεί sys.exit(2).
       Λύση: καθαρίζουμε προσωρινά το sys.argv πριν την κλήση.
    2. Το argparse σε σφάλμα κάνει sys.exit(...), που πετάει SystemExit —
       μια κλάση που ΔΕΝ είναι Exception (κληρονομεί απευθείας από
       BaseException), άρα ένα απλό `except Exception:` γύρω από την
       κλήση ΔΕΝ το πιάνει· το σφάλμα ανέβαινε ως unhandled, ο FastAPI
       handler το έβλεπε ως πραγματικό crash, και επέστρεφε 500.
       Λύση: πιάνουμε ρητά και BaseException/SystemExit εδώ.

    Επιστρέφει (success, exit_code, error_message).
    """
    from setmore_to_cal_sync import main as setmore_to_cal_main

    original_argv = sys.argv
    sys.argv = [sys.argv[0] if sys.argv else "setmore_to_cal_sync.py"]
    try:
        exit_code = setmore_to_cal_main()
        return (exit_code == 0, exit_code, None)
    except SystemExit as exc:
        # argparse (ή ρητό sys.exit) — δεν είναι Exception, πιάνεται ξεχωριστά.
        code = exc.code if isinstance(exc.code, int) else 1
        return (False, code, f"SystemExit: {exc.code}")
    except Exception as exc:
        return (False, None, str(exc)[:300])
    finally:
        sys.argv = original_argv
_SYNC_BOTH_LOCK = asyncio.Lock()


async def _run_setmore_to_cal_sync_async() -> tuple[bool, int | None, str | None]:
    """
    Η σωστή, async-native έκδοση. Καλεί απευθείας το run_backfill() με
    await, παρακάμπτοντας πλήρως το argparse/asyncio.run() layer του
    setmore_to_cal_sync.py's main() — κανένα νέο event loop δεν φτιάχνεται,
    άρα λειτουργεί σωστά μέσα από ένα ήδη τρέχον FastAPI request handler.

    Αναπαράγει τις ίδιες default τιμές με το CLI script (setmore_to_cal_sync.py).
    """
    from setmore_to_cal_backfill import run_backfill

    try:
        tz = ZoneInfo("Asia/Nicosia")
        today_local = datetime.now(tz).date()
        from_date = today_local
        lookahead_days = int(os.getenv("SETMORE_TO_CAL_LOOKAHEAD_DAYS", "14"))
        to_date = today_local + timedelta(days=lookahead_days)

        result = await run_backfill(
            from_date=from_date,
            to_date=to_date,
            apply=True,
            limit=int(os.getenv("SETMORE_TO_CAL_LIMIT", "0")),
            requested_clinic="both",
            default_clinic=(os.getenv("SETMORE_DEFAULT_CLINIC", "").strip().lower() or None),
            include_today=True,
            setmore_page_limit=500,
            cal_page_size=100,
            cal_max_pages=50,
            cal_existing_status="upcoming",
            pause_ms=int(os.getenv("SETMORE_TO_CAL_PAUSE_MS", "350")),
            cal_api_key=None,
            limassol_event_type_id=341357,
            evrychou_event_type_id=341358,
            clinic_by_staff={},
            clinic_by_service={},
        )
        error_msg = None
        if not result.success:
            error_msg = "; ".join(result.create_errors[:5]) or "unknown error"
        return (result.success, 0 if result.success else 1, error_msg)
    except Exception as exc:
        return (False, None, str(exc)[:300])


ZADARMA_API_KEY = os.getenv("ZADARMA_API_KEY", "").strip()
ZADARMA_API_SECRET = os.getenv("ZADARMA_API_SECRET", "").strip()
ZADARMA_CALLBACK_FROM = os.getenv("ZADARMA_CALLBACK_FROM", "").strip()
ZADARMA_CALLBACK_SIP = os.getenv("ZADARMA_CALLBACK_SIP", "").strip()
ZADARMA_CALLBACK_PREDICTED = _env_bool("ZADARMA_CALLBACK_PREDICTED", default=False)
ZADARMA_SMS_SENDER = os.getenv("ZADARMA_SMS_SENDER", "").strip()
ZADARMA_ENABLE_CALLBACK = _env_bool("ZADARMA_ENABLE_CALLBACK", default=False)
ZADARMA_ENABLE_SMS = _env_bool("ZADARMA_ENABLE_SMS", default=False)
ZADARMA_LAST_ERROR: str = ""

_ZADARMA_CONFIGURED = bool(ZADARMA_API_KEY and ZADARMA_API_SECRET)
print(
    "ZADARMA CONFIG: "
    f"configured={_ZADARMA_CONFIGURED} "
    f"callback_enabled={ZADARMA_ENABLE_CALLBACK} "
    f"callback_from={'set' if ZADARMA_CALLBACK_FROM else 'MISSING'} "
    f"callback_sip={'set' if ZADARMA_CALLBACK_SIP else 'MISSING'} "
    f"api_key={'set' if ZADARMA_API_KEY else 'MISSING'} "
    f"api_secret={'set' if ZADARMA_API_SECRET else 'MISSING'}"
)

if not BREVO_API_KEY:
    print("WARNING: BREVO_API_KEY is not set")
if not BREVO_SENDER_EMAIL:
    print("WARNING: BREVO_SENDER_EMAIL is not set")
if not BREVO_RECIPIENT_EMAIL:
    print("WARNING: BREVO_RECIPIENT_EMAIL is not set")


def send_brevo_email(
    subject: str,
    text_content: str,
    html_content: Optional[str] = None,
    recipient_email: Optional[str] = None,
) -> None:
    """
    Στέλνει email μέσω Brevo SMTP API.
    """
    target_email = recipient_email or BREVO_RECIPIENT_EMAIL
    if not (BREVO_API_KEY and BREVO_SENDER_EMAIL and target_email):
        raise RuntimeError("Missing Brevo configuration env vars")

    url = "https://api.brevo.com/v3/smtp/email"

    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    payload = {
        "sender": {
            "email": BREVO_SENDER_EMAIL,
            "name": BREVO_SENDER_NAME,
        },
        "to": [
            {
                "email": target_email,
            }
        ],
        "subject": subject,
        "textContent": text_content,
    }

    if html_content:
        payload["htmlContent"] = html_content

    print("BREVO PAYLOAD:", payload)

    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code not in (200, 201, 202):
        print("BREVO ERROR BODY:", resp.text)
        raise RuntimeError(f"Brevo API error {resp.status_code}: {resp.text}")

    print("BREVO RESPONSE:", resp.text)


def send_slack_notification(
    subject: str,
    text_content: str,
    emoji: str = ":telephone_receiver:",
    urgent: bool = False,
) -> None:
    """
    Στέλνει Slack notification μέσω Incoming Webhook.
    Αν δεν υπάρχει SLACK_WEBHOOK_URL, κάνει skip σιωπηλά.
    """
    if not SLACK_WEBHOOK_URL:
        return

    color = "#FF0000" if urgent else "#36a64f"
    icon = ":rotating_light:" if urgent else emoji

    # Μορφοποίηση: subject ως header, text ως body
    lines = text_content.strip().split("\n")
    body_text = "\n".join(f"> {line}" for line in lines if line.strip())

    payload = {
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{icon}  {subject}",
                            "emoji": True,
                        },
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": body_text or "_χωρίς λεπτομέρειες_",
                        },
                    },
                ],
            }
        ]
    }

    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            json=payload,
            timeout=5.0,
        )
        if resp.status_code != 200:
            print(f"SLACK ERROR {resp.status_code}: {resp.text[:200]}")
        else:
            print("SLACK OK")
    except Exception as exc:
        print(f"SLACK EXCEPTION: {exc}")


async def _enrich_call_audit_from_dograh(
    call_id: str,
    workflow_id: int,
    run_id: int,
) -> None:
    """Fetch optional Dograh call details after the call, never in the voice path."""
    if not call_id or not workflow_id or not run_id or not DOGRAH_API_KEY or not DOGRAH_CALL_DETAIL_URL_TEMPLATE:
        return
    await asyncio.sleep(max(int(os.getenv("DOGRAH_CALL_DETAIL_DELAY_SECONDS", "15")), 0))
    try:
        url = DOGRAH_CALL_DETAIL_URL_TEMPLATE.format(
            call_id=call_id,
            workflow_id=workflow_id,
            run_id=run_id,
        )
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                url,
                headers={"X-API-Key": DOGRAH_API_KEY},
            )
        response.raise_for_status()
        detail = response.json()
        if not isinstance(detail, dict):
            return
        from call_audit import save_reasoning_enrichment

        result = await asyncio.to_thread(save_reasoning_enrichment, call_id, detail)
        print(
            "DOGRAH REASONING ENRICHMENT: "
            f"call_id={call_id} delays={len(result.get('reasoning_delays_ms') or [])} "
            f"total_ms={result.get('reasoning_total_ms')}"
        )
    except Exception as exc:
        print(f"DOGRAH REASONING ENRICHMENT skipped for {call_id}: {exc}")


async def _call_quality_monitor_loop() -> None:
    initial_delay = int(os.getenv("CALL_QUALITY_INITIAL_DELAY_SECONDS", "120"))
    interval = int(os.getenv("CALL_QUALITY_CHECK_INTERVAL_SECONDS", "3600"))
    await asyncio.sleep(max(initial_delay, 0))
    while True:
        try:
            from call_audit import audit_summary
            from call_quality import (
                due_quality_alerts,
                mark_quality_alerts_sent,
                write_due_snapshots,
            )

            await asyncio.to_thread(write_due_snapshots)
            summary = await asyncio.to_thread(audit_summary, 7)
            alerts = await asyncio.to_thread(due_quality_alerts, summary)
            if alerts:
                lines = [
                    "Το εβδομαδιαίο scorecard ξεπέρασε τα συμφωνημένα όρια:",
                    f"Κλήσεις δείγματος: {summary.get('calls', 0)}",
                ]
                for item in alerts:
                    unit = "ms" if item["metric"] == "p95_reasoning" else "%"
                    direction = "κάτω από" if item["direction"] == "below" else "πάνω από"
                    lines.append(
                        f"- {item['label']}: {item['value']}{unit} "
                        f"({direction} όριο {item['threshold']}{unit})"
                    )
                text = "\n".join(lines)
                delivered = False
                if SLACK_WEBHOOK_URL:
                    await asyncio.to_thread(
                        send_slack_notification,
                        subject="Voice agent quality alert",
                        text_content=text,
                        emoji=":warning:",
                        urgent=True,
                    )
                    delivered = True
                if BREVO_API_KEY and BREVO_SENDER_EMAIL and BREVO_RECIPIENT_EMAIL:
                    await asyncio.to_thread(
                        send_brevo_email,
                        subject="Voice agent quality alert",
                        text_content=text,
                    )
                    delivered = True
                if delivered:
                    await asyncio.to_thread(mark_quality_alerts_sent, alerts)
                else:
                    print("CALL QUALITY ALERT pending: no Slack/Brevo destination configured")
        except Exception as exc:
            print(f"CALL QUALITY MONITOR error: {exc}")
        await asyncio.sleep(max(interval, 900))


async def _upcoming_booking_directory(days: int = 14) -> list[dict[str, Any]]:
    now_local = datetime.now(TZ_NICOSIA)
    end_local = now_local + timedelta(days=max(1, days))
    response = await CalClient().get_bookings(
        status="upcoming",
        take=100,
        skip=0,
        sort_start="asc",
    )
    bookings = response.get("data", response)
    if isinstance(bookings, dict):
        bookings = bookings.get("bookings") or []
    result: list[dict[str, Any]] = []
    for booking in bookings if isinstance(bookings, list) else []:
        start = booking.get("start") or booking.get("startTime") or ""
        try:
            start_local = datetime.fromisoformat(
                str(start).replace("Z", "+00:00")
            ).astimezone(TZ_NICOSIA)
        except Exception:
            continue
        if not (now_local <= start_local <= end_local):
            continue
        attendees = booking.get("attendees") or []
        attendee = attendees[0] if attendees else {}
        fields = booking.get("bookingFieldsResponses") or {}
        phone = (
            attendee.get("phoneNumber")
            or fields.get("attendeePhoneNumber")
            or ""
        )
        if not phone:
            email = str(attendee.get("email") or fields.get("email") or "")
            prefix = email.split("@", 1)[0]
            phone = prefix if prefix.isdigit() else ""
        text = " ".join(
            str(booking.get(key) or "")
            for key in ("title", "location", "meetingUrl")
        ).lower()
        clinic = (
            "evrychou"
            if "evrychou" in text or "ευρύχου" in text
            else "limassol"
            if "limassol" in text or "λεμεσ" in text
            else ""
        )
        result.append(
            {
                "phone": phone,
                "name": attendee.get("name") or fields.get("name") or "",
                "start": start_local.isoformat(),
                "friendly_start": (
                    f"{make_friendly_day_gr(start_local)} "
                    f"{make_friendly_time_gr(start_local)}"
                ),
                "clinic": clinic,
            }
        )
    return result


async def _build_end_of_day_digest(target_date: date) -> dict[str, Any]:
    from call_audit import load_call_audits
    from daily_call_digest import (
        build_daily_call_digest,
        render_daily_call_digest_text,
    )

    audits, bookings = await asyncio.gather(
        asyncio.to_thread(load_call_audits, 3),
        _upcoming_booking_directory(
            int(os.getenv("DAILY_CALL_DIGEST_LOOKAHEAD_DAYS", "14"))
        ),
    )
    digest = build_daily_call_digest(
        audits,
        bookings,
        target_date=target_date.isoformat(),
    )
    digest["text"] = render_daily_call_digest_text(digest)
    return digest


async def _daily_call_digest_loop() -> None:
    if not _env_bool("DAILY_CALL_DIGEST_ENABLED", default=False):
        print("DAILY CALL DIGEST: disabled")
        return
    hour = int(os.getenv("DAILY_CALL_DIGEST_HOUR", "19"))
    minute = int(os.getenv("DAILY_CALL_DIGEST_MINUTE", "30"))
    last_sent_date = ""
    while True:
        now_local = datetime.now(TZ_NICOSIA)
        today = now_local.date().isoformat()
        if (
            now_local.hour == hour
            and now_local.minute >= minute
            and last_sent_date != today
        ):
            try:
                digest = await _build_end_of_day_digest(now_local.date())
                text = str(digest.get("text") or "")
                if SLACK_WEBHOOK_URL:
                    await asyncio.to_thread(
                        send_slack_notification,
                        subject=f"Ημερήσια σύνοψη κλήσεων {today}",
                        text_content=text,
                        emoji=":clipboard:",
                    )
                if BREVO_API_KEY and BREVO_SENDER_EMAIL and BREVO_RECIPIENT_EMAIL:
                    await asyncio.to_thread(
                        send_brevo_email,
                        subject=f"Ημερήσια σύνοψη κλήσεων {today}",
                        text_content=text,
                    )
                last_sent_date = today
                print(
                    "DAILY CALL DIGEST sent: "
                    f"date={today} callbacks={digest.get('callbacks_needed')}"
                )
            except Exception as exc:
                print(f"DAILY CALL DIGEST error: {exc}")
        await asyncio.sleep(60)


@app.on_event("startup")
async def _start_call_quality_monitor() -> None:
    enabled = os.getenv("CALL_QUALITY_MONITOR_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        print("CALL QUALITY MONITOR: disabled")
    else:
        asyncio.create_task(_call_quality_monitor_loop())
    asyncio.create_task(_daily_call_digest_loop())



# ============================================================
# Cypriot BERT – Embeddings
# ============================================================

CY_BERT_MODEL_NAME = os.getenv("CY_BERT_MODEL_NAME", "petros/bert-base-cypriot-uncased-v1")
if ENABLE_CY_BERT:
    print("Loading Cypriot BERT encoder:", CY_BERT_MODEL_NAME)
    try:
        cy_tokenizer = AutoTokenizer.from_pretrained(CY_BERT_MODEL_NAME)
        cy_model = AutoModel.from_pretrained(CY_BERT_MODEL_NAME)
        cy_model.eval()
        CY_BERT_READY = True
        print("Cypriot BERT encoder loaded successfully.")
    except Exception as e:
        print("ERROR loading Cypriot BERT encoder:", e)
        cy_tokenizer = None
        cy_model = None
        CY_BERT_READY = False
else:
    print("Cypriot BERT encoder disabled (ENABLE_CY_BERT=false).")
    cy_tokenizer = None
    cy_model = None
    CY_BERT_READY = False


def encode_cypriot_text(text: str) -> list[float]:
    if not CY_BERT_READY or cy_tokenizer is None or cy_model is None:
        raise RuntimeError("Cypriot BERT encoder is not loaded")

    if not text:
        return []

    with torch.no_grad():
        inputs = cy_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        outputs = cy_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding[0].tolist()

# ============================================================
# Cypriot Intent Classifier – Fine-tuned local model
# ============================================================

CY_INTENT_MODEL_PATH = os.getenv(
    "CY_INTENT_MODEL_PATH",
    "/data/cypriot-intent-model"  # προσαρμόζεις αν το mount είναι άλλο
)

DEFAULT_CY_INTENT_LABELS = [
    "appointment",
    "message",
    "confirmation",
    "change",
    "cancellation",
    "urgent",
    "other",
]
CY_INTENT_CONFIDENCE_THRESHOLD = float(
    os.getenv("CY_INTENT_CONFIDENCE_THRESHOLD", "0.70")
)
CY_INTENT_LABEL_ALIASES = {
    "cancelation": "cancellation",
    "emergency": "urgent",
    "results": "message",
}


def normalize_intent_label(label: str) -> str:
    label = (label or "").strip().lower()
    return CY_INTENT_LABEL_ALIASES.get(label, label)


cy_intent_id2label = {
    idx: label for idx, label in enumerate(DEFAULT_CY_INTENT_LABELS)
}

CY_INTENT_ONNX_PATH = os.getenv("CY_INTENT_ONNX_PATH", "/data/cypriot_intent.onnx")
CY_INTENT_TOKENIZER_PATH = os.getenv(
    "CY_INTENT_TOKENIZER_PATH", "/data/onnx_tokenizer/tokenizer.json"
)

print("Loading Cypriot intent model (ONNX) from disk:", CY_INTENT_ONNX_PATH)
if ENABLE_CY_INTENT:
    try:
        cy_intent_session = _cy_intent_ort.InferenceSession(
            CY_INTENT_ONNX_PATH, providers=["CPUExecutionProvider"]
        )
        cy_intent_tokenizer = _CyIntentTokenizer.from_file(CY_INTENT_TOKENIZER_PATH)
        # ΣΗΜΕΙΩΣΗ: το πραγματικό tokenizer.json (από convert_to_onnx.py's
        # tokenizer.save_pretrained()) ΗΔΗ περιέχει αυτές τις ρυθμίσεις
        # (max_length=128, pad_id=0). Αυτά τα ρητά calls είναι ΣΚΟΠΙΜΑ
        # redundant — λειτουργούν ως safety net σε περίπτωση που μελλοντικά κάποιος
        # αλλάξει το conversion script. Idempotent, καμία βλάβη.
        cy_intent_tokenizer.enable_padding(length=128, pad_id=0, pad_token="[PAD]")
        cy_intent_tokenizer.enable_truncation(max_length=128)
        # Εδώ διαφέρει από το PyTorch path: το ONNX session δεν κουβαλάει id2label
        # (αυτό ζούσε στο model.config του PyTorch μοντέλου) — χρησιμοποιούμε
        # τη σταθερή σειρά labels με την οποία εκπαιδεύτηκε το μοντέλο
        # (DEFAULT_CY_INTENT_LABELS), προαιρετικά παρακαμπτόμενη από env var σε
        # JSON μορφή αν ποτέ χρειαστεί.
        _id2label_override = os.getenv("CY_INTENT_ID2LABEL_JSON", "").strip()
        if _id2label_override:
            cy_intent_id2label = {
                int(k): normalize_intent_label(v)
                for k, v in json.loads(_id2label_override).items()
            }
        else:
            cy_intent_id2label = {
                idx: label for idx, label in enumerate(DEFAULT_CY_INTENT_LABELS)
            }
        print("Cypriot intent labels:", cy_intent_id2label)
        CY_INTENT_READY = True
        print("Cypriot intent ONNX model loaded successfully from disk.")
    except Exception as e:
        print("ERROR loading Cypriot intent ONNX model from disk:", e)
        cy_intent_session = None
        cy_intent_tokenizer = None
        cy_intent_id2label = {
            idx: label for idx, label in enumerate(DEFAULT_CY_INTENT_LABELS)
        }
        CY_INTENT_READY = False
else:
    print("Cypriot intent model disabled (ENABLE_CY_INTENT=false).")
    cy_intent_session = None
    cy_intent_tokenizer = None
    cy_intent_id2label = {
        idx: label for idx, label in enumerate(DEFAULT_CY_INTENT_LABELS)
    }
    CY_INTENT_READY = False


CY_INTENT_PER_INTENT_THRESHOLDS = {
    "urgent": float(os.getenv("CY_INTENT_THRESHOLD_URGENT", "0.55")),
}


def _cy_intent_threshold_for(intent: str) -> float:
    """Επιστρέφει το κατώφλι confidence για το συγκεκριμένο intent label,
    ή το γενικό CY_INTENT_CONFIDENCE_THRESHOLD αν δεν υπάρχει override."""
    return CY_INTENT_PER_INTENT_THRESHOLDS.get(intent, CY_INTENT_CONFIDENCE_THRESHOLD)


def _softmax_numpy(logits):
    """Αριθμητικά ασφαλής softmax χωρίς PyTorch — μόνο numpy."""
    shifted = logits - _cy_intent_np.max(logits, axis=-1, keepdims=True)
    exp = _cy_intent_np.exp(shifted)
    return exp / _cy_intent_np.sum(exp, axis=-1, keepdims=True)


def predict_cypriot_intent(text: str) -> tuple[str, float]:
    """
    Τρέχει το fine-tuned Κυπριακό intent model (μέσω ONNX Runtime, ΧΩΡΙΣ
    PyTorch) και επιστρέφει (label, confidence).

    ΣΗΜΑΝΤΙΚΟ ΓΙΑ ΤΑΧΥΤΗΤΑ: αυτό καλείται ΜΟΝΟ ως fallback όταν το
    rule_based_intent() δεν είναι confident.
    """
    if not CY_INTENT_READY or cy_intent_tokenizer is None or cy_intent_session is None:
        raise RuntimeError("Cypriot intent model is not loaded")

    if not text:
        return "other", 0.0

    encoded = cy_intent_tokenizer.encode(text)
    input_ids = _cy_intent_np.array([encoded.ids], dtype=_cy_intent_np.int64)
    attention_mask = _cy_intent_np.array([encoded.attention_mask], dtype=_cy_intent_np.int64)

    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    session_input_names = {i.name for i in cy_intent_session.get_inputs()}
    if "token_type_ids" in session_input_names:
        onnx_inputs["token_type_ids"] = _cy_intent_np.zeros_like(input_ids)

    outputs = cy_intent_session.run(None, onnx_inputs)
    logits = outputs[0]
    probs = _softmax_numpy(logits)[0]
    idx = int(_cy_intent_np.argmax(probs))
    conf = float(probs[idx])
    label = cy_intent_id2label.get(idx, "other")
    return label, conf


# ============================================================
# Models
# ============================================================

class TimeSlot(BaseModel):
    start: datetime
    end: datetime


class GetAvailabilityRequest(BaseModel):
    clinic: str = Field(..., description="\"limassol\" ή \"evrychou\"")
    duration_minutes: int = Field(default=20, description="Διάρκεια ραντεβού σε λεπτά")
    max_slots: int = Field(default=25, description="Πόσα κοντινά διαθέσιμα slots να επιστρέψει")
    from_date: Optional[date | datetime] = None  # δέχεται και YYYY-MM-DD


class GetAvailabilityResponse(BaseModel):
    slots: List[TimeSlot]


class BookRequest(BaseModel):
    first_name: str
    last_name: str
    phone_number: str
    clinic: str              # "limassol" ή "evrychou"
    appointment_date: str    # π.χ. "2026-02-12"
    appointment_time: str    # π.χ. "11:30"
    duration_minutes: int = 20
    notes: Optional[str] = None

class CypriotEncodeRequest(BaseModel):
    text: str = Field(..., description="Κείμενο σε κυπριακά/ελληνικά για κωδικοποίηση")


class CypriotEncodeResponse(BaseModel):
    embedding: list[float]


class CypriotIntentRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: Optional[str] = Field(default=None, description="Αρχικό μήνυμα caller από το workflow.")
    message_clean: Optional[str] = Field(default=None, description="Επεξεργασμένο/καθαρό μήνυμα caller, αν υπάρχει.")
    text: Optional[str] = Field(default=None, description="Backward-compatible alias. Prefer message/message_clean.")
    caller_number: Optional[str] = Field(
        default=None,
        description="Από το webhook/Dograh {{caller_number}}. Ποτέ δεν ζητιέται από τον caller.",
    )
    phone_number: Optional[str] = Field(
        default=None,
        description="Backward-compatible alias. Prefer caller_number from webhook.",
    )
    call_id: Optional[str] = None


class CypriotIntentResponse(BaseModel):
    intent: str
    confidence: float
    threshold: float
    is_confident: bool
    should_ask_clarification: bool
    source: Optional[str] = None
    message_clean: Optional[str] = None
    message_category: Optional[str] = None
    priority_level: Optional[str] = None
    clinic: Optional[str] = None
    day_preference: Optional[str] = None
    time_preference: Optional[str] = None
    from_date: Optional[str] = None
    search_mode: Optional[str] = None
    appointment_triage_needed: Optional[bool] = None
    current_stage: Optional[str] = None
    next_action: Optional[str] = None
    name_collection_required: Optional[bool] = None
    latency_notice: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    patient_lookup_source: Optional[str] = None
    flow_stage: Optional[str] = None
    action_tool: Optional[str] = None
    agent_instruction: Optional[str] = None
    ask_text: Optional[str] = None
    state_machine: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None


class ConversationStateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    caller_number: Optional[str] = Field(
        default=None,
        description="Από το webhook/Dograh {{caller_number}}. Ποτέ δεν ζητιέται από τον caller.",
    )
    phone_number: Optional[str] = None
    call_id: Optional[str] = None
    message_already_captured: Optional[bool] = None
    message_sent: Optional[bool] = None
    current_intent: Optional[str] = None
    current_message: Optional[str] = None
    current_message_category: Optional[str] = None
    current_clinic: Optional[str] = None
    current_day_preference: Optional[str] = None
    current_time_preference: Optional[str] = None
    current_stage: Optional[str] = None
    next_action: Optional[str] = None
    name_collection_required: Optional[bool] = None
    patient_name_captured: Optional[bool] = None
    clinic_known: Optional[bool] = None
    caller_wants_callback: Optional[bool] = None
    booking_slot_offered: Optional[bool] = None
    booking_slot_confirmed: Optional[bool] = None
    selected_slot_confirmed: Optional[bool] = None
    booking_completed: Optional[bool] = None
    cancel_completed: Optional[bool] = None
    urgent_handoff_sent: Optional[bool] = None
    latency_notice_spoken: Optional[bool] = None


class ConversationStateResponse(BaseModel):
    success: bool
    state: Dict[str, Any]


class PreRouteIntentRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: Optional[str] = Field(default=None, description="Αρχικό μήνυμα caller από το workflow.")
    message_clean: Optional[str] = Field(default=None, description="Επεξεργασμένο/καθαρό μήνυμα caller, αν υπάρχει.")
    text: Optional[str] = Field(default=None, description="Backward-compatible alias. Prefer message/message_clean.")
    caller_number: Optional[str] = Field(
        default=None,
        description="Από το webhook/Dograh {{caller_number}}. Ποτέ δεν ζητιέται από τον caller.",
    )
    phone_number: Optional[str] = Field(
        default=None,
        description="Backward-compatible alias. Prefer caller_number from webhook.",
    )
    call_id: Optional[str] = None


class PreRouteIntentResponse(CypriotIntentResponse):
    next_node: Optional[str] = None
    route_label: Optional[str] = None
    route_locked: Optional[bool] = None
    should_speak: bool = False
    spoken_response: str = ""


class BackendRouteRequest(PreRouteIntentRequest):
    current_node: Optional[str] = None
    current_intent: Optional[str] = None
    clinic: Optional[str] = None
    day_preference: Optional[str] = None
    time_preference: Optional[str] = None
    from_date: Optional[str] = None
    selected_start_iso: Optional[str] = None
    selected_friendly_day: Optional[str] = None
    selected_friendly_time: Optional[str] = None
    notes: Optional[str] = None
    duration_minutes: Optional[int | str] = None
    booking_status: Optional[str] = None
    booking_uid: Optional[str] = None
    calendar_event_id: Optional[str] = None


class BackendRouteResponse(CypriotIntentResponse):
    success: bool = True
    next_node: str
    route_label: str
    route_locked: bool = False
    should_ask_clarification: bool
    should_speak: bool = False
    spoken_response: str = ""
    booking_can_be_announced: bool = False
    state: Optional[Dict[str, Any]] = None

class CancelRequest(BaseModel):
    first_name: str
    last_name: str
    phone_number: str
    clinic: str              # "limassol" ή "evrychou"
    appointment_day: int     # π.χ. 3
    appointment_month: int   # π.χ. 3
    appointment_year: Optional[int] = None
    appointment_time: str    # "HH:MM"
    duration_minutes: int = 20


class BookResponse(BaseModel):
    success: bool
    appointment_id: Optional[str] = None
    message: Optional[str] = None


class MessagePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    call_id: Optional[str] = None
    caller_number: Optional[str] = None
    phone_number: Optional[str] = None
    message: str
    reason_for_call: Optional[str] = None
    clinic: Optional[str] = None
    is_urgent: Optional[bool] = None
    priority_level: Optional[str] = None
    message_category: Optional[str] = None
    message_clean: Optional[str] = None
    language: Optional[str] = None
    notes: Optional[str] = None
    appointment_date: Optional[str] = None
    appointment_time: Optional[str] = None


class MessageResponse(BaseModel):
    success: bool
    message: str
    spoken_response: str = "Θα ενημερώσω τον γιατρό σχετικά."
    state: Optional[Dict[str, Any]] = None


class OfferedSlotOption(BaseModel):
    start_iso: Optional[str] = None
    friendly_day: Optional[str] = None
    friendly_time: Optional[str] = None


class SlotSelectionValidationRequest(BaseModel):
    caller_text: str = Field(default="", description="Τελευταία φράση caller μετά από προτεινόμενα slots.")
    offered_options: List[OfferedSlotOption] = Field(default_factory=list)


class SlotSelectionValidationResponse(BaseModel):
    can_finalize: bool
    reason: str
    selected_start_iso: Optional[str] = None
    confirmation_prompt: Optional[str] = None


class UrgentOutboundRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    caller_number: Optional[str] = None
    phone_number: Optional[str] = None
    call_id: Optional[str] = None
    destination_number: Optional[str] = None
    reason_for_call: Optional[str] = None
    clinic: Optional[str] = None
    language: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    # Dograh can send empty template strings for booleans (for example
    # "{{is_urgent}}" -> ""), so keep this permissive and normalize downstream.
    is_urgent: Optional[Any] = True
    priority_level: Optional[str] = "A"
    message: Optional[str] = None
    message_clean: Optional[str] = None
    notes: Optional[str] = None
    source: Optional[str] = "dograh"


class UrgentOutboundResponse(BaseModel):
    success: bool
    message: str
    provider: Optional[str] = None
    target_number: Optional[str] = None
    detail: Optional[str] = None


class PriorityAlertRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    caller_number: Optional[str] = None
    phone_number: Optional[str] = None
    priority_level: Optional[str] = None
    reason_for_call: Optional[str] = None
    message: Optional[str] = None
    message_clean: Optional[str] = None
    message_category: Optional[str] = None
    clinic: Optional[str] = None
    source: Optional[str] = "dograh"
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class PriorityAlertResponse(BaseModel):
    success: bool
    priority_level: str
    message: str
    webhook_sent: bool = False
    email_sent: bool = False


class CallWatchdogRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    caller_number: Optional[str] = None
    phone_number: Optional[str] = None
    call_id: Optional[str] = None
    event_type: Optional[str] = None
    reason_for_call: Optional[str] = None
    message: Optional[str] = None
    message_clean: Optional[str] = None
    last_node: Optional[str] = None
    seconds_elapsed: Optional[float] = None
    source: Optional[str] = "dograh"


class CallWatchdogResponse(BaseModel):
    success: bool
    message: str
    webhook_sent: bool = False
    email_sent: bool = False


def _state_flags_from_intent_result(result: dict[str, Any]) -> dict[str, Any]:
    intent = str(result.get("intent") or "")
    message_clean = str(result.get("message_clean") or "")
    clinic = str(result.get("clinic") or "")
    appointment_reason_done = intent == "appointment" and appointment_reason_present(result)
    duration_minutes = infer_appointment_duration_minutes(message_clean, default_minutes=20) if appointment_reason_done else None
    appointment_triage_still_needed = bool(result.get("appointment_triage_needed")) and not appointment_reason_done
    flags: dict[str, Any] = {
        "current_intent": intent,
        "current_message": message_clean,
        "current_message_category": str(result.get("message_category") or ""),
        "current_clinic": clinic,
        "current_day_preference": str(result.get("day_preference") or ""),
        "current_time_preference": str(result.get("time_preference") or ""),
        "current_from_date": str(result.get("from_date") or ""),
        "current_search_mode": str(result.get("search_mode") or ""),
        "appointment_triage_needed": appointment_triage_still_needed or None,
        "appointment_triage_done": appointment_reason_done or None,
        "notes": message_clean if appointment_reason_done else None,
        "duration_minutes": duration_minutes,
        "clinic_known": bool(clinic) or None,
        "priority_level": str(result.get("priority_level") or "") or None,
    }
    normalized_message = normalize_guard_text(message_clean)
    callback_phrases = (
        "να με παρει",
        "παρε με",
        "παρε με τηλεφωνο",
        "παρεμε",
        "παρεμε τηλεφωνο",
        "παρεμενε τηλεφωνο",
        "παρεμεινε τηλεφωνο",
        "παρεμειναν τηλεφωνο",
        "παρτε με",
        "πατηστε με",
        "να επικοινωνησει",
        "να μιλησω με τον γιατρο",
    )
    if intent == "message" and message_clean:
        flags["message_already_captured"] = True
    if any(normalize_guard_text(phrase) in normalized_message for phrase in callback_phrases):
        flags["caller_wants_callback"] = True
    if intent == "urgent":
        flags["urgent_handoff_sent"] = False
    return flags


_BACKEND_ROUTE_NODE_BY_INTENT = {
    "appointment": "APPOINTMENT FLOW",
    "change": "9.4 CHANGE APPOINTMENT FLOW",
    "cancellation": "9.5 CANCEL APPOINTMENT FLOW",
    "confirmation": "9.3 CHECK APPOINTMENT FLOW",
    "message": "9.2 SEND MESSAGE FLOW",
    "urgent": "9.6 URGENT FLOW",
    "other": "9.7 OTHER FLOW",
}


def _sticky_state_value(
    state: dict[str, Any],
    *keys: str,
    fallback: str = "",
) -> str:
    for key in keys:
        value = state.get(key)
        if value not in (None, "", "any"):
            return str(value)
    return fallback


def _has_explicit_message_request_token(routing_token: str) -> bool:
    """True only when caller clearly asks to send/leave a message or callback.

    This protects an active appointment flow from being derailed by context
    words such as "παραπεμπτικό" that can also be admin-message intents.
    """
    token = normalize_guard_text(routing_token or "")
    explicit_phrases = (
        "αφησω μηνυμα",
        "να αφησω μηνυμα",
        "στειλε μηνυμα",
        "στειλτε μηνυμα",
        "γραψε του",
        "γραψτε του",
        "ενημερωσε τον γιατρο",
        "ενημερωστε τον γιατρο",
        "να τον ενημερωσεις",
        "να τον ενημερωσετε",
        "να με παρει",
        "να με παρει τηλεφωνο",
        "να μου τηλεφωνησει",
        "να με καλεσει",
        "callback",
        "call back",
        "δεν θελω ραντεβου",
        "δεν θελω να κλεισω ραντεβου",
    )
    return any(phrase in token for phrase in explicit_phrases)


def _appointment_context_reason_from_token(routing_token: str) -> str:
    """Return a safe appointment reason/note for ambiguous reason-collection turns."""
    token = normalize_guard_text(routing_token or "")
    referral_terms = (
        "παραπεμπτικ",
        "παρα πεμπτικ",
        "παραπεπτικ",
        "παρα πεπτικ",
        "referral",
        "συσταση",
        "με εστειλε",
        "με εστειλαν",
        "χαρτι",
        "γνωματευση",
    )
    if any(term in token for term in referral_terms):
        return "Ιατρική επίσκεψη με παραπεμπτικό"
    return "Γενική ορθοπεδική επίσκεψη"


def _merge_sticky_route_state(
    result: dict[str, Any],
    *,
    state: dict[str, Any],
    body: BackendRouteRequest | PreRouteIntentRequest,
) -> dict[str, Any]:
    """Keep already locked clinic/date/time unless the new utterance changes them.

    This is intentionally conservative: the classifier decides intent, but
    previously confirmed details stay sticky so Dograh does not re-ask or drift
    from Limassol to Evrychou, or from an exact date back to a generic search.
    """
    merged = dict(result)
    previous_intent = str(state.get("current_intent") or "").strip().lower()
    routing_token = normalize_guard_text(_routing_message_from_body(body))
    explicit_second_booking = any(
        phrase in routing_token
        for phrase in (
            "κρατησω και το παλιο",
            "κρατησουμε και το παλιο",
            "δευτερο ραντεβου",
            "και αλλο ραντεβου",
            "επιπλεον ραντεβου",
            "χωρις να ακυρωσω",
            "χωρις να το ακυρωσω",
        )
    )
    if (
        previous_intent == "change"
        and merged.get("intent") == "appointment"
        and not explicit_second_booking
    ):
        merged["intent"] = "change"
        merged["confidence"] = max(float(merged.get("confidence") or 0.0), 0.98)
        merged["is_confident"] = True

    active_appointment_stage_name = str(state.get("current_stage") or "")
    active_appointment_stage = active_appointment_stage_name in {
        "offering_slots",
        "awaiting_slot_selection",
        "collecting_clinic",
        "collecting_reason",
    }

    # Intent lock for active appointment/change flows.
    #
    # Real-call failure this protects:
    #   Caller: "Ένα ραντεβού θα ήθελα."
    #   Agent:  "Για ποιο λόγο θέλετε να σας δει ο γιατρός;"
    #   Caller: "Έχουμε ένα παραπεμπτικό."
    #
    # "Παραπεμπτικό" can be an admin-message intent when there is no active
    # appointment. While collecting the appointment reason, it is the visit
    # reason/note and the booking flow must continue.
    active_appointment_context_guard = (
        previous_intent in {"appointment", "change"}
        and active_appointment_stage
        and active_appointment_stage_name in {
            "collecting_reason",
            "collecting_clinic",
            "offering_slots",
            "awaiting_slot_selection",
        }
        and str(merged.get("intent") or "").strip().lower() in {"message", "other"}
        and str(merged.get("priority_level") or "C").upper() != "A"
        and not _has_explicit_message_request_token(routing_token)
    )
    if active_appointment_context_guard:
        merged["intent"] = previous_intent
        merged["confidence"] = max(float(merged.get("confidence") or 0.0), 0.98)
        merged["is_confident"] = True
        merged["source"] = "sticky_appointment_context_guard"
        merged["message_category"] = ""
        merged["appointment_triage_needed"] = False
        merged["appointment_triage_done"] = True
        merged["message_clean"] = _appointment_context_reason_from_token(routing_token)

    noisy_appointment_followup = (
        previous_intent in {"appointment", "change"}
        and merged.get("intent") == "other"
        and active_appointment_stage
        and not bool(merged.get("is_confident"))
    )
    active_appointment_followup = (
        previous_intent in {"appointment", "change"}
        and merged.get("intent") == "other"
        and active_appointment_stage
        and (
            bool(merged.get("clinic"))
            or bool(merged.get("day_preference"))
            or bool(merged.get("time_preference"))
            or bool(merged.get("from_date"))
            or bool(classify_slot_selection_utterance(routing_token).get("can_finalize"))
        )
    )
    if active_appointment_followup or noisy_appointment_followup:
        merged["intent"] = previous_intent
        merged["confidence"] = max(float(merged.get("confidence") or 0.0), 0.96)
        merged["is_confident"] = True
        merged["source"] = "sticky_appointment_noise_guard" if noisy_appointment_followup else "sticky_appointment_followup"
        if noisy_appointment_followup:
            current_text = str(merged.get("message_clean") or "")
            if not appointment_reason_present(text=current_text):
                merged["message_clean"] = str(state.get("current_message") or current_text)

    body_clinic = getattr(body, "clinic", None)
    if not merged.get("clinic"):
        merged["clinic"] = _sticky_state_value(
            state,
            "current_clinic",
            "booked_clinic",
            fallback=str(body_clinic or ""),
        )

    body_day = getattr(body, "day_preference", None)
    if not merged.get("day_preference"):
        merged["day_preference"] = _sticky_state_value(
            state,
            "current_day_preference",
            fallback=str(body_day or ""),
        )

    body_time = getattr(body, "time_preference", None)
    if not merged.get("time_preference"):
        merged["time_preference"] = _sticky_state_value(
            state,
            "current_time_preference",
            fallback=str(body_time or ""),
        )

    body_from_date = getattr(body, "from_date", None)
    if not merged.get("from_date"):
        merged["from_date"] = _sticky_state_value(
            state,
            "current_from_date",
            fallback=str(body_from_date or ""),
        )
    if merged.get("from_date") and not merged.get("search_mode"):
        merged["search_mode"] = _sticky_state_value(
            state,
            "current_search_mode",
            fallback="exact",
        )

    if str(merged.get("intent") or "") == "appointment":
        reason_done = (
            bool(state.get("appointment_triage_done"))
            or bool(merged.get("appointment_triage_done"))
            or appointment_reason_present(merged)
        )
        merged["appointment_triage_needed"] = not reason_done

    return merged


def _booking_can_be_announced_from_route(body: BackendRouteRequest) -> bool:
    status = (body.booking_status or "").strip().lower()
    uid = (body.booking_uid or "").strip()
    event_id = (body.calendar_event_id or "").strip()
    return status == "confirmed" and bool(uid or event_id)


def _route_label_for_intent(intent: str) -> str:
    return {
        "appointment": "new_appointment_reason_first",
        "change": "change_existing_appointment",
        "cancellation": "cancel_existing_appointment",
        "confirmation": "check_existing_appointment",
        "message": "send_message_or_callback",
        "urgent": "urgent_handoff",
        "other": "other_admin_or_clarification",
    }.get(intent, "other_admin_or_clarification")


def _state_machine_directive(
    result: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
    current_stage: str,
    next_action: str,
) -> dict[str, Any]:
    """Return the deterministic next step the voice workflow should follow.

    The LLM should translate the caller into fields; this directive tells the
    workflow whether to ask one short question, call a specific tool now, or
    close. It is intentionally small and backend-owned to avoid prompt loops.
    """
    current_state = state or {}
    clinic = str(result.get("clinic") or current_state.get("current_clinic") or "").strip()
    intent = str(result.get("intent") or current_state.get("current_intent") or "other")
    action_tool = ""
    agent_instruction = "ask_one_question"
    ask_text = ""

    if next_action == "call_suggestions":
        if clinic == "limassol":
            action_tool = "cal_limassol_suggestions"
            agent_instruction = "call_tool_now"
        elif clinic == "evrychou":
            action_tool = "cal_evrychou_suggestions"
            agent_instruction = "call_tool_now"
        else:
            ask_text = "Λεμεσό ή Ευρύχου;"
    elif next_action == "ask_clinic":
        ask_text = "Λεμεσό ή Ευρύχου;"
    elif next_action == "ask_reason":
        ask_text = "Για ποιο λόγο θέλετε να σας δει ο γιατρός;"
    elif next_action == "ask_message_content":
        ask_text = "Τι μήνυμα θέλετε να αφήσω στον γιατρό;"
    elif next_action == "validate_slot_selection":
        action_tool = "validate_slot_selection"
        agent_instruction = "call_tool_now"
    elif next_action == "book_appointment":
        if clinic == "limassol":
            action_tool = "cal_limassol_book"
        elif clinic == "evrychou":
            action_tool = "cal_evrychou_book"
        else:
            action_tool = "book_appointment"
        agent_instruction = "call_tool_now"
    elif next_action == "check_appointment_by_phone":
        action_tool = "check_appointment_by_phone"
        agent_instruction = "call_tool_now"
    elif next_action == "confirm_cancellation":
        agent_instruction = "ask_one_question"
        ask_text = "Θέλετε να ακυρώσω αυτό το ραντεβού;"
    elif next_action == "cancel_confirmed":
        action_tool = (
            "cal_limassol_cancel"
            if clinic == "limassol"
            else "cal_evrychou_cancel"
            if clinic == "evrychou"
            else ""
        )
        agent_instruction = "call_tool_now" if action_tool else "ask_one_question"
    elif next_action == "send_message":
        action_tool = "send_message"
        agent_instruction = "call_tool_now"
    elif next_action == "send_message_then_urgent_outbound":
        action_tool = "send_message_then_urgent_outbound"
        agent_instruction = "call_tool_now"
    elif next_action == "answer_or_get_info":
        info_text = normalize_guard_text(
            f"{result.get('message_clean') or ''} {result.get('message') or ''}"
        )
        if any(
            phrase in info_text
            for phrase in (
                "γιατρος",
                "εργαζεται",
                "δουλευει",
                "ειναι μεσα",
                "ποτε εξεταζει",
                "τι μερες",
            )
        ):
            action_tool = "get_doctor_presence_info"
        else:
            action_tool = "get_clinic_info"
        agent_instruction = "call_tool_now"
    elif next_action == "end_or_new_request":
        agent_instruction = "close_or_wait_for_new_request"

    return {
        "flow_stage": current_stage,
        "action_tool": action_tool,
        "agent_instruction": agent_instruction,
        "ask_text": ask_text,
        "state_machine": {
            "enabled": True,
            "intent": intent,
            "stage": current_stage,
            "next_action": next_action,
            "clinic": clinic,
            "action_tool": action_tool,
            "instruction": agent_instruction,
            "ask_text": ask_text,
            "action_arguments": {
                "cal_uid": str(current_state.get("pending_appointment_uid") or ""),
                "start_iso": str(current_state.get("pending_appointment_start_iso") or ""),
                "clinic": str(current_state.get("pending_appointment_clinic") or clinic),
            },
        },
    }


async def _post_optional_webhook(url: str, payload: dict[str, Any]) -> bool:
    if not url:
        return False
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload)
        if 200 <= resp.status_code < 300:
            return True
        print("OPTIONAL WEBHOOK ERROR:", url, resp.status_code, resp.text[:300])
        return False


def _priority_label(level: str) -> str:
    if level == "A":
        return "Προτεραιότητα Α - Επείγον τώρα"
    if level == "B":
        return "Προτεραιότητα Β - Υψηλή"
    return "Προτεραιότητα Γ - Ρουτίνα"


_CALLBACK_REQUEST_PHRASES = (
    "να με παρει",
    "παρε με",
    "παρε με τηλεφωνο",
    "παρεμε",
    "παρεμενε τηλεφωνο",
    "παρτε με",
    "πατηστε με",
    "να επικοινωνησει",
    "να μιλησω",
    "να μιλησω με τον γιατρο",
)


def _caller_wants_callback_from_text(*parts: str) -> bool:
    normalized = normalize_guard_text(" ".join(part for part in parts if part))
    return any(
        normalize_guard_text(phrase) in normalized
        for phrase in _CALLBACK_REQUEST_PHRASES
    )


def _is_doctor_likely_in_clinic_now() -> bool:
    """
    Ελέγχει αν ΑΥΤΗ ΤΗ ΣΤΙΓΜΗ είναι μέσα στο ωράριο ΚΑΠΟΙΑΣ κλινικής
    (Λεμεσός ή Ευρύχου), με βάση το CLINIC_SCHEDULE.

    ΣΗΜΑΝΤΙΚΟ: πριν, η _message_spoken_response() έλεγε ΠΑΝΤΑ «Ο γιατρός
    εξετάζει αυτή τη στιγμή» για επείγοντα/callbacks, ΧΩΡΙΣ να ελέγξει αν
    είναι όντως ώρα/ημέρα λειτουργίας — π.χ. Σάββατο μεσημέρι, που το
    ιατρείο είναι κλειστό. Αυτό παραπλανά τον καλούντα.
    """
    now_local = datetime.now(TZ_NICOSIA)
    weekday = now_local.weekday()
    current_time = now_local.time()
    for clinic_hours in CLINIC_SCHEDULE.values():
        if weekday in clinic_hours:
            day_start, day_end = clinic_hours[weekday]
            if day_start <= current_time <= day_end:
                return True
    return False


def _message_spoken_response(
    message_category: str | None = "",
    *parts: str,
    priority_level: str | None = "",
) -> str:
    doctor_in_clinic = _is_doctor_likely_in_clinic_now()
    if str(priority_level or "").strip().upper() == "A":
        if doctor_in_clinic:
            return "Ο γιατρός εξετάζει αυτή τη στιγμή. Θα τον ενημερώσω άμεσα και θα σας καλέσει μόλις μπορέσει."
        return "Το ιατρείο είναι αυτή τη στιγμή κλειστό. Θα ενημερώσω άμεσα τον γιατρό για το επείγον περιστατικό σας."
    if (message_category or "").strip() == "doctor_callback" or _caller_wants_callback_from_text(*parts):
        if doctor_in_clinic:
            return "Ο γιατρός εξετάζει αυτή τη στιγμή. Θα τον ενημερώσω να σας καλέσει μόλις μπορέσει."
        return "Θα ενημερώσω τον γιατρό να σας καλέσει μόλις είναι διαθέσιμος."
    return "Θα ενημερώσω τον γιατρό σχετικά."


def _message_state(phone_number: Optional[str], call_id: Optional[str]) -> dict[str, Any]:
    state = get_conversation_state(phone_number=phone_number)
    if call_id:
        call_state = get_conversation_state(call_id=call_id)
        state = {**state, **call_state}
    return state


def _merge_message_context(existing: str | None, incoming: str | None) -> str:
    existing_text = (existing or "").strip()
    incoming_text = (incoming or "").strip()
    if not existing_text:
        return incoming_text
    if not incoming_text:
        return existing_text

    existing_norm = normalize_guard_text(existing_text)
    incoming_norm = normalize_guard_text(incoming_text)
    if not incoming_norm or incoming_norm in existing_norm:
        return existing_text
    if existing_norm and existing_norm in incoming_norm:
        return incoming_text
    return f"{existing_text}\nΠρόσθετες πληροφορίες: {incoming_text}"


def _message_ledger_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    ledger = state.get("message_ledger")
    if isinstance(ledger, list):
        return [item for item in ledger if isinstance(item, dict)]
    return []


def _append_message_ledger(
    state: dict[str, Any],
    *,
    action: str,
    text: str,
    category: str,
    priority_level: str,
) -> list[dict[str, Any]]:
    ledger = _message_ledger_from_state(state)
    text_norm = normalize_guard_text(text)
    if text_norm and any(
        item.get("action") == action
        and normalize_guard_text(str(item.get("text") or "")) == text_norm
        for item in ledger
    ):
        return ledger
    ledger.append(
        {
            "action": action,
            "text": text,
            "message_category": category,
            "priority_level": priority_level,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return ledger[-10:]


def _should_update_existing_message(
    previous_state: dict[str, Any],
    message_category: str,
    priority_level: str,
    incoming_text: str,
) -> bool:
    if not previous_state.get("message_sent"):
        return False

    existing_message = str(previous_state.get("current_message") or "").strip()
    incoming_norm = normalize_guard_text(incoming_text)
    if not existing_message or len(incoming_norm) < 8:
        return False

    merged = _merge_message_context(existing_message, incoming_text)
    if normalize_guard_text(merged) == normalize_guard_text(existing_message):
        return False
    if str(priority_level or "").upper() == "A":
        return True

    previous_category = str(previous_state.get("current_message_category") or "").strip()
    if message_category and previous_category:
        return message_category == previous_category
    if message_category and str(previous_state.get("current_intent") or "") == "message":
        return True

    inferred_existing_category = (
        rule_based_intent(existing_message).get("message_category") or ""
    )
    return bool(message_category and inferred_existing_category == message_category)


def _build_alert_lines(payload: dict[str, Any]) -> list[str]:
    return [
        f"Προτεραιότητα: {_priority_label(str(payload.get('priority_level') or 'C'))}",
        f"Τηλέφωνο: {payload.get('caller_number') or payload.get('phone_number') or '-'}",
        f"Λόγος: {payload.get('reason_for_call') or '-'}",
        f"Κατηγορία μηνύματος: {payload.get('message_category') or '-'}",
        f"Κλινική: {payload.get('clinic') or '-'}",
        f"Πηγή: {payload.get('source') or '-'}",
        "",
        f"Σύνοψη: {payload.get('message_clean') or '-'}",
        f"Μήνυμα: {payload.get('message') or '-'}",
    ]


_PHONE_PAYLOAD_KEYS = (
    "caller_number",
    "phone_number",
    "callerNumber",
    "phoneNumber",
    "from_number",
    "fromNumber",
    "from",
    "ani",
    "caller_id",
    "callerId",
    "customer_phone",
    "customerPhone",
)


def _extra_payload_value(body: Any, key: str) -> Any:
    extra = getattr(body, "model_extra", None) or {}
    if isinstance(extra, dict) and key in extra:
        return extra.get(key)
    if isinstance(body, dict):
        return body.get(key)
    return None


def _extract_phone_from_mapping(data: Any, *, max_depth: int = 4) -> Optional[str]:
    if max_depth < 0:
        return None
    if isinstance(data, dict):
        for key in _PHONE_PAYLOAD_KEYS:
            if key in data:
                phone = normalize_booking_phone(str(data.get(key) or ""))
                if phone:
                    return phone
        for value in data.values():
            phone = _extract_phone_from_mapping(value, max_depth=max_depth - 1)
            if phone:
                return phone
    elif isinstance(data, list):
        for value in data:
            phone = _extract_phone_from_mapping(value, max_depth=max_depth - 1)
            if phone:
                return phone
    return None


def _caller_number_from_body(body: Any) -> Optional[str]:
    """Prefer webhook caller-number aliases; never collect this from speech."""
    for key in _PHONE_PAYLOAD_KEYS:
        candidate = getattr(body, key, None) or _extra_payload_value(body, key)
        phone = normalize_booking_phone(candidate)
        if phone:
            return phone
    return None


def _routing_message_from_body(body: Any) -> str:
    """Prefer the cleaned workflow message, then raw message, then old text alias."""
    return (
        getattr(body, "message_clean", None)
        or getattr(body, "message", None)
        or getattr(body, "text", None)
        or ""
    )


def _is_false_booking_failure_after_success(payload: "MessagePayload", phone_number: Optional[str]) -> bool:
    text = f"{payload.message_clean or ''} {payload.message or ''}"
    normalized = normalize_guard_text(text)
    if not _looks_like_booking_failure_handoff(payload):
        return False
    state = get_conversation_state(phone_number=phone_number)
    if payload.call_id:
        call_state = get_conversation_state(call_id=payload.call_id)
        state = {**state, **call_state}
    if not state.get("booking_completed"):
        return False

    booked_start = str(
        state.get("booked_start_iso")
        or state.get("current_booking_start_iso")
        or ""
    ).strip()
    booked_clinic = normalize_guard_text(str(state.get("booked_clinic") or state.get("current_clinic") or ""))
    message_clinic = normalize_guard_text(payload.clinic or payload.message or "")

    if booked_start and booked_start in text:
        return True
    if booked_start and booked_start[:16] in text:
        return True
    if booked_clinic and booked_clinic in message_clinic:
        return True
    return bool(booked_start)


def _is_false_cancellation_handoff_after_success(
    payload: "MessagePayload",
    phone_number: Optional[str],
) -> bool:
    text = f"{payload.message_clean or ''} {payload.message or ''}"
    normalized = normalize_guard_text(text)
    has_cancel_context = any(
        phrase in normalized
        for phrase in (
            "ακυρωση",
            "ακυρωσει",
            "ακυρωσουμε",
            "ακυρωθηκε",
            "δεν θα ερθω",
            "cancel",
        )
    )
    if not has_cancel_context:
        return False

    state = get_conversation_state(phone_number=phone_number)
    if payload.call_id:
        call_state = get_conversation_state(call_id=payload.call_id)
        state = {**state, **call_state}

    return bool(
        state.get("cancel_completed")
        or state.get("cancellation_status") == "cancelled"
    )


def _looks_like_booking_failure_handoff(payload: "MessagePayload") -> bool:
    text = f"{payload.message_clean or ''} {payload.message or ''}"
    normalized = normalize_guard_text(text)
    has_booking_context = any(
        phrase in normalized
        for phrase in (
            "booking",
            "κρατηση",
            "κρατησης",
            "ραντεβου",
            "κλειστηκε",
            "κλεισω",
            "αλλαγη ραντεβου",
        )
    )
    has_failure_context = any(
        phrase in normalized
        for phrase in (
            "απετυχε",
            "δεν ολοκληρωθηκε",
            "δεν εγινε",
            "δεν καταφερα",
            "τεχνικο προβλημα",
            "προβλημα με το συστημα",
        )
    )
    return has_booking_context and has_failure_context


def _patient_identity_from_cache_by_phone(phone_number: Optional[str]) -> dict[str, str]:
    if not phone_number:
        return {}
    match = None
    try:
        match = lookup_patient_directory_by_phone(phone_number)
    except Exception as exc:
        print("PATIENT DIRECTORY PHONE LOOKUP ERROR:", str(exc))
        match = None
    if not match:
        try:
            match = lookup_patient_cache_by_phone(phone_number)
        except Exception as exc:
            print("PATIENT CACHE PHONE LOOKUP ERROR:", str(exc))
            return {}
    if not match:
        return {}
    identity: dict[str, str] = {
        "first_name": match.first_name or "",
        "last_name": match.last_name or "",
        "patient_lookup_source": match.source or "patient_directory",
    }
    return {key: value for key, value in identity.items() if value}


async def _patient_identity_from_cache_by_phone_fast(phone_number: Optional[str]) -> dict[str, str]:
    if not phone_number or not PRE_ROUTE_PATIENT_LOOKUP:
        return {}
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_patient_identity_from_cache_by_phone, phone_number),
            timeout=PRE_ROUTE_PATIENT_LOOKUP_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print(
            "PATIENT LOOKUP TIMEOUT: "
            f"phone={phone_number} timeout={PRE_ROUTE_PATIENT_LOOKUP_TIMEOUT_SECONDS:.2f}s"
        )
        return {}
    except Exception as exc:
        print("PATIENT LOOKUP FAST PATH ERROR:", str(exc))
        return {}


def _patient_identity_from_cache_by_name(
    first_name: Optional[str],
    last_name: Optional[str],
) -> Optional[Any]:
    match = None
    try:
        match = lookup_patient_directory_by_name(first_name, last_name)
    except Exception as exc:
        print("PATIENT DIRECTORY NAME LOOKUP ERROR:", str(exc))
        match = None
    if match:
        return match
    try:
        return lookup_patient_cache_by_name(first_name, last_name)
    except Exception as exc:
        print("PATIENT CACHE NAME LOOKUP ERROR:", str(exc))
        return None


def _merge_patient_identity_from_text(
    patient_identity: dict[str, str],
    text: Optional[str],
) -> dict[str, str]:
    first_name, last_name = extract_patient_name_from_text(text)
    if not (first_name and last_name):
        return patient_identity
    merged = dict(patient_identity)
    merged.setdefault("first_name", first_name)
    merged.setdefault("last_name", last_name)
    merged.setdefault("patient_lookup_source", "caller_text")
    return merged


def _stability_skip_reason_triage(result: dict[str, Any]) -> bool:
    if not RECEPTION_STABILITY_MODE:
        return False
    if str(result.get("intent") or "") != "appointment":
        return False
    if appointment_reason_present(result):
        return False
    message_token = normalize_guard_text(
        f"{result.get('message_clean') or ''} {result.get('message') or ''}"
    )
    caller_cannot_give_reason = any(
        phrase in message_token
        for phrase in (
            "δεν μπορω να απαντησω",
            "δεν μπορω να πω",
            "δεν ξερω να απαντησω",
            "δεν ξερω",
            "εν μπορω να απαντησω",
            "εν ξερω",
        )
    )
    if caller_cannot_give_reason:
        return True
    return bool(
        str(result.get("clinic") or "").strip()
        or str(result.get("day_preference") or "").strip()
        or str(result.get("time_preference") or "").strip()
    )




# ============================================================
# Parallel slot pre-fetch (warm Cal.com slot caches)
# ============================================================

async def _prefetch_clinic_slots(clinic: str) -> None:
    """Fire-and-forget: ζεσταίνει το slots cache μιας κλινικής
    καλώντας τη suggestions function με default request."""
    try:
        if clinic == "evrychou":
            await evrychou_suggestions(EvrychouSuggestionRequest())
        elif clinic == "limassol":
            await limassol_suggestions(LimassolSuggestionRequest())
        print(f"SLOT PREFETCH done: {clinic}")
    except Exception as exc:
        print(f"SLOT PREFETCH failed for {clinic}: {exc}")


def _maybe_prefetch_slots(intent: str, clinic: str) -> None:
    """Αν ο caller θέλει ραντεβού αλλά δεν έχει πει ακόμα κλινική,
    προ-φορτώνουμε slots ΚΑΙ των δύο κλινικών ώστε όταν απαντήσει
    «Λεμεσό»/«Ευρύχου» η suggestions απάντηση να είναι σχεδόν στιγμιαία.
    Αν η κλινική είναι ήδη γνωστή, προ-φορτώνουμε μόνο αυτή."""
    if intent != "appointment":
        return
    try:
        if clinic in ("limassol", "evrychou"):
            asyncio.create_task(_prefetch_clinic_slots(clinic))
        else:
            asyncio.create_task(_prefetch_clinic_slots("limassol"))
            asyncio.create_task(_prefetch_clinic_slots("evrychou"))
    except RuntimeError:
        # Όχι μέσα σε event loop (π.χ. tests) — απλώς παράλειψη
        pass

@app.post("/actions/pre_route_intent", response_model=PreRouteIntentResponse, dependencies=[Depends(require_action_token)])
async def pre_route_intent_action(body: PreRouteIntentRequest):
    """
    Deterministic pre-router for Dograh/LiveKit.

    Use this before the generative node when the utterance clearly matches a
    known reception pattern. If `is_confident` is false, the workflow should ask
    one short clarification or fall back to the trained classifier.
    """
    total_start = perf_counter()
    routing_message = _routing_message_from_body(body)
    rule_start = perf_counter()
    result = rule_based_intent(routing_message)
    rule_ms = (perf_counter() - rule_start) * 1000

    caller_number = _caller_number_from_body(body)
    patient_lookup_ms = 0.0
    patient_identity: dict[str, str] = {}
    if PRE_ROUTE_PATIENT_LOOKUP and caller_number:
        patient_lookup_start = perf_counter()
        patient_identity = await _patient_identity_from_cache_by_phone_fast(caller_number)
        patient_lookup_ms = (perf_counter() - patient_lookup_start) * 1000
    patient_identity = _merge_patient_identity_from_text(patient_identity, routing_message)
    previous_state = get_conversation_state(
        phone_number=caller_number,
        call_id=body.call_id,
    )
    if previous_state.get("conversation_done"):
        clear_conversation_state(
            phone_number=caller_number,
            call_id=body.call_id,
        )
        previous_state = {}
        print("PRE_ROUTE_INTENT: cleared terminal state from a previous call")
    confirmation_token = normalize_guard_text(routing_message)
    awaiting_cancel_confirmation = (
        previous_state.get("current_stage") == "awaiting_cancel_confirmation"
        or previous_state.get("next_action") == "confirm_cancellation"
    )
    if awaiting_cancel_confirmation and confirmation_token in {
        "ναι",
        "ναι παρακαλω",
        "βεβαιως",
        "σωστα",
        "ενταξει",
        "οκ",
        "ok",
        "να το ακυρωσετε",
        "ακυρωστε το",
    }:
        result = {
            "intent": "cancellation",
            "confidence": 1.0,
            "is_confident": True,
            "source": "cancellation_confirmation",
            "message_clean": routing_message,
            "clinic": str(previous_state.get("pending_appointment_clinic") or ""),
            "priority_level": "C",
            "next_action": "cancel_confirmed",
        }
        previous_state["cancellation_confirmed"] = True
        mark_cancellation_confirmed(
            str(previous_state.get("pending_appointment_uid") or ""),
            str(previous_state.get("pending_appointment_start_iso") or ""),
        )
    elif awaiting_cancel_confirmation and confirmation_token in {
        "οχι",
        "οχι ευχαριστω",
        "μην το ακυρωσετε",
        "δεν θελω",
    }:
        result = {
            "intent": "cancellation",
            "confidence": 1.0,
            "is_confident": True,
            "source": "cancellation_declined",
            "message_clean": routing_message,
            "clinic": str(previous_state.get("pending_appointment_clinic") or ""),
            "priority_level": "C",
            "next_action": "end_or_new_request",
        }
        previous_state["cancellation_confirmed"] = False
        previous_state["conversation_done"] = True
    result = _merge_sticky_route_state(result, state=previous_state, body=body)
    if _stability_skip_reason_triage(result):
        result["appointment_triage_needed"] = False
        result["appointment_triage_done"] = True
        result.setdefault("message_clean", "Γενική ορθοπεδική επίσκεψη")
    current_stage = infer_call_stage(result, previous_state)
    next_action = infer_next_action(result, previous_state)
    result["current_stage"] = current_stage
    result["next_action"] = next_action
    directive = _state_machine_directive(
        result,
        state=previous_state,
        current_stage=current_stage,
        next_action=next_action,
    )
    appointment_reason_done = str(result.get("intent") or "") == "appointment" and appointment_reason_present(result)
    if appointment_reason_done:
        result["appointment_triage_needed"] = False
        result["appointment_triage_done"] = True
    _maybe_prefetch_slots(str(result.get("intent") or ""), str(result.get("clinic") or ""))
    latency_notice = "Μισό λεπτό να το ελέγξω." if should_say_latency_notice(next_action) else ""
    next_state = {**previous_state, **patient_identity}
    name_collection_required = name_collection_required_for_route(result, next_state)
    state = update_conversation_state(
        phone_number=caller_number,
        call_id=body.call_id,
        **_state_flags_from_intent_result(result),
        current_stage=current_stage,
        next_action=next_action,
        name_collection_required=name_collection_required,
        cancellation_confirmed=previous_state.get("cancellation_confirmed"),
        conversation_done=previous_state.get("conversation_done"),
        **patient_identity,
        patient_name_captured=bool(
            patient_identity.get("first_name") and patient_identity.get("last_name")
        ) or None,
    )
    total_ms = (perf_counter() - total_start) * 1000
    print(
        "PRE_ROUTE_INTENT "
        f"intent={result.get('intent') or 'other'} "
        f"confident={bool(result.get('is_confident'))} "
        f"source={result.get('source') or 'rule'} "
        f"rule_ms={rule_ms:.1f} "
        f"patient_lookup_ms={patient_lookup_ms:.1f} "
        f"total_ms={total_ms:.1f} "
        f"patient_lookup_enabled={PRE_ROUTE_PATIENT_LOOKUP}"
    )
    is_confident = bool(result.get("is_confident"))
    intent = str(result.get("intent") or "other")
    rule_intent_before_fallback = intent  # τι είπε το rule ΠΡΙΝ οποιοδήποτε fallback
    ml_fallback_label = ""    # τι πρότεινε το ONNX, ΑΣΧΕΤΑ αν πέρασε το threshold
    ml_fallback_conf = None   # confidence αυτής της πρόβλεψης
    ml_fallback_used = False  # αν τελικά χρησιμοποιήθηκε για την απόφαση

    # ── ML fallback ΜΟΝΟ όταν το rule-based δεν είναι confident ───────────
    # ΣΗΜΑΝΤΙΚΟ ΓΙΑ ΤΑΧΥΤΗΤΑ: αυτό ΔΕΝ τρέχει στην πλειοψηφία των κλήσεων —
    # μόνο όταν το γρήγορο, rule-based intent δεν έπιασε με σιγουριά κάτι
    # γνωστό (is_confident=False). Στις άλλες περιπτώσεις μηδέν επιπλέον
    # latency. Strict timeout (CY_INTENT_FALLBACK_TIMEOUT_SECONDS, default
    # 0.4s) ώστε ένα αργό/προβληματικό μοντέλο να ΠΟΤΕ δεν κρατήσει τον
    # καλούντα να περιμένει — σε timeout/σφάλμα, ξαναπέφτουμε στο ίδιο
    # "ζήτησε διευκρίνιση" συμπεριφορά που υπήρχε πριν από αυτό το fallback.
    #
    # A/B DISAGREEMENT LOGGING: καταγράφουμε ΠΑΝΤΑ την πρόβλεψη του ONNX
    # εδώ (ml_fallback_label/conf), ΑΚΟΜΑ ΚΑΙ ΌΤΑΝ δεν περνά το threshold —
    # αυτό δείχνει στο /dashboard/rule_gaps περιπτώσεις όπου το rule δεν απάντησε
    # ΚΑΙ το BERT δεν ήταν αρκετά σίγουρο, που είναι διαφορετικό (και χρήσιμο)
    # σήμα από «το BERT απάντησε σωστά και χρησιμοποιήθηκε».
    if not is_confident and ENABLE_CY_INTENT and CY_INTENT_READY:
        try:
            fallback_timeout = float(os.getenv("CY_INTENT_FALLBACK_TIMEOUT_SECONDS", "0.4"))
            cy_label, cy_conf = await asyncio.wait_for(
                asyncio.to_thread(predict_cypriot_intent, routing_message),
                timeout=fallback_timeout,
            )
            ml_fallback_label = cy_label
            ml_fallback_conf = cy_conf
            print(f"PRE_ROUTE_INTENT: ML fallback used. label={cy_label} conf={cy_conf:.3f}")
            if cy_conf >= _cy_intent_threshold_for(cy_label):
                intent = cy_label
                result["intent"] = cy_label
                result["confidence"] = cy_conf
                result["source"] = "ml_fallback"
                is_confident = True
                ml_fallback_used = True
        except asyncio.TimeoutError:
            print(f"PRE_ROUTE_INTENT: ML fallback timed out after {fallback_timeout}s, using rule-based clarification.")
        except Exception as exc:
            print(f"PRE_ROUTE_INTENT: ML fallback error ({exc}), using rule-based clarification.")

    spoken_response = ""
    should_speak = False
    if is_confident and str(result.get("next_action") or "") in {"send_message", "send_message_then_urgent_outbound"}:
        # Do not let the router sound as if a doctor message was sent before
        # send_message actually succeeds. The workflow must speak this short
        # acknowledgement, then call send_message/urgent tools in the same turn.
        spoken_response = "Ένα λεπτό."
        should_speak = True
    elif is_confident and str(result.get("spoken_response") or "").strip():
        spoken_response = str(result.get("spoken_response") or "").strip()
        should_speak = True
    elif is_confident and str(result.get("next_action") or "") == "ask_reason":
        spoken_response = "Για ποιο λόγο θέλετε να σας δει ο γιατρός;"
        should_speak = True
    elif is_confident and str(result.get("next_action") or "") == "ask_clinic":
        spoken_response = "Λεμεσό ή Ευρύχου;"
        should_speak = True
    elif is_confident and str(result.get("next_action") or "") in {"send_message", "send_message_then_urgent_outbound"}:
        spoken_response = "Μάλιστα, ένα λεπτό να το καταγράψω."
        should_speak = True
    elif not is_confident:
        spoken_response = (
            "Δεν το κατάλαβα καλά. Μπορείτε να μου πείτε λίγο πιο συγκεκριμένα "
            "αν αφορά ραντεβού, αλλαγή, ακύρωση, μήνυμα στον γιατρό ή κάτι άλλο;"
        )
        should_speak = True
    elif intent in {"message", "urgent"}:
        spoken_response = _message_spoken_response(
            str(result.get("message_category") or ""),
            str(result.get("message_clean") or ""),
            priority_level=str(result.get("priority_level") or ""),
        )
        should_speak = True
    elif latency_notice and str(result.get("next_action") or "") != "call_suggestions":
        # Αργή ενέργεια (π.χ. call_suggestions / check_appointment_by_phone):
        # δώσε στον agent μια φράση-γέφυρα να πει ΑΜΕΣΩΣ. Για call_suggestions
        # όμως αφήνουμε το Appointment node να πει το acknowledgement και να
        # καλέσει αμέσως το tool. Αν μιλήσει ο Router πρώτος, δημιουργείται
        # extra turn/κενό και ο agent συχνά ρωτά «είστε στη γραμμή;».
        spoken_response = latency_notice
        should_speak = True

    # ── Log call to dashboard (captures phone ΠΡΙΝ φτάσει το webhook) ──
    _pri_phone = normalize_booking_phone(body.caller_number or body.phone_number or "")
    if _pri_phone:
        _log_call_event({
            "type": intent if intent in ("message", "urgent") else "call",
            "call_id": body.call_id or "",
            "phone": _pri_phone,
            "first_name": patient_identity.get("first_name") or "",
            "last_name": patient_identity.get("last_name") or "",
            "reason": intent,
            "message": str(result.get("message_clean") or body.message or ""),
            "clinic": str(result.get("clinic") or ""),
            "priority": str(result.get("priority_level") or "C"),
            "is_urgent": intent == "urgent" or str(result.get("priority_level")).upper() == "A",
            # ΝΕΟ: ποια μέθοδος αποφάσισε το intent — "rule" (deterministic,
            # κανόνας) ή "ml_fallback" (ONNX μοντέλο). Αυτό επιτρέπει στο
            # /dashboard/weekly_triage και σε χειροκίνητο review να εντοπίζει
            # ΠΟΙΕΣ κλήσεις θα άξιζε να γίνουν νέος μόνιμος κανόνας.
            "intent_source": str(result.get("source") or "rule"),
            # A/B disagreement logging — μόνο όταν το ML fallback ΌΝΤΩΣ
            # ενεργοποιήθηκε (δηλ. rule δεν ήταν confident). Δείχνει τι είπε
            # ο κανόνας ΠΡΙΝ το fallback (συνήθως "other"), τι πρότεινε το ONNX,
            # και αν το αποτέλεσμα τελικά χρησιμοποιήθηκε ή όχι (π.χ. το ONNX μίλησε
            # αλλά με confidence κάτω από το threshold).
            "rule_intent_before_fallback": rule_intent_before_fallback,
            "ml_fallback_label": ml_fallback_label,
            "ml_fallback_confidence": ml_fallback_conf,
            "ml_fallback_used": ml_fallback_used,
        })

    return PreRouteIntentResponse(
        intent=intent,
        confidence=float(result.get("confidence") or 0.0),
        threshold=CY_INTENT_CONFIDENCE_THRESHOLD,
        is_confident=is_confident,
        should_ask_clarification=not is_confident,
        source=str(result.get("source") or "rule"),
        message_clean=str(result.get("message_clean") or ""),
        message_category=str(result.get("message_category") or ""),
        clinic=str(result.get("clinic") or ""),
        day_preference=str(result.get("day_preference") or ""),
        time_preference=str(result.get("time_preference") or ""),
        from_date=str(result.get("from_date") or ""),
        search_mode=str(result.get("search_mode") or ""),
        appointment_triage_needed=bool(result.get("appointment_triage_needed")) and not appointment_reason_present(result),
        current_stage=current_stage,
        next_action=next_action,
        name_collection_required=name_collection_required,
        latency_notice=latency_notice,
        priority_level=str(result.get("priority_level") or ""),
        first_name=patient_identity.get("first_name"),
        last_name=patient_identity.get("last_name"),
        patient_lookup_source=patient_identity.get("patient_lookup_source"),
        flow_stage=directive["flow_stage"],
        action_tool=directive["action_tool"],
        agent_instruction=directive["agent_instruction"],
        ask_text=directive["ask_text"],
        state_machine=directive["state_machine"],
        next_node=_BACKEND_ROUTE_NODE_BY_INTENT.get(intent, "9.7 OTHER FLOW"),
        route_label=_route_label_for_intent(intent),
        route_locked=is_confident,
        should_speak=should_speak,
        spoken_response=spoken_response,
        state=state,
    )


@app.post("/actions/backend_route", response_model=BackendRouteResponse, dependencies=[Depends(require_action_token)])
async def backend_route_action(body: BackendRouteRequest):
    """
    Single backend source of truth for Dograh routing.

    Dograh should call this after each meaningful caller utterance and route
    from `next_node`; it should not reinterpret the intent locally.
    """
    total_start = perf_counter()
    routing_message = _routing_message_from_body(body)
    caller_number = _caller_number_from_body(body)
    previous_state = get_conversation_state(phone_number=caller_number, call_id=body.call_id)

    result = rule_based_intent(routing_message)
    result = _merge_sticky_route_state(result, state=previous_state, body=body)
    if _stability_skip_reason_triage(result):
        result["appointment_triage_needed"] = False
        result["appointment_triage_done"] = True
        result.setdefault("message_clean", "Γενική ορθοπεδική επίσκεψη")
    current_stage = infer_call_stage(result, previous_state)
    next_action = infer_next_action(result, previous_state)
    result["current_stage"] = current_stage
    result["next_action"] = next_action
    directive = _state_machine_directive(
        result,
        state=previous_state,
        current_stage=current_stage,
        next_action=next_action,
    )
    appointment_reason_done = str(result.get("intent") or "") == "appointment" and appointment_reason_present(result)
    if appointment_reason_done:
        result["appointment_triage_needed"] = False
        result["appointment_triage_done"] = True
    _maybe_prefetch_slots(str(result.get("intent") or ""), str(result.get("clinic") or ""))
    latency_notice = "Μισό λεπτό να το ελέγξω." if should_say_latency_notice(next_action) else ""

    patient_identity = _merge_patient_identity_from_text({}, routing_message)
    intent = str(result.get("intent") or "other")
    if intent not in _BACKEND_ROUTE_NODE_BY_INTENT:
        intent = "other"
        result["intent"] = intent

    booking_can_be_announced = _booking_can_be_announced_from_route(body)
    next_state = {**previous_state, **patient_identity}
    name_collection_required = name_collection_required_for_route(result, next_state)
    state_flags = {
        **_state_flags_from_intent_result(result),
        **patient_identity,
        "current_stage": current_stage,
        "next_action": next_action,
        "name_collection_required": name_collection_required,
        "patient_name_captured": bool(
            patient_identity.get("first_name") and patient_identity.get("last_name")
        )
        or None,
    }
    if body.selected_start_iso:
        state_flags["selected_start_iso"] = body.selected_start_iso
        state_flags["selected_slot_confirmed"] = True
    if body.selected_friendly_day:
        state_flags["selected_friendly_day"] = body.selected_friendly_day
    if body.selected_friendly_time:
        state_flags["selected_friendly_time"] = body.selected_friendly_time
    if body.notes:
        state_flags["notes"] = body.notes
    if body.duration_minutes:
        state_flags["duration_minutes"] = body.duration_minutes
    if body.booking_status:
        state_flags["booking_status"] = body.booking_status
    if body.booking_uid:
        state_flags["booking_uid"] = body.booking_uid
    if body.calendar_event_id:
        state_flags["calendar_event_id"] = body.calendar_event_id
    if booking_can_be_announced:
        state_flags["booking_completed"] = True

    state = update_conversation_state(
        phone_number=caller_number,
        call_id=body.call_id,
        **state_flags,
    )

    is_confident = bool(result.get("is_confident"))
    next_node = _BACKEND_ROUTE_NODE_BY_INTENT[intent] if is_confident else "9.7 OTHER FLOW"
    spoken_response = ""
    should_speak = False
    if is_confident and str(result.get("next_action") or "") in {"send_message", "send_message_then_urgent_outbound"}:
        # Do not let the router sound as if a doctor message was sent before
        # send_message actually succeeds. The workflow must speak this short
        # acknowledgement, then call send_message/urgent tools in the same turn.
        spoken_response = "Ένα λεπτό."
        should_speak = True
    elif is_confident and str(result.get("spoken_response") or "").strip():
        spoken_response = str(result.get("spoken_response") or "").strip()
        should_speak = True
    elif is_confident and str(result.get("next_action") or "") == "ask_reason":
        spoken_response = "Για ποιο λόγο θέλετε να σας δει ο γιατρός;"
        should_speak = True
    elif is_confident and str(result.get("next_action") or "") == "ask_clinic":
        spoken_response = "Λεμεσό ή Ευρύχου;"
        should_speak = True
    elif is_confident and str(result.get("next_action") or "") in {"send_message", "send_message_then_urgent_outbound"}:
        spoken_response = "Μάλιστα, ένα λεπτό να το καταγράψω."
        should_speak = True
    elif not is_confident:
        spoken_response = (
            "Δεν το κατάλαβα καλά. Μπορείτε να μου πείτε λίγο πιο συγκεκριμένα "
            "αν αφορά ραντεβού, αλλαγή, ακύρωση, μήνυμα στον γιατρό ή κάτι άλλο;"
        )
        should_speak = True
    elif intent in {"message", "urgent"}:
        spoken_response = _message_spoken_response(
            str(result.get("message_category") or ""),
            str(result.get("message_clean") or ""),
            priority_level=str(result.get("priority_level") or ""),
        )
        should_speak = True

    total_ms = (perf_counter() - total_start) * 1000
    print(
        "BACKEND_ROUTE "
        f"intent={intent} confident={is_confident} "
        f"next_node={next_node!r} "
        f"booking_can_be_announced={booking_can_be_announced} "
        f"total_ms={total_ms:.1f}"
    )

    return BackendRouteResponse(
        success=True,
        intent=intent,
        confidence=float(result.get("confidence") or 0.0),
        threshold=CY_INTENT_CONFIDENCE_THRESHOLD,
        is_confident=is_confident,
        should_ask_clarification=not is_confident,
        source=str(result.get("source") or "backend_rule"),
        message_clean=str(result.get("message_clean") or ""),
        message_category=str(result.get("message_category") or ""),
        priority_level=str(result.get("priority_level") or ""),
        clinic=str(result.get("clinic") or ""),
        day_preference=str(result.get("day_preference") or ""),
        time_preference=str(result.get("time_preference") or ""),
        from_date=str(result.get("from_date") or ""),
        search_mode=str(result.get("search_mode") or ""),
        appointment_triage_needed=bool(result.get("appointment_triage_needed")) and not appointment_reason_present(result),
        current_stage=current_stage,
        next_action=next_action,
        name_collection_required=name_collection_required,
        latency_notice=latency_notice,
        first_name=patient_identity.get("first_name"),
        last_name=patient_identity.get("last_name"),
        patient_lookup_source=patient_identity.get("patient_lookup_source"),
        flow_stage=directive["flow_stage"],
        action_tool=directive["action_tool"],
        agent_instruction=directive["agent_instruction"],
        ask_text=directive["ask_text"],
        state_machine=directive["state_machine"],
        next_node=next_node,
        route_label=_route_label_for_intent(intent),
        route_locked=is_confident,
        should_speak=should_speak,
        spoken_response=spoken_response,
        booking_can_be_announced=booking_can_be_announced,
        state=state,
    )


@app.post("/actions/conversation_state", response_model=ConversationStateResponse, dependencies=[Depends(require_action_token)])
async def conversation_state_action(body: ConversationStateRequest):
    flags = body.model_dump(
        exclude={"caller_number", "phone_number", "call_id"},
        exclude_none=True,
    )
    if flags:
        state = update_conversation_state(
            phone_number=_caller_number_from_body(body),
            call_id=body.call_id,
            **flags,
        )
    else:
        state = get_conversation_state(
            phone_number=_caller_number_from_body(body),
            call_id=body.call_id,
        )
    return ConversationStateResponse(success=True, state=state)


@app.post("/actions/validate_slot_selection", response_model=SlotSelectionValidationResponse, dependencies=[Depends(require_action_token)])
async def validate_slot_selection_action(body: SlotSelectionValidationRequest):
    """
    Conservative helper for Dograh booking turns.

    It prevents the common failure where the caller repeats only the date
    (e.g. "Τρίτη πέντε Μαΐου") and the agent finalizes the first slot without
    the caller accepting the time.
    """
    offered_dicts = [
        {
            "start_iso": opt.start_iso,
            "friendly_day": opt.friendly_day,
            "friendly_time": opt.friendly_time,
        }
        for opt in (body.offered_options or [])
    ]
    result = classify_slot_selection_utterance(
        body.caller_text,
        offered_options=offered_dicts or None,
    )
    reason = str(result.get("reason", ""))
    can_finalize = bool(result.get("can_finalize"))
    classification = str(result.get("classification", ""))

    if can_finalize:
        return SlotSelectionValidationResponse(
            can_finalize=True,
            reason=reason,
            selected_start_iso=result.get("selected_start_iso"),
        )

    if classification == "preference_change":
        return SlotSelectionValidationResponse(
            can_finalize=False,
            reason=reason,
            confirmation_prompt=None,
        )

    if reason == "date_only_needs_time_confirmation" and body.offered_options:
        caller_norm = normalize_guard_text(body.caller_text)
        for option in body.offered_options:
            friendly_day = option.friendly_day or ""
            friendly_time = option.friendly_time or ""
            if friendly_day and normalize_guard_text(friendly_day) in caller_norm:
                return SlotSelectionValidationResponse(
                    can_finalize=False,
                    reason=reason,
                    selected_start_iso=option.start_iso,
                    confirmation_prompt=f"{friendly_day} {friendly_time}. Σας βολεύει αυτή η ώρα;",
                )

        first = body.offered_options[0]
        if first.friendly_day and first.friendly_time:
            return SlotSelectionValidationResponse(
                can_finalize=False,
                reason=reason,
                selected_start_iso=first.start_iso,
                confirmation_prompt=(
                    f"{first.friendly_day} {first.friendly_time}. "
                    "Σας βολεύει αυτή η ώρα;"
                ),
            )

    return SlotSelectionValidationResponse(can_finalize=False, reason=reason)


@app.post("/actions/priority_alert", response_model=PriorityAlertResponse, dependencies=[Depends(require_action_token)])
async def priority_alert_action(body: PriorityAlertRequest):
    phone = _caller_number_from_body(body)
    text_for_priority = " ".join(
        part for part in [body.message_clean, body.message, body.reason_for_call] if part
    )
    priority = (body.priority_level or "").strip().upper() or classify_priority_level(
        text_for_priority,
        body.reason_for_call,
    )
    if priority not in {"A", "B", "C"}:
        priority = "C"

    payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    payload["caller_number"] = phone
    payload["priority_level"] = priority
    payload["timestamp"] = datetime.now(TZ_NICOSIA).isoformat()

    subject_prefix = {"A": "ΑΜΕΣΟ", "B": "ΥΨΗΛΟ", "C": "ΡΟΥΤΙΝΑ"}[priority]
    subject = f"[{subject_prefix}] Κλήση ασθενούς - {_priority_label(priority)}"
    text_content = "\n".join(_build_alert_lines(payload))

    email_sent = False
    try:
        await asyncio.to_thread(
            send_slack_notification,
            subject=subject,
            text_content=text_content,
            emoji=":rotating_light:",
            urgent=True,
        )
        await asyncio.to_thread(
            send_brevo_email,
            subject=subject,
            text_content=text_content,
            html_content="<br/>".join(_build_alert_lines(payload)),
            recipient_email=PRIORITY_ALERT_RECIPIENT_EMAIL,
        )
        email_sent = True
    except Exception as e:
        print("PRIORITY ALERT EMAIL ERROR:", e)

    webhook_sent = False
    try:
        webhook_sent = await _post_optional_webhook(PRIORITY_ALERT_WEBHOOK_URL, payload)
    except Exception as e:
        print("PRIORITY ALERT WEBHOOK ERROR:", e)

    if priority == "A":
        try:
            await _post_urgent_outbound(
                {
                    **payload,
                    "destination_number": os.getenv("URGENT_OUTBOUND_TARGET_NUMBER", ""),
                    "is_urgent": True,
                    "source": payload.get("source") or "dograh-priority-alert",
                }
            )
        except Exception as e:
            print("PRIORITY ALERT URGENT OUTBOUND ERROR:", e)

    return PriorityAlertResponse(
        success=email_sent or webhook_sent,
        priority_level=priority,
        message="Priority alert processed.",
        webhook_sent=webhook_sent,
        email_sent=email_sent,
    )


@app.post("/actions/call_watchdog", response_model=CallWatchdogResponse, dependencies=[Depends(require_action_token)])
async def call_watchdog_action(body: CallWatchdogRequest):
    phone = _caller_number_from_body(body)
    payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    payload["caller_number"] = phone
    payload["timestamp"] = datetime.now(TZ_NICOSIA).isoformat()

    event_type = (body.event_type or "unknown").strip()
    subject = f"[WATCHDOG] Κλήση χωρίς ολοκλήρωση - {event_type}"
    text_lines = [
        f"Συμβάν: {event_type}",
        f"Τηλέφωνο: {phone or '-'}",
        f"Call ID: {body.call_id or '-'}",
        f"Τελευταίο node: {body.last_node or '-'}",
        f"Δευτερόλεπτα: {body.seconds_elapsed if body.seconds_elapsed is not None else '-'}",
        f"Λόγος κλήσης: {body.reason_for_call or '-'}",
        "",
        f"Σύνοψη: {body.message_clean or '-'}",
        f"Μήνυμα: {body.message or '-'}",
    ]

    email_sent = False
    try:
        await asyncio.to_thread(
            send_slack_notification,
            subject=subject,
            text_content="\n".join(text_lines),
            emoji=":bell:",
        )
        await asyncio.to_thread(
            send_brevo_email,
            subject=subject,
            text_content="\n".join(text_lines),
            html_content="<br/>".join(text_lines),
            recipient_email=PRIORITY_ALERT_RECIPIENT_EMAIL,
        )
        email_sent = True
    except Exception as e:
        print("CALL WATCHDOG EMAIL ERROR:", e)

    webhook_sent = False
    try:
        webhook_sent = await _post_optional_webhook(CALL_WATCHDOG_WEBHOOK_URL, payload)
    except Exception as e:
        print("CALL WATCHDOG WEBHOOK ERROR:", e)

    return CallWatchdogResponse(
        success=email_sent or webhook_sent,
        message="Call watchdog event processed.",
        webhook_sent=webhook_sent,
        email_sent=email_sent,
    )


@app.post("/actions/urgent_outbound", response_model=UrgentOutboundResponse, dependencies=[Depends(require_action_token)])
async def urgent_outbound_action(request: Request):
    """
    Triggers an urgent callback/transfer webhook for Cloudonix/Zadarma or any
    configured telephony provider.

    The target webhook URL is configured through environment variables.
    """
    try:
        raw_body_bytes = await request.body()
        raw_body_str = raw_body_bytes.decode("utf-8") if raw_body_bytes else ""

        data: dict[str, Any]
        try:
            parsed = json.loads(raw_body_str) if raw_body_str else {}
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            data = parsed
        elif isinstance(parsed, str):
            try:
                nested = json.loads(parsed)
                data = nested if isinstance(nested, dict) else {}
            except Exception:
                data = {}
        else:
            data = {}

        if isinstance(data.get("payload"), dict):
            data = data["payload"]
        elif isinstance(data.get("data"), dict):
            data = data["data"]

        body = UrgentOutboundRequest(**data)
        payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
        result = await _post_urgent_outbound(payload)
        return UrgentOutboundResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        print("URGENT OUTBOUND ERROR:", e)
        raise HTTPException(status_code=500, detail=f"Urgent outbound failed: {str(e)}")

class DograhWebhookPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    caller_number: Optional[str] = None
    phone_number: Optional[str] = None
    call_id: Optional[str] = None
    workflow_id: Optional[int] = None
    workflow_run_id: Optional[int] = None
    run_id: Optional[int] = None
    reason_for_call: Optional[str] = None
    clinic: Optional[str] = None
    language: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_urgent: Optional[bool] = None
    priority_level: Optional[str] = None
    event_type: Optional[str] = None
    last_node: Optional[str] = None
    seconds_elapsed: Optional[float] = None
    duration_seconds: Optional[float] = None
    call_duration_seconds: Optional[float] = None
    user_turns: Optional[int] = None
    user_turn_count: Optional[int] = None
    day_preference: Optional[str] = None
    time_preference: Optional[str] = None
    message: Optional[str] = None
    appointment_date: Optional[str] = None  # "YYYY-MM-DD"
    appointment_time: Optional[str] = None  # "HH:MM"
    notes: Optional[str] = None
    duration_minutes: Optional[int] = None
    message_clean: Optional[str] = None
    proceed_to_finalize: Optional[Any] = None
    selected_slot_confirmed: Optional[Any] = None
    current_stage: Optional[str] = None
    next_action: Optional[str] = None
    appointment_triage_needed: Optional[bool] = None
    appointment_triage_done: Optional[bool] = None
    name_collection_required: Optional[bool] = None
    booking_status: Optional[str] = None
    booking_uid: Optional[str] = None
    calendar_event_id: Optional[str] = None
    conversation_done: Optional[bool] = None
    cancel_completed: Optional[bool] = None
    cancellation_status: Optional[str] = None
    selected_start_iso: Optional[str] = None
    selected_friendly_day: Optional[str] = None
    selected_friendly_time: Optional[str] = None
    booking_completed: Optional[Any] = None
    booking_status: Optional[str] = None
    booking_uid: Optional[str] = None
    calendar_event_id: Optional[str] = None
    call_disposition: Optional[str] = None
    mapped_call_disposition: Optional[str] = None
    call_tags: Optional[Any] = None
    nodes_visited: Optional[Any] = None


DOGRAH_SHORT_NO_TURN_ALERT_SECONDS = float(
    os.getenv("DOGRAH_SHORT_NO_TURN_ALERT_SECONDS", "5")
)


def _payload_extra(payload: BaseModel) -> dict[str, Any]:
    extra = getattr(payload, "model_extra", None)
    return extra if isinstance(extra, dict) else {}


def _nested_get(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _coerce_float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", ".").strip()
        return float(value)
    except Exception:
        return None


def _coerce_int_or_none(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).replace(",", ".").strip()))
    except Exception:
        return None


def _dograh_payload_value(payload: DograhWebhookPayload, data: dict[str, Any], *keys: str) -> Any:
    extra = _payload_extra(payload)
    extracted = data.get("extracted_variables") if isinstance(data, dict) else {}
    runtime = data.get("runtime_configuration") if isinstance(data, dict) else {}
    candidates = [payload, data, extra, extracted, runtime]
    for source in candidates:
        if isinstance(source, BaseModel):
            for key in keys:
                value = getattr(source, key, None)
                if value not in (None, ""):
                    return value
        elif isinstance(source, dict):
            value = _nested_get(source, *keys)
            if value not in (None, ""):
                return value
    return None


def _dograh_duration_seconds(payload: DograhWebhookPayload, data: dict[str, Any]) -> Optional[float]:
    value = _dograh_payload_value(
        payload,
        data,
        "duration_seconds",
        "call_duration_seconds",
        "call_duration",
        "duration",
        "seconds_elapsed",
    )
    seconds = _coerce_float_or_none(value)
    if seconds is not None:
        return seconds / 1000.0 if seconds > 10000 else seconds
    minutes = _coerce_float_or_none(_dograh_payload_value(payload, data, "duration_minutes"))
    if minutes is not None:
        return minutes * 60.0
    return None


def _dograh_user_turn_count(payload: DograhWebhookPayload, data: dict[str, Any]) -> Optional[int]:
    value = _dograh_payload_value(
        payload,
        data,
        "user_turns",
        "user_turn_count",
        "caller_turns",
        "user_message_count",
        "caller_message_count",
    )
    turns = _coerce_int_or_none(value)
    if turns is not None:
        return turns
    messages = data.get("messages") if isinstance(data, dict) else None
    if isinstance(messages, list):
        return sum(1 for msg in messages if isinstance(msg, dict) and str(msg.get("role") or "").lower() in {"user", "caller"})
    return None


def _is_short_no_user_turn_noise(payload: DograhWebhookPayload, data: dict[str, Any], *, has_content: bool) -> bool:
    duration = _dograh_duration_seconds(payload, data if isinstance(data, dict) else {})
    user_turns = _dograh_user_turn_count(payload, data if isinstance(data, dict) else {})
    if duration is None or user_turns is None:
        return False
    return duration < DOGRAH_SHORT_NO_TURN_ALERT_SECONDS and user_turns == 0 and not has_content


async def _recover_start_only_message_from_webhook(
    payload: DograhWebhookPayload,
    *,
    phone: str,
    message_text: str,
) -> bool:
    """Recover calls where Dograh stayed on start node but captured a full message."""
    message_text = (message_text or "").strip()
    if not phone or len(message_text) < 8:
        return False

    classified = rule_based_intent(message_text)
    intent = str(classified.get("intent") or "").strip().lower()
    message_category = str(classified.get("message_category") or "").strip()
    if intent not in {"message", "urgent", "appointment"} and not message_category:
        return False

    message_clean = str(classified.get("message_clean") or message_text).strip()
    if intent == "appointment" and not message_category:
        message_category = "appointment_start_only"
    priority_level = str(
        classified.get("priority_level")
        or classify_priority_level(message_clean, intent or "message")
        or "C"
    ).upper()
    dedupe_key = message_guard_key(phone, message_text, message_clean)
    if not claim_action_once(f"webhook_recovery:{dedupe_key}", ttl_seconds=15 * 60):
        print("WEBHOOK START MESSAGE RECOVERY DEDUPED:", phone)
        return False

    subject = (
        "Κλήση για ραντεβού που έκλεισε νωρίς"
        if intent == "appointment"
        else "Μήνυμα από κλήση που έκλεισε νωρίς"
    )
    full_name = " ".join(
        part for part in (payload.first_name or "", payload.last_name or "") if part
    ).strip() or "-"
    text_content = "\n".join(
        [
            "Η κλήση έμεινε στο αρχικό node ή δεν ολοκλήρωσε tool action, αλλά καταγράφηκε ουσιαστικό αίτημα.",
            f"Όνομα ασθενούς: {full_name}",
            f"Τηλέφωνο: {phone}",
            f"Κατηγορία: {intent or 'message'}",
            f"Υποκατηγορία: {message_category or '-'}",
            f"Προτεραιότητα: {_priority_label(priority_level)}",
            "",
            f"Σύνοψη: {message_clean}",
            "",
            "Αρχικό μήνυμα:",
            message_text,
        ]
    )
    html_content = (
        f"<h3>{subject}</h3>"
        "<p>Η κλήση έμεινε στο αρχικό node ή δεν ολοκλήρωσε tool action, αλλά καταγράφηκε ουσιαστικό αίτημα.</p>"
        f"<p><strong>Όνομα:</strong> {full_name}<br/>"
        f"<strong>Τηλέφωνο:</strong> {phone}<br/>"
        f"<strong>Κατηγορία:</strong> {intent or 'message'}<br/>"
        f"<strong>Υποκατηγορία:</strong> {message_category or '-'}<br/>"
        f"<strong>Προτεραιότητα:</strong> {_priority_label(priority_level)}<br/>"
        f"<strong>Σύνοψη:</strong> {message_clean}</p>"
        f"<p><strong>Αρχικό μήνυμα:</strong><br/>{message_text}</p>"
    )

    await asyncio.to_thread(
        send_slack_notification,
        subject=subject,
        text_content=text_content,
        emoji=":rescue_worker_helmet:",
    )
    await asyncio.to_thread(
        send_brevo_email,
        subject=subject,
        text_content=text_content,
        html_content=html_content,
    )
    state = get_conversation_state(phone_number=phone, call_id=payload.call_id) or {}
    update_conversation_state(
        phone_number=phone,
        call_id=payload.call_id,
        caller_number=phone,
        message_already_captured=True,
        message_sent=True,
        current_stage="message_sent",
        next_action="end_or_new_request",
        current_intent=intent or "message",
        current_message=message_clean,
        current_message_category=message_category,
        priority_level=priority_level,
        message_ledger=_append_message_ledger(
            state,
            action="webhook_recovered",
            text=message_clean,
            category=message_category,
            priority_level=priority_level,
        ),
        caller_wants_callback=_caller_wants_callback_from_text(message_clean) or None,
        webhook_start_message_recovered=True,
    )
    print("WEBHOOK START MESSAGE RECOVERED:", {"phone": phone, "intent": intent, "category": message_category, "priority": priority_level})
    return True


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = normalize_guard_text(value)
        return normalized in {"true", "yes", "y", "1", "ναι", "nai"}
    return False


def _booking_recovery_notes(payload: DograhWebhookPayload) -> str:
    for candidate in (payload.notes, payload.message_clean, payload.message):
        value = (candidate or "").strip()
        if value:
            return value
    return "Ιατρική επίσκεψη"


def _booking_recovery_email_body(
    payload: DograhWebhookPayload,
    *,
    phone: Optional[str],
    result_status: str,
    error_message: Optional[str] = None,
) -> str:
    lines = [
        "Το Dograh έφτασε σε confirmed appointment slot αλλά δεν πέρασε από finalize booking.",
        f"Recovery status: {result_status}",
        f"Call ID: {payload.call_id or '-'}",
        f"Όνομα: {(payload.first_name or '').strip()} {(payload.last_name or '').strip()}".strip(),
        f"Τηλέφωνο: {phone or payload.caller_number or payload.phone_number or '-'}",
        f"Κλινική: {payload.clinic or '-'}",
        f"Slot: {payload.selected_start_iso or '-'}",
        f"Φιλικά: {payload.selected_friendly_day or '-'} {payload.selected_friendly_time or '-'}",
    ]
    if payload.message:
        lines.append(f"Αρχικό μήνυμα: {payload.message}")
    if payload.message_clean:
        lines.append(f"Καθαρό μήνυμα: {payload.message_clean}")
    if error_message:
        lines.append(f"Σφάλμα: {error_message}")
    return "\n".join(lines)


def _booking_recovery_placeholder_identity(phone: Optional[str]) -> tuple[str, str]:
    """Use a safe temporary identity when the caller hangs up before giving a name."""
    digits = re.sub(r"\D+", "", phone or "")
    suffix = digits[-4:] if len(digits) >= 4 else "άγνωστο"
    return "Ασθενής", f"τηλ {suffix}"


async def _recover_pending_finalize_booking(
    payload: DograhWebhookPayload,
    *,
    phone: Optional[str],
) -> Optional[dict[str, Any]]:
    """Finalize a selected appointment slot if Dograh ended before 9.1B."""
    if normalize_guard_text(payload.reason_for_call or "") != "appointment":
        return None
    selected_start_iso = (payload.selected_start_iso or "").strip()
    if not selected_start_iso:
        return None
    selected_slot_confirmed = (
        _coerce_bool(payload.proceed_to_finalize)
        or _coerce_bool(payload.selected_slot_confirmed)
        or bool(selected_start_iso and (payload.selected_friendly_time or "").strip())
    )
    if not selected_slot_confirmed:
        return None
    if _coerce_bool(payload.booking_completed):
        return {"attempted": False, "reason": "payload_already_booked"}

    clinic = normalize_guard_text(payload.clinic or "")
    if clinic not in {"limassol", "evrychou"}:
        return {"attempted": False, "reason": "missing_or_unknown_clinic"}

    call_state = get_conversation_state(call_id=payload.call_id) if payload.call_id else {}
    phone_state = get_conversation_state(phone_number=phone) if phone else {}
    state = {**phone_state, **call_state}
    if state.get("booking_completed"):
        return {"attempted": False, "reason": "state_already_booked"}

    resolved_phone = normalize_booking_phone(phone)
    if not resolved_phone:
        for candidate in (
            payload.caller_number,
            payload.phone_number,
            state.get("caller_number"),
            state.get("phone_number"),
        ):
            resolved_phone = normalize_booking_phone(str(candidate or ""))
            if resolved_phone:
                break

    first_name = (payload.first_name or state.get("first_name") or "").strip()
    last_name = (payload.last_name or state.get("last_name") or "").strip()
    used_placeholder_identity = False
    if not first_name or not last_name:
        patient_identity = _patient_identity_from_cache_by_phone(resolved_phone or phone)
        first_name = first_name or patient_identity.get("first_name", "")
        last_name = last_name or patient_identity.get("last_name", "")
    if not first_name or not last_name:
        first_name, last_name = _booking_recovery_placeholder_identity(resolved_phone or phone)
        used_placeholder_identity = True

    notes = _booking_recovery_notes(payload)
    if used_placeholder_identity:
        notes = (
            f"{notes}. Recovery booking: ο καλών έκλεισε τη γραμμή πριν δοθεί όνομα, "
            "χρησιμοποιήθηκε προσωρινό όνομα."
        )
    try:
        duration_minutes = int(
            payload.duration_minutes
            or infer_appointment_duration_minutes(notes, default_minutes=20)
            or 20
        )
    except Exception:
        duration_minutes = infer_appointment_duration_minutes(notes, default_minutes=20)
    print(
        "DOGRAH BOOKING RECOVERY ATTEMPT:",
        {
            "call_id": payload.call_id,
            "clinic": clinic,
            "start_iso": selected_start_iso,
            "phone": resolved_phone or phone,
            "first_name": first_name,
            "last_name": last_name,
            "placeholder_identity": used_placeholder_identity,
        },
    )

    try:
        if clinic == "limassol":
            result = await limassol_book(
                LimassolBookRequest(
                    start_iso=selected_start_iso,
                    first_name=first_name,
                    last_name=last_name,
                    phone_number=resolved_phone or phone,
                    caller_number=resolved_phone or phone or payload.caller_number,
                    call_id=payload.call_id,
                    notes=notes,
                    duration_minutes=duration_minutes,
                )
            )
        else:
            result = await evrychou_book(
                EvrychouBookRequest(
                    start_iso=selected_start_iso,
                    first_name=first_name,
                    last_name=last_name,
                    phone_number=resolved_phone or phone,
                    caller_number=resolved_phone or phone or payload.caller_number,
                    call_id=payload.call_id,
                    notes=notes,
                    duration_minutes=duration_minutes,
                )
            )
    except Exception as e:
        error_message = str(e)
        print("DOGRAH BOOKING RECOVERY ERROR:", error_message)
        try:
            await asyncio.to_thread(
                send_brevo_email,
                subject="Προσοχή: επιλεγμένο ραντεβού δεν ολοκληρώθηκε",
                text_content=_booking_recovery_email_body(
                    payload,
                    phone=resolved_phone or phone,
                    result_status="exception",
                    error_message=error_message,
                ),
            )
        except Exception as email_error:
            print("DOGRAH BOOKING RECOVERY ALERT ERROR:", str(email_error))
        return {"attempted": True, "success": False, "status": "exception", "error": error_message}

    success = bool(getattr(result, "success", False))
    status = str(getattr(result, "status", "") or "")
    error_message = getattr(result, "error_message", None)
    if success:
        update_conversation_state(
            phone_number=resolved_phone or phone,
            call_id=payload.call_id,
            booking_slot_confirmed=True,
            selected_slot_confirmed=True,
            booking_completed=True,
            current_clinic=clinic,
            booked_start_iso=selected_start_iso,
            current_booking_start_iso=selected_start_iso,
            booked_clinic=clinic,
            first_name=first_name,
            last_name=last_name,
        )
        print("DOGRAH BOOKING RECOVERY SUCCESS:", {"status": status, "call_id": payload.call_id})
    else:
        print("DOGRAH BOOKING RECOVERY FAILED:", {"status": status, "error": error_message})
        try:
            await asyncio.to_thread(
                send_brevo_email,
                subject="Προσοχή: επιλεγμένο ραντεβού δεν ολοκληρώθηκε",
                text_content=_booking_recovery_email_body(
                    payload,
                    phone=resolved_phone or phone,
                    result_status=status or "failed",
                    error_message=error_message,
                ),
            )
        except Exception as email_error:
            print("DOGRAH BOOKING RECOVERY ALERT ERROR:", str(email_error))

    return {
        "attempted": True,
        "success": success,
        "status": status,
        "error": error_message,
    }


class CheckRequestedDayRequest(BaseModel):
    day_preference: Optional[str] = None
    from_date: Optional[date | datetime] = None
    message: Optional[str] = None
    message_clean: Optional[str] = None
    now_iso: Optional[str] = None


class CheckRequestedDayResponse(BaseModel):
    has_specific_day: bool
    target_date: Optional[str] = None
    friendly_day: Optional[str] = None
    is_non_working_day: bool = False
    is_weekend: bool = False
    is_public_holiday: bool = False
    reason: Optional[str] = None


class CheckDoctorStatusRequest(BaseModel):
    day_preference: Optional[str] = None
    from_date: Optional[date | datetime] = None
    message: Optional[str] = None
    message_clean: Optional[str] = None
    now_iso: Optional[str] = None


class CheckDoctorStatusResponse(BaseModel):
    has_specific_day: bool
    target_date: Optional[str] = None
    friendly_day: Optional[str] = None
    status: str
    is_away: bool = False
    matched_keyword: Optional[str] = None
    matched_event_title: Optional[str] = None
    reason: Optional[str] = None


class ClinicInfoRequest(BaseModel):
    clinic: str


class ClinicInfoResponse(BaseModel):
    clinic: str
    label: str
    address: str
    landmark: str
    spoken_address: str


class DoctorPresenceInfoRequest(BaseModel):
    day_preference: Optional[str] = None
    from_date: Optional[date | datetime] = None
    clinic: Optional[str] = None
    time_preference: Optional[str] = None
    now_iso: Optional[str] = None

    @field_validator("from_date", mode="before")
    @classmethod
    def _coerce_from_date(cls, v: Any) -> Any:
        """Αγνοεί κενά strings — ο agent μερικές φορές στέλνει from_date=''."""
        if v == "" or v is None:
            return None
        return v


class DoctorPresenceInfoResponse(BaseModel):
    has_specific_day: bool
    target_date: Optional[str] = None
    friendly_day: Optional[str] = None
    is_non_working_day: bool = False
    is_weekend: bool = False
    is_public_holiday: bool = False
    doctor_status: str
    is_away: bool = False
    doctor_location: Optional[str] = None
    doctor_location_label: Optional[str] = None
    clinic_requested: Optional[str] = None
    clinic_matches_request: Optional[bool] = None
    clinic_open: Optional[bool] = None
    clinic_supports_requested_time: Optional[bool] = None
    weekly_schedule_summary: Optional[str] = None
    spoken_summary: Optional[str] = None
    next_step_hint: Optional[str] = None
    matched_keyword: Optional[str] = None
    matched_event_title: Optional[str] = None
    reason: Optional[str] = None


class CheckNoSlotsRuleRequest(BaseModel):
    clinic: str
    day_preference: Optional[str] = None
    time_preference: Optional[str] = "any"
    from_date: Optional[date | datetime] = None
    message: Optional[str] = None
    message_clean: Optional[str] = None
    now_iso: Optional[str] = None


class NoSlotAlternative(BaseModel):
    start_iso: str
    friendly_day: Optional[str] = None
    friendly_time: str


class CheckNoSlotsRuleResponse(BaseModel):
    has_specific_day: bool
    target_date: Optional[str] = None
    friendly_day: Optional[str] = None
    requested_time_preference: str = "any"
    status: str
    next_action: str
    is_non_working_day: bool = False
    has_slots_in_requested_time: bool = False
    has_slots_on_same_day: bool = False
    has_future_requested_time_slots: bool = False
    requested_time_slot_count: int = 0
    same_day_slot_count: int = 0
    alternative_time_preferences: List[str] = Field(default_factory=list)
    alternative_same_day_slots: List[NoSlotAlternative] = Field(default_factory=list)
    future_requested_time_slots: List[NoSlotAlternative] = Field(default_factory=list)
    clinic_operates_on_day: bool = True
    clinic_supports_requested_time: bool = True
    clinic_schedule_reason: Optional[str] = None
    reason: Optional[str] = None

class ShadowSetmoreRequest(BaseModel):
    first_name: str
    last_name: str
    phone_number: str
    start_iso: str
    clinic: Optional[str] = None
    email: Optional[str] = None
    notes: Optional[str] = None
    cal_uid: Optional[str] = None
    duration_minutes: int = 20

class ShadowSetmoreResponse(BaseModel):
    success: bool
    setmore_appointment_id: str | None = None
    error: str | None = None


class ReplaceAppointmentAttendeeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    booking_uid: Optional[str] = None
    cal_uid: Optional[str] = None
    booking_id: Optional[int | str] = None
    attendee_id: Optional[int | str] = None
    setmore_appointment_key: Optional[str] = None
    start_iso: Optional[str] = None
    clinic: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    caller_number: Optional[str] = None
    email: Optional[str] = None
    notes: Optional[str] = None
    language: Optional[str] = "el"
    timezone: Optional[str] = "Asia/Nicosia"
    call_id: Optional[str] = None


class ReplaceAppointmentAttendeeResponse(BaseModel):
    success: bool
    booking_uid: Optional[str] = None
    booking_id: Optional[str] = None
    attendee_id: Optional[str] = None
    cal_updated: bool = False
    cal_added_attendee: bool = False
    cal_title_updated: bool = False
    setmore_updated: bool = False
    setmore_appointment_key: Optional[str] = None
    message: str = ""
    errors: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class CalSetmoreSyncResponse(BaseModel):
    success: bool
    scanned_cal: int
    scanned_setmore: int
    created: int
    recreated: int
    already_synced: int
    skipped: int
    soft_cancelled: int
    create_errors: List[str] = Field(default_factory=list)
    cancel_errors: List[str] = Field(default_factory=list)


URGENT_OUTBOUND_URL_ENV_VARS = (
    "URGENT_OUTBOUND_WEBHOOK_URL",
    "URGENT_OUTBOUND_URL",
    "CLOUDONIX_URGENT_OUTBOUND_URL",
    "ZADARMA_URGENT_OUTBOUND_URL",
)


def _resolve_urgent_outbound_urls() -> List[str]:
    urls: List[str] = []
    for env_name in URGENT_OUTBOUND_URL_ENV_VARS:
        raw = (os.getenv(env_name) or "").strip()
        if not raw:
            continue
        urls.extend([part.strip() for part in raw.split(",") if part.strip()])

    # De-duplicate while preserving order.
    deduped: List[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _zadarma_auth_header(method: str, params: dict[str, Any]) -> str:
    if not (ZADARMA_API_KEY and ZADARMA_API_SECRET):
        raise RuntimeError("Missing ZADARMA_API_KEY or ZADARMA_API_SECRET")
    clean_params = {
        key: str(value)
        for key, value in sorted(params.items())
        if value is not None and str(value) != ""
    }
    params_str = urlencode(clean_params)
    digest = hashlib.md5(params_str.encode("utf-8")).hexdigest()
    sign_raw = hmac.new(
        ZADARMA_API_SECRET.encode("utf-8"),
        f"{method}{params_str}{digest}".encode("utf-8"),
        hashlib.sha1,
    ).digest()
    signature = base64.b64encode(sign_raw).decode("ascii")
    return f"{ZADARMA_API_KEY}:{signature}"


async def _zadarma_post(method: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"https://api.zadarma.com{method}"
    headers = {
        "Authorization": _zadarma_auth_header(method, params),
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, data=params, headers=headers)
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    if resp.status_code >= 300 or body.get("status") == "error":
        raise RuntimeError(f"Zadarma {method} failed {resp.status_code}: {body}")
    return body


async def _zadarma_get(method: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"https://api.zadarma.com{method}"
    headers = {
        "Authorization": _zadarma_auth_header(method, params),
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, params=params, headers=headers)
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    if resp.status_code >= 300 or body.get("status") == "error":
        raise RuntimeError(f"Zadarma {method} failed {resp.status_code}: {body}")
    return body


async def _zadarma_callback_to_caller(payload: dict[str, Any], target_number: str) -> dict[str, Any]:
    caller_number = (payload.get("caller_number") or "").strip()
    if not caller_number:
        raise RuntimeError("Missing caller_number for Zadarma callback")
    callback_from = ZADARMA_CALLBACK_FROM or target_number
    if not callback_from:
        raise RuntimeError("Missing ZADARMA_CALLBACK_FROM or urgent outbound target number")
    params: dict[str, Any] = {
        "from": callback_from,
        "to": caller_number,
    }
    if ZADARMA_CALLBACK_SIP:
        params["sip"] = ZADARMA_CALLBACK_SIP
    if ZADARMA_CALLBACK_PREDICTED:
        params["predicted"] = 1
    # Zadarma documents the callback method as GET /v1/request/callback/.
    result = await _zadarma_get("/v1/request/callback/", params)
    return {
        "success": True,
        "provider": "zadarma_callback",
        "target_number": target_number,
        "message": "Zadarma callback requested.",
        "detail": json.dumps(result, ensure_ascii=False),
    }


async def _zadarma_sms_to_doctor(payload: dict[str, Any], target_number: str) -> dict[str, Any]:
    text = (
        f"ΕΠΕΙΓΟΝ Α: {payload.get('caller_number') or '-'} | "
        f"{payload.get('message_clean') or payload.get('message') or payload.get('notes') or 'Επείγουσα κλήση'}"
    )
    params: dict[str, Any] = {
        "number": target_number,
        "message": text[:500],
    }
    if ZADARMA_SMS_SENDER:
        params["sender"] = ZADARMA_SMS_SENDER
    result = await _zadarma_post("/v1/sms/send/", params)
    return {
        "success": True,
        "provider": "zadarma_sms",
        "target_number": target_number,
        "message": "Zadarma SMS sent.",
        "detail": json.dumps(result, ensure_ascii=False),
    }


async def _post_urgent_outbound(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Posts an urgent callback/transfer request to one or more configured webhook endpoints.

    The endpoint is intentionally provider-agnostic. Configure the destination webhook
    through one of:
      - URGENT_OUTBOUND_WEBHOOK_URL
      - URGENT_OUTBOUND_URL
      - CLOUDONIX_URGENT_OUTBOUND_URL
      - ZADARMA_URGENT_OUTBOUND_URL
    """
    raw_reason = str(payload.get("reason_for_call") or "").strip().lower()
    raw_priority = str(payload.get("priority_level") or "").strip().upper()
    raw_is_urgent = payload.get("is_urgent")
    is_urgent_flag = (
        raw_is_urgent is True
        or (isinstance(raw_is_urgent, str) and raw_is_urgent.strip().lower() in {"1", "true", "yes", "ναι"})
    )
    urgent_text = " ".join(
        str(payload.get(field) or "")
        for field in ("message_clean", "message", "notes")
    )
    inferred_priority = classify_priority_level(urgent_text, raw_reason)
    should_send_urgent_alert = (
        raw_priority == "A"
        or raw_reason == "urgent"
        or is_urgent_flag
        or inferred_priority == "A"
    )
    if not should_send_urgent_alert:
        return {
            "success": True,
            "provider": "urgent_outbound_skipped",
            "target_number": "",
            "message": "Urgent outbound skipped for non-urgent call.",
            "detail": json.dumps(
                {
                    "reason_for_call": raw_reason,
                    "priority_level": raw_priority or inferred_priority,
                    "is_urgent": raw_is_urgent,
                },
                ensure_ascii=False,
            ),
        }

    target_number = (payload.get("destination_number") or "").strip()
    if not target_number:
        target_number = (os.getenv("URGENT_OUTBOUND_TARGET_NUMBER") or "").strip()
    if not target_number:
        raise HTTPException(
            status_code=400,
            detail="Missing urgent outbound destination number",
        )

    caller_number = _extract_phone_from_mapping(payload) or ""
    call_id = (payload.get("call_id") or "").strip()
    caller_number_source = "payload" if caller_number else ""
    if not caller_number and call_id:
        state = get_conversation_state(call_id=call_id)
        caller_number = _extract_phone_from_mapping(state) or ""
        if caller_number:
            caller_number_source = "conversation_state"
    if not caller_number:
        cache_match = _patient_identity_from_cache_by_name(
            payload.get("first_name"),
            payload.get("last_name"),
        )
        if cache_match:
            caller_number = cache_match.phone_e164
            caller_number_source = cache_match.source

    outbound_payload = {
        "caller_number": caller_number,
        "phone_number": caller_number,
        "call_id": call_id,
        "destination_number": target_number,
        "reason_for_call": payload.get("reason_for_call") or "urgent",
        "clinic": payload.get("clinic") or "",
        "language": payload.get("language") or "el",
        "first_name": (payload.get("first_name") or "").strip(),
        "last_name": (payload.get("last_name") or "").strip(),
        "is_urgent": True,
        "priority_level": raw_priority or inferred_priority or "A",
        "message": (payload.get("message") or "").strip(),
        "message_clean": (payload.get("message_clean") or "").strip(),
        "notes": (payload.get("notes") or "").strip(),
        "source": payload.get("source") or "dograh",
        "caller_number_source": caller_number_source,
        "timestamp": datetime.now(TZ_NICOSIA).isoformat(),
    }

    if ZADARMA_ENABLE_CALLBACK and ZADARMA_API_KEY and ZADARMA_API_SECRET:
        if outbound_payload["caller_number"]:
            try:
                return await _zadarma_callback_to_caller(outbound_payload, target_number)
            except Exception as zad_exc:
                global ZADARMA_LAST_ERROR
                ZADARMA_LAST_ERROR = str(zad_exc)[:200]
                print(f"ZADARMA CALLBACK ERROR: {zad_exc}")
        else:
            print("URGENT OUTBOUND WARNING: Missing caller_number; Zadarma callback skipped")
        if not ZADARMA_ENABLE_SMS:
            return {
                "success": False,
                "provider": "zadarma_callback",
                "target_number": target_number,
                "message": "Missing caller_number; urgent callback was not placed.",
                "detail": "Dograh did not provide caller_number.",
            }

    if ZADARMA_ENABLE_SMS and ZADARMA_API_KEY and ZADARMA_API_SECRET:
        return await _zadarma_sms_to_doctor(outbound_payload, target_number)

    urls = _resolve_urgent_outbound_urls()
    if not urls:
        raise HTTPException(
            status_code=400,
            detail="No urgent outbound webhook URL configured and Zadarma urgent channel is disabled",
        )

    last_error: str | None = None
    async with httpx.AsyncClient(timeout=15.0) as client:
        for url in urls:
            try:
                resp = await client.post(url, json=outbound_payload)
                if 200 <= resp.status_code < 300:
                    return {
                        "success": True,
                        "provider": url,
                        "target_number": target_number,
                        "message": "Urgent outbound request sent.",
                        "detail": None,
                    }
                last_error = f"{url} returned {resp.status_code}: {resp.text[:300]}"
            except Exception as e:
                last_error = f"{url}: {e}"

    raise HTTPException(
        status_code=502,
        detail=last_error or "Urgent outbound request failed",
    )


async def _safe_post_urgent_outbound(payload: dict[str, Any]) -> None:
    try:
        await _post_urgent_outbound(payload)
    except Exception as e:
        print("AUTO URGENT OUTBOUND ERROR:", str(e))


# ============================================================
# Clinic schedule & availability helpers
# ============================================================

TZ_NICOSIA = ZoneInfo("Asia/Nicosia")


CLINIC_SCHEDULE = {
    "evrychou": {
        0: (time(9, 0), time(18, 0)),   # Monday
        2: (time(9, 0), time(18, 0)),   # Wednesday
        4: (time(9, 0), time(13, 0)),   # Friday
    },
    "limassol": {
        1: (time(8, 0), time(18, 0)),   # Tuesday
        3: (time(8, 0), time(18, 0)),   # Thursday
        4: (time(14, 30), time(18, 0)), # Friday
    },
}

def get_today_local() -> datetime:
    return datetime.now(TZ_NICOSIA)

def get_tomorrow_local_start() -> datetime:
    today = get_today_local().date()
    tomorrow = today + timedelta(days=1)
    # 09:00 Asia/Nicosia, μπορείς να αλλάξεις ώρα αν θες άλλο default
    return datetime(
        year=tomorrow.year,
        month=tomorrow.month,
        day=tomorrow.day,
        hour=9,
        minute=0,
        tzinfo=TZ_NICOSIA,
    )


def _normalize_time_preference(value: Optional[str]) -> str:
    normalized = normalize_time_preference(value, default="any")
    if normalized == "afternoon":
        return "after_16"
    return normalized


def _classify_slot_time_preference(start_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(TZ_NICOSIA)
    except Exception:
        return "any"

    hour = dt.hour
    if 8 <= hour < 12:
        return "morning"
    if 12 <= hour < 16:
        return "noon"
    if 16 <= hour < 20:
        return "after_16"
    return "any"


DOCTOR_AWAY_STATUS_KEYWORDS = {
    "away_conference": [
        "συνέδριο",
        "conference",
        "congress",
        "symposium",
    ],
    "away_surgery": [
        "χειρουργείο",
        "χειρουργειο",
        "surgery",
        "operation",
        "operating room",
        "χειρουργ",
    ],
    "away_leave": [
        "άδεια",
        "αδεια",
        "leave",
        "vacation",
        "holiday",
        "εκτός ιατρείου",
        "εκτος ιατρειου",
        "out of office",
        "ooo",
    ],
}


def _normalize_status_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", (value or "").strip().lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = re.sub(r"[^a-z0-9α-ω\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _find_doctor_away_keyword(text: str) -> tuple[str | None, str | None]:
    for status, keywords in DOCTOR_AWAY_STATUS_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text):
                return status, keyword
    return None, None


def _match_doctor_away_status(summary: str, description: str) -> tuple[str | None, str | None]:
    summary_text = _normalize_status_text(summary)
    description_text = _normalize_status_text(description)

    doctor_context_markers = (
        "γιατρ",
        "ιατρει",
        "doctor",
        "out of office",
        "ooo",
        "away",
        "status",
    )

    def _has_doctor_context(text: str) -> bool:
        return any(marker in text for marker in doctor_context_markers)

    # Only trust away keywords when the same event clearly looks like doctor-availability
    # metadata. This prevents generic notes such as "conference until Wednesday" from
    # incorrectly triggering an away status.
    status, keyword = _find_doctor_away_keyword(summary_text)
    if status and _has_doctor_context(summary_text):
        return status, keyword

    if description_text and _has_doctor_context(summary_text + " " + description_text):
        return _find_doctor_away_keyword(description_text)

    return None, None


SUPPORTED_DAY_PREFERENCES = {
    "today",
    "tomorrow",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
}

REFERENCE_DATE_MAX_DRIFT_DAYS = int(
    os.getenv("REFERENCE_DATE_MAX_DRIFT_DAYS", "7")
)


def _resolve_reference_date_safely(
    from_date: date | datetime | None,
    now_iso: str | None = None,
) -> date:
    """
    Dograh/Synthflow occasionally sends a stale current_date/from_date template.
    For relative expressions like today/tomorrow we prefer the provider's now_iso,
    and otherwise fall back to the server-local date when the incoming reference
    date is implausibly far away.
    """
    server_today = datetime.now(TZ_NICOSIA).date()

    if now_iso:
        try:
            return _parse_now_local(now_iso).date()
        except Exception:
            pass

    reference_date = resolve_reference_date(from_date)
    if abs((reference_date - server_today).days) > REFERENCE_DATE_MAX_DRIFT_DAYS:
        return server_today
    return reference_date


def _resolve_supported_target_date(
    day_preference: str | None,
    from_date: date | datetime | None,
    now_iso: str | None = None,
) -> tuple[str | None, date | None]:
    if not day_preference:
        return None, None

    normalized = normalize_day_preference(day_preference, default="any")
    if normalized not in SUPPORTED_DAY_PREFERENCES:
        return normalized, None

    reference_date = _resolve_reference_date_safely(from_date, now_iso)
    return normalized, resolve_target_date(normalized, reference_date)


def _resolve_exact_from_date_target(
    day_preference: str | None,
    from_date: date | datetime | None,
) -> date | None:
    """Treat from_date as the requested date when the caller gave an exact date.

    Dograh uses from_date both for "current call date" and for explicit dates.
    When day_preference is empty/any and from_date is not today, this is almost
    always a caller-specified date such as "26/5" or "είκοσι έξι πέμπτου".
    In that case do not apply stale-current-date drift protection.
    """
    if from_date is None:
        return None

    normalized = normalize_day_preference(day_preference, default="any")
    if normalized != "any":
        return None

    target_date = resolve_reference_date(from_date)
    if target_date == datetime.now(TZ_NICOSIA).date():
        return None
    return target_date


def _resolve_current_month_date_from_text(
    text: str | None,
    now_iso: str | None = None,
) -> date | None:
    """Resolve phrases like "14 του μηνός" to the current call month.

    Dograh can hallucinate a month for monthless Greek date phrases. When the
    caller says "του μηνός/του μήνα", the backend should override any supplied
    from_date month and use the current month from now_iso/server date.
    """
    normalized = normalize_guard_text(text or "")
    if not normalized:
        return None
    if not re.search(r"\bτου\s+μην(?:ος|α)\b", normalized):
        return None

    day: int | None = None
    digit_match = re.search(r"\b([0-3]?\d)\s+του\s+μην(?:ος|α)\b", normalized)
    if digit_match:
        day = int(digit_match.group(1))
    else:
        for day_words, candidate_day in sorted(
            DAY_WORDS_TO_INT.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if re.search(rf"\b{re.escape(day_words)}\s+του\s+μην(?:ος|α)\b", normalized):
                day = candidate_day
                break

    if day is None or not (1 <= day <= 31):
        return None

    reference_date = _resolve_reference_date_safely(None, now_iso)
    try:
        return date(reference_date.year, reference_date.month, day)
    except ValueError:
        return None


def _doctor_away_status_for_date(target_date: date) -> tuple[str, str | None, str | None]:
    day_start = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=0,
        minute=0,
        tzinfo=TZ_NICOSIA,
    )
    day_end = day_start + timedelta(days=1)

    events = get_gcal_events(
        calendar_id=CALENDAR_ID,
        time_min=day_start,
        time_max=day_end,
    )
    for event in events:
        summary = (event.get("summary") or "").strip()
        description = (event.get("description") or "").strip()
        status, keyword = _match_doctor_away_status(summary, description)
        if status:
            return status, keyword, summary or None

    return "available", None, None


def _parse_now_local(now_iso: str | None) -> datetime:
    if not now_iso:
        return datetime.now(TZ_NICOSIA)
    try:
        return datetime.fromisoformat(now_iso.replace("Z", "+00:00")).astimezone(TZ_NICOSIA)
    except Exception:
        return datetime.now(TZ_NICOSIA)


def _clinic_label(clinic: str | None) -> str | None:
    info = get_clinic_info(clinic)
    return info["label"] if info else None


def _doctor_presence_summary(
    target_date: date,
    friendly_day: str,
    doctor_status: str,
    doctor_location: str | None,
    clinic_requested: str | None,
    is_non_working_day: bool,
    is_public_holiday: bool,
    now_local: datetime,
) -> tuple[str, str]:
    if is_non_working_day:
        if is_public_holiday:
            return (
                f"Την {friendly_day} το ιατρείο είναι κλειστό λόγω αργίας.",
                "ask_direction_after_status",
            )
        return (
            f"Την {friendly_day} το ιατρείο είναι κλειστό.",
            "ask_direction_after_status",
        )

    away_messages = {
        "away_conference": f"Την {friendly_day} ο γιατρός είναι σε συνέδριο.",
        "away_surgery": f"Την {friendly_day} ο γιατρός είναι σε χειρουργείο.",
        "away_leave": f"Την {friendly_day} ο γιατρός είναι σε άδεια.",
    }
    if doctor_status in away_messages:
        return away_messages[doctor_status], "ask_direction_after_status"

    if target_date == now_local.date() and now_local.hour >= 18:
        return "Ο γιατρός εργάζεται μέχρι τις έξι.", "ask_direction_after_status"

    requested_label = _clinic_label(clinic_requested)
    location_label = _clinic_label(doctor_location)

    if clinic_requested and doctor_location and clinic_requested == doctor_location:
        return (
            f"Την {friendly_day} ο γιατρός είναι στην κλινική {requested_label}.",
            "ask_direction_after_status",
        )

    if clinic_requested and doctor_location and clinic_requested != doctor_location:
        return (
            f"Την {friendly_day} ο γιατρός είναι στην κλινική {location_label}, όχι στην {requested_label}.",
            "ask_direction_after_status",
        )

    if doctor_location:
        return (
            f"Την {friendly_day} ο γιατρός είναι στην κλινική {location_label}.",
            "ask_direction_after_status",
        )

    return (
        doctor_weekly_schedule_summary(clinic_requested),
        "ask_direction_after_status",
    )


@app.post("/actions/check_requested_day", response_model=CheckRequestedDayResponse, dependencies=[Depends(require_action_token)])
async def check_requested_day_action(body: CheckRequestedDayRequest):
    """
    Helper για Dograh πριν από τα suggestions.
    Λύνει τη ζητούμενη μέρα με βάση το current_date της κλήσης και δηλώνει
    αν πρόκειται για μη εργάσιμη ημέρα.
    """
    current_month_target = _resolve_current_month_date_from_text(
        body.message_clean or body.message,
        body.now_iso,
    )
    exact_target_date = _resolve_exact_from_date_target(body.day_preference, body.from_date)
    if current_month_target is not None:
        target_date = current_month_target
    elif exact_target_date is not None:
        target_date = exact_target_date
    elif not body.day_preference:
        return CheckRequestedDayResponse(
            has_specific_day=False,
            reason="missing_day_preference",
        )
    else:
        normalized, target_date = _resolve_supported_target_date(
            body.day_preference,
            body.from_date,
        )
        if normalized not in SUPPORTED_DAY_PREFERENCES or target_date is None:
            return CheckRequestedDayResponse(
                has_specific_day=False,
                reason="non_specific_day_preference",
            )

    probe_dt = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=9,
        minute=0,
        tzinfo=TZ_NICOSIA,
    )

    weekend = is_weekend_day(target_date)
    public_holiday = is_cyprus_public_holiday(target_date)
    non_working = is_non_working_day(target_date)

    reason = "working_day"
    if public_holiday:
        reason = "public_holiday"
    elif weekend:
        reason = "weekend"

    return CheckRequestedDayResponse(
        has_specific_day=True,
        target_date=target_date.isoformat(),
        friendly_day=make_friendly_day_gr(probe_dt),
        is_non_working_day=non_working,
        is_weekend=weekend,
        is_public_holiday=public_holiday,
        reason=reason,
    )


@app.post("/actions/check_doctor_status", response_model=CheckDoctorStatusResponse, dependencies=[Depends(require_action_token)])
async def check_doctor_status_action(body: CheckDoctorStatusRequest):
    """
    Ελέγχει αν ο γιατρός λείπει/είναι εκτός ιατρείου τη ζητούμενη ημέρα,
    με βάση ειδικά events του Google Calendar όπως συνέδριο ή χειρουργείο.
    """
    current_month_target = _resolve_current_month_date_from_text(
        body.message_clean or body.message,
        body.now_iso,
    )
    exact_target_date = _resolve_exact_from_date_target(body.day_preference, body.from_date)
    if current_month_target is not None:
        target_date = current_month_target
    elif exact_target_date is not None:
        target_date = exact_target_date
    elif not body.day_preference:
        return CheckDoctorStatusResponse(
            has_specific_day=False,
            status="unknown_day",
            reason="missing_day_preference",
        )
    else:
        normalized, target_date = _resolve_supported_target_date(
            body.day_preference,
            body.from_date,
        )
        if normalized not in SUPPORTED_DAY_PREFERENCES or target_date is None:
            return CheckDoctorStatusResponse(
                has_specific_day=False,
                status="unknown_day",
                reason="non_specific_day_preference",
            )
    probe_dt = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=9,
        minute=0,
        tzinfo=TZ_NICOSIA,
    )

    try:
        status, keyword, event_title = await asyncio.to_thread(
            _doctor_away_status_for_date, target_date
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch calendar events: {str(e)}")

    if status != "available":
        return CheckDoctorStatusResponse(
            has_specific_day=True,
            target_date=target_date.isoformat(),
            friendly_day=make_friendly_day_gr(probe_dt),
            status=status,
            is_away=True,
            matched_keyword=keyword,
            matched_event_title=event_title,
            reason="doctor_away_event_found",
        )

    return CheckDoctorStatusResponse(
        has_specific_day=True,
        target_date=target_date.isoformat(),
        friendly_day=make_friendly_day_gr(probe_dt),
        status="available",
        is_away=False,
        reason="no_doctor_away_event_found",
    )


@app.post("/actions/get_clinic_info", response_model=ClinicInfoResponse, dependencies=[Depends(require_action_token)])
async def get_clinic_info_action(body: ClinicInfoRequest):
    info = get_clinic_info(body.clinic)
    if not info:
        raise HTTPException(status_code=400, detail="unknown clinic")
    return ClinicInfoResponse(**info)


@app.post("/actions/get_doctor_presence_info", response_model=DoctorPresenceInfoResponse, dependencies=[Depends(require_action_token)])
async def get_doctor_presence_info_action(body: DoctorPresenceInfoRequest):
    clinic_requested = normalize_clinic_name(body.clinic, default="") or None
    normalized_time = _normalize_time_preference(body.time_preference)
    now_local = _parse_now_local(body.now_iso)

    if not body.day_preference:
        weekly_summary = doctor_weekly_schedule_summary(clinic_requested)
        return DoctorPresenceInfoResponse(
            has_specific_day=False,
            doctor_status="weekly_schedule",
            clinic_requested=clinic_requested,
            clinic_supports_requested_time=None,
            weekly_schedule_summary=weekly_summary,
            spoken_summary=weekly_summary,
            next_step_hint="ask_direction_after_status",
            reason="missing_day_preference",
        )

    normalized_day, target_date = _resolve_supported_target_date(
        body.day_preference,
        body.from_date,
        body.now_iso,
    )
    if normalized_day not in SUPPORTED_DAY_PREFERENCES or target_date is None:
        return DoctorPresenceInfoResponse(
            has_specific_day=False,
            doctor_status="unknown_day",
            reason="non_specific_day_preference",
        )
    probe_dt = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=9,
        minute=0,
        tzinfo=TZ_NICOSIA,
    )
    friendly_day = make_friendly_day_gr(probe_dt)
    weekend = is_weekend_day(target_date)
    public_holiday = is_cyprus_public_holiday(target_date)
    non_working = is_non_working_day(target_date)

    doctor_status = "available"
    matched_keyword = None
    matched_event_title = None
    if not non_working:
        try:
            doctor_status, matched_keyword, matched_event_title = await asyncio.to_thread(
                _doctor_away_status_for_date, target_date
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch calendar events: {str(e)}")

    doctor_location = None if non_working or doctor_status != "available" else doctor_location_for_date(
        target_date,
        normalized_time,
    )
    clinic_open = clinic_operates_on_date(clinic_requested, target_date) if clinic_requested else None
    clinic_supports_requested_time = (
        clinic_supports_time_preference(clinic_requested, target_date, normalized_time)
        if clinic_requested
        else None
    )
    clinic_matches_request = (
        (doctor_location == clinic_requested)
        if clinic_requested and doctor_location
        else None
    )
    spoken_summary, next_step_hint = _doctor_presence_summary(
        target_date=target_date,
        friendly_day=friendly_day,
        doctor_status=doctor_status,
        doctor_location=doctor_location,
        clinic_requested=clinic_requested,
        is_non_working_day=non_working,
        is_public_holiday=public_holiday,
        now_local=now_local,
    )

    reason = "working_day"
    if non_working:
        reason = "public_holiday" if public_holiday else "weekend"
    elif doctor_status != "available":
        reason = "doctor_away_event_found"

    return DoctorPresenceInfoResponse(
        has_specific_day=True,
        target_date=target_date.isoformat(),
        friendly_day=friendly_day,
        is_non_working_day=non_working,
        is_weekend=weekend,
        is_public_holiday=public_holiday,
        doctor_status=doctor_status,
        is_away=(doctor_status != "available"),
        doctor_location=doctor_location,
        doctor_location_label=_clinic_label(doctor_location),
        clinic_requested=clinic_requested,
        clinic_matches_request=clinic_matches_request,
        clinic_open=clinic_open,
        clinic_supports_requested_time=clinic_supports_requested_time,
        weekly_schedule_summary=doctor_weekly_schedule_summary(clinic_requested),
        spoken_summary=spoken_summary,
        next_step_hint=next_step_hint,
        matched_keyword=matched_keyword,
        matched_event_title=matched_event_title,
        reason=reason,
    )


@app.post("/actions/check_no_slots_rule", response_model=CheckNoSlotsRuleResponse, dependencies=[Depends(require_action_token)])
async def check_no_slots_rule_action(body: CheckNoSlotsRuleRequest):
    """
    Helper για το no-slots flow στο Dograh.

    Επιστρέφει μία καθαρή κατηγοριοποίηση:
    - non_working_day
    - requested_time_available
    - same_day_other_times_available
    - no_slots_on_day
    - non_specific_day
    """
    clinic = normalize_clinic_name(body.clinic, default="")
    if clinic not in {"limassol", "evrychou"}:
        raise HTTPException(status_code=400, detail="clinic must be limassol or evrychou")

    day_preference = normalize_day_preference(body.day_preference, default="any")
    supported_days = {
        "today",
        "tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
    }
    requested_time_preference = _normalize_time_preference(body.time_preference)
    current_month_target = _resolve_current_month_date_from_text(
        body.message_clean or body.message,
        body.now_iso,
    )
    exact_target_date = _resolve_exact_from_date_target(body.day_preference, body.from_date)

    if current_month_target is not None:
        target_date = current_month_target
    elif exact_target_date is not None:
        target_date = exact_target_date
    elif day_preference not in supported_days:
        return CheckNoSlotsRuleResponse(
            has_specific_day=False,
            requested_time_preference=requested_time_preference,
            status="non_specific_day",
            next_action="ask_day_or_proceed_normally",
            reason="non_specific_day_preference",
        )
    else:
        reference_date = _resolve_reference_date_safely(body.from_date)
        target_date = resolve_target_date(day_preference, reference_date)
        if target_date is None:
            return CheckNoSlotsRuleResponse(
                has_specific_day=False,
                requested_time_preference=requested_time_preference,
                status="non_specific_day",
                next_action="ask_day_or_proceed_normally",
                reason="could_not_resolve_target_date",
            )

    probe_dt = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=9,
        minute=0,
        tzinfo=TZ_NICOSIA,
    )
    friendly_day = make_friendly_day_gr(probe_dt)

    if is_non_working_day(target_date):
        return CheckNoSlotsRuleResponse(
            has_specific_day=True,
            target_date=target_date.isoformat(),
            friendly_day=friendly_day,
            requested_time_preference=requested_time_preference,
            status="non_working_day",
            next_action="offer_next_working_day_or_callback",
            is_non_working_day=True,
            reason="non_working_day",
        )

    clinic_day_open = clinic_operates_on_date(clinic, target_date)
    clinic_time_open = clinic_supports_time_preference(
        clinic,
        target_date,
        requested_time_preference,
    )
    schedule_reason = clinic_schedule_reason(
        clinic,
        target_date,
        requested_time_preference,
    )
    if not clinic_day_open or not clinic_time_open:
        return CheckNoSlotsRuleResponse(
            has_specific_day=True,
            target_date=target_date.isoformat(),
            friendly_day=friendly_day,
            requested_time_preference=requested_time_preference,
            status="no_slots_on_day",
            next_action="offer_other_day_or_callback",
            has_slots_in_requested_time=False,
            has_slots_on_same_day=False,
            requested_time_slot_count=0,
            same_day_slot_count=0,
            clinic_operates_on_day=clinic_day_open,
            clinic_supports_requested_time=clinic_time_open,
            clinic_schedule_reason=schedule_reason,
            reason=schedule_reason,
        )

    if clinic == "limassol":
        requested_payload = LimassolSuggestionRequest(
            day_preference=day_preference,
            time_preference=requested_time_preference,
            from_date=target_date if exact_target_date is not None else body.from_date or reference_date,
        )
        requested_result = await limassol_suggestions(requested_payload)
    else:
        requested_payload = EvrychouSuggestionRequest(
            day_preference=day_preference,
            time_preference=requested_time_preference,
            from_date=target_date if exact_target_date is not None else body.from_date or reference_date,
        )
        requested_result = await evrychou_suggestions(requested_payload)

    requested_options = requested_result.options
    if requested_options:
        return CheckNoSlotsRuleResponse(
            has_specific_day=True,
            target_date=target_date.isoformat(),
            friendly_day=friendly_day,
            requested_time_preference=requested_time_preference,
            status="requested_time_available",
            next_action="proceed_to_suggestions",
            has_slots_in_requested_time=True,
            has_slots_on_same_day=True,
            requested_time_slot_count=len(requested_options),
            same_day_slot_count=len(requested_options),
            clinic_operates_on_day=True,
            clinic_supports_requested_time=True,
            reason="requested_time_available",
        )

    if requested_time_preference == "any":
        future_same_time_slots: List[NoSlotAlternative] = []
        return CheckNoSlotsRuleResponse(
            has_specific_day=True,
            target_date=target_date.isoformat(),
            friendly_day=friendly_day,
            requested_time_preference=requested_time_preference,
            status="no_slots_on_day",
            next_action="offer_other_day_or_callback",
            has_slots_in_requested_time=False,
            has_slots_on_same_day=False,
            requested_time_slot_count=0,
            same_day_slot_count=0,
            future_requested_time_slots=future_same_time_slots,
            clinic_operates_on_day=True,
            clinic_supports_requested_time=True,
            reason="no_slots_on_day",
        )

    if clinic == "limassol":
        same_day_any_result = await limassol_suggestions(
            LimassolSuggestionRequest(
                day_preference=day_preference,
                time_preference="any",
                from_date=target_date if exact_target_date is not None else body.from_date or reference_date,
            )
        )
    else:
        same_day_any_result = await evrychou_suggestions(
            EvrychouSuggestionRequest(
                day_preference=day_preference,
                time_preference="any",
                from_date=target_date if exact_target_date is not None else body.from_date or reference_date,
            )
        )

    same_day_any_options = same_day_any_result.options
    future_same_time_options = []
    next_day_reference = target_date + timedelta(days=1)
    if clinic == "limassol":
        future_same_time_result = await limassol_suggestions(
            LimassolSuggestionRequest(
                day_preference="any",
                time_preference=requested_time_preference,
                from_date=next_day_reference,
            )
        )
    else:
        future_same_time_result = await evrychou_suggestions(
            EvrychouSuggestionRequest(
                day_preference="any",
                time_preference=requested_time_preference,
                from_date=next_day_reference,
            )
        )
    future_same_time_options = future_same_time_result.options

    if same_day_any_options:
        alt_prefs: List[str] = []
        alt_slots: List[NoSlotAlternative] = []
        future_slots: List[NoSlotAlternative] = []

        for option in same_day_any_options:
            slot_pref = _classify_slot_time_preference(option.start_iso)
            if slot_pref != requested_time_preference and slot_pref not in alt_prefs and slot_pref != "any":
                alt_prefs.append(slot_pref)

            if len(alt_slots) < 3:
                alt_slots.append(
                    NoSlotAlternative(
                        start_iso=option.start_iso,
                        friendly_time=option.friendly_time,
                    )
                )

        for option in future_same_time_options[:3]:
            future_slots.append(
                NoSlotAlternative(
                    start_iso=option.start_iso,
                    friendly_day=option.friendly_day,
                    friendly_time=option.friendly_time,
                )
            )

        return CheckNoSlotsRuleResponse(
            has_specific_day=True,
            target_date=target_date.isoformat(),
            friendly_day=friendly_day,
            requested_time_preference=requested_time_preference,
            status="same_day_other_times_available",
            next_action="offer_same_day_other_times_or_future_same_time",
            has_slots_in_requested_time=False,
            has_slots_on_same_day=True,
            has_future_requested_time_slots=bool(future_same_time_options),
            requested_time_slot_count=0,
            same_day_slot_count=len(same_day_any_options),
            alternative_time_preferences=alt_prefs,
            alternative_same_day_slots=alt_slots,
            future_requested_time_slots=future_slots,
            clinic_operates_on_day=True,
            clinic_supports_requested_time=True,
            reason="same_day_other_times_available",
        )

    future_slots: List[NoSlotAlternative] = []
    for option in future_same_time_options[:3]:
        future_slots.append(
            NoSlotAlternative(
                start_iso=option.start_iso,
                friendly_day=option.friendly_day,
                friendly_time=option.friendly_time,
            )
        )

    return CheckNoSlotsRuleResponse(
        has_specific_day=True,
        target_date=target_date.isoformat(),
        friendly_day=friendly_day,
        requested_time_preference=requested_time_preference,
        status="no_slots_on_day",
        next_action="offer_other_day_or_callback",
        has_slots_in_requested_time=False,
        has_slots_on_same_day=False,
        has_future_requested_time_slots=bool(future_same_time_options),
        requested_time_slot_count=0,
        same_day_slot_count=0,
        future_requested_time_slots=future_slots,
        clinic_operates_on_day=True,
        clinic_supports_requested_time=True,
        reason="no_slots_on_day",
    )

def generate_clinic_slots(
    clinic: str,
    from_date: datetime,
    duration_minutes: int,
    max_slots: int,
) -> List[TimeSlot]:
    """
    Γυρίζει slots σύμφωνα με το πρόγραμμα του ιατρείου (χωρίς conflicts).
    ΟΛΑ τα slots είναι timezone-aware (Asia/Nicosia).

    Αντί για max_slots*25, κοιτάμε παράθυρο ~30 ημερών,
    ώστε να βρούμε διαθέσιμα ακόμη κι αν οι επόμενες μέρες είναι γεμάτες.
    """
    clinic_key = clinic.lower()
    if clinic_key not in CLINIC_SCHEDULE:
        raise ValueError(f"Unknown clinic: {clinic}")

    if from_date.tzinfo is None:
        from_date = from_date.replace(tzinfo=TZ_NICOSIA)
    else:
        from_date = from_date.astimezone(TZ_NICOSIA)

    slots: List[TimeSlot] = []
    current_date = from_date

    # Μέχρι και 30 μέρες μπροστά
    days_ahead_limit = 30
    days_checked = 0

    while days_checked < days_ahead_limit:
        weekday = current_date.weekday()  # Monday=0

        if weekday in CLINIC_SCHEDULE[clinic_key]:
            day_start_time, day_end_time = CLINIC_SCHEDULE[clinic_key][weekday]

            day_start = datetime(
                year=current_date.year,
                month=current_date.month,
                day=current_date.day,
                hour=day_start_time.hour,
                minute=day_start_time.minute,
                tzinfo=TZ_NICOSIA,
            )
            day_end = datetime(
                year=current_date.year,
                month=current_date.month,
                day=current_date.day,
                hour=day_end_time.hour,
                minute=day_end_time.minute,
                tzinfo=TZ_NICOSIA,
            )

            if current_date > day_start:
                current_slot_start = current_date
            else:
                current_slot_start = day_start

            while current_slot_start + timedelta(minutes=duration_minutes) <= day_end:
                current_slot_end = current_slot_start + timedelta(minutes=duration_minutes)
                slots.append(TimeSlot(start=current_slot_start, end=current_slot_end))
                current_slot_start = current_slot_end

        # επόμενη μέρα
        current_date = (
            current_date + timedelta(days=1)
        ).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=TZ_NICOSIA)
        days_checked += 1

    return slots


def slot_overlaps_event(slot_start: datetime, slot_end: datetime, event: dict) -> bool:
    """
    Ελέγχει αν ένα slot τέμνεται χρονικά με ένα calendar event.
    Κάνει normalize ώστε ΟΛΑ να είναι timezone-aware (Asia/Nicosia).
    """
    start_info = event.get("start", {})
    end_info = event.get("end", {})

    if "dateTime" not in start_info or "dateTime" not in end_info:
        # All-day events ή περίεργα formats τα θεωρούμε conflict για ασφάλεια
        return True

    event_start = datetime.fromisoformat(start_info["dateTime"])
    event_end = datetime.fromisoformat(end_info["dateTime"])

    if slot_start.tzinfo is None:
        slot_start = slot_start.replace(tzinfo=TZ_NICOSIA)
    else:
        slot_start = slot_start.astimezone(TZ_NICOSIA)

    if slot_end.tzinfo is None:
        slot_end = slot_end.replace(tzinfo=TZ_NICOSIA)
    else:
        slot_end = slot_end.astimezone(TZ_NICOSIA)

    if event_start.tzinfo is None:
        event_start = event_start.replace(tzinfo=TZ_NICOSIA)
    else:
        event_start = event_start.astimezone(TZ_NICOSIA)

    if event_end.tzinfo is None:
        event_end = event_end.replace(tzinfo=TZ_NICOSIA)
    else:
        event_end = event_end.astimezone(TZ_NICOSIA)

    return slot_start < event_end and slot_end > event_start


def filter_slots_with_calendar(
    slots: List[TimeSlot],
    events: List[dict],
) -> List[TimeSlot]:
    """
    Φιλτράρει slots που πέφτουν πάνω σε ήδη κλεισμένα events.
    """
    available: List[TimeSlot] = []

    for slot in slots:
        conflict = False
        for ev in events:
            if slot_overlaps_event(slot.start, slot.end, ev):
                conflict = True
                break
        if not conflict:
            available.append(slot)

    return available

def find_matching_event(
    events: List[dict],
    start_dt: datetime,
    end_dt: datetime,
    clinic: str,
    first_name: str,
    last_name: str,
    phone_number: str,
) -> Optional[str]:
    """
    Βρίσκει το σωστό event για ακύρωση, με λογική:
    1. Χρονικό match + clinic tag στο summary.
    2. PHONE-FIRST: αν το τηλέφωνο ταιριάζει στο description, δεχόμαστε το event
       ακόμα κι αν το όνομα έχει μικρολάθη (Ιγνατίου / Ηγνατίου κ.λπ.).
    3. Αν δεν βρούμε με τηλέφωνο, δοκιμάζουμε πιο χαλαρά με επώνυμο.
    4. Αν υπάρχει ακριβώς ένα event στο παράθυρο, το παίρνουμε ως fallback.
    """
    full_name = f"{first_name or ''} {last_name or ''}".strip()
    clinic_tag = f"({clinic})".lower()
    phone_number = (phone_number or "").strip()

    # 1. PHONE-FIRST matching
    for ev in events:
        start_info = ev.get("start", {})
        end_info = ev.get("end", {})
        summary = (ev.get("summary") or "").lower()
        description = (ev.get("description") or "").lower()

        if "dateTime" not in start_info or "dateTime" not in end_info:
            continue

        try:
            ev_start = datetime.fromisoformat(start_info["dateTime"])
            ev_end = datetime.fromisoformat(end_info["dateTime"])
        except Exception:
            continue

        ev_start = ev_start.astimezone(TZ_NICOSIA)
        ev_end = ev_end.astimezone(TZ_NICOSIA)

        # Χρονικό match με tolerance 60"
        if abs((ev_start - start_dt).total_seconds()) > 60:
            continue
        if abs((ev_end - end_dt).total_seconds()) > 60:
            continue

        if "ραντεβού ortho papachristou" not in summary:
            continue
        if clinic_tag not in summary:
            continue

        # PHONE-FIRST: αν ταιριάζει το τηλέφωνο, αυτό είναι αρκετό
        if phone_number and phone_number.lower() in description:
            return ev.get("id")

    # 2. Αν δεν βρήκαμε με τηλέφωνο, δοκιμάζουμε πιο χαλαρό όνομα (επώνυμο)
    last_name_norm = (last_name or "").strip().lower()
    if last_name_norm:
        for ev in events:
            start_info = ev.get("start", {})
            end_info = ev.get("end", {})
            summary = (ev.get("summary") or "").lower()
            description = (ev.get("description") or "").lower()

            if "dateTime" not in start_info or "dateTime" not in end_info:
                continue

            try:
                ev_start = datetime.fromisoformat(start_info["dateTime"])
                ev_end = datetime.fromisoformat(end_info["dateTime"])
            except Exception:
                continue

            ev_start = ev_start.astimezone(TZ_NICOSIA)
            ev_end = ev_end.astimezone(TZ_NICOSIA)

            if abs((ev_start - start_dt).total_seconds()) > 60:
                continue
            if abs((ev_end - end_dt).total_seconds()) > 60:
                continue

            if "ραντεβού ortho papachristou" not in summary:
                continue
            if clinic_tag not in summary:
                continue

            # πιο χαλαρά: μόνο αν το επώνυμο εμφανίζεται κάπως στο description
            if last_name_norm in description:
                return ev.get("id")

    # 3. Fallback: αν υπάρχει ακριβώς 1 event στο παράθυρο, χρησιμοποίησέ το
    if len(events) == 1:
        only_event_id = events[0].get("id")
        print("FALLBACK: using only event in window for cancel:", only_event_id)
        return only_event_id

    return None


# ============================================================
# Synthflow Actions – generic get_availability
# ============================================================

@app.post("/actions/get_availability", response_model=GetAvailabilityResponse, dependencies=[Depends(require_action_token)])
async def get_availability_action(body: GetAvailabilityRequest):
    """
    Action για Synthflow: Natasa → get_availability

    Δέχεται JSON body:

    {
      "clinic": "limassol" ή "evrychou",
      "duration_minutes": 20,
      "max_slots": 25,
      "from_date": "2026-02-19T00:00:00+02:00"  (προαιρετικό)
    }

    Λογική:
    1. Καθορίζει από πότε ξεκινάμε (from_date ή τώρα).
    2. Παράγει slots σύμφωνα με το πρόγραμμα της κλινικής (generate_clinic_slots).
    3. Φέρνει events από Google Calendar σε παράθυρο [from_date, from_date + 14 μέρες].
    4. Πετάει slots που συγκρούονται με events.
    5. Επιστρέφει τα πρώτα max_slots διαθέσιμα.
    """
    print("GET_AVAILABILITY BODY:", body)

    # 1. Από ποια στιγμή ξεκινάμε: από body.from_date ή από τώρα (Asia/Nicosia).
    # Κρατάμε την ίδια normalisation λογική που χρησιμοποιούν και τα Cal suggestions
    # routes, ώστε να μην υπάρχει διαφορετική ερμηνεία ημερομηνιών μέσα στο backend.
    if body.from_date is None:
        start_from = datetime.now(TZ_NICOSIA)
    elif isinstance(body.from_date, datetime):
        if body.from_date.tzinfo is None:
            start_from = body.from_date.replace(tzinfo=TZ_NICOSIA)
        else:
            start_from = body.from_date.astimezone(TZ_NICOSIA)
    else:
        reference_date = resolve_reference_date(body.from_date)
        start_from = datetime(
            year=reference_date.year,
            month=reference_date.month,
            day=reference_date.day,
            hour=9,
            minute=0,
            tzinfo=TZ_NICOSIA,
        )

    clinic = normalize_clinic_name(body.clinic, default="")
    if clinic not in {"limassol", "evrychou"}:
        raise HTTPException(status_code=400, detail=f"Unknown clinic: {body.clinic}")

    # 2. Παράγουμε slots σύμφωνα με το πρόγραμμα της κλινικής
    try:
        raw_slots = generate_clinic_slots(
            clinic=clinic,
            from_date=start_from,
            duration_minutes=body.duration_minutes,
            max_slots=body.max_slots,
        )
    except ValueError as e:
        # Άγνωστη κλινική
        print("ERROR GENERATING SLOTS:", e)
        raise HTTPException(status_code=400, detail=str(e))

    if not raw_slots:
        # Δεν βρέθηκαν ραντεβού με βάση το πρόγραμμα
        return GetAvailabilityResponse(slots=[])

    # 3. Ορίζουμε παράθυρο για Calendar events
    explicit_date = body.from_date is not None

    if explicit_date:
        # Αν ο ασθενής ζήτησε συγκεκριμένη μέρα, περιορίζουμε το παράθυρο σε εκείνη +1 μέρα
        window_start = start_from.replace(hour=0, minute=0, second=0, microsecond=0)
        window_end = window_start + timedelta(days=1)
    else:
        # Default: 14 μέρες μπροστά
        window_start = start_from
        window_end = start_from + timedelta(days=14)

    # 4. Φέρνουμε events από Google Calendar
    try:
        events = await asyncio.to_thread(
            get_gcal_events,
            calendar_id=CALENDAR_ID,
            time_min=window_start,
            time_max=window_end,
        )
    except Exception as e:
        # Αν σπάσει η κλήση στο Calendar, για ασφάλεια επιστρέφουμε μόνο raw_slots
        print("ERROR FETCHING GCAL EVENTS:", e)
        return GetAvailabilityResponse(slots=raw_slots[: body.max_slots])

    # 5. Φιλτράρουμε slots που συγκρούονται με events
    filtered_slots = filter_slots_with_calendar(raw_slots, events)

    # 6. Παίρνουμε μέχρι max_slots
    final_slots = filtered_slots[: body.max_slots]

    return GetAvailabilityResponse(slots=final_slots)

# ============================================================
# Hard-coded actions για Limassol / Evrychou
# ============================================================

@app.post("/actions/get_availability_limassol", response_model=GetAvailabilityResponse, dependencies=[Depends(require_action_token)])
async def get_availability_limassol():
    """
    Action μόνο για Λεμεσό.
    Στη Synthflow δεν χρειάζεται καθόλου variable clinic.
    """
    body = GetAvailabilityRequest(
        clinic="limassol",
        duration_minutes=20,
        max_slots=25,
        from_date=None,
    )
    return await get_availability_action(body)


@app.post("/actions/get_availability_evrychou", response_model=GetAvailabilityResponse, dependencies=[Depends(require_action_token)])
async def get_availability_evrychou():
    """
    Action μόνο για Ευρύχου.
    """
    body = GetAvailabilityRequest(
        clinic="evrychou",
        duration_minutes=20,
        max_slots=25,
        from_date=None,
    )
    return await get_availability_action(body)

# ============================================================
# Book
# ============================================================

def infer_duration_from_notes(notes: Optional[str], default_minutes: int = 20) -> int:
    """
    Επιστρέφει διάρκεια ραντεβού (σε λεπτά) με βάση το κείμενο notes.
    Κανόνες (ενημερωμένο 2026-06-20):
      - Aclasta → 60'
      - Οστεοπόρωση (πρώτη επίσκεψη) → 60', επαναξιολόγηση → 40'
      - Τροχαίο / παραπομπή από δικηγόρο → 40'
      - Ενέσεις (π.χ. Prolia) / ακτινογραφίες / MRI / αναλύσεις → 10'
      - Αλλιώς → default (π.χ. 20')
    """
    return infer_appointment_duration_minutes(notes, default_minutes=default_minutes)


@app.post("/actions/book", response_model=BookResponse, dependencies=[Depends(require_action_token)])
async def book_action(body: BookRequest):
    """
    Action για booking ραντεβού στο Google Calendar.

    Κανόνες διάρκειας:
      - Οστεοπόρωση → 60 λεπτά
      - Ακτινογραφίες / MRI / Prolia → 15 λεπτά
      - Αλλιώς → body.duration_minutes (default 20')
    """
    print("BOOK BODY:", body)

    # 1. Υπολογίζουμε start_dt από ημερομηνία + ώρα
    try:
        start_str = f"{body.appointment_date}T{body.appointment_time}"
        start_dt = datetime.fromisoformat(start_str)
    except Exception as e:
        print("ERROR PARSING DATETIME:", e)
        raise HTTPException(status_code=400, detail=f"Invalid date/time format: {str(e)}")

    # 2. Υπολογίζουμε duration ανάλογα με το notes
    try:
        duration_minutes = infer_duration_from_notes(body.notes, default_minutes=body.duration_minutes)
    except Exception as e:
        # αν γίνει κάτι πολύ περίεργο, πέφτουμε πίσω στο default
        print("ERROR INFERRING DURATION:", e)
        duration_minutes = body.duration_minutes

    end_dt = start_dt + timedelta(minutes=duration_minutes)

    start_iso = start_dt.isoformat()
    end_iso = end_dt.isoformat()

    try:
        full_name = f"{body.first_name} {body.last_name}".strip()
        summary = f"Ραντεβού Ortho Papachristou ({body.clinic})"
        description_parts = [
            f"Ασθενής: {full_name}",
            f"Τηλ.: {body.phone_number}",
            f"Κλινική: {body.clinic}",
            f"Διάρκεια: {duration_minutes} λεπτά",
        ]
        if body.notes:
            description_parts.append(f"Σημειώσεις: {body.notes}")
        description = "\n".join(description_parts)

        event_id = await asyncio.to_thread(
            create_gcal_event,
            calendar_id=CALENDAR_ID,
            start_iso=start_iso,
            end_iso=end_iso,
            summary=summary,
            description=description,
            timezone_str="Asia/Nicosia",
        )

        return BookResponse(
            success=True,
            appointment_id=event_id,
            message="Το ραντεβού καταχωρήθηκε στο Google Calendar.",
        )

    except Exception as e:
        print("BOOKING ERROR:", e)
        raise HTTPException(status_code=500, detail=f"Booking failed: {str(e)}")

# ============================================================
# Cancel appointment
# ============================================================

@app.post("/actions/cancel_appointment", response_model=BookResponse, dependencies=[Depends(require_action_token)])
async def cancel_appointment_action(body: CancelRequest):
    """
    Action για ακύρωση ραντεβού στο Google Calendar.
    Χρησιμοποιεί day/month (και optional year) αντί για appointment_date.

    Λογική:
    - Αν appointment_year είναι None, χρησιμοποιεί το τρέχον έτος.
    - Αν appointment_year είναι "κουφό", το διορθώνει στο τρέχον έτος.
    - Βρίσκει event με βάση:
      * ώρα / διάρκεια
      * κλινική (limassol / evrychou)
      * πρώτα τηλεφωνο, μετά επώνυμο, μετά fallback μοναδικό event στο παράθυρο.
    """
    print("CANCEL BODY:", body)

    now = datetime.now(TZ_NICOSIA)

    # 1. Επιλογή έτους
    year = body.appointment_year or now.year
    if year < now.year - 1 or year > now.year + 1:
        print(f"ADJUSTING YEAR in cancel from {year} to {now.year}")
        year = now.year

    # 2. Δημιουργία start/end datetime
    try:
        hour = int(body.appointment_time[:2])
        minute = int(body.appointment_time[3:5])

        start_dt = datetime(
            year=year,
            month=body.appointment_month,
            day=body.appointment_day,
            hour=hour,
            minute=minute,
            tzinfo=TZ_NICOSIA,
        )
        end_dt = start_dt + timedelta(minutes=body.duration_minutes)
    except Exception as e:
        print("ERROR PARSING DATETIME (cancel):", e)
        raise HTTPException(status_code=400, detail=f"Invalid date/time format: {str(e)}")

    # 3. Παράθυρο γύρω από την ώρα (10' πριν, 10' μετά)
    window_start = start_dt - timedelta(minutes=10)
    window_end = end_dt + timedelta(minutes=10)

    try:
        events = await asyncio.to_thread(
            get_gcal_events,
            calendar_id=CALENDAR_ID,
            time_min=window_start,
            time_max=window_end,
        )
    except Exception as e:
        print("ERROR FETCHING GCAL EVENTS (cancel):", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch calendar events: {str(e)}")

    print(f"GCAL LIST EVENTS (cancel) COUNT: {len(events)}")

    # 4. Βρίσκουμε το event με τη νέα phone-first λογική
    event_id = find_matching_event(
        events=events,
        start_dt=start_dt,
        end_dt=end_dt,
        clinic=body.clinic,
        first_name=body.first_name,
        last_name=body.last_name,
        phone_number=body.phone_number,
    )

    print("MATCHING EVENT ID (cancel):", event_id)

    if not event_id:
        return BookResponse(
            success=False,
            appointment_id=None,
            message="Δεν βρέθηκε ραντεβού με αυτά τα στοιχεία για ακύρωση.",
        )

    # 5. Διαγραφή event στο Google Calendar
    try:
        session = get_gcal_session()
        url = f"https://www.googleapis.com/calendar/v3/calendars/{CALENDAR_ID}/events/{event_id}"
        print("GCAL DELETE EVENT URL:", url)
        resp = await asyncio.to_thread(session.delete, url)

        if resp.status_code not in (200, 204):
            print("GCAL DELETE ERROR BODY:", resp.text)
            raise RuntimeError(
                f"Google Calendar API delete error {resp.status_code}: {resp.text}"
            )

        return BookResponse(
            success=True,
            appointment_id=event_id,
            message="Το ραντεβού ακυρώθηκε στο Google Calendar.",
        )
    except Exception as e:
        print("CANCEL ERROR:", e)
        raise HTTPException(status_code=500, detail=f"Cancel failed: {str(e)}")



# ============================================================
# Cancel appointment by phone — GCal fallback
# ============================================================
#
# Χρησιμοποιείται όταν το check_appointment_by_phone επιστρέφει
# found=False (δεν βρέθηκε ραντεβού στο Cal.com) κατά τη ροή
# ακύρωσης. Ψάχνει στο Google Calendar για επερχόμενα events με
# το τηλέφωνο του caller στο description και τα ακυρώνει.
#
# Endpoint: POST /actions/gcal_cancel_by_phone

class GCalCancelByPhoneRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    phone_number: Optional[str] = None
    caller_number: Optional[str] = None
    call_id: Optional[str] = None
    clinic: Optional[str] = None          # "limassol" | "evrychou" | "any" | None
    lookahead_days: int = 90              # πόσες μέρες μπροστά να ψάξει
    cancellation_reason: Optional[str] = "Caller requested cancellation via phone"


class GCalCancelByPhoneResponse(BaseModel):
    success: bool
    found: bool
    cancelled_count: int = 0
    cancelled_events: List[Dict[str, Any]] = []
    spoken_response: str
    state: Optional[Dict[str, Any]] = None


@app.post("/actions/gcal_cancel_by_phone", response_model=GCalCancelByPhoneResponse, dependencies=[Depends(require_action_token)])
async def gcal_cancel_by_phone_action(body: GCalCancelByPhoneRequest):
    """
    Ψάχνει στο Google Calendar για επερχόμενα ραντεβού που έχουν
    το τηλέφωνο του caller στο description, και τα ακυρώνει.

    Χρησιμοποιείται ως fallback όταν το check_appointment_by_phone
    δεν βρίσκει τίποτα στο Cal.com κατά τη ροή ακύρωσης.
    """
    phone = (
        _caller_number_from_body(body)
        or (body.phone_number or "").strip()
        or (body.caller_number or "").strip()
    )

    if not phone:
        return GCalCancelByPhoneResponse(
            success=False,
            found=False,
            spoken_response="Δεν μπόρεσα να εντοπίσω αριθμό τηλεφώνου για αναζήτηση στο ημερολόγιο.",
        )

    now_local = datetime.now(TZ_NICOSIA)
    search_end = now_local + timedelta(days=body.lookahead_days)

    try:
        events = await asyncio.to_thread(
            get_gcal_events,
            calendar_id=CALENDAR_ID,
            time_min=now_local,
            time_max=search_end,
        )
    except Exception as e:
        print("GCAL_CANCEL_BY_PHONE: error fetching events:", e)
        return GCalCancelByPhoneResponse(
            success=False,
            found=False,
            spoken_response="Παρουσιάστηκε πρόβλημα κατά την αναζήτηση στο ημερολόγιο.",
        )

    # Φιλτράρισμα: events που έχουν το τηλέφωνο στο description
    # Το description έχει τη μορφή: "Τηλ.: +35799xxxxxx"
    phone_norm = phone.strip().lower()

    # Αν ζητηθεί φιλτράρισμα κλινικής
    from appointment_utils import normalize_clinic_name
    clinic_filter = normalize_clinic_name(body.clinic or "", default="")
    clinic_tag = f"({clinic_filter})" if clinic_filter else None

    matching_events = []
    for ev in events:
        description = (ev.get("description") or "").lower()
        summary = (ev.get("summary") or "").lower()

        if phone_norm not in description:
            continue

        # Αν υπάρχει clinic filter, ελέγχουμε το summary
        if clinic_tag and clinic_tag not in summary:
            continue

        matching_events.append(ev)

    if not matching_events:
        print(f"GCAL_CANCEL_BY_PHONE: no events found for phone={phone}")
        return GCalCancelByPhoneResponse(
            success=True,
            found=False,
            spoken_response="Δεν βρήκα καταχωρημένο ραντεβού στο ημερολόγιο για αυτόν τον αριθμό.",
        )

    # Ακύρωση όλων των matching events
    session = get_gcal_session()
    cancelled = []
    errors = []

    for ev in matching_events:
        ev_id = ev.get("id")
        ev_summary = ev.get("summary", "")
        ev_start = (ev.get("start") or {}).get("dateTime", "")

        try:
            url = f"https://www.googleapis.com/calendar/v3/calendars/{CALENDAR_ID}/events/{ev_id}"
            resp = await asyncio.to_thread(session.delete, url)
            if resp.status_code in (200, 204):
                cancelled.append({
                    "event_id": ev_id,
                    "summary": ev_summary,
                    "start": ev_start,
                })
                print(f"GCAL_CANCEL_BY_PHONE: cancelled event {ev_id} ({ev_summary})")
            else:
                errors.append({"event_id": ev_id, "error": resp.text})
                print(f"GCAL_CANCEL_BY_PHONE: failed to cancel {ev_id}: {resp.text}")
        except Exception as e:
            errors.append({"event_id": ev_id, "error": str(e)})
            print(f"GCAL_CANCEL_BY_PHONE: exception cancelling {ev_id}: {e}")

    # Ενημέρωση conversation state
    state = update_conversation_state(
        phone_number=phone,
        call_id=body.call_id,
        cancel_completed=bool(cancelled),
        cancellation_status="cancelled" if cancelled else "failed",
        current_intent="cancellation",
        current_stage="cancelled" if cancelled else "cancel_failed",
        next_action="end_or_new_request" if cancelled else "offer_message",
        current_clinic=clinic_filter or None,
    )

    if cancelled:
        # Φιλική απάντηση με στοιχεία ραντεβού αν είναι δυνατόν
        if len(cancelled) == 1:
            ev_start_str = cancelled[0].get("start", "")
            try:
                dt = datetime.fromisoformat(ev_start_str).astimezone(TZ_NICOSIA)
                from appointment_utils import make_friendly_day_gr, make_friendly_time_gr
                friendly = f"{make_friendly_day_gr(dt)} {make_friendly_time_gr(dt)}"
                spoken = f"Το ραντεβού {friendly} ακυρώθηκε."
            except Exception:
                spoken = "Το ραντεβού σας ακυρώθηκε."
        else:
            spoken = f"Ακυρώθηκαν {len(cancelled)} ραντεβού."

        return GCalCancelByPhoneResponse(
            success=True,
            found=True,
            cancelled_count=len(cancelled),
            cancelled_events=cancelled,
            spoken_response=spoken,
            state=state,
        )

    return GCalCancelByPhoneResponse(
        success=False,
        found=True,
        cancelled_count=0,
        spoken_response="Βρήκα ραντεβού αλλά παρουσιάστηκε πρόβλημα κατά την ακύρωση. Θα ενημερώσω τον γιατρό.",
        state=state,
    )


# ============================================================
# Replace Appointment Attendee
# ============================================================

def _booking_data_from_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = raw.get("data") if isinstance(raw, dict) else None
    return data if isinstance(data, dict) else raw if isinstance(raw, dict) else {}


def _attendee_list_from_payload(*payloads: Dict[str, Any]) -> List[Dict[str, Any]]:
    attendees: List[Dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        data = payload.get("data", payload)
        if isinstance(data, list):
            attendees.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            nested = data.get("attendees")
            if isinstance(nested, list):
                attendees.extend(item for item in nested if isinstance(item, dict))
            elif data.get("id") or data.get("email") or data.get("name"):
                attendees.append(data)
    return attendees


def _first_present(*values: Any) -> Optional[str]:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _sms_email_from_phone_or_name(phone_number: Optional[str], full_name: str, cal_client: CalClient) -> str:
    phone = normalize_booking_phone(phone_number)
    if phone:
        digits = re.sub(r"\D", "", phone)
        if digits:
            return f"{digits}@sms.cal.com"
    return cal_client._email_from_name(full_name)


def _booking_title_for_attendee(booking: Dict[str, Any], clinic: Optional[str], full_name: str) -> str:
    normalized_clinic = normalize_clinic_name(clinic, default="")
    if normalized_clinic == "limassol":
        return f"Limassol booking between Athanasios Papachristou and {full_name}"
    if normalized_clinic == "evrychou":
        return f"Evrychou booking between Athanasios Papachristou and {full_name}"

    existing_title = str(booking.get("title") or "").strip()
    marker = " and "
    if marker in existing_title:
        return f"{existing_title.rsplit(marker, 1)[0]}{marker}{full_name}"
    return f"Booking between Athanasios Papachristou and {full_name}"


async def _find_setmore_key_for_booking(
    *,
    cal_uid: Optional[str],
    start_iso: Optional[str],
) -> Optional[str]:
    if not cal_uid and not start_iso:
        return None
    try:
        if start_iso:
            start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(TZ_NICOSIA)
            start_date = start_dt.date() - timedelta(days=1)
            end_date = start_dt.date() + timedelta(days=1)
        else:
            today = datetime.now(TZ_NICOSIA).date()
            start_date = today - timedelta(days=30)
            end_date = today + timedelta(days=365)
        snapshots = await list_setmore_synced_appointments(
            start_date=start_date,
            end_date=end_date,
            page_limit=10,
        )
    except Exception as exc:
        print("SETMORE FIND KEY ERROR:", exc)
        return None

    for snapshot in snapshots:
        if cal_uid and snapshot.cal_uid == cal_uid:
            return snapshot.appointment_key
    if start_iso:
        target = start_iso.replace(".000Z", "Z")
        for snapshot in snapshots:
            if (snapshot.start_time or "").replace(".000Z", "Z") == target:
                return snapshot.appointment_key
    return None


@app.post("/actions/replace_appointment_attendee", response_model=ReplaceAppointmentAttendeeResponse, dependencies=[Depends(require_action_token)])
async def replace_appointment_attendee_action(body: ReplaceAppointmentAttendeeRequest):
    """Replace the visible patient/attendee for an existing appointment when possible.

    Cal.com has an official endpoint for adding attendees. For true replacement,
    we first try the existing v1 attendee patch that is already used by this
    backend for name corrections, then fall back to adding the new person.
    """
    booking_uid = (body.booking_uid or body.cal_uid or "").strip()
    full_name = (body.full_name or " ".join([body.first_name or "", body.last_name or ""])).strip()
    if not booking_uid:
        return ReplaceAppointmentAttendeeResponse(
            success=False,
            message="Λείπει το Cal booking UID.",
            errors=["missing_booking_uid"],
        )
    if not full_name:
        return ReplaceAppointmentAttendeeResponse(
            success=False,
            booking_uid=booking_uid,
            message="Λείπει το νέο όνομα ασθενούς.",
            errors=["missing_full_name"],
        )

    cal_client = CalClient()
    errors: List[str] = []
    raw: Dict[str, Any] = {}
    cal_updated = False
    cal_added_attendee = False
    cal_title_updated = False
    attendee_id = str(body.attendee_id or "").strip() or None
    booking_id = str(body.booking_id or "").strip() or None

    try:
        booking_raw = await cal_client.get_booking_by_uid_v2(booking_uid)
        raw["booking"] = booking_raw
        booking = _booking_data_from_response(booking_raw)
        booking_id = booking_id or _first_present(booking.get("id"), booking.get("bookingId"))
    except Exception as exc:
        booking = {}
        errors.append(f"cal_get_booking_failed: {str(exc)}")

    try:
        attendees_raw = await cal_client.get_booking_attendees_v2(booking_uid)
        raw["attendees"] = attendees_raw
        attendees = _attendee_list_from_payload(attendees_raw, booking)
        if attendees and not attendee_id:
            attendee_id = _first_present(attendees[0].get("id"), attendees[0].get("attendeeId"))
        if attendees and not booking_id:
            booking_id = _first_present(attendees[0].get("bookingId"), attendees[0].get("booking_id"))
    except Exception as exc:
        errors.append(f"cal_get_attendees_failed: {str(exc)}")

    timezone_name = body.timezone or "Asia/Nicosia"
    language = body.language or "el"
    email = (body.email or "").strip() or _sms_email_from_phone_or_name(
        body.phone_number or body.caller_number,
        full_name,
        cal_client,
    )
    phone_number = normalize_booking_phone(body.phone_number or body.caller_number)

    if attendee_id and booking_id:
        try:
            raw["cal_attendee_patch"] = await cal_client.update_attendee_name_v1(
                attendee_id=attendee_id,
                booking_id=booking_id,
                name=full_name,
                email=email,
                time_zone=timezone_name,
                phone_number=phone_number,
            )
            cal_updated = True
        except Exception as exc:
            errors.append(f"cal_patch_attendee_failed: {str(exc)}")

    if not cal_updated:
        try:
            raw["cal_attendee_add"] = await cal_client.add_booking_attendee_v2(
                booking_uid=booking_uid,
                name=full_name,
                email=email,
                phone_number=phone_number,
                timezone_str=timezone_name,
                language=language,
            )
            cal_added_attendee = True
        except Exception as exc:
            errors.append(f"cal_add_attendee_failed: {str(exc)}")

    if booking_id:
        try:
            title = _booking_title_for_attendee(booking, body.clinic, full_name)
            raw["cal_title_patch"] = await cal_client.update_booking_title_v1(
                booking_id=booking_id,
                title=title,
            )
            cal_title_updated = True
        except Exception as exc:
            errors.append(f"cal_update_title_failed: {str(exc)}")

    setmore_key = (body.setmore_appointment_key or "").strip() or await _find_setmore_key_for_booking(
        cal_uid=booking_uid,
        start_iso=body.start_iso or str(booking.get("start") or ""),
    )
    setmore_updated = False
    if setmore_key:
        setmore_notes = body.notes or f"customer replaced via backend | cal_uid={booking_uid}"
        setmore_result = await update_setmore_appointment_customer(
            appointment_key=setmore_key,
            full_name=full_name,
            phone_number=phone_number,
            email=email,
            notes=setmore_notes,
        )
        raw["setmore_update"] = {
            "success": setmore_result.success,
            "appointment_id": setmore_result.appointment_id,
            "error": setmore_result.error,
        }
        setmore_updated = bool(setmore_result.success)
        if not setmore_updated and setmore_result.error:
            errors.append(f"setmore_update_customer_failed: {setmore_result.error}")

    success = cal_updated or cal_added_attendee
    message = (
        "Το Cal ενημερώθηκε με το νέο πρόσωπο."
        if cal_updated
        else "Το νέο πρόσωπο προστέθηκε ως attendee στο Cal."
        if cal_added_attendee
        else "Δεν μπόρεσε να ενημερωθεί το Cal."
    )
    if success and not setmore_key:
        message += " Δεν βρέθηκε αντίστοιχο Setmore appointment για αλλαγή customer."
    elif success and setmore_updated:
        message += " Ενημερώθηκε και το Setmore."
    elif success and setmore_key:
        message += " Το Setmore δεν επιβεβαίωσε αλλαγή customer."

    state = update_conversation_state(
        phone_number=normalize_booking_phone(body.caller_number or body.phone_number),
        call_id=body.call_id if hasattr(body, "call_id") else None,
        current_intent="appointment_substitution",
        current_message=body.notes or "",
    )
    raw["state"] = state

    return ReplaceAppointmentAttendeeResponse(
        success=success,
        booking_uid=booking_uid,
        booking_id=booking_id,
        attendee_id=attendee_id,
        cal_updated=cal_updated,
        cal_added_attendee=cal_added_attendee,
        cal_title_updated=cal_title_updated,
        setmore_updated=setmore_updated,
        setmore_appointment_key=setmore_key,
        message=message,
        errors=errors,
        raw=raw,
    )


# ============================================================
# Message
# ============================================================

@app.post("/actions/message", response_model=MessageResponse, dependencies=[Depends(require_action_token)])
async def message_action(request: Request):
    """
    Action για μήνυμα προς τον γιατρό.
    Δέχεται:
      - είτε κανονικό JSON object,
      - είτε stringified JSON (όπως το στέλνει το Synthflow).
    """
    try:
        raw_body_bytes = await request.body()
        raw_body_str = raw_body_bytes.decode("utf-8") if raw_body_bytes else ""
        print("RAW MESSAGE BODY:", raw_body_str)

        data: dict

        try:
            parsed = json.loads(raw_body_str)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            data = parsed
        elif isinstance(parsed, str):
            try:
                data = json.loads(parsed)
            except Exception as e:
                print("ERROR PARSING NESTED JSON STRING:", e)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid nested JSON: {str(e)}",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Request body must be JSON object or JSON string",
            )

        print("PARSED MESSAGE DATA:", data)

        payload = MessagePayload(**data)
        message_phone = _caller_number_from_body(payload) or _extract_phone_from_mapping(data)
        if not message_phone and payload.call_id:
            message_phone = _extract_phone_from_mapping(
                get_conversation_state(call_id=payload.call_id)
            )
        priority_level = (
            payload.priority_level
            or classify_priority_level(
                f"{payload.message_clean or ''} {payload.message or ''}",
                payload.reason_for_call,
            )
        )
        message_category = (
            payload.message_category
            or rule_based_intent(f"{payload.message_clean or ''} {payload.message or ''}").get("message_category")
            or ""
        )
        if _looks_like_booking_failure_handoff(payload):
            # Dograh can time out just before Cal.com returns success. Give the
            # booking endpoint a moment to mark state before sending a false
            # "technical problem" email.
            await asyncio.sleep(2.0)
        if _is_false_booking_failure_after_success(payload, message_phone):
            print("MESSAGE SUPPRESSED: false booking failure after successful booking")
            state = update_conversation_state(
                phone_number=message_phone,
                call_id=payload.call_id,
                booking_completed=True,
                false_booking_failure_suppressed=True,
            )
            return MessageResponse(
                success=True,
                message="Το booking είχε ήδη ολοκληρωθεί επιτυχώς. Το ψεύτικο failure μήνυμα αγνοήθηκε.",
                spoken_response="Το ραντεβού σας κλείστηκε.",
                state=state,
            )
        if _is_false_cancellation_handoff_after_success(payload, message_phone):
            print("MESSAGE SUPPRESSED: cancellation already completed")
            state = update_conversation_state(
                phone_number=message_phone,
                call_id=payload.call_id,
                cancel_completed=True,
                cancellation_status="cancelled",
                false_cancellation_handoff_suppressed=True,
            )
            return MessageResponse(
                success=True,
                message="Η ακύρωση είχε ήδη ολοκληρωθεί επιτυχώς. Το περιττό μήνυμα ακύρωσης αγνοήθηκε.",
                spoken_response="Το ραντεβού σας ακυρώθηκε.",
                state=state,
            )

        incoming_message = (payload.message_clean or payload.message or "").strip()
        spoken_response = _message_spoken_response(
            message_category,
            incoming_message,
            payload.message or "",
            payload.message_clean or "",
            priority_level=priority_level,
        )
        previous_state = _message_state(message_phone, payload.call_id)
        merged_message = ""  # Θα ενημερωθεί αν υπάρχει merge/dedupe
        urgent_payload = {
            "caller_number": message_phone or payload.caller_number or payload.phone_number or "",
            "phone_number": message_phone or payload.caller_number or payload.phone_number or "",
            "call_id": payload.call_id or "",
            "reason_for_call": payload.reason_for_call or "urgent",
            "clinic": payload.clinic or "",
            "language": payload.language or "el",
            "first_name": payload.first_name or "",
            "last_name": payload.last_name or "",
            "is_urgent": True,
            "priority_level": "A",
            "message": payload.message or payload.message_clean or "",
            "message_clean": payload.message_clean or payload.message or "",
            "notes": payload.notes or incoming_message,
            "source": "send_message_auto_urgent",
        }
        if _should_update_existing_message(
            previous_state,
            message_category,
            priority_level,
            incoming_message,
        ):
            merged_message = _merge_message_context(
                str(previous_state.get("current_message") or ""),
                incoming_message,
            )
            merged_category = (
                message_category
                or str(previous_state.get("current_message_category") or "")
            )
            caller_wants_callback = bool(
                previous_state.get("caller_wants_callback")
            ) or _caller_wants_callback_from_text(merged_message)
            message_ledger = _append_message_ledger(
                previous_state,
                action="updated",
                text=incoming_message,
                category=merged_category,
                priority_level=priority_level,
            )
            print("MESSAGE MERGE: existing message updated with new context")
            state = update_conversation_state(
                phone_number=message_phone,
                call_id=payload.call_id,
                message_already_captured=True,
                message_sent=True,
                current_stage="message_sent",
                next_action="end_or_new_request",
                current_intent=payload.reason_for_call
                or str(previous_state.get("current_intent") or "")
                or "message",
                current_message=merged_message,
                current_clinic=payload.clinic
                or str(previous_state.get("current_clinic") or ""),
                current_message_category=merged_category,
                message_ledger=message_ledger,
                message_updates_count=int(previous_state.get("message_updates_count") or 0) + 1,
                caller_wants_callback=caller_wants_callback or None,
                urgent_handoff_sent=previous_state.get("urgent_handoff_sent") or None,
            )
            if str(priority_level or "").upper() == "A" and not previous_state.get("urgent_handoff_sent"):
                asyncio.create_task(_safe_post_urgent_outbound(urgent_payload))
                state = update_conversation_state(
                    phone_number=message_phone,
                    call_id=payload.call_id,
                    urgent_handoff_sent=True,
                )
            return MessageResponse(
                success=True,
                message="Το υπάρχον μήνυμα ενημερώθηκε.",
                spoken_response="Το σημείωσα.",
                state=state,
            )

        dedupe_key = message_guard_key(
            message_phone,
            payload.message,
            payload.message_clean,
        )
        if not claim_action_once(dedupe_key, ttl_seconds=15 * 60):
            print("MESSAGE DEDUPE: duplicate message ignored", dedupe_key)
            merged_message = _merge_message_context(
                str(previous_state.get("current_message") or ""),
                incoming_message,
            )
            state = update_conversation_state(
                phone_number=message_phone,
                call_id=payload.call_id,
                message_already_captured=True,
                message_sent=True,
                current_stage="message_sent",
                next_action="end_or_new_request",
                current_intent=payload.reason_for_call or "message",
                current_message=merged_message or incoming_message,
                current_message_category=message_category
                or str(previous_state.get("current_message_category") or ""),
                message_ledger=_append_message_ledger(
                    previous_state,
                    action="deduped",
                    text=incoming_message,
                    category=message_category
                    or str(previous_state.get("current_message_category") or ""),
                    priority_level=priority_level,
                ),
            )
            return MessageResponse(
                success=True,
                message="Το μήνυμα είχε ήδη καταγραφεί.",
                spoken_response="Το σημείωσα.",
                state=state,
            )

        full_name = " ".join(
            [x for x in [payload.first_name, payload.last_name] if x]
        ).strip() or "-"

        subject = "Νέο μήνυμα ασθενούς από Natasa"

        lines = []
        lines.append(f"Όνομα ασθενούς: {full_name}")
        if message_phone:
            lines.append(f"Τηλέφωνο: {message_phone}")
        if payload.reason_for_call:
            lines.append(f"Κατηγορία: {payload.reason_for_call}")
        if message_category:
            lines.append(f"Υποκατηγορία: {message_category}")
        if payload.clinic:
            lines.append(f"Κλινική: {payload.clinic}")
        if payload.is_urgent is not None:
            lines.append(f"Επείγον: {'ναι' if payload.is_urgent else 'όχι'}")
        if priority_level:
            lines.append(f"Προτεραιότητα: {_priority_label(priority_level)}")
        if payload.appointment_date or payload.appointment_time:
            lines.append(
                f"Ραντεβού αναφοράς: {(payload.appointment_date or '-')} {(payload.appointment_time or '-')}"
            )
        if payload.message_clean:
            lines.append(f"Σύνοψη: {payload.message_clean}")
        lines.append("")
        lines.append("Μήνυμα:")
        lines.append(payload.message)

        text_content = "\n".join(lines)

        html_content = f"""
        <h3>Νέο μήνυμα ασθενούς από Natasa</h3>
        <p><strong>Όνομα:</strong> {full_name}<br/>
        <strong>Τηλέφωνο:</strong> {message_phone or '-'}<br/>
        <strong>Κατηγορία:</strong> {payload.reason_for_call or '-'}<br/>
        <strong>Υποκατηγορία:</strong> {message_category or '-'}<br/>
        <strong>Κλινική:</strong> {payload.clinic or '-'}<br/>
        <strong>Επείγον:</strong> {('ναι' if payload.is_urgent else 'όχι') if payload.is_urgent is not None else '-'}<br/>
        <strong>Προτεραιότητα:</strong> {_priority_label(priority_level)}<br/>
        <strong>Ραντεβού αναφοράς:</strong> {(payload.appointment_date or '-')} {(payload.appointment_time or '-')}<br/>
        <strong>Σύνοψη:</strong> {payload.message_clean or '-'}<br/></p>
        <p><strong>Μήνυμα:</strong><br/>{payload.message}</p>
        """

        await asyncio.to_thread(
            send_slack_notification,
            subject=subject,
            text_content=text_content,
            emoji=":speech_balloon:",
        )
        await asyncio.to_thread(
            send_brevo_email,
            subject=subject,
            text_content=text_content,
            html_content=html_content,
        )

        _log_call_event({
            "type": "message",
            "phone": message_phone or payload.caller_number or payload.phone_number or "",
            "first_name": payload.first_name or "",
            "last_name": payload.last_name or "",
            "message": merged_message or incoming_message or "",
            "clinic": payload.clinic or "",
            "priority": priority_level or "C",
            "is_urgent": bool(str(priority_level or "").upper() == "A"),
        })
        caller_wants_callback = _caller_wants_callback_from_text(
            payload.message_clean or "",
            payload.message or "",
        )
        urgent_handoff_sent = False
        if str(priority_level or "").upper() == "A":
            asyncio.create_task(_safe_post_urgent_outbound(urgent_payload))
            urgent_handoff_sent = True
        state = update_conversation_state(
            phone_number=message_phone,
            call_id=payload.call_id,
            message_already_captured=True,
            message_sent=True,
            current_stage="message_sent",
            next_action="end_or_new_request",
            current_intent=payload.reason_for_call or "message",
            current_message=payload.message_clean or payload.message,
            current_clinic=payload.clinic or "",
            current_message_category=message_category,
            message_ledger=_append_message_ledger(
                previous_state,
                action="sent",
                text=payload.message_clean or payload.message,
                category=message_category,
                priority_level=priority_level,
            ),
            caller_wants_callback=caller_wants_callback or None,
            urgent_handoff_sent=urgent_handoff_sent or None,
        )

        return MessageResponse(
            success=True,
            message="Το μήνυμά σας καταγράφηκε και εστάλη email στον γιατρό.",
            spoken_response=spoken_response,
            state=state,
        )
    except HTTPException:
        if "dedupe_key" in locals():
            release_action_claim(dedupe_key)
        raise
    except Exception as e:
        if "dedupe_key" in locals():
            release_action_claim(dedupe_key)
        print("MESSAGE ERROR:", e)
        raise HTTPException(status_code=500, detail=f"Message failed: {str(e)}")


@app.post("/actions/send_message", response_model=MessageResponse, dependencies=[Depends(require_action_token)])
async def send_message_action(request: Request):
    """
    Backward-compatible alias for workflows/tools that call `send_message`
    instead of `message`.
    """
    return await message_action(request)

# ============================================================
# Cypriot BERT Endpoints
# ============================================================

@app.post("/cypriot/encode", response_model=CypriotEncodeResponse)
async def cypriot_encode(body: CypriotEncodeRequest):
    """
    Δέχεται κείμενο (κυπριακά/ελληνικά) και επιστρέφει sentence embedding
    από το Κυπριακό BERT.

    Χρήσιμο για:
      - semantic search
      - intent classification (από άλλο service)
      - αποθήκευση σε vector DB
    """
    if not CY_BERT_READY:
        raise HTTPException(status_code=503, detail="Cypriot BERT model not available")

    try:
        emb = encode_cypriot_text(body.text)
    except Exception as e:
        print("ERROR in cypriot_encode:", e)
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")

    return CypriotEncodeResponse(embedding=emb)


# ============================================================
# Zadarma Integration
# ============================================================

logger = logging.getLogger("zadarma_webhook")
logging.basicConfig(level=logging.INFO)

@app.api_route("/zadarma/events", methods=["GET", "POST"], response_class=PlainTextResponse)
async def zadarma_events(
    request: Request,
    zd_echo: Optional[str] = Query(None)
) -> PlainTextResponse:
    """
    Endpoint for Zadarma event notifications (PBX calls & events).

    Behavior:
    - If zd_echo is present: echo it back and exit (verification step).
    - Otherwise: parse the event payload and handle it.
    """

    # 1. Verification step
    if zd_echo is not None:
        logger.info(f"Zadarma verification request with zd_echo={zd_echo}")
        return PlainTextResponse(content=zd_echo, status_code=200)

    # 2. Real event handling
    received_at = datetime.now(timezone.utc).isoformat()

    try:
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            payload: Any = await request.json()
        else:
            form = await request.form()
            payload = {k: v for k, v in form.items()}
    except Exception as e:
        logger.error(f"Error parsing Zadarma event payload: {e}")
        payload = {}

    # Χοντρό log για να δούμε ακριβώς τι στέλνει ο Zadarma
    try:
        headers_dict = dict(request.headers)
    except Exception:
        headers_dict = {}

    logger.info("########## ZADARMA REAL EVENT ##########")
    logger.info(json.dumps(
        {
            "received_at": received_at,
            "headers": headers_dict,
            "payload": payload,
        },
        ensure_ascii=False
    ))
    logger.info("########################################")

    # Επιπλέον σύντομο log (προαιρετικό)
    logger.info("Received Zadarma event (summary)")
    logger.info(json.dumps(
        {
            "received_at": received_at,
            "payload_keys": list(payload.keys()) if isinstance(payload, dict) else None,
        },
        ensure_ascii=False
    ))

    return PlainTextResponse(content="OK", status_code=200)

# ============================================================
# Webhook POST
# ============================================================



# ============================================================
# Abandoned-call detection
# ============================================================

ABANDONED_CALL_CHECK_DELAY_SECONDS = float(
    os.getenv("ABANDONED_CALL_CHECK_DELAY_SECONDS", "600")  # 10 λεπτά μετά το call_started
)


async def _abandoned_call_check(phone: str, call_id: str | None) -> None:
    """Τρέχει N λεπτά μετά την έναρξη κλήσης. Αν η κλήση δεν κατέληξε σε
    booking / μήνυμα / ακύρωση, στέλνει email στον γιατρό με το τηλέφωνο,
    ώστε να μπορεί να καλέσει πίσω τον ασθενή."""
    try:
        await asyncio.sleep(ABANDONED_CALL_CHECK_DELAY_SECONDS)
        state = get_conversation_state(phone_number=phone, call_id=call_id)

        # Κατέγραψε ΠΑΝΤΑ μετρικές για την κλήση (completed ή abandoned).
        try:
            from call_metrics import record_call_metric
            record_call_metric(state, phone=phone, call_id=call_id)
        except Exception as metric_exc:
            print(f"CALL METRICS error: {metric_exc}")

        completed = bool(
            state.get("booking_completed")
            or state.get("message_sent")
            or state.get("cancel_completed")
            or state.get("urgent_handoff_sent")
        )
        if completed or state.get("call_noise_ignored"):
            return

        intent = str(state.get("current_intent") or "").strip() or "—"
        last_msg = str(state.get("current_message") or "").strip() or "—"
        clinic = str(state.get("current_clinic") or "").strip() or "—"

        body = "\n".join(
            [
                "Πιθανή εγκαταλελειμμένη κλήση (δεν ολοκληρώθηκε booking/μήνυμα/ακύρωση).",
                f"Τηλέφωνο: {phone}",
                f"Πρόθεση: {intent}",
                f"Κλινική: {clinic}",
                f"Τελευταίο μήνυμα: {last_msg}",
                "",
                "Ίσως αξίζει ένα τηλεφώνημα πίσω στον ασθενή.",
            ]
        )
        await asyncio.to_thread(
            send_slack_notification,
            subject="📵 Εγκαταλελειμμένη κλήση",
            text_content=body,
            emoji=":missed_call:",
        )
        await asyncio.to_thread(
            send_brevo_email,
            subject="Εγκαταλελειμμένη κλήση — χρειάζεται follow-up",
            text_content=body,
        )
        print(f"ABANDONED CALL EMAIL sent for {phone}")
    except Exception as exc:
        print(f"ABANDONED CALL CHECK error for {phone}: {exc}")

async def require_webhook_token(request: Request) -> None:
    """
    Όπως το require_action_token, αλλά δέχεται το secret είτε από το header
    X-Reception-Token / Authorization Bearer, ΕΙΤΕ από query param (?token= / ?key=).
    Το Dograh webhook config μπορεί να μη στέλνει custom headers, οπότε το
    secret μπαίνει στο ίδιο το URL του webhook.
    """
    if not ACTIONS_SHARED_SECRET:
        return  # auth disabled
    provided = (request.headers.get(ACTION_TOKEN_HEADER) or "").strip()
    if not provided:
        auth = (request.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    if not provided:
        provided = (request.query_params.get("token") or request.query_params.get("key") or "").strip()
    if not hmac.compare_digest(provided, ACTIONS_SHARED_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/dograh/webhook")
async def dograh_webhook(request: Request):
    """
    Accepts flexible Dograh payloads (object, stringified object, nested body)
    and never rejects with 422 due to schema shape.
    """
    raw_body = await request.body()
    raw_text = raw_body.decode("utf-8", errors="ignore") if raw_body else ""

    data: Any = {}
    try:
        parsed = json.loads(raw_text) if raw_text else {}
    except Exception:
        parsed = {}

    if isinstance(parsed, dict):
        data = parsed
    elif isinstance(parsed, str):
        try:
            nested = json.loads(parsed)
            data = nested if isinstance(nested, dict) else {}
        except Exception:
            data = {}
    else:
        data = {}

    if isinstance(data, dict) and isinstance(data.get("payload"), dict):
        data = data["payload"]
    elif isinstance(data, dict) and isinstance(data.get("data"), dict):
        data = data["data"]

    try:
        if hasattr(DograhWebhookPayload, "model_validate"):
            payload = DograhWebhookPayload.model_validate(data if isinstance(data, dict) else {})
        else:
            payload = DograhWebhookPayload.parse_obj(data if isinstance(data, dict) else {})
    except Exception as e:
        print("DOGRAH WEBHOOK VALIDATION ERROR:", str(e), "raw:", raw_text[:500])
        payload = DograhWebhookPayload()

    if not payload.call_id and payload.workflow_id and (payload.workflow_run_id or payload.run_id):
        payload.call_id = (
            f"dograh-{payload.workflow_id}-{payload.workflow_run_id or payload.run_id}"
        )

    phone = _caller_number_from_body(payload) or _extract_phone_from_mapping(data)

    # Fallback: αν δεν βρέθηκε τηλέφωνο από τα mapped πεδία, ψάξε στο raw body
    # (Dograh μερικές φορές στέλνει caller_number σε nested context ή σε πεδία
    # εκτός του payload template)
    if not phone and raw_text:
        import re as _re
        # Ψάξε Κυπριακά νούμερα (+357...)
        _phone_match = _re.search(r'\+357\d{8}', raw_text)
        if _phone_match:
            phone = normalize_booking_phone(_phone_match.group(0))

    # Debug: log raw webhook data for troubleshooting missing phones
    _phone_debug = phone or "NO_PHONE"
    _call_id_debug = (data.get("call_id") or "")[:12]
    _nodes = data.get("nodes_visited") or data.get("call_tags") or []
    print(f"WEBHOOK: phone={_phone_debug} call_id={_call_id_debug} nodes={_nodes} keys={list(data.keys())[:10]}")

    # Robust extraction: ψάχνει σε payload, raw data, extracted_variables
    _extracted = data.get("extracted_variables") if isinstance(data, dict) else {}
    if not isinstance(_extracted, dict):
        _extracted = {}
    _webhook_message = (
        payload.message_clean or payload.message
        or (data.get("message_clean") if isinstance(data, dict) else None)
        or (data.get("message") if isinstance(data, dict) else None)
        or _extracted.get("message_clean")
        or _extracted.get("message")
        or ""
    ).strip()
    _webhook_reason = (
        payload.reason_for_call
        or (data.get("reason_for_call") if isinstance(data, dict) else None)
        or _extracted.get("reason_for_call")
        or ""
    ).strip().lower()

    if phone or payload.call_id:
        update_conversation_state(
            phone_number=phone,
            call_id=payload.call_id,
            caller_number=phone,
            current_intent=_webhook_reason or None,
            current_message=_webhook_message or None,
            current_clinic=payload.clinic or None,
            priority_level=payload.priority_level or None,
        )


    booking_recovery = await _recover_pending_finalize_booking(payload, phone=phone)

    # Προγραμμάτισε έλεγχο εγκαταλελειμμένης κλήσης + prefetch (fire-and-forget)
    event_type = (data.get("event_type") or "").strip()
    if event_type == "call_started":
        if phone:
            try:
                asyncio.create_task(_abandoned_call_check(phone, payload.call_id))
            except RuntimeError:
                pass

    email_lines = [
        f"Κλήση από {(payload.first_name or '').strip()} {(payload.last_name or '').strip()}".strip(),
        f"Τηλέφωνο: {phone}",
        f"Λόγος κλήσης: {payload.reason_for_call}",
        f"Κλινική: {payload.clinic}",
    ]

    if payload.language:
        email_lines.append(f"Γλώσσα: {payload.language}")
    if payload.is_urgent is not None:
        email_lines.append(f"Επείγον: {'ναι' if payload.is_urgent else 'όχι'}")
    priority_level = payload.priority_level or classify_priority_level(
        f"{payload.message_clean or ''} {payload.message or ''}",
        payload.reason_for_call,
    )
    if priority_level:
        email_lines.append(f"Προτεραιότητα: {_priority_label(priority_level)}")
    if payload.day_preference:
        email_lines.append(f"Προτίμηση μέρας: {payload.day_preference}")
    if payload.time_preference:
        email_lines.append(f"Προτίμηση ώρας: {payload.time_preference}")

    if payload.appointment_date or payload.appointment_time:
        email_lines.append(f"Ραντεβού: {payload.appointment_date or '-'} {payload.appointment_time or '-'}")

    if payload.notes:
        email_lines.append(f"Σημειώσεις: {payload.notes}")
    if payload.message:
        email_lines.append(f"Αρχικό μήνυμα: {payload.message}")
    if payload.message_clean:
        email_lines.append(f"Καθαρό μήνυμα: {payload.message_clean}")
    if booking_recovery:
        email_lines.append(f"Booking recovery: {booking_recovery}")

    _email_body = "\n".join(email_lines)

    # Safety net: if an older Dograh draft still sends an urgent payload here,
    # forward it to the same urgent outbound path used by the active workflow.
    if payload.is_urgent or (payload.reason_for_call or "").strip().lower() == "urgent":
        try:
            await _post_urgent_outbound(
                {
                    "caller_number": phone or payload.caller_number or payload.phone_number,
                    "phone_number": phone or payload.caller_number or payload.phone_number,
                    "call_id": payload.call_id or "",
                    "reason_for_call": payload.reason_for_call or "urgent",
                    "clinic": payload.clinic or "",
                    "language": payload.language or "el",
                    "first_name": payload.first_name or "",
                    "last_name": payload.last_name or "",
                    "is_urgent": True,
                    "priority_level": "A",
                    "message": payload.message or payload.message_clean or "",
                    "message_clean": payload.message_clean or payload.message or "",
                    "notes": payload.notes or "",
                    "source": "dograh",
                }
            )
        except Exception as e:
            print("DOGRAH URGENT OUTBOUND FORWARD ERROR:", str(e))

    # Ανίχνευση διακοπτόμενης/σύντομης κλήσης από το disposition του Dograh.
    disposition = (payload.call_disposition or payload.mapped_call_disposition or "").strip().lower()
    nodes = payload.nodes_visited if isinstance(payload.nodes_visited, list) else []
    # Η κλήση θεωρείται "διακόπηκε" αν ο καλών έκλεισε χωρίς ουσιαστική πρόοδο:
    # δηλαδή έμεινε μόνο στο αρχικό node και δεν συνέλεξε κανένα στοιχείο.
    has_content = bool(
        (_webhook_reason or "").strip()
        or (_webhook_message or "").strip()
        or (payload.first_name or "").strip()
        or str(payload.booking_uid or "").strip()
        or str(payload.booking_status or "").strip().lower() in {"confirmed", "success"}
    )
    minimal_progress = len(nodes) <= 1
    is_short_no_turn_noise = _is_short_no_user_turn_noise(payload, data if isinstance(data, dict) else {}, has_content=has_content)
    if is_short_no_turn_noise and (phone or payload.call_id):
        update_conversation_state(
            phone_number=phone,
            call_id=payload.call_id,
            call_noise_ignored=True,
            call_noise_reason="short_under_5s_zero_user_turns",
        )
    is_interrupted = (
        disposition in {"user_hangup", "no_answer", "failed", "busy", "abandoned"}
        and minimal_progress
        and not has_content
        and not is_short_no_turn_noise
    )

    # Καταγραφή στο call log για dashboard
    # Εμπλουτισμός: οι extraction variables του Dograh είναι συχνά κενά στο τέλος
    # σύντομων κλήσεων. Αλλά το conversation state (που γέμισε κατά τη διάρκεια
    # της κλήσης μέσω pre_route_intent/suggestions/book/send_message) τα έχει ΟΛΑ.
    _state = {}
    if phone:
        _state = get_conversation_state(phone_number=phone) or {}
    elif payload.call_id:
        _state = get_conversation_state(call_id=payload.call_id) or {}

    # Κατηγοριοποίηση τύπου κλήσης
    _intent = _webhook_reason or (_state.get("current_intent") or "").strip().lower()
    _booking_uid = str(payload.booking_uid or _state.get("booking_uid") or "").strip()
    _booking_status = str(payload.booking_status or _state.get("booking_status") or "").strip().lower()
    _booking_done = bool(_booking_uid) or _booking_status in {"confirmed", "success"} or _state.get("booking_completed", False)
    _message_sent = _state.get("message_sent", False)
    _cancel_done = _state.get("cancel_completed", False)

    _has_message_content = bool(
        (payload.message_clean or payload.message
         or _state.get("current_message") or "").strip()
    )
    _webhook_start_message_recovered = False
    if (
        disposition in {"user_hangup", "abandoned"}
        and (minimal_progress or _intent in {"message", "urgent"})
        and _has_message_content
        and not _message_sent
        and not _booking_done
        and not _cancel_done
        and phone
    ):
        _webhook_start_message_recovered = await _recover_start_only_message_from_webhook(
            payload,
            phone=phone,
            message_text=_webhook_message or str(_state.get("current_message") or ""),
        )
        if _webhook_start_message_recovered:
            _message_sent = True
            _intent = "message"
            _state = get_conversation_state(phone_number=phone, call_id=payload.call_id) or _state

    if _booking_done:
        _call_type = "booking"
    elif _cancel_done:
        _call_type = "cancellation"
    elif _intent in ("message", "urgent") or _message_sent:
        _call_type = "message"
    elif _intent == "appointment":
        _call_type = "appointment"
    elif _has_message_content:
        _call_type = "message"
    elif is_interrupted:
        _call_type = "interrupted"
    else:
        _call_type = "call"


    _duration_seconds = _dograh_duration_seconds(payload, data if isinstance(data, dict) else {})
    _user_turns = _dograh_user_turn_count(payload, data if isinstance(data, dict) else {})
    from call_benchmark import evaluate_call_outcome

    _outcome_evaluation = evaluate_call_outcome(
        state=_state,
        payload={
            "reason_for_call": _webhook_reason or _state.get("current_intent") or "",
            "booking_status": payload.booking_status or _state.get("booking_status") or "",
            "booking_uid": _booking_uid,
            "calendar_event_id": payload.calendar_event_id or _state.get("calendar_event_id") or "",
            "conversation_done": payload.conversation_done or _state.get("conversation_done") or False,
            "current_stage": payload.current_stage or _state.get("current_stage") or "",
            "next_action": payload.next_action or _state.get("next_action") or "",
        },
        nodes_visited=nodes,
    )
    _call_log_event = {
        "type": _call_type,
        "call_id": payload.call_id or "",
        "phone": phone or _state.get("phone_number") or _state.get("caller_number") or "",
        "first_name": payload.first_name or _state.get("first_name") or "",
        "last_name": payload.last_name or _state.get("last_name") or "",
        "reason": _webhook_reason or _state.get("current_intent") or "",
        "message": _webhook_message or _state.get("current_message") or "",
        "clinic": payload.clinic or _state.get("current_clinic") or "",
        "priority": payload.priority_level or _state.get("priority_level") or "C",
        "booking_uid": _booking_uid,
        "is_urgent": bool(payload.is_urgent) or (_state.get("priority_level") == "A"),
        "booking_recovery": bool(booking_recovery),
        "disposition": disposition or "",
        "incomplete": is_interrupted,
        "noise_ignored": is_short_no_turn_noise,
        "webhook_start_message_recovered": _webhook_start_message_recovered,
        "duration_seconds": _duration_seconds,
        "user_turns": _user_turns,
        "time_preference": _state.get("current_time_preference") or "",
        "day_preference": _state.get("current_day_preference") or "",
        "booking_start_iso": _state.get("booked_start_iso") or _state.get("current_booking_start_iso") or "",
        "booking_clinic": _state.get("booked_clinic") or "",
        "caller_wants_callback": _state.get("caller_wants_callback", False),
        "intended_outcome": _outcome_evaluation["intended_outcome"],
        "actual_outcome": _outcome_evaluation["actual_outcome"],
        "intent_completed": _outcome_evaluation["intent_completed"],
        "failure_point": _outcome_evaluation["failure_point"],
        "flow_quality": _outcome_evaluation["flow_quality"],
        # Dograh does not currently include per-turn delays in every webhook.
        # Keep this explicit instead of treating missing latency as failure.
        "latency_quality": _outcome_evaluation["latency_quality"],
    }
    _log_call_event(_call_log_event)
    try:
        from call_benchmark import record_call_benchmark

        _benchmark_row = record_call_benchmark(
            state=_state,
            payload={
                **_call_log_event,
                "reason_for_call": _webhook_reason or _state.get("current_intent") or "",
                "message_clean": _webhook_message or _state.get("current_message") or "",
                "current_stage": payload.current_stage or _state.get("current_stage") or "",
                "next_action": payload.next_action or _state.get("next_action") or "",
                "booking_status": payload.booking_status or _state.get("booking_status") or "",
                "booking_uid": _booking_uid,
                "calendar_event_id": payload.calendar_event_id or _state.get("calendar_event_id") or "",
                "conversation_done": payload.conversation_done or _state.get("conversation_done") or False,
                "call_id": payload.call_id or "",
            },
            phone=phone or _state.get("phone_number") or _state.get("caller_number") or "",
            call_id=payload.call_id or "",
            nodes_visited=nodes,
            duration_seconds=_duration_seconds,
            user_turns=_user_turns,
            tool_calls=len(nodes) if isinstance(nodes, list) else None,
            disposition=disposition or "",
        )
        print(
            "CALL BENCHMARK: "
            f"intent={_benchmark_row.get('intent')} "
            f"success={_benchmark_row.get('success_by_intent')} "
            f"failure_point={_benchmark_row.get('failure_point')} "
            f"duration={_benchmark_row.get('duration_seconds')}"
        )
    except Exception as exc:
        print(f"CALL BENCHMARK: hook failed ({exc}).")
    try:
        from call_audit import record_call_audit

        _audit_row = record_call_audit(
            state=_state,
            payload={
                **_call_log_event,
                "caller_number": phone or _state.get("caller_number") or "",
                "phone_number": phone or _state.get("phone_number") or "",
                "reason_for_call": _webhook_reason or _state.get("current_intent") or "",
                "message_clean": _webhook_message or _state.get("current_message") or "",
                "current_stage": payload.current_stage or _state.get("current_stage") or "",
                "next_action": payload.next_action or _state.get("next_action") or "",
                "booking_status": payload.booking_status or _state.get("booking_status") or "",
                "booking_uid": _booking_uid,
                "calendar_event_id": payload.calendar_event_id or _state.get("calendar_event_id") or "",
                "conversation_done": payload.conversation_done or _state.get("conversation_done") or False,
                "call_id": payload.call_id or "",
            },
            raw_data=data if isinstance(data, dict) else {},
            raw_text=raw_text,
            phone=phone or _state.get("phone_number") or _state.get("caller_number") or "",
            call_id=payload.call_id or "",
            disposition=disposition or "",
        )
        print(
            "CALL AUDIT: "
            f"success={_audit_row.get('success')} "
            f"failure_point={_audit_row.get('failure_point')} "
            f"reasoning_total_ms={_audit_row.get('reasoning_total_ms')}"
        )
        if (
            payload.call_id
            and _audit_row.get("reasoning_total_ms") is None
            and DOGRAH_API_KEY
            and DOGRAH_CALL_DETAIL_URL_TEMPLATE
            and payload.workflow_id
            and (payload.workflow_run_id or payload.run_id)
        ):
            asyncio.create_task(
                _enrich_call_audit_from_dograh(
                    payload.call_id,
                    payload.workflow_id,
                    payload.workflow_run_id or payload.run_id,
                )
            )
    except Exception as exc:
        print(f"CALL AUDIT: hook failed ({exc}).")
    cleared = clear_conversation_state(
        phone_number=phone,
        call_id=payload.call_id,
    )
    if cleared:
        print(
            "CONVERSATION STATE CLEARED: "
            f"phone={phone or ''} call_id={payload.call_id or ''}"
        )
    # send_email_to_doctor(subject="Νέα κλήση από Dograh", body=_email_body)
    return {"status": "ok", "booking_recovery": booking_recovery}


# ============================================================

# ============================================================

# ============================================================
# Call Log (in-memory ring buffer για dashboard)
# ============================================================

from collections import deque
import threading

_CALL_LOG: deque = deque(maxlen=200)  # τελευταίες 200 κλήσεις
_CALL_LOG_LOCK = threading.Lock()
_CALL_LOG_FILE = "/data/call_log.json"

_CLOSING_UTTERANCES = {
    "ευχαριστω", "ευχαριστω πολυ", "μαλιστα", "οκ", "okay", "ok",
    "εντασσει", "ενταξει", "να ειστε καλα", "να στε καλα", "γεια σας",
    "παρακαλω", "ωραια", "τελεια", "καλημερα", "καληνυχτα",
    "καλο βραδυ", "γεια", "αντιο", "φιλια", "ευχαριστουμε",
}


def _is_closing_utterance(text: str) -> bool:
    from appointment_utils import normalize_guard_text
    normalized = normalize_guard_text(text)
    return normalized in _CLOSING_UTTERANCES or len(normalized) <= 3


def _choose_display_message(existing_msg: str, incoming_msg: str) -> str:
    incoming = (incoming_msg or "").strip()
    existing = (existing_msg or "").strip()
    if not incoming:
        return existing
    if _is_closing_utterance(incoming):
        return existing
    return incoming


def _sanitize_display_message(event: dict) -> str:
    candidate = (
        event.get("display_message")
        or event.get("message_clean")
        or event.get("reason_for_call")
        or event.get("reason")
        or event.get("message")
        or ""
    )
    result = _choose_display_message("", candidate)
    if not result and (
        str(event.get("priority") or event.get("priority_level") or "").upper() == "A"
        or event.get("is_urgent")
    ):
        result = "Επείγον αίτημα ασθενούς"
    return result


def _persist_call_log() -> None:
    """
    Αποθηκεύει τις τελευταίες 200 κλήσεις στο disk (μόνο σήμερα + χθες).

    ΣΗΜΑΝΤΙΚΟ: το πραγματικό write γίνεται σε background thread (μέσω
    run_in_executor), όχι synchronously μέσα στο request. Πριν, κάθε κλήση
    αυτής της συνάρτησης (json.dumps + file write) μπλόκαρε το critical
    path του αιτήματος (pre_route_intent/send_message/webhook) για ~5-15ms.

    Επίσης: αυτή η συνάρτηση καλείται από το _log_call_event() ΕΝΩ ακόμα
    κρατάμε το _CALL_LOG_LOCK (στα merge paths). Για να αποφύγουμε deadlock
    (το threading.Lock δεν είναι reentrant), παίρνουμε το snapshot ΕΔΩ —
    χωρίς να ξανα-κλειδώσουμε, απλά διαβάζοντας το ήδη-κλειδωμένο _CALL_LOG
    (ο caller το κρατά ήδη) — και περνάμε το snapshot στο sync write, που
    ποτέ δεν αγγίζει το lock.
    """
    calls_snapshot = list(_CALL_LOG)
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _persist_call_log_sync, calls_snapshot)
    except RuntimeError:
        # Καλείται εκτός event loop (π.χ. σε script/CLI context) — fallback σε sync write.
        _persist_call_log_sync(calls_snapshot)


def _persist_call_log_sync(calls_snapshot: list) -> None:
    import json as _json, datetime as _dt
    from pathlib import Path
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Asia/Nicosia")
    today = _dt.datetime.now(tz).date()
    yesterday = today - _dt.timedelta(days=1)
    done_state = _load_todo_done()
    to_save = []
    for call in calls_snapshot:
        try:
            dt = _dt.datetime.fromisoformat(
                call.get("logged_at", "").replace("Z", "+00:00")
            ).astimezone(tz)
            if dt.date() in (today, yesterday):
                to_save.append(call)
            else:
                tid = "td_" + (call.get("call_id") or ((call.get("phone") or "") + "@" + (call.get("logged_at") or "")[:16]))
                if tid not in done_state:
                    to_save.append(call)
        except Exception:
            pass

    try:
        Path(_CALL_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(_CALL_LOG_FILE).write_text(_json.dumps(to_save, ensure_ascii=False))
    except Exception as exc:
        print(f"[CALL_LOG] persist error: {exc}")


def _load_call_log() -> None:
    """Φορτώνει το call log από disk κατά την εκκίνηση."""
    import json as _json
    from pathlib import Path
    try:
        p = Path(_CALL_LOG_FILE)
        if not p.exists():
            return
        data = _json.loads(p.read_text())
        if isinstance(data, list):
            with _CALL_LOG_LOCK:
                for item in reversed(data):
                    _CALL_LOG.appendleft(item)
            print(f"[CALL_LOG] loaded {len(data)} entries from disk")
    except Exception as exc:
        print(f"[CALL_LOG] load error: {exc}")


def _log_call_event(event: dict) -> None:
    """
    Προσθέτει ή ενημερώνει event στο call log (thread-safe) και αποθηκεύει στο disk.
    Merge κατά προτεραιότητα: (1) πραγματικό call_id, (2) phone+timestamp ±5 λεπτά.
    """
    import datetime as _dt
    event.setdefault("logged_at", _dt.datetime.now(tz=_dt.timezone.utc).isoformat())

    def _usable_call_id(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        # Dograh/local tests can send placeholder IDs. If we merge on those,
        # unrelated calls collapse into one dashboard row and the list appears stale.
        if raw.lower() in {
            "dograh-default-call-id",
            "default-call-id",
            "default",
            "unknown",
            "none",
            "null",
        }:
            return ""
        return raw

    call_id = _usable_call_id(event.get("call_id"))
    event["call_id"] = call_id
    phone = (event.get("phone") or "").strip()

    def _merge_into(existing):
        orig_logged_at = existing.get("logged_at")
        for k, v in event.items():
            if k in ("incomplete", "disposition"):
                existing[k] = v
            elif k in ("message", "reason", "display_message") and _is_closing_utterance(str(v or "")):
                if existing.get(k):
                    continue
            elif v not in (None, "", "C", False):
                existing[k] = v
        existing["display_message"] = _choose_display_message(
            existing.get("display_message") or existing.get("message_clean") or existing.get("reason") or "",
            event.get("message_clean") or event.get("reason") or event.get("message") or "",
        )
        if not existing["display_message"]:
            existing["display_message"] = _sanitize_display_message({**existing, **event})
        existing["logged_at"] = orig_logged_at
        existing.pop("placeholder", None)

    def _within_minutes(iso1, iso2, minutes=5):
        try:
            t1 = _dt.datetime.fromisoformat(iso1.replace("Z", "+00:00"))
            t2 = _dt.datetime.fromisoformat(iso2.replace("Z", "+00:00"))
            return abs((t1 - t2).total_seconds()) < minutes * 60
        except Exception:
            return False

    # ΣΗΜΕΙΩΣΗ: το _persist_call_log() καλείται είτε μέσα στο with block
    # (στα merge paths, ενώ κρατάμε ακόμα το lock) είτε μετά (νέο entry).
    # Είναι ασφαλές και στις δύο περιπτώσεις: η _persist_call_log() παίρνει
    # το snapshot με ένα plain list(_CALL_LOG) ΧΩΡΙΣ να ξανακλειδώνει (βλ.
    # docstring της), άρα δεν υπάρχει πια κίνδυνος deadlock.
    with _CALL_LOG_LOCK:
        # 1. Merge by call_id
        if call_id:
            for existing in _CALL_LOG:
                if _usable_call_id(existing.get("call_id")) == call_id:
                    _merge_into(existing)
                    _persist_call_log()
                    return
        # 2. Merge by phone + timestamp ±5 min (avoid duplicates from webhook + action)
        #    ΑΣΦΑΛΕΙΑ: αν τα ονόματα είναι ΚΑΙ τα δύο γνωστά ΚΑΙ διαφορετικά,
        #    ΔΕΝ κάνουμε merge — πιθανότατα είναι δύο διαφορετικοί ασθενείς
        #    που καλούν από κοινό (π.χ. οικογενειακό) τηλέφωνο εντός 5 λεπτών,
        #    όχι το ίδιο call_id φτάνοντας από δύο σημεία (webhook + action).
        if phone:
            event_time = event.get("logged_at", "")
            event_first = (event.get("first_name") or "").strip().lower()
            event_last = (event.get("last_name") or "").strip().lower()
            for existing in _CALL_LOG:
                ex_phone = (existing.get("phone") or "").strip()
                if ex_phone != phone or not _within_minutes(event_time, existing.get("logged_at", ""), 5):
                    continue
                existing_call_id = _usable_call_id(existing.get("call_id"))
                if call_id and existing_call_id and existing_call_id != call_id:
                    continue
                ex_first = (existing.get("first_name") or "").strip().lower()
                ex_last = (existing.get("last_name") or "").strip().lower()
                names_conflict = (
                    event_first and ex_first and event_first != ex_first
                ) or (
                    event_last and ex_last and event_last != ex_last
                )
                if names_conflict:
                    continue  # διαφορετικοί ασθενείς, ίδιο τηλέφωνο — ΟΧΙ merge
                _merge_into(existing)
                # Αν τώρα μάθαμε call_id, αποθηκεύουμέ το
                if call_id and not (existing.get("call_id") or "").strip():
                    existing["call_id"] = call_id
                _persist_call_log()
                return
        event["display_message"] = _sanitize_display_message(event)
        _CALL_LOG.appendleft(event)
    _persist_call_log()


# Φόρτωσε call log από disk κατά την εκκίνηση
_load_call_log()



# ============================================================
# Still Waiting (latency fallback)
# ============================================================

# ============================================================
# LiveKit Call Complete (audit + benchmark for LiveKit calls)
# ============================================================

@app.post("/livekit/call_complete", dependencies=[Depends(require_action_token)])
async def livekit_call_complete(request: Request):
    """
    End-of-call summary from the LiveKit agent.
    Records audit + benchmark so LiveKit calls appear in the same
    dashboard/quality pipeline as Dograh calls.
    """
    try:
        data = await request.json()
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}

    phone = str(
        data.get("caller_number")
        or data.get("phone_number")
        or ""
    ).strip()
    call_id = str(data.get("call_id") or "").strip()

    state = {
        "current_intent": data.get("reason_for_call") or "",
        "current_message": data.get("message") or "",
        "current_message_category": data.get("message_category") or "",
        "current_clinic": data.get("clinic") or "",
        "current_stage": data.get("current_stage") or "",
        "next_action": data.get("next_action") or "",
        "caller_number": phone,
        "phone_number": phone,
        "call_id": call_id,
        "message_sent": bool(data.get("message_sent")),
        "booking_completed": bool(data.get("booking_completed")),
        "cancel_completed": bool(data.get("cancel_completed")),
        "urgent_handoff_sent": bool(data.get("urgent_handoff_sent")),
        "conversation_done": bool(data.get("conversation_done")),
        "priority_level": data.get("priority_level") or "C",
    }

    payload = {
        **state,
        "reason_for_call": data.get("reason_for_call") or "",
        "message_clean": data.get("message_clean") or data.get("message") or "",
        "booking_status": data.get("booking_status") or "",
        "booking_uid": data.get("booking_uid") or "",
        "calendar_event_id": data.get("calendar_event_id") or "",
        "duration_seconds": data.get("duration_seconds"),
        "user_turns": data.get("user_turns"),
        "first_name": data.get("first_name") or "",
        "last_name": data.get("last_name") or "",
        "source": "livekit",
    }

    disposition = str(data.get("disposition") or "").strip()
    nodes_visited = data.get("nodes_visited") or ["livekit"]
    duration_seconds = data.get("duration_seconds")
    user_turns = data.get("user_turns")
    tool_calls = data.get("tool_calls")

    _call_event = {
        "type": data.get("outcome") or "call",
        "call_id": call_id,
        "phone": phone,
        "first_name": data.get("first_name") or "",
        "last_name": data.get("last_name") or "",
        "reason": data.get("reason_for_call") or "",
        "reason_for_call": data.get("reason_for_call") or "",
        "message": data.get("message_clean") or data.get("message") or "",
        "message_clean": data.get("message_clean") or "",
        "clinic": data.get("clinic") or "",
        "priority": data.get("priority_level") or "C",
        "is_urgent": str(data.get("priority_level") or "").upper() == "A",
        "intent_source": "livekit",
        "source": "livekit",
    }
    _call_event["display_message"] = _sanitize_display_message(_call_event)
    _log_call_event(_call_event)

    audit_row = {}
    benchmark_row = {}
    try:
        from call_audit import record_call_audit
        audit_row = record_call_audit(
            state=state,
            payload=payload,
            raw_data={"source": "livekit", "nodes_visited": nodes_visited},
            raw_text="",
            phone=phone,
            call_id=call_id,
            disposition=disposition,
        )
        print(
            f"LIVEKIT AUDIT: success={audit_row.get('success')} "
            f"failure_point={audit_row.get('failure_point')}"
        )
    except Exception as exc:
        print(f"LIVEKIT AUDIT: failed ({exc})")

    try:
        from call_benchmark import record_call_benchmark
        benchmark_row = record_call_benchmark(
            state=state,
            payload=payload,
            phone=phone,
            call_id=call_id,
            nodes_visited=nodes_visited,
            duration_seconds=float(duration_seconds) if duration_seconds is not None else None,
            user_turns=int(user_turns) if user_turns is not None else None,
            tool_calls=int(tool_calls) if tool_calls is not None else None,
            disposition=disposition,
        )
        print(
            f"LIVEKIT BENCHMARK: intent_completed={benchmark_row.get('intent_completed')} "
            f"flow_quality={benchmark_row.get('flow_quality')}"
        )
    except Exception as exc:
        print(f"LIVEKIT BENCHMARK: failed ({exc})")

    return {
        "success": True,
        "source": "livekit",
        "audit_recorded": bool(audit_row and not audit_row.get("error")),
        "benchmark_recorded": bool(benchmark_row and not benchmark_row.get("error")),
    }


@app.post("/actions/still_waiting", dependencies=[Depends(require_action_token)])
async def still_waiting_action():
    """
    Ο agent το καλεί αν περιμένει >6s για αποτέλεσμα tool.
    Επιστρέφει spoken_response για να κρατήσει ζωντανή τη συνομιλία.
    """
    import random
    messages = [
        "Ένα λεπτό ακόμα, το σύστημα ανταποκρίνεται.",
        "Παρακαλώ περιμένετε λίγο ακόμα.",
        "Επεξεργάζομαι το αίτημά σας, ένα λεπτό.",
    ]
    return {"spoken_response": random.choice(messages), "status": "waiting"}


# Health Check
# ============================================================

@app.get("/version")
async def version_check():
    """
    Ελαφρύ endpoint (καμία εξωτερική κλήση) για να επιβεβαιωθεί ΑΜΕΣΑ ποια
    έκδοση του main.py τρέχει live στο Render — χωρίς να χρειάζεται να
    θυμάται κανείς ή να μαντεύει. Πρόσθεσε μια νέα γραμμή εδώ σε κάθε
    σημαντικό deploy ώστε να μένει ιστορικό.
    """
    return {
        "build_marker": "2026-06-21-sync-mutex-and-blockout-filter-v3",
        "has_inprocess_sync": True,
        "has_argv_safe_setmore_sync": True,
        "has_persistent_db_connection": True,
        "has_dashboard_action_queue": True,
        "has_weekly_triage": True,
        "has_yesterday_completed_filter": True,
        "has_onnx_cy_intent": True,
        "has_rule_gaps_endpoint": True,
        "has_per_intent_threshold": True,
        "has_prolia_message_category_fix": True,
        "has_async_setmore_sync_fix": True,
        "has_sync_mutex": True,
        "has_blocked_out_service_filter": True,
        "has_dograh_noise_alert_filter": True,
        "has_backend_tool_retry_cap": True,
    }


@app.get("/health")
async def health_check():
    """
    Ελέγχει τη σύνδεση με Cal.com και Setmore.
    Επιστρέφει 200 αν όλα εντάξει, 503 αν κάτι αποτυχαίνει.
    Στέλνει email alert αν κάποια σύνδεση αποτύχει.
    """
    import time
    results: dict[str, Any] = {}
    failures: list[str] = []

    # Cal.com check
    try:
        t0 = time.monotonic()
        from cal_config import CAL_BASE_URL
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{CAL_BASE_URL}/event-types",
                headers={
                    "Authorization": f"Bearer {os.getenv('CAL_API_KEY', '')}",
                    "cal-api-version": "2024-06-14",
                },
            )
        latency_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code == 200:
            results["cal_com"] = {"status": "ok", "latency_ms": latency_ms}
        else:
            results["cal_com"] = {"status": "error", "http": resp.status_code}
            failures.append(f"Cal.com HTTP {resp.status_code}")
    except Exception as exc:
        results["cal_com"] = {"status": "error", "error": str(exc)[:100]}
        failures.append(f"Cal.com: {str(exc)[:60]}")

    # Setmore check
    try:
        t0 = time.monotonic()
        from setmore_client import get_setmore_headers
        headers = await get_setmore_headers()
        latency_ms = int((time.monotonic() - t0) * 1000)
        if headers:
            results["setmore"] = {"status": "ok", "latency_ms": latency_ms}
        else:
            results["setmore"] = {"status": "error", "error": "no token"}
            failures.append("Setmore: failed to get token")
    except Exception as exc:
        results["setmore"] = {"status": "error", "error": str(exc)[:100]}
        failures.append(f"Setmore: {str(exc)[:60]}")

    # Brevo / email check (static — just verify env var present)
    brevo_key = os.getenv("BREVO_API_KEY", "")
    results["brevo"] = {"status": "ok" if brevo_key else "missing_key"}

    # Zadarma check
    results["zadarma"] = {
        "configured": _ZADARMA_CONFIGURED,
        "callback_enabled": ZADARMA_ENABLE_CALLBACK,
        "last_error": ZADARMA_LAST_ERROR or None,
    }

    # Αν υπάρχουν failures, στείλε alert email
    if failures:
        try:
            await asyncio.to_thread(
                send_slack_notification,
                subject=f"⚠️ Health Check Failure: {chr(44).join(failures[:2])}",
                text_content="\n".join(f"- {f}" for f in failures),
                emoji=":warning:",
                urgent=True,
            )
            await asyncio.to_thread(
                send_brevo_email,
                subject=f"⚠️ Health Check Failure: {', '.join(failures[:2])}",
                text_content=(
                    f"Ο health check απέτυχε στις {__import__('datetime').datetime.now().isoformat()}\n\n"
                    + "\n".join(f"- {f}" for f in failures)
                    + "\n\nΛεπτομέρειες:\n"
                    + str(results)
                ),
            )
        except Exception:
            pass
        from fastapi.responses import JSONResponse, HTMLResponse
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "failures": failures, "details": results},
        )

    return {"status": "ok", "details": results}


# ============================================================
# Dashboard API + UI
# ============================================================

@app.get("/dashboard/calls")
@app.get("/dashboard/calls")
async def dashboard_calls_api(key: str = ""):
    """Επιστρέφει κλήσεις σήμερα + χθες, ΣΥΝΝ παλαιότερες ΜΗ τικαρισμένες."""
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    import datetime as _dt
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Asia/Nicosia")
    today = _dt.datetime.now(tz).date()
    yesterday = today - _dt.timedelta(days=1)
    done_state = _load_todo_done()

    with _CALL_LOG_LOCK:
        all_calls = list(_CALL_LOG)

    filtered = []
    for call in all_calls:
        logged_at = call.get("logged_at", "")
        if not logged_at:
            continue
        try:
            dt = _dt.datetime.fromisoformat(logged_at.replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        call = dict(call)
        call["is_today"] = (dt.date() == today)
        if dt.date() in (today, yesterday):
            filtered.append(call)
        else:
            tid = "td_" + (call.get("call_id") or ((call.get("phone") or "") + "@" + (call.get("logged_at") or "")[:16]))
            if tid not in done_state:
                filtered.append(call)

    return {"calls": filtered, "total": len(filtered)}


# ── Todo completion state ──────────────────────────────────
_TODO_FILE = "/data/todo_done.json"


def _load_todo_done() -> dict:
    from pathlib import Path
    try:
        p = Path(_TODO_FILE)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {}


def _save_todo_done(state: dict) -> None:
    from pathlib import Path
    try:
        Path(_TODO_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(_TODO_FILE).write_text(json.dumps(state, ensure_ascii=False))
    except Exception as exc:
        print(f"[TODO] save error: {exc}")


@app.post("/dashboard/todos/toggle")
async def toggle_todo(request: Request, key: str = ""):
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json()
    todo_id = str(body.get("id", ""))
    if not todo_id:
        return {"error": "missing id"}
    state = _load_todo_done()
    if todo_id in state:
        del state[todo_id]
        done = False
    else:
        import datetime as _dt
        state[todo_id] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        done = True
    _save_todo_done(state)
    return {"id": todo_id, "done": done}


@app.get("/dashboard/todos/state")
async def get_todo_state(key: str = ""):
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return _load_todo_done()


_SCHEDULE_CACHE: dict[str, tuple[float, dict]] = {}
_SCHEDULE_CACHE_TTL_SECONDS = 3600  # 1 ώρα — τα ραντεβού της ημέρας δεν αλλάζουν συχνά


@app.get("/dashboard/schedule")
async def dashboard_schedule(key: str = "", target_date: str = ""):
    """
    Ραντεβού ημέρας από Cal.com — default σήμερα, accept target_date=YYYY-MM-DD.

    ΣΗΜΑΝΤΙΚΟ: cache 60s ανά target_date. Το dashboard κάνει polling κάθε
    30s (και επιπλέον σε κάθε checkbox click), χωρίς αυτό το cache κάθε ένα
    από αυτά θα έκανε ζωντανό fetch στο Cal.com API — πράγμα που έχει ήδη
    προκαλέσει timeouts στα live action endpoints που χρησιμοποιεί το Dograh
    (ίδιος server, shared resources). Το cache εξασφαλίζει ότι ανεξάρτητα από
    πόσα browser tabs/clicks γίνονται, το πραγματικό αίτημα στο Cal.com τρέχει
    το πολύ 1 φορά/λεπτό ανά ημερομηνία.
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    import datetime as _dt
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Asia/Nicosia")
    now_local = _dt.datetime.now(tz)

    if target_date:
        try:
            the_date = _dt.date.fromisoformat(target_date)
        except ValueError:
            the_date = now_local.date()
    else:
        the_date = now_local.date()

    cache_key = the_date.isoformat()
    cached = _SCHEDULE_CACHE.get(cache_key)
    now_monotonic = _time_module.monotonic()
    if cached and (now_monotonic - cached[0]) < _SCHEDULE_CACHE_TTL_SECONDS:
        # Επιστρέφουμε το cached αποτέλεσμα, αλλά ξαναυπολογίζουμε is_past/is_today
        # ζωντανά (είναι τοπικός υπολογισμός, δεν χρειάζεται Cal.com call).
        # ΣΗΜΑΝΤΙΚΟ: deepcopy (όχι shallow dict()) — το payload περιέχει
        # nested λίστα από dicts (appointments). Ένα shallow dict(cached[1])
        # θα μοιραζόταν τα ΙΔΙΑ appointment dicts με το cache, και το
        # `appt["is_past"] = ...` παρακάτω θα μετέβαλε το ίδιο το cached
        # αντικείμενο αντί για το αντίγραφο.
        import copy as _copy
        cached_payload = _copy.deepcopy(cached[1])
        for appt in cached_payload.get("appointments", []):
            try:
                bk_dt = _dt.datetime.fromisoformat(appt["start"].replace("Z", "+00:00")).astimezone(tz)
                appt["is_past"] = bk_dt < now_local
            except Exception:
                pass
        cached_payload["is_today"] = (the_date == now_local.date())
        cached_payload["current_time"] = now_local.strftime("%H:%M") if cached_payload["is_today"] else ""
        cached_payload["_cache_age_s"] = round(now_monotonic - cached[0], 1)
        return cached_payload

    appointments = []
    cal = CalClient()
    try:
        day_start = the_date.isoformat() + "T00:00:00Z"
        day_end = (the_date + _dt.timedelta(days=1)).isoformat() + "T00:00:00Z"
        resp = await cal.get_bookings(
            status="upcoming",
            take=50,
            sort_start="asc",
            after_start=day_start,
            before_end=day_end,
        )
        bookings_data = resp.get("data", resp)
        bookings_list = bookings_data if isinstance(bookings_data, list) else bookings_data.get("bookings", [])
        for bk in (bookings_list if isinstance(bookings_list, list) else []):
            start_str = bk.get("start", "")
            if not start_str:
                continue
            try:
                bk_dt = _dt.datetime.fromisoformat(start_str.replace("Z", "+00:00")).astimezone(tz)
            except Exception:
                continue
            # Δίχτυ ασφαλείας: το Cal.com API μπορεί μερικές φορές να αγνοήσει τα
            # afterStart/beforeEnd filters — ξαναφιλτράρουμε εδώ σε κάθε περίπτωση.
            if bk_dt.date() != the_date:
                continue
            attendees = bk.get("attendees", [])
            name = attendees[0].get("name", "—") if attendees else "—"
            phone = attendees[0].get("phoneNumber", "") if attendees else ""
            dur = bk.get("duration", 30)
            loc = (bk.get("location") or bk.get("meetingUrl") or "").lower()
            clinic = "evrychou" if "evrychou" in loc else ("limassol" if "limassol" in loc else "")
            appointments.append({
                "clinic": clinic,
                "start": start_str,
                "time": bk_dt.strftime("%H:%M"),
                "name": name,
                "phone": phone,
                "duration": dur,
                "is_past": bk_dt < now_local,
            })
    except Exception as exc:
        print(f"[SCHEDULE] bookings error: {exc}")

    appointments.sort(key=lambda a: a.get("start", ""))
    weekdays = ["Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"]
    # Clinic by day of week
    wd = the_date.weekday()
    if wd in (0, 2):    # Mon, Wed → evrychou all day
        main_clinic = "evrychou"
    elif wd in (1, 3):  # Tue, Thu → limassol all day
        main_clinic = "limassol"
    elif wd == 4:        # Fri → evrychou morning + limassol afternoon
        main_clinic = "evrychou + limassol"
    else:
        main_clinic = ""

    # ── Schedule intelligence: κενά + patient context ──────────────────
    # Κενά (>30 λεπτά μεταξύ διαδοχικών ραντεβού) — χρήσιμο για walk-ins.
    GAP_THRESHOLD_MINUTES = 30
    gaps = []
    for i in range(len(appointments) - 1):
        try:
            end_a = datetime.fromisoformat(appointments[i]["start"].replace("Z", "+00:00")).astimezone(tz)
            end_a = end_a + timedelta(minutes=appointments[i].get("duration", 30))
            start_b = datetime.fromisoformat(appointments[i + 1]["start"].replace("Z", "+00:00")).astimezone(tz)
            gap_minutes = (start_b - end_a).total_seconds() / 60
            if gap_minutes >= GAP_THRESHOLD_MINUTES:
                gaps.append({
                    "from": end_a.strftime("%H:%M"),
                    "to": start_b.strftime("%H:%M"),
                    "duration_minutes": round(gap_minutes),
                })
        except Exception:
            continue

    # Patient context: αν ο ασθενής έχει πρόσφατη κλήση/callback στο _CALL_LOG
    # (τελευταίες 48 ώρες), επισημαίνεται στο ραντεβού του.
    with _CALL_LOG_LOCK:
        recent_calls_by_phone: dict[str, dict] = {}
        for c in _CALL_LOG:
            p = (c.get("phone") or "").strip()
            if p and p not in recent_calls_by_phone:
                recent_calls_by_phone[p] = c

    for appt in appointments:
        ph = (appt.get("phone") or "").strip()
        recent = recent_calls_by_phone.get(ph)
        if recent:
            appt["recent_call_context"] = {
                "wanted_callback": bool(recent.get("caller_wants_callback")),
                "was_booking_recovery": bool(recent.get("booking_recovery")),
                "last_message": recent.get("display_message") or recent.get("message_clean") or recent.get("message") or recent.get("reason") or "",
            }
    # ─────────────────────────────────────────────────────────────────

    is_today = (the_date == now_local.date())
    payload = {
        "date": the_date.isoformat(),
        "weekday": weekdays[wd],
        "clinic": main_clinic,
        "appointments": appointments,
        "total": len(appointments),
        "is_today": is_today,
        "current_time": now_local.strftime("%H:%M") if is_today else "",
        "gaps": gaps,
    }
    _SCHEDULE_CACHE[cache_key] = (_time_module.monotonic(), payload)
    return payload


@app.get("/dashboard/health")
async def dashboard_health(key: str = ""):
    """
    Uptime / latency Cal.com + Setmore τελευταίων 24 ωρών, με βάση το
    background polling (_external_health_polling_loop) — ΟΧΙ ζωντανό call
    εδώ, μόνο ανάγνωση του ήδη-συλλεγμένου ιστορικού.
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    with _HEALTH_HISTORY_LOCK:
        history = list(_HEALTH_HISTORY)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    recent = []
    for entry in history:
        try:
            ts = datetime.fromisoformat(entry["checked_at"].replace("Z", "+00:00"))
            if ts >= cutoff:
                recent.append(entry)
        except Exception:
            continue

    def _uptime_pct(field: str) -> float | None:
        vals = [e.get(field) for e in recent if field in e]
        if not vals:
            return None
        return round(100.0 * sum(1 for v in vals if v) / len(vals), 1)

    def _avg_latency(field: str) -> int | None:
        vals = [e.get(field) for e in recent if e.get(field) is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals))

    last = recent[-1] if recent else None
    return {
        "checks_last_24h": len(recent),
        "cal_com": {
            "uptime_pct": _uptime_pct("cal_com_ok"),
            "avg_latency_ms": _avg_latency("cal_com_latency_ms"),
            "currently_ok": last.get("cal_com_ok") if last else None,
        },
        "setmore": {
            "uptime_pct": _uptime_pct("setmore_ok"),
            "avg_latency_ms": _avg_latency("setmore_latency_ms"),
            "currently_ok": last.get("setmore_ok") if last else None,
        },
        "last_checked_at": last.get("checked_at") if last else None,
    }


@app.get("/dashboard/insights")
async def dashboard_insights(key: str = ""):
    """
    Σύνοψη ημέρας σε φυσική γλώσσα + ωριαία κατανομή κλήσεων + completion
    rate + επαναλαμβανόμενοι καλούντες. Βασίζεται στο ήδη-υπάρχον in-memory
    _CALL_LOG (σήμερα) και στο call_metrics CSV (ιστορικό completion rate),
    χωρίς κανένα νέο ζωντανό external API call.
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Asia/Nicosia")
    now_local = datetime.now(tz)
    today = now_local.date()

    with _CALL_LOG_LOCK:
        calls_today = []
        for c in _CALL_LOG:
            try:
                dt = datetime.fromisoformat(c.get("logged_at", "").replace("Z", "+00:00")).astimezone(tz)
                if dt.date() == today:
                    calls_today.append(c)
            except Exception:
                continue

    total = len(calls_today)
    bookings = sum(1 for c in calls_today if c.get("type") == "booking" or c.get("booking_uid"))
    callbacks_requested = [c for c in calls_today if c.get("caller_wants_callback")]
    callbacks_pending = sum(1 for i, c in enumerate(callbacks_requested) if not _is_todo_done(c, i))
    urgent = [c for c in calls_today if c.get("is_urgent")]
    urgent_pending = sum(1 for i, c in enumerate(urgent) if not _is_todo_done(c, i))

    # Ωριαία κατανομή (ώρα της ημέρας -> πλήθος κλήσεων)
    hourly: dict[int, int] = {}
    for c in calls_today:
        try:
            dt = datetime.fromisoformat(c.get("logged_at", "").replace("Z", "+00:00")).astimezone(tz)
            hourly[dt.hour] = hourly.get(dt.hour, 0) + 1
        except Exception:
            continue
    peak_hour = max(hourly.items(), key=lambda kv: kv[1])[0] if hourly else None

    # Επαναλαμβανόμενοι καλούντες σήμερα (ίδιο τηλέφωνο 2+ φορές)
    phone_counts: dict[str, int] = {}
    for c in calls_today:
        p = (c.get("phone") or "").strip()
        if p:
            phone_counts[p] = phone_counts.get(p, 0) + 1
    repeat_callers = [{"phone": p, "count": n} for p, n in phone_counts.items() if n > 1]
    repeat_callers.sort(key=lambda x: -x["count"])

    # Completion rate από το call_metrics CSV (τελευταίες 7 μέρες, αν υπάρχει)
    completion = {}
    try:
        from call_metrics import weekly_summary
        summary = weekly_summary(days=7)
        by_outcome = summary.get("by_outcome", {})
        total_metrics = summary.get("calls", 0)
        completed = sum(v for k, v in by_outcome.items() if k != "abandoned")
        completion = {
            "total_7d": total_metrics,
            "completed_7d": completed,
            "abandoned_7d": by_outcome.get("abandoned", 0),
            "completion_rate_pct": round(100.0 * completed / total_metrics, 1) if total_metrics else None,
            "by_outcome_7d": by_outcome,
        }
    except Exception as exc:
        print(f"[INSIGHTS] weekly_summary unavailable: {exc}")

    # Σύνοψη ημέρας σε φυσική γλώσσα (ελληνικά)
    parts = [f"Σήμερα {total} {'κλήση' if total == 1 else 'κλήσεις'}"]
    if bookings:
        parts.append(f"{bookings} κράτησαν ραντεβού")
    if callbacks_requested:
        cb_word = "ζήτησε" if len(callbacks_requested) == 1 else "ζήτησαν"
        pending_note = f" ({callbacks_pending} ακόμα εκκρεμούν)" if callbacks_pending else " (όλα εξυπηρετήθηκαν)"
        parts.append(f"{len(callbacks_requested)} {cb_word} callback{pending_note}")
    if urgent:
        urg_verb = "εξυπηρετήθηκε" if len(urgent) == 1 else "εξυπηρετήθηκαν"
        urg_note = f" ({urgent_pending} εκκρεμούν)" if urgent_pending else f" ({urg_verb})"
        parts.append(f"{len(urgent)} επείγον{'τα' if len(urgent) > 1 else ''}{urg_note}")
    summary_text = ", ".join(parts) + "." if total else "Καμία κλήση σήμερα ακόμη."

    return {
        "summary_text": summary_text,
        "total_calls_today": total,
        "bookings_today": bookings,
        "callbacks_requested_today": len(callbacks_requested),
        "callbacks_pending_today": callbacks_pending,
        "urgent_today": len(urgent),
        "urgent_pending_today": urgent_pending,
        "hourly_distribution": hourly,
        "peak_hour": peak_hour,
        "repeat_callers_today": repeat_callers,
        "completion": completion,
    }


@app.get("/dashboard/call_benchmark")
async def dashboard_call_benchmark(key: str = "", days: int = 1):
    """Post-call benchmark: success by intent and where failed calls stopped."""
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        from call_benchmark import benchmark_summary

        return benchmark_summary(days=max(1, min(int(days or 1), 30)))
    except Exception as exc:
        print(f"[CALL_BENCHMARK] summary failed: {exc}")
        return JSONResponse({"error": "benchmark unavailable", "detail": str(exc)}, status_code=500)


@app.post("/dashboard/call_benchmark_probe")
async def dashboard_call_benchmark_probe(key: str = ""):
    """Protected write probe to verify live benchmark persistence on Render."""
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        from call_benchmark import CSV_PATH, benchmark_summary, record_call_benchmark

        row = record_call_benchmark(
            state={"current_intent": "message", "message_sent": True},
            payload={
                "reason_for_call": "message",
                "message_clean": "benchmark probe",
                "conversation_done": True,
                "call_id": f"benchmark-probe-{int(_time_module.time())}",
            },
            phone="benchmark-probe",
            nodes_visited=["probe"],
            duration_seconds=0,
            user_turns=0,
            tool_calls=0,
            disposition="probe",
        )
        return {"ok": True, "benchmark_file": str(CSV_PATH), "row": row, "summary": benchmark_summary(days=1)}
    except Exception as exc:
        print(f"[CALL_BENCHMARK] probe failed: {exc}")
        return JSONResponse({"error": "benchmark probe failed", "detail": str(exc)}, status_code=500)


@app.get("/dashboard/call_audits")
async def dashboard_call_audits(key: str = "", days: int = 1):
    """Persistent live-call audit summary written on Render disk."""
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        from call_audit import audit_summary

        return audit_summary(days=max(1, min(int(days or 1), 30)))
    except Exception as exc:
        print(f"[CALL_AUDIT] summary failed: {exc}")
        return JSONResponse({"error": "call audit unavailable", "detail": str(exc)}, status_code=500)


class IntentLabelRequest(BaseModel):
    call_id: str
    expected_intent: str
    reviewer: str = "dashboard"


@app.post("/dashboard/call_audits/label")
async def dashboard_call_audit_label(body: IntentLabelRequest, key: str = ""):
    """Persist a human intent label used for real live intent accuracy."""
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        from call_audit import audit_summary, save_intent_label

        label = await asyncio.to_thread(
            save_intent_label,
            body.call_id,
            body.expected_intent,
            body.reviewer,
        )
        return {"ok": True, "label": label, "summary": audit_summary(days=7)}
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        print(f"[CALL_AUDIT] intent label failed: {exc}")
        return JSONResponse({"error": "intent label failed", "detail": str(exc)}, status_code=500)


@app.post("/dashboard/call_quality/snapshot")
async def dashboard_call_quality_snapshot(key: str = ""):
    """Create/refresh daily and weekly snapshots on demand."""
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        from call_quality import write_due_snapshots

        return await asyncio.to_thread(write_due_snapshots)
    except Exception as exc:
        return JSONResponse({"error": "snapshot failed", "detail": str(exc)}, status_code=500)


@app.get("/dashboard/end_of_day_digest")
async def dashboard_end_of_day_digest(key: str = "", target_date: str = ""):
    """Cross-reference unresolved calls with upcoming appointments."""
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        requested_date = (
            date.fromisoformat(target_date)
            if target_date
            else datetime.now(TZ_NICOSIA).date()
        )
    except ValueError:
        return JSONResponse({"error": "invalid target_date"}, status_code=400)
    try:
        return await _build_end_of_day_digest(requested_date)
    except Exception as exc:
        print(f"DAILY CALL DIGEST endpoint error: {exc}")
        return JSONResponse(
            {"error": "digest failed", "detail": str(exc)},
            status_code=500,
        )


def _is_todo_done(call: dict, idx: int) -> bool:
    """Βοηθητική: ελέγχει αν ένα call entry έχει σημειωθεί ως ολοκληρωμένο (todo)."""
    tid = "td_" + (call.get("call_id") or ((call.get("phone") or "") + "@" + (call.get("logged_at") or "")[:16]))
    return tid in _load_todo_done()


@app.get("/dashboard/action_queue")
async def dashboard_action_queue(key: str = ""):
    """
    Δομημένη ουρά ενεργειών: μόνο τα items που χρειάζονται ανθρώπινη ενέργεια
    (callbacks, επείγοντα, αποτυχημένα bookings που δεν recovered), με
    context — τι χρειάζεται, πόση ώρα περιμένει, ποιο priority, ποιο ήταν
    το τελευταίο μήνυμα. Ταξινομημένα κατά urgency × χρόνο αναμονής, με
    auto-escalation αν κάτι περιμένει >2 ώρες χωρίς να σημειωθεί done.

    ΣΗΜΑΝΤΙΚΟ: δεν προσθέτει νέα δεδομένα — απλά φιλτράρει/ταξινομεί το ήδη
    υπάρχον _CALL_LOG με βάση πεδία που ήδη καταγράφουμε (is_urgent,
    caller_wants_callback, priority, booking_recovery).
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    ESCALATION_HOURS = float(os.getenv("ACTION_QUEUE_ESCALATION_HOURS", "2"))
    now_utc = datetime.now(tz=timezone.utc)

    with _CALL_LOG_LOCK:
        calls = list(_CALL_LOG)

    items = []
    for idx, c in enumerate(calls):
        is_urgent = bool(c.get("is_urgent"))
        wants_callback = bool(c.get("caller_wants_callback"))
        is_message = c.get("type") == "message"
        failed_booking = bool(c.get("booking_recovery")) and not (c.get("type") == "booking" or c.get("booking_uid"))

        # Μόνο items που πραγματικά χρειάζονται ενέργεια
        if not (is_urgent or wants_callback or is_message or failed_booking):
            continue

        done = _is_todo_done(c, idx)
        if done:
            continue  # ήδη ολοκληρωμένο, δεν εμφανίζεται στην ουρά

        # Χρόνος αναμονής
        wait_minutes = None
        try:
            logged_at = datetime.fromisoformat(c.get("logged_at", "").replace("Z", "+00:00"))
            wait_minutes = round((now_utc - logged_at).total_seconds() / 60)
        except Exception:
            pass

        # Τύπος ενέργειας
        if failed_booking:
            action_type = "failed_booking"
            action_label = "Αποτυχημένη κράτηση — χρειάζεται έλεγχος"
        elif is_urgent:
            action_type = "urgent"
            action_label = "Επείγον — χρειάζεται άμεση επικοινωνία"
        elif wants_callback:
            action_type = "callback"
            action_label = "Ζήτησε να τον/την καλέσετε"
        else:
            action_type = "message"
            action_label = "Μήνυμα προς τον γιατρό"

        # Priority: A (επείγον) > B (callback/failed booking) > C (μήνυμα)
        priority = c.get("priority") or ("A" if is_urgent else "C")

        # Auto-escalation: αν περιμένει πάνω από το όριο, ανεβαίνει σε A
        escalated = False
        if wait_minutes is not None and wait_minutes > ESCALATION_HOURS * 60 and priority != "A":
            priority = "A"
            escalated = True

        items.append({
            "phone": c.get("phone", ""),
            "first_name": c.get("first_name", ""),
            "last_name": c.get("last_name", ""),
            "clinic": c.get("clinic", ""),
            "action_type": action_type,
            "action_label": action_label,
            "last_message": c.get("message") or c.get("reason") or "",
            "priority": priority,
            "escalated": escalated,
            "wait_minutes": wait_minutes,
            "logged_at": c.get("logged_at", ""),
            "call_id": c.get("call_id", ""),
        })

    # Ταξινόμηση: priority (A πρώτα) × χρόνος αναμονής (πιο παλιό πρώτο)
    priority_rank = {"A": 0, "B": 1, "C": 2}
    items.sort(key=lambda it: (priority_rank.get(it["priority"], 3), -(it["wait_minutes"] or 0)))

    return {
        "total_pending": len(items),
        "escalated_count": sum(1 for it in items if it["escalated"]),
        "items": items,
    }


@app.get("/dashboard/anomalies")
async def dashboard_anomalies(key: str = ""):
    """
    Επιφανειακή προβολή ανωμαλιών που ο κώδικας ΗΔΗ ανιχνεύει αλλά μέχρι
    τώρα στέλνει μόνο με email/Slack:
      - Booking failures που ΔΕΝ recovered (booking_recovery=True αλλά
        χωρίς booking_uid — ο _recover_pending_finalize_booking απέτυχε).
      - Ασθενείς που ζήτησαν callback πριν >2 ώρες και δεν τσεκαρίστηκαν
        (ίδιο escalation κατώφλι με το action_queue).
      - Ασυνήθιστη αύξηση κλήσεων: συγκρίνει τις κλήσεις της τελευταίας
        ώρας με τον μέσο όρο ανά ώρα των τελευταίων 7 ημερών (από το
        call_metrics CSV) — αν είναι σημαντικά πάνω, σήμα προς τον γιατρό.
      - Cal.com/Setmore downtime (από το ήδη υπάρχον /dashboard/health
        ιστορικό — ίδια πηγή δεδομένων, εδώ απλά επισημαίνεται ως anomaly).
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    anomalies = []
    now_utc = datetime.now(tz=timezone.utc)

    with _CALL_LOG_LOCK:
        calls = list(_CALL_LOG)

    # 1. Unrecovered booking failures
    for c in calls:
        if c.get("booking_recovery") and not (c.get("type") == "booking" or c.get("booking_uid")):
            anomalies.append({
                "type": "unrecovered_booking",
                "severity": "high",
                "message": f"Αποτυχημένη κράτηση δεν ανακτήθηκε: {c.get('phone', '—')}",
                "phone": c.get("phone", ""),
                "logged_at": c.get("logged_at", ""),
            })

    # 2. Stale callback requests (>2 ώρες, δεν σημειώθηκαν done)
    ESCALATION_HOURS = float(os.getenv("ACTION_QUEUE_ESCALATION_HOURS", "2"))
    for idx, c in enumerate(calls):
        if not c.get("caller_wants_callback"):
            continue
        if _is_todo_done(c, idx):
            continue
        try:
            logged_at = datetime.fromisoformat(c.get("logged_at", "").replace("Z", "+00:00"))
            wait_hours = (now_utc - logged_at).total_seconds() / 3600
        except Exception:
            continue
        if wait_hours > ESCALATION_HOURS:
            anomalies.append({
                "type": "stale_callback",
                "severity": "medium",
                "message": f"Callback εκκρεμεί {round(wait_hours, 1)} ώρες: {c.get('phone', '—')}",
                "phone": c.get("phone", ""),
                "logged_at": c.get("logged_at", ""),
            })

    # 3. Ασυνήθιστη αύξηση κλήσεων (τελευταία ώρα vs μέσος όρος 7 ημερών)
    last_hour_count = 0
    for c in calls:
        try:
            logged_at = datetime.fromisoformat(c.get("logged_at", "").replace("Z", "+00:00"))
            if (now_utc - logged_at).total_seconds() <= 3600:
                last_hour_count += 1
        except Exception:
            continue
    try:
        from call_metrics import weekly_summary
        summary = weekly_summary(days=7)
        total_7d = summary.get("calls", 0)
        avg_per_hour_7d = total_7d / (7 * 24) if total_7d else 0
        if avg_per_hour_7d > 0 and last_hour_count > avg_per_hour_7d * 3 and last_hour_count >= 5:
            anomalies.append({
                "type": "call_volume_spike",
                "severity": "low",
                "message": f"{last_hour_count} κλήσεις την τελευταία ώρα (μ.ο. {round(avg_per_hour_7d, 1)}/ώρα)",
                "phone": "",
                "logged_at": now_utc.isoformat(),
            })
    except Exception as exc:
        print(f"[ANOMALIES] volume check unavailable: {exc}")

    # 4. Cal.com/Setmore downtime (από το ήδη υπάρχον health history)
    with _HEALTH_HISTORY_LOCK:
        last_health = _HEALTH_HISTORY[-1] if _HEALTH_HISTORY else None
    if last_health:
        if last_health.get("cal_com_ok") is False:
            anomalies.append({
                "type": "cal_com_down",
                "severity": "high",
                "message": "Cal.com δεν απαντά στον τελευταίο έλεγχο.",
                "phone": "",
                "logged_at": last_health.get("checked_at", ""),
            })
        if last_health.get("setmore_ok") is False:
            anomalies.append({
                "type": "setmore_down",
                "severity": "medium",
                "message": "Setmore δεν απαντά στον τελευταίο έλεγχο.",
                "phone": "",
                "logged_at": last_health.get("checked_at", ""),
            })

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    anomalies.sort(key=lambda a: severity_rank.get(a["severity"], 3))

    return {"total": len(anomalies), "anomalies": anomalies}


@app.get("/dashboard/weekly_triage")
async def dashboard_weekly_triage(key: str = "", days: int = 7):
    """
    Εβδομαδιαίο triage: κατηγοριοποιεί τις κλήσεις σε «Πέτυχαν» / «Χρειάζονται
    έλεγχο» / «Αποτυχημένες», με ΣΑΦΗ, ελέγξιμα κριτήρια — όχι «AI κρίση».
    Κάθε κλήση δείχνει ΓΙΑΤΙ κατηγοριοποιήθηκε έτσι.

    Κύριο κριτήριο είναι η ολοκλήρωση του intent. Το telephony disposition
    (π.χ. user_hangup) είναι μόνο πληροφοριακό και δεν αναιρεί booking/message.
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    with _CALL_LOG_LOCK:
        calls = list(_CALL_LOG)

    recent = []
    for c in calls:
        try:
            logged_at = datetime.fromisoformat(c.get("logged_at", "").replace("Z", "+00:00"))
            if logged_at >= cutoff:
                recent.append(c)
        except Exception:
            continue

    # Group by phone+5min window για να μην μετράμε διπλά το ίδιο πραγματικό
    # τηλεφωνικό συμβάν (webhook+action logging για την ίδια κλήση).
    failed, needs_review, succeeded = [], [], []
    intent_stats: dict[str, dict[str, int]] = {}

    for c in recent:
        disposition = (c.get("disposition") or "").strip().lower()
        intent = (c.get("reason") or "").strip().lower()
        incomplete = bool(c.get("incomplete"))
        wants_callback = bool(c.get("caller_wants_callback"))
        has_booking = bool(c.get("booking_uid"))
        call_type = c.get("type") or ""
        is_message_type = call_type == "message"
        is_cancellation_type = call_type == "cancellation"
        is_booking_type = call_type == "booking"
        has_real_outcome = is_message_type or is_cancellation_type or is_booking_type or has_booking
        intent_completed = c.get("intent_completed")
        if intent_completed is None:
            if intent in {"appointment", "change"}:
                intent_completed = has_booking or is_booking_type
            elif intent in {"message", "urgent"}:
                intent_completed = is_message_type
            elif intent == "cancellation":
                intent_completed = is_cancellation_type
            else:
                intent_completed = has_real_outcome
        intent_completed = bool(intent_completed)
        intent_key = intent or "unknown"
        intent_bucket = intent_stats.setdefault(intent_key, {"calls": 0, "completed": 0})
        intent_bucket["calls"] += 1
        if intent_completed:
            intent_bucket["completed"] += 1
        reasons = []

        # ── 🔴 Σαφώς αποτυχημένη ──────────────────────────────────────
        # ΣΗΜΑΝΤΙΚΟ: «incomplete» (από το backend, ήδη συνυπολογίζει
        # nodes_visited<=1 ΚΑΙ απουσία περιεχομένου) είναι αξιόπιστο σήμα
        # αποτυχίας. Το disposition=user_hangup ΑΠΟ ΜΟΝΟ ΤΟΥ ΔΕΝ είναι —
        # ΚΑΘΕ κλήση τελειώνει με τον καλούντα να κλείνει το τηλέφωνο,
        # ακόμα και μετά από πλήρως επιτυχημένο booking+cancellation.
        if disposition == "pipeline_error":
            reasons.append("Pipeline error στην κλήση (τεχνικό σφάλμα μέσα στην κλήση)")
        if incomplete and not intent_completed:
            reasons.append("Η κλήση διεκόπη πολύ σύντομα, χωρίς ουσιαστική πρόοδο")
        if intent == "appointment" and not intent_completed and not wants_callback and not incomplete:
            reasons.append("Ζητήθηκε ραντεβού/αλλαγή αλλά δεν καταγράφηκε ολοκληρωμένο αποτέλεσμα")
        if (intent == "message" or wants_callback) and not intent_completed and not incomplete:
            reasons.append("Ζητήθηκε μήνυμα/callback αλλά δεν καταγράφηκε αποστολή μηνύματος")

        if reasons:
            failed.append({**c, "_triage_reasons": reasons})
            continue

        # ── 🟢 Πέτυχε — ελέγχεται ΠΡΙΝ το "needs_review", ώστε ένα
        #    φυσιολογικό user_hangup ΜΕΤΑ από πραγματικό αποτέλεσμα να
        #    μην ταξινομηθεί λάθος ως "χρειάζεται έλεγχο" ──────────────
        if intent_completed:
            outcome_labels = {
                "booking": "Ολοκληρώθηκε το intent: κλείστηκε ραντεβού",
                "message": "Ολοκληρώθηκε το intent: στάλθηκε μήνυμα",
                "urgent": "Ολοκληρώθηκε το intent: στάλθηκε επείγουσα ενημέρωση",
                "cancel": "Ολοκληρώθηκε το intent: ακυρώθηκε ραντεβού",
            }
            actual_outcome = c.get("actual_outcome") or (
                "booking" if has_booking or is_booking_type else
                "cancel" if is_cancellation_type else
                "message" if is_message_type else "completed"
            )
            success_reasons = [outcome_labels.get(actual_outcome, "Ολοκληρώθηκε το ζητούμενο της κλήσης")]
            if disposition == "user_hangup":
                success_reasons.append("Ο καλών έκλεισε αφού ολοκληρώθηκε το ζητούμενο")
            succeeded.append({**c, "_triage_reasons": success_reasons})
            continue

        # ── 🟡 Χρειάζεται έλεγχος ─────────────────────────────────────
        if wants_callback and not is_message_type:
            reasons.append("Ζήτησε callback — επιβεβαίωσε ότι στάλθηκε μήνυμα στον γιατρό")
        if disposition in ("user_hangup", "no_answer", "busy", "abandoned") and not incomplete and not has_real_outcome:
            reasons.append(f"Disposition «{disposition}» χωρίς καταγεγραμμένο αποτέλεσμα — άξιζε σύντομος έλεγχος του transcript")
        if c.get("_repeat"):
            reasons.append("Επαναλαμβανόμενος καλών (ίδιο τηλέφωνο, ίδια μέρα)")

        if reasons:
            needs_review.append({**c, "_triage_reasons": reasons})
            continue

        # ── 🟢 Πέτυχε (υπόλοιπες περιπτώσεις) ───────────────────────
        if disposition in ("completed", "") and not incomplete:
            succeeded.append({**c, "_triage_reasons": ["Κλήση ολοκληρώθηκε κανονικά"]})
        else:
            needs_review.append({**c, "_triage_reasons": ["Δεν ταιριάζει σε γνωστή κατηγορία — χρειάζεται χειροκίνητος έλεγχος"]})

    total = len(recent)
    by_intent = {
        intent_name: {
            **counts,
            "completion_rate_pct": round(100.0 * counts["completed"] / counts["calls"], 1)
            if counts["calls"] else None,
        }
        for intent_name, counts in sorted(intent_stats.items())
    }
    latency_summary = {}
    try:
        from call_audit import audit_summary

        audit = audit_summary(days=days)
        latency_summary = {
            "calls_with_audit": audit.get("calls", 0),
            "avg_reasoning_total_ms": audit.get("avg_reasoning_total_ms"),
            "p50_reasoning_total_ms": audit.get("p50_reasoning_total_ms"),
            "p95_reasoning_total_ms": audit.get("p95_reasoning_total_ms"),
            "max_reasoning_total_ms": audit.get("max_reasoning_total_ms"),
            "failed_checks": audit.get("failed_checks", {}),
        }
        scorecard = {
            "intent_accuracy_pct": audit.get("intent_accuracy_pct"),
            "intent_accuracy_labeled_calls": audit.get("intent_accuracy_labeled_calls", 0),
            "intent_completion_rate_pct": audit.get("intent_completion_rate_pct"),
            "wrong_route_rate_pct": audit.get("wrong_route_rate_pct"),
            "p95_reasoning_total_ms": audit.get("p95_reasoning_total_ms"),
            "stuck_rate_pct": audit.get("stuck_rate_pct"),
            "avg_user_turns": audit.get("avg_user_turns"),
            "per_intent": audit.get("per_intent", {}),
            "route_status": audit.get("route_status", {}),
        }
        labels_by_call = {
            str(row.get("call_id") or ""): row.get("expected_intent")
            for row in audit.get("recent_calls", [])
            if row.get("call_id") and row.get("expected_intent")
        }
        for group in (failed, needs_review, succeeded):
            for item in group:
                call_id = str(item.get("call_id") or "")
                if call_id in labels_by_call:
                    item["_expected_intent"] = labels_by_call[call_id]
    except Exception as exc:
        print(f"[WEEKLY_TRIAGE] latency audit unavailable: {exc}")
        scorecard = {}

    intent_completion_rate = round(100.0 * len(succeeded) / total, 1) if total else None
    latency_avg = latency_summary.get("avg_reasoning_total_ms")
    if total < 10:
        quality_gate = {
            "status": "insufficient_sample",
            "reason": "Χρειάζονται τουλάχιστον δέκα κλήσεις για συγκρίσιμη περίοδο.",
        }
    elif latency_avg is None:
        quality_gate = {
            "status": "missing_latency",
            "reason": "Υπάρχει μέτρηση intent completion, αλλά λείπουν reasoning delays.",
        }
    elif (intent_completion_rate or 0) >= 85 and latency_avg < 6000:
        quality_gate = {
            "status": "good_period",
            "reason": "Ολοκλήρωση intent τουλάχιστον ογδόντα πέντε τοις εκατό και μέσο reasoning κάτω από έξι δευτερόλεπτα.",
        }
    else:
        quality_gate = {
            "status": "below_target",
            "reason": "Η περίοδος δεν περνά ακόμη τα όρια ολοκλήρωσης intent και latency.",
        }

    return {
        "period_days": days,
        "total_calls": total,
        "succeeded_count": len(succeeded),
        "needs_review_count": len(needs_review),
        "failed_count": len(failed),
        "success_rate_pct": intent_completion_rate,
        "failed": failed,
        "needs_review": needs_review,
        "succeeded": succeeded,
        "intent_completion_rate_pct": intent_completion_rate,
        "by_intent": by_intent,
        "latency": latency_summary,
        "scorecard": scorecard,
        "quality_gate": quality_gate,
        "note": (
            "Η επιτυχία μετριέται από την ολοκλήρωση του intent. Το user_hangup "
            "δεν θεωρείται αποτυχία όταν έχει ήδη κλειστεί ραντεβού, σταλεί "
            "μήνυμα ή ολοκληρωθεί ακύρωση. Η ποιότητα latency εμφανίζεται μόνο "
            "όταν το Dograh παρέχει reasoning delays στο webhook."
        ),
    }


@app.get("/dashboard/rule_gaps")
async def dashboard_rule_gaps(key: str = "", days: int = 7):
    """
    Συγκεντρώνει τις κλήσεις της εβδομάδας όπου το rule_based_intent ΔΕΝ
    ήταν confident, ομαδοποιημένες σε δύο κατηγορίες:

    1. "fallback_used": το ONNX μοντέλο απάντησε με αρκετό confidence (πέρασε
       το per-intent threshold) και χρησιμοποιήθηκε για την πραγματική
       απόφαση routing.
    2. "fallback_uncertain": το ONNX μοντέλο δοκιμάστηκε αλλά ΟΥΤΕ αυτό ήταν
       αρκετά σίγουρο (κάτω από threshold) — εδώ ο καλών έλαβε «δεν το
       κατάλαβα καλά» ΑΚΟΜΑ ΚΑΙ μετά το ML fallback. Αυτές είναι οι πιο
       σημαντικές υποψήφιες για είτε νέο ρητό κανόνα είτε νέο training παράδειγμα.

    ΣΚΟΠΟΣ: το rule_based_intent είναι σκόπιμα deterministic — ίδιο κείμενο,
    ίδιο αποτέλεσμα πάντα, χωρίς μεταβλητότητα LLM reasoning. Το ML fallback
    υπάρχει μόνο για να καλύπτει το κενό μεταξύ ενημερώσεις κανόνων. Αυτό
    το endpoint δείχνει ΠΟΙΕΣ φράσεις χρειάστηκαν το fallback, ομαδοποιημένες
    κατά ομοιότητα κειμένου — ώστε επαναλαμβανόμενα μοτίβα να γίνονται
    υποψήφιες για νέο μόνιμο, ρητό κανόνα στο appointment_utils.py, μειώνοντας
    προοδευτικά την εξάρτηση από το fallback με την πάροδο του χρόνου.
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    with _CALL_LOG_LOCK:
        calls = list(_CALL_LOG)

    fallback_used_calls = []
    fallback_uncertain_calls = []
    for c in calls:
        # "ml_fallback_label" είναι κενό string όταν το ONNX ΔΕΝ δοκιμάστηκε
        # καθόλου (π.χ. ENABLE_CY_INTENT=false, ή rule ήταν ήδη confident).
        if not c.get("ml_fallback_label"):
            continue
        try:
            logged_at = datetime.fromisoformat(c.get("logged_at", "").replace("Z", "+00:00"))
            if logged_at < cutoff:
                continue
        except Exception:
            continue
        if c.get("ml_fallback_used"):
            fallback_used_calls.append(c)
        else:
            fallback_uncertain_calls.append(c)

    # Απλή ομαδοποίηση κατά κοινές λέξεις-κλειδιά (χωρίς εξωτερική βιβλιοθήκη
    # ομοιότητας κειμένου) — μετράμε ποιες λέξεις (>3 γράμματα, εκτός
    # συνηθισμένων stop-words) εμφανίζονται σε πολλές κλήσεις μαζί.
    import re as _re
    from collections import Counter as _Counter

    _STOP = {"θελω", "να", "και", "για", "μου", "σας", "ειμαι", "που", "ηθελα", "κατι", "αλλα"}

    def _keywords(text):
        words = _re.findall(r"[\u03b1-\u03c9\u03af\u03ca\u0390\u03cc\u03ac\u03ad\u03cd\u03cb\u03b0\u03ae\u03ce]+", (text or "").lower())
        return {w for w in words if len(w) > 3 and w not in _STOP}

    def _summarize(call_list):
        keyword_counts = _Counter()
        for c in call_list:
            keyword_counts.update(_keywords(c.get("message", "")))
        repeated = {k: v for k, v in keyword_counts.items() if v >= 2}
        return {
            "count": len(call_list),
            "repeated_keywords": dict(sorted(repeated.items(), key=lambda kv: -kv[1])),
            "calls": [
                {
                    "phone": c.get("phone", ""),
                    "message": c.get("message", ""),
                    "logged_at": c.get("logged_at", ""),
                    "rule_intent_before_fallback": c.get("rule_intent_before_fallback", ""),
                    "ml_fallback_label": c.get("ml_fallback_label", ""),
                    "ml_fallback_confidence": c.get("ml_fallback_confidence"),
                }
                for c in call_list
            ],
        }

    return {
        "period_days": days,
        "fallback_used": _summarize(fallback_used_calls),
        "fallback_uncertain": _summarize(fallback_uncertain_calls),
        "note": (
            "fallback_used: το ONNX απάντησε με αρκετό confidence και "
            "χρησιμοποιήθηκε. fallback_uncertain: ΟΥΤΕ το ONNX ήταν σίγουρο "
            "— ο καλών έλαβε «δεν κατάλαβα» ακόμα και μετά το fallback. "
            "Επαναλαμβανόμενες λέξεις-κλειδιά υποδεικνύουν μοτίβα που αξίζει "
            "να γίνουν νέος ρητός κανόνας ή νέο training παράδειγμα."
        ),
    }


@app.get("/dashboard/rule_gaps")
async def dashboard_rule_gaps(key: str = "", days: int = 7):
    """
    Συγκεντρώνει τις κλήσεις της εβδομάδας όπου το rule_based_intent ΔΕΝ
    ήταν confident, ομαδοποιημένες σε δύο κατηγορίες:

    1. "fallback_used": το ONNX μοντέλο απάντησε με αρκετό confidence (πέρασε
       το per-intent threshold) και χρησιμοποιήθηκε για την πραγματική
       απόφαση routing.
    2. "fallback_uncertain": το ONNX μοντέλο δοκιμάστηκε αλλά ΟΥΤΕ αυτό ήταν
       αρκετά σίγουρο (κάτω από threshold) — εδώ ο καλών έλαβε «δεν το
       κατάλαβα καλά» ΑΚΟΜΑ ΚΑΙ μετά το ML fallback. Αυτές είναι οι πιο
       σημαντικές υποψήφιες για είτε νέο ρητό κανόνα είτε νέο training παράδειγμα.

    ΣΚΟΠΟΣ: το rule_based_intent είναι σκόπιμα deterministic — ίδιο κείμενο,
    ίδιο αποτέλεσμα πάντα, χωρίς μεταβλητότητα LLM reasoning. Το ML fallback
    υπάρχει μόνο για να καλύπτει το κενό μεταξύ ενημερώσεις κανόνων. Αυτό
    το endpoint δείχνει ΠΟΙΕΣ φράσεις χρειάστηκαν το fallback, ομαδοποιημένες
    κατά ομοιότητα κειμένου — ώστε επαναλαμβανόμενα μοτίβα να γίνονται
    υποψήφιες για νέο μόνιμο, ρητό κανόνα στο appointment_utils.py, μειώνοντας
    προοδευτικά την εξάρτηση από το fallback με την πάροδο του χρόνου.
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    with _CALL_LOG_LOCK:
        calls = list(_CALL_LOG)

    fallback_used_calls = []
    fallback_uncertain_calls = []
    for c in calls:
        # "ml_fallback_label" είναι κενό string όταν το ONNX ΔΕΝ δοκιμάστηκε
        # καθόλου (π.χ. ENABLE_CY_INTENT=false, ή rule ήταν ήδη confident).
        if not c.get("ml_fallback_label"):
            continue
        try:
            logged_at = datetime.fromisoformat(c.get("logged_at", "").replace("Z", "+00:00"))
            if logged_at < cutoff:
                continue
        except Exception:
            continue
        if c.get("ml_fallback_used"):
            fallback_used_calls.append(c)
        else:
            fallback_uncertain_calls.append(c)

    # Απλή ομαδοποίηση κατά κοινές λέξεις-κλειδιά (χωρίς εξωτερική βιβλιοθήκη
    # ομοιότητας κειμένου) — μετράμε ποιες λέξεις (>3 γράμματα, εκτός
    # συνηθισμένων stop-words) εμφανίζονται σε πολλές κλήσεις μαζί.
    import re as _re
    from collections import Counter as _Counter

    _STOP = {"θελω", "να", "και", "για", "μου", "σας", "ειμαι", "που", "ηθελα", "κατι", "αλλα"}

    def _keywords(text):
        words = _re.findall(r"[\u03b1-\u03c9\u03af\u03ca\u0390\u03cc\u03ac\u03ad\u03cd\u03cb\u03b0\u03ae\u03ce]+", (text or "").lower())
        return {w for w in words if len(w) > 3 and w not in _STOP}

    def _summarize(call_list):
        keyword_counts = _Counter()
        for c in call_list:
            keyword_counts.update(_keywords(c.get("message", "")))
        repeated = {k: v for k, v in keyword_counts.items() if v >= 2}
        return {
            "count": len(call_list),
            "repeated_keywords": dict(sorted(repeated.items(), key=lambda kv: -kv[1])),
            "calls": [
                {
                    "phone": c.get("phone", ""),
                    "message": c.get("message", ""),
                    "logged_at": c.get("logged_at", ""),
                    "rule_intent_before_fallback": c.get("rule_intent_before_fallback", ""),
                    "ml_fallback_label": c.get("ml_fallback_label", ""),
                    "ml_fallback_confidence": c.get("ml_fallback_confidence"),
                }
                for c in call_list
            ],
        }

    return {
        "period_days": days,
        "fallback_used": _summarize(fallback_used_calls),
        "fallback_uncertain": _summarize(fallback_uncertain_calls),
        "note": (
            "fallback_used: το ONNX απάντησε με αρκετό confidence και "
            "χρησιμοποιήθηκε. fallback_uncertain: ΟΥΤΕ το ONNX ήταν σίγουρο "
            "— ο καλών έλαβε «δεν κατάλαβα» ακόμα και μετά το fallback. "
            "Επαναλαμβανόμενες λέξεις-κλειδιά υποδεικνύουν μοτίβα που αξίζει "
            "να γίνουν νέος ρητός κανόνας ή νέο training παράδειγμα."
        ),
    }


@app.get("/dashboard/patient_volume")
async def dashboard_patient_volume(key: str = "", days: int = 60):
    """
    Πραγματικός μέσος όρος ασθενών/ημέρα από το ιστορικό Setmore (όχι
    εκτίμηση) — για να επιβεβαιωθεί ο σχεδιασμός νέου προγράμματος ραντεβού.
    Μετράει μόνο εργάσιμες ημέρες (Δευτ-Παρ, εξαιρουμένων Σαβ/Κυρ) στο
    διάστημα [σήμερα - days, σήμερα], ξεχωριστά ανά κλινική.
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Asia/Nicosia")
    today = datetime.now(tz).date()
    start_date = today - timedelta(days=days)

    try:
        snapshots = await list_setmore_synced_appointments(
            start_date=start_date,
            end_date=today,
            page_limit=20,
        )
    except Exception as exc:
        return JSONResponse({"error": f"Setmore fetch failed: {exc}"}, status_code=502)

    by_day: dict[str, dict[str, int]] = {}
    for snap in snapshots:
        start_time = getattr(snap, "start_time", None) or ""
        if not start_time:
            continue
        try:
            dt = datetime.fromisoformat(start_time.replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        if dt.date() < start_date or dt.date() > today:
            continue
        if dt.weekday() >= 5:  # Σαββατοκύριακο — δεν δουλεύει το ιατρείο
            continue
        day_key = dt.date().isoformat()
        clinic = (getattr(snap, "clinic", None) or "unknown")
        by_day.setdefault(day_key, {"total": 0})
        by_day[day_key]["total"] += 1
        by_day[day_key][clinic] = by_day[day_key].get(clinic, 0) + 1

    working_days = len(by_day)
    total_appointments = sum(d["total"] for d in by_day.values())
    avg_per_day = round(total_appointments / working_days, 1) if working_days else None

    # Κατανομή ανά κλινική
    clinic_totals: dict[str, int] = {}
    for d in by_day.values():
        for k, v in d.items():
            if k != "total":
                clinic_totals[k] = clinic_totals.get(k, 0) + v
    clinic_avgs = {
        k: round(v / working_days, 1) if working_days else None
        for k, v in clinic_totals.items()
    }

    return {
        "period_days": days,
        "working_days_with_data": working_days,
        "total_appointments": total_appointments,
        "avg_patients_per_working_day": avg_per_day,
        "avg_per_clinic": clinic_avgs,
        "note": (
            "Υπολογισμένο από πραγματικά Setmore δεδομένα, μόνο εργάσιμες "
            "ημέρες (Δευτ-Παρ) με τουλάχιστον 1 ραντεβού."
        ),
    }


# Λέξεις-κλειδιά που υποδεικνύουν ότι ο ασθενής ζητά κάτι που το σύστημα
# (ή το ιατρείο) δεν προσφέρει σήμερα — πιθανό feature request.
_FEATURE_REQUEST_PATTERNS = [
    ("video_call", ["βιντεοκλήση", "βιντεο κληση", "online ραντεβού", "τηλεϊατρικη", "τηλεϊατρική"]),
    ("sms_reminder", ["sms υπενθύμιση", "μήνυμα υπενθύμιση", "να μου στείλετε μήνυμα"]),
    ("online_booking", ["να κλείσω online", "από το ίντερνετ", "ιστοσελίδα για ραντεβού", "app για ραντεβού"]),
    ("payment_online", ["να πληρώσω online", "κάρτα μέσω τηλεφώνου", "πληρωμή ηλεκτρονικά"]),
    ("multilingual", ["αγγλικά", "english", "russian", "ρωσικά"]),
    ("weekend_hours", ["σάββατο", "κυριακή", "σαββατοκύριακο"]),
    ("evening_hours", ["μετά τις 18", "βραδινό ραντεβού", "αργά το απόγευμα"]),
    ("results_followup", ["αποτελέσματα εξετάσεων", "να μάθω τα αποτελέσματα", "έτοιμες εξετάσεις"]),
]


@app.get("/dashboard/feature_requests")
async def dashboard_feature_requests(key: str = "", days: int = 30):
    """
    Σκανάρει τα μηνύματα ασθενών (ήδη υπάρχοντα στο _CALL_LOG) για patterns
    που υποδεικνύουν αίτημα για κάτι που δεν υποστηρίζεται σήμερα (π.χ.
    online booking, βραδινό ωράριο, SMS υπενθυμίσεις). Δεν είναι AI
    ανάλυση — απλό keyword matching, ώστε να φαίνεται ξεκάθαρα ΤΙ
    εντοπίστηκε και ΓΙΑΤΙ, χωρίς false-confidence από LLM εκτίμηση.
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    import unicodedata as _ud

    def _norm(text: str) -> str:
        text = (text or "").lower()
        text = "".join(c for c in _ud.normalize("NFD", text) if _ud.category(c) != "Mn")
        return text

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    with _CALL_LOG_LOCK:
        calls = list(_CALL_LOG)

    matches: dict[str, list[dict]] = {}
    for c in calls:
        msg = c.get("message") or c.get("reason") or ""
        if not msg:
            continue
        try:
            logged_at = datetime.fromisoformat(c.get("logged_at", "").replace("Z", "+00:00"))
            if logged_at < cutoff:
                continue
        except Exception:
            continue
        norm_msg = _norm(msg)
        for category, keywords in _FEATURE_REQUEST_PATTERNS:
            if any(_norm(kw) in norm_msg for kw in keywords):
                matches.setdefault(category, []).append({
                    "phone": c.get("phone", ""),
                    "message": msg,
                    "logged_at": c.get("logged_at", ""),
                })

    summary = [
        {"category": cat, "count": len(items), "examples": items[:3]}
        for cat, items in matches.items()
    ]
    summary.sort(key=lambda s: -s["count"])

    return {
        "period_days": days,
        "categories_found": len(summary),
        "requests": summary,
        "note": "Βασίζεται σε λέξεις-κλειδιά στα μηνύματα ασθενών — όχι σε AI ερμηνεία.",
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_ui(key: str = ""):
    """
    Live dashboard κλήσεων.
    Προστατεύεται με query param ?key=DASHBOARD_KEY
    """
    dashboard_key = os.getenv("DASHBOARD_KEY", "")
    if dashboard_key and key != dashboard_key:
        return HTMLResponse("<h1>401 Unauthorized</h1>", status_code=401)

    html = """<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ενημέρωση Κλήσεων — Ιατρείο Παπαχρήστου</title>
<style>
  :root {
    --paper:#f6f2e9; --card:#fffdf8; --ink:#1c1b18; --ink-soft:#6f6a5e;
    --rule:#ddd6c6; --accent:#0f5f54; --clay:#b06a3b; --urgent:#a32b22;
    --serif:'Iowan Old Style','Palatino Linotype','Palatino','Georgia',serif;
    --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
    --mono:'SF Mono',ui-monospace,Menlo,Consolas,monospace;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:var(--sans); background:var(--paper); color:var(--ink);
         min-height:100vh; -webkit-font-smoothing:antialiased; line-height:1.5; }
  .wrap { max-width:1100px; margin:0 auto; padding:28px 22px 80px; }

  /* Masthead */
  .masthead { border-bottom:2px solid var(--ink); padding-bottom:18px; margin-bottom:8px; }
  .eyebrow { font-size:.72rem; letter-spacing:.22em; text-transform:uppercase;
             color:var(--accent); font-weight:700; }
  .masthead h1 { font-family:var(--serif); font-weight:600; font-size:2.9rem;
                 line-height:1.02; letter-spacing:-.01em; margin:.12em 0 .14em; }
  .masthead h1 em { font-style:italic; }
  .dek { font-family:var(--serif); font-size:1.12rem; color:var(--ink); max-width:60ch;
         font-style:italic; }
  .dek b { font-style:normal; font-weight:600; color:var(--accent); }

  /* Stat line */
  .stats { display:flex; flex-wrap:wrap; gap:0; margin:18px 0 30px;
           border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); }
  .stat { flex:1 1 0; min-width:84px; padding:12px 14px; border-right:1px solid var(--rule); }
  .stat:last-child { border-right:0; }
  .stat .n { font-family:var(--serif); font-size:1.7rem; font-weight:600; line-height:1; }
  .stat .l { font-size:.7rem; letter-spacing:.08em; text-transform:uppercase;
             color:var(--ink-soft); margin-top:5px; }
  .stat.alert .n { color:var(--urgent); }

  /* Sections */
  .section { margin:34px 0 0; }
  .section-head { display:flex; align-items:baseline; gap:10px;
                  border-bottom:1px solid var(--ink); padding-bottom:6px; margin-bottom:14px; }
  .section-head h2 { font-family:var(--serif); font-size:1.18rem; font-weight:600; }
  .section-head .count { font-size:.74rem; color:var(--ink-soft); font-variant-numeric:tabular-nums; }

  /* Action items (callbacks / urgent) */
  .item { background:var(--card); border:1px solid var(--rule); border-radius:10px;
          padding:14px 16px; margin-bottom:11px; position:relative; }
  .item.urgent { border-left:3px solid var(--urgent); }
  .item.new { animation:flash 1.6s ease-out; }
  @keyframes flash { 0%{background:#fdf3d6;} 100%{background:var(--card);} }
  .item-top { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
  .who { font-family:var(--serif); font-size:1.05rem; font-weight:600; }
  .who .ph { font-family:var(--mono); font-size:.9rem; font-weight:400; color:var(--accent); }
  .when { margin-left:auto; font-family:var(--mono); font-size:.76rem; color:var(--ink-soft); }
  .body { margin:7px 0 2px; font-size:.95rem; color:#33312b; }
  .meta-tags { display:flex; gap:6px; flex-wrap:wrap; margin-top:9px; }
  .tag { font-size:.68rem; letter-spacing:.04em; padding:2px 8px; border-radius:999px;
         border:1px solid var(--rule); color:var(--ink-soft); background:#fbf8f0; }
  .tag.u { color:var(--urgent); border-color:#e6c3bf; background:#fbeeed; font-weight:600; }
  .tag.c { color:var(--accent); border-color:#bcd8d2; background:#eef6f4; }
  .tag.b { color:#7a5a16; border-color:#e6d6a8; background:#fbf3da; }
  .tag.y { color:#8a6a2a; }
  .tag.i { color:var(--clay); border-color:#e6cdb6; background:#fbf0e6; }
  .acts { display:flex; gap:8px; margin-top:12px; }
  .btn { font-family:var(--sans); font-size:.82rem; font-weight:600; text-decoration:none;
         padding:7px 14px; border-radius:8px; display:inline-flex; align-items:center; gap:6px; }
  .btn.call { background:var(--accent); color:#fff; }
  .btn.sms { background:var(--clay); color:#fff; }
  .btn:active { transform:translateY(1px); }

  /* Booking rows */
  .bk { display:flex; align-items:baseline; gap:12px; padding:10px 2px;
        border-bottom:1px solid var(--rule); }
  .bk:last-child { border-bottom:0; }
  .bk .who { font-size:.98rem; }
  .bk .ph { font-family:var(--mono); font-size:.84rem; color:var(--ink-soft); font-weight:400; }
  .bk .when { font-size:.78rem; }

  /* Timeline (all calls) */
  .tl { border-left:2px solid var(--rule); margin-left:4px; padding-left:0; }
  .row { display:flex; gap:10px; padding:9px 0 9px 16px; position:relative; align-items:flex-start;
         border-bottom:1px solid #ebe5d6; }
  .row:last-child { border-bottom:0; }
  .row::before { content:''; position:absolute; left:-5px; top:15px; width:8px; height:8px;
                 border-radius:50%; background:var(--rule); }
  .row.u::before { background:var(--urgent); }
  .row.b::before { background:var(--accent); }
  .row .rtime { font-family:var(--mono); font-size:.74rem; color:var(--ink-soft);
                min-width:78px; padding-top:1px; }
  .row .rmain { flex:1; }
  .row .rwho { font-weight:600; font-size:.9rem; }
  .row .rwho .ph { font-family:var(--mono); font-weight:400; color:var(--ink-soft); }
  .row .rmsg { font-size:.86rem; color:#52503f; margin-top:2px; }

  .empty { text-align:center; color:var(--ink-soft); font-family:var(--serif);
           font-style:italic; padding:26px; font-size:1.02rem; }

  /* Toolbar */
  .toolbar { display:flex; align-items:center; gap:10px; margin-top:14px;
             font-size:.78rem; color:var(--ink-soft); }
  .toolbar .dot { width:8px; height:8px; border-radius:50%; background:#22c55e; }
  .toolbar .sp { flex:1; }
  .tbtn { font:inherit; font-weight:600; color:var(--accent); background:transparent;
          border:1px solid var(--rule); border-radius:7px; padding:5px 11px; cursor:pointer; }
  .tbtn:hover { border-color:var(--accent); }
  .picker { margin-top:8px; display:none; gap:8px; align-items:center; }
  .picker input { font:inherit; padding:5px 8px; border:1px solid var(--rule); border-radius:7px; }

  /* Grid layout */
  .grid { display:grid; grid-template-columns:1.4fr 1fr; gap:22px; margin-top:8px; }
  .grid .col-main { min-width:0; }
  .grid .col-side { min-width:0; }
  .full-w { grid-column:1/-1; }

  /* ── Φάση 3: Mode badge ── */
  .mode-badge { display:inline-flex; align-items:center; gap:6px; font-size:.7rem;
    letter-spacing:.06em; text-transform:uppercase; font-weight:700; padding:3px 10px;
    border-radius:999px; margin-left:10px; vertical-align:middle; cursor:pointer;
    border:1px solid var(--rule); background:#fbf8f0; color:var(--ink-soft); }
  .mode-badge.morning { color:#7a5a16; border-color:#e6d6a8; background:#fbf3da; }
  .mode-badge.live { color:var(--accent); border-color:#bcd8d2; background:#eef6f4; }
  .mode-badge.endday { color:var(--clay); border-color:#e6cdb6; background:#fbf0e6; }

  /* ── Φάση 3: Sticky condensed stats banner (mobile) ── */
  .sticky-stats { display:none; position:sticky; top:0; z-index:50; background:var(--paper);
    border-bottom:1px solid var(--rule); padding:8px 14px; margin:0 -22px 12px;
    font-size:.78rem; gap:12px; align-items:center; }
  .sticky-stats b { font-size:.95rem; }

  /* ── Φάση 3: Mobile swipeable tabs ── */
  .mtabs { display:none; gap:6px; margin:14px 0; position:sticky; top:38px; z-index:40;
    background:var(--paper); padding:4px 0; }
  .mtab { flex:1; text-align:center; padding:10px 6px; border-radius:9px; font-size:.82rem;
    font-weight:600; color:var(--ink-soft); background:#fbf8f0; border:1px solid var(--rule);
    cursor:pointer; -webkit-tap-highlight-color:transparent; }
  .mtab.active { color:#fff; background:var(--accent); border-color:var(--accent); }
  .mtab .badge-count { display:inline-block; min-width:16px; padding:0 4px; margin-left:4px;
    border-radius:8px; background:var(--urgent); color:#fff; font-size:.68rem; line-height:1.3; }

  .mobile-summary-toggle, .mobile-summary-panel { display:none; }
  .mobile-summary-toggle {
    width:100%; align-items:center; justify-content:space-between; gap:10px;
    margin:6px 0 8px; padding:9px 12px; border:1px solid var(--rule);
    border-radius:10px; background:#fbf8f0; color:var(--accent);
    font-weight:700; font-size:.84rem; cursor:pointer;
  }
  .mobile-summary-panel {
    margin:0 0 10px; padding:10px 12px; border:1px solid var(--rule);
    border-radius:10px; background:var(--card); color:var(--ink-soft);
    font-size:.82rem;
  }
  .mobile-summary-panel.open { display:block; }
  .mobile-summary-kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin:8px 0; }
  .mobile-summary-kpi { border:1px solid var(--rule); border-radius:8px; padding:7px 6px; background:#fbf8f0; }
  .mobile-summary-kpi b { display:block; font-family:var(--serif); font-size:1.15rem; line-height:1; color:var(--ink); }
  .mobile-summary-kpi span { display:block; margin-top:3px; font-size:.62rem; letter-spacing:.07em; text-transform:uppercase; color:var(--ink-soft); }
  .mobile-health-line { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; font-size:.76rem; }

  @media (max-width:768px){
    .grid { grid-template-columns:1fr; }
  }
  @media (max-width:520px){
    .wrap{ padding:8px 16px 72px; }
    .masthead, .stats, #health-strip, #triage-wrap, #triage-panel{ display:none !important; }
    .when{ margin-left:0; width:100%; }
    .sticky-stats{ display:flex; margin:0 -16px 8px; padding:7px 16px; }
    .mobile-summary-toggle{ display:flex; }
    .mtabs{ display:flex; top:34px; margin:8px 0 10px; }
    .btn{ padding:11px 18px; font-size:.88rem; min-height:40px; }
    .chk, [onclick^="toggleTodo"]{ min-width:28px !important; height:28px !important; }
    .panel-mobile{ display:none; }
    .panel-mobile.active{ display:block; }
  }
  @media (prefers-reduced-motion:reduce){ .item.new{ animation:none; } }
</style>
</head>
<body>
<div class="wrap">
  <div class="sticky-stats" id="sticky-stats">
    <span><b id="sticky-total">0</b> κλήσεις</span>
    <span><b id="sticky-pending" style="color:var(--urgent)">0</b> εκκρεμούν</span>
    <span class="sp" style="flex:1"></span>
    <span class="dot" id="sticky-dot"></span>
  </div>

  <header class="masthead">
    <div class="eyebrow" id="brief-date">— <span class="mode-badge" id="mode-badge" onclick="cycleMode()">Live</span></div>
    <h1>Ενημέρωση <em>Κλήσεων</em></h1>
    <p class="dek" id="dek">Φόρτωση…</p>
    <p class="dek" id="insights-summary" style="margin-top:6px;font-style:normal;font-size:.92rem;color:var(--ink-soft)"></p>
  </header>

  <button class="mobile-summary-toggle" id="mobile-summary-toggle" onclick="toggleMobileSummary()">
    <span id="mobile-summary-title">Σύνοψη ημέρας</span><span id="mobile-summary-caret">▾</span>
  </button>
  <div class="mobile-summary-panel" id="mobile-summary-panel"></div>

  <div class="stats" id="stats"></div>
  <div id="health-strip" style="margin:10px 0 0;font-size:.78rem;color:var(--ink-soft);display:flex;gap:14px;flex-wrap:wrap"></div>
  <div id="triage-wrap" style="margin:12px 0">
    <button class="tbtn" id="triage-btn" onclick="runWeeklyTriage()">📋 Έλεγχος εβδομάδας</button>
  </div>
  <div id="triage-panel"></div>

  <div class="mtabs" id="mtabs">
    <div class="mtab active" data-panel="actions" onclick="switchMobileTab('actions')">Ενέργειες<span class="badge-count" id="mtab-actions-count" style="display:none">0</span></div>
    <div class="mtab" data-panel="schedule" onclick="switchMobileTab('schedule')">Πρόγραμμα</div>
    <div class="mtab" data-panel="calls" onclick="switchMobileTab('calls')">Κλήσεις</div>
  </div>

  <div id="content"><div class="empty">Φόρτωση…</div></div>

  <div class="toolbar">
    <span class="dot" id="status-dot"></span>
    <span>Ανανέωση: <span id="last-refresh">—</span></span>
    <span class="sp"></span>
    <button class="tbtn" id="sync-btn" onclick="runSync()">Συγχρονισμός</button>

    <button class="tbtn" onclick="toggleHeidiPicker()">Heidi CSV</button>
  </div>
  <div class="picker" id="heidi-picker">
    <input type="date" id="heidi-date">
    <button class="tbtn" onclick="downloadHeidiCSV()">Κατέβασμα</button>
  </div>
</div>

<script>
const KEY = new URLSearchParams(window.location.search).get('key') || '';
let lastIds = new Set(); let isFirst = true;

function fmtTime(iso){
  if(!iso) return '—';
  try { return new Date(iso).toLocaleString('el-GR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit',timeZone:'Asia/Nicosia'}); }
  catch(e){ return iso; }
}
function clinicGr(c){ return c==='limassol'?'Λεμεσός':(c==='evrychou'?'Ευρύχου':''); }

function toggleMobileSummary(){
  const panel = document.getElementById('mobile-summary-panel');
  const caret = document.getElementById('mobile-summary-caret');
  if(!panel) return;
  const willOpen = !panel.classList.contains('open');
  if(willOpen && typeof renderFromCache === 'function'){
    try{ renderFromCache(); }catch(e){}
  }
  panel.classList.toggle('open');
  const open = panel.classList.contains('open');
  if(caret) caret.textContent = open ? '▴' : '▾';
}

function updateMobileSummary(stats, messagesCount, bookingsCount, openCount){
  const panel = document.getElementById('mobile-summary-panel');
  if(!panel) return;
  const nowH = parseInt(new Date().toLocaleString('en',{hour:'numeric',hour12:false,timeZone:'Asia/Nicosia'}));
  const titleEl = document.getElementById('mobile-summary-title');
  if(titleEl) titleEl.textContent = nowH >= 19 ? 'Βραδινή σύνοψη ημέρας' : 'Σύνοψη ημέρας';
  const dekText = (document.getElementById('dek')?.innerText || '').trim();
  const insightsText = (document.getElementById('insights-summary')?.innerText || '').trim();
  const healthHtml = window._lastHealthCompactHtml || '';
  panel.innerHTML = `
    <div>${dekText}</div>
    ${insightsText ? `<div style="margin-top:5px">${insightsText}</div>` : ''}
    <div class="mobile-summary-kpis">
      <div class="mobile-summary-kpi"><b>${stats.total}</b><span>Κλήσεις</span></div>
      <div class="mobile-summary-kpi"><b style="color:var(--urgent)">${stats.urgent}</b><span>Επείγ.</span></div>
      <div class="mobile-summary-kpi"><b>${messagesCount}</b><span>Μηνύμ.</span></div>
      <div class="mobile-summary-kpi"><b>${bookingsCount}</b><span>Ραντ.</span></div>
    </div>
    <div><b style="color:var(--urgent)">${openCount}</b> ανοιχτές εκκρεμότητες</div>
    ${healthHtml ? `<div class="mobile-health-line">${healthHtml}</div>` : ''}
    <div style="margin-top:9px"><button class="tbtn" onclick="runWeeklyTriage(); toggleMobileSummary();">📋 Έλεγχος εβδομάδας</button></div>
  `;
}

// Καθαρό, πράσινο εικονίδιο τηλεφώνου (αντί για το σκούρο/μονόχρωμο emoji 📞
// που δεν διακρίνεται καλά σε μικρή οθόνη κινητού). Στρογγυλό πράσινο φόντο
// με λευκό ακουστικό-εικόνα, παραμετρικό μέγεθος ώστε να ταιριάζει σε κάθε
// σημείο χρήσης (κάρτες επειγόντων, χρονολόγιο, todoItem).
// Καθαρά εικονίδια τηλεφώνου/μηνύματος, ΧΩΡΙΣ δικό τους background κύκλο —
// μόνο το ίδιο το contour σχήμα, σε χρώμα currentColor. Έτσι κληρονομούν
// αυτόματα το χρώμα του κουμπιού που τα περιέχει (λευκό μέσα στο filled
// .btn.call, πράσινο μέσα στο outline .btn.sms) αντί να έχουν δικό τους
// ασύμβατο πράσινο κύκλο φόντο που δεν ταιριάζει με το background του ίδιου
// του κουμπιού.
function phoneIconSVG(sizePx){
  const s = sizePx || 18;
  return `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;vertical-align:middle">
    <path d="M7 3.5c-1.4 0-3 1.3-3 3 0 7.5 6 13.5 13.5 13.5 1.7 0 3-1.6 3-3 0-.5-.2-1-.6-1.3l-3-2.5c-.5-.4-1.2-.4-1.7 0l-1 .8c-.3.2-.7.2-1 0a11 11 0 0 1-4.2-4.2c-.2-.3-.2-.7 0-1l.8-1c.4-.5.4-1.2 0-1.7l-2.5-3c-.3-.4-.8-.6-1.3-.6z"
      fill="currentColor"/>
  </svg>`;
}

function messageIconSVG(sizePx){
  const s = sizePx || 18;
  return `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;vertical-align:middle">
    <path d="M4 5.5C4 4.4 4.9 3.5 6 3.5H18C19.1 3.5 20 4.4 20 5.5V14.5C20 15.6 19.1 16.5 18 16.5H10.5L6 20V16.5H6C4.9 16.5 4 15.6 4 14.5V5.5Z"
      stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" fill="none"/>
  </svg>`;
}

function telOf(call){
  const tel=(call.phone||'').replace(/[^+0-9]/g,'');
  return tel ? (tel.startsWith('+')?tel:'+357'+tel) : '';
}
function nameOf(call){ return [call.first_name,call.last_name].filter(Boolean).join(' '); }
function isUrgent(c){ return c.is_urgent || c.priority==='A'; }
function isBooking(c){ return c.type==='booking' || !!c.booking_uid; }
function isMessage(c){
  if(c.type==='message' || c.caller_wants_callback) return true;
  if(c.type==='booking' || c.type==='cancellation') return false;
  return !!(c.message && c.message.length > 10);
}
function isCallback(c){ return !!c.caller_wants_callback; }
function typeLabel(c){
  if(isBooking(c)) return 'Ραντεβού';
  if(c.type==='cancellation') return 'Ακύρωση';
  if(isCallback(c)) return 'Callback';
  if(isMessage(c)) return 'Μήνυμα';
  if(c.type==='appointment') return 'Ραντεβού (εξέλιξη)';
  if(c.incomplete) return 'Διεκόπη';
  return 'Κλήση';
}

const SMS = encodeURIComponent('Καλησπέρα, σας καλώ από το ιατρείο του Δρ. Παπαχρήστου σχετικά με το αίτημά σας.');

function actionCard(call,isNew){
  const tel=telOf(call); const nm=nameOf(call); const msg=call.display_message||call.message_clean||call.reason_for_call||call.reason||call.message||'';
  const tags=[
    isUrgent(call)?'<span class="tag u">Επείγον</span>':'',
    call.clinic?`<span class="tag c">${clinicGr(call.clinic)}</span>`:'',
    isCallback(call)?'<span class="tag u">Ζητά callback</span>':'',
    isMessage(call)&&!isCallback(call)?'<span class="tag i">Μήνυμα</span>':'',
    call.is_today===false?'<span class="tag y">Χθες</span>':'',
    call.incomplete?'<span class="tag">Διεκόπη</span>':'',
  ].filter(Boolean).join('');
  const acts=tel?`<div class="acts">
      <a class="btn call" href="tel:${tel}" style="display:inline-flex;align-items:center;gap:6px">${phoneIconSVG(18)} Κλήση</a>
      <a class="btn sms" href="sms:${tel}?&body=${SMS}" style="display:inline-flex;align-items:center;gap:6px">${messageIconSVG(16)} SMS</a></div>`:'';
  return `<div class="item${isUrgent(call)||isCallback(call)?' urgent':''}${isNew?' new':''}">
    <div class="item-top">
      <span class="who">${nm||'Άγνωστος'} <span class="ph">${call.phone||''}</span></span>
      <span class="when">${fmtTime(call.logged_at)}</span>
    </div>
    ${msg?`<div class="body">${msg}</div>`:''}
    <div class="meta-tags">${tags}</div>
    ${acts}
  </div>`;
}
function bookingRow(call){
  const nm=nameOf(call);
  const bkTime = call.booking_start_iso ? fmtTime(call.booking_start_iso) : '';
  const bkClinic = clinicGr(call.booking_clinic || call.clinic) || '';
  return `<div class="bk">
    <span class="who">${nm||'Ραντεβού'} <span class="ph">${call.phone||''}</span></span>
    <span class="tag c">${bkClinic||'—'}</span>
    ${bkTime?`<span class="tag b">📅 ${bkTime}</span>`:''}
    <span class="when">${fmtTime(call.logged_at)}</span>
  </div>`;
}
function timelineRow(call, idx){
  const nm=nameOf(call); const msg=call.display_message||call.message_clean||call.reason_for_call||call.reason||call.message||'';
  const cls=isUrgent(call)?'u':(isBooking(call)?'b':'');
  const typeLbl=typeLabel(call);
  // ΣΗΜΑΝΤΙΚΟ: χρησιμοποιούμε το ΙΔΙΟ todoId() helper που χρησιμοποιεί και το
  // todoItem() (Εκκρεμότητες). Πριν, το Χρονολόγιο έφτιαχνε δικό του ID με
  // πρόθεμα "tl_" ενώ το todoItem χρησιμοποιούσε "td_" — δύο διαφορετικά
  // keys για την ΙΔΙΑ κλήση. Αποτέλεσμα: το φίλτρο «χθεσινές τικαρισμένες
  // κρύβονται» (στο renderFromCache, που ελέγχει μέσω todoId()) ταίριαζε
  // σωστά στις Εκκρεμότητες αλλά ΠΟΤΕ στο Χρονολόγιο — οι χθεσινές
  // τικαρισμένες παρέμεναν ορατές εκεί. Με κοινό ID, το tick σε ΟΠΟΙΟ
  // σημείο αντιστοιχεί πάντα στην ΙΔΙΑ εγγραφή κατάστασης.
  const tid = todoId(call);
  const done=!!todoDone[tid];
  const disp=call.incomplete?'📵 Διεκόπη':'';
  const ph=call.phone||'';
  const label = nm||ph||disp||'Άγνωστος';
  const repeat = call._repeat ? '<span style="font-size:.65rem;background:#fef3c7;color:#92400e;padding:1px 6px;border-radius:9px;margin-left:6px">🔁 επανάληψη</span>' : '';
  const tel = telOf(call);
  const callBtn = tel ? `<a class="btn call" href="tel:${tel}" style="font-size:.7rem;padding:3px 8px;margin-left:8px;flex-shrink:0;display:inline-flex;align-items:center" onclick="event.stopPropagation()">${phoneIconSVG(16)}</a>
    <a class="btn sms" href="sms:${tel}?&body=${SMS}" style="font-size:.7rem;padding:3px 8px;margin-left:4px;flex-shrink:0;display:inline-flex;align-items:center" onclick="event.stopPropagation()">${messageIconSVG(14)}</a>` : '';
  return `<div class="row ${cls}" style="${done?'opacity:.4':'opacity:1'}">
    <div onclick="toggleTodo('${tid}',this)" style="cursor:pointer;min-width:22px;height:22px;border:2px solid ${done?'var(--accent)':'#c5bfaf'};border-radius:4px;display:flex;align-items:center;justify-content:center;background:${done?'var(--accent)':'#fff'};margin-top:2px;flex-shrink:0;transition:.15s">${done?'<span style="color:#fff;font-size:13px">✓</span>':''}</div>
    <span class="rtime">${fmtTime(call.logged_at)}<br><span style="font-size:.66rem;color:var(--ink-soft)">${typeLbl}</span></span>
    <span class="rmain" style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px">
      <span style="flex:1;min-width:0">
        <span class="rwho" style="${done?'text-decoration:line-through;color:var(--ink-soft)':'color:var(--ink)'}">${label}${ph&&nm?` <span class="ph">${ph}</span>`:''}${repeat}</span>
        ${msg?`<div class="rmsg" style="${done?'text-decoration:line-through':''}">${msg}</div>`:''}
      </span>
      ${callBtn}
    </span>
  </div>`;
}

function buildDek(s){
  if(!s.total) return 'Καμία κλήση σήμερα ή χθες ακόμη. Μόλις ο ψηφιακός γραμματέας απαντήσει κλήση, θα εμφανιστεί εδώ.';
  const parts=[];
  parts.push(`Ο ψηφιακός γραμματέας διαχειρίστηκε <b>${s.total} ${s.total===1?'κλήση':'κλήσεις'}</b>`);
  const tail=[];
  if(s.action) tail.push(`${s.action} ${s.action===1?'χρειάζεται':'χρειάζονται'} ενέργεια`);
  if(s.booking) tail.push(`${s.booking} ${s.booking===1?'ραντεβού':'ραντεβού'}`);
  let t = parts[0];
  if(tail.length) t += ' — ' + tail.join(', ');
  return t + '.';
}

async function runSync(){
  const b=document.getElementById('sync-btn'); b.textContent='…'; b.disabled=true;
  try{ const r=await fetch('/sync/both'+(KEY?'?key='+KEY:'')); const d=await r.json();
    b.textContent=d.status==='ok'?'OK':'Μερικό'; setTimeout(()=>{b.textContent='Συγχρονισμός';b.disabled=false;},3500); fetchCalls();
  }catch(e){ b.textContent='Σφάλμα'; b.disabled=false; }
}
function toggleHeidiPicker(){
  const p=document.getElementById('heidi-picker'); const vis=p.style.display==='flex';
  p.style.display=vis?'none':'flex';
  if(!vis){ const t=new Date(); t.setDate(t.getDate()+1);
    const ds=`${t.getFullYear()}-${String(t.getMonth()+1).padStart(2,'0')}-${String(t.getDate()).padStart(2,'0')}`;
    const i=document.getElementById('heidi-date'); if(i&&!i.value) i.value=ds; }
}
function downloadHeidiCSV(){
  const i=document.getElementById('heidi-date'); const di=i?i.value:'';
  if(!di){ alert('Επίλεξε ημερομηνία.'); return; }
  document.getElementById('heidi-picker').style.display='none';
  window.location.href=`/dashboard/heidi-csv?date=${di}`+(KEY?`&key=${KEY}`:'');
}

let todoDone = {};

function todoId(call,i){ return 'td_'+(call.call_id || ((call.phone||'')+'@'+(call.logged_at||'').slice(0,16))); }

function todoItem(call, idx){
  const id = todoId(call,idx);
  const done = !!todoDone[id];
  const tel = telOf(call); const nm = nameOf(call);
  const msg = call.display_message||call.message_clean||call.reason_for_call||call.reason||call.message||'';
  const tags = [
    isUrgent(call)?'<span class="tag u">Επείγον</span>':'',
    isCallback(call)?'<span class="tag u">Callback</span>':'',
    call.clinic?`<span class="tag c">${clinicGr(call.clinic)}</span>`:'',
  ].filter(Boolean).join(' ');
  const acts = tel ? `<a class="btn call" href="tel:${tel}" style="font-size:.75rem;padding:4px 10px;display:inline-flex;align-items:center" onclick="event.stopPropagation()">${phoneIconSVG(17)}</a>
    <a class="btn sms" href="sms:${tel}?&body=${SMS}" style="font-size:.75rem;padding:4px 10px;display:inline-flex;align-items:center" onclick="event.stopPropagation()">${messageIconSVG(15)}</a>` : '';
  return `<div class="item${done?' done':''}" style="${done?'opacity:.55;':''}">
    <div style="display:flex;align-items:flex-start;gap:10px">
      <div class="chk${done?' checked':''}" onclick="toggleTodo('${id}',this)" style="cursor:pointer;min-width:22px;height:22px;border:2px solid ${done?'var(--accent)':'var(--rule)'};border-radius:5px;margin-top:2px;display:flex;align-items:center;justify-content:center;background:${done?'var(--accent)':'transparent'};transition:.2s">${done?'<span style="color:#fff;font-size:14px">✓</span>':''}</div>
      <div style="flex:1">
        <div class="item-top">
          <span class="who">${nm||'Ασθενής'} <span class="ph">${call.phone||''}</span></span>
          <span class="when">${fmtTime(call.logged_at)}</span>
        </div>
        ${msg?`<div class="body" style="${done?'text-decoration:line-through':''}">${msg}</div>`:''}
        <div style="display:flex;align-items:center;gap:8px;margin-top:6px">${tags} ${acts}</div>
      </div>
    </div>
  </div>`;
}

async function toggleTodo(id, el){
  // Optimistic local update — απλά αλλάζει την κλάση του ίδιου checkbox,
  // ΔΕΝ ξανακαλεί fetchCalls()/fetchSchedule() (αυτό θα έκανε νέο live
  // αίτημα στο Cal.com API σε κάθε κλικ — η βασική αιτία της υπερφόρτωσης).
  if(todoDone[id]) delete todoDone[id]; else todoDone[id]=true;
  if(typeof renderFromCache === 'function') renderFromCache();
  try{
    await fetch('/dashboard/todos/toggle'+(KEY?'?key='+KEY:''), {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id})
    });
    // ΣΗΜΑΝΤΙΚΟ: το /dashboard/anomalies υπολογίζει «callback εκκρεμεί» με
    // βάση το αν η εκκρεμότητα έχει τικαριστεί (todoDone) — αν δεν το
    // ξανακαλέσουμε εδώ, το anomaly banner δείχνει «εκκρεμεί» ακόμα και
    // μετά που ο γιατρός το τικάρει, μέχρι το επόμενο 45-second poll.
    // Είναι ασφαλές να το ξανακαλέσουμε σε κάθε tick: διαβάζει μόνο
    // in-memory _CALL_LOG + το ήδη-ενημερωμένο todoDone από disk, καμία
    // εξωτερική κλήση σε Cal.com/Setmore — άρα δεν ξαναφέρνει το
    // πρόβλημα υπερφόρτωσης που είχαμε με τα fetchCalls/fetchSchedule.
    if(typeof fetchAnomalies === 'function') fetchAnomalies();
  }catch(e){ console.error('toggle sync error:',e); }
}

let scheduleOffset = 0; // 0=σήμερα, 1=αύριο

function scheduleRow(a){
  const c = clinicGr(a.clinic)||'';
  const past = a.is_past;
  return `<div class="bk" style="${past?'opacity:.45;text-decoration:line-through':''}">
    <span style="font-family:var(--mono);font-size:.88rem;min-width:48px;font-weight:600;color:${past?'var(--ink-soft)':'var(--accent)'}">${a.time||'—'}</span>
    <span class="who">${a.name||'—'} <span class="ph">${a.phone||''}</span></span>
    ${c?`<span class="tag c">${c}</span>`:''}
    <span class="tag" style="margin-left:auto">${a.duration||30} λεπ</span>
  </div>`;
}

function switchDay(delta){
  scheduleOffset = Math.max(0, Math.min(1, scheduleOffset+delta));
  fetchSchedule();
}

async function fetchSchedule(){
  try{
    // Υπολόγισε target date
    const d = new Date();
    d.setDate(d.getDate() + scheduleOffset);
    const ds = d.toISOString().slice(0,10);
    const r = await fetch('/dashboard/schedule?target_date='+ds+(KEY?'&key='+KEY:''));
    if(!r.ok) return;
    const data = await r.json();
    const el = document.getElementById('schedule-panel');
    if(!el) return;
    const label = data.is_today ? 'Σημερινό Πρόγραμμα' : 'Αυριανό Πρόγραμμα';
    const arrows = `<span style="display:inline-flex;gap:4px;margin-left:8px">
      <button onclick="switchDay(-1)" class="tbtn" style="padding:2px 8px;font-size:.8rem" ${scheduleOffset<=0?'disabled':''}>◀</button>
      <button onclick="switchDay(1)" class="tbtn" style="padding:2px 8px;font-size:.8rem" ${scheduleOffset>=1?'disabled':''}>▶</button></span>`;
    if(!data.appointments||!data.appointments.length){
      el.innerHTML=`<div class="section"><div class="section-head"><h2>${label}</h2>${arrows}</div><div class="empty">Δεν βρέθηκαν ραντεβού (${data.weekday||''}).</div></div>`;
      return;
    }
    const clinic = data.clinic||'';
    const past = data.appointments.filter(a=>a.is_past).length;
    const active = data.total - past;
    el.innerHTML=`<div class="section"><div class="section-head"><h2>${label}</h2>${arrows}<span class="count">${data.weekday} — ${clinic} — ${active} εκκρεμούν${past?' / '+past+' ολοκληρώθηκαν':''}</span></div>`
      + data.appointments.map(scheduleRow).join('') + `</div>`;
  }catch(e){}
}

let lastCalls = [];

// ── Φάση 3: Morning / Live / End-of-day modes ──────────────────────────
// Morning (πριν τις 09:00): briefing-style — η σύνοψη ημέρας είναι το
// πρώτο πράγμα που βλέπει ο γιατρός. Live (09:00-19:00): ό,τι κάνει σήμερα,
// δηλαδή real-time monitor. End-of-day (μετά τις 19:00): wrap-up. Ο γιατρός
// μπορεί να πατήσει το badge για να κάνει manual override — το mode τότε
// μένει «κλειδωμένο» μέχρι να ξανανοίξει τη σελίδα.
let _modeOverride = null;

function detectMode(){
  if(_modeOverride) return _modeOverride;
  const h = parseInt(new Date().toLocaleString('en',{hour:'numeric',hour12:false,timeZone:'Asia/Nicosia'}));
  if(h < 9) return 'morning';
  if(h >= 19) return 'endday';
  return 'live';
}

function cycleMode(){
  const order = ['live','morning','endday'];
  const current = detectMode();
  const next = order[(order.indexOf(current)+1) % order.length];
  _modeOverride = next;
  updateModeBadge();
}

function updateModeBadge(){
  const mode = detectMode();
  const el = document.getElementById('mode-badge');
  if(!el) return;
  el.className = 'mode-badge ' + mode;
  el.textContent = mode==='morning' ? 'Πρωινό Briefing' : (mode==='endday' ? 'Απολογισμός Ημέρας' : 'Live');
  // Στο morning/endday mode, η σύνοψη ημέρας προτάσσεται — sticky προτεραιότητα στο insights-summary
  const insightsEl = document.getElementById('insights-summary');
  if(insightsEl){
    insightsEl.style.fontWeight = (mode!=='live') ? '600' : 'normal';
    insightsEl.style.fontSize = (mode!=='live') ? '1.02rem' : '.92rem';
  }
}

let _currentMobilePanel = 'actions';

function switchMobileTab(panel){
  _currentMobilePanel = panel;
  document.querySelectorAll('.mtab').forEach(t=>t.classList.toggle('active', t.dataset.panel===panel));
  document.querySelectorAll('.panel-mobile').forEach(p=>p.classList.toggle('active', p.dataset.panel===panel));
}

function _restoreMobilePanel(){
  // ΣΗΜΑΝΤΙΚΟ: το renderFromCache() ξαναχτίζει ΟΛΟΚΛΗΡΟ το grid HTML από την
  // αρχή — κάθε φορά (π.χ. σε κάθε tick ενός checkbox) το νέο HTML έχει
  // πάντα hardcoded "active" στο πρώτο tab (Ενέργειες). Χωρίς αυτή τη
  // συνάρτηση, ο χρήστης σε κινητό που βρίσκεται στο tab "Κλήσεις" θα
  // πετάγεται πίσω στο "Ενέργειες" σε κάθε refresh/tick. Επαναφέρουμε το
  // tab που ήταν ενεργό ΠΡΙΝ το rebuild, χωρίς να αλλάξουμε το _currentMobilePanel.
  if(_currentMobilePanel === 'actions') return;  // ήδη το default, καμία αλλαγή χρειάζεται
  document.querySelectorAll('.mtab').forEach(t=>t.classList.toggle('active', t.dataset.panel===_currentMobilePanel));
  document.querySelectorAll('.panel-mobile').forEach(p=>p.classList.toggle('active', p.dataset.panel===_currentMobilePanel));
}

// ── Φάση 3: Pull-to-refresh (mobile) ───────────────────────────────────
(function setupPullToRefresh(){
  let startY = null; let pulling = false;
  document.addEventListener('touchstart', (e)=>{
    if(window.scrollY===0) startY = e.touches[0].clientY;
  }, {passive:true});
  document.addEventListener('touchmove', (e)=>{
    if(startY===null) return;
    const dy = e.touches[0].clientY - startY;
    if(dy > 80 && !pulling){
      pulling = true;
      const dot = document.getElementById('status-dot');
      if(dot) dot.style.background = '#3b82f6';
    }
  }, {passive:true});
  document.addEventListener('touchend', ()=>{
    if(pulling){ fetchCalls(); }
    startY = null; pulling = false;
  }, {passive:true});
})();



const DASHBOARD_CALL_REFRESH_MS = 600000;      // εργάσιμες ώρες: 10 λεπτά
const DASHBOARD_CALL_REFRESH_OFF_HOURS_MS = 1200000; // 18:00-07:30: 20 λεπτά
const DASHBOARD_SCHEDULE_REFRESH_MS = 600000;  // πρόγραμμα: 10 λεπτά
const DASHBOARD_INSIGHTS_REFRESH_MS = 600000;  // analytics/anomalies/features: 10 λεπτά
const DASHBOARD_HEALTH_REFRESH_MS = 600000;    // health strip: 10 λεπτά

let _fetchCallsInFlight = false;
let _lastScheduleFetchAt = 0;
let _lastInsightsFetchAt = 0;
let _lastHealthFetchAt = 0;
let _lastAnomaliesFetchAt = 0;
let _lastFeatureRequestsFetchAt = 0;

function _due(lastTs, intervalMs){
  return !lastTs || (Date.now() - lastTs) >= intervalMs;
}

function _localNicosiaMinutes(){
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Nicosia',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false
  }).formatToParts(new Date());
  const h = parseInt(parts.find(p => p.type === 'hour')?.value || '0', 10);
  const m = parseInt(parts.find(p => p.type === 'minute')?.value || '0', 10);
  return h * 60 + m;
}

function _isReceptionOffHours(){
  const mins = _localNicosiaMinutes();
  return mins >= 18 * 60 || mins < (7 * 60 + 30);
}

function _nextCallRefreshMs(){
  return _isReceptionOffHours() ? DASHBOARD_CALL_REFRESH_OFF_HOURS_MS : DASHBOARD_CALL_REFRESH_MS;
}

function _scheduleRefreshAllowed(){
  return !_isReceptionOffHours();
}

async function fetchCalls(){
  // ΑΣΦΑΛΕΙΑ: αν ο server αργήσει (π.χ. λόγω φόρτου από πραγματική κλήση),
  // ένα νέο setInterval tick ή ένα visibilitychange event μπορεί να
  // ξεκινήσει ΠΡΙΝ τελειώσει το προηγούμενο fetchCalls(). Χωρίς αυτό το
  // guard, τα αποτελέσματα μπορούν να φτάσουν εκτός σειράς (π.χ. ένα
  // αργό, παλιό αίτημα να αντικαταστήσει με stale δεδομένα ένα πιο
  // πρόσφατο, γρήγορο αίτημα που είχε ήδη ολοκληρωθεί).
  if(_fetchCallsInFlight) return;
  _fetchCallsInFlight = true;
  try{
    const [callsR, todoR] = await Promise.all([
      fetch('/dashboard/calls'+(KEY?'?key='+KEY:'')),
      fetch('/dashboard/todos/state'+(KEY?'?key='+KEY:''))
    ]);
    if(!callsR.ok){ document.getElementById('status-dot').style.background='#ef4444'; return; }
    const d=await callsR.json(); const calls=d.calls||[];
    try{ todoDone = await todoR.json(); }catch(e){ todoDone={}; }
    lastCalls = calls;
    document.getElementById('status-dot').style.background='#22c55e';
    document.getElementById('last-refresh').textContent=new Date().toLocaleTimeString('el-GR',{timeZone:'Asia/Nicosia'});
    document.getElementById('brief-date').textContent=new Date().toLocaleDateString('el-GR',{weekday:'long',day:'numeric',month:'long',year:'numeric',timeZone:'Asia/Nicosia'});

    const curIds=new Set(calls.map((c,i)=>(c.logged_at||'')+'_'+i));
    const newIds=isFirst?new Set():new Set([...curIds].filter(id=>!lastIds.has(id)));
    lastIds=curIds; isFirst=false;

    if(newIds.size>0){
      const n = calls[0];
      const isUrgentCall = isUrgent(n);
      const isBookingCall = isBooking(n);
      const hourNow = parseInt(new Date().toLocaleString('en',{hour:'numeric',hour12:false,timeZone:'Asia/Nicosia'}));
      const quietHours = hourNow >= 19 || hourNow < 7;
      const playSound = !quietHours || isUrgentCall; // σίγαση μετά τις 19:00, εκτός priority A/urgent

      if(playSound){
        try{
          const ctx=new (window.AudioContext||window.webkitAudioContext)();
          const beep=(freq,gain,delay,dur)=>setTimeout(()=>{
            const o=ctx.createOscillator(); const g=ctx.createGain();
            o.connect(g); g.connect(ctx.destination);
            o.frequency.value=freq; g.gain.value=gain;
            o.start(); o.stop(ctx.currentTime+dur);
          },delay);
          if(isUrgentCall){
            // Επείγον: 2 έντονα beeps
            beep(1000,0.2,0,0.18); beep(1000,0.2,260,0.18);
          } else if(isBookingCall){
            // Ραντεβού κλεισμένο: ευχάριστο ανεβαστικό chime
            beep(880,0.12,0,0.14); beep(1175,0.12,140,0.16);
          } else {
            // Κανονική κλήση/μήνυμα: 1 διακριτικό beep
            beep(880,0.13,0,0.16);
          }
        }catch(e){}
      }

      // Vibration (mobile, κυρίως για επείγοντα)
      if('vibrate' in navigator && isUrgentCall){
        navigator.vibrate([120,60,120]);
      }

      // Browser notification (ανεξάρτητο από quiet hours — ο γιατρός βλέπει το badge όταν θέλει)
      if('Notification' in window && Notification.permission==='granted'){
        const prefix = isUrgentCall ? '🚨 Επείγον — ' : (isBookingCall ? '✅ Νέο ραντεβού — ' : 'Νέα κλήση — ');
        new Notification(prefix+(n.phone||''),{body:(n.display_message||n.message_clean||n.reason||n.message||'').slice(0,100),tag:'ortho-call'});
      }
    }

    renderFromCache();

    // After 19:00 show tomorrow by default (μόνο στο πρώτο load, όχι σε κάθε refresh)
    if(isFirstScheduleLoad){
      const nowH=new Date().toLocaleString('en',{hour:'numeric',hour12:false,timeZone:'Asia/Nicosia'});
      if(parseInt(nowH)>=19) scheduleOffset=1;
      isFirstScheduleLoad=false;
      _lastScheduleFetchAt = Date.now();
      await fetchSchedule();
    }
    else if(_scheduleRefreshAllowed() && _due(_lastScheduleFetchAt, DASHBOARD_SCHEDULE_REFRESH_MS)){
      _lastScheduleFetchAt = Date.now();
      await fetchSchedule();
    }
    if(_due(_lastInsightsFetchAt, DASHBOARD_INSIGHTS_REFRESH_MS)){
      _lastInsightsFetchAt = Date.now();
      fetchInsights();
    }
    if(_due(_lastHealthFetchAt, DASHBOARD_HEALTH_REFRESH_MS)){
      _lastHealthFetchAt = Date.now();
      fetchHealth();
    }
    if(_due(_lastAnomaliesFetchAt, DASHBOARD_INSIGHTS_REFRESH_MS)){
      _lastAnomaliesFetchAt = Date.now();
      fetchAnomalies();
    }
    if(_due(_lastFeatureRequestsFetchAt, DASHBOARD_INSIGHTS_REFRESH_MS)){
      _lastFeatureRequestsFetchAt = Date.now();
      fetchFeatureRequests();
    }
    updateModeBadge();
  }catch(e){ document.getElementById('status-dot').style.background='#ef4444'; }
  finally{ _fetchCallsInFlight = false; }
}

let isFirstScheduleLoad = true;

async function fetchInsights(){
  try{
    const r = await fetch('/dashboard/insights'+(KEY?'?key='+KEY:''));
    if(!r.ok) return;
    const d = await r.json();
    const el = document.getElementById('insights-summary');
    if(!el) return;
    let html = d.summary_text || '';
    if(d.peak_hour !== null && d.peak_hour !== undefined && d.total_calls_today > 2){
      html += ` Περισσότερες κλήσεις γύρω στις ${String(d.peak_hour).padStart(2,'0')}:00.`;
    }
    if(d.repeat_callers_today && d.repeat_callers_today.length){
      const top = d.repeat_callers_today[0];
      html += ` 🔁 ${top.phone} κάλεσε ${top.count} φορές σήμερα.`;
    }
    if(d.completion && d.completion.completion_rate_pct !== null && d.completion.completion_rate_pct !== undefined){
      html += ` Τις τελευταίες 7 μέρες: ${d.completion.completion_rate_pct}% των κλήσεων ολοκληρώθηκαν επιτυχώς.`;
    }
    el.innerHTML = html;
  }catch(e){}
}

async function fetchHealth(){
  try{
    const r = await fetch('/dashboard/health'+(KEY?'?key='+KEY:''));
    if(!r.ok) return;
    const d = await r.json();
    const el = document.getElementById('health-strip');
    if(!el) return;
    const dot = (ok) => ok===true ? '🟢' : (ok===false ? '🔴' : '⚪');
    const cal = d.cal_com||{}; const set = d.setmore||{};
    let html = '';
    if(cal.uptime_pct !== null && cal.uptime_pct !== undefined){
      html += `<span>${dot(cal.currently_ok)} Cal.com: ${cal.uptime_pct}% uptime (24ω)${cal.avg_latency_ms?', '+cal.avg_latency_ms+'ms':''}</span>`;
    }
    if(set.uptime_pct !== null && set.uptime_pct !== undefined){
      html += `<span>${dot(set.currently_ok)} Setmore: ${set.uptime_pct}% uptime (24ω)${set.avg_latency_ms?', '+set.avg_latency_ms+'ms':''}</span>`;
    }
    el.innerHTML = html;
    window._lastHealthCompactHtml = html;
  }catch(e){}
}

async function fetchAnomalies(){
  try{
    const r = await fetch('/dashboard/anomalies'+(KEY?'?key='+KEY:''));
    if(!r.ok) return;
    const d = await r.json();
    let el = document.getElementById('anomaly-banner');
    const highOrMedium = (d.anomalies||[]).filter(a => a.severity==='high' || a.severity==='medium');
    if(!highOrMedium.length){
      if(el) el.remove();
      return;
    }
    if(!el){
      el = document.createElement('div');
      el.id = 'anomaly-banner';
      el.style.cssText = 'margin:10px 0;padding:10px 14px;border-radius:8px;background:#fef2f2;border:1px solid #fecaca;color:#991b1b;font-size:.85rem';
      const statsEl = document.getElementById('stats');
      if(statsEl && statsEl.parentNode) statsEl.parentNode.insertBefore(el, statsEl);
    }
    el.innerHTML = '⚠️ ' + highOrMedium.map(a => a.message).join(' · ');
  }catch(e){}
}

async function fetchFeatureRequests(){
  try{
    const r = await fetch('/dashboard/feature_requests'+(KEY?'?key='+KEY:''));
    if(!r.ok) return;
    const d = await r.json();
    const el = document.getElementById('feature-requests-panel');
    if(!el) return;
    if(!d.requests || !d.requests.length){ el.innerHTML=''; return; }
    const catLabels = {
      video_call:'Βιντεοκλήση/Τηλεϊατρική', sms_reminder:'SMS Υπενθυμίσεις',
      online_booking:'Online κράτηση', payment_online:'Online πληρωμή',
      multilingual:'Άλλη γλώσσα', weekend_hours:'Σαββατοκύριακο ωράριο',
      evening_hours:'Βραδινό ωράριο', results_followup:'Αποτελέσματα εξετάσεων online'
    };
    let html = `<div class="section"><div class="section-head"><h2>Αιτήματα Ασθενών (${d.period_days}ημ)</h2><span class="count">${d.categories_found}</span></div>`;
    html += d.requests.map(r => `<div class="bk"><span class="who">${catLabels[r.category]||r.category}</span><span class="tag" style="margin-left:auto">${r.count}×</span></div>`).join('');
    html += `</div>`;
    el.innerHTML = html;
  }catch(e){}
}

async function runWeeklyTriage(){
  // On-demand (όχι σε κάθε auto-refresh) — η εβδομαδιαία ανασκόπηση δεν
  // χρειάζεται να φορτώνει κάθε 45 δευτερόλεπτα, μόνο όταν ο γιατρός
  // πατήσει το κουμπί στο τέλος της εβδομάδας.
  const btn = document.getElementById('triage-btn');
  const panel = document.getElementById('triage-panel');
  btn.textContent = '⏳ Έλεγχος...'; btn.disabled = true;
  try{
    const r = await fetch('/dashboard/weekly_triage'+(KEY?'?key='+KEY:''));
    const d = await r.json();
    if(!r.ok){ panel.innerHTML = `<div class="empty">Σφάλμα: ${d.error||'άγνωστο'}</div>`; return; }

    const sevColor = {failed:'#ef4444', needs_review:'#f59e0b', succeeded:'#22c55e'};
    const sevLabel = {failed:'🔴 Αποτυχημένες', needs_review:'🟡 Χρειάζονται έλεγχο', succeeded:'🟢 Πέτυχαν'};
    const intentOptions = ['appointment','message','confirmation','change','cancellation','urgent','other'];

    let html = `<div class="section"><div class="section-head"><h2>Έλεγχος ${d.period_days} ημερών</h2>
      <span class="count">${d.total_calls} κλήσεις${d.intent_completion_rate_pct!==null?' · '+d.intent_completion_rate_pct+'% ολοκλήρωση intent':''}</span></div>`;
    const sc = d.scorecard || {};
    const metric = (label, value, suffix='') =>
      `<div class="stat"><div class="n">${value===null||value===undefined?'—':value+suffix}</div><div class="l">${label}</div></div>`;
    html += `<div class="stats" style="margin:8px 0 10px">
      ${metric('Intent accuracy', sc.intent_accuracy_pct, '%')}
      ${metric('Intent completion', sc.intent_completion_rate_pct, '%')}
      ${metric('Wrong route', sc.wrong_route_rate_pct, '%')}
      ${metric('P95 reasoning', sc.p95_reasoning_total_ms, 'ms')}
      ${metric('Stuck rate', sc.stuck_rate_pct, '%')}
    </div>`;
    if(!sc.intent_accuracy_labeled_calls){
      html += `<div style="font-size:.75rem;color:var(--ink-soft);margin-bottom:8px">
        Intent accuracy: δεν υπάρχουν ακόμη human/golden labels. Δεν υπολογίζεται από την πρόβλεψη του ίδιου του agent.
      </div>`;
    }
    const perIntent = sc.per_intent || {};
    if(Object.keys(perIntent).length){
      html += `<div style="font-size:.78rem;color:var(--ink-soft);margin-bottom:8px">`+
        Object.entries(perIntent).map(([name,v]) =>
          `${name}: ${v.completed}/${v.calls}${v.completion_rate_pct!==null?' ('+v.completion_rate_pct+'%)':''}`
        ).join(' · ')+`</div>`;
    }
    if(d.latency && d.latency.calls_with_audit){
      const avg = d.latency.avg_reasoning_total_ms;
      const p95 = d.latency.p95_reasoning_total_ms;
      const max = d.latency.max_reasoning_total_ms;
      html += `<div style="font-size:.78rem;color:var(--ink-soft);margin:4px 0 10px">
        Latency audit: ${d.latency.calls_with_audit} κλήσεις${avg!==null?' · μέσο '+avg+'ms':''}${p95!==null?' · P95 '+p95+'ms':''}${max!==null?' · μέγιστο '+max+'ms':''}
      </div>`;
    }

    for(const group of ['failed','needs_review','succeeded']){
      const items = d[group] || [];
      if(!items.length) continue;
      html += `<div style="margin:10px 0">
        <div style="font-weight:700;color:${sevColor[group]};font-size:.85rem;margin-bottom:6px">${sevLabel[group]} (${items.length})</div>`;
      html += items.slice(0,20).map(c => {
        const tel = telOf(c);
        const callBtn = tel ? `<a class="btn call" href="tel:${tel}" style="font-size:.7rem;padding:3px 8px;margin-left:8px;display:inline-flex;align-items:center">${phoneIconSVG(15)}</a>` : '';
        const reasons = (c._triage_reasons||[]).map(r=>`<div style="font-size:.78rem;color:var(--ink-soft)">· ${r}</div>`).join('');
        const labelControl = c.call_id ? `<div style="margin-top:6px;display:flex;align-items:center;gap:6px">
          <span style="font-size:.72rem;color:var(--ink-soft)">Σωστό intent:</span>
          <select onchange="labelCallIntent('${c.call_id}',this)" style="font-size:.72rem;padding:3px 5px;border-radius:6px">
            <option value="">— επιλέξτε —</option>
            ${intentOptions.map(i=>`<option value="${i}" ${c._expected_intent===i?'selected':''}>${i}</option>`).join('')}
          </select>
          <span class="intent-label-status" style="font-size:.7rem;color:var(--ink-soft)"></span>
        </div>` : '';
        return `<div class="bk" style="display:block;border-left:3px solid ${sevColor[group]};padding-left:10px">
          <div style="display:flex;align-items:center;justify-content:space-between">
            <span class="who">${nameOf(c)||c.phone||'Άγνωστος'}</span>
            <span style="display:flex;align-items:center">${fmtTime(c.logged_at)}${callBtn}</span>
          </div>
          ${reasons}
          ${labelControl}
        </div>`;
      }).join('');
      if(items.length>20) html += `<div style="font-size:.78rem;color:var(--ink-soft);margin-top:4px">...και ${items.length-20} ακόμη</div>`;
      html += `</div>`;
    }
    html += `</div>`;
    panel.innerHTML = html;
  }catch(e){
    panel.innerHTML = `<div class="empty">Σφάλμα φόρτωσης ελέγχου.</div>`;
  }finally{
    btn.textContent = '📋 Έλεγχος εβδομάδας'; btn.disabled = false;
  }
}

async function labelCallIntent(callId, select){
  const expectedIntent = select.value;
  const status = select.parentElement.querySelector('.intent-label-status');
  if(!expectedIntent) return;
  select.disabled = true;
  status.textContent = 'Αποθήκευση...';
  try{
    const url = '/dashboard/call_audits/label'+(KEY?'?key='+KEY:'');
    const r = await fetch(url,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({call_id:callId,expected_intent:expectedIntent,reviewer:'dashboard'})
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.error||'label failed');
    status.textContent = 'Αποθηκεύτηκε';
  }catch(e){
    status.textContent = 'Σφάλμα';
  }finally{
    select.disabled = false;
  }
}


function renderFromCache(){
  // Ξαναζωγραφίζει ΟΛΟΚΛΗΡΟ το UI (εκκρεμότητες/ραντεβού/χρονολόγιο) από τα
  // ήδη-φορτωμένα δεδομένα (lastCalls), ΧΩΡΙΣ κανένα network call. Καλείται
  // είτε μετά από fetchCalls() είτε μετά από ένα τοπικό checkbox click.
  //
  // ΦΙΛΤΡΟ: οι ΧΘΕΣΙΝΕΣ κλήσεις που έχουν ήδη τικαριστεί (todoDone) δεν
  // εμφανίζονται καθόλου σήμερα — ούτε στα Επείγοντα/Εκκρεμότητες, ούτε στο
  // Χρονολόγιο, ούτε στα στατιστικά. Οι ΣΗΜΕΡΙΝΕΣ τικαρισμένες παραμένουν
  // ορατές (απλά ξεθωριασμένες/με γραμμή) όπως πριν — ο γιατρός θέλει να
  // βλέπει τι έγινε σήμερα μέχρι τώρα, αλλά όχι να ξαναβλέπει χθεσινά
  // θέματα που έχει ήδη κλείσει.
  const calls = lastCalls.filter(c => {
    if(c.is_today === false){
      const id = todoId(c);
      if(todoDone[id]) return false;  // χθεσινή + τικαρισμένη -> κρύβεται
    }
    return true;
  });

  const urgent=[], messages=[], bookings=[], other=[];
  calls.forEach(c=>{ if(isUrgent(c)) urgent.push(c); else if(isBooking(c)) bookings.push(c); else if(isMessage(c)) messages.push(c); else other.push(c); });

  // Detect duplicate phones (same number called multiple times)
  const phoneCounts={};
  calls.forEach(c=>{ const p=c.phone||''; if(p){ phoneCounts[p]=(phoneCounts[p]||0)+1; }});
  calls.forEach(c=>{ c._repeat = (phoneCounts[c.phone||'']||0)>1; });

  const stats={ total:calls.length, action:urgent.length+messages.length, booking:bookings.length, urgent:urgent.length };
  document.getElementById('dek').innerHTML=buildDek(stats);
  document.getElementById('stats').innerHTML=`
    <div class="stat"><div class="n">${stats.total}</div><div class="l">Κλήσεις</div></div>
    <div class="stat ${stats.urgent?'alert':''}"><div class="n">${stats.urgent}</div><div class="l">Επείγοντα</div></div>
    <div class="stat"><div class="n">${messages.length}</div><div class="l">Μηνύματα</div></div>
    <div class="stat"><div class="n">${bookings.length}</div><div class="l">Ραντεβού</div></div>`;

  // Sticky condensed banner (mobile) + browser tab title badge count
  const openCount = [...urgent, ...messages].filter((c,i)=>!todoDone[todoId(c,i)]).length;
  const stickyTotal = document.getElementById('sticky-total');
  const stickyPending = document.getElementById('sticky-pending');
  if(stickyTotal) stickyTotal.textContent = stats.total;
  if(stickyPending) stickyPending.textContent = openCount;
  document.title = openCount>0 ? `(${openCount}) Κλήσεις` : 'Ενημέρωση Κλήσεων';
  const mtabCount = document.getElementById('mtab-actions-count');
  if(mtabCount){
    if(openCount>0){ mtabCount.style.display='inline-block'; mtabCount.textContent=openCount; }
    else { mtabCount.style.display='none'; }
  }
  updateMobileSummary(stats, messages.length, bookings.length, openCount);

  // ── Build grid layout ──
  const action=[...urgent,...messages];
  let leftCol='', rightCol='', bottomFull='';

  // LEFT: Επείγοντα (ξεχωριστή λίστα)
  if(urgent.length){
    leftCol+=`<div class="section"><div class="section-head"><h2 style="color:var(--urgent)">⚠ Επείγοντα</h2><span class="count">${urgent.length}</span></div>`;
    leftCol+=urgent.map((c,i)=>todoItem(c,i)).join('')+`</div>`;
  }

  // LEFT: Εκκρεμότητες (μη-επείγοντα)
  if(messages.length){
    const pendingCount = messages.filter((_,i)=>!todoDone[todoId(_,urgent.length+i)]).length;
    leftCol+=`<div class="section"><div class="section-head"><h2>Εκκρεμότητες</h2><span class="count">${pendingCount} ανοιχτές / ${messages.length}</span></div>`;
    leftCol+=messages.map((c,i)=>todoItem(c,urgent.length+i)).join('')+`</div>`;
  }
  if(!urgent.length && !messages.length){
    leftCol+=`<div class="section"><div class="section-head"><h2>Εκκρεμότητες</h2></div><div class="empty">Δεν υπάρχουν εκκρεμότητες.</div></div>`;
  }

  // LEFT (below todos): Νέα Ραντεβού (created today)
  if(bookings.length){
    leftCol+=`<div class="section"><div class="section-head"><h2>Σημερινές Κρατήσεις</h2><span class="count">${bookings.length}</span></div>`;
    leftCol+=bookings.map(bookingRow).join('')+`</div>`;
  }

  // RIGHT: Πρόγραμμα ημέρας — ΔΕΝ ξαναφορτώνει εδώ, κρατά το ήδη-υπάρχον
  // περιεχόμενο του #schedule-panel αν υπάρχει ήδη στο DOM.
  const existingSchedule = document.getElementById('schedule-panel');
  rightCol+= existingSchedule ? existingSchedule.outerHTML : `<div id="schedule-panel"></div>`;

  // BOTTOM: Χρονολόγιο
  bottomFull+=`<div class="section full-w"><div class="section-head"><h2>Χρονολόγιο Κλήσεων</h2><span class="count">${calls.length}</span></div>`;
  bottomFull+= calls.length ? `<div class="tl">`+calls.map((c,i)=>timelineRow(c,i)).join('')+`</div>` : `<div class="empty">Δεν υπάρχουν κλήσεις ακόμη.</div>`;
  bottomFull+=`</div>`;
  bottomFull+=`<div id="feature-requests-panel" class="full-w"></div>`;

  const gridHtml = calls.length
    ? `<div class="grid"><div class="col-main panel-mobile active" data-panel="actions">${leftCol}</div><div class="col-side panel-mobile" data-panel="schedule">${rightCol}</div><div class="panel-mobile full-w" data-panel="calls">${bottomFull}</div></div>`
    : `<div class="grid"><div class="col-main panel-mobile active" data-panel="actions"><div class="empty">Δεν υπάρχουν κλήσεις σήμερα ή χθες ακόμη.</div></div><div class="col-side panel-mobile" data-panel="schedule"><div id="schedule-panel"></div></div></div>`;

  document.getElementById('content').innerHTML = gridHtml;
  _restoreMobilePanel();
}


if('Notification' in window && Notification.permission==='default') Notification.requestPermission();
fetchCalls();
function scheduleNextFetchCalls(){
  setTimeout(async()=>{
    await fetchCalls();
    scheduleNextFetchCalls();
  }, _nextCallRefreshMs());
}
scheduleNextFetchCalls();
document.addEventListener('visibilitychange',()=>{ if(!document.hidden) fetchCalls(); });
</script>
</body>
</html>"""
    return HTMLResponse(html)


# ============================================================
# Sync Both + Heidi CSV
# ============================================================

@app.get("/sync/both")
async def sync_both_endpoint(key: str = Query(default="")):
    """
    Τρέχει setmore→cal και cal→setmore. Προστατεύεται με DASHBOARD_KEY.

    ΣΗΜΑΝΤΙΚΟ: τρέχει ΕΝΤΟΣ της ίδιας Python διαδικασίας (in-process), όχι με
    subprocess.run(). Το subprocess spawn-άρει νέο Python interpreter που
    ξανακάνει import ΟΛΟ το main.py από την αρχή (FastAPI, Pydantic, httpx,
    και αν είναι ενεργά τα ENABLE_CY_* flags — torch/transformers επίσης),
    προσθέτοντας εκατοντάδες MB πάνω στην ήδη τρέχουσα διαδικασία· σε
    μηχανή 2GB αυτό προκαλεί άμεσο OOM. Χρησιμοποιούμε τις ίδιες in-process
    συναρτήσεις που ήδη χρησιμοποιεί το /admin/trigger-sync.
    """
    if DASHBOARD_KEY and key != DASHBOARD_KEY:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    if _SYNC_BOTH_LOCK.locked():
        return JSONResponse(
            {"status": "already_running", "message": "sync already running, wait"},
            status_code=409,
        )

    async with _SYNC_BOTH_LOCK:
        results = {}
        try:
            cal_result = await sync_cal_to_setmore()
            results["cal_to_setmore"] = cal_result.to_dict()
        except Exception as exc:
            results["cal_to_setmore"] = {"success": False, "error": str(exc)[:300]}
        success, exit_code, error_msg = await _run_setmore_to_cal_sync_async()
        results["setmore_to_cal"] = {"success": success, "exit_code": exit_code}
        if error_msg:
            results["setmore_to_cal"]["error"] = error_msg
        overall = "ok" if all(r.get("success") for r in results.values()) else "partial"
        return {"status": overall, "results": results}


@app.get("/dashboard/heidi-csv")
async def heidi_csv_export(key: str = Query(default=""), date: str = Query(default="")):
    """Παράγει CSV ραντεβού μιας ημέρας στη μορφή Heidi: patient_name, date, time."""
    if DASHBOARD_KEY and key != DASHBOARD_KEY:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    import datetime as _dt
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Asia/Nicosia")
    if date:
        try:
            target_date = _dt.date.fromisoformat(date)
        except ValueError:
            return JSONResponse({"error": "invalid date, use YYYY-MM-DD"}, status_code=400)
    else:
        target_date = (_dt.datetime.now(tz) + _dt.timedelta(days=1)).date()
    date_str = target_date.isoformat()
    try:
        cal = CalClient()
        # Fetch με pagination — μπορεί να χρειαστεί και δεύτερη σελίδα
        all_bookings: list = []
        skip = 0
        while True:
            result = await cal.get_bookings(status="upcoming", take=100, skip=skip, sort_start="asc")
            page_data = result.get("data", [])
            if not isinstance(page_data, list):
                break
            bookings_in_page = [b for b in page_data if isinstance(b, dict)]
            all_bookings.extend(bookings_in_page)
            pagination = result.get("pagination", {})
            if not pagination.get("hasNextPage", False):
                break
            skip += 100
            if skip > 500:  # safety cap
                break
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    rows = []
    for b in all_bookings:
        start_iso = b.get("start", "")
        if not start_iso:
            continue
        try:
            local_dt = _dt.datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        if local_dt.date() != target_date:
            continue
        attendees = b.get("attendees", [])
        name = attendees[0].get("name", "Unknown") if isinstance(attendees, list) and attendees else "Unknown"
        rows.append((local_dt, name, date_str, local_dt.strftime("%H:%M")))
    rows.sort(key=lambda x: x[0])
    import io, csv as _csv
    output = io.StringIO()
    writer = _csv.writer(output)
    writer.writerow(["patient_name", "date (YYYY-MM-DD) - please specify date format", "time"])
    for _, name, d, t in rows:
        writer.writerow([name, d, t])
    from fastapi.responses import Response
    return Response(
        content=output.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=heidi-schedule-{date_str}.csv"},
    )



# Root
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "ortho-reception-backend (Synthflow only, Google Calendar + Brevo)",
        "version": "3.1.0",
    }



# ============================================================
# Rooters
# ============================================================

app.include_router(limassol_router)
app.include_router(evrychou_router)
app.include_router(check_appointment_router)
app.include_router(check_by_phone_router)

# ============================================================
# Bert Endpoint
# ============================================================

@app.post("/cypriot/intent", response_model=CypriotIntentResponse)
async def cypriot_intent(body: CypriotIntentRequest):
    routing_message = _routing_message_from_body(body)
    rule_result = rule_based_intent(routing_message)
    previous_state = get_conversation_state(
        phone_number=_caller_number_from_body(body),
        call_id=body.call_id,
    )
    current_stage = infer_call_stage(rule_result, previous_state)
    next_action = infer_next_action(rule_result, previous_state)
    patient_identity = _merge_patient_identity_from_text(
        _patient_identity_from_cache_by_phone(_caller_number_from_body(body)),
        routing_message,
    )
    next_state = {**previous_state, **patient_identity}
    name_collection_required = name_collection_required_for_route(rule_result, next_state)
    if bool(rule_result.get("is_confident")):
        state = update_conversation_state(
            phone_number=_caller_number_from_body(body),
            call_id=body.call_id,
            **_state_flags_from_intent_result(rule_result),
            current_stage=current_stage,
            next_action=next_action,
            name_collection_required=name_collection_required,
            **patient_identity,
            patient_name_captured=bool(
                patient_identity.get("first_name") and patient_identity.get("last_name")
            ) or None,
        )
        return CypriotIntentResponse(
            intent=str(rule_result.get("intent") or "other"),
            confidence=float(rule_result.get("confidence") or 0.0),
            threshold=CY_INTENT_CONFIDENCE_THRESHOLD,
            is_confident=True,
            should_ask_clarification=False,
            source=str(rule_result.get("source") or "rule"),
            message_clean=str(rule_result.get("message_clean") or ""),
            message_category=str(rule_result.get("message_category") or ""),
            clinic=str(rule_result.get("clinic") or ""),
            day_preference=str(rule_result.get("day_preference") or ""),
            time_preference=str(rule_result.get("time_preference") or ""),
            from_date=str(rule_result.get("from_date") or ""),
            search_mode=str(rule_result.get("search_mode") or ""),
            appointment_triage_needed=bool(rule_result.get("appointment_triage_needed")),
            current_stage=current_stage,
            next_action=next_action,
            name_collection_required=name_collection_required,
            priority_level=str(rule_result.get("priority_level") or ""),
            first_name=patient_identity.get("first_name"),
            last_name=patient_identity.get("last_name"),
            patient_lookup_source=patient_identity.get("patient_lookup_source"),
            state=state,
        )

    if not CY_INTENT_READY:
        state = update_conversation_state(
            phone_number=_caller_number_from_body(body),
            call_id=body.call_id,
            **_state_flags_from_intent_result(rule_result),
            current_stage=current_stage,
            next_action=next_action,
            name_collection_required=name_collection_required,
            **patient_identity,
            patient_name_captured=bool(
                patient_identity.get("first_name") and patient_identity.get("last_name")
            ) or None,
        )
        return CypriotIntentResponse(
            intent=str(rule_result.get("intent") or "other"),
            confidence=float(rule_result.get("confidence") or 0.0),
            threshold=CY_INTENT_CONFIDENCE_THRESHOLD,
            is_confident=False,
            should_ask_clarification=True,
            source="rule_fallback",
            message_clean=str(rule_result.get("message_clean") or ""),
            message_category=str(rule_result.get("message_category") or ""),
            clinic=str(rule_result.get("clinic") or ""),
            day_preference=str(rule_result.get("day_preference") or ""),
            time_preference=str(rule_result.get("time_preference") or ""),
            from_date=str(rule_result.get("from_date") or ""),
            search_mode=str(rule_result.get("search_mode") or ""),
            appointment_triage_needed=bool(rule_result.get("appointment_triage_needed")),
            current_stage=current_stage,
            next_action=next_action,
            name_collection_required=name_collection_required,
            priority_level=str(rule_result.get("priority_level") or ""),
            first_name=patient_identity.get("first_name"),
            last_name=patient_identity.get("last_name"),
            patient_lookup_source=patient_identity.get("patient_lookup_source"),
            state=state,
        )

    try:
        intent, conf = predict_cypriot_intent(routing_message)
    except Exception as e:
        print("ERROR in cypriot_intent:", e)
        raise HTTPException(status_code=500, detail=f"Intent classification failed: {str(e)}")

    is_confident = conf >= CY_INTENT_CONFIDENCE_THRESHOLD
    model_result = {
        "intent": intent,
        "is_confident": is_confident,
        "message_clean": routing_message if intent == "message" else "",
        "appointment_triage_needed": should_triage_appointment(routing_message, intent, {}),
    }
    model_stage = infer_call_stage(model_result, previous_state)
    model_next_action = infer_next_action(model_result, previous_state)
    model_next_state = {**previous_state, **patient_identity}
    model_name_collection_required = name_collection_required_for_route(
        model_result, model_next_state
    )
    state = update_conversation_state(
        phone_number=_caller_number_from_body(body),
        call_id=body.call_id,
        current_intent=intent,
        current_message=routing_message if intent == "message" else "",
        current_stage=model_stage,
        next_action=model_next_action,
        name_collection_required=model_name_collection_required,
        appointment_triage_needed=should_triage_appointment(routing_message, intent, {}) or None,
        message_already_captured=(intent == "message" and is_confident) or None,
        **patient_identity,
        patient_name_captured=bool(
            patient_identity.get("first_name") and patient_identity.get("last_name")
        ) or None,
    )
    return CypriotIntentResponse(
        intent=intent,
        confidence=conf,
        threshold=CY_INTENT_CONFIDENCE_THRESHOLD,
        is_confident=is_confident,
        should_ask_clarification=not is_confident,
        source="model",
        message_clean=routing_message if intent == "message" else "",
        message_category="",
        from_date="",
        search_mode="",
        appointment_triage_needed=should_triage_appointment(routing_message, intent, {}),
        current_stage=model_stage,
        next_action=model_next_action,
        name_collection_required=model_name_collection_required,
        priority_level=classify_priority_level(routing_message, intent),
        first_name=patient_identity.get("first_name"),
        last_name=patient_identity.get("last_name"),
        patient_lookup_source=patient_identity.get("patient_lookup_source"),
        state=state,
    )

class CypriotFillMaskRequest(BaseModel):
    text: str = Field(..., description="Κείμενο με ένα [MASK] token")

class CypriotFillMaskResponse(BaseModel):
    candidates: list[str]

@app.post("/cypriot/fill_mask", response_model=CypriotFillMaskResponse)
async def cypriot_fill_mask(body: CypriotFillMaskRequest):
    if not CY_MLM_READY:
        raise HTTPException(status_code=503, detail="Cypriot masked LM not available")

    text = body.text
    if "[MASK]" not in text:
        raise HTTPException(status_code=400, detail="Text must contain a [MASK] token")

    try:
        import torch
        with torch.no_grad():
            input_ids = cy_mlm_tokenizer.encode(text, return_tensors="pt")
            outputs = cy_mlm_model(input_ids)[0]  # logits
            mask_index = (input_ids[0] == cy_mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
            mask_logits = outputs[0, mask_index]
            top_k = torch.topk(mask_logits, k=5)
            tokens = [cy_mlm_tokenizer.decode([idx]) for idx in top_k.indices]
    except Exception as e:
        print("ERROR in cypriot_fill_mask:", e)
        raise HTTPException(status_code=500, detail=f"Fill-mask failed: {str(e)}")

    return CypriotFillMaskResponse(candidates=tokens)


# ============================================================
# Setmore shadowing
# ============================================================

@app.post("/shadow/setmore", response_model=ShadowSetmoreResponse)
async def shadow_setmore(req: ShadowSetmoreRequest):
    result = await create_setmore_shadow_appointment(
        first_name=req.first_name,
        last_name=req.last_name,
        phone_number=req.phone_number,
        start_iso=req.start_iso,
        clinic=req.clinic,
        email=req.email,
        notes=req.notes,
        cal_uid=req.cal_uid,
        duration_minutes=req.duration_minutes,
    )

    return ShadowSetmoreResponse(
        success=result.success,
        setmore_appointment_id=result.appointment_id,
        error=result.error,
    )


@app.post("/sync/cal-to-setmore", response_model=CalSetmoreSyncResponse)
async def sync_cal_to_setmore_endpoint(
    x_sync_token: str | None = Header(default=None, alias="X-Sync-Token"),
    token: str | None = Query(default=None),
    force_recreate: bool = Query(default=False),
):
    sync_admin_token = os.getenv("SYNC_ADMIN_TOKEN")
    if not sync_admin_token:
        raise HTTPException(
            status_code=503,
            detail="SYNC_ADMIN_TOKEN is not configured",
        )

    provided_token = x_sync_token or token
    if provided_token != sync_admin_token:
        raise HTTPException(status_code=403, detail="Invalid sync token")

    result = await sync_cal_to_setmore(force_recreate=force_recreate)
    return CalSetmoreSyncResponse(**result.to_dict())


@app.post("/admin/trigger-sync")
async def trigger_sync(request: Request):
    """
    Triggers both Cal->Setmore and Setmore->Cal syncs.
    Protected by SYNC_ADMIN_TOKEN.
    """
    if not SYNC_ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="SYNC_ADMIN_TOKEN not configured")
    auth = (request.headers.get("Authorization") or "").strip()
    provided = ""
    if auth.lower().startswith("bearer "):
        provided = auth[7:].strip()
    if not provided:
        provided = (request.headers.get("X-Admin-Token") or "").strip()
    if not hmac.compare_digest(provided, SYNC_ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if _SYNC_BOTH_LOCK.locked():
        return JSONResponse(
            content={"success": False, "error": "sync already running, wait"},
            status_code=409,
        )

    async with _SYNC_BOTH_LOCK:
        results = {}
        # 1. Cal.com -> Setmore sync
        try:
            cal_result = await sync_cal_to_setmore()
            results["cal_to_setmore"] = cal_result.to_dict()
            print("TRIGGER-SYNC cal_to_setmore:", results["cal_to_setmore"])
        except Exception as exc:
            results["cal_to_setmore"] = {"success": False, "error": str(exc)}
            print("TRIGGER-SYNC cal_to_setmore ERROR:", exc)
        # 2. Setmore -> Cal.com sync
        success, exit_code, error_msg = await _run_setmore_to_cal_sync_async()
        results["setmore_to_cal"] = {"success": success, "exit_code": exit_code}
        if error_msg:
            results["setmore_to_cal"]["error"] = error_msg
        print("TRIGGER-SYNC setmore_to_cal:", results["setmore_to_cal"])
        overall_success = any(r.get("success") for r in results.values())
        return JSONResponse(content={
            "success": overall_success,
            "results": results,
        })
