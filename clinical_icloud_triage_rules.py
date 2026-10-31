"""Conservative header-only iCloud triage hints for one-doctor pilot.

Suggestions, NEVER patient identity or clinical attestation. No AI/provider
lookup, file parsing or clinical content persistence. Every alias message
remains recoverable in the full All inbox, including administrative mail.
"""
from __future__ import annotations

import re
import unicodedata
from email.utils import getaddresses

CLINICAL_WORDS = (
    "radiolog", "radiology", "ακτινολογ", "διαγνωστικ", "μαγνητικ",
    "mri", "m.r.i", "xray", "x-ray", "x ray", "ct scan", "ultrasound",
    "υπερηχο", "ακτινογραφ", "εργαστηρι", "αποτελεσμα", "εξετασ",
    "αιματολογ", "βιοχημικ", "γνωματευ", "πορισμα", "patient",
    "medical", "hospital", "lab result", "laborator", "scan",
    "referral", "οστεοπορωσ", "physiotherapy",
)
ADMIN_SUBJECT = (
    "payout", "payment", "invoice", "receipt", "billing",
    "statement of account", "transfer completed", "bank transfer",
    "αποδειξη πληρωμ", "εμβασμα", "καταβολη", "ενοικιο", "μισθωμα",
    "rental", "rent payment", "tax", "λογαριασμοσ", "παραστατικο",
)
ADMIN_SENDERS = (
    "cal.com", "calendly.com", "stripe.com", "paypal.com",
    "payout", "billing@", "accounts@", "invoices@",
    "pastides", "παστιδ",  # Practice-specific rental/administrative correspondence.
)
ADMIN_EVENT_WORDS = (
    "booking confirmation", "event confirmation", "evrychou between",
    "limassol between", "appointment scheduled", "meeting between",
)
BULK_INDICATORS = {"bulk", "list", "junk"}


def fold(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value).casefold())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value).strip()


def _sender(message) -> tuple[str, str]:
    name, address = (getaddresses([str(message.get("From", ""))]) or [("", "")])[0]
    return fold(name), fold(address)


def suggest_category(message) -> str:
    """Return a navigational hint, never a verified sender or patient link.

    clinical: plausible medical correspondence; needs doctor's medical action.
    administrative: booking, rent, payout or accounts; recoverable under All.
    other: nonmedical bulk; recoverable under All.
    review: ambiguous mail, visible in clinical triage rather than hidden.
    """
    subject = fold(message.get("Subject", ""))
    display, address = _sender(message)
    sender = display + " " + address

    # Practice-specific false positives must not dominate the clinical queue.
    if (any(token in sender for token in ADMIN_SENDERS)
            or any(token in subject for token in ADMIN_SUBJECT)
            or any(token in subject for token in ADMIN_EVENT_WORDS)):
        return "administrative"

    # A sender display name can be spoofed. This is only a routing hint,
    # NEVER patient identity evidence, clinical review or trusted provider.
    if any(token in subject or token in sender for token in CLINICAL_WORDS):
        return "clinical"

    auto = fold(message.get("Auto-Submitted", ""))
    bulk = (bool(message.get("List-Unsubscribe"))
            or fold(message.get("Precedence", "")) in BULK_INDICATORS
            or bool(auto and auto != "no"))
    return "other" if bulk else "review"
