"""Synthetic header-only sorting; NEVER used to infer a patient identity."""
from email.message import EmailMessage
from clinical_icloud_triage_rules import suggest_category


def message(subject="", sender="patient@example.invalid", *, bulk=False):
    m = EmailMessage()
    m["From"] = sender
    m["To"] = "practice.alias@icloud.com"
    m["Subject"] = subject
    if bulk:
        m["Auto-Submitted"] = "auto-generated"
    return m


def test_radiology_sender_without_subject_is_a_clinical_hint():
    assert suggest_category(message(sender="Radiology Department <radiology@example.invalid>")) == "clinical"
    assert suggest_category(message(sender="Ακτινολογικό κέντρο <mail@example.invalid>", bulk=True)) == "clinical"


def test_generated_mri_transfer_never_becomes_generic_review():
    assert suggest_category(message(
        "Imaging MRI files delivered by WeTransfer",
        "noreply@wetransfer.com", bulk=True,
    )) == "clinical"


def test_calendar_payout_invoice_and_rent_are_administrative():
    for subj, sender in (
        ("Limassol between doctor and patient", "hello@cal.com"),
        ("Appointment scheduled", "events@example.invalid"),
        ("Payout payment received", "finance@example.invalid"),
        ("Your invoice receipt", "billing@example.invalid"),
        ("Ενοίκιο Οκτωβρίου", "sender@example.invalid"),
    ):
        assert suggest_category(message(subj, sender)) == "administrative"


def test_practice_only_sender_rules_are_private_and_reversible(monkeypatch):
    fake_person = "synthetic-rental-contact"
    candidate = message("Καλησπέρα", fake_person + "@example.invalid")
    assert suggest_category(candidate) == "review"
    monkeypatch.setenv("ICLOUD_TRIAGE_ADMIN_HINTS", fake_person)
    assert suggest_category(candidate) == "administrative"
    monkeypatch.delenv("ICLOUD_TRIAGE_ADMIN_HINTS")
    assert suggest_category(candidate) == "review"


def test_ambiguous_patient_mail_visible_and_bulletin_recoverable():
    assert suggest_category(message("Hello from a patient")) == "clinical"
    assert suggest_category(message("Καλημέρα")) == "review"
    assert suggest_category(message("Weekly digest", "news@example.invalid", bulk=True)) == "other"


def test_sender_and_category_do_not_overwrite_visitor_message():
    candidate = message("MRI results", "Fake Radiology <anything@example.invalid>")
    before = list(candidate.items())
    assert suggest_category(candidate) == "clinical"
    assert list(candidate.items()) == before
