from datetime import datetime, timezone

from clinical_calendar import classify_appointment


def test_explicit_medication_categories_win():
    assert classify_appointment("Ένεση Prolia", "", 10) == "prolia"
    assert classify_appointment("Aclasta infusion", "", 60) == "aclasta"


def test_osteoporosis_first_and_review_require_semantics():
    assert classify_appointment("Πρώτη επίσκεψη οστεοπόρωσης", "", 60) == "osteoporosis_first"
    assert classify_appointment("Review osteoporosis", "", 60) == "osteoporosis_review"


def test_duration_does_not_invent_first_vs_review():
    assert classify_appointment("Οστεοπόρωση", "", 60) == "osteoporosis_unspecified"


def test_unrelated_appointment_stays_other():
    assert classify_appointment("Γόνατο", "follow up", 40) == "other"
