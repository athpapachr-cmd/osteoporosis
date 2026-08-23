from clinical_calendar import _clinical_reason, classify_appointment


def test_explicit_medication_categories_win():
    assert classify_appointment("Ένεση Prolia", "", 10) == "prolia"
    assert classify_appointment("Πρόλια", "", 10) == "prolia"
    assert classify_appointment("Aclasta infusion", "", 60) == "aclasta"


def test_explicit_osteoporosis_visit_semantics_outrank_duration():
    assert classify_appointment("Πρώτη επίσκεψη οστεοπόρωσης", "", 40) == "osteoporosis_first"
    assert classify_appointment("Review osteoporosis", "", 60) == "osteoporosis_review"
    assert classify_appointment("Οστεοπόρωση - επανέλεγχος", "", 60) == "osteoporosis_review"


def test_current_secretary_duration_refines_known_osteoporosis_only():
    assert classify_appointment("Οστεοπόρωση", "", 40) == "osteoporosis_review"
    assert classify_appointment("Οστεοπόρωση", "", 60) == "osteoporosis_first"
    assert classify_appointment("Πόνος γόνατος", "", 40) == "other"
    assert classify_appointment("Τροχαίο", "", 60) == "other"


def test_ambiguous_osteoporosis_reason_stays_unspecified():
    assert classify_appointment("Οστεοπενία", "", 20) == "osteoporosis_unspecified"


def test_reason_hides_setmore_transport_metadata():
    assert _clinical_reason(
        "clinic=limassol | source=cal_sync | cal_uid=abc | Οστεοπόρωση - επανέλεγχος"
    ) == "Οστεοπόρωση - επανέλεγχος"
