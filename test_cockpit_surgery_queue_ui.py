from pathlib import Path


ROOT = Path(__file__).resolve().parent
HTML = (ROOT / "static/cockpit/index.html").read_text(encoding="utf-8")
JS = (ROOT / "static/cockpit/app.js").read_text(encoding="utf-8")
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")


def test_cockpit_surgery_queue_exposes_required_fields_and_controls():
    for required_id in (
        "surgeryQueueState",
        "surgeryTableBody",
        "surgeryForm",
        "surgeryFullName",
        "surgeryIdentityNumber",
        "surgeryDateOfBirth",
        "surgeryPhone",
        "surgeryProcedureType",
        "surgeryLaterality",
        "surgeryDate",
        "surgerySubmit",
        "surgeryCancelEdit",
    ):
        assert f'id="{required_id}"' in HTML

    for sort_key in (
        "queue_position",
        "full_name",
        "identity_number",
        "date_of_birth",
        "procedure_type",
        "laterality",
        "phone",
        "surgery_date",
    ):
        assert f'data-sort="{sort_key}"' in HTML

    assert "Ολοκληρώθηκε" in JS
    assert 'actionButton("🗑", "delete"' in JS
    assert 'actionButton("Διαγραφή", "delete"' not in JS
    assert "Διαγραφή" in JS
    assert 'method: "DELETE"' in JS
    assert '"up"' in JS
    assert '"down"' in JS
    assert "/clinical/surgeries" in JS


def test_surgery_queue_does_not_persist_patient_identity_in_browser_storage():
    assert "localStorage" not in JS
    assert "sessionStorage" not in JS
    assert ".innerHTML" not in JS
    assert "textContent" in JS


def test_surgery_queue_is_wired_into_clinical_core():
    assert "from clinical_surgery_queue import build_surgery_queue_router" in MAIN
    assert "app.include_router(build_surgery_queue_router(engine))" in MAIN


def test_calendar_note_reflects_daily_feed_not_future_slice():
    assert "ενημερώνεται καθημερινά" in JS
    assert "θα συνδεθεί στο επόμενο integration slice" not in JS
