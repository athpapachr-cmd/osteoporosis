from clinical_data import resolve_encounter_status


def test_draft_can_remain_draft_or_complete():
    assert resolve_encounter_status("draft", "draft", content_changed=True) == "draft"
    assert resolve_encounter_status("draft", "completed", content_changed=True) == "completed"


def test_completed_noop_save_cannot_regress_to_draft():
    assert resolve_encounter_status("completed", "draft", content_changed=False) == "completed"


def test_completed_content_change_becomes_amended_even_if_client_requests_draft():
    assert resolve_encounter_status("completed", "draft", content_changed=True) == "amended"


def test_completed_content_change_becomes_amended_even_if_finish_is_pressed_again():
    assert resolve_encounter_status("completed", "completed", content_changed=True) == "amended"


def test_amended_status_is_sticky():
    assert resolve_encounter_status("amended", "draft", content_changed=False) == "amended"
    assert resolve_encounter_status("amended", "completed", content_changed=True) == "amended"
