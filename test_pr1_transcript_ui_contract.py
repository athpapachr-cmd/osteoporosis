from pathlib import Path


def test_transcript_ui_is_ephemeral_non_authoritative_and_loaded():
    js = Path("static/baseline-audit/transcript-capture.js").read_text(encoding="utf-8")
    bootstrap = Path("static/baseline-audit/app.js").read_text(encoding="utf-8")
    assert '/clinical/transcript/extract' in js
    assert 'data-nav-action="heidi"' in js
    assert 'AI-extracted candidate ≠ επιβεβαιωμένο κλινικό δεδομένο.' in js
    assert 'pagehide' in js and 'pageshow' in js and 'clearState' in js
    assert 'credentials: "same-origin"' in js
    assert 'document.querySelector("#encounterArchetype")' in js
    assert 'loadScript("./transcript-capture.js")' in bootstrap
    for forbidden in ("localStorage", "sessionStorage", "indexedDB"):
        assert forbidden not in js
    assert "data-transcript-submit" in js
    assert "Accept" not in js and "Αποδοχή" not in js
    assert "currentCase" not in js
    assert "patient-registry" not in js


def test_transcript_ui_renders_provider_data_with_text_content_not_html():
    js = Path("static/baseline-audit/transcript-capture.js").read_text(encoding="utf-8")
    assert "node.textContent" in js
    assert "candidate.evidence_snippet" in js
    assert "resultsNode.replaceChildren" in js
