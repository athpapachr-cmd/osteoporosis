import json
from pathlib import Path


def test_eval_fixture_is_synthetic_and_covers_required_cases():
    cases = json.loads(Path("evals/transcript_v1/cases.json").read_text(encoding="utf-8"))
    assert len(cases) >= 10
    ids = {item["id"] for item in cases}
    assert {
        "fracture_relative_time",
        "explicit_negative_smoking",
        "dxa_objective",
        "labs_objective",
        "options_one_final",
        "preference_only",
        "followup_vague",
        "garbled_speech",
        "frax_original_adjusted",
        "speaker_ambiguity",
    }.issubset(ids)
    joined = json.dumps(cases, ensure_ascii=False).lower()
    for forbidden in ("gesy id", "@gmail.com", "+357 9"):
        assert forbidden not in joined


def test_provider_eval_runner_is_fail_closed_and_does_not_print_transcript_content():
    runner = Path("evals/transcript_v1/run_provider_eval.py").read_text(encoding="utf-8")
    assert "provider-eval BLOCKED" in runner
    assert 'status["phi_provider_approved"]' in runner
    assert "item['transcript']" not in runner
    assert "result.candidates" in runner
