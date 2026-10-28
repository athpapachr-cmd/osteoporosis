"""One fictional provider extraction, no clinical Save or patient creation."""
import json
import os
from pathlib import Path
import httpx

text = "Κυρία 68 χρονών με πρόσφατο κάταγμα πηχεοκαρπικής πριν δύο χρόνια, εμφανίζει πόνο στην θωρακική μοίρα της σπονδυλικής στήλης εδώ και τρεις μήνες, κυρίως κατά την όρθια στάση και βάδιση, χωρίς νευρολογική σημειολογία. Ζητήθηκαν ακτινογραφίες, σε T-score -2,8 στη σπονδυλική στήλη, σύσταση για έναρξη alendronate μετά από εκτίμηση από οδοντίατρο."
base = "https://ortho-reception-backend.onrender.com"
with httpx.Client(base_url=base, timeout=120) as client:
    r = client.post("/clinical/login", json={"key": os.environ["CLINICAL_DATA_KEY"]})
    assert r.status_code == 200, f"login_status={r.status_code}"
    r = client.get("/clinical/patient/cockpit-cp1-test/chart")
    assert r.status_code == 200, f"chart_status={r.status_code}"
    chart = r.json()
    assert chart["demographics"].get("synthetic_scenario") == "cockpit-cp1-v1"
    # Read the existing synthetic latest candidate without logging other visits.
    saved = chart["visits"][0].get("clinical") if chart["visits"] else None
    r = client.post("/clinical/visit-capture/context", json={
        "patient_id": "cockpit-cp1-test", "client_session_id": "fictional-provider-proof-20261028"})
    assert r.status_code == 200, f"context_status={r.status_code}"
    r = client.post("/clinical/visit-capture/extract", json={
        "context_id": r.json()["context_id"], "text": text, "encounter_date": "2026-10-28"})
    if r.status_code != 200:
        print(f"extract_status={r.status_code}")
        raise SystemExit(1)
    candidate = r.json()["candidate"]
    def checks(value):
        content = json.dumps(value, ensure_ascii=False).lower()
        return {
            "fracture_and_interval": "κάταγμα" in content and ("δύο χρόνια" in content or "2 χρόνια" in content),
            "pain_duration": "πόνο" in content and ("τρεις μήνες" in content or "3 μήν" in content),
            "t_score": "2,8" in content or "2.8" in content,
            "conditional_alendronate": "alendronate" in content and "οδοντ" in content,
        }
    proof = {"candidate": candidate, "extracted_checks": checks(candidate),
             "existing_latest_checks": checks(saved), "writes": 0}
    Path("fictional-summary-proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k,v in proof.items() if k != "candidate"}, ensure_ascii=False))
    assert all(proof["extracted_checks"].values()), "fictional clinical fact omitted"
