"""Read-only check of latest signed synthetic visit; no provider or Save."""
import json, os, unicodedata
from pathlib import Path
import httpx
with httpx.Client(base_url="https://ortho-reception-backend.onrender.com", timeout=30) as client:
    r=client.post("/clinical/login",json={"key":os.environ["CLINICAL_DATA_KEY"]})
    assert r.status_code==200
    r=client.get("/clinical/patient/cockpit-cp1-test/chart")
    assert r.status_code==200
    data=r.json()
    assert data["demographics"].get("synthetic_scenario")=="cockpit-cp1-v1"
    c=data["visits"][0]["clinical"]
    raw=json.dumps(c,ensure_ascii=False).lower()
    text="".join(x for x in unicodedata.normalize("NFD",raw) if unicodedata.category(x)!="Mn")
    result={"candidate":c,"checks":{
        "fracture": "καταγμα" in text,
        "two_years": "δυο χρον" in text or "2 χρον" in text,
        "three_months": "τρεις μην" in text or "3 μην" in text,
        "t_score": "2,8" in text or "2.8" in text,
        "conditional_alendronate": "alendronate" in text and "οδοντ" in text},
        "provider_calls":0,"clinical_writes":0}
    Path("fictional-summary-proof.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="candidate"}))
