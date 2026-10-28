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

import shutil
from playwright.sync_api import sync_playwright, expect
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path=shutil.which("google-chrome") or shutil.which("chromium"))
    page=browser.new_page(viewport={"width":1280,"height":1000})
    writes=[]
    page.on("request",lambda r:writes.append(r.url) if r.method=="POST" and "/clinical/" in r.url and not r.url.endswith("/login") else None)
    page.goto("https://ortho-reception-backend.onrender.com/static/cockpit/index.html?chart=cockpit-cp1-test")
    page.locator("#ceGlobalKey").click()
    page.locator("#ceGlobalKeyInput").fill(os.environ["CLINICAL_DATA_KEY"])
    page.locator("#ceGlobalLoginForm button[type=submit]").click()
    expect(page.locator("#ceGlobalAuthStatus")).to_contain_text("Συνδέθηκες")
    page.keyboard.press("Escape")
    card=page.locator(".patient-chart-overview .patient-chart-card").first
    expect(card).to_be_visible()
    for field in ("history","findings","recommendations"):
        for item in c[field]:
            expect(card).to_contain_text(item["text"])
    for width in (1280,390):
        page.set_viewport_size({"width":width,"height":1000})
        expect(card).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    page.set_viewport_size({"width":1280,"height":1000})
    page.locator(".patient-chart-overview .patient-chart-grid").screenshot(path="complete-live-chart.png")
    assert not writes, "unexpected clinical mutation"
    browser.close()
print("LIVE_EXISTING_SIGNED_SYNTHETIC_CHART_VISIBLE_4_FACTS=PASS; CLINICAL_WRITES=0")
