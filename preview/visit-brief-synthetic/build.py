"""Build actual Cockpit Visit Brief assets with an isolated synthetic API facade.

Only artifacts inside dist are published; no secret, patient or provider inputs.
"""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[2]
target = root / "visit-brief-demo-dist"
if target.exists():
    shutil.rmtree(target)
(target / "static").mkdir(parents=True)
shutil.copytree(root / "static/cockpit", target / "static/cockpit")
(target / "demo").mkdir()
shutil.copyfile(root / "preview/visit-brief-synthetic/mock-api.js", target / "demo/mock-api.js")

banner = """
<div class="synthetic-banner" role="status">
<strong>ΔΟΚΙΜΑΣΤΙΚΟ VISIT BRIEF · ΣΥΝΘΕΤΙΚΑ ΔΕΔΟΜΕΝΑ</strong>
<span>Πάτησε ένα από τα ραντεβού «Προηγούμενο / Τωρινό / Επόμενο», επίλεξε φάκελο και επιβεβαίωσε.
Καμία σύνδεση με πραγματική βάση, γραμματέα ή στοιχεία ασθενών.</span>
</div>
"""
style = """
<style>
.synthetic-banner {position:sticky;top:0;z-index:5000;background:#172c3c;color:#fff;
 padding:13px 20px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;font:14px/1.35 system-ui;}
.synthetic-banner strong{letter-spacing:.03em;color:#baf2cd}
.synthetic-banner span{max-width:900px}
</style>
<script src="/demo/mock-api.js"></script>
"""
html = (root / "static/cockpit/index.html").read_text(encoding="utf-8")
assert '<body>' in html and '</head>' in html
assert 'static/cockpit/visit-brief.js' in html
html = html.replace('</head>', style + '\n</head>', 1)
html = html.replace('<body>', '<body>' + banner, 1)
(target / "index.html").write_text(html, encoding="utf-8")
# The same preview is reachable from canonical cockpit path; no PHI source.
(target / "static/cockpit/index.html").write_text(html, encoding="utf-8")
print("PASS: standalone synthetic Visit Brief preview with real Cockpit JS/CSS")
