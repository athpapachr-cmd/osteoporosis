"""Build and smoke the downloadable synthetic prototype, without production app/data."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUTPUT=ROOT/'artifacts'
OUTPUT.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as temp:
    staging=Path(temp)/'physio-knee-oa-prototype'
    staging.mkdir()
    for relative in ['clinic_utilities/contracts','clinic_utilities/physio_profiles','clinic_utilities/physio_referral_product']:
        shutil.copytree(ROOT/relative,staging/relative,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.venv'))
    for name in ['__init__.py','physio_referral_runtime.py','physio_referral_formatter_el.py','physio_referral_formatter_el_v2.py']:
        shutil.copy2(ROOT/'clinic_utilities'/name,staging/'clinic_utilities'/name)
    shutil.copy2(HERE/'README.md',staging/'START_HERE.md')
    files={str(path.relative_to(staging)):hashlib.sha256(path.read_bytes()).hexdigest()
           for path in sorted(staging.rglob('*')) if path.is_file()}
    provenance={'source_commit':os.environ.get('GITHUB_SHA','local_uncommitted_build'),
                'synthetic_only':True,'production_registration':False,'file_sha256':files}
    (staging/'BUILD_PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    # This single check proves packaged dependency closure, not a repeat full regression.
    probe="""from clinic_utilities.physio_referral_product.prototype.test_server import request
from clinic_utilities.physio_referral_product.prototype.server import project, bootstrap
assert bootstrap()['synthetic_only'] is True
assert project(request())['gate']['allowed'] is True
print('Packaged real-CU1 dependency closure PASS')
"""
    subprocess.run([sys.executable,'-c',probe],cwd=staging,check=True,timeout=30)
    for cache in staging.rglob('__pycache__'):
        shutil.rmtree(cache)
    archive=shutil.make_archive(str(OUTPUT/'physio-knee-oa-prototype'),'zip',root_dir=temp,base_dir=staging.name)
    print('Synthetic prototype archive:',Path(archive).name)
