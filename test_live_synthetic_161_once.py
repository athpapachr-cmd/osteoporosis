import os, json, urllib.request
try:
    req=urllib.request.Request('https://ortho-reception-backend.onrender.com/clinical/clinic-utilities/medical-report/api/contract',headers={'X-Clinical-Key':os.environ['CLINICAL_DATA_KEY']})
    with urllib.request.urlopen(req,timeout=45) as response:
        data=json.load(response)['ai']
    print(json.dumps({k:data.get(k) for k in ('provider','enabled','api_key_configured','phi_provider_approved','analysis_model')},ensure_ascii=False))
except Exception:
    print('Provider metadata unavailable');raise SystemExit(1)
