from openai import OpenAI
from clinical_data import VisitCaptureCandidateV1
try:
 response=OpenAI(timeout=90,max_retries=0).responses.parse(model='gpt-5.6',store=False,max_output_tokens=16000,reasoning={'effort':'high'},text_format=VisitCaptureCandidateV1,input='Synthetic visit only: knee pain, patient will bring MRI CD. Date2026-10-18. source_ref dia_summary, role dia_summary, comparison unavailable. Return candidate from this source.')
 print('SCHEMA PROBE SUCCESS',response.status)
except Exception as e:
 code=getattr(e,'code',None);status=getattr(e,'status_code',None)
 print('SCHEMA PROBE ERROR',type(e).__name__,'status',status,'code',code,'param',getattr(e,'param',None))
 if status==400 and code in ('invalid_json_schema','unsupported_parameter','unsupported_value'):
  body=getattr(e,'body',{})
  print('SCHEMA DETAIL:',str(body.get('message',''))[:2000])
