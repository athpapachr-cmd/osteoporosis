# Direct audio summary correction

Product Owner requests Stop -> automatic clinical summary, no identity checkbox, optional editing and working paste. Base main 8974383f. R2: explicit confirmation boundary changes.

Reuse audio pipeline/validated renderer. Show result.detail immediately read-only; transcript/editor remain optional collapsed controls. Explicit Save is acceptance of displayed patient/current candidate; send recording_patient_confirmed=true only in this click path. Automatic processing/background preview never assert it. Backend/local checks unchanged. Save first previews exactly current bytes, guards generation/patient/context/candidate across awaits, then writes once. Renew context under same guards.

Transcript-only has no checkbox/Preview/Save. Enable paste after processing finishes. Manual extraction abandons audio authority, hides audio presentation and invalidates older async results. Preserve recovery Blob/transcript until discard or successful manual extraction. Existing manual owner handles paste; identity-blocked audio stays nonretryable.

Clarify prompt: absent spoken name/DOB/identifier is consistent; generic age/sex/history do not identify another person or conflict with synthetic chart labels. Only explicit identifying contradiction or genuinely ambiguous named subjects cause conflict/uncertain. Never override those model verdicts.

Evidence: browser Stop -> summary/no extra click/no automatic Save; optional editor; edit -> explicit Save previews latest bytes once; transcript-only paste works/no checkbox; stale results rejected; route identity/expiry checks preserved. Prompt request fixture covers demographic example. Actual model quality needs fictional provider qualification. No model/retention/schema change.
