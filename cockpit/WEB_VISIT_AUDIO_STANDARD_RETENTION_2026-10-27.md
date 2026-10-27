# Web Visit Capture — standard OpenAI API retention change

> 2026-10-27, Asia/Nicosia. The Product Owner explicitly chose activation with standard OpenAI API retention, superseding the earlier project-ZDR condition. This is a material privacy-boundary change to the already released web Visit Capture flow, not a claim of Zero Data Retention or whole-service GDPR compliance. Prior release evidence and all patient/visit, exact-source review, and one explicit Save contracts remain in `WEB_VISIT_AUDIO_2026-10-27.md`.

## Provider behavior and clinician notice

OpenAI's current data controls distinguish application state from abuse-monitoring logs. The existing foreground `/v1/responses` request keeps `store:false`; it uses no tools, background mode, or file-storage endpoint. OpenAI's standard abuse-monitoring logs for Responses may contain customer content for up to 30 days, and retention may be longer in the documented legal/safety exceptions. The documented `/v1/audio/transcriptions` row has no application-state or abuse-monitoring retention. These statements describe the provider policy, not an app-enforced deletion guarantee. Source: https://developers.openai.com/api/docs/guides/your-data .

Before browser recording, the clinician sees a concise notice that audio is sent to OpenAI for transcription and the transcript to OpenAI for structuring; `store:false` does not remove the possible standard Responses abuse-log retention (normally up to 30 days, with documented exceptions). Start requires an explicit checkbox asserting that the patient was informed of and agreed to recording and this processing. Unchecking during recording discards it; the protected draft endpoint also requires a matching per-request clinician acknowledgement before reading the request body or calling the provider. This is an operational attestation, not proof of a legal basis or patient consent record. The clinician is responsible for the applicable privacy notice and lawful basis.

After processing, the UI describes only removal of the app's temporary copy. It does not claim to delete provider logs. Raw audio/transcript remain absent from app records and logs, while the exact-source editable draft and final reviewed Save keep their existing owners.

## Configuration and authority

The dedicated route is enabled only when `OPENAI_API_KEY` is present, `VISIT_AUDIO_OPENAI_PHI_APPROVED=1`, and `VISIT_AUDIO_RETENTION_MODE=standard`. The old `VISIT_AUDIO_ZDR_VERIFIED` and `VISIT_AUDIO_ZDR_PROJECT_ID` settings are not part of this selected policy and must not be set merely to bypass the old gate. With no explicit project ID, API requests use the existing project-scoped key without an `OpenAI-Project` override. The key stays server-only. The capability endpoint returns only an enabled boolean; its status is not a legal certification. Other OpenAI purposes and flags confer no authority.

The new acknowledgement header is `X-Visit-Retention-Acknowledged: standard`. The server rejects absent/mismatched acknowledgement with 422 before reading audio. It is checked again after browser Start and before upload. The existing protected context, patient-change checks, size/magic bounds, cancellation, temporary-file cleanup, source-unit validation, provisional clinical review, preview and one-shot Save remain unchanged.

## Evidence and release

Pre-code independent affected R2 reviews provider-policy accuracy, the replaced gate, acknowledgement/no-upload behavior, and preservation of clinical authority. After implementation, synthetic tests verify gate combinations, no project override/key leak, `store:false`, pre-body acknowledgement rejection, checkbox withdrawal, honest post-processing wording, and inherited selected-patient/Save regressions. One independent post-code affected R2, exact-head CI, native Render auto-deploy, no-PHI asset/auth/capability/log smoke, then clinician first informed-patient acceptance. Agent verification sends no real patient audio and performs no clinical write. A Render configuration update is a separate deployment transition; observe its terminal state and smoke it before declaring the feature available.
