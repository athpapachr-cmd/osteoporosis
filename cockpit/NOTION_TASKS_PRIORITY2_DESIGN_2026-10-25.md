# Priority 2 — Notion tasks and bounded voice proposal

Product Owner accepted the live Priority 1 result and authorized Priority 2 on 2026-10-25. The purpose is one small topbar view of the existing Notion database **Οι εργασίες μου**, plus a dictation path that proposes completing a selected task and adding a follow-up. No ClinicalPending or Notion mirror is created.

## Actual source and ownership

- Notion database `241379c6-877a-40fc-8ab4-a0efc042c887`, data source `02c1d4a7-8768-4ba5-970d-18245b06eeb2`.
- Existing **Εκκρεμείς** view includes `Status = Not started | In progress`, sorted by `Due` ascending. The data source has `Task name`, `Status`, `Due`, `Κατηγορία`, `Προτεραιότητα`, `Σημειώσεις`, `Assignee`.
- Notion alone owns task status, due date, notes and new follow-up rows. The existing ☑ ClinicalPending popover remains separate and keeps its clinical owner.
- A dedicated server-side Notion connection with content read, insert and update access to **this database only** is required. Its token stays in Render env; the browser never receives it. The connected Codex Notion tool cannot be reused as an app runtime credential. Source is OFF when that scoped credential is absent.

## Proposed bounded behavior

1. A distinct topbar button opens the existing popover. Protected `GET /clinical/notion-tasks/peek` queries the fixed data source for up to four active rows in the view's due-date order and returns only task ID, title, status, due, category, revision and an exact Notion page URL. No aggregate count is inferred from four rows. Each row links to its own Notion page; a separate link opens the full database.
2. Dictation starts from a selected row. Browser `MediaRecorder` holds a short recording in memory and posts it to a Clinical-Key-protected endpoint. The server limits size and duration, does not persist audio or raw transcript, and calls OpenAI `gpt-transcribe` for file transcription only when a distinct voice/provider enable gate is configured. A small `gpt-6-luna` structured response proposes `mark_done`, completion wording and one follow-up title/notes. It has no tools and no write access. The transcript is source data, never instructions.
3. A compact review shows the selected Notion task, editable transcript-derived fields and the proposed status/follow-up. The clinician may correct, clear or cancel all fields. **No read or microphone action changes Notion.** Only an explicit confirmation submits a write request.
4. The confirmation endpoint re-reads the exact selected Notion page, checks its parent data source, active status and observed revision, and writes only fields the clinician confirmed. It marks the original `Status = Done` if chosen and creates at most one new `Not started` follow-up in the same data source, carrying an editable title/notes and the selected category. No clinical patient linkage or task is created. The endpoint does not blindly retry uncertain external writes; a partial/unknown result is shown honestly with a link to Notion for reconciliation.
5. Refresh the popover from Notion after confirmed success and on next opening. Logout, panel switch or stale responses must clear the preview and discard audio. DOM text is rendered as text, with exact trusted Notion URLs only.

## Gates and limits

- `NOTION_TASKS_ENABLED=1`, `NOTION_TASKS_TOKEN` and the fixed data source ID are required for read/write. The key must be a dedicated Notion internal connection shared manually with the database; no personal or broad token fallback.
- `NOTION_TASKS_VOICE_ENABLED=1`, `OPENAI_API_KEY` and `NOTION_TASKS_VOICE_PHI_PROVIDER_APPROVED=1` are separately required before identifiable task audio/transcripts are sent to OpenAI. Existing clinical-documents flags do not authorize this purpose. Text review remains available without microphone support.
- No real task mutation in tests or smoke. Synthetic fixtures only. Production smoke is bounded to source read/shape, auth rejection, configured feature state and UI assets; Product Owner performs the first real confirmed task action.
- This is R2 because it adds an external authoritative write and possible identifiable audio transfer. Independent pre-code design and post-code exact-head reviews are required. Implementation stays in the Cockpit repo, not the Reception/Voice Agent/backend workstreams.
- Missing scoped Notion credential or task-specific provider approval blocks activation of the affected feature, not safe code review/deploy. No Priority 3 work follows without Product Owner acceptance.
