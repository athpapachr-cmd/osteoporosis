# Priority 2 correction — new Notion task and speech-filled review

## Product Owner correction and observed cause

The first live dictation on an existing task showed the transcript but left the editable suggestion fields empty. The spoken example expressed an intended future action, not an accomplished task. The installed `gpt-6-luna` prompt asks only for completion/follow-up and explicitly allows empty proposal strings; the UI copies those strings directly. The observed empty form is consistent with that contract. It is not evidence that transcription failed. The Product Owner requests a prominent general **new task / dictate new task** action above the task rows and expects spoken information to appear in editable fields, leaving only review and explicit confirmation.

This is a bounded correction to the existing **Οι εργασίες μου** integration. Notion data source `02c1d4a7-8768-4ba5-970d-18245b06eeb2` remains the sole task owner; ClinicalPending and other Cockpit tasks remain separate. Existing clinical authentication, scoped Notion token, voice/provider approval, 1–60 second / 2 MB limits, ephemeral audio/transcript, no provider tools, `store=false`, and generic external-error handling remain.

## Existing task dictation

- Model proposal must distinguish completed work from a plan or progress note. Future tense alone never implies `Done`.
- Any clear, relevant short dictation can propose a note on the selected task even when nothing has yet been completed. Rename the review field to **Ενημέρωση / σημείωση στην υπάρχουσα εργασία**.
- Keep `mark_done`, note, and at most one follow-up editable. A response with all proposed text empty despite a nonempty transcript uses the exact trimmed transcript as an editable note when it fits the existing 800-character note limit. This fallback reports no completion and creates no follow-up. If over the limit, show the transcript and require the clinician to edit; do not silently truncate.
- The existing `/confirm` endpoint, source revision recheck, note append preserving previous rich text, page/revision unique guard, and unknown/partial outcome freeze remain unchanged. Voice and manual actions still have no Notion side effect until explicit confirmation.

## New task entrance and contract

- At the top of the Notion popover, before the four-row preview, show **Νέα εργασία** and, only when `voice_available=true`, **Υπαγόρευση νέας**. Both are available even when the preview has zero active rows. They are not tied to a selected existing task.
- Manual creation opens one compact form. Dictation uses the existing browser recording flow, transcription model and small structured proposal model, then opens the same form prefilled. The transcript is displayed for correction, never persisted by the app. The proposal may populate **title** (required, up to 180 characters), **notes** (optional, up to 1000), **due date** (optional ISO calendar day), **category** (optional: `Ιατρείο`, `Cockpit`, `Προσωπικά`) and **priority** (optional: `Υψηλή`, `Κανονική`, `Χαμηλή`). The Notion status is fixed to `Not started`; no patient linkage or assignee is inferred.
- The model receives the Cyprus calendar date to interpret explicit relative wording such as “today” or “tomorrow”. It may leave ambiguous date/category/priority blank. The server validates all values against the fixed source schema; invalid model output fails closed to manual review. A short transcript with an empty model title may prefill the exact transcript as title only if within 180 characters. No invented details. A planned action creates a new `Not started` task, never a completed one.
- `POST /clinical/notion-tasks/propose-new` accepts the same bounded `audio/webm` body and duration header as the selected-task voice route. It returns only transcript, editable structured suggestion and a new attempt UUID. It makes no Notion write.
- `POST /clinical/notion-tasks/create` requires a new UUID `attempt_id` and validated editable fields. Before the single Notion `POST /pages`, it atomically claims the existing PostgreSQL `cockpit_notion_task_attempts` guard using `source_page_id=00000000-0000-0000-0000-000000000000` and `source_revision=attempt_id`; the primary key and unique pair reject exact replay across workers. No new table or migration. The guard stores no title, notes, transcript or audio.
- Build a page under the fixed `data_source_id` with `Task name`, `Status=Not started`, and only confirmed optional `Σημειώσεις`, `Due`, `Κατηγορία`, `Προτεραιότητα`. Validate the returned page belongs to that source and show its exact trusted Notion URL on success, even if the new row falls outside the four-row preview. A failed/unknown POST remains frozen in the local guard; client and server never auto-retry it. The clinician checks the Notion database before deciding whether to start a distinct attempt.
- While a create request is pending, disable confirmation. Logout, closing the popover or switching panels discards draft/transcript/audio and stale responses. Render all source and proposal strings as text, not HTML.

## Acceptance and release boundary

Synthetic affected tests cover future-tense existing-task note prefill without Done, model-empty fallback, new-task manual/voice review, category/priority/date validation, auth and feature gates, one-shot create replay/concurrency, unknown POST, no unintended Notion write, stale response and mobile layout. Reuse the established task tests and no-PHI production smoke; do not include real patient data in fixtures, screenshots, CI or logs.

One independent affected pre-code R2 design review and one corrected post-code R2 review gate apply because this adds a new external authoritative write. Exact-head CI, controlled merge, Render deploy and bounded production UI/source smoke follow. The agent must not create a real task during smoke; the clinician performs the first actual confirmed creation. Priority 3 remains outside scope.
