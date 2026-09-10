# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1D — MERGED / DEPLOY PENDING
> **Updated:** 2026-09-10 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh implementation base:** `6feb03dbaea101f11828db9b71f57ebb718ccaa6`.
> **Slice:** `CORE-LEARNING-HUB-L1D-CLIPBOARD-HANDOFF-2026-09-10`.
> **PR #86:** CLOSED / MERGED.
> **Reviewed PR head:** `5e524c552001c2899e5a274e7af103ae575fd669`.
> **Exact tested runtime head:** `d94b9a17b60299e098a5c58722542b6d8d68827d`.
> **Squash merge SHA:** `e0ecf43b5c4e96f20be7a777bfec65a87641d339`.
> **ACTIVE RUNTIME/DESIGN WRITER:** NONE.
> **L-1C:** MERGED / DEPLOYED; Project completion protocol installed by product owner in weekly Osteoporosis Clinical Mentoring.
> **Production config/secret authority exercised:** NO.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Merged L-1D workflow

The default Learning Hub Inbox now includes a clinician-initiated quick handoff surface:

`Paste & Send Challenge`

Workflow:

```text
Challenge conversation
→ Copy structured synthetic Challenge artifact
→ Clinical Learning Hub / Inbox
→ explicit Paste & Send Challenge tap
→ existing protected POST /clinical/learning/api/imports
→ pending_review candidate
→ clinician Review & Save
```

No accepted Challenge is created by the clipboard action itself.

---

# 2. Supported copied artifact shapes

The client accepts:

1. bare canonical `ClinicalLearningChallengeV1`;
2. bare bounded rich legacy `ClinicalLearningChallengeV1` with `schema_version=1.0`;
3. wrapper `{episode, source_event_id?, loop_plan?, resources?}`;
4. the same JSON inside a Markdown `json` code fence.

For a bare episode carrying top-level `source_event_id`, the client lifts that identifier into the existing import envelope before server validation.

A successful server response is surfaced with:

```text
state
import_id
source_event_id
source_format
idempotent
```

The Inbox then refreshes and opens the returned candidate.

---

# 3. iPhone / browser fallback

Clipboard access occurs only after the clinician taps the primary button.

If the browser does not expose Clipboard API access, permission is denied, or the clipboard is empty:

```text
NO network write
→ reveal compact manual paste fallback in Inbox
→ clinician pastes the same artifact
→ Send pasted Challenge
→ same protected /api/imports flow
```

The Advanced importer remains available unchanged.

---

# 4. Security / authority preserved

```text
LEARNING RECORD != PATIENT RECORD
clipboard access requires explicit user gesture
clipboard content is not persisted in browser storage
raw transcript is not an accepted learning artifact
existing PHI guard remains authoritative
pending_review only
no auto-accept
no imported clinician-review authority
no imported reference-verification authority
no Foundation state mutation
no Signal promotion
no patient reads/writes
no new backend endpoint/table/schema
no CLINICAL_LEARNING_INGEST_KEY configuration
no MCP/cron transport claim
```

A `pending_review` receipt proves only that an Inbox candidate exists.

---

# 5. Verification and merge evidence

Exact tested runtime head:

`d94b9a17b60299e098a5c58722542b6d8d68827d`

Exact PR head:

`5e524c552001c2899e5a274e7af103ae575fd669`

GitHub Actions on the PR head:

- Clinical Learning L1C challenge transport gate `34431909196` — **SUCCESS**.
- Clinical Learning L1B regression gate `34431909214` — **SUCCESS**.
- Clinical Learning L1 regression gate `34431909207` — **SUCCESS**.
- L0 contract validation step — **SUCCESS**; overall L0 workflow failure is the expected design-only scope rejection for a non-L0 runtime slice.

Squash merge:

```text
PR = #86
merged = YES
merge SHA = e0ecf43b5c4e96f20be7a777bfec65a87641d339
merged_at = 2026-09-10T03:13:20Z
```

No backend, database, frozen schema or adjacent-owner file changed.

---

# 6. Lifecycle state

```text
L-1D IMPLEMENTED = YES
L-1D TESTED = YES
L-1D REVIEWED = PASS
L-1D MERGED = YES
PASTE & SEND UX = MERGED
MANUAL INBOX FALLBACK = MERGED
DEPLOYED = PENDING NORMAL RENDER AUTO-DEPLOY
PRODUCTION CLIPBOARD SMOKE = NO
WRITER LOCK = NONE
PRODUCTION INGEST KEY = NOT CONFIGURED
NATIVE CHATGPT WRITE TOOL = NOT CONNECTED
```

Exact next action: allow normal Render `autoDeploy=yes` behavior from `main`; do not manually trigger a duplicate deployment. Verify the auto-created deployment reaches LIVE at the L-1D merge commit or a docs-only descendant carrying the same runtime, then perform one authenticated production clipboard smoke.