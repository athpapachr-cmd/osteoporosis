# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1D — IMPLEMENTED / TESTED / RELEASE HOLD
> **Updated:** 2026-09-10 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh implementation base:** `6feb03dbaea101f11828db9b71f57ebb718ccaa6`.
> **Branch:** `feat/clinical-learning-l1d-clipboard-handoff-2026-09-10`.
> **Slice:** `CORE-LEARNING-HUB-L1D-CLIPBOARD-HANDOFF-2026-09-10`.
> **Exact tested runtime head:** `d94b9a17b60299e098a5c58722542b6d8d68827d`.
> **ACTIVE RUNTIME/DESIGN WRITER:** NONE.
> **L-1C:** MERGED / DEPLOYED; Project completion protocol installed by product owner in weekly Osteoporosis Clinical Mentoring.
> **Production config/secret authority exercised:** NO.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Implemented L-1D workflow

The default Learning Hub Inbox now receives a clinician-initiated quick handoff surface:

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

# 5. Exact-head verification

Exact tested runtime head:

`d94b9a17b60299e098a5c58722542b6d8d68827d`

GitHub Actions:

- Clinical Learning L1C challenge transport gate `34431713117` — **SUCCESS**.
- Clinical Learning L1B regression gate `34431712733` — **SUCCESS**.
- Clinical Learning L1 regression gate `34431712837` — **SUCCESS**.

The bounded diff versus base changes only:

```text
.github/workflows/clinical-learning-l1-tests.yml
.github/workflows/clinical-learning-l1b-tests.yml
.github/workflows/clinical-learning-l1c-tests.yml
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
static/clinical-learning/l1b.js
test_clinical_learning_l1_ui_contract.py
```

No backend, database, schema or adjacent-owner file changed.

---

# 6. Lifecycle state

```text
L-1D IMPLEMENTED = YES
L-1D TESTED = YES
EXACT-HEAD REVIEW = PASS
PASTE & SEND UX = READY
MANUAL INBOX FALLBACK = READY
MERGED = NO
DEPLOYED = NO
PRODUCTION CLIPBOARD SMOKE = NO
WRITER LOCK = NONE
```

Exact next action: open bounded Draft PR / RELEASE HOLD. Merge requires separate explicit product-owner authority. After merge, allow normal Render auto-deploy and perform one authenticated production clipboard smoke.