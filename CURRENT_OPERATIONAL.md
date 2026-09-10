# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1D — CLIPBOARD HANDOFF IMPLEMENTATION ACTIVE
> **Updated:** 2026-09-10 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh implementation base:** `6feb03dbaea101f11828db9b71f57ebb718ccaa6`.
> **Branch:** `feat/clinical-learning-l1d-clipboard-handoff-2026-09-10`.
> **Slice:** `CORE-LEARNING-HUB-L1D-CLIPBOARD-HANDOFF-2026-09-10`.
> **ACTIVE RUNTIME/DESIGN WRITER:** THIS bounded L-1D slice only.
> **L-1C:** MERGED / DEPLOYED; Project completion protocol installed by product owner in the weekly Osteoporosis Clinical Mentoring automation.
> **Production config/secret authority exercised:** NO.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Problem now being solved

The deployed L-1C protocol correctly makes Cockpit handoff part of Challenge completion, but native ChatGPT write transport is not connected. The current Advanced JSON fallback works but is unnecessarily cumbersome on iPhone.

The bounded L-1D objective is a clinician-initiated clipboard handoff:

```text
Challenge conversation
→ Copy structured episode
→ open Clinical Learning Hub
→ tap Paste & Send Challenge
→ existing protected manual import endpoint
→ Pending Imports / Inbox
→ clinician Review & Save
```

The button must not create an accepted Challenge directly. It may create only the same `pending_review` candidate already owned by the L-1B import path.

---

# 2. Reuse boundary

L-1D must reuse the existing protected endpoint:

`POST /clinical/learning/api/imports`

and therefore reuse the existing:

- rich/canonical learning adapter;
- deterministic PHI guard;
- source-event/idempotency behavior;
- pending-review persistence;
- Inbox review path;
- clinician review before accepted Challenge persistence.

No new backend write authority, environment variable or secret is needed.

---

# 3. Required UX

Default Inbox surface gains one visible quick-handoff control:

`Paste & Send Challenge`

On explicit clinician tap only:

1. read clipboard through the browser Clipboard API when available;
2. accept a copied JSON object, a fenced `json` code block, or a bounded wrapper containing `episode` plus optional `source_event_id`, `loop_plan`, and `resources`;
3. normalize to the existing `/api/imports` request envelope;
4. submit once;
5. surface returned `pending_review` receipt fields;
6. refresh Inbox and open the created/imported candidate.

If clipboard read is unavailable or denied, reveal a small manual paste fallback in the Inbox itself rather than forcing navigation to Advanced.

---

# 4. Safety / privacy invariants

```text
LEARNING RECORD != PATIENT RECORD
explicit tap required before clipboard read
clipboard content is not persisted client-side
raw transcript is not an accepted handoff format
existing PHI guard remains authoritative
pending_review only
no auto-accept
no Foundation mutation
no Signal authority
no patient reads/writes
no CLINICAL_LEARNING_INGEST_KEY configuration
```

A clipboard/import error must not clear the user's source text automatically or claim that Cockpit was updated.

---

# 5. Allowed mutation scope

```text
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
static/clinical-learning/index.html
static/clinical-learning/l1b.js
static/clinical-learning/styles.css                 # only if required for bounded UX
 test_clinical_learning_l1_ui_contract.py
.github/workflows/clinical-learning-l1b-tests.yml   # additive scope-gate maintenance only
.github/workflows/clinical-learning-l1c-tests.yml   # additive scope-gate maintenance only
osteoporosis-change-log.md                          # append-only at material milestone
```

Frozen L-0/L-1 schema owners remain read-only. No backend contract/schema change is authorized unless implementation proves the existing import seam insufficient; that would be a REPLAN trigger.

---

# 6. Exact next action

```text
implement Inbox quick handoff
→ add clipboard/manual-fallback UI contract regressions
→ run inherited L1B/L1 tests and browser syntax
→ exact-head review
→ Draft PR / RELEASE HOLD
```

Do not configure production ingest credentials and do not implement cron/MCP transport in this slice.