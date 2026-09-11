# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1D — MERGED / DEPLOYED / PRODUCTION-SMOKE-VERIFIED / CLOSED
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CORE-LEARNING-HUB-L1D-CLIPBOARD-HANDOFF-2026-09-10`.
> **PR #86:** CLOSED / MERGED.
> **Reviewed PR head:** `5e524c552001c2899e5a274e7af103ae575fd669`.
> **Exact tested runtime head:** `d94b9a17b60299e098a5c58722542b6d8d68827d`.
> **Squash merge SHA:** `e0ecf43b5c4e96f20be7a777bfec65a87641d339`.
> **Production-verified deploy commit:** `833fddb15f41e78071b25c5807a120c8237377ac`.
> **Render deploy:** `dep-dah21hpsrm7s7392hu90` — LIVE.
> **ACTIVE RUNTIME/DESIGN WRITER:** NONE.
> **Production config/secret authority exercised:** NO.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Released L-1D workflow

The default Clinical Learning Hub Inbox includes the clinician-initiated quick handoff action:

`Paste & Send Challenge`

Production workflow:

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

# 2. Production verification

Render auto-deploy completed successfully on the current docs-only descendant carrying the merged L-1D runtime:

```text
service = srv-d5qfk31r0fns73di596g
deploy = dep-dah21hpsrm7s7392hu90
commit = 833fddb15f41e78071b25c5807a120c8237377ac
trigger = new_commit
status = live
finished = 2026-09-10T03:17:02Z
```

The product owner then completed an authenticated production smoke and reported that:

```text
Copy from Challenge = worked
Paste & Send Challenge = worked
Inbox import/review flow = worked
```

This closes the L-1D acceptance objective. The prior L-1B clinician review / Learning Loop activation path was already production-smoke-verified separately; this smoke specifically verifies the new clipboard handoff layer reaches and works with that existing review path.

---

# 3. Security / authority preserved

```text
LEARNING RECORD != PATIENT RECORD
clipboard access requires explicit user gesture
clipboard content is not persisted in browser storage
raw transcript is not an accepted learning artifact
existing PHI guard remains authoritative
pending_review only at handoff
no auto-accept
no imported clinician-review authority
no imported reference-verification authority
no Foundation state mutation
no Signal promotion
no patient reads/writes
no new backend endpoint/table/schema
no CLINICAL_LEARNING_INGEST_KEY configuration
no native MCP/write-tool claim
```

The L-1D convenience surface reuses the existing protected `/clinical/learning/api/imports` path and does not expand write authority.

---

# 4. Verification evidence

Exact tested runtime head:

`d94b9a17b60299e098a5c58722542b6d8d68827d`

PR-head gates:

- Clinical Learning L1C challenge transport gate `34431909196` — **SUCCESS**.
- Clinical Learning L1B regression gate `34431909214` — **SUCCESS**.
- Clinical Learning L1 regression gate `34431909207` — **SUCCESS**.
- L0 contract validation step — **SUCCESS**; overall L0 workflow failure was only the expected design-only scope rejection for a non-L0 runtime slice.

---

# 5. Lifecycle state

```text
L-1D IMPLEMENTED = YES
L-1D TESTED = YES
L-1D REVIEWED = PASS
L-1D MERGED = YES
L-1D DEPLOYED = YES
PRODUCTION CLIPBOARD SMOKE = PASS
PASTE & SEND UX = PRODUCTION VERIFIED
MANUAL INBOX FALLBACK = AVAILABLE
WRITER LOCK = NONE
PRODUCTION INGEST KEY = NOT CONFIGURED
NATIVE CHATGPT WRITE TOOL = NOT CONNECTED
```

L-1D is closed. Future zero-click ChatGPT → Cockpit transport, MCP/app integration or cron reconciliation remains a separate explicitly governed integration slice and must not be inferred from this closeout.
