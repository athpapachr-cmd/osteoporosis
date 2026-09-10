# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1D Clipboard Handoff

> **STATUS:** DESIGN FROZEN / IMPLEMENTATION ACTIVE
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1D-CLIPBOARD-HANDOFF-2026-09-10`.
> **Base:** `6feb03dbaea101f11828db9b71f57ebb718ccaa6`.
> **Branch:** `feat/clinical-learning-l1d-clipboard-handoff-2026-09-10`.
> **L-1C:** MERGED / DEPLOYED; Project completion protocol installed in weekly mentoring automation.
> **Frozen L-0/L-1 schema owners:** READ-ONLY / unchanged.
> **Writer lock:** L-1D bounded writer active.

---

# 1. Product problem

The Challenge completion protocol already produces a structured synthetic learning episode, but native ChatGPT → Cockpit write transport is not connected. The existing Advanced import requires navigating to a debug surface and manually pasting JSON.

The desired clinician workflow is intentionally simpler:

```text
copy Challenge artifact
→ open Learning Hub
→ one explicit Paste & Send Challenge tap
→ pending_review Inbox candidate
```

This is a convenience layer over the existing protected manual import path, not a new authority surface.

---

# 2. UX contract

The Inbox default view gets a compact quick-handoff card above Pending Imports.

Primary action:

`Paste & Send Challenge`

Behavior:

- clipboard is read only after the clinician taps the button;
- successful clipboard read is never stored in localStorage/sessionStorage;
- supported copied shapes:
  1. bare canonical `ClinicalLearningChallengeV1`;
  2. bare bounded rich legacy `ClinicalLearningChallengeV1` with `schema_version=1.0`;
  3. wrapper `{episode, source_event_id?, loop_plan?, resources?}`;
  4. the same JSON inside a Markdown `json` code fence;
- if a bare episode contains a top-level `source_event_id`, L-1D lifts it into the import envelope and removes it from the episode before adapter validation;
- successful import shows `state`, `import_id`, `source_event_id`, `source_format`, and whether the import was idempotent;
- Inbox refreshes and opens the returned candidate automatically.

---

# 3. Clipboard failure contract

Clipboard API support/permission varies by browser surface. Failure must degrade cleanly:

```text
clipboard unavailable / denied / empty
→ no network write
→ reveal manual paste fallback in Inbox
→ clinician pastes same artifact
→ Send pasted Challenge
→ same protected /api/imports flow
```

The fallback is not the old Advanced workflow; it is a small local input on the Inbox quick-handoff card.

---

# 4. Existing server authority reused

L-1D sends only to:

`POST /clinical/learning/api/imports`

The server already owns:

- rich/canonical adapter validation;
- PHI scanning;
- source-event UUID validation/derivation;
- deterministic normalized-hash conflict behavior;
- `pending_review` creation;
- no raw source payload persistence;
- Inbox → clinician review → accepted Challenge linkage.

No endpoint, database table, schema, or environment secret is added.

---

# 5. Security and authority boundaries

Permanent for this slice:

```text
clipboard access requires user gesture
clipboard content is ephemeral client input
no patient identifiers
no raw transcript persistence
no auto-accept
no imported clinician-review authority
no imported reference-verification authority
no Foundation state mutation
no Signal promotion
no patient data mutation
no native MCP/write-tool claim
no production ingest key configuration
```

A successful `pending_review` receipt means only that an Inbox candidate exists.

---

# 6. Acceptance evidence

L-1D is implementation-complete when focused/inherited tests prove:

- Inbox contains the visible `Paste & Send Challenge` control;
- clipboard read occurs only inside its explicit click handler;
- Markdown fenced JSON extraction is bounded;
- bare episode and wrapper normalization both target `/api/imports`;
- top-level `source_event_id` is lifted into the envelope;
- success receipt includes pending-review identity and opens Inbox candidate;
- clipboard failure exposes manual fallback without network write;
- no local/session storage of clipboard content;
- Advanced import remains available unchanged;
- browser JS syntax passes;
- inherited L-1B/L-1 runtime tests remain green;
- frozen schema owners unchanged;
- no Clinic Utilities/RF/physio/CU-1 spillover.

---

# 7. Allowed files

```text
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
static/clinical-learning/index.html
static/clinical-learning/l1b.js
static/clinical-learning/styles.css                 # only if needed
 test_clinical_learning_l1_ui_contract.py
.github/workflows/clinical-learning-l1b-tests.yml   # scope-gate maintenance only
.github/workflows/clinical-learning-l1c-tests.yml   # scope-gate maintenance only
osteoporosis-change-log.md                          # append-only when materially complete
```

---

# 8. Release path

```text
bounded UI implementation
→ focused UI contract + inherited regressions
→ exact-head review
→ Draft PR / RELEASE HOLD
→ explicit product-owner merge authority
→ normal Render auto-deploy
→ one authenticated production clipboard smoke
```

MCP/native write transport and Render cron reconciliation remain separate future integration slices.