# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-1 PROGRESSIVE GUIDANCE FOUNDATION IMPLEMENTED / TESTED / RELEASE GATE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main` at G-1 bootstrap:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Implementation branch:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
> **Parent accepted design:** `design/module01-progressive-guidance-foundations-2026-08-30` @ `298d8d525f1bac97ffb6904fe09800519bd1a584`.
> **Runtime/test head before canonical closeout:** `d66ac728e379a90542f95a8ecdde6d945420f6ae`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30` — canonical closeout only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — bounded G-1 implementation complete.
> **Merge/deploy/production smoke:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product outcome implemented

G-1 installs only the first progressive guidance foundations; it does not attempt a complete osteoporosis taxonomy.

Current runtime model:

```text
existing first-page encounter dropdown
+ existing first-page quick_notes / short context
+ protected prior completed/amended encounters when a patient is loaded
→ read-only longitudinal projection
→ deterministic current encounter context
→ minimum Visit Plan
→ existing cards visually prioritized with WHY NOW
```

The dropdown remains clinician-declared coarse intent. `quick_notes` remains context only and is not parsed into new authoritative clinical facts or an inferred visit class.

---

# 2. Implemented runtime files

```text
static/baseline-audit/progressive-guidance-core.js   NEW
static/baseline-audit/progressive-guidance-ui.js     NEW
static/baseline-audit/progressive-guidance.css       NEW
static/baseline-audit/adaptive-applicability.css     bounded ownership-compatible change
static/baseline-audit/app.js                         bootstrap wiring
```

Tests/CI:

```text
test_progressive_guidance_node.js                    NEW
test_progressive_guidance_wiring.js                  NEW
.github/workflows/g1-progressive-guidance-tests.yml  NEW
```

No clinical storage schema or database migration was added.

---

# 3. Longitudinal projection implemented

The pure guidance core builds an ephemeral projection from:

```text
GET /clinical/patient/{patient_id}/encounters
```

using completed/amended historical encounters only and excluding the current encounter when identifiable.

Implemented invariants:

- scheduled/planned administration without an exact `actual_date` does not count as administered;
- repeated representation of the same exact `agent + actual_date` is not double-counted;
- stable administration IDs are retained when useful;
- contradictory representations remain explicit projection conflicts;
- actual administration count is never reconstructed from expected cadence or elapsed treatment time;
- a later blank treatment snapshot does not itself erase older history;
- unresolved prior planned tasks may resurface when the same semantic task has not later been explicitly completed/not applicable;
- no new patient-level treatment table/source of truth is created.

G-1 uses explicit stored `next_due_date`/due-status context only. It does not derive medication-specific next-dose timing from clinical cadence.

---

# 4. Current Visit Plan behavior

Current G-1 reason classes:

```text
NEW_EVENT
UNRESOLVED_PRIOR
EXPLICIT_DUE_STATE
TREATMENT_CONTEXT
VISIT_TYPE_CORE
CONTEXTUAL
```

Priority:

```text
NEW_EVENT
→ UNRESOLVED_PRIOR
→ EXPLICIT_DUE_STATE
→ TREATMENT_CONTEXT
→ VISIT_TYPE_CORE
→ CONTEXTUAL
```

Examples implemented as mechanics:

- first-assessment archetypes surface a broader base flow;
- routine treatment continuation remains narrower;
- an explicit new fracture overrides the routine flow and surfaces fracture/risk/treatment domains;
- explicit fracture-on-treatment additionally surfaces administration/transition context;
- unresolved prior tasks resurface follow-up context;
- explicitly recorded due/overdue treatment state prioritizes administration/follow-up cards;
- longitudinal treatment conflicts are shown rather than silently resolved.

These are visit-flow mechanics, not medication-specific treatment recommendations.

---

# 5. UI ownership correction found during implementation review

Initial implementation temporarily removed coarse adaptive collapse classes when higher-priority G-1 guidance surfaced a card. Review identified that this could leave the card visually open after the higher-priority trigger disappeared unless the coarse applicability module reran.

The final implementation preserves ownership instead:

```text
adaptive-applicability.js/classes remain untouched
+
guidance-surfaced class provides a temporary visual override
```

CSS applies coarse collapse only to:

```text
.adaptive-collapsed:not(.guidance-surfaced)
```

When the G-1 reason disappears, the underlying coarse applicability state automatically becomes visible again without repair logic or state mutation.

---

# 6. Exact test evidence

Runtime/test head:

```text
d66ac728e379a90542f95a8ecdde6d945420f6ae
```

GitHub Actions:

```text
workflow: G1 progressive guidance foundation
run:      33327717796
head:     d66ac728e379a90542f95a8ecdde6d945420f6ae
result:   SUCCESS
```

Successful gates:

- JavaScript syntax checks;
- progressive guidance pure-core regressions;
- progressive guidance bootstrap/ownership/wiring regression;
- pre-existing authoritative Finish browser regression;
- pre-existing server finalization lifecycle regression.

Core regressions explicitly cover:

1. first-assessment broad base flow;
2. `other + quick_notes` stays context only and is not silently classified;
3. repeated exact actual administration deduplication;
4. scheduled-only administration not counted;
5. conflicting explicit next-due facts remain conflicts;
6. unresolved prior task and later explicit completion behavior;
7. no due state inferred from actual administration date alone;
8. new fracture/fracture-on-treatment override of routine flow;
9. current/draft encounter excluded from historical projection;
10. deterministic same structured context → same Visit Plan.

Wiring regression proves the finalization bootstrap remains:

```text
finalization coordinator
→ patient registry
→ pilot completion
```

and that G-1 loads without taking Finish ownership.

---

# 7. What is NOT proven / NOT done

```text
G-1 IMPLEMENTED                         YES
G-1 TESTED                              YES
G-1 MERGED                              NO
G-1 DEPLOYED                            NO
G-1 PRODUCTION-SMOKE-VERIFIED           NO
G-1 REAL-CLINIC-USABILITY-VALIDATED     NO
FULL VISIT TAXONOMY                     NO / deliberately deferred
MEDICATION-SPECIFIC MILESTONE RULES     NO
PR-1 HEIDI EXTRACTION                   NO
PR-2 INLINE ACCEPT/EDIT/REJECT          NO
5-CASE SYSTEM-ASSISTED PILOT            NOT STARTED
30-CASE SYSTEM-ASSISTED BASELINE        NOT STARTED
MODULE 01 CLOSED                        NO
```

No real patient/transcript fixture was used in G-1 tests.

---

# 8. C1 authoritative Finish state preserved

The branch still inherits the previously tested C1 implementation:

```text
fix/module01-c1-authoritative-finish-2026-08-30
@ a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871
IMPLEMENTED / TESTED
MERGED NO
DEPLOYED NO
PRODUCTION-SMOKE NO
```

G-1 testing re-ran and passed the authoritative Finish browser and server lifecycle regressions. This does not constitute release/deploy authority.

---

# 9. Physiotherapy remains parked/preserved

```text
feat/cu1-rich-referral-global-evidence-2026-08-29
@ bdd23b83a8252405f5aa5a0c0b67f303ccfcef5f
IMPLEMENTED / TESTED / PRODUCT-OWNER REVIEWED
MERGED NO / DEPLOYED NO
```

No physiotherapy mutation occurred.

---

# 10. Exact next action

STOP runtime mutation at the G-1 release gate.

A separate product-owner release decision is required before PR/merge/deploy. Because this branch includes the inherited C1 correction and accepted Module-01 design ancestry, any release action must fresh-bootstrap, inspect the complete compare against current `main`, and verify that no unrelated parked work is included.

After an authorized merge/deploy, production smoke must distinguish:

```text
C1 authoritative Finish integrity
+
G-1 guidance loading / dropdown + context / longitudinal WHY NOW behavior
```

PR-1 Heidi extraction and PR-2 inline provisional population remain later bounded slices required before the five-case system-assisted real pilot unless separately replanned.