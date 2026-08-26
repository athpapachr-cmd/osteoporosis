# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Current major phase:** Baseline/pilot integrity + Clinical Practice Review foundation.
> **Active phase plan:** `CLINICAL_EXCELLENCE_PLAN.md`.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — PR-1 Transcript Intake + Candidate Extraction v1, pre-code design review.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** PR-1 NOT STARTED and NOT YET AUTHORIZED; detailed design review must close first.
> **CALENDAR / DIGITAL SECRETARY:** intentionally paused.

This file is the sole owner of operational **NOW**. Do not infer current mutation authority from `HANDOFF_CURRENT.md`, chat history or old PR text.

---

# 1. Current runtime foundation

Latest merged runtime hardening before PR-1:

```text
PR #29 — fix: preserve completed encounter finalization
merge commit: 0a2147b8ae5fb8316bde16c8fbb4c0d96aba2194
```

Server lifecycle rule:

```text
draft + ordinary Save                 → draft
draft + Finish Visit                  → completed
completed + no-op Save                → completed
completed + content/date modification → amended
amended + later Save                  → amended
```

Focused deterministic transition tests exist.

## Live synthetic browser smoke — PASSED 3/3

On 2026-08-26 the product owner reported all three agreed synthetic browser checks passed:

1. `completed` encounter + Save with no content change → remained `completed`;
2. material field/date change + Save → became `amended`;
3. reload/reopen → remained `amended` and loadable.

Operational conclusion:

```text
PR #29 finalization integrity gate = CLOSED
```

This removes the previous runtime-integrity blocker to PR-1 planning. It does **not** itself authorize PR-1 code; the next gate is pre-code design review.

---

# 2. Baseline/patient persistence truth

Already proven before PR-1:

- Baseline Audit Steps 1–6 implemented;
- P1–P8 pre-pilot hardening implemented;
- `labs_date` + Step-6 conflict clear-on-collapse implemented;
- 14-scenario synthetic form smoke passed;
- PostgreSQL clinical storage configured and previously verified online;
- protected browser-session `/clinical/*` access implemented;
- Patient → Encounters[] + LabSnapshots[] persistence implemented;
- patient search/load + encounter save/reload browser smoke passed;
- longitudinal laboratory save/history browser smoke passed;
- duplicate lab-history UI removed and `Νέες αναλύσεις` reset smoke passed;
- encounter finalization live browser smoke passed 3/3.

Important caveat:

> Do not claim whole-service GDPR/privacy compliance. Legacy public `/osteoporosis/*`/CORS exposure remains a separate production-security concern.

---

# 3. Product boundary — Clinical Excellence, not an osteoporosis audit application

The project target is a reusable **Personal Clinical Excellence System before, during and after the clinic**. Osteoporosis remains Module 01/proving ground, not the whole product.

Permanent implementation question:

```text
Is this reusable Clinical Excellence Core behavior?
or
Is this Module 01 osteoporosis-specific clinical content?
```

For PR-1 specifically:

```text
CORE owns:
transcript/source intake
→ semantic candidate envelope
→ provenance/temporality/negation/uncertainty
→ provider boundary
→ privacy/logging rules
→ validation/failure semantics
→ candidate review transport

MODULE 01 owns:
osteoporosis concepts
→ fracture/DXA/VFA/risk/labs/treatment/task mapping
→ existing osteoporosis schema targets
```

Do not build an osteoporosis-only transcript engine that must later be rewritten for other modules.

---

# 4. Baseline methodology gate

The next clinical-use sequence remains:

```text
5 pilot encounters
→ one deliberate usability/data-contract refinement
→ freeze Baseline Form + KPI contract
→ 30 scored consecutive unique encounters
→ baseline lock
```

During the scored baseline:

- routine KPI coaching hidden;
- routine Practice Review coaching hidden;
- safety-critical exception path allowed;
- any intervention exposure must be recorded if methodology changes.

PR-1 may be engineered and tested in shadow/capture mode without activating routine coaching.

---

# 5. Active next engineering program — PR-1 PRE-CODE DESIGN REVIEW

Active slice:

```text
PR-1 — Transcript Intake + Candidate Extraction v1
```

Current status:

```text
DESIGN DIRECTION APPROVED
SOURCE INSPECTION STARTED
DETAILED PRE-CODE DESIGN UNDER REVIEW
IMPLEMENTATION NOT STARTED
RUNTIME WRITER NOT CLAIMED
```

The product owner explicitly requested a strong design gate before any PR-1 runtime code, following the project’s canonical discipline.

The design must be sufficiently explicit that a fresh implementation conversation can bootstrap from `main`, understand the exact contracts and seams, claim one writer lock, and implement without reconstructing architecture from chat history.

Frozen first-slice product boundary remains:

```text
paste Heidi transcript
→ protected Core extraction endpoint
→ provider adapter
→ validated typed candidates
→ Module 01 mapping where possible
→ clinically readable preview
→ no authoritative encounter write
→ no raw transcript persistence
```

Full design, failure modes, acceptance evidence and REPLAN triggers live in `SLICE_PLAN_CURRENT.md`.

---

# 6. Source-inspection findings that constrain PR-1 design

Verified repository/runtime facts:

- `main.py` composes the current clinical routers over `legacy_main.py`;
- `/clinical/*` browser access uses the existing clinical-session cookie plus server-side `X-Clinical-Key` injection;
- the new transcript endpoint should remain under `/clinical/*` and must use the same protection model;
- `legacy_main.py` already initializes an OpenAI API client from `OPENAI_API_KEY` and uses Chat Completions for legacy AI functions;
- `requirements.txt` currently specifies `openai>=1.6.0` without an exact pin;
- the Baseline UI is modularly bootstrapped through `static/baseline-audit/app.js`, so transcript intake can be additive rather than rewriting the form;
- existing osteoporosis schemas already define anthropometrics, fracture events, formal risk, DXA/VFA, laboratories, treatment episodes/administrations/decision and follow-up tasks.

Design implication:

> PR-1 should introduce a reusable Clinical Excellence transcript Core plus an osteoporosis mapping profile, not import new transcript semantics directly into Step-specific UI code.

Provider/privacy implication:

> `store=false`/local non-persistence is not equivalent to verified Zero Data Retention. Real identifiable transcript use must remain a separate privacy-readiness decision. PR-1 development and acceptance testing use synthetic/de-identified transcripts only until provider retention/data-control configuration is explicitly verified.

---

# 7. Exact next authorized actions — DESIGN ONLY

The next conversation/session should **not begin by writing runtime code**. It should:

```text
1. bootstrap fresh `main` from all six canonicals
2. confirm PR #29 finalization gate is closed 3/3
3. read `SLICE_PLAN_CURRENT.md` in full
4. inspect only the exact PR-1 seams still needed
5. challenge the Core-vs-Module boundary
6. freeze request/response/candidate contracts
7. freeze privacy/provider/logging behavior
8. freeze module mapping strategy and conflict behavior
9. freeze UI lifecycle and ephemeral-data behavior
10. freeze deterministic + model/eval test matrix
11. list red-line merge blockers and REPLAN triggers
12. present the final implementation plan to the product owner
13. obtain explicit product-owner approval to IMPLEMENT
```

Only after that approval:

```text
claim one runtime writer lock
→ create PR-1 implementation branch
→ implement the frozen slice only
```

---

# 8. Explicitly deferred / forbidden before implementation approval

Do not yet:

```text
WRITE PR-1 runtime code
AUTO-WRITE extracted transcript values into patient records
PERSIST raw transcript
LOG raw transcript or candidate clinical values
SHOW routine Practice Review coaching during scored baseline
CHANGE KPI scoring contract
REDESIGN entire Baseline form
MODIFY Digital Secretary / Setmore / Zadarma
BUILD a polished Clinical Excellence Home
COMMIT identifiable patient/transcript fixtures
ASSUME OpenAI/API retention configuration is suitable for identifiable transcripts
```

---

# 9. New-conversation implementation policy

Starting the actual implementation in a fresh conversation is **preferred**, not a problem.

That is the purpose of the six-canonical control plane. A new implementation conversation should not need a pasted chat summary. It should be able to recover:

```text
fresh main/source identity
product purpose
active slice
approved design
privacy/baseline invariants
active writer state
exact implementation seams
acceptance tests
forbidden scope
next authorized action
```

Recommended fresh-chat opening after design approval:

> Continue the Clinical Excellence project from the repository canonicals. Bootstrap from fresh `main` and all six canonical files. The active slice is PR-1 Transcript Intake + Candidate Extraction v1. Do not rely on prior chat history. Confirm the current writer lock and exact next authorized action before mutating anything.

---

# 10. Handoff completeness rule

Any session ending after a material design/implementation step must update this file so a new conversation can resume without prior chat.

At minimum preserve:

```text
fresh main/source identity
active writer/branch/PR
what was designed
what was implemented
what was tested
what was merged/deployed
what remains unverified
current blocker/HOLD
exact next action
forbidden overlapping action
```

If those cannot be reconstructed from this file plus `SLICE_PLAN_CURRENT.md`, the operational handoff is incomplete.
