# Visit Capture — Dia / Heidi / GESY → Cockpit bounded independent R2 pre-code request

> **DATE:** 2026-10-08 Asia/Nicosia.
> **STATUS:** PREPARED / NOT DISPATCHED / READ-ONLY / NO VERDICT.
> **TIER:** R2 — new protected clinical write boundary, patient identity binding, candidate-to-record signoff and longitudinal amendment semantics.
> **TARGET REPOSITORY:** `athpapachr-cmd/osteoporosis`.
> **TARGET BRANCH:** `docs/cockpit-visit-brief-clinical-inbox-2026-10-07`.
> **PARENT DESIGN:** `cockpit/VISIT_BRIEF_CLINICAL_INBOX_DESIGN_2026-10-07.md`, blob `8ee79a96350f9ac42059eeb2a3e836cbe9d11245`.
> **PARENT R2 RECEIPT:** `cockpit/reviews/VISIT_BRIEF_CLINICAL_INBOX_PRECODE_R2_PASS_RECEIPT_2026-10-08.md`.
> **DELTA DESIGN:** `cockpit/VISIT_CAPTURE_DIA_HEIDI_GESY_DELTA_2026-10-08.md`.
> **DELTA DESIGN BLOB:** `7e19ce29a3c33a0d06e6c0ec485952f41c0ecf72`.

## Review purpose

Review ONE bounded material delta only: clinician-reviewed Visit Capture.

The existing Visit Brief + Independent Clinical Inbox review is already PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0. Do not re-review its Q1–Q6 unless this delta directly contradicts an already-closed contract.

Do not implement, edit files, merge, deploy, access live Heidi/GESY/Dia, use patient data, activate Gmail/Zadarma or broaden into Calendar, Reception voice/booking, generic EHR design or GESY automation.

## Product reality

The clinician runs Cockpit inside Dia.

At the end of a visit, selected tabs may be:

1. Heidi summary — today;
2. GESY — today;
3. GESY — immediately previous visit;
4. Cockpit — already-confirmed patient context.

Desired workflow:

```text
selected source tabs
→ Dia candidate synthesis
→ text/structured insertion into Cockpit
→ human preview
→ clinician ONE explicit Save
→ protected existing encounter owner
```

No Dia API integration is proposed. No autonomous clinical write is allowed.

## Mandatory fresh bootstrap

Fresh-verify remote `main`, read the six canonicals in required order, then applicable `PROCEDURES.md`, `cockpit/CURRENT.md`, Product Constitution, parent design, prior PASS receipt and exact delta design.

Classify current drift before relying on any seam. The target may remain on a docs branch even if main has not changed.

Inspect only the source needed to dispose the questions below.

## D1 — Existing ownership / mechanism reuse

Verify whether Visit Capture correctly EXTENDS rather than replaces:
- `clinical_patients.patient_id`;
- `clinical_encounters`;
- current encounter payload/finalization semantics;
- G3/validated longitudinal projections;
- any existing pending/CareTask mechanism if applicable.

Check the current limitation: completed encounter content can become `amended`, but current source may overwrite payload rather than preserve immutable revision history.

PASS if the delta uses the existing rightful clinical owner and names the minimum missing extension.
BLOCK if it creates a competing patient/encounter/longitudinal truth store or assumes amendment history that source does not provide.

## D2 — Identity + write safety

Assess destination binding:

- only confirmed internal `patient_id` may select the write target;
- Dia/GESY/name/phone/tab title cannot retarget it;
- candidate source identity is evidence, not authority;
- changing patient/context invalidates unsaved candidate;
- stale/mismatched capture context/version fails closed;
- clinician sees the candidate before one explicit Save.

Trace the synthetic stale-context case: candidate prepared under patient A; Cockpit changes to B before Save. It must not silently write A content to B.

PASS/BLOCK with one concrete source-grounded reason.

## D3 — Encounter contract / three projections

Assess whether one structured encounter can safely support:
- Snapshot;
- Visit Brief;
- Encounter Detail;

without storing three competing prose truths.

Check minimum semantic support for:
- reason for visit;
- comparison basis / what changed;
- findings;
- clinical impression;
- coding as separate incomplete layer;
- decisions;
- partial medications;
- next contact;
- safety-net only when source-stated;
- uncertainty/conflict;
- provenance/verification/certainty separation.

PASS if sufficient and source-grounded.
BLOCK only for a material missing contract that would force redesign.

## D4 — Longitudinal separation / pending / amendments

Confirm:

```text
ClinicalEncounter
!= PendingItem/CareTask
!= later LongitudinalEvent
!= ClinicalAttentionItem
!= CommunicationAction
!= authoritative accepted lab result
```

Check:
- pending items are first-class and can distinguish responsible owner from external dependency;
- later MRI/lab/review/communication events do not rewrite the historical encounter;
- silent post-signoff overwrite is forbidden;
- either a minimum immutable revision mechanism exists/is added, or post-signoff edit stays fail-closed until it does.

PASS/BLOCK.

## D5 — Dia / Heidi / GESY browser boundary

Assess this first product boundary only:

```text
Heidi today = primary rich source
GESY today = visit/coding/admin supporting source
GESY previous = bounded comparison source
Dia = candidate synthesis/insertion
Cockpit = authenticated clinical write boundary
clinician = final Save authority
```

Check:
- no Dia-selected destination patient;
- no autonomous final record creation;
- no silent source conflict resolution;
- GESY coding not treated as complete clinical truth;
- previous-visit comparison fails closed when unsafe;
- raw page bodies are not persisted merely because they were used;
- real identifiable Dia/Heidi/GESY use remains a live processor/privacy qualification.

PASS/BLOCK.

## D6 — Narrow first-code eligibility

If D1–D5 pass, state the smallest implementation that can start under separate writer authority.

Expected maximum first-code boundary:

1. confirmed-patient Visit Capture surface;
2. synthetic/manual `VisitCaptureCandidateV1` insertion;
3. strict validation and capture-context stale guard;
4. one preview rendering Snapshot / Visit Brief / Encounter Detail;
5. one explicit Save to the existing protected encounter owner;
6. bounded pending integration;
7. source provenance without raw source persistence;
8. no live Dia/Heidi/GESY access;
9. no Gmail;
10. no Zadarma;
11. no authoritative lab-result acceptance.

Required synthetic cases:
- straightforward capture;
- no safe previous comparison;
- cross-patient/conflicting source;
- stale patient context;
- partial medication;
- later event leaves encounter unchanged;
- post-signoff correction cannot silently overwrite;
- all 3 projections come from one structured encounter.

List every remaining LIVE-ACTIVATION gate separately. Do not treat an open live gate as a synthetic-code blocker unless it is actually required to build the synthetic slice.

## Evidence discipline

Use finite evidence. Do not rerun broad suites. Do not inspect live patient/source systems.

A reachable material blocker:
- finish only its direct actionable trace;
- return BLOCK;
- give the smallest correction;
- STOP.

If all D1–D6 are disposed:
- PASS / COMPLETE_FOR_DECLARED_SCOPE;
- P0:P1:P2 count;
- exact narrow first-code boundary;
- remaining live gates;
- STOP.

No correction implementation by the reviewer. No review-of-review. No second speculative review.

## Required handback

Return:

```text
REVIEW =
VISIT CAPTURE DIA/HEIDI/GESY → COCKPIT DELTA R2 PRE-CODE

REMOTE MAIN =
<fresh SHA>

TARGET =
<branch/head + exact delta design blob>

VERDICT =
PASS / BLOCK / UNKNOWN

COVERAGE =
COMPLETE / PARTIAL

P0:P1:P2 =
x:y:z

D1 =
...

D2 =
...

D3 =
...

D4 =
...

D5 =
...

D6 =
...

NARROW FIRST-CODE BOUNDARY =
...

REMAINING LIVE-ACTIVATION GATES =
...

MATERIAL FINDINGS =
...

STOP =
why the review stops here
```

A PASS grants design disposition only. Implementation still requires a separately recorded writer/contract authorization under current governance.
