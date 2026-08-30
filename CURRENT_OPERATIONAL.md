# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-1 RELEASE REVIEW BLOCKED / RUNTIME CORRECTION REQUIRED.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Release-review branch:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
> **Reviewed branch head before this canonical update:** `a7cc4277b57075dd6f0f0e721b12052da77eed25`.
> **Inherited tested C1 head:** `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30` — release-review closeout only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR/merge/deploy/production smoke:** NOT DONE / NOT AUTHORIZED.

---

# 1. Release-review result

A fresh six-canonical bootstrap was completed from current remote `main`, followed by an exact compare/review of the accepted Module-01 branch chain.

Ancestry/scope findings:

```text
main 08ecd3ab33e98d567c47042a8a1de482df6952b9
→ C1 authoritative Finish a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871
→ G-1 progressive guidance a7cc4277b57075dd6f0f0e721b12052da77eed25
```

The G-1 branch is directly descended from the exact C1 head. The complete `main...G-1` compare contains only accepted Module-01 canonicals/contracts, C1 finalization files/tests and G-1 guidance files/tests. No parked physiotherapy/RF runtime work is included.

Open draft PR #63 remains unrelated parked CU-1 work and must not be merged as part of Module 01.

---

# 2. Existing test evidence remains valid but is not sufficient for release

Exact-head G-1 CI before release review:

```text
workflow: G1 progressive guidance foundation
run:      33327944349
head:     a7cc4277b57075dd6f0f0e721b12052da77eed25
result:   SUCCESS
```

Passed:

- JavaScript syntax;
- progressive-guidance core regressions;
- guidance wiring/ownership regression;
- authoritative Finish browser regression;
- server finalization lifecycle regression.

The release review found runtime states that these tests do not cover.

---

# 3. RELEASE BLOCKER G1-R1 — unavailable history is silently presented as empty history

Current `progressive-guidance-ui.js` behavior:

```text
protected GET /clinical/patient/{patient_id}/encounters fails
→ catch
→ historicalEncounters = []
→ render continues
→ active patient can be shown as having 0 previous completed/amended encounters
```

This is not acceptable for clinical guidance.

Required invariant:

```text
HISTORY UNAVAILABLE != NO HISTORY
AUTH/NETWORK/SERVER FAILURE != ZERO PRIOR ENCOUNTERS
```

The guidance layer must carry an explicit history-load state such as:

```text
not_loaded / loading / loaded / unavailable
```

When history is unavailable:

- do not claim `0 previous encounters`;
- do not imply longitudinal guidance is complete;
- visibly state that prior context could not be loaded;
- keep current-visit/local guidance usable where safe;
- do not derive absence-based longitudinal conclusions.

This is a release blocker because silent loss of longitudinal context can suppress unresolved-prior or treatment-timeline guidance.

---

# 4. RELEASE BLOCKER G1-R2 — live UI clearing can fall back to stale persisted context

`currentCaseSnapshot()` currently uses truthy fallback patterns such as:

```text
DOM value || persisted value
```

and only replaces fracture-event state when the live DOM contains one or more events.

Consequences before Save can include:

- clearing `quick_notes` still displaying the previous persisted note;
- clearing/resetting an optional current field falling back to the old stored value;
- deleting all visible fracture events leaving old persisted fracture events in the guidance snapshot;
- stale guidance remaining surfaced until another persistence/synchronization step updates the cache.

Required invariant:

```text
IF A LIVE CONTROL EXISTS
→ its current value, including explicit blank/empty state, owns today's in-memory guidance snapshot
```

Persisted state is only fallback when the corresponding live control is absent, not when its current value is empty.

For live fracture-event UI, an empty rendered event list must project as an empty current list rather than resurrecting prior local-cache events.

This is a data-state integrity release blocker for the guidance presentation layer.

---

# 5. C1 authoritative Finish review

C1 ancestry remains intact and its tested ownership model is preserved:

```text
coordinator
→ local Save/flush
→ local pilot-completion payload marker
→ strict protected completed sync
→ protected success shown only after server confirmation
```

On protected finalization failure, local data including the completion marker are intentionally retained for retry while protected completion is explicitly reported as unconfirmed. This behavior is already an accepted C1 contract; the release review did not identify a new regression in C1 wiring.

Production C1 behavior remains unproven until merge/deploy/synthetic smoke.

---

# 6. Current status matrix

```text
C1 IMPLEMENTED / TESTED                    YES
C1 MERGED / DEPLOYED / PROD-SMOKED         NO
G-1 IMPLEMENTED / PREVIOUSLY TESTED        YES
G-1 RELEASE-READY                          NO — BLOCKED G1-R1 + G1-R2
G-1 MERGED / DEPLOYED / PROD-SMOKED        NO
PR-1 HEIDI                                 NOT IMPLEMENTED
PR-2 INLINE REVIEW/POPULATION               NOT IMPLEMENTED
5-CASE SYSTEM-ASSISTED PILOT               NOT STARTED
30-CASE SYSTEM-ASSISTED BASELINE            NOT STARTED
MODULE 01 CLOSED                           NO
```

---

# 7. Exact next authorized action

STOP release activity. Do not open/merge/deploy G-1 in the present state.

Next action requires a separate product-owner authorization for one bounded runtime correction:

```text
fresh main verification
→ claim runtime writer on the existing G-1 branch or a dedicated correction branch
→ fix G1-R1 explicit history availability state
→ fix G1-R2 live-DOM-over-persisted snapshot semantics
→ add focused browser/core regressions for both failure modes
→ run full G-1 + C1 regression workflow at exact correction head
→ release-review exact compare again
→ STOP at release gate
```

Only after that correction passes may a separate release decision authorize PR/merge/Render auto-deploy/production smoke.

No physiotherapy/RF mutation. No PR-1/PR-2 expansion in this correction.