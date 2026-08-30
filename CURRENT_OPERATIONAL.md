# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-1 R1/R2 BOUNDED RUNTIME CORRECTION ACTIVE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Implementation/correction branch:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
> **Release-review block head:** `56267d08dc5d68b8c5e4208f2ae3761fa15156b5`.
> **Inherited tested C1 head:** `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30` — correction state/evidence only.
> **ACTIVE RUNTIME WRITER/LOCK:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30` — **ONLY G1-R1 + G1-R2 + focused regression/CI wiring**.
> **PR/merge/deploy/production smoke:** NOT DONE / NOT AUTHORIZED.

---

# 1. Product-owner authorization

The product owner explicitly authorized:

```text
fix R1 + R2
+ new focused regressions
+ full G-1 + C1 CI
```

This does **not** authorize PR creation, merge, deployment, production smoke, PR-1/PR-2 work, taxonomy expansion, medication-specific milestone logic, physiotherapy or RF mutation.

---

# 2. Preserved release-review findings

Fresh release review verified the accepted ancestry:

```text
main 08ecd3ab33e98d567c47042a8a1de482df6952b9
→ C1 authoritative Finish a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871
→ G-1 progressive guidance
```

The branch contains accepted Module-01 canonical/contracts, C1 finalization files/tests and G-1 guidance files/tests only. Parked physiotherapy/RF work is excluded.

Exact-head G-1 CI before this correction was successful at `a7cc4277b57075dd6f0f0e721b12052da77eed25` (run `33327944349`), but release review identified two uncovered state-integrity blockers.

---

# 3. G1-R1 — explicit longitudinal-history availability state

Defect:

```text
protected historical encounter fetch fails
→ historicalEncounters = []
→ UI can imply zero prior encounters
```

Required invariant:

```text
HISTORY UNAVAILABLE != NO HISTORY
AUTH/NETWORK/SERVER FAILURE != ZERO PRIOR ENCOUNTERS
```

Bounded correction requirements:

- carry explicit `not_loaded / loading / loaded / unavailable` history state;
- clear old-patient history immediately when a different/current patient history load begins;
- on failure, do not claim zero previous encounters;
- visibly state that longitudinal context is unavailable/incomplete;
- keep current local visit guidance usable;
- no absence-based longitudinal conclusion from unavailable history.

---

# 4. G1-R2 — live DOM owns current in-memory guidance snapshot

Defect:

```text
live DOM value || persisted cache value
```

can resurrect stale persisted values after the clinician clears a live field before Save.

Required invariant:

```text
IF A LIVE CONTROL EXISTS
→ its present value, including blank/empty, owns today's in-memory guidance snapshot

persisted cache
→ fallback only when the live control/root is absent
```

This applies at minimum to:

- encounter archetype;
- encounter date;
- quick notes;
- interval fracture status;
- live fracture-event collection, including an explicitly empty rendered list.

---

# 5. Required regression evidence

Add focused synthetic regressions that prove:

```text
R1-A protected history fetch failure → state=unavailable
R1-B unavailable history summary never claims 0 previous visits
R1-C beginning a new patient/history load does not retain prior patient's historical rows
R1-D successful empty history is distinct: state=loaded + 0 is allowed
R2-A live blank quick_notes overrides persisted nonblank quick_notes
R2-B live blank select/date/status overrides persisted value when control exists
R2-C live empty fracture-event container overrides persisted fracture events with []
R2-D persisted fallback remains available only when corresponding live control/root is absent
```

Then run the complete existing G-1 workflow, preserving:

- progressive guidance core regressions;
- guidance wiring/ownership regression;
- authoritative Finish browser regression;
- server finalization lifecycle regression.

---

# 6. Current status matrix

```text
C1 IMPLEMENTED / TESTED                    YES
C1 MERGED / DEPLOYED / PROD-SMOKED         NO
G-1 BASE IMPLEMENTED / PREVIOUSLY TESTED   YES
G1-R1 CORRECTION                           ACTIVE
G1-R2 CORRECTION                           ACTIVE
G-1 RELEASE-READY                          NO — pending correction + exact-head CI
PR-1 HEIDI                                 NOT IMPLEMENTED
PR-2 INLINE REVIEW/POPULATION              NOT IMPLEMENTED
5-CASE SYSTEM-ASSISTED PILOT               NOT STARTED
30-CASE SYSTEM-ASSISTED BASELINE            NOT STARTED
MODULE 01 CLOSED                           NO
```

---

# 7. Exact next action

```text
1. modify only progressive-guidance state/snapshot seams required by R1/R2
2. add focused synthetic regression(s)
3. wire focused test into G-1 workflow
4. run exact-head full G-1 + C1 CI
5. inspect exact correction diff and CI evidence
6. update canonicals, release runtime writer
7. STOP at release gate
```

No PR/merge/deploy/production smoke in this authorization.