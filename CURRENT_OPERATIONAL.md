# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-1 RELEASE PR OPEN / MERGE GATE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Release branch:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
> **Pre-PR tested head:** `8628dbcd9320e91497d49e1c223cc51a810cd51b`.
> **Release PR:** `#64` → `main`.
> **Inherited tested C1 head:** `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **ACTIVE CANONICAL WRITER/LOCK:** release branch — PR/release closeout only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **Merge/deploy/production smoke:** MERGE AUTHORIZED BY PRODUCT OWNER / NOT YET DONE.

---

# 1. Release authorization

The product owner explicitly authorized release of G-1 after G1-R1/R2 were closed and the full G-1+C1 suite passed.

Authorized path:

```text
fresh six-canonical bootstrap
→ fresh main verification
→ exact full compare/review
→ PR
→ squash merge
→ normal Render auto-deploy
→ production synthetic smoke
→ canonical release closeout
```

PR-1/PR-2, new taxonomy/milestone rules, physiotherapy/RF mutation and real pilot data remain outside this release.

---

# 2. Release evidence before PR

Fresh release bootstrap verified:

```text
main = 08ecd3ab33e98d567c47042a8a1de482df6952b9
release head = 8628dbcd9320e91497d49e1c223cc51a810cd51b
```

Exact compare is directly ahead of main and contains only accepted Module-01 canonical/design artifacts, C1 finalization files/tests and G-1 guidance files/tests. No parked physiotherapy/RF runtime is included.

Fresh exact-head check-run evidence at `8628dbcd...`:

```text
workflow: G1 progressive guidance foundation
run:      33329427497
result:   SUCCESS
```

This includes:

- JavaScript syntax;
- G-1 core regression;
- G-1 wiring/ownership regression;
- G1-R1/R2 UI-state regression;
- C1 authoritative Finish browser regression;
- C1 server finalization lifecycle regression.

---

# 3. PR state

```text
PR:       #64
base:     main @ 08ecd3ab33e98d567c47042a8a1de482df6952b9
head:     feat/module01-g1-progressive-guidance-foundation-2026-08-30
state:    OPEN
mergeable: YES at initial PR inspection
```

Because this operational closeout commit moves the PR head, PR-head CI must pass again before merge.

---

# 4. Release invariants

Must remain true before merge:

```text
main has not moved unexpectedly
PR head contains no unrelated parked scope
G1 full suite SUCCESS at exact PR head
PR mergeable
C1 ancestry preserved
no real patient data/transcript committed
```

After squash merge:

```text
Render auto-deploy only; do not manually trigger duplicate deploy
verify exact deployed main commit
production synthetic smoke:
- G-1 page/bootstrap loads
- dropdown + quick context produce guidance / WHY NOW
- longitudinal history unavailable state is explicit and never false zero
- loaded empty history can legitimately show zero
- C1 authoritative Finish confirms protected completed/amended state and reload behavior using synthetic data
```

Do not mark production-smoke verified without direct evidence.

---

# 5. Current status matrix

```text
C1 IMPLEMENTED / TESTED                    YES
G-1 IMPLEMENTED / TESTED                   YES
G1-R1 / G1-R2                              CLOSED / TESTED
PR #64                                     OPEN
MERGED                                     NO
DEPLOYED                                   NO
PRODUCTION-SMOKE-VERIFIED                  NO
PR-1 HEIDI                                 NOT IMPLEMENTED
PR-2 REVIEW/POPULATION                     NOT IMPLEMENTED
REAL 5-CASE PILOT                          NOT STARTED
MODULE 01 CLOSED                           NO
```

---

# 6. Exact next action

```text
wait for exact PR-head CI
→ inspect complete PR diff / mergeability / main freshness
→ if PASS, squash-merge PR #64 using exact expected head SHA
→ verify new main SHA
→ verify Render auto-deploy exact commit
→ perform production synthetic smoke if available through connected/browser tooling
→ record release evidence and append changelog
```

If any release invariant fails, STOP before merge or before declaring production validation.