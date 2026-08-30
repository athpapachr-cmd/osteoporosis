# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-1 PROGRESSIVE GUIDANCE FOUNDATION IMPLEMENTATION ACTIVE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Implementation branch:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
> **Parent accepted design:** `design/module01-progressive-guidance-foundations-2026-08-30` @ `298d8d525f1bac97ffb6904fe09800519bd1a584`.
> **Parent G-0 design:** `design/module01-dynamic-guided-visit-replan-2026-08-30` @ `6aadc8ef55719be98233afa6a80a179f43512c1d`.
> **Parent tested runtime ancestry:** `fix/module01-c1-authoritative-finish-2026-08-30` @ `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
> **ACTIVE RUNTIME WRITER/LOCK:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30` — bounded G-1 files only.
> **Merge/deploy/production smoke:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product-owner authorization

The product owner authorized continuation after clarifying that the first runtime should establish only extensible foundations rather than a complete visit taxonomy.

Approved minimum runtime intent:

```text
coarse visit dropdown
+ short first-page visit context/free text
+ minimum read-only longitudinal context
+ basic dynamic card relevance/ordering
+ WHY NOW
+ event/unresolved-prior/treatment-due plumbing
```

The runtime must be capable of later learning from real visit/transcript patterns, but it must not autonomously mutate taxonomy or clinical guidance.

---

# 2. Explicit G-1 scope

G-1 may implement:

- read-only longitudinal projection from protected historical encounters;
- current encounter context using existing runtime fields;
- current `encounter_archetype` as coarse clinician-declared visit intent;
- existing first-page free text/quick notes as fallback encounter context;
- deterministic, non-treatment-recommending card relevance/order mechanics;
- `why now` reasons;
- event/fracture override plumbing;
- unresolved prior item resurfacing where deterministically available;
- treatment administration/due context plumbing from existing data;
- synthetic tests for the above.

G-1 must remain minimal and extensible.

---

# 3. Explicitly out of scope / forbidden

Do not in G-1:

- build a complete osteoporosis visit taxonomy;
- add one form per Prolia/dose number;
- invent exact 4th/8th/10th-dose rules;
- invent medication-specific monitoring/safety cadence;
- implement PR-1 transcript provider/API extraction;
- implement PR-2 Accept/Edit/Reject population;
- implement Practice Review runtime;
- mutate physiotherapy/RF scope;
- merge/deploy to `main` without separate authorization;
- use real patient/transcript content in tests or public repository files.

---

# 4. Progressive classification invariant

```text
encounter_archetype = coarse clinician-declared label
quick_notes / short visit summary = fallback context
future transcript patterns = evidence for later taxonomy/card refinement
```

`other` plus concise context remains valid. No forced inaccurate category.

Taxonomy/card changes after the 5-case pilot, 30-case baseline or later milestones remain clinician-reviewed deliberate changes, not autonomous learning.

---

# 5. C1 authoritative Finish preserved

```text
branch: fix/module01-c1-authoritative-finish-2026-08-30
head:   a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871
state:  IMPLEMENTED / TESTED
CI:     exact-head run 33323204227 SUCCESS
MERGED: NO
DEPLOYED: NO
PRODUCTION-SMOKE: NO
```

G-1 inherits the tested code ancestry. It does not authorize C1 release.

---

# 6. Physiotherapy remains parked/preserved

```text
feat/cu1-rich-referral-global-evidence-2026-08-29
@ bdd23b83a8252405f5aa5a0c0b67f303ccfcef5f
IMPLEMENTED / TESTED / PRODUCT-OWNER REVIEWED
MERGED NO / DEPLOYED NO
```

Do not mutate/merge/deploy during G-1.

---

# 7. Current status

```text
G-0 DYNAMIC GUIDANCE DESIGN                  COMPLETE
G-0.1 PROGRESSIVE FOUNDATION REFINEMENT      COMPLETE
G-1 IMPLEMENTATION                           ACTIVE
G-1 TESTED                                   NO
G-1 MERGED                                   NO
G-1 DEPLOYED                                 NO
PR-1 / PR-2                                  NOT IMPLEMENTED
5-CASE SYSTEM-ASSISTED PILOT                 NOT STARTED
30-CASE SYSTEM-ASSISTED BASELINE             NOT STARTED
MODULE 01 CLOSED                             NO
```

---

# 8. Exact next action

```text
inspect exact G-1 runtime/schema seams
→ implement bounded progressive guidance mechanics
→ add synthetic regressions
→ run focused tests
→ update canonicals with exact evidence
→ STOP at tested implementation / release gate
```

No merge/deploy unless separately authorized.