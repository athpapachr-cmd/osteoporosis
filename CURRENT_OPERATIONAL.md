# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — PROGRESSIVE GUIDANCE FOUNDATION DESIGN COMPLETE / PRE-RUNTIME STOP.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Current design branch:** `design/module01-progressive-guidance-foundations-2026-08-30`.
> **Parent G-0 design:** `design/module01-dynamic-guided-visit-replan-2026-08-30` @ `6aadc8ef55719be98233afa6a80a179f43512c1d`.
> **Parent tested runtime ancestry:** `fix/module01-c1-authoritative-finish-2026-08-30` @ `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE — refinement complete.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **Runtime mutation:** NOT AUTHORIZED by this design refinement.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product-owner clarification now authoritative

The dynamic visit architecture is intentionally progressive.

The product owner does **not** require a complete visit taxonomy/card model before first real use. The immediate objective is to install extensible foundations and let actual encounters/transcripts drive later refinement.

Current practical capture model:

```text
first-page dropdown
→ coarse clinician-declared visit type

first-page short notes / free text
→ fallback visit context when dropdown is too coarse or no option fits

later Heidi transcript
→ richer evidence about what actually occurred
```

The system may later propose better classification, but clinician confirmation remains required.

---

# 2. Progressive classification policy

Do not front-load a complete set of visit subtypes or card rules.

Start with coarse categories such as:

```text
initial assessment
results/work-up review with management decision
treatment start
treatment administration / continuation
repeat prescription / routine maintenance
treatment change / transition
fracture / fracture-on-treatment
adverse effect / intolerance
other
```

Existing runtime archetypes may be reused where sufficiently close. `other` plus concise summary is valid and preferable to forcing an inaccurate category.

---

# 3. Learning-from-use checkpoints

Refine visit taxonomy/cards deliberately from real use:

```text
minimum foundation
→ 5 system-assisted pilot encounters
→ one deliberate refinement
→ 30-case system-assisted baseline
→ post-baseline evidence-from-use refinement
→ later periodic/milestone refinement when enough new evidence accumulates
```

Review at these checkpoints:

- actual dropdown usage;
- frequency/content of `other`;
- recurring free-text visit summaries;
- recurring transcript patterns;
- missing/irrelevant cards;
- duplicate questioning/data entry;
- clinically important omissions;
- repeated unresolved tasks;
- treatment/event patterns that may justify new milestone rules.

No autonomous taxonomy mutation from frequency or model output alone.

---

# 4. Minimum G-1 runtime ambition

If separately authorized, G-1 should build only the useful extensible foundation:

```text
coarse visit intent
+ short visit context
+ minimum read-only longitudinal context
+ basic dynamic card applicability/ordering
+ WHY NOW
+ safety/new-event override
+ unresolved-prior override
+ treatment/due plumbing
```

G-1 does not need complete drug-specific guidance, complete therapy milestones, predictive encounter classification, or a complete card library.

The richer G-0 objects remain extension seams, not first-runtime completeness requirements.

---

# 5. Heidi position

Heidi remains required before the five-case real pilot unless separately replanned because the known manual workflow is not the intended product.

Preferred future flow:

```text
coarse clinician visit label
+ optional short summary
+ Heidi transcript extraction
→ structured candidates
→ provisional in-place population
→ Accept / Edit / Reject
→ authoritative encounter data
```

Transcript patterns may later inform category/card redesign but do not silently reclassify encounters or mutate taxonomy.

---

# 6. C1 authoritative Finish state preserved

```text
branch: fix/module01-c1-authoritative-finish-2026-08-30
head:   a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871
state:  IMPLEMENTED / TESTED
CI:     exact-head run 33323204227 SUCCESS
MERGED: NO
DEPLOYED: NO
PRODUCTION-SMOKE: NO
```

No real pilot should start until finalization integrity is released/deployed/smoke-verified and the intended minimum system-assisted workflow is available.

---

# 7. Physiotherapy remains parked/preserved

```text
feat/cu1-rich-referral-global-evidence-2026-08-29
@ bdd23b83a8252405f5aa5a0c0b67f303ccfcef5f
IMPLEMENTED / TESTED / PRODUCT-OWNER REVIEWED
MERGED NO / DEPLOYED NO
```

Do not mutate/merge/deploy during Module 01 work without separate authorization.

---

# 8. Status matrix

```text
G-0 DYNAMIC GUIDANCE ARCHITECTURE             DESIGN-COMPLETE
G-0.1 PROGRESSIVE-FOUNDATION REFINEMENT       COMPLETE
FULL VISIT TAXONOMY REQUIRED NOW               NO
COARSE DROPDOWN AS FIRST-LINE LABEL            YES
FREE-TEXT FALLBACK CONTEXT                     YES
LATER TRANSCRIPT-INFORMED REFINEMENT           YES
AUTOMATIC TAXONOMY MUTATION                     NO
G-1 RUNTIME                                    NOT IMPLEMENTED
PR-1 / PR-2                                    NOT IMPLEMENTED
5-CASE SYSTEM-ASSISTED PILOT                   NOT STARTED
30-CASE SYSTEM-ASSISTED BASELINE               NOT STARTED
MODULE 01 CLOSED                               NO
```

---

# 9. Exact next action

STOP design refinement.

If the product owner separately authorizes runtime implementation:

```text
fresh six-canonical bootstrap
→ create bounded G-1 runtime branch
→ implement minimum progressive guidance foundation only
→ synthetic regression
→ stop at tested implementation / release gate
```

C1 merge/deploy remains a separate release decision. PR-1/PR-2 remain later bounded runtime slices before real pilot use.
