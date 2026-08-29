# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE — CU-1 GLOBAL RICH PHYSIOTHERAPY REFERRAL + CLINICIAN EVIDENCE PANEL.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Approved LET prototype parent:** `feat/cu1-rich-referral-lateral-elbow-2026-08-29` @ `aebfb5a6ee14a0e44d80dd6183a1877d74567b46`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **ACTIVE RUNTIME WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **Runtime authorization:** GLOBAL HORIZONTAL RICH-REFERRAL MODEL + CLINICIAN-ONLY EVIDENCE VIEW + TESTS.
> **Latest clinically tested head before docs-only closeout:** `3364d1b6f9e749ccad8bac059a4cb6d5b54d4ed4`.
> **Latest focused CI evidence:** workflow run `33268968382` / run #357 — compile PASS, browser JavaScript syntax PASS, Python acceptance suite **116/116 PASS**.
> **Deploy/merge authorization:** NO — global coverage review and product-owner representative output review first.
> **Further route-by-route evidence research:** HOLD unless a concrete evidence gap blocks safe global rendering.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused.

---

# 1. Product-owner approval — LOCKED

On 2026-08-29 the product owner approved the lateral-elbow referral product shape and authorized horizontal application across the physiotherapy condition registry using the corresponding guideline/literature authority for each route.

The global model must not be implemented as one custom formatter branch per disease. It must use one shared rehabilitation-document model with route/subtype/context-specific content supplied as structured data/evidence authority.

The approved referral principles are:

```text
COMPACT CLINICAL CONTEXT
+
EARLY MANAGEMENT / INITIAL CAPACITY
+
PROGRESSIVE CAPACITY RESTORATION
+
FUNCTIONAL RETURN
+
RECURRENCE-RISK REDUCTION / SELF-MANAGEMENT
```

For detailed output, the document may organize this into clinically meaningful stages with:

```text
STAGE
→ GOALS
→ INTERVENTION DIRECTIONS (how goals are pursued)
→ TELEGRAPHIC PROGRESS MARKERS
```

The stage structure is document/clinical organization. It must never be falsely labelled as a universal evidence-validated phase protocol when the cited source does not establish that protocol.

---

# 2. GeSY output policy

Hard ceiling:

```text
2000 characters
```

Target for ordinary Detailed output:

```text
~1500–1850 characters
```

The formatter must compose below the limit by design. Mechanical clipping is only an abnormal final fail-safe and must not be relied upon to remove safety/reassessment content.

Clinical context may include only information actually supplied/selected:
- diagnosis/presentation wording;
- relevant findings;
- functional limitation;
- explicit work/sport/activity context;
- material restriction/protocol context when applicable.

Do not repeat unnecessary field labels. Do not invent patient facts, tasks, causality or deficits.

Short output is a coherent narrative with flow, not a checklist. Detailed output remains compact but preserves goals, intervention directions and progression logic.

---

# 3. Global rehabilitation invariants

1. **Goal without method is insufficient.** A goal such as pain reduction, mobility, strength or return to activity must be accompanied by the route-appropriate way it is pursued.
2. **Passive-only care is not a complete routine rehabilitation plan** unless a route/protocol context explicitly prevents active progression.
3. **No micromanagement.** Do not prescribe universal sets, repetitions, kilograms, hold times, fixed weeks or unsupported numerical clearance thresholds.
4. **Therapist execution remains therapist-owned.** Evidence tagged `therapist_execution_detail` must not automatically enter referral prose.
5. **Evidence applicability is route/subtype/context-specific.** Never borrow a recommendation from a neighboring diagnosis simply to fill a gap.
6. **Evidence gap is explicit.** A blocked or insufficiently covered route must not receive a generic evidence-labelled treatment sequence.
7. **Patient-specific protocol wins.** Written postoperative/fracture/healing restrictions override conflicting route defaults.
8. **Short and Detailed are two renderings of the same clinical truth.**
9. **Stage 3 includes both function and prevention.** Where clinically applicable, return to actual daily/work/sport demands is paired with load self-management, relevant ergonomic/technical modification and maintenance of required capacity to reduce recurrence risk.
10. **Reassessment remains compact but preserved** where route-specific safety or failure-to-progress authority requires it.

---

# 4. Evidence-source policy

The existing structured CU-1 evidence corpus remains the source of literature identity, applicability, recommendation direction, source class, strength/certainty, freshness and evidence-gap state.

The runtime must preserve these distinctions:

```text
referral_core
therapist_execution_detail
clinician_ui_only
```

Only `referral_core` authority may automatically support referral treatment prose. `therapist_execution_detail` remains available for evidence transparency but is not converted into physician exercise prescription. `clinician_ui_only` may inform the clinician evidence view, limitations and safety context but does not automatically enter the GeSY referral.

Conflicting guideline frameworks, low/very-low certainty findings and evidence gaps must remain visible as such; they must not be silently blended into a stronger recommendation.

---

# 5. Clinician-only evidence panel — AUTHORIZED

Add a separate UI section, outside the GeSY text box, for the clinician to inspect the evidence behind the selected route/subtype/context.

Preferred label:

```text
Τεκμηρίωση / Παραπομπές
```

It should show human-readable information only:
- source title;
- organization/authors;
- year/version;
- journal/guideline reference;
- DOI/link when available;
- recommendation strength/certainty where attached to the relevant claim;
- short `Υποστηρίζει:` explanation for the claim/component;
- freshness / reviewed-on state;
- evidence gaps or important limitations/conflicts.

Internal machine IDs must not be shown to the user-facing referral. The evidence panel is clinician-only UI and is not copied or printed as part of the GeSY referral by default.

---

# 6. Current implementation boundary

Authorized now:
- shared/global rich-referral document model;
- route/subtype/context evidence resolution from structured machine artifacts;
- Short + Detailed rendering under the 2000-character ceiling;
- explicit evidence-gap behavior instead of generic fallback;
- clinician-only evidence panel/API;
- deterministic cross-route tests;
- canonical/changelog/PR documentation.

Not authorized now:
- direct merge to `main`;
- deployment before exact-head review;
- weakening route/subtype safety gates;
- runtime interpretation of profile Markdown for trigger/validation logic;
- inventing evidence to make every route appear complete;
- persistence changes;
- CU-2 or PR-1 restart.

## 6.1 Frozen shoulder route — CLOSED / TESTED ON FEATURE BRANCH

`shoulder.adhesive_capsulitis_frozen_shoulder` is complete for the currently approved CU-1 rich-referral scope.

Validated behavior:

```text
formal_diagnosis + frozen_shoulder_scope=primary_frozen_shoulder
→ context-gated rich referral + exact clinician evidence projection

presentation-only
secondary_or_other_stiff_shoulder
not_stated / unresolved scope
→ no primary-frozen-shoulder rich authority; fail closed to the non-rich path / evidence-context gap
```

The rich projection is deliberately a **single evidence-bounded organizational stage** for individualized mobility/ROM, function and self-management. It is not a `freezing → frozen → thawing` protocol and contains no universal treatment duration, fixed ROM threshold or disease-stage transition rule.

Evidence-scope boundaries remain locked:
- manual therapy including ROM may be considered within primary frozen shoulder authority;
- self-stretching remains `therapist_execution_detail` and no fixed dose is generated;
- strengthening is not a mandatory routine direction because current evidence is insufficient/very low certainty;
- BESS uncertainty about supervised physiotherapy versus natural history remains visible and is not converted into a superiority claim;
- post-injection physiotherapy remains a context-specific evidence claim and is not auto-rendered into the general primary-frozen-shoulder plan;
- secondary/post-traumatic/postoperative or otherwise different stiff-shoulder contexts do not borrow primary frozen shoulder authority.

Acceptance evidence at `3364d1b6f9e749ccad8bac059a4cb6d5b54d4ed4`:
- frozen route test is part of `.github/workflows/cu1-tests.yml`;
- exact context-gating and invalid-enum fail-closed behavior PASS;
- primary-vs-secondary/unresolved evidence isolation PASS;
- clinician evidence-panel context isolation PASS;
- coverage amendment / fixture linkage PASS;
- Short/Detailed content and character ceilings PASS;
- no fixed-stage, fixed-dose or mandatory-strengthening leakage PASS;
- whole focused suite: **116/116 PASS**.

State distinction:

```text
FROZEN SHOULDER DESIGNED      YES
IMPLEMENTED                   YES
TESTED                        YES
MERGED                        NO
DEPLOYED                      NO
PRODUCTION-SMOKE-VERIFIED     NO
```

No further frozen-shoulder implementation work is required unless later evidence review or product feedback creates a specific replan trigger.

---

# 7. Exact next action

1. Treat frozen shoulder as closed for the current feature-branch scope; do not reopen it for cosmetic refinement.
2. Continue the horizontal CU-1 rollout with the next unresolved registry route, preserving route/context-specific evidence authority and explicit block states.
3. Keep exact-head CI green after each bounded route batch.
4. When representative route classes and remaining coverage are adequate, perform the global representative Short/Detailed + clinician-evidence review before PR/merge/deploy.
5. Merge/deploy remains explicitly on HOLD until product-owner review and global acceptance are complete.
