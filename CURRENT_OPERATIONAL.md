# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE — CU-1 GLOBAL RICH PHYSIOTHERAPY REFERRAL + CLINICIAN EVIDENCE PANEL.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Approved LET prototype parent:** `feat/cu1-rich-referral-lateral-elbow-2026-08-29` @ `aebfb5a6ee14a0e44d80dd6183a1877d74567b46`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **ACTIVE RUNTIME WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **Runtime authorization:** GLOBAL HORIZONTAL RICH-REFERRAL MODEL + CLINICIAN-ONLY EVIDENCE VIEW + TESTS.
> **Latest product-correction runtime head:** `8007f9f7bddba4a75f102454c347737445bb0cea`.
> **Latest focused CI evidence:** workflow run `33293621220` / run #364 — compile PASS, browser JavaScript syntax PASS, focused Python suite PASS.
> **Deploy/merge authorization:** NO — product-owner representative output review first.
> **Preview deployment:** NOT REQUESTED / NOT AUTHORIZED by product owner.
> **Further route-by-route evidence rollout:** HOLD until context-gated generation behavior is product-reviewed.
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
11. **Selection input is not referral prose.** Checkbox/catalog selections are source data for composition and must not become the output merely because a rich plan failed to resolve.
12. **Context-gated unresolved means generation blocked.** Missing/unsupported context must never fall through to the legacy formatter.

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
- deployment or preview deployment without explicit product-owner instruction;
- weakening route/subtype safety gates;
- runtime interpretation of profile Markdown for trigger/validation logic;
- inventing evidence to make every route appear complete;
- persistence changes;
- CU-2 or PR-1 restart.

## 6.1 Product acceptance defect — context-gated legacy fallback — CORRECTED / TESTED

Product-owner browser testing on 2026-08-30 intentionally omitted `frozen_shoulder_scope` while generating a frozen-shoulder referral. The deployed/main formatter produced a generic checklist-like referral. Inspection of the feature branch then identified the same architectural defect in its fallback path: when a context-gated rich variant did not resolve, `physio_referral_formatter_el_v2.py` could call the legacy formatter.

That behavior is now forbidden horizontally.

Corrected runtime behavior:

```text
context_gated route
+ exactly one reviewed rich variant resolves
→ formatter allowed
→ rich route-specific referral

context_gated route
+ missing / unresolved / unsupported context
→ validation error: rich_referral_context_required
→ formatter_blocked = true
→ no referral text from the API
→ direct formatter fallback also raises CU1ContractError
```

The correction is implemented in:
- `clinic_utilities/physio_route_context.py` — validation-level context gate;
- `clinic_utilities/physio_referral_formatter_el_v2.py` — defense-in-depth ban on legacy fallback.

Regression coverage was updated in:
- `test_cu1_route_context_intake.py`;
- `test_cu1_wording_labels_and_une.py`;
- `test_cu1_shoulder_frozen_rich.py`.

The tests now reproduce deliberate context omission and unsupported contexts for frozen shoulder, post-traumatic neck, cervical dizziness and UNE rather than merely checking `renderer.supports()`.

Exact runtime/test acceptance:

```text
8007f9f7bddba4a75f102454c347737445bb0cea
workflow run 33293621220 / run #364
compile PASS
browser JavaScript syntax PASS
focused Python acceptance suite PASS
```

### Frozen shoulder state after product correction

`shoulder.adhesive_capsulitis_frozen_shoulder` remains evidence-curated for the explicit primary/formal context, but its prior product acceptance is superseded by the 2026-08-30 correction.

```text
formal_diagnosis + frozen_shoulder_scope=primary_frozen_shoulder
→ rich referral allowed

presentation-only
secondary_or_other_stiff_shoulder
not_stated / omitted scope
→ generation BLOCKED
→ no legacy referral text
```

The earlier wording `fail closed to the non-rich path` is explicitly superseded. For a context-gated route, **non-rich legacy fallback is not fail-closed behavior**.

---

# 7. Exact next action

1. Do not create a preview service unless the product owner later asks for one.
2. Keep the context-gated generation correction fixed and exact-head green.
3. Do not resume horizontal route rollout yet.
4. Next product step is representative referral-output inspection using controlled generated examples or a later explicitly authorized preview/deployment path.
5. Merge/deploy remains explicitly on HOLD.
