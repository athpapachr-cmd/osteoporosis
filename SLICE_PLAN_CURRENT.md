# SLICE_PLAN_CURRENT.md — Knee-OA v5 Product Owner review HOLD

> **STATUS:** REVIEW HOLD — v4 remains released/live; authenticated live product smoke passed; v5 is implemented/tested but unmerged/unreleased.
> **Production runtime:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Authenticated live smoke:** run `34681808255`, attempt `3` — SUCCESS.
> **V5 branch:** `fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12`.
> **V5 exact tested substantive head:** `a47357c602120d3678e8f2f23b99775e616c79e1`.
> **V5 full gate:** `34693751545` — SUCCESS.
> **Writer:** NONE.
> **Next implementation slice:** NONE ACTIVE.

## 1. Closed production boundary

The v4 production lifecycle now has authenticated verification in addition to the earlier public-asset and unauthenticated auth-boundary smoke.

Authenticated run `34681808255`, attempt `3`, proved protected page/bootstrap access, deterministic Knee-OA projection and safety fail-closed behavior using only synthetic/non-identifiable smoke state. No patient persistence or patient identifiers were used.

## 2. Tested v5 candidate under Product Owner review

The prior Product Owner session produced a bounded v5 candidate addressing post-use friction only.

### Optional symptom refinement

```text
Πόνος / Δυσκαμψία / Αδυναμία inactive first tap
→ select generic symptom only
→ no forced popup

second tap while active
→ optional focused detail sheet
```

`Λειτουργικότητα` opens its chooser on first tap because the meaningful state is the chosen functional limitation, not a generic function flag.

### Weakness clarity

Visible focused options become:

- `Μυϊκή αδυναμία στην εξέταση`
- `Αδυναμία τετρακεφάλου στην εξέταση`
- `Ατροφία τετρακεφάλου`

Bare `Τετρακέφαλος` / `Περιαρθρικά` sibling choices are not exposed. Backward-compatible acceptance of historical `peri_knee_general` state remains below the presentation layer.

### Referral prose

- richer pain qualifier owns location specificity and prevents duplicated/fused legacy location wording;
- clinical/functional handoff and physiotherapy assessment/plan are separated by a blank line;
- `Επιπλέον στόχος:` / `Επιπλέον στόχοι:` become connected prose rather than generated labels;
- physiotherapist autonomy and the existing deterministic active-plan semantics remain unchanged.

## 3. Exact v5 evidence

At head `a47357c602120d3678e8f2f23b99775e616c79e1`, run `34693751545` passed bounded scope, syntax, focused v5 prose/state tests, inherited real CU-1/HTTP tests, frozen Step-3 fixtures, qualifier tests, protected Cockpit integration, focused/inherited Chromium acceptance, adjacent-owner isolation and package closure.

The candidate is therefore `IMPLEMENTED / TESTED`, not `MERGED / DEPLOYED / PRODUCTION-SMOKE-VERIFIED` as v5.

## 4. Out of scope / forbidden inference

V5 does not authorize or contain:

- evidence changes;
- Cyprus/GeSY clinical or administrative semantics;
- jurisdiction-overlay runtime activation;
- safety-rule changes;
- second diagnosis;
- patient persistence;
- analytics;
- autonomous evidence updates.

Cyprus/GeSY audit can proceed as a separate read-only evidence/design slice now that the production-auth boundary is closed.

## 5. Next decision

Product Owner review of the tested v5 candidate is the only release decision pending for that refinement. Any PR/merge/deploy requires separate explicit authority.

Jurisdiction work may proceed independently through audit/design only. No live product mutation is authorized merely by finding local differences.