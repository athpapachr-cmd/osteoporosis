# Knee OA post-review bounded amendment — result v1

> **Date:** 2026-09-12 Asia/Nicosia  
> **Branch:** `feat/physio-knee-oa-review-amendments-v1-2026-09-12`  
> **Synthesis/disposition parent:** `554ecfa30bf8d0c04a19510a5db0276e6844edd5`  
> **Tested substantive head:** `83a5acd4e5b25413708bbda1715c58b5c78bce08`  
> **Successful workflow:** `Physio Knee OA prototype gate` run `34672654522`  
> **Result:** **PASS**  
> **Production / real-patient authority:** NONE

## 1. Purpose

Implement only the bounded corrections accepted after the four specialist reviews and Product Owner utility-gate discussion, then return the synthetic Knee-OA candidate for visual Product Owner review.

The implementation explicitly applies the reusable rule:

```text
clinically meaningful
!= workflow-useful
!= receiver-useful
!= worth adding
```

No new structured functional-baseline field was introduced and FFD was not promoted to the routine physician workflow.

## 2. Implemented amendments

### Diagnosis and missing prerequisites

The former checkbox-like `Διάγνωση επιβεβαιωμένη` presentation was replaced in the prototype by an explicit diagnosis-selection surface. Opening the route alone still does not assert the diagnosis. A clinician selection does; the diagnosis then projects automatically into the live referral.

Missing prerequisites are now specific and local:

```text
Επίλεξε διάγνωση
Επίλεξε πλευρά
```

Only the next unresolved prerequisite is emphasized, with text/non-colour semantics as well as restrained error styling. Export remains blocked.

### Weakness

Generic weakness remains contextual/reported. The former ambiguous quadriceps localization value is rejected. The explicit product-local value is now `quadriceps_exam`, displayed as `Τετρακέφαλος στην εξέταση`, before mapping to canonical `quadriceps_weakness`.

No MRC/dynamometry workflow was added.

### Passive extension deficit / FFD

FFD remains advanced/optional examination content. It is displayed and projected as a passive extension deficit, distinct from active extension lag.

```text
Παθητικό έλλειμμα έκτασης 10°
```

`0°` is rejected if a number is supplied; unknown degree remains `None`. The word `μόνιμο` and the English parenthetical label were removed. A selected FFD can make the existing mobility suggestion eligible but never selects the intervention.

### Referral proportionality and physiotherapy autonomy

A diagnosis+side-only referral remains shorter than a patient-specific referral while retaining the evidence-supported core rehabilitation priorities.

For richer referrals, plan wording is framed as physiotherapy assessment plus **indicative priorities**, explicitly dependent on physiotherapy findings and functional goals. Adjuncts are described as options for physiotherapy assessment rather than as a fixed technique menu. No clinician-selected structured state is silently removed.

### Qualifier / evidence / suggestion UX

- opening another qualifier group collapses the prior one;
- parent deselection also synchronizes hidden/ARIA expansion state;
- multiselect pain localization remains possible;
- routine supported/conditional evidence positions move behind an explicit deeper disclosure;
- mixed-guideline evidence still exposes all material source positions immediately;
- the suggestion presentation is flatter but retains explicit Add, evidence access, dismissal and stale-candidate protection;
- mobile manual-reconciliation gets a direct action.

### Jurisdiction profile seam

The prototype now visibly identifies:

```text
Πλαίσιο · Κύπρος · ΓεΣΥ
```

This is only a bounded `CY_GESY` profile signal. It does **not** import item-level local recommendations or claim that announced GeSY IT integration is already live. The intended reusable architecture remains:

```text
international clinical-evidence core
+
optional jurisdiction/local-system overlay
```

Country selection is not added to the routine Knee-OA surface. Future Greece/England profiles remain demand-driven hypotheses.

## 3. Test evidence

The first amendment run, `34672583682`, correctly failed because the inherited adapter test still required amended product prose to remain byte-for-byte equal to the frozen Step-3 output. This was treated as an ownership/test-boundary problem rather than deleting the frozen regression.

The test architecture was corrected so:

- the frozen Step-3 renderer still must match all original exact-output fixtures;
- the reviewed product overlay has separate explicit amendment tests.

Successful exact substantive gate:

```text
run                                      34672654522
head                                     83a5acd4e5b25413708bbda1715c58b5c78bce08
scope guard                              PASS
Python / JS syntax                       PASS
real CU-1 / HTTP suite                   15 / 15 PASS
frozen Step-3 exact outputs              15 PASS
Greek source-summary coverage            54 positions
post-review clinical/output tests        11 / 11 PASS
inherited Chromium browser tests         12 / 12 PASS
post-review Chromium tests                9 / 9 PASS
packaged real-CU1 dependency closure     PASS
```

The browser tests specifically cover diagnosis selection versus checkbox semantics, specific missing states, `CY_GESY` label, no new functional-baseline control, qualifier visible/ARIA consistency, examined quadriceps semantics, passive FFD semantics, supported evidence progressive disclosure and immediate all-source display for mixed guidance.

## 4. Limits

This PASS is synthetic technical/product-behavior evidence only. It does not prove:

```text
actual iPhone Safari / VoiceOver acceptance
complete measured accessibility / contrast acceptance
receiving-physiotherapist real-user validation
Cyprus/GeSY item-level guideline audit
Greece / England product need
production privacy / hosting / auth / billing readiness
willingness to pay / retention
clinical or commercial pilot validation
```

The Cyprus profile label is not a local-evidence state and cannot influence defaults/suggestions until a separately reviewed local-guidance contract exists.

## 5. Next boundary

Return the tested synthetic artifact and screenshots to the Product Owner for visual/use review.

No second diagnosis, jurisdiction expansion, PR, merge, deploy or production integration is authorized by this technical PASS.
