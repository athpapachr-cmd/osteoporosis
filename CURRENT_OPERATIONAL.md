# CURRENT_OPERATIONAL.md — Physiotherapy Referral Knee-OA V5.1 examination refinement

> **STATUS:** ACTIVE IMPLEMENTATION SLICE.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CU-PHYSIO-KNEE-OA-V5-1-EXAM-2026-09-13`.
> **Bootstrap main:** `53ea38318d99e81218e5402e3c206ba30c51a8a9`.
> **Branch:** `feat/physio-knee-oa-v5-1-exam-discoverability-2026-09-13`.
> **Writer:** ChatGPT / this bounded slice only.
> **Released baseline:** Knee-OA V5 + `CY_GESY`, production-smoke-verified.
> **Patient/referral persistence authority:** NONE.

## 1. Product Owner authority

The Product Owner authorized a bounded post-V5 usability/examination refinement after real-device use.

Authorized changes:

1. After first-tap selection of `Πόνος`, `Δυσκαμψία` or `Αδυναμία`, show a restrained inline hint that a second tap opens optional details.
2. Refine objective weakness choices to directional knee strength findings:
   - `Αδυναμία έκτασης γόνατος / τετρακεφάλου`;
   - `Αδυναμία κάμψης γόνατος / ισχιοκνημιαίων`;
   - `Αδυναμία κάμψης και έκτασης γόνατος`.
3. Add a progressive `Εύρος κίνησης` examination group with:
   - extension lag / active extension deficit;
   - passive extension deficit;
   - active flexion restriction;
   - passive flexion restriction.
4. Add `Κριγμός στην κίνηση` as an objective examination finding.
5. Refine tenderness into clinically distinct, on-demand palpation findings:
   - medial/lateral joint-line tenderness;
   - medial/lateral bony tenderness;
   - pes-anserine tenderness;
   - extensor-mechanism tenderness.
6. Add objective stability/laxity findings:
   - valgus instability;
   - varus instability;
   - anterior instability / ACL context;
   - posterior instability / PCL context.
7. Preserve current effusion/swelling semantics; do not add a new generic enlargement control.
8. Surface a restrained review bubble for explicitly captured atypical/review-clue states. Bony tenderness by itself is NOT atypical and must not trigger the bubble.
9. Make international and Cyprus source links directly discoverable from the evidence/source heading instead of requiring an additional nested disclosure step.

## 2. Evidence boundary

Evidence reviewed for this slice supports the examination concepts but does not authorize diagnosis inference:

- EULAR knee-OA diagnostic recommendations identify crepitus and restricted movement among the most useful examination signs and also recognize bony enlargement.
- OARSI describes physical examination including tenderness, range of motion, swelling, crepitus/grating, joint stability and varus/valgus alignment.
- 2026 systematic review/meta-analysis shows reduced flexion and extension ROM and reduced flexor/extensor strength in symptomatic knee OA; certainty is low/very low and this is used only to justify capture vocabulary, not automated treatment.
- NICE NG226 states imaging is not routine unless atypical features or possible alternative/additional diagnosis are present; atypical examples include recent trauma, prolonged morning stiffness, rapid worsening/deformity and hot swollen joint.

No SIFK/SONK diagnosis checkbox is authorized. No finding may infer a second diagnosis.

## 3. Hard invariants

This slice must not change:

- international evidence states or source positions;
- `CY_GESY` local-position semantics;
- treatment defaults or auto-selection;
- diagnosis/laterality authority;
- safety fail-closed behavior;
- suggestion != selection;
- clinical guidance vs GeSY admin/reimbursement separation;
- no patient/referral persistence;
- manual-edit reconciliation;
- single-diagnosis Knee-OA scope.

## 4. Current implementation strategy

Use product-local qualifier/examination state and existing generic CU-1 findings where semantically valid. Do not expand frozen CU-1 meanings by pretending a specific finding is something else.

Preferred mapping:

```text
extension weakness             -> existing quadriceps_weakness + product-local wording
flexion weakness               -> objective_weakness + product-local flexion wording
flexion+extension weakness     -> objective_weakness + product-local combined wording

extension lag                  -> existing extension_lag
passive extension deficit      -> existing passive_rom_restricted + current FFD qualifier semantics
active flexion restriction     -> existing active_rom_restricted + product-local direction
passive flexion restriction    -> existing passive_rom_restricted + product-local direction

all specific tenderness        -> existing generic tenderness + product-local location/type
crepitus                       -> product-local examination qualifier only
objective instability          -> product-local examination qualifier only
```

Specific objective instability must not be relabelled as subjective giving-way or recurrent instability episode.

## 5. Atypical/review bubble

The current `>30′` morning-stiffness clue remains valid and should become more discoverable as a non-blocking review bubble.

Atypical-feature messaging must mean:

> consider further clinical assessment / possible additional or alternative cause

and must NOT mean:

> the tool has diagnosed SIFK, inflammatory arthritis, infection or another disease.

Safety flags retain their existing stronger blocking behavior and are not replaced by this bubble.

## 6. Source links

The international evidence model already exposes validated HTTPS `locator` URLs per source, and the `CY_GESY` overlay already exposes `source_provenance.source_url`.

The change is presentation-only:

```text
source label + visible external-link affordance
→ official reviewed source in new tab
```

No new evidence resolution logic is required.

## 7. Release boundary

Lifecycle for this slice:

```text
DESIGN / EVIDENCE CHECK      ACTIVE
IMPLEMENTATION               PENDING
EXACT-HEAD TESTS             PENDING
PR                            PENDING
MERGE                         NOT AUTHORIZED BY IMPLEMENTATION ALONE
DEPLOY                        NOT AUTHORIZED BY IMPLEMENTATION ALONE
PRODUCTION SMOKE              NOT YET APPLICABLE
```

## 8. Exact next action

Implement the bounded V5.1 UI/qualifier/prose/source-link changes plus focused and inherited regressions on this branch. Do not open a second diagnosis or change evidence-state semantics.