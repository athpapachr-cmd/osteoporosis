# Knee OA v1 — current commercial release record

> **STATUS:** V4 RUNTIME RELEASED / DEPLOYED / AUTHENTICATED LIVE PRODUCT SMOKE PASS; V5 TESTED CANDIDATE AWAITS PRODUCT OWNER REVIEW.
> **Diagnosis:** Knee Osteoarthritis only.
> **Current production runtime SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Render deploy:** `dep-daiivp0jo6nc73bl6pug` — LIVE at exact v4 runtime SHA.
> **Public/auth-boundary smoke:** `34689920602` — SUCCESS.
> **Authenticated live product smoke:** `34681808255`, attempt `3` — SUCCESS.
> **Authenticated smoke closeout:** `KNEE_OA_V1_AUTH_LIVE_SMOKE_CLOSEOUT.md`.
> **V5 candidate branch/head:** `fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12` / `a47357c602120d3678e8f2f23b99775e616c79e1`.
> **V5 gate:** `34693751545` — SUCCESS.

## Product surface currently released

- explicit OA diagnosis assertion + laterality;
- live deterministic Greek referral, no routine Generate button;
- compact clinical-picture surface;
- direct manual editing with stale-text reconciliation;
- reviewed active-rehab defaults;
- evidence-aware suggestions and mixed-guideline disclosure;
- advanced examination only on demand;
- `★ Συχνά / Σχετικά τώρα / Όλα` scan-first advanced UI;
- mobile controls with ≥44 px targets and large-text reflow;
- Cyprus/GeSY context seam without item-level activation;
- ephemeral patient draft; no analytics/patient persistence.

## Production architecture

The product is deployed at:

`/clinical/clinic-utilities/physio-referral`

Existing Clinical Excellence authentication and real CU-1 validation/safety remain authoritative. Shared product projection is deterministic. No unauthenticated clinical endpoint or autonomous treatment decision is introduced.

## Authenticated production verification

Run `34681808255`, attempt `3`, completed `SUCCESS` after the Product Owner made the existing production `CLINICAL_DATA_KEY` available to GitHub Actions as a protected repository secret.

The run proved:

```text
authenticated protected page             PASS
authenticated product bootstrap          PASS
deterministic allowed Knee-OA projection PASS
safety fail-closed projection            PASS
non-identifiable smoke-data boundary     PASS
```

The credential value was not printed. No patient identifiers, patient history or patient persistence were used.

Therefore full authenticated live product smoke is now a supported lifecycle claim for released v4.

## V5 tested candidate — not released

A prior Product Owner session produced a bounded v5 correction for post-use friction. It is implemented/tested at exact head `a47357c602120d3678e8f2f23b99775e616c79e1`, gate `34693751545` SUCCESS, but remains unmerged and undeployed.

Candidate behavior:

- generic Pain/Stiffness/Weakness first tap does not force optional refinement;
- second tap opens the focused refinement sheet;
- Function chooser remains first-tap;
- weakness/atrophy wording is reduced to explicit, non-overlapping concepts;
- product pain qualifiers own location specificity over overlapping legacy location prose;
- rich referral uses a blank-line boundary before physiotherapy assessment/plan;
- generated `Επιπλέον στόχος:` label is removed in favor of connected prose.

V4 remains production authority until explicit v5 release approval.

## Remaining validation

```text
actual iPhone Safari / VoiceOver                    NOT YET PROVEN
receiving-physiotherapist field validation         NOT YET PROVEN
paid conversion / retention                        NOT YET PROVEN
real clinical pilot                                NOT YET PROVEN
Cyprus/GeSY item-level overlay                     NOT ACTIVATED
second diagnosis                                   NOT SELECTED / NOT AUTHORIZED
```

The Cyprus/GeSY primary-source audit may proceed as a separate design/evidence slice now that the production authenticated-smoke gate is closed.