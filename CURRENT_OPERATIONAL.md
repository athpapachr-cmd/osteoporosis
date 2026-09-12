# CURRENT_OPERATIONAL.md — Knee-OA v1 authenticated live smoke closed / v5 review HOLD

> **STATUS:** PHYSIO REFERRAL KNEE-OA V1 — V4 RELEASED / DEPLOYED / AUTHENTICATED LIVE PRODUCT SMOKE PASS; V5 TESTED CANDIDATE AWAITS PRODUCT OWNER REVIEW.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Current production runtime SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Render deploy:** `dep-daiivp0jo6nc73bl6pug` — LIVE at exact production runtime SHA.
> **Public/auth-boundary smoke:** `34689920602` — SUCCESS.
> **Authenticated live smoke:** run `34681808255`, attempt `3` — SUCCESS.
> **Authenticated smoke closeout:** `commercial_products/physio_referral/releases/KNEE_OA_V1_AUTH_LIVE_SMOKE_CLOSEOUT.md`.
> **V5 tested candidate branch:** `fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12`.
> **V5 tested substantive head:** `a47357c602120d3678e8f2f23b99775e616c79e1`.
> **V5 gate:** `34693751545` — SUCCESS.
> **ACTIVE RUNTIME / DESIGN WRITER:** NONE.
> **PR / MERGE / DEPLOY AUTHORITY:** NONE for v5; Product Owner review HOLD.
> **Real-patient data:** NOT USED in smoke or v5 technical validation.

## 1. Production lifecycle boundary

The previously open authenticated-production verification boundary is now closed.

GitHub Actions run `34681808255`, attempt `3`, used the authorized production credential only through the protected repository secret `CLINICAL_DATA_KEY`; the value was not printed. The run completed `SUCCESS` and proved:

```text
authenticated protected product page           PASS
authenticated product bootstrap                PASS
authenticated deterministic Knee-OA projection PASS
authenticated safety fail-closed projection    PASS
synthetic/non-identifiable smoke-data boundary PASS
```

No patient identifier, patient history, patient persistence or production credential value was written by the smoke.

Current released v4 lifecycle:

```text
MERGED                              YES
DEPLOYED                            YES
PUBLIC-ASSET LIVE SMOKE             PASS
UNAUTHENTICATED AUTH BOUNDARY       PASS
AUTHENTICATED LIVE PRODUCT SMOKE    PASS
PILOT-VALIDATED                     NO
RECEIVER-VALIDATED                  NO
COMMERCIAL/PAID VALIDATED           NO
```

## 2. V5 tested candidate discovered from prior Product Owner session

A prior conversation had already completed the bounded post-use v5 refinement on:

`fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12`

Exact substantive head `a47357c602120d3678e8f2f23b99775e616c79e1` passed run `34693751545`.

V5 addresses only Product Owner usability/prose findings:

- first inactive tap on `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without forcing a modal;
- second tap opens optional detail;
- `Λειτουργικότητα` still opens its chooser because no honest generic function-only state exists;
- weakness refinement surface is reduced to explicit objective weakness, quadriceps weakness on examination and quadriceps atrophy;
- bare `Περιαρθρικά` is removed from routine/advanced UI while backward-compatible state acceptance remains;
- pain qualifiers own location specificity and remove overlapping legacy pain-location prose;
- rich referrals use a paragraph boundary before physiotherapy assessment/plan;
- machine-like `Επιπλέον στόχος:` labels become connected human prose.

V5 is not merged or deployed. V4 remains production authority until separate Product Owner release approval.

## 3. Hard product boundaries remain unchanged

```text
international evidence core + optional jurisdiction overlay
suggestion != clinician selection
clinical adaptation != reimbursement/admin rule
resource policy != stronger clinical evidence
```

No new evidence state, GeSY item-level rule, second diagnosis, persistence, analytics, autonomous literature update or treatment-decision automation is active.

## 4. Exact next legitimate actions

1. Product Owner reviews the exact tested v5 behavior/prose before any release decision.
2. Cyprus/GeSY OA primary-source audit may now proceed because the authenticated production lifecycle gate is closed.
3. Jurisdiction audit/design remains separate from v5 and must not mutate live evidence/UI merely because a local difference exists.

No active writer exists at this canonical state.