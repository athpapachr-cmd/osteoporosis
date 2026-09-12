# CURRENT_OPERATIONAL.md — Knee-OA v1 bounded usability/prose refinement

> **STATUS:** ACTIVE — bounded post-release usability/prose refinement only.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh bootstrap main:** `90e4377fc92e76d767a4b911a0dcff523b50b71e`.
> **Current production runtime SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Render deploy:** `dep-daiivp0jo6nc73bl6pug` — LIVE.
> **Authenticated production smoke:** GitHub Actions run `34681808255`, attempt `3` — SUCCESS on 2026-09-12.
> **ACTIVE RUNTIME / DESIGN WRITER:** `fix/physio-knee-oa-usability-v5-2026-09-12`.
> **Mutation scope:** Knee-OA clinical-picture interaction + deterministic Greek presentation prose + focused tests/canonicals only.
> **Real-patient data:** NOT USED.

## 1. Lifecycle boundary now closed

The previously unproven authenticated production boundary is now proven with an authorized production credential supplied through the protected GitHub Actions secret `CLINICAL_DATA_KEY`.

Run `34681808255`, attempt `3`, completed `SUCCESS` and proved:

```text
authenticated protected product page          PASS
authenticated product bootstrap               PASS
authenticated deterministic Knee-OA projection PASS
authenticated safety fail-closed projection   PASS
synthetic/non-identifiable smoke-data boundary PASS
```

The run sent no patient identifiers/history and did not print the production credential.

Therefore the released Knee-OA v4 lifecycle is now:

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

## 2. Active bounded refinement

Product-owner use identified three workflow/output defects that are independent of the planned Cyprus/GeSY jurisdiction audit:

1. selecting `Πόνος`, `Δυσκαμψία` or `Αδυναμία` immediately opens a modal detail sheet even though qualifiers are optional;
2. the weakness detail sheet exposes confusing overlapping labels (`Αντικειμενική στην εξέταση`, two different `Τετρακέφαλος` meanings, generic `Ατροφία`, `Περιαρθρικά`);
3. rich referral prose is visually dense, uses `Επιπλέον στόχος`, and can produce a pain-location overlap such as `...χηνείου ποδός στη μεσάρθρια γραμμή` when a richer qualifier overlaps a legacy specific pain finding.

Approved bounded design:

```text
inactive Pain/Stiffness/Weakness tap
→ select parent only; no modal

second tap on already-active parent
→ optional focused detail sheet

Function tap
→ focused functional-detail sheet remains, because no generic function fact exists

weakness detail sheet
→ Μυϊκή αδυναμία στην εξέταση
→ Αδυναμία τετρακεφάλου
→ Εμφανής ατροφία τετρακεφάλου

referral prose
→ clinical/functional summary paragraph
→ blank line
→ physiotherapy assessment/active-plan paragraph
→ "Επιπρόσθετη λειτουργική προτεραιότητα" wording
→ no semantically duplicated pain-location suffix
```

## 3. Hard preservation boundaries

This slice MUST NOT change:

- CU-1 taxonomy or safety authority;
- evidence states/source positions/default recommendations;
- Cyprus/GeSY recommendation semantics;
- jurisdiction profile behavior;
- clinician autonomy;
- patient persistence or analytics;
- diagnosis count;
- autonomous literature-to-live-rule behavior.

`international evidence core + jurisdiction overlay` remains unchanged and the Cyprus/GeSY audit is not implemented through this refinement.

## 4. Acceptance evidence required before closeout

At an exact tested branch head:

- first tap on inactive Pain/Stiffness/Weakness does not open `#sheet`;
- parent selection immediately updates referral state;
- second tap opens the existing focused sheet;
- Function still opens its focused choices;
- weakness sheet exposes only the three clarified routine choices;
- quadriceps atrophy still maps to explicit `visible_atrophy=true` + `atrophy_location=quadriceps` semantics;
- referral output contains a paragraph break before physiotherapy assessment;
- `Επιπλέον στόχος:` / `Επιπλέον στόχοι:` no longer appear in presented output;
- overlapping qualifier + legacy joint-line pain does not create fused/redundant wording;
- inherited safety/evidence/manual-edit/no-storage/mobile regressions remain green.

## 5. Exact next action

Implement the bounded refinement on the active branch, run the focused/inherited Knee-OA gates, then perform exact-head review and canonical closeout. Merge/deploy remains a separate release lifecycle step.

No second diagnosis is authorized. Cyprus/GeSY jurisdiction work remains a separate evidence/audit slice and must not be smuggled into this UX correction.