# Knee OA v1 — post-release smoke finding v4

> **STATUS:** IMPLEMENTATION AUTHORIZED / ACTIVE.
> **Date:** 2026-09-12 Asia/Nicosia.
> **Parent main:** `c19c66bffbbeebb5181510fb28a62794292079b6`.
> **Scope:** bounded clinical-picture UX + referral prose correction only.

## Product-owner finding

During live user trial preparation, the relationship between a selected clinical-picture parent and its dependent qualifiers was not visually obvious. Inline qualifier lists appeared below the parent chips as a separate-looking set of controls, especially on mobile.

The accepted correction is:

- compact clinical-picture surface;
- mobile: 2×2 grid;
- desktop: four equal columns in one row;
- each clinical item opens a focused contextual sheet/pop-up;
- the main surface shows only the parent label plus a small active-detail count;
- qualifier details remain inside the sheet rather than consuming vertical space;
- the same interaction model is used on desktop and mobile;
- no clinical ID, evidence state, suggestion rule or safety rule changes.

Accepted referral-prose corrections:

- remove unsupported `κυρίως` when multiple pain locations are selected without ranking;
- explicit `quadriceps_exam` must render as an examination finding, not collapse to generic `μυϊκή αδυναμία`;
- keep one human continuous final sentence; replace machine-like priority wording with natural `με έμφαση σε ... ανάλογα με τα ευρήματα της αξιολόγησης και τους λειτουργικούς στόχους`.

## Hard boundaries

No second diagnosis. No evidence update. No jurisdiction change. No patient persistence. No clinical-rule expansion. No autonomous recommendation changes.
