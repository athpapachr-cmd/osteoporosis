# CURRENT_OPERATIONAL.md — Knee-OA v1 v5 tested candidate / Product Owner review next

> **STATUS:** PHYSIO REFERRAL KNEE-OA V1 — V4 REMAINS RELEASED/LIVE; V5 IMPLEMENTED / EXACT-HEAD TECHNICAL GATE PASS / PRODUCT OWNER REVIEW NEXT.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Released production baseline:** PR `#89`, runtime SHA `bf527e3836a18491b2758fd293b42f82e0924382`, Render `dep-daiivp0jo6nc73bl6pug`, live boundary-smoke `34689920602` SUCCESS.
> **V5 branch:** `fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12`.
> **V5 tested substantive head:** `a47357c602120d3678e8f2f23b99775e616c79e1`.
> **V5 gate:** `34693751545` — SUCCESS.
> **V5 artifact:** `10298267537`, digest `sha256:a22fd402c49efcf3a9db20cc7751ed74363264437b911cd62c9de772fd520315`.
> **ACTIVE RUNTIME / DESIGN WRITER:** NONE.
> **PR / MERGE / DEPLOY AUTHORITY:** NONE / HOLD pending Product Owner review.
> **Second diagnosis:** NOT AUTHORIZED.

## 1. Trigger and disposition

Authenticated Product Owner use of released v4 identified four bounded defects. V5 corrects them without expanding the clinical/evidence model.

### Optional symptom refinement

For `Πόνος`, `Δυσκαμψία`, `Αδυναμία`:

```text
first inactive tap
→ selects the generic symptom
→ no popup / no forced qualifier

second tap while active
→ opens the focused optional refinement sheet
```

`Λειτουργικότητα` still opens its chooser on first tap because there is no meaningful generic function-only state.

### Weakness / atrophy clarity

New UI presents only explicit concepts:

- `Μυϊκή αδυναμία στην εξέταση`
- `Αδυναμία τετρακεφάλου στην εξέταση`
- `Ατροφία τετρακεφάλου`

Bare `Τετρακέφαλος` and bare `Περιαρθρικά` are no longer exposed as sibling UI choices. Legacy `peri_knee_general` acceptance remains only for backward compatibility; v5 does not create it through routine or advanced UI.

### Pain prose ownership

When product pain-location qualifiers exist, they own pain-location specificity. Overlapping legacy `joint_line_pain` / `anterior_peripatellar_pain` findings are removed from the product rendering state so phrases such as `χηνείου ποδός στη μεσάρθρια γραμμή` cannot be produced.

### Referral readability

When clinical/functional content and a physiotherapy plan coexist, the referral is rendered as two paragraphs:

```text
[indication + clinical picture + functional impact]

[physiotherapy assessment / active plan + connected goal emphasis]
```

Machine-like `Επιπλέον στόχος:` / `Επιπλέον στόχοι:` labels are replaced by connected prose (`Παράλληλα, στους λειτουργικούς στόχους ...`).

## 2. Exact gate evidence

At exact substantive head `a47357c602120d3678e8f2f23b99775e616c79e1`, run `34693751545` completed `SUCCESS`.

It proved:

- bounded v5 scope PASS;
- syntax PASS;
- focused v5 prose/state regressions 3/3 PASS;
- inherited real server/HTTP tests 15 PASS;
- frozen Step-3 exact-output fixtures 15 PASS;
- qualifier/clinical projection tests 11/11 PASS;
- protected Cockpit integration tests 6/6 PASS;
- focused v5 Chromium PASS;
- inherited v2/v3/v4 browser acceptance PASS;
- protected production Cockpit Chromium PASS;
- adjacent-owner isolation PASS;
- packaged dependency closure PASS.

A first v5 run failed only because a new browser test used an ambiguous selector that matched both `Σχετικά τώρα` and the `Εξέταση` category row. The selector was corrected without changing product behavior; the full exact-head run then passed.

## 3. Release boundary

V4 remains the production authority. V5 is **not merged and not deployed**.

The next legitimate action is Product Owner visual/use review of the exact tested v5 candidate. Any PR/merge/deploy requires a separate explicit decision after that review.

No evidence update, safety-rule change, jurisdiction change, persistence, analytics, autonomous recommendation behavior or second diagnosis is part of v5.
