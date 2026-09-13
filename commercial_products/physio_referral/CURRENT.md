# CURRENT.md — Physio Referral commercial product track

> **STATUS:** KNEE-OA V1 RELEASED / CY_GESY OVERLAY RELEASED / V5 FRESH-MAIN INTEGRATED + TESTED / PRODUCT OWNER RELEASE HOLD.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Diagnosis vertical:** Knee Osteoarthritis only.
> **Released jurisdiction runtime:** `e52a4851b504476c1e361575d08664c05467ff53`.
> **Production jurisdiction profile:** `CY_GESY` via explicit server-side configuration.
> **Authenticated CY_GESY production smoke evidence:** `34703453615` — SUCCESS.
> **Original reviewed V5:** `a47357c602120d3678e8f2f23b99775e616c79e1`, gate `34693751545` — SUCCESS.
> **Current integrated V5 candidate:** `feat/physio-knee-oa-v5-integration-2026-09-13` @ `9a7f360745710deacb0ff82f03249723bdfe87d6`.
> **Integrated V5 gate:** `34736389860` — SUCCESS.
> **Artifact:** `10311108002` / `sha256:c51e3e29c4057b0e90773011d54648703eca5d15c66a9f64e612952f1dd324d6`.
> **Release state:** NOT MERGED / NOT DEPLOYED.
> **Root operational authority:** `CURRENT_OPERATIONAL.md`.

## Product state

The Knee-OA product remains released inside the authenticated Clinical Excellence physiotherapy utility with the reviewed `CY_GESY` jurisdiction overlay active in production.

The currently released architecture remains:

```text
international evidence core
+
reviewed CY_GESY jurisdiction overlay
+
deterministic Knee-OA referral projection
+
progressive evidence disclosure
```

V5 is now a **fresh-main integrated/tested candidate**, not merely the older isolated branch candidate. Production remains on the pre-V5 released behavior until a separate release decision/merge/deploy occurs.

## Independent review disposition — resolved at candidate level

Independent review of the historical exact V5 candidate returned:

```text
ACCEPT WITH REQUIRED CHANGES
PATCH V5 THEN MERGE
```

The required change was not clinical redesign. It was integration onto current production ancestry because `CY_GESY` and shared integrations landed after the original V5 test ancestry.

That blocker is now resolved on exact integrated head `9a7f3607...`, where V5, current jurisdiction behavior and protected Cockpit integration passed together.

## Product Owner-approved V5 simplification

The Product Owner additionally accepted the independent review's first simplification:

```text
Ατροφία τετρακεφάλου
!= duplicated weakness second-tap option

Ατροφία τετρακεφάλου
= objective finding under Περισσότερα → Εξέταση
```

Final integrated behavior:

- first tap on inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without a forced popup;
- second tap exposes optional refinement;
- `Λειτουργικότητα` keeps its first-tap chooser;
- weakness second-tap contains only `Μυϊκή αδυναμία στην εξέταση` and `Αδυναμία τετρακεφάλου στην εξέταση`;
- `Ατροφία τετρακεφάλου` remains available only via `Περισσότερα → Εξέταση` within this compact/duplicate-access question;
- weakness badge/count does not count the separately selected atrophy finding;
- bare `Περιαρθρικά` remains absent from visible routine/advanced UI;
- pain-location duplication is reconciled;
- clinical picture/function and physiotherapy plan use separate paragraphs;
- `Επιπλέον στόχος:` becomes connected natural prose;
- low-information output remains compact.

## Integrated test evidence

Run `34736389860` — SUCCESS on exact head `9a7f360745710deacb0ff82f03249723bdfe87d6`.

It passed:

- V5 focused prose/server tests;
- inherited deterministic projection/qualifier tests;
- current `CY_GESY` jurisdiction regressions;
- current protected Cockpit integration regressions;
- V5/inherited/CY_GESY Chromium acceptance;
- atrophy route de-duplication regression;
- manual-edit fail-closed and no-storage boundaries;
- adjacent-owner isolation;
- package closure.

No jurisdiction runtime/data, international evidence semantics, safety rules, treatment defaults, patient persistence, Medical Report runtime, or second diagnosis were changed.

## Production lifecycle

Supported current claims:

```text
Knee-OA production release                       yes
CY_GESY jurisdiction overlay                     yes
explicit production jurisdiction config          yes
authenticated CY_GESY product smoke              pass
V5 fresh-main integration                        complete
V5 exact-head integrated gate                    pass
V5 merged                                        no
V5 deployed                                      no
V5 authenticated production smoke                no
real clinical pilot                              no
paid/commercial validation                       no
second diagnosis                                 no
```

## External feedback policy

Formal physiotherapist/receiver evaluation remains optional later external evidence, not a blocking gate for V5.

```text
external feedback
!= release prerequisite by default
!= evidence automatically
!= implementation authority automatically
```

Receiver utility is not claimed as proven.

## Next product boundary

The implementation/testing command has been completed.

Next bounded step:

```text
open/review V5 integration PR
→ preserve RELEASE HOLD
→ explicit Product Owner release decision
→ only if accepted: merge / Render deploy / authenticated V5 production smoke
```

Do not infer second-diagnosis, analytics, persistence, billing, Greece/England profile or new treatment-surface authority from V5 completion.

Permanent rule:

```text
CLINICALLY INTERESTING != WORKFLOW-USEFUL != RECEIVER-USEFUL != WORTH ADDING
PRODUCT OWNER REQUEST != EVIDENCE != IMPLEMENTATION AUTHORITY
EXTERNAL FEEDBACK != AUTOMATIC IMPLEMENTATION AUTHORITY
```