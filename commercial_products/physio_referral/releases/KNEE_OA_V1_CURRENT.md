# Knee OA v1 — current commercial release record

> **STATUS:** RELEASED PRODUCTION FOUNDATION + CY_GESY OVERLAY; V5 FRESH-MAIN INTEGRATED/TESTED CANDIDATE UNDER RELEASE HOLD.
> **Diagnosis:** Knee Osteoarthritis only.
> **Released jurisdiction runtime:** `e52a4851b504476c1e361575d08664c05467ff53`.
> **Production jurisdiction profile:** `CY_GESY` via explicit server-side configuration.
> **Authenticated CY_GESY production smoke:** `34703453615` — SUCCESS.
> **Original reviewed V5:** `a47357c602120d3678e8f2f23b99775e616c79e1`, gate `34693751545` — SUCCESS.
> **Current integrated V5 candidate:** `9a7f360745710deacb0ff82f03249723bdfe87d6`, gate `34736389860` — SUCCESS.
> **V5 release:** NOT MERGED / NOT DEPLOYED.

## Product surface currently released

The production product currently includes:

- explicit Knee-OA diagnosis assertion + laterality;
- live deterministic Greek referral, no routine Generate button;
- compact clinical-picture surface;
- direct manual editing with stale-text reconciliation;
- reviewed active-rehab defaults;
- international evidence-aware suggestions and mixed-guideline disclosure;
- advanced examination only on demand;
- `★ Συχνά / Σχετικά τώρα / Όλα` scan-first advanced UI;
- Cyprus/GeSY jurisdiction overlay as separate progressive evidence context;
- explicit server-side `CY_GESY` activation, no patient-location inference;
- ephemeral patient/referral draft; no analytics/patient persistence.

The current release does **not** yet include V5 because the fresh-main integrated candidate remains under Product Owner release HOLD.

## Production architecture

The product is deployed at:

`/clinical/clinic-utilities/physio-referral`

Architecture:

```text
real CU-1 validation/safety
+
deterministic Knee-OA projection/presentation
+
international evidence core
+
separate CY_GESY jurisdiction overlay
+
protected Clinical Excellence transport
```

No unauthenticated clinical endpoint, autonomous treatment decision or patient-draft persistence is introduced.

## Current jurisdiction verification

Authenticated production smoke `34703453615` proved the released `CY_GESY` behavior with protected credentials and non-identifiable synthetic smoke state only.

Verified release invariants include:

```text
CY_GESY explicit configuration              PASS
international acupuncture state unchanged  PASS
separate Cyprus acupuncture direction       PASS
manual/default clinical selection unchanged PASS
referral prose free of Cyprus/GeSY leakage  PASS
safety fail-closed                          PASS
```

Therefore the previous statement that Cyprus/GeSY item-level overlay was not activated is superseded: **the reviewed `CY_GESY` overlay is released and active in production.**

## V5 independent review and integration

Independent review accepted the V5 product/UX/clinical-copy direction but found the historical exact V5 head was no longer release-ready against current production ancestry because `CY_GESY` and shared integrations landed later.

Disposition:

```text
ACCEPT WITH REQUIRED CHANGES
PATCH V5 THEN MERGE
```

The required integration patch has now been implemented/tested from fresh main `8aeb91ae37b83caaa188054128db98e04b638fd8`.

Current exact candidate:

```text
branch   feat/physio-knee-oa-v5-integration-2026-09-13
head     9a7f360745710deacb0ff82f03249723bdfe87d6
gate     34736389860 — SUCCESS
artifact 10311108002
```

## V5 candidate behavior — not yet released

The integrated candidate provides:

- generic Pain/Stiffness/Weakness first tap without forced optional refinement;
- second tap opens focused optional detail;
- Function remains a first-tap chooser;
- weakness second-tap contains only explicit weakness exam options;
- duplicated `Ατροφία τετρακεφάλου` is removed from weakness second-tap and retained at `Περισσότερα → Εξέταση`;
- weakness badge/count does not count the separate atrophy finding;
- ambiguous bare `Περιαρθρικά` remains absent from visible routine/advanced UI;
- pain qualifier prose owns location specificity without legacy duplicate tails;
- rich referral uses a paragraph boundary before physiotherapy assessment/priorities;
- generated `Επιπλέον στόχος:` wording is replaced by connected prose;
- low-information output remains compact.

## Exact integrated candidate evidence

Run `34736389860` passed on exact head `9a7f3607...`:

```text
bounded-scope / no-cache guard             PASS
syntax                                     PASS
V5 focused server/prose                    PASS
current jurisdiction regressions           PASS
protected Cockpit integration              PASS
V5 + inherited + CY_GESY Chromium          PASS
atrophy route de-duplication               PASS
manual-edit fail-closed                    PASS
no browser patient/referral storage        PASS
adjacent-owner isolation                   PASS
package closure                            PASS
```

Thus the independent-review integration blocker is resolved at the tested-candidate level.

## Remaining validation / release boundary

```text
V5 PR / merge                                  PENDING
V5 Render deployment                           NOT DONE
V5 authenticated production smoke              NOT DONE
actual iPhone Safari / VoiceOver               NOT YET PROVEN
formal receiving-physiotherapist validation    DEFERRED / NON-BLOCKING
paid conversion / retention                    NOT YET PROVEN
real clinical pilot                            NOT YET PROVEN
second diagnosis                               NOT SELECTED / NOT AUTHORIZED
```

Formal physiotherapist/receiver feedback is optional later external evidence, not a V5 release gate.

Production authority remains the current released Knee-OA + CY_GESY state until an explicit Product Owner release decision permits V5 merge/deploy.