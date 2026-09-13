# CURRENT_OPERATIONAL.md — Clinical Documents Engine Phase 2 / Medical Report V1

> **STATUS:** MERGED / DEPLOYED / AI RUNTIME ENABLED — AUTHENTICATED PRODUCTION SMOKE PENDING.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CU-CLINICAL-DOCUMENTS-P2-MEDICAL-REPORT-V1-2026-09-13`.
> **PR:** `#99` — SQUASH MERGED.
> **Implementation review head:** `3a19a09280ff682badfd4866e19d6a4f3cb9c208`.
> **Release commit:** `261015be5a2921c6d67ad6b48d14196c17b1c34f`.
> **Clinical Documents implementation gate:** `34733760918` — SUCCESS.
> **Clinical Documents PR gate:** `34734187010` — SUCCESS.
> **Release deploy:** `dep-daj11b95efls739bog40` — LIVE.
> **Configuration deploy:** `dep-daj128u7bikc73acabt0` — LIVE.
> **Writer:** none.
> **Patient/case persistence authority:** NONE.

## 1. Product Owner authority

On 2026-09-13, after the implementation checkpoint was placed in release hold, the Product Owner instructed:

`Προχωρά μέχρι τέλους`

This authorized completion of the bounded release path: PR, squash merge, normal Render deployment and activation of the already-designed Medical Report AI runtime gates. It did not authorize persistent case storage, OCR, billing, autonomous medico-legal opinion, compensation calculations or unrelated product mutations.

## 2. Released workflow

Medical Report V1 is now on `main` and deployed for accident medical reports and medico-legal expert reports.

```text
case details
+ clinician history/instructions
+ selected source documents
+ clinician source classification
→ request-scoped extraction
→ Evidence Ledger + Timeline + work-incapacity intervals
→ AI-assisted draft
→ clinician review/edit
→ optional targeted literature research
→ clinician review/edit
→ explicit final confirmation
→ optional session-only signature
→ multi-page Greek PDF
```

## 3. Safety and authority model

The released runtime preserves the distinction between source fact, patient report, clinician finding, specialist opinion, literature evidence, AI inference and final clinician opinion.

Source/page/evidence references are validated deterministically. Work-absence date order is validated and overlaps/gaps are surfaced. Uploaded records are treated as quoted data rather than executable model instructions. Diagnosis, causation, prognosis and future-needs text remain clinician-review-required. Final PDF generation requires explicit clinician confirmation.

## 4. Privacy / persistence boundary

Still true in production:

- no Clinical Documents patient/case database;
- no browser PHI/case persistence or autosave;
- uploaded source bytes are request scoped;
- no patient identifiers in query strings;
- signature is session/request scoped only;
- direct patient identifiers are excluded from external literature-search prompts;
- no real patient data are present in repository tests or fixtures.

## 5. Runtime configuration and deployment

The Medical Report AI runtime enable and identifiable-record approval gates were activated on the Render `osteoporosis` service. The configuration change triggered deploy `dep-daj128u7bikc73acabt0`, which reached `LIVE` on the exact release commit at `2026-09-13T02:59:14Z`.

No provider credential value was exposed, copied or changed in this release session. The available deployment control plane cannot attest the value or validity of an existing secret credential, and the runtime remains fail-closed if a required provider credential is unavailable.

## 6. Verification evidence

The implementation and PR Clinical Documents gates both passed. Existing Sick Leave, CU-1 and G3 regressions also passed on the release path. Red results from unrelated Clinical Learning / Physio workflows were deliberate scope/adjacent-owner guards for this non-owner slice; inspected substantive Clinical Learning tests passed before the scope guard.

Render startup logs confirm the configured release instance completed application startup and became live. Existing protected clinical authentication remains configured.

## 7. Release-state distinction

```text
IMPLEMENTED                 YES
TESTED                      YES
MERGED                      YES
DEPLOYED                    YES
AI RUNTIME GATES ENABLED    YES
RENDER STARTUP/LIVE         YES
AUTHENTICATED UI SMOKE      PENDING
LIVE AI DRAFT CALL          PENDING
LIVE RESEARCH CALL          PENDING
LIVE FINAL PDF USER FLOW    PENDING
```

The remaining validation cannot be performed from the deployment connector because the Medical Report routes correctly require the existing protected user authentication and the connector does not inherit the Product Owner's browser session.

## 8. Exact next action

Run one short authenticated production smoke with a synthetic/non-identifiable test case:

```text
Clinic Utilities → Ιατρικές εκθέσεις
→ generate draft
→ inspect Evidence Ledger / Timeline
→ run literature research
→ edit + confirm
→ preview/download PDF
```

Until that authenticated smoke passes, Phase 2 is released and live but must not be labelled `PRODUCTION-SMOKE-VERIFIED`.