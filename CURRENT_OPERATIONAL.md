# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 PRODUCTION — IMAGING-ATTACHMENT SEMANTIC GUARD TESTED / EXACT-HEAD REVIEW PASS — PR + SQUASH MERGE + DEPLOY AUTHORIZED
> **Updated:** 2026-09-06 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified production `main`:** `e8bf4bac16eff5e0c2101ec891483b81b14765e1`.
> **Production deploy:** `dep-daei1sh42hec73ccthr0` — LIVE.
> **Hotfix branch:** `fix/rf-imaging-attachment-semantic-guard-2026-09-06`.
> **Implementation base / merge base:** `e8bf4bac16eff5e0c2101ec891483b81b14765e1`.
> **Exact tested runtime head before release-authority docs commit:** `814a62d3b31ae76d19c6f5da3f824e9137011e96`.
> **Exact successful gate:** `RF v2 hotfix regression gate`, run `34031607422` — SUCCESS.
> **Current frozen slice:** `CU-RF-IMAGING-SEMANTIC-GUARD-2026-09-06`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — implementation/test/review closed.
> **ACTIVE CANONICAL WRITER/LOCK:** release closeout only.
> **Implementation/test authority:** CONSUMED.
> **PR authority:** GRANTED by product owner on 2026-09-06 for this bounded hotfix.
> **Merge authority:** GRANTED by product owner on 2026-09-06; squash merge required by repository discipline.
> **Deploy authority:** GRANTED by product owner on 2026-09-06; normal Render auto-deploy from `main`, no redundant manual deploy.
> **Production config authority:** NONE; no config change is required for this hotfix.
> **Production-smoke authority:** not implied by merge/deploy; smoke remains a separate post-deploy verification step.

---

# 1. Production truth before this release

Native RF v2 was released through PR #75 and correction PR #76.

Current production identity before the imaging-guard release:

```text
main: e8bf4bac16eff5e0c2101ec891483b81b14765e1
PR #76: merged by squash
Render: dep-daei1sh42hec73ccthr0
status: LIVE
```

Server-side production configuration is already present for:

```text
RF_PRODUCT_CATALOG_JSON
RF_DOCTOR_PROFILE_JSON
```

No environment/config mutation is part of the imaging semantic-guard release.

---

# 2. Product-owner smoke evidence that triggered the hotfix

Product-owner smoke established:

```text
clinical authentication/session                    PASS
native RF v2 UI visible                             PASS
Category A A.1/A.2 UI                               PASS
product catalog                                     PASS
doctor profile                                      PASS
single-side rule / derived target                   MERGED + DEPLOYED; re-smoke pending
medication capacity 0..3                            MERGED + DEPLOYED; re-smoke pending
Narox/Melox/Panadol/Parcoten parser corrections     MERGED + DEPLOYED; re-smoke pending
A.1 official form generation                        PASS on earlier smoke
uploaded PDF concatenation                          PASS on earlier smoke
attachment semantic suitability                     DEFECT FOUND / HOTFIX TESTED
full A.1 end-to-end production smoke                NOT YET PASS
full A.2 end-to-end production smoke                NOT YET PASS
PILOT-VALIDATED                                     NO
```

The deliberately unrelated smoke attachment was a laboratory report. Production accepted it merely because it was a valid PDF. This proved:

```text
PDF PRESENT != IMAGING-REPORT EVIDENCE PRESENT
```

---

# 3. Frozen imaging semantic-guard behavior

The bounded hotfix implements deterministic in-memory classification:

```text
IMAGING_SUPPORTED
→ create allowed without extra confirmation

CLEARLY_NON_IMAGING
→ create rejected fail-closed
→ clinician confirmation cannot override

AMBIGUOUS_OR_UNREADABLE
→ explicit clinician confirmation required
→ intended for scanned/image-only or otherwise unclassifiable PDFs
```

Server authority and privacy invariants:

- PDF must parse and contain at least one page; `%PDF` magic bytes alone are insufficient;
- extracted text is bounded and ephemeral;
- `POST /clinical/clinic-utilities/rf/api/validate-imaging` returns only bounded status/confirmation/message fields;
- `/api/create` independently repeats semantic assessment;
- browser state cannot override `CLEARLY_NON_IMAGING`;
- persistence may retain only bounded provenance (`auto_supported` or `clinician_confirmed`), never extracted attachment text;
- the classifier establishes document-type suitability only and does not interpret imaging findings or prove the selected diagnosis.

The same hotfix corrects stale medication UI copy to `έως 3` / `0..3`, matching the already-authoritative backend rule.

---

# 4. Exact automated evidence and review

Substantive tested clean head:

```text
814a62d3b31ae76d19c6f5da3f824e9137011e96
```

Full gate:

```text
workflow: RF v2 hotfix regression gate
run: 34031607422
result: SUCCESS
```

Final canonical-closeout head before this authority update:

```text
4fb623e7afe15cac502333d69d6d44c7f648e45b
run: 34031820267
result: SUCCESS
```

Passed evidence includes:

```text
Python / JavaScript syntax                          PASS
Official 12-page RF template identity/geometry      PASS
Real packaged-template A.1 generation               PASS
Real packaged-template A.2 generation               PASS
Existing RF v2 focused regressions                  PASS
Synthetic radiology attachment → supported          PASS
Synthetic laboratory attachment → rejected          PASS
Textless valid PDF → confirmation required          PASS
Ambiguous without confirmation → rejected           PASS
Ambiguous with confirmation → accepted               PASS
Clearly non-imaging even if confirmed → rejected    PASS
Fake %PDF bytes → rejected                          PASS
Preview endpoint does not return extracted text     PASS
UI semantic-validation wiring                       PASS
Adjacent CU-1 regressions                           PASS
Legacy RF gateway rollback regressions              PASS
G4/G3/G2/G1/C1 regression ancestry                  PASS
Full branch-vs-production diff hygiene              PASS
```

Exact-head review found no release-blocking scope, privacy, trust-boundary or dependency issue. The branch was `behind_by: 0` with merge base exactly equal to current production `main`.

---

# 5. Lifecycle matrix

```text
RF #76 CORRECTION IMPLEMENTED/TESTED/MERGED/DEPLOYED  YES
RF #76 PRODUCTION RE-SMOKE                            PENDING
IMAGING SEMANTIC-GUARD DESIGN                         FROZEN
IMAGING SEMANTIC-GUARD IMPLEMENTED                    YES
IMAGING SEMANTIC-GUARD TESTED                         YES
IMAGING SEMANTIC-GUARD EXACT-HEAD REVIEW              PASS
IMAGING SEMANTIC-GUARD PR                             AUTHORIZED / TO OPEN
IMAGING SEMANTIC-GUARD MERGE                          AUTHORIZED / PENDING
IMAGING SEMANTIC-GUARD DEPLOY                         AUTHORIZED / PENDING AUTO-DEPLOY
FULL RF A.1 PRODUCTION-SMOKE-VERIFIED                 NO
FULL RF A.2 PRODUCTION-SMOKE-VERIFIED                 NO
PILOT-VALIDATED                                       NO
```

---

# 6. Exact next action

Product owner explicitly authorized **merge and deploy** on 2026-09-06. Repository discipline requires the bounded branch to pass through PR and squash merge.

Execute now:

```text
rerun exact-head RF hotfix gate after this docs-only authority commit
→ open bounded PR to main
→ verify PR is mergeable and exact head has green checks
→ squash merge using expected head SHA
→ allow normal Render auto-deploy from main
→ verify Render reaches LIVE on the exact merge SHA
→ HOLD for production re-smoke
```

Do not manually trigger an additional Render deploy if auto-deploy succeeds.

Post-deploy smoke still must separately establish:

```text
#76 single-side/derived-location behavior
#76 medication parser/capacity behavior
obvious laboratory PDF rejected
real imaging PDF accepted OR scanned report explicitly clinician-confirmed
final A.1 PDF inspected
A.2 path exercised
```

No production configuration change and no claim of full production-smoke verification are authorized by this release action.
