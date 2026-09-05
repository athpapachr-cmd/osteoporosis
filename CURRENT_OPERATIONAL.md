# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 NATIVE CLINIC UTILITY — IMPLEMENTED / RELEASE-CANDIDATE TESTED / EXACT-HEAD REVIEW PASS / PR AUTHORIZED — PRE-PR HOLD
> **Updated:** 2026-09-05 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified production `main`:** `8aa8b38e3fa9a8f8ba0618868b452b1835be0d47`.
> **Active branch:** `feat/clinic-utilities-rf-v2-native-2026-09-05`.
> **Implementation base / merge base:** `8aa8b38e3fa9a8f8ba0618868b452b1835be0d47`.
> **Current frozen slice:** `CU-RF-V2-NATIVE-2026-09-05`.
> **Exact tested runtime head before docs-only closeout:** `aa2f92cce5d4cd2cfd02cafc59413be7bdc0d5fb`.
> **Exact workflow:** `RF v2 native clinic utility`, run `33988642002` — SUCCESS.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — implementation/test phase closed.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE — canonical pre-PR closeout complete.
> **Implementation/test authority:** CONSUMED.
> **PR authority:** GRANTED — bounded native RF v2 release PR only.
> **Merge authority:** NONE.
> **Production config/secret authority:** NONE.
> **Deploy authority:** NONE.
> **Production-smoke authority:** NONE.

---

# 1. Production baseline

Production remains on the old G4 authenticated RF gateway at:

```text
main:
8aa8b38e3fa9a8f8ba0618868b452b1835be0d47

release origin:
PR #73 — G4 hotfix: authenticated RF gateway
```

The old gateway auth/form leg was later proven in production after server-side RF key configuration:

```text
Cockpit RF route 200
→ upstream RF /rf 200
→ old form rendered
```

Full old history/create/PDF smoke was not completed before the authoritative RF form changed. The old RF form contract is now obsolete for the requested workflow.

---

# 2. Approved native RF ownership

Product owner approved moving RF ownership from `ortho-reception-backend-v2` into the Clinical Excellence runtime.

Target/implemented active architecture:

```text
authenticated Clinical Excellence browser
→ /clinical/clinic-utilities/rf
→ native RF router
→ existing clinical auth boundary
→ separate RF persistence tables
→ official RF v2 PDF stamping + imaging append
```

The legacy gateway code is retained unmounted as rollback/reference only.

RF remains a reusable Clinic Utility and is not written into osteoporosis encounter payloads.

---

# 3. Release-candidate implementation proven at aa2f92cc...

The exact tested runtime head implements:

- Category-A-only A.1/A.2 workflow for this clinician;
- fixed server-side doctor/product configuration without exposing values to the browser/repo;
- user-specific indication subset plus dynamic Other;
- required imaging PDF upload;
- A.1 pain/date validation and SI/hip intervention requirements;
- deterministic medication parsing/deduplication;
- fail-closed requirement for exactly 3 NSAID trials + 3 other analgesic trials before A.1 create;
- deterministic physiotherapy-date parsing;
- separate RF application-request and actual-procedure-history persistence;
- A.2 lookup by patient identity plus site/laterality;
- manual actual-procedure backfill with `clinician_manual` provenance for transition-period or later missing records;
- no obsolete hard-coded 10-week repeat rule;
- data minimization: raw pasted medication text, raw pasted physiotherapy dates and medication `source_text` lines are not persisted in application JSON;
- native Clinical Excellence UI;
- official PDF page assembly using the exact supplied template.

Hard invariant:

```text
RF APPLICATION REQUEST
!=
ACTUAL RF PROCEDURE
```

A generated approval/request never creates actual-procedure history by inference.

---

# 4. Authoritative official PDF identity

Packaged release template:

```text
clinic_utilities/rf/templates/rf_official_form_v2.pdf
```

Exact identity:

```text
source upload: Radiotherapy Eligibility Form.pdf
size: 310238 bytes
SHA-256: 998e99e6b0a51d4a19431dd2e31e595282d7adf17eb29f5f91eeab94e3647252
Git blob SHA: c6c234e99095be38c47a1c6f078dacdd47f4199f
pages: 12
geometry: approximately 595 x 842 pt per page
```

The gate verifies those values and exercises real A.1 and A.2 generation against the packaged official binary.

---

# 5. Exact automated evidence

Workflow:

```text
RF v2 native clinic utility
run: 33988642002
head: aa2f92cce5d4cd2cfd02cafc59413be7bdc0d5fb
result: SUCCESS
```

Successful exact-head evidence includes:

```text
Python syntax                              PASS
JavaScript syntax                          PASS
Official template size/hash/blob/page gate PASS
Real packaged-template A.1 generation      PASS
Real packaged-template A.2 generation      PASS
Native RF focused regressions              PASS
3+3 / data-minimization hardening tests    PASS
Native RF UI integrity                     PASS
Native route/dependency ownership          PASS
Adjacent CU-1 regressions                  PASS
Legacy RF gateway rollback regressions     PASS
G4 workspace regression                    PASS
G3 regressions                             PASS
G2 regressions                             PASS
G1 regressions                             PASS
C1 finalization regressions                PASS
```

Synthetic tests contain no identifiable patient data.

---

# 6. Exact-head review

Final source/security/scope review found and corrected before the final tested head:

1. A.1 previously allowed fewer than the required 3+3 medication rows.
   - corrected to fail closed unless 3 NSAIDs + 3 other analgesics are resolved;
2. application JSON previously retained raw pasted medication/physio text and selected medication source lines.
   - corrected to persist normalized evidence only;
3. manual actual-procedure history was labelled `legacy_manual`.
   - corrected to `clinician_manual` so provenance remains true for both transition-period and later clinician-entered missing procedure records.

Post-correction compare remains:

```text
branch ahead of production main
behind: 0
merge base: exact production main
scope: RF v2 + canonical slice files only
```

No Ortho-Reception runtime/config/secret mutation occurred.

---

# 7. Lifecycle matrix

```text
RF v2 DESIGN                         APPROVED / FROZEN
RF v2 IMPLEMENTATION                 COMPLETE
OFFICIAL TEMPLATE PACKAGED           YES — exact byte identity verified
RF v2 RELEASE-CANDIDATE TESTED       YES @ aa2f92cc... / run 33988642002
RF v2 EXACT-HEAD REVIEW              PASS
PR                                   AUTHORIZED / NOT YET OPEN
MERGED                               NO
DEPLOYED                             NO
PRODUCTION-SMOKE-VERIFIED            NO
PILOT-VALIDATED                      NO
ACTIVE RUNTIME WRITER                NONE
```

`IMPLEMENTED != TESTED != PR != MERGED != DEPLOYED != PRODUCTION-SMOKE-VERIFIED != PILOT-VALIDATED`.

---

# 8. Exact next action / HOLD

The implementation/test phase is closed.

Product owner has granted authority through opening the bounded native RF v2 release PR. Remaining lifecycle steps remain separately gated:

```text
final docs-only drift verification
→ open bounded RF v2 release PR
→ verify exact PR-head checks
→ separate merge decision
→ normal Render auto-deploy after merge
→ authenticated production smoke
```

The stale docs-only PR #74 was closed unmerged as superseded because it predates the authoritative-form change/native ownership replan.

Forbidden under current authority:

```text
NO merge
NO production config/secret mutation
NO deploy
NO production smoke
NO Ortho-Reception mutation
NO claim of production validation
```
