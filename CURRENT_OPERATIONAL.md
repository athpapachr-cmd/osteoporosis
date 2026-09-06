# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 PRODUCTION — IMAGING-ATTACHMENT SEMANTIC GUARD MERGED / DEPLOYED — PRODUCTION RE-SMOKE PENDING
> **Updated:** 2026-09-06 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified production `main`:** `1d26195c77e186cff98086283252af2eb499dd17`.
> **Release origin:** PR #77 — squash merged.
> **Production deploy:** `dep-daeliaks728c7384f3fg` — LIVE.
> **Current frozen slice:** `CU-RF-IMAGING-SEMANTIC-GUARD-2026-09-06`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — release complete.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only post-deploy closeout branch pending smoke evidence.
> **Implementation/test authority:** CONSUMED.
> **PR/merge/deploy authority:** CONSUMED for PR #77 release.
> **Production config authority:** NONE; no config mutation occurred.
> **Production-smoke authority:** not yet exercised for this release.

---

# 1. Production truth

Native RF v2 is now live with the imaging-attachment semantic guard.

```text
main:    1d26195c77e186cff98086283252af2eb499dd17
PR:      #77 — merged by squash
Render:  dep-daeliaks728c7384f3fg
status:  LIVE
trigger: new_commit
```

The prior #76 production deploy `dep-daei1sh42hec73ccthr0` was superseded normally by the #77 auto-deploy.

Existing server-side configuration remains unchanged and present:

```text
RF_PRODUCT_CATALOG_JSON
RF_DOCTOR_PROFILE_JSON
```

No environment/config mutation was part of PR #77.

---

# 2. Released semantic-guard contract

The production RF create path now distinguishes:

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

Server-side invariants:

- PDF must parse and contain at least one page; `%PDF` magic bytes alone are insufficient;
- extracted text is bounded and ephemeral;
- protected preview returns only bounded status / confirmation requirement / message;
- `/api/create` independently repeats the semantic assessment;
- browser state cannot override a clearly non-imaging classification;
- persistence may retain only bounded review provenance (`auto_supported` or `clinician_confirmed`), never extracted attachment text;
- classifier establishes document-type suitability only; it does not interpret imaging findings or prove the selected RF diagnosis.

The same release also corrects stale medication UI copy to `0..3` / `έως 3`, matching the already-authoritative backend contract.

---

# 3. Release evidence

Exact PR head:

```text
aa6ebc1d2df83a1638d7070a33f4173773bbe00a
```

Pre-PR exact-head gate:

```text
workflow: RF v2 hotfix regression gate
run: 34032436268
result: SUCCESS
```

PR-triggered adjacent gate:

```text
workflow: CU-1 focused tests
run: 34032485051
result: SUCCESS
```

Squash merge:

```text
PR #77
merge SHA: 1d26195c77e186cff98086283252af2eb499dd17
merged: true
```

Render auto-deploy:

```text
dep-daeliaks728c7384f3fg
source SHA: 1d26195c77e186cff98086283252af2eb499dd17
trigger: new_commit
status: LIVE
finished: 2026-09-06T12:16:20Z
```

No redundant manual deploy was triggered.

---

# 4. Lifecycle matrix

```text
IMAGING SEMANTIC-GUARD DESIGN                         FROZEN
IMAGING SEMANTIC-GUARD IMPLEMENTED                    YES
IMAGING SEMANTIC-GUARD TESTED                         YES
IMAGING SEMANTIC-GUARD EXACT-HEAD REVIEW              PASS
PR #77                                                 MERGED
MERGED                                                  YES
DEPLOYED                                                YES
PRODUCTION-SMOKE-VERIFIED                               NO
FULL RF A.1 PRODUCTION-SMOKE-VERIFIED                  NO
FULL RF A.2 PRODUCTION-SMOKE-VERIFIED                  NO
PILOT-VALIDATED                                        NO
```

`IMPLEMENTED != TESTED != MERGED != DEPLOYED != PRODUCTION-SMOKE-VERIFIED != PILOT-VALIDATED`.

---

# 5. Exact next action

Production re-smoke is now the only release gate still open for this RF slice.

Verify in the live UI:

```text
1. single-side + system-derived exact location behaves correctly
2. medication capacity shows 0..3 and parser corrections remain intact
3. obvious laboratory PDF is rejected
4. real imaging PDF is accepted, OR scanned/textless imaging requires explicit clinician confirmation
5. inspect generated A.1 official PDF
6. exercise A.2 continuation path and inspect its PDF
```

Until those checks pass:

```text
NO claim of full RF production-smoke verification
NO pilot validation claim
```

This docs-only post-deploy closeout branch should be reconciled with final smoke evidence before any later canonical merge, avoiding an unnecessary production redeploy merely to record interim release state.
