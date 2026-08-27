# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified current main before lock-release cleanup:** `e064e1fe86a49dcf5026b4346d9e5f3fedfd3d92`.
> **CU-1 R1–R2 HARDENING PR:** PR #52 squash-merged as `f0a31b335848a1799a0b0b116a3bbe29a75fa7b3`.
> **CU-1 FINAL R2 CORRECTION PR:** PR #53 squash-merged as `213db0c31fab1582c2466b4b42921f5f2b74e299`.
> **CU-1 DESIGN-COMPLETE CLOSEOUT PR:** PR #54 squash-merged as `e064e1fe86a49dcf5026b4346d9e5f3fedfd3d92`.
> **Current major phase:** Personal Clinical Excellence foundation; bounded CU-1 pre-code design detour is complete.
> **Active slice:** CU-1 Physiotherapy Referral v2 — **DESIGN-COMPLETE / CLOSED AT PRE-CODE DESIGN GATE**.
> **Final design review:** `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V3.md`.
> **Frozen clinical profiles:** all planned CU-1 regional/shared v1.1 profiles.
> **Frozen machine contract entrypoint:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Proven CU-1 state

```text
CU-1 regional/shared clinical-content design = FROZEN
B1–B6 machine-contract blockers = RESOLVED
R1 declarative safety/consistency trigger gap = RESOLVED
R2 route-specific validation gap = RESOLVED
final shared-muscle R2 transcription defects = CORRECTED in PR #53
final repeat completeness review v3 = DESIGN-COMPLETE
canonical/roadmap closeout = MERGED in PR #54
```

`DESIGN-COMPLETE` means the pre-code design is sufficiently explicit for a future implementation slice. It does **not** mean implemented, runtime-tested, deployed, production-smoke-verified or pilot-validated.

---

# 2. Frozen first implementation boundary

If CU-1 runtime is later explicitly authorized, the first implementation direction remains:

```text
ephemeral ReferralDraftV1
→ canonical normalization / gateway / ownership resolution
→ deterministic route validation + safety/consistency rule evaluation from cu1_contract_manifest_v1
→ ShortReferralFormatter / DetailedReferralFormatter
→ generated text
→ copy / print
```

Persistence/referral patient-data storage remains outside the frozen first implementation scope.

---

# 3. Current locks and prohibitions

```text
canonical writer = NONE
runtime writer = NONE
CU-1 runtime implementation = NOT AUTHORIZED
CU-2 implementation = NOT AUTHORIZED by CU-1 completion
PR-1 runtime = PAUSED
```

No new session may infer implementation authority from `DESIGN-COMPLETE`.

No frozen clinical taxonomy should be reopened unless a concrete contradiction is demonstrated.

No runtime may interpret clinical profile Markdown to invent safety-trigger or route-validation semantics; the normative machine contract is the manifest and its listed artifacts.

---

# 4. Exact next authorized action

There is **no automatic engineering next step** after CU-1 design completion.

The next action requires an explicit product-owner choice:

```text
A. authorize a dedicated CU-1 runtime implementation slice
   → fresh six-canonical bootstrap
   → inspect current runtime/navigation integration seams
   → freeze implementation slice
   → claim a new runtime writer branch

OR

B. leave CU-1 frozen at DESIGN-COMPLETE
   → resume another explicitly selected roadmap item (for example the paused PR-1/baseline work or another separately authorized Clinic Utility)
```

Until that choice is made:

```text
STOP
NO ACTIVE WRITER
NO RUNTIME MUTATION
```
