# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified main before final v3 documentation branch:** `213db0c31fab1582c2466b4b42921f5f2b74e299`.
> **CU-1 R1–R2 HARDENING PR:** PR #52 squash-merged as `f0a31b335848a1799a0b0b116a3bbe29a75fa7b3`.
> **CU-1 FINAL R2 CORRECTION PR:** PR #53 squash-merged as `213db0c31fab1582c2466b4b42921f5f2b74e299`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 — final design-completeness documentation and canonical closeout.
> **Design status:** `DESIGN-COMPLETE` per `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V3.md` on this branch, pending exact docs-only review/merge.
> **Frozen clinical profiles:** all planned CU-1 regional/shared v1.1 profiles.
> **Frozen machine contract entrypoint:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-design-complete-v3-2026-08-27`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. What is now proven

```text
CU-1 regional/shared clinical-content design = FROZEN
B1–B6 machine-contract blockers = RESOLVED
R1 declarative safety/consistency trigger gap = RESOLVED
R2 route-specific validation gap = RESOLVED
final shared-muscle R2 transcription defects = CORRECTED in PR #53
final repeat completeness review v3 = DESIGN-COMPLETE on active docs branch
```

`DESIGN-COMPLETE` means pre-code design is sufficiently explicit for a future implementation slice. It does not mean implemented, runtime-tested, deployed or pilot-validated.

---

# 2. Frozen first implementation boundary

```text
ephemeral ReferralDraftV1
→ deterministic validation/rule evaluation from cu1_contract_manifest_v1
→ ShortReferralFormatter / DetailedReferralFormatter
→ generated text
→ copy / print
```

Persistence remains out of the frozen first implementation scope.

---

# 3. Hard constraints

```text
NO production CU-1 runtime code under the current docs writer
NO FastAPI CU-1 endpoints
NO referral persistence/patient-data storage
NO identifiable patient data in repository or fixtures
NO reopening frozen clinical taxonomy without a proven contradiction
NO runtime interpretation of profile Markdown for trigger/validation logic
NO PR-1 runtime work while this closeout branch owns the canonical writer lock
```

---

# 4. Exact next action

```text
1. finish canonical/design-complete documentation on docs/cu1-design-complete-v3-2026-08-27
2. exact branch-vs-main docs-only review
3. open PR and independent exact-head review
4. squash merge only if clean
5. post-merge canonical cleanup: release writer lock, record exact merged main SHA, append durable milestone
6. STOP with CU-1 DESIGN-COMPLETE and runtime still NOT AUTHORIZED
7. a future runtime implementation requires separate explicit product-owner authorization and a fresh implementation slice/branch
```
