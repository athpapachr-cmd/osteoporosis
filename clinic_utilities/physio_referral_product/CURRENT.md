# CURRENT.md — Physio Referral productization local NOW

> **STATUS:** STEP 3 KNEE-OA DYNAMIC REFERRAL/TEMPLATE DESIGN FROZEN / CLOSED.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Branch:** `design/physio-referral-knee-oa-template-v1-2026-09-11`.
> **Fresh main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen Step-2 parent:** `ab4b349223cd4c461837ab3125967a06d169a7e1`.
> **Reviewed substantive head:** `cd4a42b4582921df7eb64d6ff3fb7c718a141c3a`.
> **Review artifact head:** `11f04d9ea316ba7fc30a03f4aaa13b60ea5c1e65`.
> **Vertical slice:** Knee Osteoarthritis only.
> **Runtime writer:** NONE.
> **Patient-data mutation:** NONE.
> **Production config/deploy authority:** NONE.

---

# Proven

- Step 1 minimal/mobile-first UX contract is frozen, including the six-state evidence model introduced by Step 2.
- Step 2 Knee-OA evidence knowledge module is frozen and source-traceable.
- Step 3 human dynamic-referral design exists at `KNEE_OA_TEMPLATE_DESIGN_V1.md`.
- Step 3 machine template contract exists at `contracts/knee_oa_template_contract_v1.yaml`.
- Primary and edge fixtures prove deterministic Greek referral composition, supported power-user preservation, semantic de-duplication, fail-closed unsupported selections and Copy-readiness boundaries.
- The product-local phenotype overlay preserves `stiffness symptom != ROM restriction` and `generic weakness != objective weakness`.
- Final copied Knee-OA referral requires explicit clinician OA assertion and right/left/bilateral laterality; opening the fixed Knee-OA screen does not create a diagnosis.
- Evidence suggestions do not become referral content until selected.
- Evidence UI metadata/citations do not leak into the copied referral.
- Generic goals may be suppressed only by explicit semantic de-duplication while their structured state is preserved.
- Explicit restrictions and clinician notes are preserved in copied output.
- `true_locking_or_major_mechanical_rom_block` remains deliberately unavailable on the first product surface because the inherited CU-1 rule catalog does not currently derive a safety trigger from that finding itself; exposure is deferred until a bounded safety/reassessment mapping exists.
- Exact Step-3 design review disposition: `DESIGN PASS / MATERIAL OPEN FINDING NONE`.
- Machine gate run `34561575795` passed on reviewed substantive head `cd4a42b...`.
- Machine gate run `34561638107` passed on review-artifact head `11f04d9...`.
- Full Step-3 diff contains no production CU-1 runtime/API/formatter/static UI/database mutation.

---

# Not yet proven

```text
evidence interaction/traceability contract    NOT FROZEN
visual/mobile prototype                       NOT BUILT
new product UX                                NOT IMPLEMENTED
functional Knee-OA vertical slice             NOT BUILT
product-owner usability test                  NOT PERFORMED
independent multi-axis product review          NOT PERFORMED
external clinician willingness-to-pay         NOT VALIDATED
commercial pilot                              NOT STARTED
```

---

# Exact next action

```text
STEP 4 — Evidence Interaction / Traceability Layer

Define:
colour + non-colour evidence cues
+ compact contextual bubbles
+ `i` evidence sheet
+ source/year vs reviewed-on display
+ guideline-conflict presentation
+ suggestion evidence provenance
+ accessibility semantics
```

Step 4 remains design-only until separately completed/reviewed. Runtime implementation does not become authorized merely because Steps 1–3 are frozen.

---

# Hold

```text
NO production UI rewrite yet
NO CU-1 runtime mutation yet
NO second diagnosis
NO billing/auth/entitlement implementation
NO patient persistence
NO autonomous literature-to-rule update
NO PR/merge/deploy/production smoke from Step-3 authority
```
