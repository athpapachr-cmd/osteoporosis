# CURRENT.md — Physio Referral productization local NOW

> **STATUS:** STEP 2 KNEE-OA EVIDENCE DESIGN PASS / FREEZE CLOSEOUT.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Branch:** `design/physio-referral-knee-oa-evidence-v1-2026-09-11`.
> **Base main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Reviewed substantive head:** `6b82691c8431b699d20752c83b443793989f6402`.
> **Review artifact head:** `f20f01f34abf1796f075640a397d085443834c59`.
> **Vertical slice:** Knee Osteoarthritis only.
> **Runtime writer:** NONE.
> **Patient-data mutation:** NONE.
> **Production config/deploy authority:** NONE.

---

# Proven

- Step-1 minimal/mobile-first UX contract exists and has been replanned to a six-state evidence model.
- Step-2 human evidence design exists at `KNEE_OA_EVIDENCE_DESIGN_V1.md`.
- Step-2 machine contract exists at `contracts/knee_oa_evidence_contract_v1.yaml`.
- Six major guideline/CPG sources are represented separately rather than silently hybridized.
- `guideline_conflict_or_mixed` is explicit; Greek surface semantic is `Οι οδηγίες διαφέρουν`.
- Source-claim scope prevents broad evidence strength from being falsely transferred to narrower interventions.
- Reviewed visible smart default is therapeutic exercise + progressive strengthening + education/self-management.
- Graded activity is context-dependent, not a universal default.
- Manual therapy, soft-tissue techniques and acupuncture retain explicit mixed-guideline semantics.
- Dry needling remains excluded from the Knee-OA selectable pathway and is not misrepresented as consensus evidence.
- Walking-aid and weight-management integration seams are explicit rather than silently patched.
- Machine gate run `34559461372` passed on substantive head `6b82691...`.
- Post-review gate run `34559680326` passed on review-artifact head `f20f01f...`.
- Exact design review disposition: `DESIGN PASS / MATERIAL OPEN FINDING NONE`.
- No production CU-1 runtime/API/formatter/UI/database code was changed.

---

# Not yet proven

```text
dynamic referral/template contract          NOT FROZEN
new evidence interaction UI                 NOT IMPLEMENTED
new product UX                              NOT IMPLEMENTED
functional Knee-OA vertical slice           NOT BUILT
independent multi-axis product review        NOT PERFORMED
external clinician willingness-to-pay       NOT VALIDATED
commercial pilot                            NOT STARTED
```

---

# Exact next action

```text
STEP 3 — dynamic Knee-OA referral/template contract

Define how:
diagnosis + laterality
+ phenotype/findings
+ functional limitations
+ selected evidence-aware plan
+ optional power-user selections
→ live concise referral text
```

The Step-3 design must preserve physiotherapist autonomy, use no invented exercise prescription detail and remain separate from production runtime implementation.

---

# Hold

```text
NO production UI rewrite yet
NO CU-1 runtime mutation yet
NO second diagnosis
NO billing/auth/entitlement implementation
NO patient persistence
NO autonomous literature-to-rule update
NO merge/deploy/production smoke from Step-2 authority
```
