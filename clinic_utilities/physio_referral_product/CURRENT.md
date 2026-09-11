# CURRENT.md — Physio Referral productization local NOW

> **STATUS:** STEP 1 UX CONTRACT FROZEN FOR PROTOTYPE.
> **Updated:** 2026-09-07 Asia/Nicosia.
> **Branch:** `design/physio-referral-product-ux-v1-knee-oa-2026-09-07`.
> **Base main:** `43ed3090c5d4fe849e5f500edf22182fade36cd8`.
> **Vertical slice:** Knee Osteoarthritis only.
> **Runtime writer:** NONE.
> **Patient-data mutation:** NONE.
> **Production config/deploy authority:** NONE.

---

# Proven

- Existing CU-1 Physiotherapy Referral v2 is already implemented/deployed and provides a deterministic structured referral foundation.
- Existing frozen Knee v1.1 clinical profile already contains a Knee-OA pathway and evidence-sensitive adjunct semantics.
- Product owner selected ~€9.99/month as the plausible initial price point for a genuinely evidence-aware product, rather than €19.99 for a simple referral generator.
- Product owner selected one-diagnosis vertical slicing rather than five-diagnosis parallel development.
- Product owner approved the minimal/mobile-first interaction direction captured in `UX_CONTRACT_CURRENT.md`.

---

# Not yet proven

```text
Knee-OA evidence knowledge module          NOT DESIGNED/FROZEN
exact evidence source set                  NOT FROZEN
exact evidence-state mapping               NOT FROZEN
dynamic referral template contract         NOT FROZEN
new product UX                             NOT IMPLEMENTED
new product UX                             NOT TESTED
independent review                         NOT PERFORMED
external clinician willingness-to-pay      NOT VALIDATED
commercial pilot                           NOT STARTED
```

---

# Exact next action

```text
STEP 2
→ design the Knee-OA evidence knowledge module
→ map each intervention/recommendation to explicit evidence state + provenance
→ preserve source year separately from reviewed-on date
→ define positive suggestion and caution triggers
→ do not implement runtime until the Step-2 contract is reviewed/frozen
```

---

# Explicit hold

Until separate product-owner continuation authority:

```text
NO runtime UI rewrite
NO CU-1 taxonomy expansion
NO second diagnosis
NO billing/auth implementation
NO patient persistence
NO autonomous literature-to-rule updates
NO production deploy
```
