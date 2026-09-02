# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-3 PRODUCTION-SMOKE-VERIFIED / G-4 WORKSPACE UX + RF UTILITY INTEGRATION ACTIVE.
> **Updated:** 2026-09-02 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `ab94c6286bdc49cb8304b072e557c5eb0a96b0c6`.
> **G-3 hotfix PR:** `#71` — squash merged.
> **G-3 hotfix merge/runtime SHA:** `ab94c6286bdc49cb8304b072e557c5eb0a96b0c6`.
> **G-3 Render deploy:** `dep-dabolap5efls739s9am0` — `live`, trigger `new_commit`, exact commit `ab94c628...`.
> **G-3 product-owner production re-smoke:** PASS — `Νέο` and `Σύνοψη ασθενούς` visible and reported working well.
> **G-4 branch:** `feat/module01-g4-collapsible-sticky-summary-rf-utility-2026-09-02`.
> **ACTIVE CANONICAL WRITER/LOCK:** G-4 workspace UX + RF utility integration.
> **ACTIVE RUNTIME WRITER/LOCK:** G-4 workspace UX + RF utility integration.

---

# 1. Closed production state

```text
C1 / G-1 / G-2                    PRODUCTION-SMOKE-VERIFIED
G-3 IMPLEMENTED                   YES
G-3 TESTED                        YES
G-3 MERGED                        YES
G-3 DEPLOYED                      YES
G-3 PRODUCTION-SMOKE-VERIFIED     YES
G-3 PILOT-VALIDATED               NO
```

The successful re-smoke closes the G-3 production visibility defect. It is production smoke, not real-clinic pilot validation.

---

# 2. Active G-4 product-owner requirements

The product owner requested three bounded workspace changes after successful G-3 re-smoke:

1. `Σύνοψη ασθενούς` must be collapsible because it occupies substantial vertical space;
2. `Σημερινή ροή` must also be collapsible;
3. at least the patient summary must remain available at the top while scrolling;
4. integrate the existing radiofrequency-treatment PDF creation page into the Cockpit.

---

# 3. G-4 implementation boundary

## A. Workspace ergonomics

- add accessible expand/collapse controls to patient summary and current flow;
- preserve dynamic G-2/G-3 content and `Νέο` semantics while collapsed/expanded;
- make the patient-summary surface sticky at the top of the encounter workspace while scrolling;
- do not create new clinical truth, rules, writes or treatment decisions;
- collapse state is UI preference only and must not become authoritative patient data.

## B. Radiofrequency Clinic Utility integration

The previous RF PDF workflow belongs to Clinic Utilities / Clinical Operations, not Osteoporosis clinical state.

Recovered prior implementation evidence identifies a protected RF utility at `/rf` in the clinic reception backend, with `/rf/create` and PDF routes, official Medikey/DIROS/Thermedico templates and tested PDF generation. The first G-4 integration must therefore expose the existing protected RF utility from the Cockpit without copying its PDF engine, templates or patient/request persistence into osteoporosis encounter payloads.

Initial integration target:

```text
Cockpit Clinic Utilities navigation
→ Radiofrequency PDF utility
→ https://ortho-reception-backend-v2.onrender.com/rf
```

The original RF service remains the PDF-generation source of truth in this slice. A future deliberate migration into this repository would require the complete source/templates/auth/storage contract, not reconstruction from changelog notes.

---

# 4. Invariants / exclusions

```text
RF UTILITY != OSTEOPOROSIS ENCOUNTER STATE
UI COLLAPSE STATE != CLINICAL DATA
STICKY SUMMARY != SECOND SUMMARY OWNER
NO CHANGE TO G-2 EVIDENCE RULES OR THRESHOLDS
NO CHANGE TO C1 FINISH
NO C2 PERSISTENCE RELEASE IN THIS SLICE
NO PR-1 / PR-2
NO MANUAL RENDER DEPLOY
```

C2 remains implemented/tested on its existing branch but is not being released inside G-4.

---

# 5. Exact next action

```text
freeze G-4 slice design
→ inspect current G-3 summary/flow renderer + CSS and Cockpit navigation
→ implement collapsible/sticky behavior through existing render owner
→ add isolated RF Clinic Utility navigation entry
→ add focused regressions + inherited G3/G2/G1/C1 gates
→ exact-head review
→ close writer lock at IMPLEMENTED / TESTED
→ STOP before PR/merge/deploy without separate release authority
```
