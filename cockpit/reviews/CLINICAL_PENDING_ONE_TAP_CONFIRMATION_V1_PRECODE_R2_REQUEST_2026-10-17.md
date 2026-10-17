# Clinical Pending One-Tap Confirmation v1 — independent PRE-CODE R2 request

**STATUS: PREPARED / NOT DISPATCHED / NO VERDICT. DO NOT ACTIVATE UNTIL EXISTING PR #161 PRIVACY/RELEASE LANE IS LAWFULLY SETTLED OR EXPLICITLY SEQUENCED WITHOUT OVERLAP.**

Date: 2026-10-17, Asia/Nicosia. **MODE WHEN ACTIVATED:** one fresh independent READ-ONLY, finite, affected PRE-CODE R2 review. No source/test edit, self-PASS, new writer, PR mutation, provider call, patient data, merge or deploy.

Repository: athpapachr-cmd/osteoporosis.
- Prospective design: cockpit/VISIT_BRIEF_CLINICAL_INBOX_DESIGN_2026-10-07.md, additive §§14–15 only; **design blob 083d32c67b0e16df2a280f6f3282bfe209b2f77e**, design-bearing commit **95603ff683bd77e69ee885ff2eef709ba1081d91**.
- Draft-only prospective design [PR #162](https://github.com/athpapachr-cmd/osteoporosis/pull/162); parent §§1–13 and prior R2 are settled/unaffected.
- Verified source baseline main: **4cf219e8f3f254caeb88134577af397afc229410**.
- Frozen source oracles: clinical_data.py **713ce588c2d7061f2f2c35f2a851a2a76b5965a1** (ClinicalPendingORM, VisitCapturePendingCandidate, signed Save, pending GET); clinical_auth.py **dda469fecdfbec3e6537f2d36e19e9a15df750c7** (shared key is NOT clinician identity); test_visit_capture.py **b5d1af910946a0387588ff33d6269e5e31b7f2a6** (signed Save/pending invariants).
- Existing [PR #161](https://github.com/athpapachr-cmd/osteoporosis/pull/161) has independently passed corrected post-code R2 at code/test head **3a98f7c046a83f05e0590661118c92c972483026** but remains DRAFT/unmerged with OPEN clinic retention/deletion/access gate. Reuse its accepted evidence; do NOT make this feature a dependency/blocker for #161 or alter its frozen code. Automatic same-UID attendee one-click is separately unqualified.

## Five finite questions

**R1 — Reuse existing authority, not another task engine.** Does §15 really extend existing ClinicalPendingORM/VisitCapturePendingCandidateV1 and protected patient store, proposing only optional approved stage-plan/revision and one same-DB append-only event owner? Ensure no new patient registry, task scheduler, AI-created clinical plan, second status owner or mutation of signed encounters; confirm backwards compatibility of existing Save and legacy pending.

**R2 — Clinically truthful checkpoint transitions.** Is unanswered UNKNOWN with no persisted negative, an explicit time-bounded “Όχι ακόμη” still unresolved, and “Ναι” affirmative only with evidence for the specific step? Receipt does not prove physician-reviewed MRI images, written radiologist report reading, completed patient communication or final closure. Expected revision/idempotency and correction/retraction must prevent duplicate/out-of-order transitions, only into stages already clinician-approved.

**R3 — Identity/role and write authorization.** One pending write requires server-side patient/pending match, signed source encounter, plan version and true trusted role; no phone/name/call/booking identity shortcut or UI-supplied doctor. clinical_auth.py only authenticates shared Clinical Key and cannot prove individual doctor identity. Live image-review attestation must remain HOLD pending qualified clinician identity/role. Synthetic injected test actor is not proof of production authorization.

**R4 — Daily Brief and shared projections.** Daily Brief is a bounded on-demand read of existing Clinical Pending events; patient card and Visit Brief consume the same pending ID and version. Preserve selected-booking Cyprus-date semantics from PR #161, signed encounter immutability and distinct Clinical Inbox/Reception action authority. Calls are operational candidates until independently linked and clinically accepted; no changes to M8 or telephony.

**R5 — One small synthetic vertical.** One signed and approved synthetic MRI-CD pending with receipt and image-review stages must cover unanswered/no write, not-yet time-limited, yes receipt only, no written-report equivalence, optional contact, final resolution only after required confirmed stages, across-day persistence, common state in two views, replay/conflict/reversal, wrong-patient and stale/auth errors, and unchanged legacy Save/Home. No scheduled Brief delivery, Gmail, model extraction, Reception import, second database, production migration or real patient data.

## Return and STOP

Return **PASS/BLOCK/UNKNOWN**, **COMPLETE_FOR_DECLARED_SCOPE/PARTIAL**, **P0:P1:P2**, R1–R5 concise source/line evidence, one smallest correction for any material finding. Reuse accepted Home, Visit Capture, Visit Brief and Clinical Inbox reviews, without unnecessary re-review. A design PASS would allow only a separate later Product Owner synthetic implementation decision after #161's release/privacy lane is settled, NOT merge/deploy or clinical data activation.

**NOT DISPATCHED NOW. STOP after one finite review when separately activated.**
