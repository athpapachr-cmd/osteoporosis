# PHYSIO P1 SYNTHETIC CASE SET

> **PURPOSE:** repeatable, non-identifiable cases for real-device, timing and receiver-comparison validation.
> **RULE:** these are synthetic test scenarios, not patient records and not treatment prescriptions.
> **RUNTIME:** use only the choices currently available in the released Knee-OA product. If a stated detail has no current control, leave it out rather than inventing a field.

---

# Case 1 — minimal unilateral referral

Synthetic facts:

- Knee Osteoarthritis asserted.
- Right knee.
- Pain present.
- One ordinary functional difficulty, for example stairs.
- No additional examination detail.
- No unresolved safety concern.

Purpose:

- low-information output proportionality;
- basic laterality and pain flow;
- time-to-copy;
- receiver reaction to a concise referral.

Expected test principle:

~~~text
LOW INFORMATION
→ SHORT REFERRAL
~~~

---

# Case 2 — richer functional / examination context

Synthetic facts:

- Knee Osteoarthritis asserted.
- Left knee.
- Symptoms for 8 months.
- Pain and stiffness selected.
- Weakness selected and refined only through currently available directional/detail controls.
- Functional difficulty with stairs and sit-to-stand.
- Add one or two currently supported examination findings under Περισσότερα → Εξέταση.
- No unresolved safety concern.

Purpose:

- second-tap refinement;
- chronicity;
- functional detail;
- examination progressive disclosure;
- richer deterministic referral without boilerplate;
- enlarged-text and VoiceOver traversal.

---

# Case 3 — bilateral / multiple functional limitations

Synthetic facts:

- Knee Osteoarthritis asserted.
- Bilateral.
- Pain present.
- Functional difficulty with walking distance and stairs.
- Use only supported current qualifiers.
- Select functional-task retraining if available.
- No unresolved safety concern.

Purpose:

- bilateral wording;
- receiver-side compression of functional retraining;
- prevention of duplicated task-list prose;
- product-versus-manual timing.

---

# Case 4 — review clue without automatic alternate diagnosis

Synthetic facts:

- Knee Osteoarthritis asserted.
- One knee.
- Include one currently supported atypical/review observation, for example recent trauma or rapid worsening, **without** selecting an explicit unresolved safety concern.
- Keep other information modest.

Purpose:

- confirm observation/review clue is not silently converted into fracture, infection, SIFK/SONK, imaging command or a second diagnosis;
- receiver reaction to wording;
- check that evidence/safety semantics remain distinct.

---

# Case 5 — manual edit / reconciliation

Start from Case 2.

Steps:

1. create the structured referral;
2. choose manual edit;
3. make an obvious synthetic wording change;
4. then change one structured selection;
5. observe stale/reconciliation behavior;
6. explicitly choose the intended final text path;
7. copy final referral.

Purpose:

- clinician-owned manual buffer;
- no silent reverse parsing;
- structured change does not silently overwrite manual text;
- timing/edit burden.

---

# Safety probe — explicit unresolved concern

Synthetic facts:

- Knee Osteoarthritis asserted.
- Valid laterality.
- Select an **explicit supported unresolved safety concern** in the current safety model.

Purpose:

- verify export/copy readiness fails closed according to the current safety contract;
- verify the blocking state is discoverable on the real device;
- verify no safety-critical action depends on opening evidence detail or Περισσότερα.

This probe is for workflow validation only. Do not infer or fabricate a specific diagnosis.

---

# Receiver comparison preparation

For Cases 1–4:

1. Generate the current product referral.
2. Independently create the Product Owner's ordinary/manual referral from the same synthetic facts.
3. Remove branding and obvious origin clues.
4. Randomly label the two versions A and B per case.
5. Give the receiver the two versions without saying which is product-generated.
6. Use the scoring/questions in P1_VALIDATION_PROTOCOL.md.

Do not include the safety probe in receiver preference scoring unless the purpose is specifically to evaluate the clarity of safety handoff language.

