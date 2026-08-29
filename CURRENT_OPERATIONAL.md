# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE — CU-1 RICH REFERRAL LATERAL-ELBOW RUNTIME PROTOTYPE.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Design parent:** `design/cu1-history-evidence-timeline-2026-08-28` @ `cc479f4a1d818481a886916e3f0f05dc56c623b3` / PR #63 draft.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/cu1-rich-referral-lateral-elbow-2026-08-29`.
> **ACTIVE RUNTIME WRITER/LOCK:** `feat/cu1-rich-referral-lateral-elbow-2026-08-29`.
> **Runtime authorization:** BOUNDED TO LATERAL-ELBOW PRODUCT-SHAPE PROTOTYPE + TESTS.
> **Deploy/merge authorization:** NO — product-owner output review first.
> **Further route-by-route evidence expansion:** HOLD.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused.

---

# 1. Current product decision — LOCKED

Product-owner review on 2026-08-29 accepted the corrected CU-1 referral target and explicitly authorized a bounded runtime prototype for `lateral_elbow_tendinopathy`.

The referral must not be either:

```text
sterile generic goals
OR
exercise-prescription micromanagement
```

The locked rehabilitation-document unit is:

```text
STAGE
→ GOALS
→ INTERVENTION DIRECTIONS: how each goal is pursued
→ PROGRESS MARKERS
→ NEXT-STAGE ORIENTATION
```

Intervention directions may name clinically meaningful treatment categories such as active ROM, isometric loading, concentric/eccentric resisted loading, load modification, manual therapy, cryotherapy/TENS or functional graded exposure when appropriate. They must not prescribe universal exercise sets, repetitions, kilograms, hold times, fixed weekly phase duration or fabricated numeric clearance thresholds.

---

# 2. Product purpose

The referral is a clinician-to-physiotherapist communication document and also a transparent expectation-setting document for the patient.

It must make clear what a complete rehabilitation process is expected to address. Passive symptom-modulation modalities may be adjuncts when evidence-compatible and clinically indicated, but must not be rendered as substitutes for active, progressive and function-oriented rehabilitation.

A detailed referral fails the product gate if it could reasonably be considered fulfilled by passive modalities alone without progressive active rehabilitation.

---

# 3. Lateral-elbow prototype scope

The current prototype must use the already-reviewed LET route authority and the accepted rehabilitation-product model to generate both:

```text
Short referral
Detailed referral
```

For the detailed output, the intended conceptual progression is:

```text
Stage 1 — irritability / symptom control + mobility + initial active loading
Stage 2 — progressive strength / endurance / load capacity
Stage 3 — functional / occupational / sport reintegration when relevant
```

This is a document/clinical-organization model, not a claim that the 2022 CPG validates a universal three-stage protocol or a universal transition threshold.

Required LET content behavior:

- pain/irritability goal must state how it is pursued: education/load modification and appropriate short-term symptom-modulation options;
- mobility goal must state how it is pursued: active mobility/ROM work and relevant mobility treatment if an actual deficit exists;
- initial load-tolerance goal may use isometric/low-demand extensor activation as an early orientation;
- progressive stage must state progressive resisted wrist-extensor loading, including concentric/eccentric orientation, without dose prescription;
- grip strength/endurance and repeated-use tolerance must be restored and tracked;
- shoulder/scapular rehabilitation is conditional on an actual proximal impairment;
- high-demand work/sport reintegration is conditional on actual patient context and must not invent job tasks;
- adjuncts remain adjuncts; ultrasound is not to be presented as a stand-alone core treatment;
- progress markers are clinical/functional observations, not invented universal thresholds;
- PRTEE/DASH/PSFS and grip/ROM measures may be used for follow-up but not converted into automatic discharge/progression cutoffs;
- atypical neurological/cervical/mechanical/traumatic findings preserve reassessment/correct-owner behavior.

---

# 4. Evidence boundary preserved

The already-reviewed route package remains authoritative for evidence applicability and grading. Preserve in particular:

```text
subacute/chronic wrist-extensor resisted exercise → JOSPT Grade B
isometric / concentric / eccentric modes → allowed, no universal dose
high-demand reintroduction → Grade F, conditional, no numeric clearance threshold
proximal shoulder/scapular training → Grade C only if impairment exists
local manual therapy → Grade B when selected/applicable
dry needling → Grade B when selected/applicable
rigid taping → Grade B in selected irritable short-term context
counterforce/wrist support → Grade F selected immediate/activity context
education / behavioral / ergonomic modification → Grade E
PRTEE/DASH/PSFS and impairment measures → tracking, not transition thresholds
```

The 2019 Day/Lucado/Uhl phased program may inform **rehabilitation organization** but its exact dose/load/transition criteria are not promoted to universal CPG rules.

The 2022 CPG also permits short-term cryotherapy/TENS and laser in defined contexts and cannot recommend therapeutic ultrasound as stand-alone treatment because evidence is conflicting. These modalities remain symptom-modulation adjuncts, not the active rehabilitation core.

---

# 5. Exact current implementation boundary

Allowed now:

```text
- route-specific lateral-elbow output synthesis in the existing formatter seam
- short + detailed generated output
- focused deterministic formatter/product tests
- canonical/PR documentation for this prototype
- review of the generated texts with the product owner
```

Not allowed yet:

```text
- deploy or merge the prototype before product-owner text review
- global rollout to all conditions before the lateral-elbow wording is accepted
- new route-by-route evidence queue work
- distal-biceps continuation
- persistence changes
- ClinicalAssertionV1 or medico-legal runtime
- CU-2 or PR-1 restart
```

---

# 6. Exact next action

1. Implement the LET short/detailed synthesis at the existing formatter seam without breaking other routes.
2. Add regression tests that enforce the locked product shape, including the passive-only failure concept and no-dose/no-false-threshold rules.
3. Run focused tests at exact branch head.
4. Render the actual Short and Detailed LET outputs and present them to the product owner.
5. Collect wording corrections.
6. After explicit text approval, redesign the implementation as a **shared/global rehabilitation document model** and apply it across the route registry in one horizontal rollout rather than manually repeating one route at a time.

Global rollout is intentionally deferred only until the LET text is approved; route-by-route product implementation is not the intended strategy.
