# PHYSIO P1 — Product Owner Real-Use Regression Cases

> **ROLE:** separate real-use usability regression set for the current Knee-OA Physio Referral.
> **DATE:** 2026-10-04 Asia/Nicosia.
> **SOURCE:** Product Owner direct use on a real mobile device.
> **IMPORTANT:** this file does **not** replace or rewrite `P1_SYNTHETIC_CASE_SET.md`.

The purpose of this set is to preserve the exact real-use scenarios that exposed the current structural information-architecture defect and to reuse the same scenarios after correction.

---

## Case 1 — routine right Knee OA

67-year-old woman with known right knee OA.

Clinical picture:
- pain mainly during walking and stairs;
- no important stiffness.

Function:
- difficulty with stairs;
- walking limited after approximately 20–30 minutes.

No recent significant trauma or other stated concerning feature.

Desired referral direction:
- physiotherapy assessment;
- therapeutic exercise;
- progressive strengthening;
- education/self-management.

Observed current-product findings:
- walking/stairs concepts felt split between symptom and Functionality;
- hidden second-tap detail was not discoverable;
- removal/deselection was low-visibility and sometimes required scrolling;
- walking limitation was hard to locate;
- referral output was good;
- product time approximately 2–3 minutes, roughly similar to simple manual referral.

---

## Case 2 — bilateral OA with stiffness/chronicity

72-year-old man with bilateral knee OA, worse on the left.

Clinical picture:
- symptoms for approximately 2 years;
- pain and stiffness after inactivity;
- morning stiffness approximately 15 minutes.

Function:
- difficulty rising from a low chair;
- difficulty walking uphill.

Observed current-product findings:
- stiffness refinements required repeated hidden interactions;
- chair-rise took material search time to find;
- hill walking was not found;
- referral output was good;
- product time approximately 2–3 minutes; manual estimate approximately 3–4 minutes.

Do not infer that hill walking requires a dedicated new control.

---

## Case 3 — richer objective examination

64-year-old woman with left knee OA.

Clinical picture:
- medial/anterior pain;
- pain particularly with stairs and sit-to-stand.

Examination:
- clear left quadriceps weakness;
- mild reduction of flexion;
- crepitus;
- medial joint-line tenderness;
- no objective instability.

Function:
- sit-to-stand;
- stairs;
- prolonged walking.

Observed current-product findings:
- objective quadriceps weakness was not readily found while atrophy was;
- ROM detail was difficult to locate and active/passive ownership was unclear;
- medial joint-line tenderness was found;
- functional tasks were searched across multiple sections;
- additional rehabilitation options were hidden;
- referral output was good;
- product time approximately 3–4 minutes; manual estimate approximately 5 minutes.

---

## Case 4 — rapid worsening / atypical stable-OA pattern

70-year-old man with known right knee OA.

Clinical picture:
- more intense pain;
- faster-than-usual worsening.

No stated fever, hot markedly swollen knee or major recent trauma.
Able to weight-bear.
Walking and stairs are difficult.

Observed current-product findings:
- intense pain and worsening were difficult to find;
- pain-related concepts felt duplicated between Pain and deeper surfaces;
- rehabilitation and adjuncts inside generic `Περισσότερα` felt structurally wrong;
- product time approximately 2–3 minutes, with search burden.

Required semantic boundary:
`rapid worsening != SIFK/fracture diagnosis != automatic imaging`.

---

## Case 5 — acute joint / routine physiotherapy inappropriate

76-year-old woman with a history of knee OA and an acutely painful left knee.

Scenario features:
- acute/severe pain;
- marked swelling / hot-inflammatory presentation;
- major difficulty weight-bearing;
- marked functional limitation.

The clinician would not send this patient for routine physiotherapy without reassessment.

Observed current-product findings:
- severe pain was not readily found;
- weight-bearing difficulty was not readily found;
- acute-joint observations were not sufficiently discoverable;
- product time approximately 3–4 minutes;
- routine physiotherapy referral would not be accepted for this scenario.

Required direction:
```text
ordinary observations
→ evidence-bounded review cue
→ clinician reassessment/disposition
!= automatic diagnosis
!= automatic imaging
```

---

## Cross-case regression expectation

After the bounded correction, rerun the same five cases.

Measure:
- approximate completion time;
- approximate taps;
- backtracking/search;
- manual edit need;
- referral acceptability;
- review-cue behavior.

Success requires:
- no hidden second tap for ordinary detail;
- no competing `Λεπτομέρειες` route;
- one obvious semantic owner per concept;
- Functionality separate from Examination;
- Rehabilitation/Proposed Plan separate from clinical data;
- no generic miscellaneous `Περισσότερα`;
- discoverable deselection;
- Case-5 observations readily accessible;
- referral output at least as useful as before;
- no diagnosis inference or automatic imaging.
