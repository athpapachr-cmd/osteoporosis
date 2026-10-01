# PHYSIO P1 EVIDENCE WORKSHEET

> **TASK:** PHYSIO-P1-KNEE-OA-REFERENCE-VALIDATION-CORE-BOUNDARY-20261001
> **USE:** append observations from the frozen P1 protocol.
> **PRIVACY:** synthetic, de-identified or aggregate data only. No patient names, IDs, contact details, identifiable narratives or real-patient screenshots.

---

# 1. Validation identity

- Current production/release identity checked: current Knee-OA V5.1/post-use released product authority; P1 repo snapshot `1097b8e64a8fd181f33b3e53bf6911ba2296ab9c`.
- Date(s) of collection: 2026-10-01 for Lane F; other lanes pending.
- Collector: PHYSIO coordinator for Lane F; Product Owner / external receiver as specified for remaining lanes.
- Device/browser when applicable:
- Notes on any material product change during collection:

If the runtime changes materially during P1, mark the affected observations as belonging to the earlier product state rather than silently combining them.

---

# 2. Lane A — real-device / accessibility

| Case | Condition | Result PASS / FRICTION / FAIL | Finding IDs | Notes |
|---|---|---|---|---|
| A | iPhone Safari portrait |  |  |  |
| B | iPhone Safari portrait |  |  |  |
| C | enlarged text |  |  |  |
| D | landscape |  |  |  |
| A11Y | VoiceOver focused pass |  |  |  |
| SAFETY | explicit unresolved safety concern |  |  |  |

Focused accessibility notes:

- focus order:
- labels/state announcement:
- sheets/dialogues:
- copy/edit reachability:
- enlarged-text clipping/overflow:
- touch usability:
- safety state discoverability:

---

# 3. Lane C — matched time / friction

| Case | Method | Total seconds | Manual edit? | Backtracks/corrections | Friction 1–5 | Material info lost? | Notes |
|---|---|---:|---|---:|---:|---|---|
| 1 | ordinary/manual |  |  |  |  |  |  |
| 1 | Physio product |  |  |  |  |  |  |
| 2 | Physio product |  |  |  |  |  |  |
| 2 | ordinary/manual |  |  |  |  |  |  |
| 3 | ordinary/manual |  |  |  |  |  |  |
| 3 | Physio product |  |  |  |  |  |  |
| 4 | Physio product |  |  |  |  |  |  |
| 4 | ordinary/manual |  |  |  |  |  |  |
| 5 | ordinary/manual |  |  |  |  |  |  |
| 5 | Physio product |  |  |  |  |  |  |

Summary:

- median/typical ordinary time:
- median/typical product time:
- repeated friction point:
- repeated benefit:
- edit burden:
- interpretation:

---

# 4. Lane B — receiver comparison

Use one row per receiver per case/version.

| Receiver code | Case | Version A/B | Clarity 1–5 | Actionability 1–5 | Useful context 1–5 | Autonomy 1–5 | Concision 1–5 | Overall 1–5 | Preferred? |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| R1 | 1 |  |  |  |  |  |  |  |  |

Receiver qualitative notes:

- material information missing:
- unnecessary content:
- too prescriptive / ambiguous wording:
- what receiver would still ask:
- reason for preferred version:
- repeated signal across receivers/cases:

Receiver evidence status:

- not started / exploratory single receiver / multi-receiver completed / explicitly deferred.

---

# 5. Lane D — aggregate diagnosis mix

Collection window:
Aggregate denominator:

| Diagnosis/referral family | Body region | Count | Typical referral friction low/medium/high | Structured generator plausibly useful? | Non-identifying note |
|---|---|---:|---|---|---|
| Knee OA | knee |  |  |  |  |

Top recurring candidate families:

1.
2.
3.

Interpretation beyond frequency:

- receiver information need:
- Core overlap:
- new vertical-content burden:
- safety complexity:
- evidence-maintenance burden:
- plausible product value:

---

# 6. Lane E — exploratory value / willingness-to-pay

| Respondent code | Would use? | Main value | Main friction | €9.99 response | What must be true to pay? |
|---|---|---|---|---|---|
| C1 |  |  |  |  |  |

Summary:

- repeated value signal:
- repeated objection:
- price signal:
- actual payment observed? yes/no:
- interpretation:

Do not label this commercial validation unless actual commercial evidence exists.

---

# 7. Lane F — evidence provenance / locator audit

| Evidence item / claim | Source + version | Exact locator present? | reviewed_on present? | International/local owner clear? | UI wording within source? | Disposition |
|---|---|---|---|---|---|---|
| Core exercise / active rehabilitation | NICE NG226; EULAR 2023 update; ACR/AF; AAOS; VA/DoD | source has precise recommendation/section; machine position does not store it | yes | yes | yes | supported; P1-F001 locator maintenance |
| Progressive strengthening scope | NICE 1.3.1; EULAR rec 3; ACR/AF; AAOS; VA/DoD rec 6 | source precise; machine position lacks exact locator | yes | yes | yes | supported; no strength laundering found |
| Education / self-management | EULAR rec 2; VA/DoD rec 4; NICE/ACR/AAOS | source precise; machine position lacks exact locator | yes | yes | yes | supported |
| Manual therapy mixed state | NICE 1.3.6–1.3.7; AAOS Manual Therapy; ACR/AF | source precise; machine position lacks exact locator | yes | yes | yes | mixed state supported |
| Acupuncture mixed state | NICE 1.3.8; AAOS Acupuncture; ACR/AF; ACE rec 6; VA/DoD rec 32 | source precise; machine position lacks exact locator | yes | yes | yes | mixed state supported |
| Dry needling non-routine / excluded | NICE 1.3.8; AAOS Dry Needling; VA/DoD rec 32 | source precise; machine position lacks exact locator | yes | yes | yes | current handling remains defensible |
| CY_GESY local positions | HIO/OAY OA artifact + adaptation | recommendation IDs + page/section stored | yes | yes | yes | supported; P1-F002 page convention normalization |
| CY_GESY lifecycle caveat | HIO/OAY May-2026 announcement + hosted OA artifact | explicit announcement + current hosted artifact | yes | yes | yes | current caveat remains appropriate |

Overall disposition:

- **NEEDS LOCATOR MAINTENANCE**.
- No sampled evidence-state/default-plan/runtime correction was indicated.
- Detailed audit: `programme/PHYSIO/P1_EVIDENCE_PROVENANCE_AUDIT.md`.

---

# 8. Finding register

| ID | Lane | Finding | Class | Repeated? | Materiality | Owner/seam | Action | Status |
|---|---|---|---|---|---|---|---|---|
| P1-F001 | F | International `positions[]` retain source/direction/scope/strength/summary but no claim-level recommendation/section/page locator | evidence-integrity maintenance | yes across material items | useful operational | Knee-OA evidence contract | future bounded contract/docs locator maintenance; no runtime state change | OPEN MAINTENANCE |
| P1-F002 | F | CY_GESY overlay and companion source audit use different implicit page conventions (printed page vs viewer/index page) while recommendation IDs agree | evidence-integrity maintenance | yes in sampled local recommendations | useful operational | CY_GESY evidence/docs | normalize page semantics in future bounded maintenance; no clinical change | OPEN MAINTENANCE |

Materiality vocabulary:

- safety/data-integrity;
- clinically meaningful;
- useful operational;
- optional/cosmetic.

---

# 9. P1 closeout summary

- Lane A:
- Lane B:
- Lane C:
- Lane D:
- Lane E:
- Lane F: COMPLETE — **NEEDS LOCATOR MAINTENANCE**; no sampled clinical-state/default/runtime correction indicated.
- Lane G:
- runtime correction required?: none from Lane F; remaining lanes pending.
- cross-project dependency?:
- receiver-validation claim allowed?:
- commercial-validation claim allowed?:
- second diagnosis authorized? **NO unless separately decided by Product Owner.**

Final P1 decision:

- no change / bounded Knee-OA correction / continue validation / prepare P2 diagnosis-selection decision / stop-defer.

