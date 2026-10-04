# PHYSIO P1 EVIDENCE WORKSHEET

> **TASK:** PHYSIO-P1-KNEE-OA-REFERENCE-VALIDATION-CORE-BOUNDARY-20261001
> **USE:** append observations from the frozen P1 protocol.
> **PRIVACY:** synthetic, de-identified or aggregate data only. No patient names, IDs, contact details, identifiable narratives or real-patient screenshots.

---

# 1. Validation identity

- Current production/release identity checked: current Knee-OA V5.1/post-use released product authority; P1 repo snapshot `1097b8e64a8fd181f33b3e53bf6911ba2296ab9c`.
- Date(s) of collection: 2026-10-01 for Lane F; 2026-10-04 Product Owner real-use A/C observations.
- Collector: PHYSIO coordinator for Lane F; Product Owner for the 2026-10-04 real-use cases.
- Device/browser when applicable: real mobile device for Product Owner cases; exact browser/VoiceOver state not separately verified in this checkpoint.
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

## Product Owner real-use structural observations — separate from frozen synthetic set

These observations came from five Product-Owner cases exercised on a real mobile device. They are **not** a rewrite of `P1_SYNTHETIC_CASE_SET.md`.

| Real-use case | Main observation | Referral output | Approximate product time |
|---|---|---|---|
| 1 | walking/stairs context split across symptom/function; hidden second-tap detail; walking limitation hard to locate; removal affordance not obvious | good | ~2–3 min |
| 2 | stiffness refinements required repeated hidden interactions; chair-rise hard to locate; hill walking not found | good | ~2–3 min |
| 3 | objective quadriceps weakness not found while atrophy was; ROM refinement hard to locate; function searched across wrong sections; added rehab options hidden | good | ~3–4 min |
| 4 | severe pain / rapid worsening hard to locate; perceived duplicate pain-related concepts; rehabilitation/adjuncts mixed into generic More | good | ~2–3 min |
| 5 | severe pain, weight-bearing difficulty and acute-joint safety observations were not readily discoverable; clinician would not send routine physiotherapy referral | not accepted as routine referral situation | ~3–4 min |

Cross-case interpretation:

- repeated failure is **structural information architecture**, not mobile-only styling;
- good deterministic referral output was preserved;
- search/navigation burden materially reduced the expected speed advantage;
- generic `Περισσότερα`, hidden second-tap detail and competing access paths are the dominant repeated friction pattern;
- Case 5 is safety-related because clinically important observations were not discoverable enough to support the clinician's decision not to proceed with routine referral.

Formal VoiceOver, enlarged-text and explicit Safari acceptance remain unproven.


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

- median/typical ordinary time: exact matched timing not collected; Product Owner estimated ~2–3 min for Case 1, ~3–4 min for Case 2 and ~5 min for Case 3; Cases 4–5 manual comparators not established.
- median/typical product time: approximate observed range ~2–4 min across Cases 1–5; not a formal stopwatch median.
- repeated friction point: searching across hidden second-tap detail, generic `Περισσότερα`, Functionality, Examination and rehabilitation-related controls.
- repeated benefit: referral output repeatedly judged good; Case 2 and Case 3 suggested potential time advantage despite search burden.
- edit burden: exact edit counts not consistently captured; structural navigation burden dominated the observations.
- interpretation: Lane C now has useful qualitative matched evidence, but not a formal exact-seconds completion. A bounded structural correction is justified before further timing claims.

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
| P1-AC001 | A/C | Hidden second-tap detail plus competing detail routes were repeatedly not discoverable in routine use | structural IA | yes | clinically meaningful | Knee-OA UI/presentation | remove hidden second-tap dependency and competing detail route; one semantic owner per concept | PRODUCT OWNER DECISION PENDING |
| P1-AC002 | A/C | Generic `Περισσότερα` mixes clinical presentation, examination and rehabilitation/adjunct worlds | structural IA | yes | clinically meaningful | Knee-OA UI/presentation | replace miscellaneous container with coherent owners | PRODUCT OWNER DECISION PENDING |
| P1-AC003 | A/C | Functionality concepts were searched for in Examination/More because semantic ownership was unclear | structural IA | yes | clinically meaningful | Knee-OA UI/presentation | canonical Functionality owner = functional_impairments; finding aliases compatibility-only | PRODUCT OWNER DECISION PENDING |
| P1-AC004 | A/C | Objective quadriceps weakness / ROM findings were not consistently discoverable | examination discoverability | yes | clinically meaningful | Knee-OA UI/presentation | reconcile subjective weakness vs objective exam and simplify finding access | PRODUCT OWNER DECISION PENDING |
| P1-AC005 | A/C | Case 5 acute-joint / weight-bearing observations were not readily discoverable even though clinician would not send routine referral | safety-related UX/semantics | yes in safety case | safety/data-integrity | Knee-OA UI | R2-B PASS: product-local review cue, explicit continue/defer disposition, no diagnosis/imaging/CU-1 block | PRE-CODE SEMANTICS PASS |
| P1-AC006 | A/C | Referral output remained good while 2–4 min completion was materially consumed by navigation/search | workflow efficiency | yes | useful operational | Knee-OA product | preserve deterministic output; correct IA before further timing validation | OPEN CORRECTION |
| P1-AC007 | A/C | Routine deselection/removal affordance was low-visibility and sometimes required scrolling | interaction discoverability | repeated | useful operational | Knee-OA UI | make removal/deselection immediately discoverable | R2 PRE-CODE REVIEW |

Materiality vocabulary:

- safety/data-integrity;
- clinically meaningful;
- useful operational;
- optional/cosmetic.

---

# 9. P1 closeout summary

- Lane A: PARTIAL / MATERIAL REAL-USE FINDING — mobile use observed; structural IA defect identified; formal Safari/VoiceOver/enlarged-text acceptance still pending.
- Lane B:
- Lane C: PARTIAL / QUALITATIVE MATCHED EVIDENCE — ~2–4 min product use with repeated search burden; exact stopwatch dataset not collected.
- Lane D:
- Lane E:
- Lane F: COMPLETE — **NEEDS LOCATOR MAINTENANCE**; no sampled clinical-state/default/runtime correction indicated.
- Lane G:
- runtime correction required?: **YES — bounded Knee-OA structural IA correction; exact R2-A product decisions pending Product Owner approval; closure review and implementation not yet authorized.**
- cross-project dependency?: **none required by R2-B; shared CU-1 remains unchanged.**
- receiver-validation claim allowed?:
- commercial-validation claim allowed?:
- second diagnosis authorized? **NO unless separately decided by Product Owner.**

Final P1 decision:

- no change / bounded Knee-OA correction / continue validation / prepare P2 diagnosis-selection decision / stop-defer.

