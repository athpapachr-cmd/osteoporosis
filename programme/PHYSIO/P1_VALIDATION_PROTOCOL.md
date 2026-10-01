# PHYSIO P1 VALIDATION PROTOCOL — Knee-OA Reference / Core Boundary

> **TASK:** PHYSIO-P1-KNEE-OA-REFERENCE-VALIDATION-CORE-BOUNDARY-20261001
> **STATUS:** DESIGN FROZEN FOR BOUNDED VALIDATION.
> **Reference product:** released Knee Osteoarthritis physiotherapy referral, V5.1 + bounded post-use refinements, CY_GESY active.
> **Runtime mutation:** none authorized by this protocol.
> **Privacy:** synthetic, de-identified, or aggregate evidence only. Never record patient names, IDs, contact details, free-text clinical narratives that could identify a patient, or screenshots containing patient identity.
> **Purpose:** validate usefulness and freeze the practical boundary between reusable Physio mechanics and Knee-specific vertical content before any second diagnosis is considered.

---

## 1. Questions P1 must answer

P1 is not a feature-expansion exercise. It asks:

1. Does the current live Knee-OA workflow work acceptably on the Product Owner's real iPhone/Safari setup, including accessibility use?
2. Does it reduce end-to-end referral friction compared with the Product Owner's ordinary referral method without degrading the handoff?
3. Do receiving physiotherapists find the output understandable, actionable, appropriately concise and respectful of physiotherapist autonomy?
4. Which existing mechanisms are demonstrated reusable Physio Core and which remain Knee-OA-specific vertical content?
5. What is the actual aggregate physiotherapy-referral diagnosis mix, so any future second vertical is selected from real workflow evidence rather than intuition?
6. Is there an early product-value / willingness-to-pay signal worth testing further?
7. Are material evidence claims and jurisdiction positions maintainable with explicit provenance and locators before paid clinical use?

A technical PASS alone cannot answer these questions.

---

## 2. Non-goals and invariants

P1 does **not** authorize:

- a second diagnosis;
- a CU-1 rewrite or generic Physio framework refactor;
- new patient/referral persistence;
- analytics, billing or entitlements;
- new Greece or England profiles;
- autonomous literature-to-live updating;
- duplication of global Cockpit navigation, auth, safety or patient-model owners;
- automatic implementation of reviewer suggestions.

Permanent interpretation rules remain:

~~~text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING

SUGGESTION
!= CLINICIAN SELECTION

SYMPTOM
!= OBJECTIVE FINDING
!= DIAGNOSIS

MISSING
!= NEGATIVE
~~~

Any runtime correction must come from a concrete observed finding and must receive its own bounded owner/scope.

---

## 3. Evidence lanes

| Lane | Question | Primary collector | External dependency | Output |
|---|---|---|---|---|
| A | Real-device / accessibility acceptance | Product Owner | none | device acceptance record |
| B | Receiver usefulness | independent receiving physiotherapist(s) | yes | blinded paired review |
| C | End-to-end time / friction | Product Owner | none | matched timing record |
| D | Referral frequency / diagnosis mix | Product Owner | none | aggregate diagnosis-mix table |
| E | Product value / willingness-to-pay | Product Owner or product researcher | clinician respondents | exploratory value record |
| F | Evidence provenance / locator maintenance | PHYSIO coordinator or separate evidence reviewer | optional independent reviewer | provenance audit |
| G | Reusable Core boundary | PHYSIO coordinator | uses A–F | final boundary ledger |

The absence of an external receiver at a particular moment does not block other lanes. If Lane B is explicitly deferred, P1 must say so and must not claim receiver validation.

---

# 4. Lane A — real iPhone Safari and accessibility acceptance

## 4.1 Device conditions

Use the actual iPhone and Safari used in clinical work.

Run the synthetic cases in P1_SYNTHETIC_CASE_SET.md under:

- portrait orientation;
- landscape at least once;
- ordinary text size;
- a materially enlarged text setting/browser zoom;
- VoiceOver for the focused accessibility pass.

No real patient should be used for this lane.

## 4.2 Required interaction coverage

Confirm, using the current production workflow:

1. authenticated entry and project load;
2. diagnosis/laterality selection;
3. first-tap versus second-tap symptom refinement;
4. functionality chooser;
5. Περισσότερα progressive disclosure;
6. examination/detail controls;
7. suggestions remain unselected until explicit clinician action;
8. evidence detail/local difference remains readable without taking over the routine surface;
9. chronicity entry and output;
10. live deterministic referral update;
11. manual edit path;
12. stale/manual reconciliation after a later structured change;
13. Αντιγραφή action;
14. blocking behavior for an explicit unresolved safety concern;
15. return/navigation without accidental patient-draft persistence.

## 4.3 Accessibility observations

Record whether:

- interactive controls can be reached in a coherent focus order;
- the visible label and announced label are understandable;
- selected/unselected state is perceivable without relying only on colour;
- sheets/dialogues can be entered, traversed and dismissed;
- the primary copy/edit actions remain reachable;
- enlarged text does not create unusable horizontal scrolling, clipping or hidden actions;
- touch targets are practically usable on the real device;
- no safety-critical state depends on opening evidence detail or Περισσότερα.

## 4.4 Result states

Use one of:

- PASS — usable without material friction;
- PASS WITH FRICTION — usable, but one or more bounded defects should enter the finding register;
- FAIL — a material usability/accessibility/safety defect prevents ordinary use.

A screenshot may be kept only if it contains synthetic/non-identifiable state.

---

# 5. Lane B — receiving-physiotherapist paired comparison

## 5.1 Purpose

Test the referral as a handoff to the professional who receives it, not as a document that merely looks complete to the referrer.

## 5.2 Sample

Target: 2–3 independent receiving physiotherapists if practical.

If only one receiver is available, record the result as exploratory single-receiver evidence. Do not label the product receiver-validated from one person's review.

## 5.3 Materials

Use 4 synthetic/de-identified case pairs from P1_SYNTHETIC_CASE_SET.md.

For each case prepare:

- A: referral produced by the current Physio product;
- B: referral produced by the Product Owner's ordinary/manual referral method from the same facts.

Remove product branding and randomize A/B order so the receiver is not told which is the product output.

## 5.4 Receiver questions

For each referral, score 1–5:

- clarity/comprehension;
- actionability for initial physiotherapy assessment;
- sufficiency of clinically useful context;
- respect for physiotherapist autonomy / absence of over-prescription;
- concision / absence of noise;
- overall usefulness as a referral.

Then ask:

- What material information is missing?
- What would you remove?
- What would you still need to ask the patient/referrer before or during the first assessment?
- Is any sentence too prescriptive, ambiguous or clinically misleading?
- Which version would you prefer to receive, and why?

Record the answer verbatim only if it contains no patient-identifiable information.

## 5.5 Interpretation

One comment does not authorize implementation.

Classify findings as:

- repeated receiver signal;
- single-person preference;
- safety/meaning problem;
- missing material context;
- unnecessary content;
- autonomy/prescriptiveness problem;
- no material difference.

---

# 6. Lane C — end-to-end time and friction

## 6.1 Comparison

Compare:

~~~text
ordinary current referral method
vs
current Physio product
~~~

Use 5 matched synthetic cases. Alternate which method is performed first to reduce order/learning bias.

## 6.2 Start and stop

Start when the clinician begins the referral task.

Stop when copy-ready referral text is available for transfer to the intended destination.

Record separately if manual editing is used.

## 6.3 Capture

For each method/case record:

- total seconds;
- whether manual editing was needed;
- number of obvious backtracks/corrections;
- any point where the clinician had to search for a control or remember hidden state;
- perceived friction 1–5;
- whether the final handoff lost material information.

Do not impose an arbitrary percentage threshold in advance. A speed advantage is meaningful only if it is repeatable and does not buy speed by producing a worse referral.

Interpretation should consider:

- median/typical time;
- consistency across cases;
- edit burden;
- receiver evidence from Lane B;
- clinically meaningful information preserved.

---

# 7. Lane D — aggregate referral-frequency / diagnosis mix

## 7.1 Purpose

Provide the evidence base for a future P2 diagnosis-selection decision. Frequency alone does not decide P2, but P2 should not begin without knowing what is actually referred.

## 7.2 Default collection window

Prospectively tally consecutive physiotherapy referrals for 4 clinic weeks.

If fewer than 20 physiotherapy referrals occur in that period, collection may continue until 20 referrals or a maximum of 8 weeks, whichever comes first.

This is a pragmatic discovery sample, not epidemiology.

## 7.3 Allowed aggregate fields

Record only:

- diagnosis/referral family;
- body region;
- count;
- whether the referral is usually simple or repeatedly requires nuanced handoff;
- whether a reusable structured generator would plausibly save work;
- optional free-text category note that contains no patient detail.

Do **not** record name, ID, date of birth, contact information, exact appointment timestamp, or identifiable clinical narrative.

## 7.4 P2 candidate interpretation

When enough aggregate evidence exists, rank candidate diagnosis families only for discussion using:

- frequency;
- current referral friction;
- receiver information need;
- degree of reusable Core overlap;
- amount of genuinely new vertical content;
- evidence-maintenance burden;
- safety complexity;
- likely commercial value.

The ranking informs a Product Owner decision. It does not itself authorize P2 implementation.

---

# 8. Lane E — exploratory product value / willingness-to-pay

## 8.1 Scope

This is exploratory discovery, not commercial validation.

Target 3–5 prescribing/referring clinicians if practical.

Show the live workflow or a short synthetic demonstration before asking value questions.

## 8.2 Questions

Ask:

1. Would you use this in your normal referral workflow? Why or why not?
2. Which part saves the most effort, if any?
3. Which part feels unnecessary or too complex?
4. What would stop you using it repeatedly?
5. At approximately €9.99/month, which best describes you?
   - definitely would pay;
   - probably would pay;
   - unsure;
   - probably would not pay;
   - definitely would not pay.
6. What would need to be true for it to be worth paying for?

Do not convert stated willingness into a claim of paid conversion. Actual payment/retention remains separate evidence.

---

# 9. Lane F — material evidence provenance / locator audit

## 9.1 Objective

Before paid clinical use, material evidence claims should be maintainable and independently traceable.

Audit the current source/evidence objects that materially affect:

- starting/default rehabilitation plan;
- suggestion state;
- evidence-state badge/detail;
- a clinically meaningful local CY_GESY difference;
- recommendation-against-routine-use or conflict/mixed messaging.

## 9.2 Required evidence fields

For each material item, verify where applicable:

- source organization/title;
- version/year;
- reviewed_on date;
- evidence state;
- exact page/section/table/recommendation locator or stable equivalent;
- DOI/PMID/official URL where appropriate;
- local-vs-international ownership;
- whether the current UI wording is narrower/equal/broader than the source position.

Result:

- PASS;
- NEEDS LOCATOR MAINTENANCE;
- NEEDS WORDING/STATE REVIEW;
- BLOCKING EVIDENCE INTEGRITY ISSUE.

A missing locator is a maintenance finding; it does not automatically mean the clinical direction is wrong.

---

# 10. Lane G — reusable Physio Core boundary

Use P1_CORE_BOUNDARY_LEDGER.md.

Final dispositions are limited to:

~~~text
KEEP AS CORE
KEEP VERTICAL-SPECIFIC
CHANGE
REMOVE
EVIDENCE GAP
COMMERCIAL HYPOTHESIS
CROSS-PROJECT DEPENDENCY
~~~

A mechanism belongs in Core only when it is genuinely diagnosis-agnostic in semantics and ownership. A mechanism that merely happens to exist in Knee OA does not become Core by naming it generic.

Do not extract a broad generic framework solely because a second diagnosis might someday need it. Generalization remains P3 and should follow at least two real uses.

---

# 11. Finding classification and escalation

Every P1 finding should be entered in P1_EVIDENCE_WORKSHEET.md with one of these classes:

- safety/data-integrity defect;
- clinically meaningful workflow defect;
- receiver/handoff defect;
- accessibility/usability defect;
- evidence-integrity maintenance;
- commercial hypothesis;
- vertical-content issue;
- Core-mechanism issue;
- cross-project dependency;
- optional/cosmetic preference.

Action rule:

~~~text
concrete material defect in Physio-owned runtime
→ define bounded correction
→ identify correct technical/product owner
→ separate implementation authority
→ tests/review/release path

cross-project dependency
→ record exact seam/owner/requested decision
→ STOP mutation of foreign owner

single preference / cosmetic idea
→ record
→ do not implement by default
~~~

---

# 12. What the Product Owner can collect immediately

Immediate, without another reviewer:

- Lane A real-device / Safari / VoiceOver evidence;
- Lane C time/friction evidence;
- Lane D aggregate diagnosis-mix tally;
- Lane E initial clinician value interviews.

The Product Owner should not need to change runtime or enter real patient data to collect any of these.

External / separate-person evidence:

- Lane B requires an actual receiving physiotherapist for receiver evidence;
- Lane F may be performed by the PHYSIO coordinator, with an independent evidence reviewer added if a material source-to-claim ambiguity is found.

Coordinator synthesis:

- Lane G is completed after the relevant evidence is available.

---

# 13. P1 completion rule

P1 may be closed only when:

- Lane A has an explicit real-device result;
- Lane C has enough matched observations to characterize friction honestly;
- Lane D has an aggregate denominator and visible diagnosis mix;
- Lane E has either completed exploratory interviews or is explicitly deferred with no commercial-validation claim;
- Lane F has an explicit provenance disposition;
- Lane G has a final reusable-boundary ledger;
- Lane B is either completed or explicitly deferred by Product Owner, with receiver-validation language matching the evidence;
- every material runtime finding is either separately bounded for correction or explicitly accepted/deferred;
- no second diagnosis or generic refactor has been smuggled into P1.

P1 closeout must end with one explicit decision:

- no change / continue current product;
- bounded Knee-OA correction;
- continue validation;
- prepare a P2 diagnosis-selection decision;
- stop/defer the product workstream.

Even if P2 is selected, implementation requires a separate bounded Product Owner authorization.

