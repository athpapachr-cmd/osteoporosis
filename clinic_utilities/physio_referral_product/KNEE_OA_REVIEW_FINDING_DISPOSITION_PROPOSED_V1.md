# Knee OA: πλήρες finding disposition ledger — Product Owner refinement 2026-09-12

**Candidate:** `6539351c592c1dc3e49931057b63925dea3cb94d`  
**Date:** 2026-09-12  
**Κατάσταση:** PARTIAL PRODUCT OWNER DISPOSITION RECORDED / IMPLEMENTATION NOT AUTHORIZED.

Οι αρχικές severities των reviewers διατηρούνται. Οι Product Owner αποφάσεις δεν αποδεικνύουν ότι ένα finding είναι σωστό ή κλειστό· ορίζουν τον τρόπο με τον οποίο πρέπει να επαληθευτεί ή να διορθωθεί. `ACCEPT` σημαίνει ότι γίνεται δεκτή η κατεύθυνση της διόρθωσης, όχι ότι η διόρθωση υλοποιήθηκε. `VERIFY` σημαίνει αναπαραγωγή/πρωτογενή επαλήθευση πριν mutation. `ADAPT` σημαίνει μικρότερη ή διαφορετική λύση. `DEFER / TEST FIRST` σημαίνει ότι δεν προστίθεται νέο UI/field πριν αποδειχθεί incremental value.

Original severity: D = MATERIAL BEFORE SECOND DIAGNOSIS; P = MATERIAL BEFORE COMMERCIAL PILOT; I = IMPROVEMENT; L = LATER / OPTIONAL.

| Source ID | Original severity | Product Owner disposition / proposed response | Τι πρέπει να αποδειχθεί για κλείσιμο |
|---|---|---|---|
| CE-01 | D | ACCEPT: explicit examined quadriceps weakness, reported generic weakness | Reproduce sibling-choice mapping· subjective localization cannot create objective finding· exam assertion can· no treatment auto-selection |
| CE-02 | D | ACCEPT SEMANTIC FIX / KEEP ADVANCED ONLY: FFD = measured passive/fixed extension deficit; no permanence, no numeric zero; do not promote to routine field | Positive/unknown/zero boundary cases· distinct active lag· no fabricated degree· reviewed Greek copy· receiver test confirms useful handoff when known |
| CE-03 | D | ACCEPT / VERIFY: exact intervention noun and claim scope | Primary-source matrix for massage/soft tissue, kinesiotaping/taping and graded activity· corrected positions independently checked |
| CE-04 | P | ADAPT / VERIFY: exact locators and review-date ownership | Trace upstream review events; distinguish content review from link check and sign-off; verified material claim locators with truthful fallback |
| CE-05 | I | DEFER exposure; clarify policy on next evidence amendment | Hidden dry-needling remains hidden; product exclusion not labelled multi-guideline consensus; source differences preserved |
| PT-01 | D | ADAPT: concise low-information output, όχι νέο υποχρεωτικό questionnaire | Diagnosis/side-only output proportional; rich detail only when entered; no invented normal/negative findings |
| PT-02 | D | ADAPT: priorities and therapist-led evaluation, όχι wholesale loss of selections | Receiver comparison; no technique/dose/progression mandate; every selected item remains visible, explicit or deliberately reconciled; restrictions stay authoritative |
| PT-03 | D | DEFER / TEST FIRST: do **not** add a priority-activity/baseline field yet. Functional goals/baseline are important in rehabilitation, but that does not prove the referring physician should collect them | Receiver study compares current function block vs one concise priority/baseline concept; only add if incremental actionability/comprehension justifies UI burden |
| PT-04 | I | ADAPT: contextual access to existing examination; FFD stays optional/advanced | Usability task demonstrates discoverability; no extra permanent exam form and no inference from stiffness |
| PT-05 | I | DEFER / REVIEW: peri-knee atrophy and anatomical granularity | Clinical/receiver judgment on specific low-value field before removal; no further pain-map expansion |
| UX-01 | D | ACCEPT / ADAPT: specific prerequisite messaging with local accessible error cue; restrained red may be used but never as sole signal | Separate missing-diagnosis, missing-side and both-missing tasks; field-specific message without hidden review sheet; error clears on resolution |
| UX-02 | D | ADAPT: predictable completion and single visible/ARIA state | Multiselect stays usable; switching sections collapses predictably; deselect/reselect/reset keep hidden state and aria-expanded consistent |
| UX-03 | D | ACCEPT: contextual semantic meaning for evidence cues | First-time tasks demonstrate meaning without memorizing glyph legend; colour remains non-exclusive cue |
| UX-04 | D | ADAPT: concise first evidence layer, detail on demand | Supported-state first view concise; all material conflicting source positions visible at first mixed disclosure; nothing misleadingly hidden |
| UX-05 | P | ACCEPT: direct mobile manual-reconciliation route | Stale manual buffer preserved; explicit authoritative choice; copy/print remain guarded; no silent text replacement |
| UX-06 | P | VERIFY / ACCEPT: measured meaningful contrast and real-device acceptance | Reproduce actual contrast/state issues; large-text dock unobscured; keyboard plus actual iPhone Safari/VoiceOver evidence distinguished from automation |
| UX-07 | I | ACCEPT: flatten suggestion presentation | One compact suggestion with explicit add, concise source support and accessible detail; stale candidate/dismissal safeguards preserved |
| COM-01 | P | ADAPT: breadth from actual referral mix, not fixed 70%/5–8 target | Aggregate volume and frequency data; bounded next diagnosis only after current slice corrected; no paid breadth claim without usage |
| COM-02 | P | ACCEPT experiment design, not immediate real-patient pilot | Timed end-to-end comparison includes navigation, typing, editing and transfer; quality does not deteriorate |
| COM-03 | P | VERIFY / REFRAME: Cyprus OA guideline is local-system guidance, not automatic clinical-evidence trump card | Obtain final official text/version; compare material divergences against NICE/international evidence; distinguish scientific rationale from feasibility/resource/reimbursement context; planned vs live GeSY integration verified separately |
| COM-04 | D | ADAPT: bounded evidence maintenance budget | Authoring/review time tracked; annual maintenance estimate explicitly forecast; no claim of measured 12-month data before observation |
| COM-05 | P | ADAPT: narrow buyer definition and aggregate baseline first | Frequency/referral-mix data; no patient identifiers; instrumentation remains a separately authorized change |
| COM-06 | P | ADAPT: keep price hypothesis and stage WTP testing | €9.99/€99 transparent; no discount/intent survey mistaken for actual payment; no billing implementation inferred |
| COM-07 | I | ADAPT with UX-04: presentation depth, not engine deletion | Separate quick evidence access from deeper-source use; disagreement and provenance still retrievable |
| COM-08 | I | DEFER catalogue deletion pending use/cost evidence | Actual relevance and maintenance effort compared; no enlarged adjunct catalogue |
| COM-09 | L | DEFER cockpit expansion | Referral module demonstrates standalone value before wider product build |
| GEN-F01 | D | VERIFY / ACCEPT if reproduced: availability-aware new defaults | New-draft source-state test against declared policy; suggestions/defaults share authority; existing clinician selections not silently erased |
| GEN-F02 | D | ADAPT with CE-04: schema now, verified claims before clinical paid use | Source-native locator fields and fallback defined before replication; exact material positions audited; original D severity retained |
| GEN-F03 | D | VERIFY contested ACE metadata | Check recommendations 2/4/6 and pinned fields; record direction/native strength/scope without majority vote or inherited reviewer acceptance |
| GEN-F04 | P | ADAPT with CE-02 and PT-02 | Reviewed passive FFD wording and receiver autonomy framing; useful rehab directions not replaced by empty generic prose |
| GEN-F05 | P | ACCEPT: actual receiving-physiotherapist feedback | Comparative handoff tasks for actionability, missing detail, autonomy and comprehension; AI perspective review not substituted for user study |
| GEN-F06 | P | VERIFY / ACCEPT with UX-06 | Actual device/AT and meaningful contrast acceptance; historical Chromium checks not treated as complete accessibility proof |
| GEN-F07 | P | ADAPT: staged commercial validation, no circular paid-user prerequisite | Distinguish discovery, WTP experiment, clinical paid pilot and scale; actual repeated use/payment/retention only claimed when observed |
| GEN-F08 | I | VERIFY: VA/DoD massage source position | Confirm exact source text/scope; add only if relevant and verified; no new state or surface control needed |
| GEN-F09 | I | DEFER / TEST FIRST with PT-03: no new functional-priority field until receiver value is demonstrated | Test current function categories + optional note against proposed concise priority/baseline information; require meaningful incremental receiver benefit before UI addition |
| GEN-F10 | I | ADAPT: prune redundant goals and catalogue presentation | Prove specific control contributes no new output/context; preserve nonredundant goals and explicit selected-item accounting |

## Product Owner UI decision not represented by a reviewer finding

### PO-UI-01 — diagnosis projection without redundant checkbox

The diagnosis should render automatically once the clinician has explicitly selected/asserted the diagnosis. A separate control whose only purpose is to include the diagnosis in the generated referral is rejected as redundant UI.

The current standalone prototype's `Διάγνωση επιβεβαιωμένη` control is tolerated only as a temporary substitute for a missing upstream diagnosis-selection event. The intended integrated product should make explicit diagnosis selection itself the formal assertion and immediately project the diagnosis into the referral.

Acceptance later must prove:

```text
no automatic diagnosis inference
explicit clinician selection/assertion exists
no second include-diagnosis checkbox
diagnosis appears automatically in live referral after selection
missing diagnosis state is explicit and accessible
```

## General Product Owner governance rule

```text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING
```

Any Product Owner, reviewer, author or assistant suggestion is a **hypothesis**, not clinical authority or implementation authority. Before a new field/qualifier/control is added, the product must identify the downstream decision, management, safety, handoff or workflow benefit and compare that incremental benefit with cognitive load and maintenance cost.

If a finding is important for the physiotherapist to assess but does not need to be supplied by the referring physician, it does not automatically belong in the referral generator. If usefulness is plausible but unproven, default to progressive disclosure, existing free text, receiver testing or deferral.

## Coverage and overlap

CE: 5 records, 4 material. PT: 5 records, 3 material. UX: 7 records, 6 material. COM: 9 records, 6 material. GEN: 10 records, 7 material. Total: 36 source records; 26 material record occurrences; 0 reviewer-reported blockers. No original finding is closed by this document.

Main overlap groups: CE-02 / GEN-F04; CE-04 / GEN-F02; PT-02 / GEN-F04 / GEN-F05; PT-03 / GEN-F09; UX-04 / COM-07; UX-06 / GEN-F06; COM-05 / COM-06 / GEN-F07; GEN-F10 / UX-07.

The central change from the 2026-09-11 proposed ledger is deliberate: **PT-03 / GEN-F09 are no longer accepted as immediate feature additions.** They are receiver-value hypotheses. FFD remains clinically meaningful but is not promoted to routine collection. Cyprus/GeSY guidance is to be evaluated as local system guidance with explicit separation of clinical evidence from resource/policy context.