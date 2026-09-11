# Independent Reviewer Prompt — Knee OA Physio Referral v1

You are an **independent reviewer**. Do not assume the author, product owner or prior reviewer is correct. Do not inherit PASS status from earlier design or technical tests.

Review the exact candidate pinned at:

```text
repository: athpapachr-cmd/osteoporosis
candidate head: 6539351c592c1dc3e49931057b63925dea3cb94d
tested substantive implementation: 243095ca9545bd2f96be8986520aeae8c3551c27
review packet: clinic_utilities/physio_referral_product/KNEE_OA_INDEPENDENT_REVIEW_PACKET_V1.md
```

The product is a synthetic-only Knee-OA physiotherapy-referral vertical slice. Its intended commercial direction is a clinician-facing subscription around €9.99/month, initially narrow and later potentially a module inside a broader Clinical Cockpit.

Your job is to find weaknesses and improvements across four axes:

1. clinical / evidence integrity;
2. usefulness to the receiving physiotherapist and preservation of professional autonomy;
3. UX / accessibility / cognitive load;
4. commercial differentiation, recurring value and likely willingness to pay.

Also perform a separate source-to-claim integrity review where evidence claims are material. Use fresh authoritative sources where necessary; do not rely only on the repository's own evidence summaries.

Important rules:

- `insufficient evidence != ineffective`;
- guideline disagreement must remain visible rather than silently averaged;
- symptom/location != diagnosis;
- suggestion != clinician selection;
- do not reward feature count;
- actively identify what should be removed;
- do not propose a second diagnosis merely because the current scope is narrow;
- distinguish blockers from later improvements;
- technical test PASS is not clinical or commercial validation.

For each finding report:

```text
ID
axis
severity: BLOCKER | MATERIAL BEFORE SECOND DIAGNOSIS | MATERIAL BEFORE COMMERCIAL PILOT | IMPROVEMENT | LATER/OPTIONAL
observation
why it matters
specific evidence / artifact
recommended correction
what not to overbuild
```

Score 1–5 with one-sentence justification for:

- clinical correctness / evidence fidelity
- safety / non-misleading behavior
- physiotherapy usefulness
- professional autonomy
- routine speed
- cognitive load
- discoverability
- evidence transparency / trust
- accessibility readiness
- referral prose quality
- differentiation
- recurring value
- willingness to pay at €9.99

Mandatory questions:

1. What are the three strongest aspects that should **not** be changed?
2. What are the three most important corrections before expansion?
3. If forced to simplify the product by 20%, what would you remove first?
4. What single change would most improve conversion?
5. What single change would most improve retention?
6. Is the evidence layer genuinely valuable or mostly decorative?
7. Is the referral clinically useful to a physiotherapist without becoming prescriptive?
8. What remains unproven that should block a commercial pilot?

Return exactly one final verdict:

```text
REVIEW PASS
CONDITIONAL PASS
REVIEW HOLD
```

Then state:

```text
MATERIAL OPEN FINDINGS: <number>
BLOCKERS: <number>
TOP 3 CORRECTIONS: ...
TOP 3 THINGS NOT TO CHANGE: ...
```

Do not implement fixes during review. The review must remain independent from implementation.
