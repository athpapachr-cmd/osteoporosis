# Visit Brief + Clinical Inbox — bounded independent R2 pre-code request

> **STATUS:** PREPARED / NOT DISPATCHED / NO VERDICT / READ-ONLY.
> **DATE:** 2026-10-07 Asia/Nicosia.
> **TIER:** R2: patient identity, clinical candidate/review semantics, identifiable Gmail material, changed intake ownership and new clinician-approved external messaging effect.
> **TARGET REPOSITORY:** `athpapachr-cmd/osteoporosis`.
> **TARGET BRANCH:** `docs/cockpit-visit-brief-clinical-inbox-2026-10-07`.
> **BASE MAIN:** `045798dfa28612b268f16a99262c6ecc9ca4829d`.
> **DESIGN:** `cockpit/VISIT_BRIEF_CLINICAL_INBOX_DESIGN_2026-10-07.md`.
> **EXACT DESIGN BLOB:** `8ee79a96350f9ac42059eeb2a3e836cbe9d11245`.
> **RECEPTION REFERENCE:** Backend `e6babb5ab758d282166767c36dd7311024406afb`; Ops `750c006ba1fb73752a87c181671355a1c69dca16`, read-only references. Fresh drift must be classified before relying on a changed seam.
> **SUPERSEDES:** the unused 2026-10-06 Visit Brief / Lab Results request, archived unchanged. No old Visit Brief verdict exists to reuse.

## Decision and authority

Review whether the revised **two-entry design** preserves product intent and has an adequately bounded, implementable shared authority/identity/review/intake/communication architecture. Identify the smallest correction and exact remaining gate before any proposed A/B/C implementation boundary may be authorized. Calendar Unification is already RELEASED / PRODUCT OWNER SMOKE VERIFIED; do not re-review its closed release or wait for its old smoke-pending checkpoint.

The Product Owner requires:

- click appointment/name → floating Visit Brief, with compact Home context;
- new Gmail lab/MRI/X-ray/imaging/GESY material → independent always-current Clinical Inbox, usable without any appointment;
- same-day source review and deliberately approved patient communication, with one later longitudinal history;
- strong/clinician-confirmed patient linkage and separate recipient confirmation;
- source-specific candidates, no automatic authoritative write/task closure;
- Reception communication admission/orchestration and Zadarma transport;
- correctness while free Cockpit sleeps, with durable intake/delivery recovery;
- existing configured callback windows, without per-patient callback booking.

This request prepares one independent pre-code review. The author is not its independent reviewer. No runtime edits, test/eval execution, dependency installation, provider/config/secret mutation, live mailbox/document access, patient messaging, booking, merge, deploy, P0-V0 release, PR-1 work or new agent delegation is authorized. A PASS is a design disposition only, not implementation/release authority or proof of integration readiness.

## Bootstrap once and establish an evidence map

Fresh-verify Osteoporosis main; read its six canonicals in `AGENTS.md` order, then `PROCEDURES.md` P2–P5.1/P7, `cockpit/CURRENT.md`, the constitution and exact design. Verify the blob from the declared branch before judging it. Record actual source heads and any drift. Root `SLICE_PLAN_CURRENT.md` still owns PR-1; this request concerns the parallel Cockpit design and does not transfer the root primary lifecycle.

When inspecting Reception source, follow its `AGENTS.md` pointer to fresh Ops governance/current authority and applicable source-grounding/finite-review procedures. That bootstrap does not authorize examining unrelated voice/booking projects or mutating Ops. Existing source observations here are navigation hints, not a reviewer verdict. No `orctl`/software-gate PASS from the author is claimed; use the receiver's applicable current gate or label its availability limit. A shared helper/reference alone is not an assignment to review every caller.

Before searching, map each Q below to the smallest source/contract sufficient to decide it. Report files actually read and the question they answered. Source inspection and the design's future regression oracles are sufficient pre-code evidence; no broad runtime suite or production probe is required.

## Finite declared questions

### Q1 — two independent entrances, one clinical-attention lifecycle

**Affected surface:** design §§1–3/6/8/12; constitution; phase-plan §34; current appointment/G3/protected-record read seams only as needed.

**Sufficient evidence:** trace the synthetic no-appointment arrival to independent Inbox review/action and its later Visit Brief projection; trace an unlinked appointment to context-only overlay then confirmed record; map review/disposition/communication references to one clinical owner. Confirm that A-only release is not labelled live Gmail completion, P0-V0 is not assumed deployed and G3 reuse preserves missing/conflict/draft/actual distinctions.

**PASS:** both entrances can represent the same scoped record without appointment dependency or duplicate history/effects. **BLOCK:** design requires attendance or duplicates/loses clinical state. **UNKNOWN:** a necessary owner/consumer cannot be grounded; name that seam and stop.

### Q2 — always-on Gmail intake, free-service delivery and freshness

**Affected surface:** design §§4/6/10–12; Reception hosting/delivery seams in `main.py`/`render.yaml` only; official Gmail sync/push/scope contracts cited by the design.

**Sufficient evidence:** trace asleep-Cockpit arrival → durable source receipt/outbox → idempotent clinical ingress → acknowledgement → visible work. Check cursor advancement, bounded/incomplete pages, expired history/reconciliation, downtime backlog, source deletion and unavailable/stale-vs-empty behavior. Determine whether proposed Reception extension is an adjacent source-delivery role rather than a second clinical/Gmail authority, and whether all required first-integration configuration/latency gates are explicit. Verify first poll/coverage requirements without inventing a guaranteed SLA or live OAuth readiness.

**PASS:** intake correctness survives sleep and failure with truthful coverage, one reader and recoverable delivery. **BLOCK:** opening Cockpit remains necessary for discovery, failed batches advance coverage or delivery loses/duplicates records. **UNKNOWN:** required hosting/provider fact is unavailable; distinguish the specific activation gap from unrelated voice service HOLD.

### Q3 — scoped patient identity, contact authority and invalidation

**Affected surface:** design §§2/5–7/9; `clinical_data.py` patient contracts; `clinical_calendar.py` effective live projection; current Reception contact seam only if used.

**Sufficient evidence:** check internal generic patient ID vs source identifiers; name/fuzzy/phone correlation cannot become clinical authority; multi-patient message/document scope; clinician confirmation evidence; unlinked source access without candidate-patient history. Trace separate approved recipient snapshot plus link/item/text revision binding, subject switch/stale response/concurrent correction and invalidated unsent action.

**PASS:** only authorized strong/confirmed links attach clinical content, and send uses independently confirmed recipient/current approval. **BLOCK:** weak evidence, inherited message-wide link or stale subject/contact can leak history or send to the wrong person. **UNKNOWN:** exact verification/contact contract needed for the proposed boundary is unresolved; do not assume demographics supplies a verified phone.

### Q4 — source-specific semantics and authoritative-result separation

**Affected surface:** design §§5–6/8; existing protected lab create/update/read seams in `clinical_data.py`/`clinical_data_ext.py`; no clinical-guideline review or automatic treatment design.

**Sufficient evidence:** one realistic example each for laboratory candidate value, MRI report summary, X-ray/imaging report, GESY referral/authorization notification, and unsupported/mixed content. Confirm units/date precision/comparators/uncertainty/provenance, source findings vs clinician interpretation, report vs image analysis and referral vs completed result. Trace reviewed/communicated states to prove they do not promote values or close CareTasks. Determine the minimum later acceptance seam without claiming the current flexible values store already supports required provenance/idempotency.

**PASS:** distinct adapters and authority states preserve the source meaning and no silent clinical promotion. **BLOCK:** lab parsing of report/notification, inferred completed investigation, diagnosis/normality or automatic authoritative write/closure. **UNKNOWN:** a required typed adapter/acceptance contract is unavailable; distinguish a deferred extension from a claim of first-slice support.

### Q5 — clinician-approved Reception → Zadarma action and callback rules

**Affected surface:** design §9/6–7; Reception `main.py` `_zadarma_auth_header`, `_zadarma_post`, `_zadarma_sms_to_doctor` and the direct urgent-outbound consumer; `render.yaml` variable names only; official Zadarma API.

**Sufficient evidence:** identify REUSE/EXTEND boundaries from existing urgent-to-doctor helper to new patient action; trace authenticated clinician approval, recipient/text/revision binding, durable idempotency admission before effect, same-key replay, target-specific rejection, failure/outcome_unknown and explicit reconciliation. Check no service token or urgency flag substitutes for clinician authority, no raw provider error escapes, no voice-agent dependency or duplicate transport owner. Preserve working-weekday 10:00–10:20 / 17:40–18:00 Asia/Nicosia configured callback policy, no booked slot, inferred outbound physician call or automatic booking. Provider delivery/reference/idempotency facts stay unknown unless supported.

**PASS:** one lawful approved transport action with truthful result and non-booked current-policy callback semantics. **BLOCK:** autonomous/stale/wrong-recipient send, blind unknown retry, HTTP success treated as delivery or competing messaging authority. **UNKNOWN:** a required provider or action contract is unproven; name the minimal source/qualification needed, not a broad telephony redesign.

### Q6 — privacy/retention, failure and bounded first-code eligibility

**Affected surface:** design §§10–13 and the Q1–Q5 cumulative affected boundaries only; root PR-1 HOLD and existing processor/privacy owners as boundaries, not redesign assignments.

**Sufficient evidence:** enumerate minimal retained source/candidate/link/action fields and rightful owner, transient raw viewing, safe document limits/errors, auth/access, browser/log/public-Git exposure, disconnect/source-unavailable behavior and exact open field-level/provider/retention gates. Assess the staged A/B/C scope and future acceptance oracles. State whether missing facts prevent a particular contract from being frozen before code; do not convert intentional live-activation deferral into a blanket design PASS or silently close it. No extra privacy workstream, processor approval or indefinite candidate retention is implied.

**PASS:** architecture is coherent and its exact pre-code/live gates are truthful and sufficient; specify which boundary is eligible only after separate authority/required contract freeze. **BLOCK:** a required field-level contract is materially underspecified, retained PHI escapes its owner, or later/privacy/synthetic approval is used as live permission. **UNKNOWN:** necessary retention/processor/source facts cannot be decided from allowed evidence; return the exact gap. Neither PASS nor UNKNOWN authorizes code in this current task.

## Evidence allow-list and expansion

Beyond mandatory canonical bootstrap, use only the exact design, constitution, root/Cockpit CURRENT, this request, the affected TODO/phase entries, current protected clinical/patient/lab/Calendar contracts, released G3 summary seam, current Reception helpers/direct consumer/hosting-delivery seam, and cited official Gmail/Zadarma documentation. Verify symbols/heads rather than infer behavior from names or config. Read the archived predecessor only to resolve a specific supersession contradiction; do not recursively audit its history.

Expand only for a source-proven material dependency needed to close one question: record the exact causal link, affected question/source and sufficient closure evidence before following it. Do not expand to general email clients, whole-service GDPR/security certification, voice-agent restoration, booking/availability redesign, full D2, GESY portal integration, P0-V0 release or automatic clinical decision support. If required scope cannot be bounded, return UNKNOWN/PARTIAL with that gap; do not keep searching for weaker substitute evidence.

## Global terminal rule

- All required Q1–Q6 disposed with sufficient affected coverage → **PASS / COMPLETE_FOR_DECLARED_SCOPE**, STOP.
- Reachable material blocker → finish only its directly affected actionable trace, return **BLOCK / PARTIAL** (or COMPLETE only if every question was already disposed), mark untouched questions NOT_REVIEWED, STOP.
- Required evidence unavailable → **UNKNOWN / PARTIAL**, exact gap and untouched questions, STOP.

`COMPLETE_FOR_DECLARED_SCOPE` means these declared questions are disposed, not absence of every conceivable defect. `NO ADDITIONAL MATERIAL FINDING` is bounded to the inspected surface. No repeated source tour, review-of-review, broad tests, provider probe or release action follows. A BLOCK receives the smallest correction and, if corrected, one delta + affected cumulative closure review.

## Requested handback

Return one compact report containing:

1. exact actual base/branch/head, verified design blob and Reception/Ops source identities (including drift/limits);
2. overall verdict, coverage COMPLETE/PARTIAL and P0:P1:P2 counts;
3. Q1–Q6 dispositions with the decisive source/evidence and files inspected;
4. each material finding: clinician/patient consequence → violated contract → exact source → smallest correction;
5. unresolved integration/retention/contract gates and the narrow first-code boundary, if any, eligible for a later authorization;
6. no implementation/merge/deploy/provider/patient-data authority, and the reason the review stopped.

Do not independently certify the author or Calendar PASS. Do not send messages to another chat or launch reviewers from this prepared request.
