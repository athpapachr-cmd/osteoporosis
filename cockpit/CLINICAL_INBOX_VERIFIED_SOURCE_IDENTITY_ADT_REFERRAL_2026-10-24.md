# Clinical Inbox — source-aware verified patient identity, ADT and referral links

**Date:** 2026-10-24 (Asia/Nicosia). **Status:** Product Owner direction APPROVED · affected R2 DESIGN CANDIDATE · NO RUNTIME/REAL PHI/PROVIDER AUTHORITY.

**Repo:** `athpapachr-cmd/osteoporosis`, verified production main `bccaf10aead6394f545ec44f735aa12abcb37fed`; predecessor synthetic Clinical Inbox PR #208 LIVE only in protected #161. **Separate UI correction:** PR #209 restores recent topbar peek/dropdowns, R1 HOLD; it has **no patient-link authority**.

## 1. Product Owner correction: no universal DOB test

The clinician rejects **date of birth (DOB) as a mandatory second identity check**, particularly when clinical work already provides better verified evidence. Prefer **already clinician-verified ΑΔΤ / official ID-card number, a formally mapped beneficiary identifier or an existing verified referral-to-patient mapping** where that kind of evidence is genuinely present and appropriate. DOB may be an optional contextual clue but is **never a required second field or authentication factor**.

A clinician's session authentication (`Clinical Key`, later true clinician role) is separate from patient-to-document identity verification. Neither knowing DOB nor entering ID-card digits authenticates the clinician. Minimize typing and avoid replaying already completed identity verification.

**Observed existing source limits:**
- `clinical_patients.patient_id` is a generated internal identifier, **not proven to be ADT or GESY beneficiary ID**. Its `demographics_json` has optional DOB/phone; no authoritative ADT issuer/verification/expiry mapping is established in the current clinical registry.
- The separate surgical queue owns `identity_number`. **Do not join it to Clinical Excellence patients by matching name or treat its identifier as verified clinical-registry authority**; separately approved, attributable verification would be required to rebind anything.
- Current protected `clinical_pending_items` is an obligation lifecycle, **not** an authoritative lab/referral-number registry. Its free-text description is not verified referral identity.
- The existing #161 synthetic ClinicalAttention fixture always seeds a DOB and requires `verified_dob`; **that is a test convenience, not the accepted product rule**. Do not ship that as the mandatory live-clinic workflow.

## 2. Source evidence versus patient identity — ordered clinical flow

**All input sources are untrusted for instructions and identity authority until qualified.** A recognized or previously approved sender suggests source provenance, not verified document/patient linkage.

1. **Scope and source first.** Classify clinical content as GESY *availability notification*, laboratory values/report, imaging report or other. Preserve mailbox account, message/attachment/subject scope, source revision and received/source-stated times separately. A PDF with mixed/unknown subjects stays **Inbox-only** until safely segmented and independently confirmed.
2. **Reuse a prior verified patient–source relationship where available.** A stable source/issuer-specific **referral/order identifier previously confirmed to belong to one `clinical_patient_id`**, with matching document/subject scope, no conflict or revocation, is strong linkage evidence. No re-entry of DOB or ADT is required merely because a new notification arrives for that already verified referral. Do not dedupe different referrals or cross-account message receipts just because the name matches.
3. **Verified identity document when the source actually contains one.** If an incoming document states an ID-card number (ΑΔΤ) or supported official beneficiary ID, compare it only to a **previously verified and issuer-qualified protected clinical registry identity mapping** with one unambiguous target and no conflict. A bare number typed earlier into demographics, sender text, or the separate surgery queue is **not such mapping**. Use a minimal masked evidence cue in UI; no raw identifier in source URLs, public logs, routine UI lists or CI.
4. **No strong match: candidate only.** Names, email addresses, shared phones, symptom/site, appointment labels, and an *unlinked* referral number/date may help propose cards, but **never prove identity alone**. The clinician may open the source and selected protected card, compare available independently recorded identifiers or an existing patient-specific request, then explicitly confirm the scoped association with an auditable evidence type. No mandatory DOB. If independent support is lacking, keep `unlinked / candidate_only`, ask for clarification via normal clinic process, and block patient chart projection, authoritative write and outbound contact.
5. **After explicit clinician confirmation, reuse that exact scope.** Bind the qualified issuer + referral/order ID + document/subject scope + approved clinical patient to protected provenance and version, without creating a competing CareTask/clinical record. Subsequent matching notifications can reuse this evidence if still current and conflict-free, with visible source and review status. Patient reassignment or corrected document invalidates stale review/disposition/unsent approvals under existing CAS/409 and append-only audit; already sent communication remains historical.

**A simple name + matching dates or a clinician's check-box without independent supporting evidence does not become 'strong auto-link'.** A deliberate clinician confirmation must say what record/source was compared, but it should normally take one action once those facts are already displayed.

## 3. Concrete synthetic GESY-style example (NEVER real patient content)

An invented notification says *“Results available; beneficiary SYN-PATIENT-X; referral REF-DEMO-A; referral date 2026-09-01; immunology.”* It **does not contain analyte values or a clinical result**.

- **Known verified link:** The protected source history already binds `REF-DEMO-A` to clinical patient `SYN-PATIENT-X`, confirmed against original referral. Show patient and reference evidence, and permit clinician review without DOB; notification remains **results available / review pending**, not a completed test or reviewed values.
- **No verified link but verified ADT in both subject and registry:** After exact strong, unique comparison, show the approved patient candidate with masked identifier/provenance for clinician to accept when required. Never copy the raw ADT to a message, open URL or task title.
- **Only name + referral number:** Propose possible chart matches; require actual clinician-supported scoped confirmation, not automatic attachment. If two same-name patients exist, show conflict; no attachment, communication, task closure or invented result interpretation.

Do **not** use a sender's email address to identify a patient. For an actual incoming GESY email, sender/recipients can contain clinical private data and require separate owner/provider security approval; the existing #208 fixture remains entirely synthetic.

## 4. Minimal protected contract (proposal only; DDL/grants/retention gate OPEN)

Rebind/extend the **existing protected Clinical Excellence patient authority**, with an adjacent *verified identity evidence* relationship if needed, rather than creating a second registry or task database.

| Evidence owner | Proposed minimal persisted fields | Required gate |
|---|---|---|
| Protected `PatientIdentityEvidence` | Internal `clinical_patient_id`, `identity_kind` (verified ADT / qualified beneficiary mapping), `issuer`, server-only normalized equality lookup (encrypted reference or controlled keyed blind index), verification `state`, `reviewer/at`, version/revocation/conflict, source/evidence provenance | Separate privacy minimization, access roles, secure encryption/key rotation, retention/deletion, DDL/grants and clinician verification before storing *real* identifiers |
| Protected `ReferralSubjectLink` (existing clinical referral owner extended if proven; otherwise adjacent verified mapping) | Issuer/source family, referral/order key and source-stated date precision, verified patient ID, original source reference/subject scope, clinician approval/evidence, version/conflict/revocation and linked ClinicalAttention item ID | Exact unique-scope constraints, issuer-qualified normalization, patient correction/audit, retention/grants; reference to existing Pending/visit request without making the Inbox a referral writer |
| Source/attention relationship | Item ID, source receipt revision, independently verified patient link reference, `candidate_only/confirmed/conflict`, review state, clinician provenance and CAS revision | Reuse previously R2-approved `ClinicalAttentionItem` source/identity/review axes; never mark lab accepted or CareTask resolved from email arrival |

**No new table, ADT extraction, backfill, database grant or persistent identifier storage is authorized in this design-only slice.** Do not make a temporary global duplicate directory of ADT values. Only protected server-mediated comparisons may surface "verified match"; lack of reliable key ownership or conflicting mappings must fail closed.

## 5. UX contract: minimal actions, explicit uncertainty

- First popup: identify source/type, source-stated name and referral (safely minimized), then suggest existing card **with reason** (e.g. *“Συνδεδεμένο παραπεμπτικό — επιβεβαιώθηκε σε προηγούμενη επίσκεψη”* or *“Ίδιο όνομα μόνο — ανεπιβεβαίωτο”*).
- **Verified unambiguous existing association:** display `Συνδεδεμένος ασθενής` + evidence, review action enabled, **no DOB/ADT re-entry**.
- **Strong but new qualified identity evidence:** show preselected candidate + one clinician *“Επιβεβαίωση ασθενούς”* after source/card comparison. Do not turn a weak name into strong evidence automatically.
- **Weak/ambiguous/no ID available:** keep manual patient search and explicit evidence-based review; allow *“Χωρίς επιβεβαιωμένη σύνδεση”*, no forced new patient creation or mandatory alternate ID entry.
- New data arrival is NOT `Εξετάσεις ελέγχθηκαν`. For GESY availability notifications, display *“Αποτελέσματα διαθέσιμα — εκκρεμεί ιατρικός έλεγχος”*. Confirmed patient link is NOT permission to accept test values, complete visit obligations, send SMS or book.
- Correcting a link must remove prior patient's read projection, invalidate still-unsent approvals, and preserve the audit/unknown external outcomes. An expired clinician session, conflicting issuer, stale form or source correction fails closed.

The top-right recent-item peek restoration is independently scoped to [PR #209](https://github.com/athpapachr-cmd/osteoporosis/pull/209), and is not changed by this design.

## 6. Finite gates and implementation sequence

**Pre-code gate now:** ONE independent **affected R2** review of this changed identity-source/ADT/referral binding and minimal persistence boundary; reuse existing Q1–Q6 / PR #208 R2-closed ownership, mixed-doc, CAS and sender untrust. Do not reopen settled review questions.

**A — later separately authorized synthetic correction:** Update #161 invented patient evidence to fake verified ADT/referral fields, remove compulsory DOB input from the synthetic confirmation flow, preserve exact patient-scoped verification/CAS, and demonstrate GESY-style *results available* notification without medical values. Separate code/test head, one post-code affected R2 before deployment; **no real ADT or email**.

**B — later separately authorized protected persistence:** Prove/source the true ID-card and referral ownership; clinic actor/role rights, field-level retention/deletion, encryption, DDL/grants, migration and stable patient/source links. No generic clinical record promotion, no use of separate surgical identity as authority. This is not approved by synthetic A.

**C — live external sources:** Qualify approved Gmail, then Outlook/Hotmail and iCloud account adapters, with independent account-specific origin, incremental coverage, disconnect, privacy and source-adapter gates. Reception still owns provider transport; Clinical Excellence retains clinical decision authority.

**Stop rule:** If no reliable source-level ADT or verified referral mapping exists, the correct behavior is **clinician confirmation with documented independent evidence or remaining unlinked**, never a hard-required DOB fallback. This product change does not authorize active mailbox integration or production patient matching.
