# Cockpit iCloud — usable one-doctor clinical pilot

**31 Oct 2026 · existing Clinical Key · no new user/role/referral identifier**

Scope: correct the real UX failure after PR #265. Before, iCloud subject-only, bulk/no-subject default caused unrelated appointment notifications, payment receipts and landlord mail to appear with the same non-actionable "Για έλεγχο" label. The icon told the clinician to navigate through Shortcuts.

## Implemented pilot behavior

- Both entry routes remain: ✉ topbar -> directly opens **Όλα τα email**, and Shortcuts -> Εισερχόμενα ασθενών. A full-screen category selector offers Κλινικά, Διοικητικά, Όλα. **All** preserves access to every To/Cc alias-scoped INBOX item (excluding none due to category), with the existing bounded continuation and read-only UID-bound IMAP operations.
- Header-only, conservative clinical / administrative / other / uncertain suggestions. Medical institution sender descriptors can classify clinical notifications without a Subject; Cal.com events, payouts, invoices, rent are administrative by default. No LLM or automatic patient matching. Practice-specific names belong in private server variable ICLOUD_TRIAGE_ADMIN_HINTS, not in source control. Manual classification overrides are stored in a doctor-only event ledger; all mail remains recoverable.
- Selecting an email retains existing plaintext and safe attachment download. It adds actions **Χαρακτηρισμός**, **Βρες επίσκεψη**, choice of exact existing patient/Pending or appointment without chart, **Είδα/ενημέρωση**, manual contact attempt/informed and explicit doctor callback closure. No automatic external calls, Gmail/Apple mailbox writes, or external links followed.
- Candidate search checks full name AND phone when both are found; weaker name-only or manually searched matches are clearly identified and **require the doctor's choice**. No GESY referral number, extra role, or forced synthetic chart. Never link solely by email address or name automatically.
- Additive private table clinical_icloud_message_decisions and append-only clinical_icloud_message_action_events store ONLY hashed mailbox+UID source key, selected internal patient/Pending/appointment references, category, timestamps and action fingerprint. No raw email subject, sender, body, personal phone or attachment data is stored. DDL attempts are scoped to the new tables and failure leaves live read-only inbox intact.
- Signed source and active Pending are validated when a link is chosen; a cancelled source invalidates new actions. The existing ClinicalPending is **not** automatically resolved. The existing chart/obligations endpoint projects that same linked mail event state inside the original Pending, not as an independent clinical task. A doctor may need to mark the original Pending complete when additional requested materials are still outstanding.
- The UI actions require the existing clinician Clinical Key, a same-origin POST and explicit doctor click. Role-wide staff credentials DO NOT authorize them. The separately built PR #270 is NOT required for this solo pilot.

## Limits / acceptance

This is a controlled, single-doctor pilot, not a general-purpose identity matcher. A telephone number shared within a family is not alone sufficient, and sender display names are untrusted navigational hints. Real iCloud provider sorting requires clinician acceptance; CLI/CI tests use synthetic mail only. Outbound patient communications are NOT implemented.

Run exact-head Python/Node/Chromium tests, verify new database tables are installed without breaking existing protected routes, no real names/senders/PHI appear in logs, test the doctor-only iCloud inbox and classify/link/review one actual message under existing login. Confirm iCloud provider remains EXAMINE/BODY.PEEK and doesn't mutate Seen.

This work responds to the current sole user, with the secretary planned in about 3-4 weeks. A separate restricted secretary interface is future work, not a gate for this physician pilot. No personal account provisioning or recurring referral-number entry is required.
