# Cockpit iCloud Mail — actual first-source implementation candidate
Date: 2026-10-30. Status: DRAFT / provider not activated / no real email read.

Goal: show only potentially patient-related messages addressed to the practice's iCloud Mail alias inside Cockpit. This is not a general mailbox mirror, clinical identity reconciliation, or results acceptance.

## Implementation
- Existing Clinical Key browser session and protected API route /clinical/icloud-mail.
- Fixed TLS IMAP endpoint imap.mail.me.com:993. Read-only EXAMINE, UID SEARCH on To/Cc for the configured alias and BODY.PEEK reads. No SMTP, STORE, EXPUNGE, flag changes, marking read, or provider writes. No timer, polling, caching, database persistence, or patient-record linkage.
- Separate server-only environment: ICLOUD_CLINICAL_READ_ENABLED=1, ICLOUD_CLINICAL_IMAP_USERNAME (the primary iCloud Mail username; NOT assumed to be the Hotmail Apple Account sign-in or the alias), ICLOUD_CLINICAL_APP_PASSWORD (new Apple app-specific password), ICLOUD_CLINICAL_ALIAS (practice recipient address). Do not put values into GitHub, chat, URLs, CI, or logs. OFF by default; missing setting fails 503.
- Header-only search and a bounded 80-recent-alias-candidate scan, maximum 4 peek / 25 full list. Visible To/Cc must match the alias exactly after the server search. Bulk/List-Unsubscribe mail is omitted. Subjects indicating likely clinical relevance are listed as possible clinical; otherwise "Για έλεγχο", not assigned to any patient. No claim of perfect filtering. Hidden/Bcc recipients, messages in other folders, very old results and mailing-list anomalies are outside this coverage.
- Detail revalidates alias, UIDVALIDITY, provider-reported size and current message headers. Maximum 12 MiB message, 14k text characters displayed as inert text; HTML fallback has markup stripped. Only bounded PDF/JPEG/PNG/WEBP attachments (<=8 MiB) are downloadable as octet-stream with no-store, nosniff, sandbox and generic filenames. No inline active HTML, no PDF extraction/AI processing, no forwarding.
- UI: the existing top-right ✉ peek links to a protected full iCloud viewer; the synthetic #161 Inbox remains isolated. Session expiry, panel switch and closing wipe transient message DOM and fence late results. Zero real-provider data in tests.

## Limitations and safe activation
- Apple's app-specific password can grant broader IMAP capability to another client; "read-only" is enforced by this application's fixed IMAP command path, NOT by the Apple credential. Store it only as a Render server secret and revoke it in Apple Account to cut access. The app itself never modifies mail.
- The fact that a message was sent to an alias, or that its subject contains clinical keywords, does not establish that it belongs to a particular patient. No automatic ClinicalPending or Visit Brief projection.
- The clinician already confirmed the practice address is an iCloud alias. Apple states IMAP username is usually the primary iCloud Mail address local-part; do not confuse the Apple Account login email with the IMAP username. Determine it privately during operator configuration, not by asking for the mailbox password in chat.
- No claim of production support, test of personal mail, or completed connection before exact-head CI, focused security review, explicit merge/deploy authority and a controlled non-PHI provider smoke.
- Rollback: keep ICLOUD_CLINICAL_READ_ENABLED unset or set to 0 (restores existing disconnected state); subsequently revert exact deployed source if needed. Do not touch the other Reception or Clinical Inbox connectors.

Official Apple source: https://support.apple.com/el-cy/102525
