# Clinical Inbox — actual source routing correction (2026-10-25)

**Product Owner decision:** One clinician currently uses Cockpit and personally holds the existing Clinical Key. Do not introduce multi-staff identity/auth refactoring merely as a first-use prerequisite. The existing key is possession-based, not proof of an individually attributable human: additional-person access/strong attribution remains a separate future gate. No production authorization follows from this document.

## True incoming sources (never publish mailbox credentials or patient information)

- **Reception / Papachristou Ortho:** the already running Digital Secretary dashboard owns patient call records, callback/message status and narratives. Gmail contains derived call notifications and ordinary booking/admin messages, **not** the primary clinical-results feed. First slice reads directly from Reception's protected dashboard API; do not parse Gmail copies into a duplicate inbox or obligation owner.
- **Microsoft personal Hotmail:** receives GESY notifications. Future *read-only* Microsoft Graph adapter, separately qualified per-account and provider. A notification that results are available is **not** the results, an accepted clinical finding or a completed Pending task.
- **iCloud Mail:** receives clinical documents/results sent by patients. Future secure read-only iCloud Mail adapter; inspect actual documents only after privacy/retention/credential and attachment-processing gates. Sender, display name, phone, DOB and subject alone may not attach documents to protected patients.

**Scope now:** bounded first-use Cockpit receptionist-call peek, existing Clinical Key only, no new account login, no recipient email access, no patient association, no copying to ClinicalPending, no Gmail, no Hotmail, no iCloud, no outbound actions.

**Code contract:** new `/clinical/reception-calls/peek` route is OFF unless explicitly enabled; server-to-server HTTPS `GET /dashboard/calls`, server-only dashboard key, timeouts, fail-closed errors, no redirects, no PHI logs, minified four-row read projection, no storage and no fabricated unread count. Reuse Home topbar popover and show original Reception dashboard link for full history/handling. Existing Reception remains sole task/handled authority. Do not display stale caller content after logout/panel switch.

**Explicit future privacy/operation gates:** a private, scoped credential rather than broad dashboard-key is preferred before enabling PHI access; authenticated transport, field/retention/backup purposes and producer freshness must be verified in an operator-controlled environment. Never store mailbox addresses or secrets in public GitHub, CI, browser or logs.

**Later:** a separate reviewed Hotmail→GESY intake and iCloud→patient-results intake, with account-scoped receipts, coverage truth, document security/revision handling and clinician-verified identity only. Reconcile old "Gmail first" product roadmap language before treating it as current instruction.
