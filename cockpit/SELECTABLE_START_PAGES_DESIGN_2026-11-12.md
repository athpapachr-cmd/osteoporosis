# Selectable Cockpit start pages — frozen R1 design

## Product Owner checkpoint

The Product Owner asked to continue the previously discussed multiple-start-page work through implementation. The clinician should be able to choose the first surface that fits their normal work, without creating another clinical owner or reopening the previous patient.

## Problem and visible change

Today the shared **Αρχική** action always points to Cockpit Home. After this slice, the clinician can mark the currently open first-class surface as their start page:

1. **Ημέρα μου** — the existing Cockpit Home and today's context;
2. **Εβδομαδιαίο ημερολόγιο** — the existing protected Clinical Calendar, opening on the current week;
3. **Κλινικά εισερχόμενα** — the existing protected iCloud source viewer;
4. **Περιστατικά ιατρείου** — the existing surgeries / traffic cases / RF surface.

Each eligible surface exposes one compact action: **Ορισμός ως αρχικής**. The selected surface shows **Αρχική σελίδα**. The shared **Αρχική** action points to the saved choice, and entering the generic `/static/cockpit/` URL routes to it.

## Reuse and ownership

- **REBIND** the existing shared navigation and existing Cockpit surfaces; do not create a second Home, calendar, inbox or office-case owner.
- Persist only the opaque page identifier in browser `localStorage` under one versioned key. No clinical, patient, authentication, mailbox or appointment data is stored.
- This is a per-browser-profile preference. Account/device synchronization, a database table and a new preferences API are outside this slice.
- Existing Clinical Key authentication, page endpoints and data loaders remain unchanged.

## Safety and routing invariants

- Invalid, missing or inaccessible preference falls back to **Ημέρα μου**.
- **Ημέρα μου** opens today's generic state, never a previous patient or stored visit.
- Calendar opens the current week; it does not restore an old week.
- Inbox opens the existing all-mail source view without selecting a message.
- Office Cases opens the existing default cases surface without selecting a patient/case.
- Existing `canLeave()` guards remain authoritative. Selecting a start page changes future entry only and never navigates away immediately.
- Direct deep links (`chart`, selected Pending, source item) are not rewritten into start-page routes.
- The generic entry redirect is allowlisted to the four fixed same-origin paths; no arbitrary URL is accepted from storage.
- No provider call, clinical write, mailbox mutation, schema, secret, role or new external integration is introduced.

## Finite acceptance evidence

1. Deterministic script evidence for default, valid selection, corrupted storage, storage denial and generic-entry redirect.
2. Shared-navigation evidence that the Home link follows the allowlisted choice and the current eligible surface can be set.
3. Cockpit route evidence for iCloud and Office Cases, including neutral initial state and return to Home.
4. Existing Cockpit Home, Calendar, iCloud and Office Cases tests remain green.
5. Real Chromium evidence at desktop and mobile widths proves the control is reachable, does not overflow and preserves existing navigation/authentication behavior.
6. One independent exact-code R1 review, followed by exact-head CI before any merge/release.

## Explicitly outside this slice

- cross-device/user preference synchronization;
- new day/month calendar modes;
- automatic AI prioritization or redesign of **Ημέρα μου**;
- restoration of a patient, message, case, week or unfinished visit;
- any clinical or provider-side write.

