# Selectable Cockpit start pages — release receipt

Status: TECHNICAL RELEASE COMPLETE. Exact source is LIVE and bounded no-PHI production browser smoke PASS. Clinician preference/usability acceptance remains separate.

- Product Owner authorized implementation through release.
- PR [#286](https://github.com/athpapachr-cmd/osteoporosis/pull/286) frozen final head `4b8aec3a77e6713e9e6ac6c900c1627d5c1de5c7`; all 18 final check runs SUCCESS, none pending.
- Independent exact-head Bugbot verdict PASS, no actionable findings, P0:P1:P2=`0:0:0`. The shared Physio test remained byte-identical to `main`; five inherited CI workflows use Playwright Python `1.63.0` for Ubuntu 26.04 compatibility.
- Squash merge `09988d485b157e6218c46b5177244211bb41138b` on `main`.
- One native Render `new_commit` deploy only: `dep-dbr21tff3r2c73bqumn0`, exact source `09988d485b157e6218c46b5177244211bb41138b`, LIVE `2026-11-12T20:28:30.535477Z`. No manual duplicate deploy.

## Released behavior

The clinician can mark any of four existing surfaces as the browser profile's start page:

- **Ημέρα μου**;
- **Εβδομαδιαίο ημερολόγιο**;
- **Κλινικά εισερχόμενα**;
- **Περιστατικά ιατρείου**.

The shared Home link follows the allowlisted preference. A generic Cockpit entry follows it too, while explicit Day, selected chart and other deep links remain authoritative. Invalid or unavailable local storage falls back to Day. The preference stores one page id only; it contains no patient, appointment, message, case or credential data. It is intentionally local to each browser profile, so another Mac or browser chooses its own start page.

## Bounded production smoke

- HTTP `200`: `/static/cockpit/?view=day`, `/static/cockpit/start-page.js`, `/static/cockpit/shared-nav.js`.
- Served Cockpit HTML includes the new bootstrap; served JavaScript includes the expected storage key, fixed routes and Greek selection control.
- Actual headless production browser selected and reopened all four choices successfully:
  - `day` -> `/static/cockpit/`;
  - `calendar` -> `/static/clinical-calendar/`;
  - `inbox` -> `/static/cockpit/#icloud-mail`;
  - `cases` -> `/static/cockpit/#office-cases`.
- Explicit `/static/cockpit/?view=day` remained on Day while Calendar was preferred.
- Release-window Render application ERROR logs: `0`; HTTP `500/502/503`: `0`.
- Observed instance after release: CPU `0.0628/0.15`, memory about `241/537 MB`. HTTP latency/request metrics had not populated in the bounded observation window and are not claimed.

No authenticated clinical request, patient/provider write, database/schema/grant/network/secret/service-setting change or production workflow mutation was performed. Source-equivalent local and CI browser tests own broader behavior evidence. Final documentation publication uses `[skip render]`.
