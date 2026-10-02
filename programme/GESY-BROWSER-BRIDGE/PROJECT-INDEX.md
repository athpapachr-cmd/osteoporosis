# GESY Browser Bridge — Project Index

> **STATUS:** feasibility workstream initialized.
> **DATE:** 2026-10-02.
> **OWNER:** Clinical Excellence / Cockpit coordinator.
> **PRODUCT OWNER:** clinician.
> **RUNTIME AUTHORITY:** none.

## Purpose

Determine whether the clinician's existing authenticated GESY provider portal session can support a bounded browser-assisted **read-only** projection into Clinical Excellence / Cockpit without relying on a public GESY API.

This is a feasibility study, not a claim that GESY exposes or authorizes an API, automation contract, or third-party integration.

## Field signal

The Product Owner reports a colleague achieved a GESY integration using Codex despite not receiving a GESY API. Treat this as a practical field signal that browser/UI automation may be viable, not as verified evidence of an official integration mechanism or permission model.

## Non-negotiable boundaries

- clinician performs portal sign-in directly;
- no GESY username/password, MFA secret, session cookie, patient export or identifiable content is committed to this repository;
- no credential is entered into ChatGPT conversation text or persisted by the feasibility artifacts;
- feasibility starts **read-only**;
- no prescribing, referral submission, authorization submission, message sending, appointment mutation or other portal write;
- no bulk scraping;
- no silent write into the protected Clinical Excellence patient record;
- browser-derived values remain source-labelled observations until accepted through the relevant clinical authority;
- failure/uncertainty must fail closed rather than infer a patient or result.

## Initial evidence target

Prove or disprove one narrow workflow:

```text
clinician already signed into GESY portal
→ user deliberately opens/selects one patient
→ browser automation reads one bounded patient page
→ extract a minimal allow-listed projection
→ show source/time/provenance
→ no portal write
→ no credential storage
```

Candidate fields for later selection include recent visits, current/recent prescriptions, referrals/investigations and authorization status. No field is approved until the actual portal surface and semantics are inspected.

## Relationship to Cockpit

This workstream is **not** on the D1.1 → Visit Brief critical path. If feasible, its output may later become one provenance-labelled source inside Visit Brief. It does not own patient identity, longitudinal clinical truth or GESY workflow.
