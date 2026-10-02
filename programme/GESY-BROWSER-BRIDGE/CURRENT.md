# GESY Browser Bridge — CURRENT

> **NOW (2026-10-02):** F0 feasibility only.
> **Implementation:** NOT STARTED.
> **Portal write authority:** NONE.
> **Credential persistence authority:** NONE.
> **Patient-data repository authority:** NONE.
> **Critical-path status:** independent of Cockpit D1.1 → Visit Brief.

## Current question

Can a clinician-authenticated GESY portal session be used through browser/computer automation to retrieve a small, explicitly requested, read-only patient projection safely enough to be useful inside Clinical Excellence?

## Known

- Product Owner field report: a colleague reports using Codex with the GESY portal without receiving a GESY API.
- This does not establish an official API, stable DOM contract, automation permission, or production-safe architecture.
- The clinician should authenticate directly in the portal; credentials must not be copied into repository/chat artifacts.

## Unknown / must be proven

- exact portal pages and fields available to the clinician;
- whether the portal behavior is stable enough for bounded automation;
- portal terms/policy relevant to automation;
- whether patient selection can be kept explicit and fail-closed;
- which fields are sufficiently unambiguous for structured read projection;
- whether a local/session-bound browser workflow is adequate or a formal computer-use integration is required.

## First bounded spike

1. inspect the authenticated portal manually with the Product Owner present;
2. choose **one** read-only page and **3–4** allow-listed fields;
3. run a browser-assisted retrieval without writes;
4. verify no credentials or patient data enter repository/log artifacts;
5. record only field names, control behavior, failure modes and feasibility verdict — never identifiable patient values.

## Stop rules

STOP and do not automate if the flow requires credential extraction, bypass of MFA/access controls, unsupported bulk retrieval, hidden write side effects, or unreliable patient identity selection.

## Next action

Prepare the F0 observation protocol when the Product Owner wants to run the portal feasibility session. Do not block D1.1 / Visit Brief work while waiting.
