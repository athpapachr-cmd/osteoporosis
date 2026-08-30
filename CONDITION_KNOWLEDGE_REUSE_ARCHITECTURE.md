# Condition Knowledge Reuse Architecture

Status: architectural design decision
Date: 2026-08-30
Scope: Personal Clinical Excellence System / future Cockpit condition modules

## Core principle

Condition-specific clinical knowledge collected while building a utility must not become disposable utility-specific logic.

For every condition, the system should progressively build one reusable structured clinical knowledge model that can later support multiple product surfaces.

```text
CONDITION KNOWLEDGE MODEL
        |
        +--> Clinical assessment card
        +--> Differential / safety support
        +--> Referral generator
        +--> Clinician evidence panel
        +--> Follow-up / reassessment
        +--> Longitudinal outcome tracking
        +--> Learning / challenge / audit
```

The referral generator is therefore a projection of condition knowledge, not the owner of that knowledge.

## Data ownership layers

Each condition-specific datum should declare one or more intended uses rather than being copied independently into each feature.

Suggested projection scopes:

- `clinical_assessment_core`
- `clinical_assessment_optional`
- `referral_context`
- `referral_core`
- `clinician_evidence_only`
- `therapist_execution_detail`
- `follow_up_reassessment`
- `longitudinal_measurement`
- `treatment_history`

A datum can belong to more than one scope, but no projection should be automatic merely because the datum exists.

## Example: primary frozen shoulder

The current frozen-shoulder evidence work should later seed a dedicated clinical assessment card rather than be recreated.

Potential reusable domains include:

### Diagnostic / assessment context
- clinician-established primary frozen shoulder vs secondary/other stiff shoulder
- laterality
- symptom duration
- pain pattern including night/rest pain when present
- active ROM restriction
- passive ROM restriction
- painful active/passive ROM
- functional limitations
- differential / structural context

### Rehabilitation-relevant context
- clinician-entered tissue irritability: high / moderate / low / uncertain-not-assessed
- irritability must not be inferred automatically from findings
- qualitative ROM restriction is sufficient for referral context
- numerical ROM may be captured only as optional measurement when actually measured

### Optional measurements
- goniometric ROM when measured with an adequately standardized method
- SPADI / DASH or other validated outcome measure when actually collected

These measurements may support longitudinal reassessment but are not mandatory referral fields.

### Treatment history
- prior intra-articular injection
- medication / other treatment
- response to previous treatment

Treatment-history data belong to the clinician's condition record. They are not automatically rendered into the physiotherapy referral; a future reviewed projection may expose them only when they materially affect the handoff.

### Referral projection
The physiotherapy referral should receive only clinically useful handoff information:
- diagnosis/presentation and laterality
- actual relevant findings
- actual functional impairment
- explicit restrictions/safety context
- rehabilitation-relevant irritability when the clinician has assessed it and when it changes treatment intensity
- evidence-bounded rehabilitation direction

The referral should not automatically include:
- separate system fields already handled elsewhere such as ICD code or session count
- unstandardized numeric ROM estimates
- treatment-history details that do not alter the physiotherapy handoff
- evidence commentary intended only for the clinician
- therapist-level exercise prescription details

## ROM measurement rule

Physician and physiotherapist measurements may serve different purposes.

- The referring clinician may record qualitative ROM restriction and may optionally record numerical ROM if actually measured.
- The physiotherapist is expected to perform their own baseline examination and measurements for treatment planning and progress assessment.
- Numerical values obtained by different observers/methods should not be treated as directly interchangeable longitudinal measurements without adequate standardization.
- The clinician may repeat their own assessment at medical follow-up using a consistent method.

## Irritability rule

Tissue irritability is clinically relevant to rehabilitation intensity and may be exposed as an optional condition-specific clinician-entered field.

It must:
- be optional unless future evidence/clinical workflow explicitly requires it;
- support `high`, `moderate`, `low`, and `uncertain_or_not_assessed` states;
- never be silently inferred by runtime from pain/ROM checkboxes;
- influence referral wording only through reviewed condition-specific authority.

## UI reuse rule

Dynamic clinical controls should ultimately resolve through:

```text
profile
 -> route / condition
 -> subtype
 -> context / management branch
```

The same condition model should determine which findings, measurements and optional contextual fields are shown in both the future clinical card and relevant utilities. Profile-wide checkbox lists are only a coarse fallback for genuinely shared options.

## Future-condition rule

This pattern applies to every condition added to the Cockpit / Clinical Excellence System. Evidence review done for physiotherapy, audit, education, or another utility should be structured so that clinically reusable facts, assessment elements, treatment-history elements, follow-up measurements and referral projections can later be composed into the condition's full clinical card without re-researching the condition from scratch.

## Non-goals

This architecture does not imply that all collected data must appear in every output. Reuse means one source of structured clinical truth with projection-specific filtering, not one giant form or one giant referral.
