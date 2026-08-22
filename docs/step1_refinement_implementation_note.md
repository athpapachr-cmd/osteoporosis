# Step 1 refinement implementation note

Implemented on `feat/baseline-step1-risk-refinement` after live review of the first deployed pilot screen.

Key implementation changes:

- explicit `new_to_service` vs `established_patient` axis;
- explicit encounter archetype axis;
- automatic BMI from weight + height when both are available;
- current/reference height and derived height loss;
- prior fragility fracture with last site + month/year instead of a flat historical-fracture checkbox;
- glucocorticoid dose + duration capture;
- falls count over 12 months;
- structured secondary/associated conditions;
- conditional frailty/immobility details;
- sarcopenia case-finding trigger with optional SARC-F;
- Heidi no longer asks for free-text correction descriptions or transcripts; optional one-click correction categories only;
- derived applicability context is stored silently and not presented as baseline coaching.

The implementation remains prototype-only: case data are browser-local, unencrypted, and no identifiable clinical data should be entered.
