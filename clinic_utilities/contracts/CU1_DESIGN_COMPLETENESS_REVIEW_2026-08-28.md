# CU-1 exact design-completeness review — 2026-08-28

> **HISTORICAL SNAPSHOT NOTE (2026-08-29):** this full report is intentionally preserved as the exact gate snapshot from 2026-08-28. Its staged-shard blocker has subsequently been resolved by tranche2/tranche3 promotion; use `CURRENT_OPERATIONAL.md` and `cu1_evidence_manifest_v1.yaml` for current state. The historical overall `BLOCK` is not itself a current shard-status report.

> **RESULT:** `BLOCK`
> **Reviewed branch:** `design/cu1-history-evidence-timeline-2026-08-28`
> **Reviewed through design head:** `af8370f5c0128ff3859bf5823295b6c9170429c8`
> **Authoritative base:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`
> **PR:** #63 — draft
> **Runtime evidence-aware generation:** NOT AUTHORIZED

---

## 1. Scope reviewed

Exact review covered the current human contract, machine design schema, evidence manifest, active core evidence registry, staged evidence tranches, route-coverage matrix and synthetic semantic fixtures:

```text
CU1_HISTORY_EVIDENCE_TIMELINE_V1.md
cu1_history_timeline_schema_v1.yaml
cu1_evidence_manifest_v1.yaml
cu1_evidence_registry_v1.yaml
cu1_evidence_tranche2_v1.yaml
cu1_evidence_tranche3_v1.yaml
cu1_evidence_coverage_matrix_v1.yaml
cu1_history_evidence_fixtures_v1.yaml
```

Frozen regional/shared profiles and route registry were inspected where needed to test route/subtype applicability.

---

## 2. PASS — object semantics

The following design defects are resolved sufficiently for the current design layer:

```text
PASS ReferralHistoryV2 explicit provenance
PASS missing history != negative history
PASS approximate duration != inferred exact onset date
PASS patient statement != objective examination finding
PASS route-specific dynamic history prompt object
PASS criteria-based rehabilitation semantics
PASS no universal MSK phase sequence
PASS element-level evidence authority
PASS separate EvidenceSource and EvidenceClaim
PASS route/subtype/applicability-scoped RouteEvidenceProfile
PASS separate strength and certainty fields
PASS output scopes: referral_core / therapist_execution_detail / clinician_ui_only
PASS clinician modification distinct from evidence authority
PASS patient-specific protocol/healing restriction distinct from evidence authority
PASS protocol override precedence is machine-represented
PASS evidence freshness/supersession fields
PASS Short referral retains same disease-specific evidence authority
```

---

## 3. PASS — active core output-scope regression

Exact review found two scope-leak candidates in the active core evidence shard:

1. a DGS intervention direction referenced a `clinician_ui_only` insufficient-evidence claim alongside a referral-core claim;
2. a planned Achilles functional-return phase used a `clinician_ui_only` claim as rendered objective authority.

Both were corrected before this gate:

```text
DGS rendered intervention now resolves only to referral_core authority
unsupported Achilles functional-return phase removed rather than rendered from UI-only evidence
```

This preserves the invariant:

```text
clinician_ui_only != automatic referral prose
```

---

## 4. PASS — subtype boundary examples

The design now prevents several clinically material leakage modes:

```text
midportion Achilles authority != insertional Achilles authority
lateral elbow loading authority != Achilles loading authority
GTPS/gluteal-tendinopathy evidence != isolated trochanteric-bursitis authority
anterior shoulder-instability evidence != posterior/multidirectional authority
full-thickness rotator-cuff route != rotator-cuff-tendinopathy route
acute isolated meniscus evidence != degenerative/chronic meniscal route
```

No frozen taxonomy mutation was required for these findings; applicability conditions and route-specific evidence identities are sufficient at the present design layer.

---

## 5. PASS — evidence-gap behavior principle

The design correctly preserves:

```text
insufficient evidence
→ explicit evidence gap
→ no invented progression threshold
→ no generic cross-route fallback
```

A one-phase evidence-bounded sequence may be valid when literature supports a broad rehabilitation direction but not a multi-phase progression. Unsupported later phases are omitted.

This is preferable to false precision.

---

# 6. BLOCKER A — routine-route coverage is materially incomplete

The routine-route gate is not closeable yet.

Large parts of the frozen CU-1 route registry still have no accepted route-specific evidence profile and no reviewed rehabilitation sequence, including substantial portions of:

```text
cervical
shoulder long-tail
elbow long-tail
wrist/hand long-tail
knee long-tail
hip/groin long-tail
ankle/foot long-tail
shared fracture rehabilitation
shared muscle/myotendinous rehabilitation
shared deconditioning/balance/gait rehabilitation
```

Therefore:

```text
route_coverage_gate = FAIL
DESIGN-COMPLETE = forbidden
```

---

# 7. BLOCKER B — staged evidence shards are not yet promoted to normative authority

The evidence corpus is now explicitly sharded for maintainability.

Current states:

```text
cu1_evidence_registry_v1.yaml
→ active design authority

cu1_evidence_tranche2_v1.yaml
→ staged candidate, not schema-frozen

cu1_evidence_tranche3_v1.yaml
→ staged candidate, not schema-frozen
```

The staged shards contain clinically useful current evidence, but they use map-key identity and some objects do not yet carry the complete normalized object shape required for promotion.

Before promotion each shard must pass:

```text
identity materialization
required-field normalization
all-reference resolution
route/subtype/applicability validation
output-scope compatibility
freshness/supersession validation
cross-shard duplicate-ID validation
exact human evidence-scope review
```

Until that occurs, staged evidence may drive the work queue but is not normative referral authority.

---

# 8. BLOCKER C — several routine routes have genuine evidence-precision gaps

This is not merely missing research bookkeeping. Fresh source review confirms that some routine routes do not currently provide the precision required for a full staged physician-facing rehabilitation sequence.

### Deep gluteal / piriformis presentation

Current conservative evidence is low quality and does not establish a superior disease-specific conservative protocol or validated progression thresholds.

Safe state:

```text
broad physiotherapy direction may be considered
specific superior technique = forbidden
invented staged thresholds = forbidden
```

### De Quervain first dorsal compartment disorder

Current comparative evidence is dominated by corticosteroid injection and orthosis/immobilization strategies. Current reviews explicitly identify active interventions such as progressive loading and education as areas needing better study.

Safe state:

```text
medical/conservative evidence context available
validated active progressive PT sequence = not established
```

### Carpal tunnel syndrome

Current AAOS/ASSH evidence is authoritative for overall CTS management, and rehabilitation literature conditionally supports selected conservative approaches, but a route-specific criteria-based PT progression sequence meeting the CU-1 contract is not established by the current high-authority sources reviewed.

### Acute isolated meniscal injury — nonoperative

AAOS 2024 permits PT/rehabilitation in selected non-displaced acute isolated tears largely as consensus/clinical opinion; it does not supply the validated staged progression sequence required by this design.

### Glenohumeral osteoarthritis

AAOS current guidance states that PT may benefit selected patients, but explicitly classifies this as consensus in the absence of reliable evidence. A precise evidence-derived route sequence cannot be fabricated from that statement.

These evidence gaps do not justify generic rehabilitation text.

---

# 9. BLOCKER D — progression semantics remain incomplete in otherwise strong routes

Several routes have strong evidence for the core intervention but not for exact transition criteria.

Examples:

```text
lateral elbow tendinopathy
→ resisted wrist-extensor exercise supported
→ phased high-demand reintroduction supported
→ exact evidence-based transition threshold not supplied

midportion Achilles
→ tendon loading is first-line
→ multiple loading modes supported
→ published return-to-sport criteria remain heterogeneous / poorly operationalized

knee osteoarthritis
→ exercise + self-management strongly supported
→ universal staged transition thresholds not established

plantar heel pain
→ stretching / resistance / selected adjuncts supported
→ mandatory multi-phase progression model not established

patellofemoral pain
→ combined hip+knee exercise strongly supported
→ exact transition to running-specific work not defined
```

The design may safely use evidence-bounded phases and repeated clinical/function measures where a source supports them, but must not invent numeric thresholds simply to make every route look equally staged.

---

# 10. BLOCKER E — route-specific history prompt coverage is incomplete

The prompt object and provenance semantics are frozen, and representative prompts now exist for multiple routes.

However:

```text
route_history_prompt_gate = FAIL
```

because many routine routes still lack their reviewed route-specific history items.

No prompt may be filled by a generic MSK history questionnaire if the route requires materially different history semantics.

---

# 11. BLOCKER F — fixture coverage is not yet route-complete

The synthetic design oracle now covers the major safety invariants requested at handoff, including subtype leakage, protocol precedence, stale evidence, missing history and output-scope behavior.

However it is not yet sufficient to prove the final corpus because:

```text
many routine routes do not yet exist as accepted evidence profiles
staged shards are not promoted
route-specific history prompts are incomplete
additional subtype-boundary cases will emerge during remaining curation
```

The fixture architecture passes; fixture corpus completeness does not.

---

# 12. Current clinical evidence findings that materially changed the design queue

Fresh review added or clarified current authorities for, among others:

```text
2025 rotator-cuff tendinopathy CPG
2025 AAOS rotator-cuff injury CPG for full-thickness tears
2026 traumatic anterior shoulder-instability formal consensus
2025 posterior shoulder-instability consensus
2024 AAOS acute isolated meniscus CPG
2023 nonarthritic hip joint pain CPG
2024/2025 GTPS/gluteal-tendinopathy systematic reviews
2025 insertional Achilles low-compression RCT
2024 AAOS/ASSH carpal tunnel CPG
2025 De Quervain network meta-analysis
```

The evidence queue was updated to preserve population/subtype limits rather than generalize these sources across broader route containers.

---

# 13. Exact gate result

```text
ReferralHistoryV2 semantics                 PASS
RehabilitationSequence object semantics     PASS
GoalPlanV2 / ReassessmentPlanV2 semantics   PASS
EvidenceSource / EvidenceClaim schema        PASS
Element-level evidence provenance            PASS
Protocol override model                      PASS
Active core output-scope audit               PASS
Subtype-boundary architecture                PASS
Evidence freshness architecture              PASS

Routine-route evidence coverage              FAIL
Staged-shard promotion/conformance            FAIL
Evidence-gap route safe-completion policy     INCOMPLETE PER ROUTE
Route-specific history prompt coverage        FAIL
Route-complete fixture corpus                 FAIL

FINAL RESULT                                  BLOCK
DESIGN-COMPLETE                               NO
RUNTIME AUTHORIZED                            NO
```

---

# 14. Exact next authorized work

Continue only on the existing active design writer:

```text
design/cu1-history-evidence-timeline-2026-08-28
```

Next order:

```text
1. normalize and exact-review tranche2, then promote only if every object passes the manifest gate
2. normalize and exact-review tranche3, then promote only if every object passes the manifest gate
3. continue remaining routine routes from cu1_evidence_coverage_matrix_v1.yaml
4. for every evidence-gap route, define the narrowest safe evidence-bounded output and keep unsupported progression empty
5. complete route-specific history prompts alongside each route profile
6. add route/subtype fixtures at the same time as each profile
7. rerun exact design-completeness review
8. STOP again only at DESIGN-COMPLETE or a newly specific BLOCK
```

Explicitly still forbidden:

```text
runtime evidence-recommendation engine
runtime formatter integration
runtime persistence changes
CU-2
PR-1 restart
generic MSK fallback
invented progression thresholds
```
