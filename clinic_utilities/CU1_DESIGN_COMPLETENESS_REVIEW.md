# CU-1 Design Completeness Review — 2026-08-27

> **STATUS:** REVIEW COMPLETE — **BLOCK** for runtime implementation authorization.
> **Base reviewed:** `4b12932cb007994ca5d998f47719ff706191d2e9` (`main`).
> **Scope:** all frozen CU-1 regional/shared v1.1 profiles + core `ReferralDraft`/formatter architecture + current runtime/schema seams.
> **Runtime changes:** none.

---

# 1. Executive conclusion

The frozen clinical/content set is clinically coherent enough to preserve. No broad re-opening of the regional taxonomies is indicated.

However CU-1 is **not yet design-complete for implementation**. The profiles currently define substantial structured semantics in prose that the frozen core contract does not yet represent deterministically. Implementing now would force runtime code to invent schema, routing, warning severity, precedence and formatter behavior.

Required classification:

```text
CLINICAL/CONTENT DESIGN = substantially complete / preserve frozen profiles
IMPLEMENTATION-CONTRACT DESIGN = BLOCK
RUNTIME AUTHORIZATION = NOT READY
```

---

# 2. What passed

## 2.1 Clinical taxonomy / diagnosis-vs-finding separation — PASS

Across profiles, the design consistently preserves:

```text
symptom != diagnosis
provocation/special test != diagnosis
imaging finding != automatically symptomatic diagnosis
not_assessed != normal
clinician-entered diagnosis may be carried but not inferred
adjunct != core rehabilitation
```

Neurological symptoms remain distinct from objective deficit in cervical/lumbar/elbow/wrist-hand contexts. Structural post-traumatic and postoperative restrictions are generally respected.

## 2.2 Shared-profile ownership concept — PASS IN PRINCIPLE

The intended ownership is sound:

```text
Shared Fracture
→ healing / stability / immobilization / WB-use / ROM-loading restrictions

Shared Muscle/Myotendinous
→ acute muscle/myotendinous injury / load progression / RTS-work semantics

Shared Deconditioning/Balance/Gait
→ generalized functional decline / established frailty / falls/function / walking-aid semantics
```

Shared Fracture correctly retains authority over fracture restrictions when functional rehabilitation is also selected.

## 2.3 Safety philosophy — PASS IN PRINCIPLE

The profiles consistently avoid reassurance from missing data and include clinician disposition for material concerns. Structural, neurological, vascular, infection, fracture, tendon rupture, DVT and postoperative concerns are generally separated from routine referral wording.

## 2.4 Persistence/privacy boundary — PASS FOR FIRST IMPLEMENTATION DIRECTION

The frozen direction remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

No patient persistence is required for the first CU-1 implementation. This is compatible with the current public-repository/privacy posture, provided real patient data is not committed/logged/persisted.

---

# 3. Blocking findings

## B1 — `ReferralDraft` is too flat for the frozen profile semantics

Current frozen core:

```text
ReferralDraft
  patient_context
  body_region
  primary_problem
  secondary_problems[]
  laterality
  chronicity
  key_findings[]
  functional_impairments[]
  precautions[]
  explicit_restrictions[]
  goals[]
  rehab_directions[]
  adjunct_options[]
  reassessment_criteria[]
  sessions_optional
  clinician_free_text_optional
```

The frozen profiles require typed structured state that has no explicit home in this contract, including:

```text
profile/route identity
formal diagnosis assertion vs presentation wording
pathway subtype
operative vs nonoperative state
procedure/protocol
healing/stability
weight-bearing / upper-limb-use state
ROM/loading restrictions
neurological tri-states
safety_screen_status
clinician_disposition
fracture context / injury phase
shared-gateway payload
functional measurement values/units
```

A list of free-form findings/restrictions cannot safely substitute for these typed states because deterministic warnings and formatter wording depend on their exact values.

### Required design resolution

Freeze a nested typed selection model before implementation, for example conceptually:

```text
ReferralDraft
  version
  patient_context
  primary_problem: ProblemSelection
  secondary_problems[]: ProblemSelection
  findings[]: FindingSelection
  functional_impairments[]
  precautions[]
  explicit_restrictions[]
  goals[]
  rehab_directions[]
  adjunct_options[]
  safety: SafetyState
  sessions_optional
  clinician_free_text_optional

ProblemSelection
  profile_id
  route_id
  assertion_state / diagnostic_wording_mode
  subtype_optional
  laterality_optional
  context{}
  restrictions{}
  shared_gateway_optional
```

Exact field names are still a design decision; implementation must not invent them.

---

## B2 — No frozen machine-readable profile/route/key registry or exact gateway map

`clinic_utilities/` currently contains profile Markdown only. There is no canonical registry/schema that enumerates:

```text
profile_id
route_id
subtype ids
finding ids
goal ids
rehab-direction ids
adjunct ids
shared gateway target
regional → shared key mapping
visibility / routine-vs-rare metadata
```

The Markdown files contain real key drift and cross-profile mapping differences. Examples include mixed acronym casing and non-identical gateway values such as:

```text
Hip gateway: ASIS_avulsion
Shared Fracture: ASIS_apophyseal_avulsion

Hip gateway: AIIS_avulsion
Shared Fracture: AIIS_apophyseal_avulsion

Hip proximal-rectus gateway key
!= Shared Muscle route/subtype key
```

Additional key-style inconsistency includes uppercase acronyms embedded in machine identifiers (`ACL_*`, `MCL_*`, `SPPB_*`, `formal_GTPS_*`, `formal_CRPS_*`, `ASIS_*`, `MRI_*`, etc.) alongside lowercase snake_case identifiers.

### Required design resolution

Freeze one canonical machine registry (YAML/JSON/Python data contract is an implementation choice, but the semantic registry must be frozen before coding) and define exact aliases/migrations for existing prose keys.

All regional→shared gateways must map to one exact canonical route/site/subtype id.

---

## B3 — Selected postoperative/structural routes have unresolved owner ambiguity

Knee correctly established exclusive routing:

```text
ACL nonoperative/prehab → K7
ACL reconstruction → K13 postoperative
MCL nonoperative → K8
MCL repair/reconstruction → K13 postoperative
```

Equivalent exclusivity is not consistently frozen elsewhere.

Examples in Wrist/Hand:

```text
WH9 digital_tendon_injury_rehabilitation
  includes extensor_tendon_repair_postoperative / flexor_tendon_repair_postoperative

WH11 postoperative_wrist_hand_rehabilitation
  examples include tendon repair
```

Likewise:

```text
WH7 thumb MCP collateral ligament injury
  allows operative/nonoperative context

WH11 postoperative route
  includes thumb collateral-ligament repair/reconstruction
```

and sagittal-band/extensor stabilization can similarly be represented in both a structural pathway and WH11.

Fracture fixation can also appear as a regional postoperative example while Shared Fracture is intended to own fracture healing/restriction semantics.

### Required design resolution

Freeze a single route-precedence/ownership table. At minimum define for each overlapping scenario whether:

```text
A) operative case routes exclusively to regional postoperative route with structural subtype
or
B) structural route remains primary and postoperative state is context
or
C) shared structural profile is primary and regional route is navigation only
```

Do not leave this to UI/runtime branching.

---

## B4 — Warning/safety severity and blocking behavior are not centrally defined

Profiles currently use multiple prose levels:

```text
soft warning
warning
high-priority prompt
reassessment prompt
routine physiotherapy deferred
no routine reassuring wording
urgent/same-day pathway
```

There is no frozen shared type describing:

```text
severity
blocking vs non-blocking
whether referral generation is prevented
whether warning is clinician-only UI or appears in generated text
required clinician disposition before continuing
precedence when multiple concerns coexist
```

This is safety-critical because different implementers could legitimately interpret the same profile text differently.

### Required design resolution

Freeze a common safety/consistency result contract, for example conceptually:

```text
INFO
SOFT_WARNING
HARD_WARNING_REQUIRES_ACKNOWLEDGEMENT
BLOCK_UNTIL_DISPOSITION
URGENT_REASSESSMENT
```

and define whether each state:

```text
blocks formatter
requires clinician disposition
is omitted from patient/referral text
is shown only in clinician UI
```

Exact enum names remain to be frozen.

---

## B5 — Formatter interface/output contract is not frozen

`ShortReferralFormatter` and `DetailedReferralFormatter` are named repeatedly, but there is no authoritative formatter contract or runtime file.

Unresolved design questions include:

```text
exact formatter input type
section/order rules
primary vs secondary problem rendering
presentation wording vs formal diagnosis wording
which normal findings may be stated
omission of not_assessed / not_stated
how restrictions are ordered and emphasized
how shared-profile context is merged with regional wording
where optional adjuncts appear
whether evidence caveats enter referral text or clinician UI only
how safety warnings interact with formatting
short-format maximum/target density
Greek labels vs machine ids
free-text escaping/normalization
empty-section behavior
```

### Required design resolution

Freeze one formatter specification with deterministic examples for at least:

```text
simple regional case
formal-diagnosis assertion case
regional + shared fracture case
postoperative/protocol case
neurological tri-state case
material safety concern case
multiple secondary problems
short vs detailed output from identical draft
```

---

## B6 — Tri-state/enumeration semantics are conceptually sound but not normalized/versioned

The design repeatedly and correctly distinguishes missing assessment from normal findings, but uses multiple local value shapes:

```text
yes / no / not_stated
normal / abnormal / not_assessed
present / absent / not_assessed
none_reported / single_fall / recurrent_falls / not_assessed
stable / unstable / not_assessed
not_stated as one subtype state in some diagnosis assertions
```

Some `formal_*` entries are booleans/tri-state, while others are direct clinician-selected subtype ids.

This is manageable only after an explicit type registry exists.

### Required design resolution

Freeze common semantic types such as:

```text
AssertionState
AssessmentState
SafetyScreenState
ClinicianDisposition
Laterality
Visibility
```

and explicitly document where a route-specific enum is required instead of a generic tri-state.

---

# 4. Important non-blocking findings

## N1 — Stale `future shared profile` wording

Some regional frozen documents still describe now-existing shared profiles as `future shared ...`. This is documentation drift, not a clinical blocker, but should be reconciled during the design-hardening pass without changing the frozen clinical decisions.

## N2 — Evidence-sensitive adjunct metadata needs centralized rendering ownership

Clinical decisions about acupuncture, dry needling and ESWT are intentionally different by region. That is acceptable. The implementation should not infer one global rule; the central registry should carry per-route adjunct visibility plus evidence-label/rendering metadata where required.

## N3 — Current runtime architecture can host CU-1 cleanly

Current production architecture is FastAPI/uvicorn with a thin `main.py` composing routers, and Pydantic is already available. `clinic_utilities/` has no runtime today. Therefore there is no legacy CU-1 runtime that must be preserved, and a bounded new router/static utility seam is technically feasible after design completion.

This is not authorization to implement.

---

# 5. Recommended bounded design-hardening scope

Do **not** reopen broad clinical taxonomy.

One docs/schema-only pass should resolve B1–B6 by producing:

```text
1. CU-1 core typed contract v1
2. canonical profile/route/key registry v1
3. exact regional→shared gateway mapping table
4. route ownership + precedence table
5. common safety/warning/disposition contract
6. ShortReferralFormatter / DetailedReferralFormatter specification
7. normalized common enum/tri-state definitions
8. focused cross-profile fixture matrix (design fixtures, no patient data)
```

Suggested fixture matrix should include at least:

```text
cervical radicular symptoms with incomplete neuro exam
lumbar cauda-equina concern
shoulder postoperative restriction
wrist flexor/extensor tendon repair
knee ACL reconstruction
hip proximal rectus → Shared Muscle
hip ASIS/AIIS avulsion → Shared Fracture
ankle fracture → Shared Fracture
SIFK with unknown loading status
fragility fracture → Deconditioning/Falls gateway
calf strain with DVT concern
frailty route with SPPB data but no autonomous diagnosis
```

---

# 6. Exit criteria for repeat review

CU-1 may be classified `DESIGN-COMPLETE` only when all are true:

```text
[ ] every frozen route has one canonical machine id
[ ] every regional→shared gateway resolves to one exact canonical target
[ ] no operative/structural scenario has two unresolved primary owners
[ ] every profile-specific structured state has a defined typed home
[ ] common tri-state/enums are normalized and versioned
[ ] safety warnings have deterministic severity/blocking/disposition behavior
[ ] formatter input/output/omission/precedence rules are frozen
[ ] short/detailed formatter design fixtures produce unambiguous expected text structure
[ ] no runtime implementation is required to answer any remaining semantic question
```

---

# 7. Final review classification

```text
BLOCK
```

Reason:

> The clinical content is substantially complete, but the cross-profile machine contract is not yet sufficiently frozen for safe deterministic implementation.

Next action is one bounded design-hardening pass. Runtime implementation remains unauthorized.