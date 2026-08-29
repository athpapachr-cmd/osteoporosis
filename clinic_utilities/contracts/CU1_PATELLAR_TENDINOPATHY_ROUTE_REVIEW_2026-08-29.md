# CU-1 Patellar Tendinopathy Route Coverage Review — 2026-08-29

> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`  
> **Runtime mutation:** not authorized and not performed.  
> **Route shard:** `clinic_utilities/contracts/cu1_evidence_route_coverage_patellar_tendinopathy_v1.yaml`  
> **Review mode:** exact source / applicability / output-scope / loading-mode / progression review.

## Decision

```text
canonical route identity                         PASS
frozen knee-profile consistency                 PASS
current exercise evidence review                PASS
absolute-effect uncertainty preserved           PASS
loading-mode hierarchy not fabricated           PASS
PTLE RCT population limitation preserved        PASS
ESWT non-default posture preserved              PASS
source / claim / profile references             PASS
explicit payload IDs                            PASS
required profile / sequence fields              PASS
output-scope compatibility                      PASS
therapist execution autonomy                    PASS
no universal numeric progression threshold      PASS
no unsupported RTS threshold                    PASS
route-specific history prompts                  PASS

PATELLAR TENDINOPATHY PROFILE                   PASS
REHABILITATION SEQUENCE                         COMPLETE — SINGLE-PHASE EVIDENCE-BOUNDED
RUNTIME AUTHORIZED                              NO
CU-1 DESIGN-COMPLETE                            NO
```

## Frozen route

Canonical route:

```text
patellar_tendinopathy
```

The frozen knee profile preserves:

```text
anterior knee pain + tendon tenderness != automatic patellar tendinopathy
imaging tendon change != automatically symptomatic tendinopathy
progressive tendon loading = rehabilitation direction
no mandatory eccentric / isometric / HSR protocol
ESWT != generator default
```

This evidence review does not alter those taxonomy or diagnosis-vs-finding rules.

## Current sources reviewed

1. Lopes AD, Rizzo RRN, Hespanhol L, Costa LOP, Kamper SJ. **Exercise for patellar tendinopathy.** Cochrane Database Syst Rev. 2025;5:CD013078. DOI `10.1002/14651858.CD013078.pub2`.
2. Liu Y, Li C, Yang F. **Comparative effectiveness of exercise interventions for patellar tendinopathy: a systematic review and network meta-analysis of randomized controlled trials.** BMC Sports Sci Med Rehabil. 2026;18:296. DOI `10.1186/s13102-026-01743-4`.
3. Breda SJ et al. **Effectiveness of progressive tendon-loading exercise therapy in patients with patellar tendinopathy: a randomised clinical trial.** Br J Sports Med. 2021;55(9):501-509. DOI `10.1136/bjsports-2020-103403`.
4. Challoumas D et al. **Management of patellar tendinopathy: a systematic review and network meta-analysis of randomised studies.** BMJ Open Sport Exerc Med. 2021;7(4):e001110. DOI `10.1136/bmjsem-2021-001110`.

## Evidence reconciliation

### 1. Absolute effect of strengthening exercise

The 2025 Cochrane review found only seven randomized trials, all in athletes, and concluded that evidence for strengthening exercise is low to very low certainty depending on outcome/comparator. It could not draw firm conclusions about pain/function versus no treatment and several alternatives.

CU-1 therefore must not render:

```text
exercise is proven to work for every patellar-tendinopathy patient
```

The Cochrane uncertainty is preserved in a clinician-facing claim rather than hidden.

### 2. Relative choice among loading strategies

The 2026 network meta-analysis searched through December 2025. In the primary network, no exercise intervention demonstrated statistical superiority over HSR, and observed differences did not establish a clinically meaningful treatment hierarchy. HSR was a reasonable reference and several progressive loading strategies appeared broadly comparable within current evidence limits.

Therefore:

```text
eccentric != mandatory universal protocol
HSR != mandatory universal protocol
isometric != mandatory universal protocol
PTLE != mandatory universal protocol
probability ranking != clinical superiority
```

Loading-mode selection/dosing remains therapist-level execution detail.

### 3. PTLE randomized-trial signal

The Breda RCT enrolled 76 mostly chronic, young athletes; median symptom duration was two years and 82% had prior treatment without full recovery. PTLE improved VISA-P more than eccentric-only exercise at 24 weeks, with adjusted between-group difference 9 points (95% CI 1–16). Return-to-sport was numerically higher with PTLE but the between-group difference was not statistically significant.

This supports progressive tendon loading as a reasonable conservative direction in an applicable population, but the later 2026 synthesis prevents extrapolation into a universal PTLE-superiority rule.

### 4. ESWT

The 2021 intervention NMA found that ESWT added to an eccentric-exercise background did not show clear short-term superiority over sham ESWT with the same exercise background for pain or function.

Therefore ESWT remains:

```text
clinician_ui_only evidence context
not automatic referral adjunct
not substitute for the loading/capacity rehabilitation direction
```

## Rehabilitation-sequence decision

A complete route can be represented safely as one evidence-bounded phase:

```text
individualized progressive tendon/quadriceps loading
-> monitor symptom/load tolerance + strength + function
-> progress by clinical response, not elapsed time alone
-> therapist selects/doses loading mode
```

No exact repetition scheme, frequency, percentage load, pain threshold or mandatory phase duration is physician-authored by CU-1.

This one-phase model is intentional. The current evidence does not justify adding a false multi-phase sequence merely because older expert frameworks contain staged numeric protocols.

## Return-to-sport boundary

The reviewed current evidence does not establish validated universal patellar-tendinopathy return-to-sport thresholds.

The Breda trial included return to sport as an outcome, but it does not validate one universal clearance threshold. Cochrane also found limited return-to-sport comparative evidence.

Therefore:

```text
sport / jumping / running goal may be captured in history
specific RTS progression remains therapist/clinician individualized
no automatic numeric RTS clearance criterion
no elapsed-time-only clearance
```

A later source may add a route-specific RTS criterion only after an exact evidence review.

## History prompts

The route adds non-inferential prompts for:

```text
recent jumping/running/gym load change
symptom location + load relationship
prior loading programme + response
US/MRI context without imaging-diagnosis equivalence
work/sport/jumping/running goal
```

## Exact route state

```text
rep_patellar_tendinopathy_v1
-> PASS

seq_patellar_tendinopathy_v1
-> COMPLETE — SINGLE-PHASE EVIDENCE-BOUNDED

loading mode
-> therapist_execution_detail

ESWT
-> NOT AUTO-RECOMMENDED

universal numeric progression / RTS threshold
-> NOT AUTHORIZED
```

The next queue item after matching fixtures and manifest/matrix reconciliation is `thumb_cmc1_osteoarthritis`.
