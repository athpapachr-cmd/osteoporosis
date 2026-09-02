# TODO.md — Clinical Excellence long-range compass

> **ROLE:** permanent broad roadmap/checklist across phases.
> **CANONICAL HOME:** `athpapachr-cmd/osteoporosis`.
> **ACTIVE DETAILED PHASE:** `CLINICAL_EXCELLENCE_PLAN.md`.
> **ACTIVE SLICE DESIGN:** `SLICE_PLAN_CURRENT.md`.
> **OPERATIONAL NOW:** `CURRENT_OPERATIONAL.md`.
> **MODULE 01:** Osteoporosis.

This file answers **where the product is going and in what broad order**. It is not the operational lock and should not become a PR/deploy diary.

---

# 0. CLOSED FOUNDATIONS / CURRENT PRODUCTION BASE

- [x] Reframe project from isolated Osteoporosis Cockpit to reusable Personal Clinical Excellence System.
- [x] Establish Osteoporosis as Module 01.
- [x] Establish Signal-first feedback architecture.
- [x] Establish gap classes: knowledge / reasoning / execution / communication-system.
- [x] Establish sustained strengths as positive Signals.
- [x] Establish STANDARD / CHALLENGE / RED TEAM / LEARNING modes.
- [x] Define transparent measurement and no-composite-score-before-baseline principles.
- [x] Define Baseline Osteoporosis Audit draft v1 + KPI Dictionary v1.
- [x] Implement Baseline Audit Steps 1–6 and pre-pilot hardening.
- [x] Pass 14-scenario synthetic form smoke.
- [x] Implement authenticated patient registry + PostgreSQL encounter/lab persistence.
- [x] Verify patient load/save/reload and longitudinal laboratory snapshots.
- [x] Implement Clinical Calendar foundation and keep live Secretary integration independently deferrable.
- [x] Merge server-side completed/amended finalization integrity semantics.
- [x] Complete prior live 3/3 server-finalization smoke.
- [x] Upgrade to six active canonicals.
- [x] Implement/test bounded authoritative browser Finish correction on `fix/module01-c1-authoritative-finish-2026-08-30`.
- [x] Reframe the clinician-facing product from manual Baseline Audit form to dynamic Clinical Excellence consultation system.
- [x] Freeze G-0 dynamic-guidance architecture and revised system-assisted baseline methodology.
- [x] Implement/test the bounded G-1 progressive-guidance runtime foundation on `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
- [x] Release C1 authoritative Finish + G-1 progressive guidance to production through PR #64.
- [x] Correct and deploy explicit `Γιατί τώρα:` discoverability in the top `Σημερινή ροή` summary through PR #66.
- [x] Complete product-owner production re-smoke confirming visible `Γιατί τώρα:` and dynamic guidance behavior.
- [x] Release G-2 evidence-backed osteoporosis guidance through PR #69.
- [x] Verify Render auto-deploy `dep-daaph5vlk1mc73940g60` is live at exact G-2 merge SHA `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
- [x] Complete authenticated product-owner G-2 production smoke.
- [x] Release G-3 guidance salience + longitudinal patient summary through PR #70.
- [x] Correct the bounded G-3 production-visibility/salience defects through PR #71.
- [x] Verify Render hotfix auto-deploy `dep-dabolap5efls739s9am0` live at exact `ab94c6286bdc49cb8304b072e557c5eb0a96b0c6`.
- [x] Complete product-owner G-3 production re-smoke confirming visible `Νέο` and `Σύνοψη ασθενούς`.

C1, G-1, G-2 and G-3 are now **implemented / tested / merged / deployed / production-smoke-verified**. Real-clinic usefulness/refinement remains distinct from production smoke and is not `PILOT-VALIDATED`.

---

# 1. CURRENT PRIMARY PROGRAM — OSTEOPOROSIS MODULE 01 CLOSURE

Primary product objective:

```text
improve today's osteoporosis encounter
+
reduce duplicate/manual capture
+
review whether what was said/reasoned/decided was appropriate
+
improve clinician performance longitudinally
```

Audit remains an underlying measurement engine, not the main clinician-facing workflow.

## 1.1 Critical finalization integrity

- [x] Authoritative Finish correction designed.
- [x] Authoritative Finish correction implemented.
- [x] Focused browser/API regression passed.
- [x] Fresh-bootstrap/review and merge exact tested ancestry after explicit release authority.
- [x] Allow normal Render auto-deploy from `main`.
- [x] Authenticated production smoke reported working by product owner; no current C1 Finish defect reported.

No real pilot before the full intended guided/capture workflow is ready.

## 1.2 G-0 Dynamic Guidance design — COMPLETE

- [x] Separate Clinical Guidance / Transcript Capture / Audit / Practice Review.
- [x] Freeze `EncounterContextV1`.
- [x] Freeze `LongitudinalGuidanceProjectionV1` + conflict semantics.
- [x] Freeze `GuidanceRuleV1`.
- [x] Freeze `VisitPlanV1`.
- [x] Freeze `GuidedCardStateV1`.
- [x] Freeze `TherapyMilestoneProfileV1` capability.
- [x] Freeze `GuidanceExposureV1`.
- [x] Freeze rule priority: safety/event → unresolved prior → agent-specific → milestone/due → archetype → contextual.
- [x] Preserve current coarse archetypes as visit intent rather than creating a form per dose number.
- [x] Verify existing protected historical encounter payloads can feed a read-only longitudinal projection without immediate DB migration.
- [x] Freeze machine contract manifest: `schemas/dynamic_guided_visit_contract_manifest_v1.yaml`.
- [x] Complete exact G-0 design review: `M01_G0_DYNAMIC_GUIDANCE_DESIGN_REVIEW_V1.md` = `DESIGN-COMPLETE`.

## 1.3 G-1 Dynamic Guidance runtime foundation — PRODUCTION-SMOKE-VERIFIED / CLOSED

- [x] Build ephemeral `LongitudinalGuidanceProjectionV1` from protected historical encounters.
- [x] Build `EncounterContextV1` resolver.
- [x] Implement deterministic minimum guidance evaluation and priority resolution.
- [x] Produce `VisitPlanV1`-compatible card guidance states for the current UI.
- [x] Render `why now` reason(s).
- [x] Prove new-event override of routine flow.
- [x] Prove unresolved-prior item resurfacing.
- [x] Prove explicit due/overdue/treatment timeline plumbing without invented cadence.
- [x] Expand prior authoritative data reuse into richer read-only longitudinal summary after real-use evidence demonstrated the need; released and production-smoke-verified in G-3.
- [x] Keep G-1 generic: no medication-specific milestone content invented.
- [x] Preserve coarse applicability ownership and authoritative Finish regressions.
- [x] Merge/deploy through reviewed release path.
- [x] Production smoke confirmed G-1 loads and existing `Τύπος σημερινής επίσκεψης` is usable.
- [x] Correct the production-smoke WHY-NOW discoverability defect without changing guidance semantics.
- [x] Product-owner re-smoke confirms literal `Γιατί τώρα: ...` is visible in top `Σημερινή ροή` after the correction deploy.
- [x] Product-owner observed that surfaced guidance changes dynamically with visit context and is informative/guiding; this remains a production observation, not pilot validation.

## 1.4 Osteoporosis guidance-content profiles — PRODUCTION-SMOKE-VERIFIED / CLOSED

- [x] Review evidence/approved clinic-policy boundary for clinically active dynamic rules.
- [x] Define first-assessment guidance profile.
- [x] Define known-patient initial-to-service profile.
- [x] Evaluate results/work-up-review-with-management-decision visit intent; retain as product-flow candidate, not activated runtime enum yet.
- [x] Define routine stable follow-up profile.
- [x] Define treatment-start profile.
- [x] Define repeated-administration/continuation profile.
- [x] Define fracture/post-fracture and fracture-on-treatment event overrides.
- [x] Define transition/exit/consolidation profile.
- [x] Define adverse-effect/intolerance profile.
- [x] Define evidence-backed denosumab/time-critical therapy timing and milestone rules.
- [x] Keep administration count and elapsed exposure separate.
- [x] Explicitly forbid generic “4th/8th/10th Prolia” rules without reviewed rationale/provenance.
- [x] Freeze machine-readable evidence/rules/profiles/milestones contract and human design review.
- [x] Pass G-2 machine-contract validation on the final contract ancestry.
- [x] Implement the reviewed G-2 runtime activation boundary over G-1.
- [x] Pass focused G-2 runtime regressions plus inherited G-1/C1 regressions.
- [x] Complete separate product-owner release review and PR #69 release path.
- [x] Squash merge to `main` as `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
- [x] Verify exact Render auto-deploy live.
- [x] Complete authenticated production smoke.

Important first-runtime exclusions remain:

- `OST_G2_R15_DENOSUMAB_EXIT_CTX_FOLLOWUP` remains blocked until reliable denosumab-exit → specific zoledronate-event linkage exists;
- `OST_G2_R16_DENOSUMAB_EXIT_NO_CTX_OPTION` remains blocked until CTX-monitoring availability is explicitly represented;
- checklist-only medication safety guidance is visibly identified as verification/checklist content and not automated safety clearance.

## 1.5 G-3 Guidance salience + longitudinal patient summary — PRODUCTION-SMOKE-VERIFIED / CLOSED

Triggered by product-owner production interaction after G-2 smoke.

- [x] Define newly-surfaced Visit Plan salience semantics.
- [x] Ensure initial plan is baseline and does not mark every card `Νέο`.
- [x] Add explicit textual `Νέο` plus stronger visual emphasis; color is not the sole signal.
- [x] Regression-test the exact `<4 cm → >=4 cm` height-loss transition surfacing VFA guidance as new.
- [x] Add deterministic/read-only `Σύνοψη ασθενούς` above `Σημερινή ροή`.
- [x] Summarize first→latest completed/amended course, fractures/risk, DXA, treatment/actual administrations, labs, latest explicit management decision and unresolved/conflicts.
- [x] Keep current visit distinct from completed historical truth.
- [x] Preserve `missing != negative`, `scheduled != actual`, conflict fail-closed and later-blank-does-not-erase-prior semantics.
- [x] Reuse existing protected encounter/lab endpoints and existing G-1 treatment/admin projection; no DB migration/new write path.
- [x] Pass focused G-3 tests plus inherited G-2/G-1/C1 regression gate.
- [x] Release through PR #70; Render auto-deploy reached live at exact merge SHA `ef17367c7b8959f51e05b80909226804951d1bc7`.
- [x] Correct production visibility/cache and same-card/new-evidence salience defects through PR #71.
- [x] Render auto-deploy `dep-dabolap5efls739s9am0` reached live at exact hotfix merge SHA `ab94c6286bdc49cb8304b072e557c5eb0a96b0c6`.
- [x] Product-owner production re-smoke confirmed both `Νέο` and `Σύνοψη ασθενούς` visible and working well.

G-3 is production-smoke-verified, not pilot-validated. Subsequent evidence-from-use UX refinements are handled as bounded slices rather than reopening G-3 clinical semantics.

## 1.6 G-4 Workspace ergonomics + RF utility navigation — IMPLEMENTED / TESTED; RELEASE REVIEW ACTIVE

Triggered by product-owner evidence from use after successful G-3 re-smoke.

- [x] Add accessible independent collapse/expand controls to `Σύνοψη ασθενούς` and `Σημερινή ροή`.
- [x] Make `Σύνοψη ασθενούς` sticky at the top of the encounter scroll context.
- [x] Keep collapse state as UI-only browser preference, not patient/clinical data.
- [x] Preserve the existing G-3 summary/guidance renderer as the single clinical owner.
- [x] Add `Clinic Utilities` navigation with existing physiotherapy referral and `Ραδιοκύματα — PDF` entry.
- [x] Keep the existing RF generator as the PDF/template/request source of truth; do not duplicate RF persistence/templates into osteoporosis encounter state.
- [x] Pass G-4 focused regression plus inherited G-3/G-2/G-1/C1 gate at exact tested runtime head `942d4e06944ebd6de97891cb8e2739c88ba85a38`, run `33599860151`.
- [x] Exact-head code/scope review passed; only canonical closeout commits followed the tested runtime head.
- [ ] Complete canonical reconciliation and final exact-head release-readiness check.
- [ ] Open release PR only after release review PASS.
- [ ] Merge/deploy only with separate explicit product-owner authority.
- [ ] Production smoke after release must include collapse/expand, sticky summary and RF navigation target.

## 1.7 Heidi-first capture — BEFORE REAL PILOT

- [ ] Restart corrected archived PR-1 v3 design as bounded implementation slice.
- [ ] Protected transcript paste/intake.
- [ ] Raw transcript ephemeral by default.
- [ ] PHI-safe request validation/logging.
- [ ] Reusable Core semantic candidates.
- [ ] Deterministic osteoporosis target mapping against actual persisted runtime paths.
- [ ] Preserve negation, temporality, speaker/source and uncertainty.
- [ ] Preserve option discussed / recommendation / preference / final decision distinctions.
- [ ] No exact-date invention from vague timing.
- [ ] No authoritative PR-1 write.

## 1.8 PR-2 Inline provisional population — BEFORE REAL PILOT

- [ ] Show mapped candidates inside destination clinical cards.
- [ ] `proposed` values are visually populated but non-authoritative.
- [ ] Accept / Reject / Edit.
- [ ] Explicit conflict handling with authoritative longitudinal data.
- [ ] No silent overwrite.
- [ ] Persist provenance after clinician acceptance.
- [ ] Keep clinically meaningful unmapped candidates visible for review.
- [ ] Show compact applicable-visit extraction coverage without treating unmentioned as negative.

## 1.9 Five-case real system-assisted pilot

Only after:

```text
G-1 production-readiness gate closed
+ G-2 evidence-backed minimum osteoporosis guidance released/smoked
+ G-3 guided UX refinements released/smoked
+ G-4 workspace ergonomics released/smoked
+ PR-1 extraction
+ PR-2 inline review/population
```

- [ ] Run 5 consecutive eligible real system-assisted osteoporosis encounters.
- [ ] Measure completion time.
- [ ] Measure manual entry/correction burden.
- [ ] Track clinically meaningful transcript omissions.
- [ ] Track false/incorrect candidates.
- [ ] Track ambiguous/conflicting candidates.
- [ ] Track wrong/missing card relevance.
- [ ] Track duplicate questioning/data entry.
- [ ] Track persistence/finalization defects.
- [ ] Track cognitive burden and safety/data-integrity issues.
- [ ] Do not redesign after each case unless safety/data-loss/persistence requires it.
- [ ] After all five, make one deliberate refinement.
- [ ] Freeze Guidance/Capture/KPI applicability/denominator contracts.

## 1.10 Quick Practice Review shadow capability

- [ ] `PracticeReviewV1`.
- [ ] `PracticeObservationV1` with direction, importance, confidence, provenance, evidence, suggested change and clinician disposition.
- [ ] Review clinical completeness relative to the actual Visit Plan, not a universal checklist.
- [ ] Review reasoning sequence.
- [ ] Review decision quality.
- [ ] Review risk interpretation.
- [ ] Review safety.
- [ ] Review communication accuracy/clarity.
- [ ] Review shared decision making.
- [ ] Review consultation flow/efficiency.
- [ ] Review follow-up execution.
- [ ] Keep routine clinician-facing Practice Review hidden during scored baseline by default.

## 1.11 Thirty-case scored system-assisted baseline

- [ ] Run 30 consecutive unique eligible osteoporosis encounters under frozen guidance/capture contracts.
- [ ] Clinical Guidance remains active.
- [ ] Transcript-assisted capture remains active.
- [ ] Routine KPI/performance feedback remains hidden.
- [ ] Routine clinician-facing Practice Review remains hidden by default.
- [ ] Safety-critical feedback remains allowed.
- [ ] Record guidance exposure where reliable.
- [ ] Distinguish pre-cue vs post-cue correct behavior where event sequence is technically trustworthy.
- [ ] Label cohort accurately as `system-assisted baseline`.
- [ ] Lock denominators/baseline and run-chart/reliability conventions.

## 1.12 Close one real improvement loop

- [ ] Clinician dispositions on important PracticeObservations.
- [ ] Aggregate repeated observations longitudinally.
- [ ] Promote denominator-aware Signals.
- [ ] Classify negative Signals: knowledge / reasoning / execution / communication-system.
- [ ] Apply root-cause-appropriate intervention.
- [ ] Re-measure later encounters.
- [ ] Record improved / unchanged / worsened / insufficient evidence.
- [ ] Where feasible, assess whether correct behavior becomes less prompt-dependent over time without overstating causality.

## 1.13 Final Module 01 closure

- [ ] No unresolved critical safety/data-integrity defect.
- [ ] Dynamic visit flow validated in real use.
- [ ] Transcript capture reduces duplicate manual entry safely.
- [ ] Practice Review produces evidence-traceable clinician-governed observations.
- [ ] Baseline locked or explicit methodology revision approved.
- [ ] At least one real improvement loop re-measured.
- [ ] Reusable Core vs osteoporosis-specific content distinguishable.
- [ ] Six canonicals reconstruct project truth without chat history.
- [ ] Only then mark `MODULE 01 CLOSED`.

---

# 2. CLINICAL GUIDANCE / DYNAMIC VISIT ENGINE

Permanent requirements:

- [x] Architecture uses archetype + longitudinal triggers, not one universal checklist.
- [x] Architecture can represent safety/event overrides.
- [x] Architecture can represent due/milestone rules.
- [x] Architecture can represent unresolved prior items.
- [x] Architecture can explain `WHY NOW?`.
- [x] Architecture avoids a form per ordinal treatment visit.
- [x] Minimum G-1 runtime foundation implemented/tested/merged/deployed.
- [x] Regression suite for representative G-1 encounter mechanics.
- [x] Explicit WHY-NOW summary presentation regression.
- [x] Final product-owner correction re-smoke.
- [x] Evidence-backed osteoporosis guidance-content registry and activation contract designed/reviewed.
- [x] G-2 evidence-backed guidance runtime implemented/tested/merged/deployed/production-smoke-verified.
- [x] G-3 newly-surfaced guidance salience implemented/tested/merged/deployed/production-smoke-verified.
- [x] G-3 deterministic always-visible longitudinal patient summary implemented/tested/merged/deployed/production-smoke-verified.
- [x] G-4 collapsible/sticky top-workspace ergonomics implemented/tested.
- [ ] G-4 release PR / merge / deploy / production smoke.
- [ ] Real-clinic usability validation and evidence-from-use card/taxonomy refinement.

---

# 3. CLINICAL PRACTICE REVIEW / LEARNING

- [ ] Quick Review sufficient for Module 01 closure.
- [ ] Evidence/provenance on material claims.
- [ ] Accept / Modify / Dismiss.
- [ ] Longitudinal recurrence/reliability logic.
- [ ] Sustained-strength detection.
- [ ] Signal → intervention → re-measurement.

Later/non-blocking by default:

- [ ] Full Deep Review.
- [ ] Full RED TEAM productization.
- [ ] Exhaustive Decision Reconstruction UI.
- [ ] Full learning/mastery breadth.

---

# 4. EVIDENCE / STANDARDS

Osteoporosis domains remain:

1. Diagnosis & case finding
2. DXA / VFA / imaging
3. Fracture-risk assessment
4. Secondary osteoporosis & laboratory evaluation
5. Pharmacologic treatment selection
6. Sequential therapy / treatment transitions
7. Monitoring / treatment response / adherence
8. Falls, frailty, exercise & nutrition
9. Communication / shared decision making / continuity

- [x] G-2 material Clinical Guidance rules carry source/version/applicability/strength/freshness where relevant.
- [ ] Material Practice Review claims link to explicit standards/evidence.
- [x] No silent framework hybridization in the G-2 guidance contract/runtime.
- [ ] Evidence-impact classification and renewal lifecycle.

Comprehensive curriculum/registry breadth is not a Module 01 closure blocker beyond what is required for safe guidance/review.

---

# 5. SAFETY / PRIVACY

- [x] Protected clinical route/session auth foundation.
- [x] PostgreSQL clinical encounter/lab storage.
- [ ] Complete broader legacy-route/CORS hardening before whole-service privacy claims.
- [ ] Sensitive-action/data-access audit trail.
- [ ] Retention/deletion/data-minimization policy.
- [ ] GDPR/privacy review appropriate to identifiable transcript use.
- [ ] Provider data-control/privacy gate before real identifiable transcripts.
- [x] Safety/event guidance outranks routine visit convenience in G-2/G-3 tested runtime.
- [x] Denosumab/time-critical therapy runtime rules use exact actual timelines and reviewed provenance.
- [x] G-3 longitudinal summary keeps current draft distinct, scheduled doses non-actual, and conflicts explicit.

---

# 6. DEFERRED / NON-BLOCKING TRACKS

Unless later evidence elevates one to a safety/data-integrity dependency:

- [ ] Patient Voice full program.
- [ ] External Benchmark Registry.
- [ ] Full Clinical Excellence Home/analytics polish.
- [ ] Calendar/Setmore/Zadarma/CareTask live integration.
- [ ] Radiofrequency utility runtime migration into this repository.
- [ ] Patient leaflets/posters/materials.
- [ ] New physiotherapy disease routes.
- [ ] Module 02/generalization.

Permanent: `Appointment != CareTask`.

---

# 7. CLINIC UTILITIES — PARKED/PRESERVED

Production CU-1 physiotherapy baseline is already historically merged/deployed.

Later rich-referral work remains preserved at:

```text
feat/cu1-rich-referral-global-evidence-2026-08-29
@ bdd23b83a8252405f5aa5a0c0b67f303ccfcef5f
IMPLEMENTED / TESTED / PRODUCT-OWNER REVIEWED
MERGED NO / DEPLOYED NO
```

G-4 adds Cockpit navigation to the existing protected RF PDF generator but does not migrate or duplicate that generator's backend/templates/persistence into this repository.

Do not mutate/merge/deploy later rich-referral or RF-engine migration work during Module 01 closure without separate authorization.

---

# 8. BROAD IMPLEMENTATION ORDER

```text
1. C1 authoritative Finish release/smoke — closed
2. G-1 dynamic-guidance mechanics — production-smoke-verified / closed
3. G-2 evidence-backed osteoporosis guidance — production-smoke-verified / closed
4. G-3 guidance salience + longitudinal patient summary — production-smoke-verified / closed
5. G-4 workspace ergonomics + RF utility navigation — implemented/tested; release review active
6. PR-1 transcript extraction
7. PR-2 inline provisional population
8. guided card UX sufficient for real use
9. 5-case system-assisted pilot
10. one refinement + contract freeze
11. Quick Practice Review shadow capability
12. 30-case system-assisted scored baseline
13. baseline lock
14. clinician-facing reviewed Signals/interventions
15. one longitudinal closed improvement loop
16. re-measurement / prompt-dependence trend where valid
17. final Module 01 closure review
18. later breadth/generalization
```

Safety/data-integrity defects always outrank this order.
