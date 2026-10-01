# R2 — Longitudinal Clinical Trajectory Review

> **Workstream:** OST-UI · **Mode:** fresh separate documentation review · **Date:** 2026-10-01 Asia/Nicosia
> **Disposition:** **R2 COMPLETE / R3 ELIGIBLE FOR COORDINATOR RECONCILIATION**
> **Scope:** current representation and reconstruction capability; no clinical-rule, runtime, schema, database, PR #121 or root-canonical mutation.

## A. Exact source identity

### Canonical Bootstrap Manifest

| Required item | Verified state |
|---|---|
| Repository / fresh remote `main` | `athpapachr-cmd/osteoporosis` / `63e903e05c1bfe22ca925374b8994355f6c92baf` (remote `refs/heads/main`, 2026-10-01) |
| Six root canonicals, fully read in `AGENTS.md` order | `AGENTS.md` → `TODO.md` → `CLINICAL_EXCELLENCE_PLAN.md` → `SLICE_PLAN_CURRENT.md` → `CURRENT_OPERATIONAL.md` → `osteoporosis-change-log.md`; their contents were checked against the verified `main` (no differences in these six files) |
| Major phase | Module-01 closure: dynamic guided consultation and transcript-assisted capture, followed by Practice Review and measurement/improvement loop |
| Active root slice / design | `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`; active, design verified, bounded implementation authorized; not a persisted longitudinal transcript solution |
| Root writer / allowed R2 scope | `CURRENT_OPERATIONAL.md` reserves overlapping root PR-1 work. OST-UI is a parallel review sidecar. R2 may write only this artifact and `programme/OST-UI/CURRENT.md` on its own branch. |
| Branch / PR / release state relevant here | OST-UI coordinator branch `docs/ost-ui-r1-current-product-audit-2026-09-27` fresh remote head `f04af35b5424328f55a11932224becb43521d831`; R1 `COMPLETE / COORDINATOR ACCEPTED`. PR #121 is **OPEN**, head `aa32f7fbd49c02653c11eb35edaf2e24939eb443`, base `main`; changes after reviewed P1 target are only `programme/OST-LIFECOURSE/CURRENT.md` and `programme/WORKSTREAM-REGISTRY.md`. S1 is merged on current `main`; its workstream checkpoint says auto-deploy live, production smoke not run. No R2 deploy/smoke applies. |
| Permanent invariants | Protected patient facts and completed/amended encounter history; missing/unavailable history is not negative history; actual administration differs from planned/scheduled and derived due; generic fracture differs from confirmed fragility; clinical guidance, transcript capture, audit and Practice Review have separate authority; public repository contains no identifiable patient data. |
| Exact authorized action | Examine R2 longitudinal adequacy, write this review and local OST-UI checkpoint, return to programme coordinator, stop. |
| Deferred / forbidden | R1 repetition, R3 interaction design, R4 architecture, synthesis, P2 LifeCourse design, PR-1/H-12 work, implementation, runtime/schema/database/root-canonical or PR #121 mutation, merge/deploy. |

### Governing and reviewed inputs

- The **full** `PROJECT-INDEX.md`, `CURRENT.md`, and accepted `R1-CURRENT-PRODUCT-CONSTITUTION-AUDIT.md` were read at OST-UI coordinator head `f04af35…`. R1's established diagnosis is a protected longitudinal substrate with history-sensitive guidance inside an encounter/audit-first six-step shell; R2 does not repeat that inventory.
- Independently reviewed LifeCourse P1 ownership input is the corrected target `b4f917b161f33b0bfb3507328df07cec2d7bd2b6`, with coordinator-recorded independent delta+cumulative **PASS**. `programme/OST-LIFECOURSE/P1-PROGRAMME-RECONCILIATION.md` at that target was read as current-state semantic ownership, not P2 architecture.
- Current-main S1 executable semantics were inspected in `app-core.js`, `osteoporosis-evidence-guidance-core.js`, `osteoporosis-longitudinal-summary-core.js`, and `progressive-guidance-ui.js`. Merged S1 commit `dd16a505662606f588a2adc8b1ae0ddddd884d19` is in the verified main ancestry. The S1 checkpoint is `programme/OST-CLINICAL/CURRENT.md` on main.
- Runtime evidence was read from the exact `63e903e…` checkout: `clinical_data.py`, `clinical_data_ext.py`, `static/baseline-audit/{app-core,patient-registry,step3,step4,step5,longitudinal,prior-dxa-inline,progressive-guidance-core,progressive-guidance-ui,osteoporosis-evidence-guidance-core,osteoporosis-longitudinal-summary-core}.js`, and the relevant G1/G2/G3/S1 regression fixtures. This was source and deterministic synthetic reasoning, not production patient-data inspection.

**Evidence labels:** **OBSERVED** means current source or an executable synthetic probe establishes the behavior; **INFERRED** means the longitudinal consequence follows from that behavior; **UNVERIFIED** means no reliable conclusion from this review. Design intent, current executable behavior, and clinician-entered truth are kept separate.

## B. Longitudinal truth map

Clinical class and technical role are both shown. “Durable” means the value can survive in a protected completed/amended encounter or dedicated lab row; it does not imply that its clinical meaning is complete or that its history is immutable.

| Clinical concept → class | Current durable owner → technical role | Derived projection / identity / time semantics | Limitation |
|---|---|---|---|
| Patient and encounter → STATE / EVENT | `clinical_patients`; `clinical_encounters` (`encounter_id`, `encounter_date`, `status`, full `payload_json`) → **authoritative durable fact** for submitted encounter content | Completed/amended rows feed G1/G3; draft/current UUID is excluded from prior truth | Encounter JSON can be amended in place; no immutable as-at-decision version is exposed (`clinical_data.py:30-49, 307-332`). |
| Fracture → EVENT; fragility mechanism → INTERPRETATION / UNCERTAINTY | `fracture_history.events[]` (`id`, site, year-month, `low_trauma`, `occurred_on_treatment`) → **encounter snapshot** of structured fact; `risk_context.prior_fragility_fracture` is compatibility summary | G2/G3 dedupe by stable ID or fallback fact key; confirmed fragility requires normalized `low_trauma=yes`; conflicting explicit mechanisms fail closed | Month rather than day; no automatic link to a specific treatment epoch. Generic event does not prove fragility. Legacy claim remains unconfirmed (`app-core.js:179-203`; evidence core `:76-107, 205-207`; summary core `:111-187`). |
| Glucocorticoid / CKD context → STATE / CONSTRAINT | `risk_context.glucocorticoids`, dose, duration, `secondary_conditions` including `ckd` → **encounter snapshot** | G2 reads current values; past encounter dates allow coarse comparison | No dated exposure interval or constraint-resolution sequence; default false/blank can conceal not-asked versus negative (`app-core.js:131, 236-246`; `index.html:163,184`). |
| Treatment episode → STATE / OUTCOME / PLAN | `step4.treatment_episodes[]` row ID, agent, status, start/end or approximate duration, adherence/tolerance/response, optional start/stop reasons → **encounter snapshot** | G1 uses **latest nonempty** completed/amended treatment snapshot, not an epoch history merger; row ID is local to stored row; dates may be unknown | Full sequence can be manually reconstructed from old encounters but current projection may omit earlier epochs. Same episode across visits has no enforced continuity (`step4.js:174-190`; guidance core `:205-224, 263-266`). |
| Administration → EVENT or PLAN | `step4.administrations[]` row ID, scheduled/actual/next-due dates and status → **encounter snapshot** | G1 actual history admits exact `actual_date`, dedupes by agent+date, checks reused ID conflict; G2 calculates evidence due from reliable actual history | Scheduled/missed rows persist but are absent from actual-event projection; a later actual event is not linked to a particular missed plan (`step4.js:193-201`; guidance core `:114-203`; evidence core `:121-166, 449-481`). |
| Clinical decision → DECISION | `step4.decision` type, selected agent, reasons, rationale, acceptance and review flags → **encounter snapshot** | G3 surfaces latest explicit decision, not full decision chain | Discussion/recommendation are not separately structured; decision-to-action linkage is inferential (`step4.js:61-93, 242-248`; summary core `:232-255`). |
| Preference / constraint → PREFERENCE / CONSTRAINT | `decision.preference_documented`, `patient_accepted`, reasons checkbox and optional rationale; Step-5 communication process flags → **encounter snapshot** | Historical encounter retrieval only | A preference's content, source, duration and effect on a particular option are generally not losslessly structured (`step4.js:80-93`; `step5.js:35-50, 112-124`). |
| Transition → PLAN | `step4.transition` type, prior end, next agent/date, explicit plan, unresolved safety, note → **encounter snapshot** | G2 exit guidance reads current transition; no separate durable transition completion projection | Planned successor is not actual administration; no enforced link among stop decision, successor episode, actual dose and monitoring (`step4.js:95-110, 249-253`; evidence core `:471-481`). |
| Task → FUTURE OBLIGATION | `step4.tasks[]` UUID, type, due date/timeframe, status → **encounter snapshot** | G1 cross-encounter key is `type | due_date | timeframe_text`; only latest status of that tuple is used. UUID remains stored row identity. | Date/text edit creates a second semantic key, not a stable obligation transition; disposition vocabulary limited to `planned`, `already_done`, `not_applicable` (`step4.js:204-212`; guidance core `:226-243`). |
| Critical close → UNCERTAINTY / FUTURE OBLIGATION | `step4.close` plan completeness, unresolved critical flag, note → **encounter snapshot** | G1 carries **latest encounter's** close flag | Older unresolved critical item can cease to surface if a later close is blank/no without a linked resolution (`step4.js:117-123,254`; guidance core `:245-253`). |
| DXA → OBJECTIVE RESULT / EVIDENCE CONTEXT | `step3.dxa` → protected **encounter snapshot**; `longitudinal_review.dxa_history[]` → **manually authored historical factual records** inside encounter payload | G3 latest DXA reads `step3.dxa`; Step-3 trends combine current DXA and manual rows; manual `_id` is row identity | Unique historical scan may exist only as manual row; same scan may be duplicate/conflicting across owners; no source-scan reconciliation (`step3.js:30-69`; `longitudinal.js:128-158`; `prior-dxa-inline.js:97-115`; summary core `:58-82`). |
| Laboratory / renal / BTM → OBJECTIVE RESULT | `step3.labs` → **encounter snapshot**; `clinical_lab_snapshots` → **patient-level longitudinal fact** with row ID, lab date and optional source encounter | Lab table and G3 latest snapshot use dedicated rows; same encounter+date is browser upsert lookup | Date change or cleared form can leave stale prior row; dual representation and optional source link require provenance reconciliation (`step3.js:106-158, 354-356`; `patient-registry.js:231-277`; `clinical_data.py:42-50`; `clinical_data_ext.py:40-62`). |
| Formal FRAX / risk → RESULT / INTERPRETATION | `risk_assessment` plus `longitudinal_review.frax_history[]` and FRAXplus fields → **encounter snapshot** and possible **overlapping manual representation** | Descriptive trends include current calculated/display point; G3 latest formal risk reads encounter risk | Historical manual rows can duplicate encounter risk; original formal and adjusted/contextual values are separate, but full provenance/time-of-evidence is limited (`app-core.js:243-246`; `longitudinal.js:43-55,124-157,190-193`; summary core `:84-106`). |
| Guidance / due / patient summary → DERIVED STATE / CLINICAL-RULE OUTPUT | No patient-fact owner: G1 projection, G2 evidence context, Visit Plan and G3 summary are **derived / rebuildable projections** | Rebuilt from completed/amended history, current state and current rule implementation | Their present output does not prove what was displayed or guideline context at an old decision (`progressive-guidance-core.js:103-108,255-284`; evidence core `:259-300, 304-320, 642-657`; summary core `:319-349`). |
| Local draft and session preference → WORKING / PRESENTATION STATE | `localStorage` working case/link cache; G4 `sessionStorage` collapse state → **working/transient** and **presentation-only** | Server sync gives protected durability to saved payload, not to every unsynced draft change | Neither is independent longitudinal clinical authority (`patient-registry.js:20-29, 268-285`; R1 §E3/G4). |

## C. Durable versus derived state map

```text
patient
  → completed/amended encounter payloads: dated clinical snapshots, decisions, plans, rows
  → clinical_lab_snapshots: dated patient-level laboratory facts
  → G1 LongitudinalGuidanceProjectionV1: read-only reconstruction
  → G1 EncounterContext / VisitPlan + G2 evidence adapter: current rule evaluation
  → G3 summary / Step-3 trends: presentation projections
```

**OBSERVED adequacy:** this chain can retain many dated facts and make prior history change today's guidance. It preserves actual-versus-scheduled distinction, source encounter IDs in several projections, history-load failure, and some conflicts. `LongitudinalGuidanceProjectionV1` has no patient-fact write authority. The G0 frozen `EncounterContextV1` design is broader than the executable G1 builder; G2's evidence context is a rule adapter, not another patient-state owner (guidance core `:294-355`; evidence core `:225-300`). Reviewed evidence/registry/manifest own rule intent; JS executes it; transient rule output is not a durable decision record.

**INFERRED limit:** encounter snapshots plus lab rows suffice to recover *some* timeline facts by opening records, but they do not by themselves give stable identity for a treatment epoch or changing obligation, link a decision to subsequent action/outcome, preserve the exact evidence context at that decision, or reconcile overlapping investigations. A missing cross-time link must not be fabricated by a projection.

## D. Ten trajectory pressure tests

| Journey | What current structures can preserve | Reconstruction verdict / precise break |
|---|---|
| **A. New patient entering care** | Initial history/risk/fracture events, DXA, dated labs, formal risk, Step-4 decision/rationale, acceptance flag and planned tasks can all live in a completed encounter. | **PARTIAL.** Recommendation and option discussion are Step-5 process flags, not option/recommendation contents. Preference detail is usually optional prose or absent. Final decision is structured; actual subsequent action is a later separate fact. |
| **B. Several years of denosumab** | Episode status/start/reasons, exact actual administration rows, planned/scheduled rows, labs and subsequent decisions can survive in successive encounters; G1 counts reliable actual events and G2 can use exact last actual date. | **PARTIAL/FRAGMENTED.** Current treatment is latest nonempty snapshot. Reasons and interruptions must be found across encounters. Recorded administrations can be counted, but incomplete capture cannot prove complete exposure; no enforced epoch continuity. No cadence is inferred here. |
| **C. Delayed/missed denosumab dose** | Last actual `actual_date`, scheduled date, `missed` status, later actual date and current derived due state are different stored/derived things. A synthetic G1 probe with an actual row and a later scheduled-only `missed` row counted **one** actual administration. | **PARTIAL.** The missed row is retrievable in its encounter but not a first-class actual-event projection or linked resolution. A late administration cannot automatically be attributed to that plan. Derived due uses current evidence and reliable actual history; it is not historical fact. |
| **D. Denosumab discontinuation / transition** | `decision.stop`/rationale, transition fields, acceptance flag, successor episode, actual administration, labs and task can each be recorded. G2 raises reviewed exit guidance without selecting therapy. | **FRAGMENTED.** There is no durable chain linking stop reason → discussed options → recommendation → preference → final choice → actual successor dose → monitoring → outcome. `transition.next_agent_date` is planned, not actual. R15/R16 remain inactive for missing linkage/availability semantics. No successor strategy is decided here. |
| **E. Fracture during treatment** | Structured event site/month, explicit `low_trauma`, `occurred_on_treatment`, treatment snapshot, actual administration dates and new decision are retrievable. | **PARTIAL.** Event-level “on treatment” assertion does not identify which epoch or prove exposure/adherence at event time. Treatment-failure interpretation is separate; no automatic failure label is justified. Outcome/response link is not explicit. |
| **F. New fragility fracture** | Merged S1 keeps generic fracture separate from confirmed low-trauma fact. G2 current fragility requires interval status and `low_trauma=yes`; G3 counts generic events separately from confirmed fragility and fails closed on mechanism conflict. | **PARTIAL but safer.** A confirmed event can change current guidance. Unknown/uncertain/legacy-only mechanism remains unconfirmed. Date precision is month and interpretation/decision follow-up remain encounter-linked only. |
| **G. Long-term glucocorticoid exposure** | Each saved encounter can hold present GC checkbox, dose and duration; G2 can use current values. | **FRAGMENTED.** There is no dedicated dated exposure start/stop/dose-change course. Comparing snapshots can show change *at visits*, but cannot reliably establish intervening exposure or whether default false meant assessed negative. |
| **H. Evolving CKD / renal constraint** | `ckd` secondary-condition flag and dated creatinine/eGFR lab snapshots can coexist with decisions and checklist-only medication safety guidance. | **PARTIAL.** Lab trend is factual if dates/provenance are sound. A specific changing renal constraint, clinician interpretation and why it changed a plan are not linked; no threshold is defined by R2. Lab dual-write can further confuse the historical value. |
| **I. Multi-year monitoring** | DXA snapshots and manual history, dated lab/BTM snapshots, formal FRAX and manual FRAX history, Step-3 comparison/LSC fields, and descriptive charts exist. | **PARTIAL/PROVENANCE-SENSITIVE.** Lab trend can be factual; DXA/FRAX overlap can double-count or disagree; BTM context is captured but longitudinal result-to-decision association is weak. Descriptive BMD change is not proof of significant change without comparability/LSC. |
| **J. Possible treatment failure / reassessment** | Prior episode response/adherence, actual rows, new fracture/result, decision reason `inadequate_response`, optional rationale, planned task can all be retrieved. | **FRAGMENTED.** A fresh clinician cannot reliably establish whether exposure was complete, what competing explanations were considered, which option was recommended, whether preference changed the choice, or which later outcome belongs to this reassessment. `inadequate_response` is a recorded interpretation, not automatic failure. |

## E. Treatment-epoch assessment

| Question | Current answer / classification |
|---|---|
| Which treatment was active? | Latest nonempty episode snapshot can answer a current asserted state when unambiguous: **RECONSTRUCTABLE WITH LIMITATIONS**. Earlier epochs require manual encounter review; multiple active rows trigger conflict. |
| Why did it start or stop? | Optional episode reason fields plus decision reasons/rationale can preserve this if entered: **FRAGMENTED**. No enforced link between repeated snapshots and specific decision. |
| When did actual exposure begin? | For administered agents, earliest captured exact `actual_date` can be retrieved; episode `start_date` is an asserted course date. These may differ. **RECONSTRUCTABLE WITH LIMITATIONS** if capture is complete; otherwise **UNCLEAR**. |
| Interruptions and delays? | Scheduled/missed/overdue rows may be recorded, and G2 derives timing from reliable actuals. No stable missed-plan→actual-resolution relation. **FRAGMENTED**. |
| Successor treatment and response/outcome? | Transition plan, later episode/actual row, `response_context`, fracture/DXA/lab facts exist. Their relationship is temporal unless separately explained. **FRAGMENTED**. |

Overall treatment-epoch adequacy: **FRAGMENTED** for a 10–15-year coherent epoch narrative, despite **adequately durable individual completed-encounter snapshots and captured actual administration facts**. This is a semantic-continuity finding, not a conclusion that encounter persistence must be replaced.

## F. Decision, preference and action continuity

| Required distinction | Current-state test |
|---|---|
| OPTION DISCUSSED | Step-5 `alternatives_tradeoffs_discussed` says whether discussion occurred. It does not identify the options or their rationale. **PARTIAL**. |
| CLINICIAN RECOMMENDATION | `decision.rationale` may narrate it, but no distinct structured recommendation owner in the persisted Step-4/5 shape. **NOT REPRESENTED as distinct durable semantic fact**. |
| PATIENT PREFERENCE / CONSTRAINT | `preference_documented`, reason `patient_preference`, acceptance and process flags survive; content and temporal applicability are generally missing. **PARTIAL**. |
| FINAL DECISION | `decision.type`, `selected_agent`, reasons/rationale and acceptance can be retrieved by encounter. **RELIABLE if explicitly completed**, with optional rationale limitation. |
| ACTUAL SUBSEQUENT ACTION | Exact administration row or later encounter/lab fact may prove an action. A decision, plan, task or appointment alone does not. **PARTIAL linkage**. |

PR-1's semantic candidate vocabulary preserves these distinctions in transient proposed extraction, but PR-1 has no authoritative write and does not retroactively repair persisted historical encounters. Later outcomes must not be assigned to a prior decision solely because they followed it.

## G. Future-obligation assessment

Step-4 rows can persist `planned`, `already_done`, `not_applicable`, exact due date or vague timeframe; prior planned rows can resurface. G1 also derives explicit due/overdue treatment state from stored next-due date, and G2 can derive a rule-based due date from reliable actual treatment history. `close.unresolved_critical` is a coarse latest-encounter flag. These are useful existing continuity mechanisms.

**OBSERVED synthetic executable probe:** two completed encounters carried the same task UUID `t1` and type `lab`, first due `2020-06-01`, later due `2020-07-01`, both `planned`. G1 returned **two** unresolved tasks and no conflict. A scheduled-only `missed` administration in the same probe did not increase actual dose count. The probe called `buildLongitudinalProjection` from exact main source without writing patient data.

| Disposition pressure | Current representation |
|---|---|
| planned | Explicit status and resurfacing by semantic tuple. |
| completed | `already_done` can close only an exact matching tuple; action evidence is not automatically linked. |
| rescheduled / changed date or timeframe | New tuple; old planned tuple may remain unresolved even with same stored row UUID. |
| deferred / patient declined / alternative plan chosen / superseded | No corresponding task status. Can be narrated elsewhere, but cannot reliably retire the original obligation in current projection. |
| not needed | `not_applicable` exists, but only exact-tuple continuity is recognized. |
| unknown | No explicit task disposition; absence or blank does not establish resolved state. |

The task UUID is durable **row identity**, not current cross-encounter obligation identity. The gap is stable continuity and richer disposition semantics. R2 makes no replacement-model decision. A later blank close can also mask an older critical close flag in the latest-only projection, without evidence that the earlier issue was resolved.

## H. Goals and target state

**INFERRED:** a desired future state is often inferable from a selected treatment, transition plan, tasks, decision rationale or free text. The current persisted shape has no general durable goal/target-state owner with identity, target date, review/revision and achievement status. A task says what to do; an episode status says what is asserted now; neither necessarily states the clinical goal. This is a **GENUINE REPRESENTATIONAL GAP** for explicit longitudinal goals, with **UNVERIFIED** clinical scope until the responsible clinical owner defines which target states matter. No target thresholds are invented here.

## I. Outcome linkage

Fracture events, DXA/lab results, treatment response context, adverse-effect/tolerance status and subsequent decisions can be ordered by encounter or measurement date. This supports **temporal relationship**. An entered interpretation or rationale may support a **documented clinical association**. Current IDs do not inherently link an outcome to an intervention/decision, and nothing supports an automatic **causal claim**. Therefore later evaluation of “did the intervention work?” is **partial** and must retain exposure completeness, competing events, measurement quality and uncertainty.

## J. Uncertainty and conflict continuity

**Preserved where implemented:** `low_trauma=uncertain`/unknown is not confirmed fragility; G2/G3 stable-ID mechanism conflict fails closed; G1 records reused-administration-ID and next-due conflicts; multiple active treatment episodes surface conflict; protected-history `unavailable` is not zero history; current draft is not completed history. Optional decision uncertainty, patient undecided, DXA comparability uncertainty, secondary-cause unresolved and critical-close flags can be stored.

**Flattening risks:** checkbox defaults (`glucocorticoids`, `ckd` secondary condition) can collapse unasked and negative; G3 “latest” risk/DXA/lab/decision hides intermediate uncertainty; G1 task tuple can multiply or lose dispositions; latest close flag can supersede an older unresolved issue without explicit resolution; overlapping DXA/lab facts may have no conflict record; an amended encounter's prior version is not reconstructable from the current row alone. Thus the current fail-closed mechanisms are valuable but domain-limited, not a universal uncertainty ledger.

## K. Investigation continuity

### DXA

`step3.dxa` is an encounter-scoped factual snapshot, including date, machine, measurements, quality and comparability/LSC review. `longitudinal_review.dxa_history[]` holds manual historical factual rows with their own `_id`; one can be the **only** recorded scan or overlap a protected encounter scan. The trend code combines manual rows and current DXA, while G3 latest DXA reads protected `step3.dxa` only. Therefore one scan can appear twice, values can conflict, and a unique manual historical fact can be absent from G3's latest-DXA summary. Manual rows have scan date/machine/value but no enforced source-report or source-encounter link. **Class:** **CONSOLIDATE EXISTING**, with provenance-sensitive factual reconciliation; never demote a unique manual fact by assumption.

### Laboratories and BTMs

`step3.labs` contains a dated encounter-captured/reviewed result, including creatinine/eGFR and CTX/P1NP/other BTMs plus BTM context. Browser sync creates/updates a patient-level `clinical_lab_snapshots` row by matching **same source encounter + same lab date**. If the encounter lab date changes, the old row is not matched and a new row is inserted; if fields/date are later cleared, `hasLabValues` skips sync and the old row remains. The dedicated lab update endpoint can change row date/source ID; the review did not find reconciliation with the source encounter payload. Duplicate observations and stale source-linked rows can therefore survive. G3 takes the latest dedicated lab row; it does not resolve dual-owner conflict. **Class:** **CONSOLIDATE EXISTING / EXTEND EXISTING OWNER** for lifecycle/provenance integrity, without specifying how.

### FRAX / risk

Formal original MOF/hip values, model/framework and contextual interpretation live in `risk_assessment`. `longitudinal_review` also stores manually added formal/adjusted risk history and FRAXplus context; Step-3 charts combine manual history with current point. This can show trends while retaining original-versus-adjusted distinctions, but historical manual rows can overlap encounter facts, and risk categories/framework or evidence context may change across time. A charted difference alone is not a causal treatment outcome. **Class:** **CONSOLIDATE EXISTING** where duplication exists; otherwise **REUSE AS-IS** for the captured factual assessment.

## L. Evidence at decision time

**Question 1: What evidence/guideline context applied when the original decision was made?** Current `step4.decision` retains the clinician's reasons/rationale and the encounter date, but not the evaluated rule IDs/source versions/strength/freshness or the actual guidance shown. The G2 registry/manifest is versioned clinical-rule intent and current JS produces source refs/rule trace for **today's** evaluation; the resulting Visit Plan is ephemeral, not a persisted patient decision record. Re-running current rules on old facts would answer a different question and may use revised evidence. Historical as-at-decision evidence is therefore **UNVERIFIED / generally not reliably reconstructable** from the persisted encounter alone.

**Question 2: What does current evidence say today?** G2 can evaluate the current patient/visit context against current reviewed rule content and show source refs, including checklist-only safety limits. This is a **REUSE AS-IS** current guidance capability, subject to its reviewed activation scope and exact history quality. R2 does not define evidence-version storage or modify medical rules.

## M. Fifteen-year reconstruction test

The following is **synthetic only**. Dates are illustrative recorded facts, not a treatment schedule or clinical recommendation. Every item uses a currently persistable protected encounter payload or `clinical_lab_snapshots`; no imagined event store or future PR-1 write is used.

| Synthetic time | What is saved in current structures | Later reconstruction |
|---|---|---|
| 2011 initial completed encounter | `fracture_history.events[{id:F1,site:vertebral,month:2011-04,low_trauma:yes}]`; `step3.dxa` dated result; `risk_assessment` formal result; dated `step3.labs`; `step4.decision.start` alendronate with optional rationale; episode `E1 active` | **RELIABLE** for entered fracture/site/month/mechanism, investigations and final selected drug; **PARTIAL** for options/recommendation/preference unless prose records them. |
| 2014 completed review | Episode `E1 stopped`, `reason_stopped` optional; decision switch; new episode `E2 planned` denosumab; task planned | **PARTIAL**: stop and plan are dated encounter assertions, but the exact relation among old/new episode rows depends on manual interpretation. |
| 2015–2020 several completed visits | Selected denosumab `administrations[]` have illustrative exact `actual_date` values `2015-04-17`, `2017-09-11`, `2019-04-20`; a separate row has `scheduled_date:2019-02-01`, `status:missed`, and no actual date; dated lab snapshots are also saved. These sparse examples do not assert a complete course or prescribing cadence. | **RELIABLE** for *captured* actual dates and that scheduled-only row was not an actual dose; **AMBIGUOUS** whether the later dose resolved that particular missed plan and whether uncaptured doses exist. |
| 2021 completed review | A manually entered `longitudinal_review.dxa_history[]` row for a 2018 scan, plus `step3.dxa` for a 2021 scan; a second manual 2021 row may duplicate the latter | **PARTIAL** for values/dates; **AMBIGUOUS** duplication and provenance; G3 may omit unique manual history. |
| 2022 completed review | `risk_context.secondary_conditions=[ckd]`; dated creatinine/eGFR `step3.labs` and dedicated lab rows; decision rationale notes renal concern | **RELIABLE** for dated captured lab values if row lifecycle is reconciled; **PARTIAL** for when the constraint arose and exactly why it altered choice. |
| 2024 completed fracture visit | `F2` with year-month, low-trauma assertion and `occurred_on_treatment=yes`; current episode snapshot, new risk/decision/task | **RELIABLE** generic fact and confirmed mechanism if consistent; **AMBIGUOUS** exact event day, actual exposure at fracture and whether response/failure interpretation is justified. |
| 2025–2026 completed transition/review | `decision.stop`, transition next agent/date, task due date changed on later visit, later successor administration if actually entered, new lab/BTM; latest episode snapshot | **PARTIAL** final plan and any actual action; **AMBIGUOUS** task continuity and successor linkage; **IMPOSSIBLE** to recover unrecorded option/recommendation/preference content or original evidence version. |

**Fresh clinician test:** opening all completed/amended encounters and lab rows can recover major recorded events, selected decisions and exact captured actual administrations. The current projection can give a useful current snapshot, but it cannot certify a complete 15-year exposure sequence; it may collapse treatment history to the latest snapshot, show duplicate obligations after a changed due date, omit unique manual DXA facts from the summary, and cannot restore erased encounter revisions or never-captured decision/evidence content. Therefore “the full patient story” is **PARTIAL**, with specific **AMBIGUOUS** and **IMPOSSIBLE** components above. The limitations are semantic and provenance limits as well as presentation limits.

## N. Reuse / consolidate / extend / genuine-gap matrix

| Responsibility / observed gap | Reuse-before-new-path questions answered | R2 classification |
|---|---|
| Protected patient, completed/amended encounters and dated lab storage | Already persisted and protected; current historical loading can safely derive bounded context. | **REUSE AS-IS** as substrate, with lifecycle caveats below. |
| G1 projection, fail-closed history and actual administration dedupe; G2 current evidence; G3 current summary | Derived from existing facts; no patient-fact write authority or duplicate clinical owner needed. | **REUSE AS-IS** for current bounded derivations. |
| DXA manual history versus encounter snapshot; FRAX manual history versus encounter risk | Information already exists in two places; provenance/duplicate ownership is the problem. | **CONSOLIDATE EXISTING**; unique manual facts remain factual. |
| Lab encounter snapshot versus dedicated lab row | Existing dated owner exists; date edits, clearing and source linkage create stale/duplicate facts. | **CONSOLIDATE EXISTING** and **EXTEND EXISTING OWNER** for lifecycle semantics. |
| Treatment epoch continuity and actual/missed/successor linkage | Rows exist, but latest-snapshot projection and row IDs do not establish one stable epoch/action chain. | **EXTEND EXISTING OWNER** for continuity; whether a genuinely new durable concept is needed is **UNVERIFIED** pending bounded semantics. |
| Task rescheduling, deferral, decline, alternative or supersession | Row UUID exists, but executable cross-encounter key is unstable; disposition vocabulary is narrow. | **EXTEND EXISTING OWNER**; stable identity/disposition semantics are the gap, not absence of task rows. |
| Option, recommendation and preference content distinct from final decision | Process flags/rationale exist but cannot losslessly reconstruct the distinctions. | **GENUINE REPRESENTATIONAL GAP** in historical persisted semantics; exact clinical scope **BLOCKED ON CLINICAL SEMANTICS** where needed. |
| Explicit desired clinical goal / target state | Sometimes inferable from plan/tasks, but inference cannot prove what target was agreed or later achieved. | **GENUINE REPRESENTATIONAL GAP**; target definitions **BLOCKED ON CLINICAL SEMANTICS**. |
| Outcome-to-decision/intervention association | Events/results already persisted; temporal order is derivable, causal relationship is not. | **EXTEND EXISTING OWNER** for explicit association semantics if clinically authorized; otherwise **UNVERIFIED**. |
| Original evidence at decision time | Current registry/rules exist, but prior rule exposure/version is not in encounter decision. Re-evaluation today is not historical evidence. | **GENUINE REPRESENTATIONAL GAP** for reliable as-at-decision reconstruction. |
| Complete life-course visible at point of care | Many source facts already exist but are scattered. | **PRESENTATION-ONLY GAP** only for already reliable facts; R3 may assess access requirements, not repair semantic gaps. |
| CKD/GC exposure trajectory | Encounter snapshots/labs exist; precise exposure/constraint intervals are not represented. | **EXTEND EXISTING OWNER** or **UNVERIFIED** for clinical scope; do not infer intervals from sparse visits. |

## O. Dependency register

| Dependency | Owner / R2 boundary |
|---|---|
| Clinical fracture/fragility meaning and S1 | OST-CLINICAL/current main. Consume merged S1; generic fracture is not confirmed fragility. R2 makes no new fragility rule. |
| LifeCourse P1 ownership and parked Q1–Q12 | P1 corrected target `b4f917b…` is reviewed **PASS** current-state input. P2, migrations and new stores remain parked with OST-LIFECOURSE/coordinator. |
| Treatment and evidence rule meaning | Reviewed G2 evidence registry/manifest and clinical owner. R2 does not define cadence, successor strategy, renal threshold, treatment failure or monitoring rule. |
| PR-1 / H-12 and future capture | Root PR-1 lifecycle is separate. Proposed candidate distinctions cannot be counted as historical persisted truth; R2 did not wait for or inspect H-12 release engineering. |
| R3 point-of-care interaction | R3 can use R2's reliable/partial/ambiguous information requirements only after coordinator reconciliation; no screen/navigation/layout proposal is made here. |
| R4 architecture / shared Core | R4 decides reuse/ownership implementation if later authorized. R2 does not prescribe tables, APIs, event sourcing, migrations or component structure. |
| Verification limit | Review establishes source behavior and one synthetic G1 projection probe, not completeness or accuracy of any real patient record, live production smoke, or clinical validity of new rules. |

## P. Direct R2 conclusion

**The present substrate is adequate for a bounded longitudinal current-context product:** it can retain protected dated encounters and lab facts, distinguish actual from scheduled administrations, carry structured fracture mechanism after S1, reconstruct a latest asserted treatment snapshot, surface some unresolved tasks/conflicts, and use prior history to change today's guidance. These are real assets for the Product Constitution.

**It is not yet semantically adequate to guarantee reconstruction of the full 10–15-year osteoporosis care trajectory.** The material failures are treatment-epoch and interruption continuity; decision/recommendation/preference/action separation; stable obligation identity and dispositions; explicit goals and outcome association; investigation duplicate/provenance lifecycle; and evidence as known at the original decision. Some facts are already stored and need consolidation or better derivation. Other distinctions were never losslessly captured. A presentation change alone cannot make those missing links true.

This is a responsibility-by-responsibility answer, not an all-or-none verdict on the existing patient→encounter→lab→projection architecture. It does **not** establish that a new store, a rebuild or a particular UI is required.

## Q. Disposition

**R2 COMPLETE / R3 ELIGIBLE FOR COORDINATOR RECONCILIATION**

R2 ends with this artifact and the local `programme/OST-UI/CURRENT.md` checkpoint. The programme coordinator must reconcile this branch/head/artifact before deciding whether to authorize R3. No R3, R4, synthesis, implementation, schema/database mutation, PR #121 mutation, root-canonical mutation, merge or deploy was performed.
