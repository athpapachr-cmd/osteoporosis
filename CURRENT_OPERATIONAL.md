# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-2 PRODUCTION-SMOKE-VERIFIED / G-3 RELEASE REVIEW PASS / PR #70 OPEN / MERGE HOLD.
> **Updated:** 2026-09-01 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **G-2 release PR:** `#69` — squash merged.
> **G-2 merge/runtime SHA:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **G-2 Render deploy:** `dep-daaph5vlk1mc73940g60` — `live`, exact commit `9cfad82d...`.
> **G-3 branch:** `feat/module01-g3-guidance-salience-longitudinal-summary-2026-09-01`.
> **G-3 exact tested runtime head:** `dab45baf9f80632ee6e58f03fa4d5005c68e0ac5`.
> **G-3 release PR:** `#70` — OPEN, non-draft, base `main`, explicit MERGE HOLD.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE after release-review closeout.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.

---

# 1. G-2 release state — CLOSED

The product owner completed the requested authenticated synthetic production smoke on the deployed G-2 ancestry and reported the smokes passed.

```text
G-2 DESIGN-COMPLETE = YES
G-2 IMPLEMENTED = YES
G-2 TESTED = YES
G-2 RELEASE REVIEW = PASS
G-2 MERGED = YES
G-2 DEPLOYED = YES
G-2 PRODUCTION-SMOKE-VERIFIED = YES
G-2 PILOT-VALIDATED = NO
```

Do not relabel product-owner production smoke as real-clinic pilot validation.

The G-2 clinical safeguards remain frozen, including inactive R15/R16 and no automatic CTX threshold retreatment command, ordinal Prolia milestones, automatic treatment failure/switch, selected-agent mutation, or automatic named-specialty referral.

---

# 2. G-3 implemented boundary

Slice:

```text
M01-G3-GUIDANCE-SALIENCE-LONGITUDINAL-SUMMARY-v1
```

Implemented runtime:

```text
static/baseline-audit/osteoporosis-longitudinal-summary-core.js
static/baseline-audit/progressive-guidance-ui.js
static/baseline-audit/progressive-guidance.css
static/baseline-audit/app.js
```

## Newly surfaced guidance

```text
initial plan = baseline, not NEW
new evidence/event/unresolved/due/treatment-context item = eligible for NEW
base-flow-only VISIT_TYPE_CORE addition = not NEW noise
new item = explicit `Νέο` text badge + stronger visual emphasis
color alone is never the only signal
new state is ephemeral/in-memory only
item removed from plan = new state cleared
patient/case switch = baseline reset
```

Product-owner fixture:

```text
height loss 3.9 cm
→ no R02 evidence trigger
→ height loss 4.0 cm
→ OST_G2_R02_VFA_STRUCTURED_TRIGGER activates
→ VFA enters newly-surfaced set
→ top guidance item + destination card receive `Νέο` emphasis
```

## Longitudinal patient summary

A protected-patient `Σύνοψη ασθενούς` is rendered above `Σημερινή ροή` from completed/amended protected encounters, protected lab snapshots, the existing G-1 longitudinal projection and the explicitly separate current visit.

Visible domains:

```text
Πορεία
Κατάγματα / κίνδυνος
DXA
Θεραπεία / actual administrations
Εργαστηριακά
Τελευταία explicit management απόφαση
Εκκρεμότητες / conflicts
```

Truth rules preserved:

```text
LATEST BLANK != ERASE PRIOR AUTHORITATIVE FACT
MISSING != NEGATIVE
CONFLICT != SILENTLY PICK LATEST
SCHEDULED DOSE != ACTUAL DOSE
DISCUSSION / OPTION != FINAL DECISION
CURRENT DRAFT != COMPLETED HISTORICAL FACT
```

No AI narrative summary, DB migration, new clinical write or treatment decision automation was introduced.

---

# 3. G-3 test evidence

Exact tested runtime head:

```text
dab45baf9f80632ee6e58f03fa4d5005c68e0ac5
```

Push workflow:

```text
G3 guidance salience longitudinal summary
run 33486322905
COMPLETED / SUCCESS
```

The full gate passed G-3 salience/summary and wiring regressions, frozen G-2 contract/runtime regressions, inherited G-1 regressions and authoritative C1 Finish/server-finalization regressions.

No runtime file changed after `dab45baf...` before release review; later commits before PR creation were canonical documentation only.

---

# 4. Fresh G-3 release-readiness review

Fresh remote main remained:

```text
9cfad82d1258a44e71080e0aa4d6d644e581cfbf
```

Reviewed pre-PR branch head:

```text
43bfc98a47d38f35b2b2e1365d702e9d6ec554b2
```

Compare `main → 43bfc98...`:

```text
status: ahead
ahead_by: 13
behind_by: 0
merge_base: exactly current main
changed_files: 10
```

The delta contains only expected G-3 runtime/test/workflow/canonical files. No PR-1/PR-2, physiotherapy or RF leakage was found.

Release PR:

```text
PR #70
feat: release G-3 guidance salience and longitudinal patient summary
base: main
head: feat/module01-g3-guidance-salience-longitudinal-summary-2026-09-01
state: OPEN
mergeable: YES at reviewed head
MERGE HOLD: YES
```

Exact PR-head checks at `43bfc98...` all passed:

```text
G3 guidance salience longitudinal summary  run 33538371613  SUCCESS
G2 evidence guidance runtime               run 33538371747  SUCCESS
G1 progressive guidance foundation         run 33538371860  SUCCESS
Baseline finalization integrity            run 33538371642  SUCCESS
```

The present commit is documentation-only and does not change runtime semantics.

---

# 5. Newly accepted product-owner priority — server-authoritative patient visits across devices

The product owner works across three computers and requires:

```text
CLINICAL DATA SAVED/UPDATED ON ONE DEVICE
→ AVAILABLE FROM THE SAME PROTECTED PATIENT RECORD ON THE OTHER DEVICES
```

Inspection of the current runtime shows this is only partially achieved today:

- PostgreSQL `clinical_patients`, `clinical_encounters` and `clinical_lab_snapshots` already exist and are protected;
- server draft/completed/amended encounters already exist;
- `localStorage` still owns the browser working case list, active case identity and module cache;
- draft server synchronization is currently tied primarily to explicit Save actions;
- server `updated_at` exists but the current PUT contract has no optimistic-concurrency guard against a stale device overwriting a newer device state.

Therefore the next bounded production-workflow slice after G-3 release must design/implement **server-authoritative cross-device encounter continuity**, not merely rename UI labels.

Required direction:

```text
patient_id + server encounter_id = authoritative identity
server draft = cross-device draft truth
localStorage = disposable working cache only
open patient/encounter = hydrate from server
meaningful save/autosave = persist complete current encounter to server
device B reload/open = see latest server state
stale write = conflict, never silent overwrite
```

Preferred first concurrency seam: reuse existing encounter `updated_at` as a version/expected-update token or add an equivalent explicit revision contract; stale PUT should fail closed with a conflict rather than last-writer-wins silently.

---

# 6. Newly accepted product-owner priority — retire pilot-case shell

The product owner also requires normal real-patient/real-visit workflow rather than a visible `PILOT CASE 1/5` shell.

Current runtime inspection confirms pilot semantics are still embedded in:

```text
PILOT_TARGET = 5
PILOT CASE n/5 pill
PILOT-nnn local case IDs
`Νέο Case` / `Cases` local-browser workflow
first_core_baseline_encounter_for_patient UI
pilot_completion local object
Finish text/alerts referring to pilot case
privacy/banner copy describing local pilot behavior
```

Next production-workflow slice must move the clinician-facing identity to:

```text
Protected Patient
→ one or more server-backed Encounters / Επισκέψεις
```

and remove pilot-specific presentation/identity from ordinary clinical use while preserving any audit/baseline methodology metadata only where it is still methodologically required in the background.

Important privacy boundary: retiring the pilot shell does not by itself authorize unsafe identifiable-data handling. Real clinical records remain inside the protected server route/storage boundary; broader audit-log/retention/GDPR work remains explicit before whole-service privacy claims.

---

# 7. Status matrix

```text
G-2 PRODUCTION-SMOKE-VERIFIED                 YES
G-3 DESIGN                                    COMPLETE
G-3 IMPLEMENTED                               YES
G-3 TESTED                                    YES
G-3 RELEASE-READINESS REVIEW                  PASS
G-3 RELEASE PR                                #70 OPEN
G-3 MERGE AUTHORITY                           NO
G-3 MERGED                                    NO
G-3 DEPLOYED                                  NO
G-3 PRODUCTION-SMOKE-VERIFIED                 NO
CROSS-DEVICE SERVER-AUTHORITATIVE WORKFLOW    NEXT DESIGN PRIORITY
PILOT-SHELL RETIREMENT                        NEXT DESIGN PRIORITY
PR-1 HEIDI                                    NOT IMPLEMENTED
PR-2 INLINE REVIEW/POPULATION                 NOT IMPLEMENTED
FORMAL 5-CASE SYSTEM-ASSISTED PILOT           NOT STARTED
MODULE 01 CLOSED                              NO
```

---

# 8. Exact next action / STOP gate

The explicitly authorized G-3 release-readiness review and PR opening are complete.

**STOP before G-3 merge/deploy.**

Only after a separate explicit product-owner merge instruction may a future session perform:

```text
fresh verify main
→ fresh six-canonical bootstrap
→ fetch exact PR #70 head
→ confirm no branch/base drift
→ confirm required checks remain green
→ inspect any review threads/comments
→ if clean, squash-merge with expected exact head
→ allow normal Render auto-deploy
→ verify deploy identity/state
→ production smoke
→ canonical release closeout
```

After G-3 release closeout, the next design slice should be the patient-centric/server-authoritative production workspace described in sections 5–6, before starting formal real-patient pilot collection.
