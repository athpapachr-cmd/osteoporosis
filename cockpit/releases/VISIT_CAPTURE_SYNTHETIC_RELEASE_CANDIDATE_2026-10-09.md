# Visit Capture — staged release candidate and release HOLD

> **DATE:** 2026-10-09 Asia/Nicosia.
> **MODE:** RELEASE PREPARATION / DRAFT STACKED PR ONLY / NO MERGE OR DEPLOY AUTHORITY.
> **PRODUCT OWNER:** approved preparation of release candidate and related PR; the preceding explicit offer excluded merge/deploy.
> **STATUS:** IMPLEMENTED / FOCUSED TESTED / INDEPENDENT R2 REVIEW CLOSED / DRAFT RELEASE CANDIDATE / PRODUCTION MERGE+DEPLOY HOLD.
> **GOVERNING OWNERS:** `CURRENT_OPERATIONAL.md` for writer/now and `cockpit/CURRENT.md` for Cockpit release state.

## 1. Scope of the release candidate

This candidate provides a protected, patient-confirmed **Καταγραφή επίσκεψης** surface in Cockpit with one clinician Save and Snapshot / Visit Brief / Encounter Detail projections from a single normalized structured encounter.

The implementation:

- extends `clinical_patients.patient_id` and existing protected `clinical_encounters`;
- accepts `VisitCaptureCandidateV1` with source/provenance/comparison semantics and typed bounded optional external dependency;
- obtains a patient/context-bound server context that fails closed after switching or stale state;
- previews three deterministic human-readable summaries from one candidate, without saving three separate truth copies;
- persists a signed encounter only after explicit clinician Save;
- retains separate first-class pending items;
- server-rejects edits to signed Visit Capture encounters via the legacy generic PUT, pending a separate future amendment mechanism;
- uses only synthetic/manual test inputs in this first slice.

Not present: live Dia or Heidi/GESY access, an API connection to Dia, Gmail intake, Zadarma, autonomous record acceptance, automatic patient communications/booking or real-patient validation. The parent Visit Brief + independent Clinical Inbox core/live integration remains **separate and not implemented** by this candidate.

## 2. Source, branch and immutable evidence

| Item | Exact identity |
|---|---|
| Verified remote `main` at preparation | `045798dfa28612b268f16a99262c6ecc9ca4829d` |
| Unmerged parent design branch | `docs/cockpit-visit-brief-clinical-inbox-2026-10-07` @ `5bc2a505758de6b24e82667bb84580d144f96b9c` |
| Parent design draft | [PR #138](https://github.com/athpapachr-cmd/osteoporosis/pull/138), OPEN / DRAFT / UNMERGED |
| Implementation branch | `feat/cockpit-visit-capture-v1-2026-10-08` |
| Frozen corrected runtime/test head | `07f68a2adc4664a4ab050b3fa10a663de591b662` |
| Corrected `clinical_data.py` blob | `ea9e3d39c124136f4818094aec4086e49cbd0919` |
| Corrected `test_visit_capture.py` blob | `00adc97ec0c726cc4a063470cf7a77e6f134a711` |
| Focused CI | [run 37963002895](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/37963002895), SUCCESS / 20 focused+UI+Home tests / syntax / diff hygiene |
| Independent post-code closure | PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 0:0:0, P2-F2-01 CLOSED |
| Closure receipt | `cockpit/reviews/VISIT_CAPTURE_P2_F2_01_FOCUSED_CLOSURE_PASS_RECEIPT_2026-10-09.md`, blob `adfb02e736d0db9ed77e305504a0b4a94ade9cb6` |

Post-correction commits changed only canonical/current/review documentation; no runtime/test drift as of this candidate. Future release work must recheck these immutable blobs if branch heads move.

## 3. Stacked PR ancestry and release sequence

The implementation branch is descended directly from the parent design branch. Do not open an implementation PR against `main` while PR #138 is unmerged; that would include and potentially duplicate the parent design changes.

**Prepared release approach:**

1. Keep PR #138 (parent design) **OPEN/DRAFT**, without merging it.
2. Open the implementation candidate as a separate **DRAFT stacked PR**, base = `docs/cockpit-visit-brief-clinical-inbox-2026-10-07`, head = `feat/cockpit-visit-capture-v1-2026-10-08`.
3. Confirm the PR diff includes only the implementation/its checkpoints, and the `canonical-impact` declaration matches the diff. Inspect applicable CI without rerunning unchanged tests solely for reassurance.
4. **STOP** at the release preparation checkpoint. There is no authority to merge PR #138, merge the implementation, deploy, flip environment variables or process live patient data.
5. If a future release is explicitly authorized, first decide the design PR #138 merge independently, then re-target/reconcile the implementation PR to the new `main` and verify unchanged runtime ancestry/CI. Never assume the stacked PR base will re-target itself automatically.

Normal Render auto-deploy follows `main`. Any merge into `main` is therefore potentially deployment-affecting and is prohibited at this preparation step.

## 4. Material release HOLD — synthetic-only UI notice is not an enforceable production gate

**Source-grounded finding for release readiness, not a reopening of reviewed implementation F1–F6:** `main.py` includes the existing `build_clinical_router(engine)` unconditionally. Its new `/clinical/visit-capture/context`, `/preview`, and `/save` endpoints in `clinical_data.py` require the same `CLINICAL_DATA_KEY` as other clinical functions but have **no distinct server-side OFF-by-default activation switch**. The Cockpit Home links to the static capture surface unconditionally. The `Synthetic / manual first-code` UI banner warns the user but does **not** restrict actual writes.

**Consequence:** merging/deploying the current candidate to the real protected Cockpit would make Visit Capture writes accessible to any user who already passes clinical authentication, including with real patient IDs/content. That is beyond the present synthetic-only approval. Therefore:

```text
DRAFT PR PREPARATION = AUTHORIZED
PRODUCTION MERGE/DEPLOY OF THIS CANDIDATE AS-IS = HOLD
REAL-PATIENT VISIT CAPTURE ACTIVATION = NOT AUTHORIZED
```

**Smallest future release-enablement work**, subject to separate Product Owner authorization: add an OFF-by-default backend admission/feature gate for the Visit Capture routes (at least context/preview/save), with matching non-misleading UI visibility, and test both disabled default and explicitly enabled isolated/synthetic behavior. This is a **new bounded release-enablement code delta** if approved; its directly affected validation/review is separate from and must not gratuitously reopen the already-closed P2-F2-01 or original clinical fidelity review.

Alternatively, retain the branch as a draft candidate without production deployment until real identifiable-data qualification and release intent are explicitly granted. Do not describe a banner, password or synthetic fixture as a production safety gate.

## 5. Open live-activation gates

Before identifiable real-use:
- Dia-browser processing/processor/privacy suitability for patient information;
- Heidi processor/terms qualification;
- GESY session, browser access and processor boundaries;
- selected-tab identity scoping and cross-patient contamination handling in real Dia behavior;
- browser transient cache/log/clipboard handling;
- field-level production retention, clinician role/access, deletion/amendment process as applicable;
- qualified clinical usage and clinician workflow/smoke with explicit authority.

These are **not** defects in the reviewed synthetic/manual implementation and are not silently converted into PASS by CI.

## 6. Release acceptance / rollback boundaries

Future merge gate must separately establish:
- explicit Product Owner merge/deploy approval;
- appropriate PR ancestry and a clean applicable canonical-impact guard;
- verified source/head/tree and inherited targeted CI;
- production release fence or resolved real-use authorization before exposing a new clinical write path;
- deploy/health checkpoint and one bounded synthetic-only smoke plan with no real patients, followed by Product Owner feedback;
- pause/rollback path if the new feature is unexpectedly visible or allows writes.

No production deployment or rollback action is performed while preparing this document.

**Next exact action after draft PR publication:** hand back the stacked PR URL, current exact branch identity and the one concrete production admission blocker to the Product Owner for a **separate release-enablement / merge/deploy decision**. Review chain remains CLOSED; no new broad reassurance review.
