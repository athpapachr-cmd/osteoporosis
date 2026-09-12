# SLICE_PLAN_CURRENT.md — CYPRUS / GESY OA JURISDICTION OVERLAY V1

> **STATUS:** ACTIVE — PRIMARY-SOURCE AUDIT + DESIGN ONLY.
> **Branch:** `design/physio-cy-gesy-oa-overlay-v1-2026-09-12`.
> **Bootstrap main:** `a2fa27c7ff26d1dd22cd6f726656ca0532daab75`.
> **Writer:** bounded design branch above.
> **Runtime/UI implementation:** NOT AUTHORIZED.
> **Diagnosis vertical:** Knee Osteoarthritis only.

## 1. Problem

The product already has a reviewed international Knee-OA evidence contract and a dormant `CY_GESY` context seam. Cyprus now has an HIO-adapted OA guideline based on NICE NG226, plus separate GeSY access/reimbursement rules. Treating all of these as one source of "local evidence" would corrupt both clinical truth and product simplicity.

The design must support:

```text
international evidence core
+
optional jurisdiction clinical overlay
+
separately classified local-system/admin policy
```

without silently hybridising them.

## 2. Audit rules

For every product-relevant local recommendation preserve at minimum:

- local recommendation ID;
- identifying short verbatim excerpt and exact source locator;
- faithful normalized local position;
- source/version/publication metadata;
- scope/intervention;
- direction;
- strength/certainty only when explicitly available from source conventions;
- relationship to NICE;
- whether Cyprus changes/adds to NICE;
- stated rationale when available;
- rationale class: clinical/evidence-based / resource-feasibility / administrative / reimbursement / mixed / unclear;
- operational publication status;
- GeSY IT-system integration status separately;
- reviewed-on date.

Do not infer unstated cost or reimbursement motives. Use `unclear` where the source does not state the rationale.

Because the public HIO-linked OA PDF still carries draft metadata while the May-2026 HIO announcement says adaptation is completed and implemented, preserve both facts. Do not invent a final document version/date that HIO has not exposed clearly.

## 3. Comparison authority

Compare verified Cyprus positions only against:

`clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_contract_v1.yaml`

Current international evidence resolution remains authoritative for the core and forbids source voting/arithmetic scoring.

A local position may be:

```text
agreement
local_difference
local_addition
administrative_only
reimbursement_or_access_only
status_unknown
not_comparable
```

None of these states may overwrite the international evidence state.

## 4. UX design target

Routine surface must remain minimal.

- local agreement: normally silent;
- genuine local difference: small contextual jurisdiction cue, detail on demand;
- local admin/resource rule: separate operational information, never styled as clinical efficacy evidence;
- local status unknown/planned: explicit uncertainty/planned wording;
- full source tables, reimbursement mechanics and guideline bureaucracy stay out of routine referral creation.

## 5. Explicit non-goals

- no live UI/runtime change;
- no evidence-state mutation;
- no second diagnosis;
- no country selector;
- no Greece/England content;
- no billing/analytics/persistence;
- no patient data;
- no automatic literature ingestion;
- no automatic local recommendation activation;
- no merging of v5 Product Owner review work into this slice.

## 6. Acceptance criteria

The slice is design-complete only if:

1. official Cyprus/HIO clinical guidance is auditable recommendation-by-recommendation for current product-relevant domains;
2. GeSY administrative/reimbursement/access rules are separately classified;
3. NICE relationship is explicit for every material local difference/addition;
4. ambiguous rationale/status stays `unclear` rather than inferred;
5. the schema is jurisdiction-generic enough for future profiles without implementing them;
6. UX preserves progressive disclosure and clinician autonomy;
7. final recommendation is bounded to `NO CHANGE`, `LOCAL INFO ONLY`, or `IMPLEMENT JURISDICTION OVERLAY V1` with evidence;
8. no runtime/UI code is changed.

## 7. REPLAN triggers

Replan if:

- an authoritative final Cyprus source contradicts the currently linked HIO artifact materially;
- a GeSY rule is found to be enforced differently from its published status;
- the overlay would require changing core evidence semantics rather than representing local context separately;
- a routine-surface addition cannot demonstrate workflow or receiver utility.

## 8. Exact next action

Finish the primary-source audit and freeze source-audit, difference-matrix, machine-schema and UX-policy artifacts for Product Owner review.