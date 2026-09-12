# SLICE_PLAN_CURRENT.md — CYPRUS / GESY OA JURISDICTION OVERLAY V1 review HOLD

> **STATUS:** DESIGN-COMPLETE / PRODUCT OWNER REVIEW HOLD.
> **Branch:** `design/physio-cy-gesy-oa-overlay-v1-2026-09-12`.
> **Bootstrap main:** `a2fa27c7ff26d1dd22cd6f726656ca0532daab75`.
> **Writer:** NONE.
> **Runtime/UI implementation:** NOT AUTHORIZED.
> **Diagnosis vertical:** Knee Osteoarthritis only.

## 1. Frozen design problem

The product needs to represent real local clinical differences and GeSY operational rules without collapsing them into the international evidence core.

Frozen architecture:

```text
international evidence core
+
optional jurisdiction clinical overlay
+
separately classified local-system/admin policy
```

## 2. Completed artifacts

- `CYPRUS_GESY_OA_SOURCE_AUDIT_V1.md`
- `CYPRUS_GESY_OA_DIFFERENCE_MATRIX_V1.md`
- `JURISDICTION_OVERLAY_SCHEMA_V1.yaml`
- `CYPRUS_GESY_OA_OVERLAY_UX_DESIGN_V1.md`
- `CYPRUS_GESY_OA_OVERLAY_DESIGN_REVIEW_V1.md`

## 3. Frozen evidence findings

- current Cyprus/HIO OA guidance contains genuine differences/additions relative to NICE;
- electrotherapy is the clearest physiotherapy-adjacent genuine difference;
- manual therapy and acupuncture show why a local position must coexist with, not replace, an international mixed state;
- radiofrequency ablation, podiatry, glucosamine/chondroitin, hyaluronan, PRP and additional imaging detail are clinically relevant local content but are not automatically routine physio-referral content;
- GeSY access/reimbursement/documentation/provider-unit rules are operational policy, not clinical evidence;
- HIO guideline implementation is announced, but GeSY information-system integration is planned/not verified active;
- current linked OA PDF still has stale draft metadata, so final-document version identity remains explicitly imperfect.

## 4. Frozen product behavior

### Local agreement

Normally silent. Optional source detail only.

### Local difference

International evidence remains visible and unchanged. A restrained `Κύπρος · διαφέρει`-type cue may appear only when the item is already relevant/inspected, with separate international and Cyprus rows in detail.

### Administrative/resource/reimbursement rule

Separate `Πληροφορία ΓεΣΥ` operational layer. Never styled as evidence strength.

### Local status unknown/planned

Explicitly label as unknown/planned. Never infer active enforcement.

## 5. Routine-surface exclusions

Do not routinely show:

- jurisdiction selector or badges on every item;
- full source tables;
- radiofrequency ablation;
- podiatry;
- glucosamine/chondroitin;
- hyaluronan;
- PRP;
- injection guidance;
- imaging algorithms;
- electrotherapy controls merely because Cyprus mentions them;
- session/reimbursement tables;
- provider unit caps;
- planned IT integration as active;
- inferred cost motives;
- another checkbox section named GeSY.

## 6. Current live-product decision

```text
international evidence states    NO CHANGE
routine Knee-OA main UI           NO CHANGE from jurisdiction evidence
routine referral prose            NO CHANGE from jurisdiction evidence
patient persistence               NO CHANGE / none
second diagnosis                  NO
```

The already-tested v5 post-use candidate remains a separate Product Owner review matter and is not part of this overlay design.

## 7. Bounded recommendation

`IMPLEMENT JURISDICTION OVERLAY V1`

This means implement the separate machine/provenance capability **only after Product Owner review in a new bounded runtime slice**. It does not mean adding routine visible controls.

## 8. Exit / next gate

Design slice is complete and writer released.

Next legitimate action:

```text
Product Owner review
→ accept / modify / reject design
→ only if accepted: fresh runtime implementation slice
```

No runtime/UI PR, merge, deploy or activation is authorized by this design freeze.