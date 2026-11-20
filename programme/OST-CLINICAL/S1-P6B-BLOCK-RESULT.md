# S1 Fracture / Fragility Semantics — Independent P6B result

> **STATUS:** BLOCK — ONE MATERIAL RESIDUAL.
> **Review mode:** fresh independent READ-ONLY exact-head post-code review.
> **Reviewed PR:** #123.
> **Reviewed head:** `f4fe36bcfbd0ca475768844387536ec71e2fd38d`.
> **Base:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Target drift:** NO.

## Verified closures

The independent review verified:
- D1 closed — generic fracture existence no longer auto-writes positive prior fragility;
- D2 closed — generic fracture no longer overwrites last-fragility compatibility fields;
- D3 closed — legacy prior=true no longer synthesizes a structured event/UUID on render;
- D4 closed — G3 no longer equates generic event existence/stale `event.fragility` with confirmed fragility;
- D5 closed — encounter archetype alone no longer creates current fragility truth;
- D6 closed — historical vertebral fragility requires explicit normalized `low_trauma === "yes"`.

Preserved behavior, scope discipline and CI ownership were otherwise PASS.

## Blocking residual

`S1-P6B-R1 — RAW LOW_TRAUMA PRESERVATION ACROSS LOAD → RENDER → SAVE`

Current render logic represents only exact canonical select values:

```text
yes
no
uncertain
""
```

A preserved raw legacy/noncanonical value such as:

```text
unknown
 YES 
Yes
yes 
other unrecognised raw value
```

may not match any `<option>` exactly. The UI therefore renders the empty option, and an ordinary save can write `""` back into the existing event.

That creates a silent no-edit rewrite of source evidence.

Material example:

```text
raw low_trauma = " YES "
→ semantic normalizer interprets as confirmed yes
→ select renders blank because raw string does not exactly equal "yes"
→ ordinary save writes ""
→ source evidence silently erased / meaning downgraded
```

## Review verdict

```text
D1–D6: PASS
BACKWARD COMPATIBILITY: BLOCKED BY R1
OTHER MATERIAL RESIDUALS: NONE FOUND
P6B VERDICT: BLOCK
```

## Review evidence limitation

The reviewer could not perform a second local checkout because the local execution container could not resolve github.com. The reviewer independently inspected exact source/diff and verified actual GitHub Actions job steps/logs rather than claiming local execution.

## Required programme response

Do not redesign S1.

Perform one bounded residual correction that preserves exact raw `low_trauma` across a no-edit load/render/save cycle while maintaining normalized clinical interpretation and allowing explicit clinician edits to write canonical values.

Then run a fresh exact-head delta+cumulative independent P6B review.

No merge/deploy/smoke is authorized.