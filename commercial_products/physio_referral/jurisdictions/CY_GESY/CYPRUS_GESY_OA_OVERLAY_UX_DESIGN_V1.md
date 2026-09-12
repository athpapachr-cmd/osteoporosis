# CYPRUS_GESY_OA_OVERLAY_UX_DESIGN_V1

> **STATUS:** PROPOSED UX / PRODUCT BEHAVIOR — REVIEW REQUIRED BEFORE IMPLEMENTATION
> **Reviewed on:** 2026-09-12
> **Runtime authority:** NONE

## 1. Product objective

Make verified Cyprus/GeSY context available **without turning the routine referral screen into a guideline dashboard**.

The design follows:

```text
power underneath
+
minimal decisions on routine surface
+
progressive disclosure
+
visible disagreement when it genuinely matters
```

Permanent utility gate:

```text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING
```

## 2. Local agreement

Example: therapeutic exercise, walking aids, device restrictions.

### Routine behavior

**Show nothing extra.**

Do not add a green Cyprus badge merely to announce that another source agrees. Agreement is not a user task.

### On-demand evidence detail

If the clinician already opens an evidence sheet, a small jurisdiction section may state:

```text
Κύπρος · ΓεΣΥ
Η τοπική κατευθυντήρια θέση είναι συμβατή με τη διεθνή θέση που εμφανίζεται εδώ.
```

Source and recommendation locator may be expanded on demand.

### Never do

- no extra recommendation weight;
- no stronger colour because Cyprus agrees;
- no arithmetic increase in confidence;
- no extra tap in routine referral creation.

## 3. Local difference

Examples:

- acupuncture: international product state is mixed; Cyprus position is against;
- a future electrotherapy item: Cyprus conditional-for short-term pain adjunct, while NICE is against.

### Routine behavior

Do not overwrite the international evidence badge.

If the item is already visible/selected/being inspected and the difference could matter, show **one restrained jurisdiction cue**, for example:

```text
Κύπρος · διαφέρει
```

The cue is informational and does not select/deselect anything.

### Expanded evidence sheet

Present two clearly separate rows:

```text
Διεθνής τεκμηρίωση
[existing product state + source-specific positions]

Κύπρος · ΓεΣΥ
[verified local position]
```

If the local position follows one side of an international disagreement, say so explicitly rather than implying that the conflict disappeared.

Example:

```text
Διεθνής θέση: Οι οδηγίες διαφέρουν
Κύπρος · ΓεΣΥ: Δεν συνιστάται
```

### Clinician autonomy

A local difference must not:

- auto-select an intervention;
- auto-remove a clinician-selected intervention;
- rewrite referral prose without explicit selection;
- block export unless an independently reviewed safety rule already owns that behavior.

## 4. Local administrative / resource / reimbursement rule

Examples:

- GeSY referral required for physiotherapy access;
- eligible diagnosis requirement;
- covered-session limits;
- PHYS02 documentation requirements;
- provider unit caps.

### Presentation rule

These belong to an **operational GeSY information layer**, never an evidence bubble.

Use language such as:

```text
Πληροφορία ΓεΣΥ
```

not:

```text
Ισχυρή σύσταση
Κλινική τεκμηρίωση
```

### Routine behavior

Usually silent if the product already satisfies the operational requirement.

Show an operational note only when the clinician has a decision to make or an export would otherwise fail for a GeSY-specific reason.

Example legitimate future use:

```text
Το παραπεμπτικό ΓεΣΥ απαιτεί επιλέξιμη διάγνωση πριν από την εξαγωγή.
```

That is a workflow constraint, not evidence that the diagnosis or treatment is clinically stronger.

### Explicit ban

Do not display provider payment mechanics such as monthly unit caps on the routine referral surface.

## 5. Local status unknown or planned

Example: HIO announcement states future integration of the OA guideline into the GeSY information system, but active IT enforcement has not been verified.

### Required wording

Use a neutral state such as:

```text
Κατάσταση εφαρμογής στο Σύστημα Πληροφορικής: δεν έχει επιβεβαιωθεί ως ενεργή.
```

or, when the source explicitly says future implementation:

```text
Προγραμματισμένη ενσωμάτωση στο Σύστημα Πληροφορικής του ΓεΣΥ.
```

### Never do

- do not say `Το ΓεΣΥ απαιτεί...` from a merely planned integration;
- do not block the clinician based on planned rules;
- do not infer enforcement from guideline publication;
- do not infer reimbursement from clinical wording.

## 6. Public-source version inconsistency

The current HIO situation requires one additional UX/data rule:

```text
implementation announcement says active
+
linked public recommendation PDF still says draft
→ preserve metadata conflict
```

If source metadata is exposed, use a restrained note in deep detail only:

```text
Ο ΟΑΥ ανακοίνωσε εφαρμογή της οδηγίας τον Μάιο 2026. Το δημόσια συνδεδεμένο κείμενο εξακολουθεί να φέρει μεταδεδομένα προσχεδίου Δεκεμβρίου 2025.
```

This is provenance information, not routine referral content.

## 7. What should NOT be shown routinely

Do **not** add the following to the normal Knee-OA referral surface:

- a country/jurisdiction selector;
- `Κύπρος · ΓεΣΥ` badges beside every intervention;
- full source tables or recommendation IDs;
- every local adaptation just because it exists;
- radiofrequency nerve ablation;
- podiatry referral;
- glucosamine/chondroitin;
- hyaluronan;
- PRP;
- corticosteroid-injection guidance;
- imaging algorithms;
- electrotherapy controls when electrotherapy is not already a justified product item;
- GeSY session-count/reimbursement tables;
- provider monthly unit caps;
- reimbursement/documentation mechanics that do not change the referring clinician's immediate task;
- planned GeSY IT integration as if it were active enforcement;
- inferred cost/resource motives;
- a generic `local guideline agrees` confirmation on every routine item;
- a recommendation feed;
- another checkbox section called `GeSY`.

The absence of these items from the routine surface does **not** mean the machine layer cannot represent them.

## 8. Existing Knee-OA UI/evidence-state decision

### International evidence states

**No current state needs to change.**

The Cyprus audit does not invalidate the reviewed international contract.

### Routine main screen

**No jurisdiction-driven change needed.**

The current defaults — individualized active rehabilitation, therapeutic exercise, progressive strengthening and education/self-management — remain compatible with verified Cyprus guidance.

### Existing evidence detail

A future overlay can add a small local-position section only for an item the clinician is already inspecting, especially where Cyprus differs or chooses one side of international conflict.

### Referral prose

No Cyprus clinical recommendation identified in this audit requires adding jurisdiction language to the referral prose.

Do not write `σύμφωνα με το ΓεΣΥ` into the referral unless a later receiver/workflow test proves such wording is useful.

## 9. Proposed first-runtime behavior if overlay is later authorized

The smallest useful runtime is deliberately boring:

```text
account/profile configured as CY_GESY
→ core product behaves exactly as today
→ local agreement stays silent
→ relevant local difference can add one detail-level cue
→ admin rule appears only at the workflow seam it actually governs
→ source/status detail remains behind progressive disclosure
```

No new routine tap is required.

## 10. Bounded recommendation

`IMPLEMENT JURISDICTION OVERLAY V1`

Meaning:

- implement the **separate machine/provenance layer** after Product Owner review;
- activate `CY_GESY` only from explicit configuration;
- initially keep routine Knee-OA UI and international evidence states unchanged;
- expose local differences progressively in evidence detail;
- keep GeSY administrative/reimbursement information in a distinct operational class;
- do not add Greece/England profiles until real market/workflow need exists.

### Why not `NO CHANGE`

Because verified clinical differences exist and the target market is GeSY. Leaving all local context as unstructured documentation would invite later silent hybridisation.

### Why not `LOCAL INFO ONLY`

Because the distinction is not merely prose. The system needs machine-level separation among international state, local clinical position, administrative policy and operational status so future UI behavior can remain deterministic and honest.

### Why this does not mean UI expansion

The overlay is primarily an **under-the-surface truth model**. Its purpose is to prevent incorrect product behavior, not to produce more buttons.
