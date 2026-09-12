# Market / jurisdiction strategy — Physiotherapy Referral

Architecture:

```text
international clinical-evidence core
+
optional jurisdiction / local-system overlay
```

Initial commercial audience: clinicians working in Cyprus / GeSY.

Current local-profile seam:

```text
CY_GESY
Κύπρος · ΓεΣΥ
```

The local profile is not permission to convert reimbursement/resource policy into stronger clinical-efficacy evidence. Cyprus/GeSY guidance must be audited recommendation-by-recommendation before it changes defaults, evidence states, suggestions or referral prose.

The product must remain portable. Greece and other jurisdictions can be added only when real workflow/market need is demonstrated. England should not receive a localization simply because the architecture permits one; NHS self-referral / First Contact Physiotherapy pathways may materially reduce the doctor-to-physio referral use case.

No country selector, location inference, billing localization or dormant country content is required in the Knee-OA v1 production slice.
