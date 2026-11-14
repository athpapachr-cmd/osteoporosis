# Compact calendar reason preview — independent R1 review

- Candidate `420bba4`, base `3741cce1008768a4bd1112a99ce6e57ca215d6d8`: independent R1 PASS, P0:P1:P2 0:0:0 within declared scope.
- Source reason is escaped, normalized for the single-line preview and retained in full in details. Empty, long and untrusted inputs, keyboard/browser density and existing filter, classification, duplicate-ID and auth paths were covered.
- Independent synthetic browser/calendar suite 26/26 PASS at desktop/mobile; no production or real-patient access.
- Subsequent CSS-only readability adjustment (11px text/13px line height and darker color) passed the focused busy-week browser checks 2/2; runtime behavior and tests otherwise unchanged.
