# Day Home agenda: correction evidence

The three exact-code R2 findings were addressed in the bounded projection:

1. Tomorrow surgery source is rendered independently when tomorrow appointment read is loading or fails. Dedicated real-browser test checks known surgery in both states.
2. Five-minute/manual/rollover asynchronous reads capture the focused opaque control, focus the loading state and restore the same key or a descriptive heading on success. They do not steal focus moved outside while loading; failure remains focusable. Dedicated browser test covers expiry and moved focus; full Cockpit browser exercises the real refresh wiring at three widths.
3. On list expansion, the >90-day group opens when it contains any of the four initially displayed rows, and can subsequently be explicitly collapsed. Synthetic tests cover five old rows and a patient with old and future obligations and exact links.

Node regressions and actual Chromium desktop/half/mobile plus dedicated two-width agenda lifecycle passed. No real patient/provider writes. One independent affected R2 closure remains.
