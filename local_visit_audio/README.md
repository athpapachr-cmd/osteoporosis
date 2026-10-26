# Local Visit Audio companion

This component keeps visit audio, Whisper output and Krikri classification on the clinician's FileVault-enabled Mac. It never has a Clinical Key, database credential or Render token.

Install the exact reviewed local dependencies:

```bash
cd "/Users/athanasios/Documents/ChatGPT/Cockpit Codex/osteoporosis-priority4"
python3 -m local_visit_audio.install
```

Then start the dedicated loopback companion:

```bash
python3 -m local_visit_audio.service
```

Open `http://127.0.0.1:8765` and use its button to enter Visit Capture. Do not open a real-patient recording until the post-code review, guarded release and clinician acceptance gates have completed.
