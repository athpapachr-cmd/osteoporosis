# D1.1 review execution — exact prompt and bounded replacement

> DATE: 2026-10-03 Asia/Nicosia.
> This is prompt/evidence provenance, not a new independent review or a runner configuration.
> General execution rules live in `PROCEDURES.md` P5.1; the active request is `D1_1_PRECODE_REVIEW_REQUEST.md`.

## Exact original coordinator-provided chat prompt

From referenced conversation `6abebc7c-018c-83eb-bd23-273f093e2b44`, assistant turn responding to “Ωραία βάλε το GESY FEASIBILITY SPIKE και ξεκίνα το D1.1 → Visit Brief”:

> Είσαι ο fresh independent READ-ONLY pre-code reviewer για το Cockpit D1.1 Global Appointment Context στο `athpapachr-cmd/osteoporosis`. Fresh-bootstrap από current remote main και ακολούθησε `AGENTS.md` + `PROCEDURES.md`. Κατανάλωσε πλήρως το `cockpit/reviews/D1_1_PRECODE_REVIEW_REQUEST.md`. Exact design target: `cockpit/D1_1_GLOBAL_APPOINTMENT_CONTEXT_DESIGN.md`, expected blob `2b438c949e30d00a1d11d9d1431d766954320f15`, branch `design/cockpit-d1-1-global-context-2026-10-02`, contextual head `9b11f722c485247754fa7219e9b0f0d9167f8734`. Παραμένεις READ-ONLY. Μην υλοποιήσεις, μην διορθώσεις code, μην κάνεις merge/deploy. Επέστρεψε ακριβώς το verdict/output που ζητά το review request και STOP.

The original on-branch request blob was `7cd035537d53d93c8063e8ceba9439dd1704faa0`; archive copy preserves those bytes. On fresh main `5801fbe20cf760eaa9c67e8aa03bc52c5928393b`, the request/design were absent; CURRENT pointed to their design branch. An unqualified “read request on main” can therefore waste time searching. The replacement explicitly reads the request from its named branch.

## Findings on duration

The original prompt does say READ-ONLY and STOP **after** output. It lacks an explicit definition of sufficient evidence. Seven broad questions ask the reviewer to establish that all material boundaries are preserved, while the output asks for COMPLETE coverage and NO ADDITIONAL MATERIAL FINDING. Without a finite question→evidence map, those terms can be misread as requiring a search for every possible counterexample. Fresh bootstrap traverses six canonicals; following their historical links or cross-repository pointers without a concrete question can expand the search indefinitely. Prior P5 bounded the number of reviews in a chain, not the evidence search within one review.

The Product Owner reports the Work-mode review continued for over seven hours until interrupted. App chat `6ac00978-8000-83ed-8ccc-a4ada5e3a252` shows the same prompt and the later BLOCK handback, but exposes no intermediate tool/reasoning trace. A separate Codex run with the same prompt, task `01a0fe25-769a-7843-9b4c-e710a08f4615`, completed in about four minutes. This contrast shows the prompt is not by itself a proven seven-hour cause. The long run could involve repeated search, a hanging tool or execution-platform delay; none is established from the available trace.

## Ready-to-use closure prompt

Είσαι fresh independent READ-ONLY reviewer για το D1.1 στο `athpapachr-cmd/osteoporosis`. Κάνε μία φορά το υποχρεωτικό bootstrap από fresh remote main / AGENTS.md και διάβασε το closure request **από branch `design/cockpit-d1-1-global-context-2026-10-02`**, path `cockpit/reviews/D1_1_PRECODE_REVIEW_REQUEST.md`. Exact design blob: `261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33`. Έλεγξε μόνο τη διόρθωση των D1.1-PRE-01/PRE-02 και τις επηρεαζόμενες συνολικές εγγυήσεις Q1–Q3. Για κάθε ερώτημα χρησιμοποίησε τα κατονομασμένα αρχεία/συναρτήσεις, κατέγραψε τη συγκεκριμένη αιτία αν χρειαστεί άμεσα εξαρτώμενο αρχείο και σταμάτα μόλις τα ερωτήματα έχουν απάντηση ή αποδειχθεί ουσιώδες BLOCK. Το «καμία πρόσθετη διαπίστωση» αφορά μόνο αυτό το δηλωμένο πεδίο, όχι έρευνα ολόκληρου του repository. Αν λείπει αναγκαίο στοιχείο ή αλλάζει ουσιαστικά το πεδίο, επέστρεψε BLOCK/PARTIAL με το ακριβές κενό. Μην υλοποιήσεις, διορθώσεις code, κάνεις merge/deploy ή ανοίξεις άλλους reviewers. Επέστρεψε το ζητούμενο verdict και STOP.

A runner timeout may guard against hangs, but it would not repair the missing evidence-completion rule. This session has not launched the closure reviewer.
