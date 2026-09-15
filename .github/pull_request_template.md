## Canonical impact declaration

<!-- Keep this block. The Canonical impact guard parses it. -->
<!-- Replace REPLACE_ME and adjust every value to match the actual PR. -->
<!-- canonical-impact:start -->
release_affecting: no
checkpoint_stage: other
root_current: none
slice_plan: none
todo: none
clinical_excellence_plan: none
workstream_current: not_applicable
workstream_current_path: not_applicable
changelog: none
reason: REPLACE_ME
<!-- canonical-impact:end -->

Allowed values:
- `release_affecting`: `yes | no`
- `checkpoint_stage`: `design | implementation | implementation_tested | release_hold | post_merge | post_deploy | post_smoke | governance | docs_only | other`
- `root_current`, `slice_plan`, `todo`, `clinical_excellence_plan`: `update | none`
- `workstream_current`: `update | not_applicable`
- `workstream_current_path`: exact repo path to `CURRENT_OPERATIONAL.md` or a workstream `/CURRENT.md`, otherwise `not_applicable`
- `changelog`: `update | defer_until_completion | none`

For `release_affecting: yes`, the PR must set `workstream_current: update` and include the declared current-state path in the diff.
