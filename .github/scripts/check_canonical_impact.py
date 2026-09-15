from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

START = '<!-- canonical-impact:start -->'
END = '<!-- canonical-impact:end -->'
REQUIRED = [
    'release_affecting',
    'checkpoint_stage',
    'root_current',
    'slice_plan',
    'todo',
    'clinical_excellence_plan',
    'workstream_current',
    'workstream_current_path',
    'changelog',
    'reason',
]
ALLOWED = {
    'release_affecting': {'yes', 'no'},
    'checkpoint_stage': {
        'design', 'implementation', 'implementation_tested', 'release_hold',
        'post_merge', 'post_deploy', 'post_smoke', 'governance', 'docs_only', 'other',
    },
    'root_current': {'update', 'none'},
    'slice_plan': {'update', 'none'},
    'todo': {'update', 'none'},
    'clinical_excellence_plan': {'update', 'none'},
    'workstream_current': {'update', 'not_applicable'},
    'changelog': {'update', 'defer_until_completion', 'none'},
}
CANONICAL_FILES = {
    'root_current': 'CURRENT_OPERATIONAL.md',
    'slice_plan': 'SLICE_PLAN_CURRENT.md',
    'todo': 'TODO.md',
    'clinical_excellence_plan': 'CLINICAL_EXCELLENCE_PLAN.md',
}


class GuardError(ValueError):
    pass


def parse_declaration(body: str) -> dict[str, str]:
    if body.count(START) != 1 or body.count(END) != 1:
        raise GuardError('PR body must contain exactly one Canonical Impact Declaration block')
    before, rest = body.split(START, 1)
    block, after = rest.split(END, 1)
    del before, after
    data: dict[str, str] = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if ':' not in line:
            raise GuardError(f'invalid declaration line: {line!r}')
        key, value = line.split(':', 1)
        key, value = key.strip(), value.strip()
        if key in data:
            raise GuardError(f'duplicate declaration key: {key}')
        data[key] = value
    missing = [key for key in REQUIRED if key not in data]
    extra = sorted(set(data) - set(REQUIRED))
    if missing:
        raise GuardError(f'missing declaration keys: {missing}')
    if extra:
        raise GuardError(f'unknown declaration keys: {extra}')
    for key, allowed in ALLOWED.items():
        if data[key] not in allowed:
            raise GuardError(f'{key} must be one of {sorted(allowed)}, got {data[key]!r}')
    reason = data['reason'].strip()
    if len(reason) < 12 or 'REPLACE_ME' in reason.upper():
        raise GuardError('reason must be a real concise explanation, not the template placeholder')
    return data


def is_changelog(path: str) -> bool:
    name = Path(path).name
    return path == 'osteoporosis-change-log.md' or name in {'CHANGELOG.md', 'PRODUCT_CHANGELOG.md'}


def validate(data: dict[str, str], changed_files: set[str]) -> None:
    errors: list[str] = []

    for key, path in CANONICAL_FILES.items():
        declared_update = data[key] == 'update'
        actually_changed = path in changed_files
        if declared_update != actually_changed:
            errors.append(
                f'{key}: declaration={data[key]!r} but {path} changed={actually_changed}'
            )

    current_mode = data['workstream_current']
    current_path = data['workstream_current_path']
    if current_mode == 'update':
        if current_path == 'not_applicable':
            errors.append('workstream_current=update requires a real workstream_current_path')
        elif not (
            current_path == 'CURRENT_OPERATIONAL.md' or current_path.endswith('/CURRENT.md')
        ):
            errors.append(
                'workstream_current_path must be CURRENT_OPERATIONAL.md or end with /CURRENT.md'
            )
        elif current_path not in changed_files:
            errors.append(
                f'workstream current path {current_path!r} is declared update but is absent from PR diff'
            )
    elif current_path != 'not_applicable':
        errors.append(
            'workstream_current=not_applicable requires workstream_current_path=not_applicable'
        )

    changed_changelogs = sorted(path for path in changed_files if is_changelog(path))
    if data['changelog'] == 'update' and not changed_changelogs:
        errors.append('changelog=update but no recognized changelog file changes')
    if data['changelog'] != 'update' and changed_changelogs:
        errors.append(
            f'changelog={data["changelog"]!r} but changelog files changed: {changed_changelogs}'
        )

    if data['release_affecting'] == 'yes' and current_mode != 'update':
        errors.append(
            'release_affecting=yes requires workstream_current=update in the same PR checkpoint'
        )

    if data['release_affecting'] == 'no' and data['checkpoint_stage'] in {
        'post_merge', 'post_deploy', 'post_smoke'
    }:
        errors.append(
            f'checkpoint_stage={data["checkpoint_stage"]} is a release lifecycle stage and requires release_affecting=yes'
        )

    if errors:
        raise GuardError('\n'.join(f'- {error}' for error in errors))


def run_self_test() -> None:
    base = {
        'release_affecting': 'no',
        'checkpoint_stage': 'governance',
        'root_current': 'none',
        'slice_plan': 'none',
        'todo': 'none',
        'clinical_excellence_plan': 'none',
        'workstream_current': 'not_applicable',
        'workstream_current_path': 'not_applicable',
        'changelog': 'none',
        'reason': 'Governance-only change with no runtime release effect.',
    }
    validate(base, {'AGENTS.md'})

    release = dict(base)
    release.update({
        'release_affecting': 'yes',
        'checkpoint_stage': 'release_hold',
        'workstream_current': 'update',
        'workstream_current_path': 'commercial_products/example/CURRENT.md',
        'reason': 'Runtime candidate enters release hold with durable workstream state.',
    })
    validate(release, {'runtime.py', 'commercial_products/example/CURRENT.md'})

    bad_release = dict(release)
    bad_release['workstream_current'] = 'not_applicable'
    bad_release['workstream_current_path'] = 'not_applicable'
    try:
        validate(bad_release, {'runtime.py'})
    except GuardError:
        pass
    else:
        raise AssertionError('release-affecting PR without current update should fail')

    bad_root = dict(base)
    bad_root['root_current'] = 'update'
    try:
        validate(bad_root, {'AGENTS.md'})
    except GuardError:
        pass
    else:
        raise AssertionError('declared root update without file change should fail')

    body = f'''\n{START}\nrelease_affecting: no\ncheckpoint_stage: governance\nroot_current: none\nslice_plan: none\ntodo: none\nclinical_excellence_plan: none\nworkstream_current: not_applicable\nworkstream_current_path: not_applicable\nchangelog: none\nreason: Governance declaration parser self-test.\n{END}\n'''
    parsed = parse_declaration(body)
    assert parsed['checkpoint_stage'] == 'governance'
    print('canonical-impact self-test PASS')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--event')
    parser.add_argument('--changed-files')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if not args.event or not args.changed_files:
        parser.error('--event and --changed-files are required unless --self-test is used')

    event = json.loads(Path(args.event).read_text(encoding='utf-8'))
    body = ((event.get('pull_request') or {}).get('body') or '')
    changed = {
        line.strip()
        for line in Path(args.changed_files).read_text(encoding='utf-8').splitlines()
        if line.strip()
    }
    try:
        data = parse_declaration(body)
        validate(data, changed)
    except GuardError as exc:
        print('CANONICAL IMPACT GUARD FAIL', file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    print('CANONICAL IMPACT GUARD PASS')
    print(f"checkpoint_stage={data['checkpoint_stage']}")
    print(f"release_affecting={data['release_affecting']}")
    print(f"changed_files={len(changed)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
