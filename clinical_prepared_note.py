"""Lossless prepared clinical note adapter. No model, SDK, network or persistence."""
import re
import unicodedata


def heading_key(value):
    return ''.join(c for c in unicodedata.normalize('NFD', value)
                   if not unicodedata.combining(c)).strip().rstrip(':').lower()


def heading(line):
    """Recognize presentation only; never classify clinical content from prose."""
    text = line.rstrip('\r\n')
    markdown = re.match(r'^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$', text)
    if markdown:
        title = markdown[1].strip().rstrip(':')
        return title, ''
    bold = re.match(r'^\s*(?:\*\*|__)(.+?)(?:\*\*|__)\s*:?[ \t]*(.*)$', text)
    if bold:
        return bold[1].strip().rstrip(':'), bold[2]
    plain = re.match(r'^\s*([^\n:]{1,160}):[ \t]*(.*)$', text)
    if plain and not re.match(r'^\s*[-*•\d]', text):
        return plain[1].strip(), plain[2]
    return None


def note_sections(text):
    """Ordered display projections; the complete original remains the authority."""
    sections = []
    title, body = '', []
    for line in text.splitlines(keepends=True):
        found = heading(line)
        if found:
            if title or body:
                sections.append((title, ''.join(body)))
            title, inline = found
            body = [inline + ('\n' if line.endswith('\n') else '')] if inline else []
        else:
            body.append(line)
    if title or body:
        sections.append((title, ''.join(body)))
    return sections


def note_reason(text):
    for title, body in note_sections(text):
        if heading_key(title) in {'λογος επισκεψης', 'λογος παρακολουθησης', 'reason for visit'}:
            value = body.strip()
            if 0 < len(value) <= 4000:
                return value
    return 'Κλινική σημείωση'


def note_tasks(text):
    """Only reserved unchecked lines in the explicit obligations section act."""
    tasks, current, active = [], [], False

    def flush():
        if current:
            excerpt = ''.join(current).strip()
            description = re.sub(r'^[-*]\s+\[ \]\s*', '', excerpt).strip()
            if not description or len(description) > 1000 or len(excerpt) > 2000:
                raise ValueError('task_length')
            tasks.append((description, excerpt))
            current.clear()

    for line in text.splitlines(keepends=True):
        found = heading(line)
        if found:
            key = heading_key(found[0])
            if (current and line[:1] in {' ', '\t'}
                    and not re.match(r'^\s*(?:#{1,6}\s|\*\*|__)', line)
                    and not key.startswith(('προτασεις ai', 'ai suggestions'))):
                # A plain indented label may be a condition or a new section.
                # Refuse ambiguity rather than detach conditions or create tasks.
                raise ValueError('ambiguous_task_heading')
            flush()
            active = key == 'υποχρεωσεις'
            # Inline prose remains note content; only standalone checkbox rows act.
            continue
        if not active:
            continue
        if re.match(r'^\s*[-*]\s+\[[ xX]\]', line):
            flush()
            if re.match(r'^\s*[-*]\s+\[ \]', line):
                current.append(line.lstrip())
        elif current:
            current.append(line)
    flush()
    if len(tasks) > 60:
        raise ValueError('task_count')
    return tasks


def prepared_candidate(text, encounter_date):
    from clinical_data import VisitCaptureCandidateV1
    ref = 'prepared_note'
    return VisitCaptureCandidateV1.model_validate({
        'schema_version': 'visit_capture_candidate_v1',
        'source_identity_state': 'consistent',
        'encounter_date': encounter_date,
        'source_bindings': [{'source_ref': ref, 'role': 'prepared_clinical_note'}],
        'comparison_basis': {'type': 'unavailable'},
        'reason_for_visit': {'text': note_reason(text), 'provenance': [ref]},
        'prepared_note': {'text': text},
        'pending': [{'description': description, 'source_excerpt': excerpt,
                     'item_type': 'other', 'status': 'open',
                     'responsible_role': 'unspecified', 'provenance': [ref]}
                    for description, excerpt in note_tasks(text)],
    })
