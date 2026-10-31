"""Read-only, alias-scoped iCloud clinical mail intake.

No mailbox data is stored. The Apple app-specific password stays server-side.
The existing Cockpit clinician cookie authorizes reads; no clinical chart,
pending task, mailbox flag, or provider write is performed.
"""
from __future__ import annotations

import imaplib
import logging
import os
import re
import ssl
import unicodedata
from contextlib import contextmanager
from datetime import timezone
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from html.parser import HTMLParser

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from clinical_reception_calls import _require_clinical_key

_HOST = "imap.mail.me.com"
_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,80}@icloud\.com$", re.I)
_ITEM_RE = re.compile(r"^([1-9][0-9]{0,18})-([1-9][0-9]{0,18})$")
_MAX_CANDIDATES = 80  # Header fetches per request, not a permanent mailbox cut-off
_UID_WINDOW = 256  # IMAP server limits each SEARCH to this bounded UID interval
_MAX_WINDOWS = 8
_MAX_SEARCH_REPLY_BYTES = _UID_WINDOW * 21
_MAX_MESSAGE_BYTES = 12 * 1024 * 1024
_MAX_HEADER_BYTES = 64 * 1024
_MAX_FETCH_METADATA_BYTES = 4096  # Bounded IMAP FETCH response atoms, not header literals
_HEADER_FETCH = "(BODY.PEEK[HEADER.FIELDS (FROM TO CC SUBJECT DATE LIST-UNSUBSCRIBE PRECEDENCE AUTO-SUBMITTED)]<0.65537> RFC822.SIZE)"
_MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
_CLINICAL_TERMS = (
    "ασθεν", "ιατρ", "ορθοπ", "ακτινογραφ", "μαγνητικ", "εξετασ",
    "αποτελεσμα", "εργαστηρι", "αιματολογ", "παραπεμπτικ", "συνταγ",
    "φυσιοθεραπ", "γνωματευ", "χειρουργ", "οστεοπορωσ", "θεραπει",
    "patient", "medical", "mri", "x-ray", "xray", "scan", "ultrasound",
    "radiolog", "laborator", "blood", "referral", "results", "report",
)
_ALLOWED_ATTACHMENTS = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class _SourceUnavailable(Exception):
    pass


# Only fixed stage names may appear in protected HTTP responses or server logs.
_DIAGNOSTIC_CODES = frozenset({
    "CONFIG_DISABLED", "CONFIG_MISSING", "TLS_CONNECTION_FAILED",
    "IMAP_AUTH_FAILED", "INBOX_SELECT_FAILED",
    "UIDVALIDITY_OR_UIDNEXT_FAILED", "IMAP_SEARCH_FAILED",
    "IMAP_HEADER_FETCH_FAILED", "IMAP_MESSAGE_FETCH_FAILED",
    "IMAP_FLAGS_FETCH_FAILED",
    "IMAP_UNEXPECTED_FAILURE",
})
_LOG = logging.getLogger(__name__)


class _StageFailure(Exception):
    def __init__(self, code: str):
        if code not in _DIAGNOSTIC_CODES:
            raise ValueError("Unknown diagnostic stage")
        super().__init__(code)
        self.code = code


class _BadMessage(Exception):
    """A single malformed/unreadable header; preserve the remaining mailbox."""


def _settings() -> tuple[str, str, str]:
    if os.getenv("ICLOUD_CLINICAL_READ_ENABLED") != "1":
        raise _StageFailure("CONFIG_DISABLED")
    username = os.getenv("ICLOUD_CLINICAL_IMAP_USERNAME", "").strip()
    password = os.getenv("ICLOUD_CLINICAL_APP_PASSWORD", "")
    alias = os.getenv("ICLOUD_CLINICAL_ALIAS", "").strip().lower()
    if not username or not password or not _ALIAS_RE.fullmatch(alias):
        raise _StageFailure("CONFIG_MISSING")
    return username, password, alias


@contextmanager
def _session(username: str, password: str):
    """Fixed TLS IMAP and read-only EXAMINE. Never expose provider exception text."""
    try:
        mail = imaplib.IMAP4_SSL(
            _HOST, 993, ssl_context=ssl.create_default_context(), timeout=8
        )
    except Exception:
        raise _StageFailure("TLS_CONNECTION_FAILED") from None
    try:
        try:
            status, _ = mail.login(username, password)
        except Exception:
            raise _StageFailure("IMAP_AUTH_FAILED") from None
        if status != "OK":
            raise _StageFailure("IMAP_AUTH_FAILED")
        try:
            status, _ = mail.select("INBOX", readonly=True)
        except Exception:
            raise _StageFailure("INBOX_SELECT_FAILED") from None
        if status != "OK":
            raise _StageFailure("INBOX_SELECT_FAILED")
        try:
            _, values = mail.response("UIDVALIDITY")
            raw = next((v for v in (values or []) if isinstance(v, bytes)), b"")
        except Exception:
            raise _StageFailure("UIDVALIDITY_OR_UIDNEXT_FAILED") from None
        if not raw.isdigit() or not (0 < int(raw) < 10**19):
            raise _StageFailure("UIDVALIDITY_OR_UIDNEXT_FAILED")
        yield mail, str(int(raw))
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _clean(value: str, limit: int) -> str:
    return "".join(c for c in str(value) if c.isprintable() or c in "\n\t").strip()[:limit]


def _normalized(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", value.casefold())
        if unicodedata.category(c) != "Mn"
    )


def _classification(message) -> str | None:
    subject = _normalized(str(message.get("Subject", "")))
    clinical_hint = any(term in subject for term in _CLINICAL_TERMS)
    auto = str(message.get("Auto-Submitted", "")).casefold()
    bulk = (bool(message.get("List-Unsubscribe"))
            or str(message.get("Precedence", "")).casefold() in ("bulk", "list", "junk")
            or (bool(auto) and auto != "no"))
    # Lab/clinical notifications can be auto-generated and include a
    # List-Unsubscribe header. Keep plausible clinical mail in the doctor's
    # review bucket instead of silently losing it.
    if bulk:
        return "review" if clinical_hint else None
    return "clinical" if clinical_hint else "review"


def _to_alias(message, alias: str) -> bool:
    recipients = getaddresses(
        [str(message.get("To", "")), str(message.get("Cc", ""))]
    )
    return any(address.casefold() == alias for _, address in recipients)


# IMAP4.uid("FETCH") may return unsolicited FETCH rows for other UIDs.
# Match a literal and its own adjacent trailer before using either metadata
# or message bytes; never combine metadata from different FETCH responses.
_FETCH_START = re.compile(rb"^[1-9][0-9]{0,18}[ \t]+\(")
_FETCH_UID_NAME = re.compile(rb"(?:^|[\s(])UID(?=[\s)]|$)", re.I)
_FETCH_UID_VALUE = re.compile(rb"(?:^|[\s(])UID[ \t]+([1-9][0-9]{0,18})(?=[\s)]|$)", re.I)
_FETCH_SIZE_NAME = re.compile(rb"(?:^|[\s(])RFC822\.SIZE(?=[\s)]|$)", re.I)
_FETCH_SIZE_VALUE = re.compile(rb"(?:^|[\s(])RFC822\.SIZE[ \t]+([0-9]{1,20})(?=[\s)]|$)", re.I)


def _fetch_literal_for_uid(items, uid: str, *, require_size: bool) -> tuple[bytes, int | None]:
    """Select one UID-matched FETCH literal, never another message's metadata."""
    if not isinstance(uid, str) or not re.fullmatch(r"[1-9][0-9]{0,18}", uid):
        raise _BadMessage()
    if not isinstance(items, (list, tuple)):
        raise _BadMessage()
    requested = uid.encode("ascii")
    selected = None
    metadata_bytes = 0
    index = 0
    while index < len(items):
        row = items[index]
        index += 1
        if not isinstance(row, tuple):
            if isinstance(row, bytes):
                metadata_bytes += len(row)
            elif row is not None:
                raise _BadMessage()
            if metadata_bytes > _MAX_FETCH_METADATA_BYTES:
                raise _BadMessage()
            continue
        if len(row) != 2 or not isinstance(row[0], bytes) or not isinstance(row[1], bytes):
            raise _BadMessage()
        prefix, literal = row
        if not _FETCH_START.match(prefix):
            raise _BadMessage()
        trailer = b""
        # imaplib always emits a standalone trailer after a literal. A
        # standalone FETCH line ("123 (...)") is another response, not trailer.
        if index < len(items) and isinstance(items[index], bytes):
            following = items[index]
            if not _FETCH_START.match(following):
                trailer = following
                index += 1
        metadata_bytes += len(prefix) + len(trailer)
        if metadata_bytes > _MAX_FETCH_METADATA_BYTES:
            raise _BadMessage()
        metadata = prefix + b" " + trailer
        uids = _FETCH_UID_VALUE.findall(metadata)
        if requested not in uids:
            continue
        # One exact UID attribute per selected FETCH response; reject multiple
        # literals for that UID rather than silently choosing the first.
        if selected is not None or len(_FETCH_UID_NAME.findall(metadata)) != 1 or len(uids) != 1:
            raise _BadMessage()
        size = None
        if require_size:
            sizes = _FETCH_SIZE_VALUE.findall(metadata)
            if len(_FETCH_SIZE_NAME.findall(metadata)) != 1 or len(sizes) != 1:
                raise _BadMessage()
            size = int(sizes[0])
        selected = (literal, size)
    if selected is None:
        raise _BadMessage()
    return selected


def _fetch_headers(mail, uid: str):
    try:
        status, items = mail.uid("FETCH", uid, _HEADER_FETCH)
    except Exception:
        raise _StageFailure("IMAP_HEADER_FETCH_FAILED") from None
    if status != "OK":
        raise _StageFailure("IMAP_HEADER_FETCH_FAILED")
    header, size = _fetch_literal_for_uid(items, uid, require_size=True)
    if len(header) > _MAX_HEADER_BYTES:
        raise _BadMessage()
    try:
        message = BytesParser(policy=policy.default).parsebytes(
            header, headersonly=True
        )
    except (TypeError, ValueError, UnicodeError):
        raise _BadMessage() from None
    return message, size


def _summary(uid: str, validity: str, message, alias: str):
    if not _to_alias(message, alias):
        return None
    classification = _classification(message)
    if classification is None:
        return None
    name, sender = (getaddresses([str(message.get("From", ""))]) or [("", "")])[0]
    try:
        dt = parsedate_to_datetime(str(message.get("Date", "")))
        date = dt.astimezone(timezone.utc).isoformat() if dt.tzinfo else ""
    except (TypeError, ValueError, IndexError):
        date = ""
    return {
        "id": validity + "-" + uid,
        "from": _clean((name + " <" + sender + ">") if name else sender, 180),
        "subject": _clean(str(message.get("Subject", "")), 220) or "(Χωρίς θέμα)",
        "date": date,
        "category": classification,
    }


def _uid_next(mail) -> int:
    try:
        kind, values = mail.response("UIDNEXT")
        raw = next((v for v in (values or []) if isinstance(v, bytes)), b"")
    except Exception:
        raise _StageFailure("UIDVALIDITY_OR_UIDNEXT_FAILED") from None
    if kind != "UIDNEXT" or not raw.isdigit() or not (0 < int(raw) < 10**19):
        raise _StageFailure("UIDVALIDITY_OR_UIDNEXT_FAILED")
    return int(raw)


def _search_window(mail, alias: str, low: int, high: int) -> list[int]:
    """Bound the IMAP server's response before it returns any UIDs."""
    quoted = '"' + alias + '"'
    try:
        # Use the provider-proven bounded UID + recipient search for both paths.
        # A combined UNSEEN/UID/OR expression may fail on this provider; the
        # peek checks exact UID-bound FLAGS separately without changing Seen.
        status, data = mail.uid(
            "SEARCH", None, "UID", f"{low}:{high}",
            "OR", "TO", quoted, "CC", quoted
        )
    except Exception:
        raise _StageFailure("IMAP_SEARCH_FAILED") from None
    if status != "OK" or not data or not isinstance(data[0], bytes):
        raise _StageFailure("IMAP_SEARCH_FAILED")
    if len(data[0]) > _MAX_SEARCH_REPLY_BYTES:
        raise _StageFailure("IMAP_SEARCH_FAILED")
    tokens = data[0].split()
    if len(tokens) > _UID_WINDOW:
        raise _StageFailure("IMAP_SEARCH_FAILED")
    result = []
    for raw in tokens:
        if not raw.isdigit():
            raise _StageFailure("IMAP_SEARCH_FAILED")
        uid = int(raw)
        if not (low <= uid <= high):
            raise _StageFailure("IMAP_SEARCH_FAILED")
        result.append(uid)
    return sorted(set(result), reverse=True)


_FETCH_FLAGS_NAME = re.compile(rb"(?:^|[\s(])FLAGS(?=[\s)]|$)", re.I)
_FETCH_FLAGS_VALUE = re.compile(rb"(?:^|[\s(])FLAGS[ \t]+\(([^()]{0,512})\)", re.I)


def _uid_is_unseen(mail, uid: str) -> bool:
    """Read exact UID-bound flags; ambiguity never becomes an unread claim."""
    try:
        status, rows = mail.uid("FETCH", uid, "(UID FLAGS)")
    except Exception:
        raise _StageFailure("IMAP_FLAGS_FETCH_FAILED") from None
    if status != "OK":
        raise _StageFailure("IMAP_FLAGS_FETCH_FAILED")
    if not isinstance(rows, (list, tuple)):
        raise _BadMessage()
    wanted = uid.encode("ascii")
    verified = None
    metadata_size = 0
    for row in rows:
        if row is None:
            continue
        if not isinstance(row, bytes):
            raise _BadMessage()
        metadata_size += len(row)
        if metadata_size > _MAX_FETCH_METADATA_BYTES:
            raise _BadMessage()
        if not _FETCH_START.match(row):
            continue  # A bounded trailer, never evidence for the requested UID.
        uids = _FETCH_UID_VALUE.findall(row)
        if wanted not in uids:
            continue
        if verified is not None or len(uids) != 1 or len(_FETCH_UID_NAME.findall(row)) != 1:
            raise _BadMessage()
        flags = _FETCH_FLAGS_VALUE.findall(row)
        if len(flags) != 1 or len(_FETCH_FLAGS_NAME.findall(row)) != 1:
            raise _BadMessage()
        verified = b"\\seen" not in (flag.lower() for flag in flags[0].split())
    if verified is None:
        raise _BadMessage()
    return verified


def _list_items(mail, validity: str, alias: str, limit: int, cursor: str | None = None, *, unseen_only: bool = False):
    last_uid = _uid_next(mail) - 1
    if cursor is not None:
        match = _ITEM_RE.fullmatch(cursor)
        if not match:
            raise HTTPException(422, "Invalid mail continuation")
        if match.group(1) != validity:
            raise HTTPException(409, "Mailbox changed; restart from newest")
        high = int(match.group(2))
        if high > last_uid:
            raise HTTPException(422, "Invalid mail continuation")
    else:
        high = last_uid

    result = []
    skipped = 0
    headers_checked = 0
    windows = 0
    while high > 0 and windows < _MAX_WINDOWS and headers_checked < _MAX_CANDIDATES:
        low = max(1, high - _UID_WINDOW + 1)
        found = _search_window(mail, alias, low, high)
        windows += 1
        exhausted_early = False
        for uid in found:
            high = uid - 1
            headers_checked += 1
            try:
                # At most 80 candidate flag reads per peek, never an unbounded
                # mailbox-wide UNSEEN query. Each FLAGS response is tied to UID.
                if unseen_only and not _uid_is_unseen(mail, str(uid)):
                    item = None
                else:
                    header, _ = _fetch_headers(mail, str(uid))
                    item = _summary(str(uid), validity, header, alias)
            except (_BadMessage, ValueError, TypeError, UnicodeError):
                skipped += 1
                item = None
            if item is not None:
                result.append(item)
            if len(result) >= limit or headers_checked >= _MAX_CANDIDATES:
                exhausted_early = True
                break
        # A page is capped at exactly its requested limit. Previously the inner
        # loop broke but the outer window scan continued, causing >4 peek items
        # (PEEK_ITEMS) or >25 full-view items, then false client failure.
        if len(result) >= limit or headers_checked >= _MAX_CANDIDATES:
            break
        if not exhausted_early:
            high = low - 1

    # Partial coverage is explicit. A page with no matches is NOT necessarily
    # an empty mailbox, and one damaged message does not hide healthy ones.
    return {
        "items": result,
        "next_cursor": f"{validity}-{high}" if high > 0 else None,
        "skipped_count": skipped,
        "coverage_complete": high == 0 and skipped == 0,
    }


def _message(mail, validity: str, alias: str, item_id: str):
    match = _ITEM_RE.fullmatch(item_id)
    if not match or match.group(1) != validity:
        raise HTTPException(404, "Mail item not found")
    uid = match.group(2)
    try:
        header, size = _fetch_headers(mail, uid)
        if not _summary(uid, validity, header, alias):
            raise HTTPException(404, "Mail item not found")
    except _BadMessage:
        raise HTTPException(422, "Mail item cannot be safely read") from None
    if size > _MAX_MESSAGE_BYTES:
        raise HTTPException(413, "Mail item exceeds safe size limit")
    try:
        status, data = mail.uid("FETCH", uid, "(BODY.PEEK[])")
        if status != "OK":
            raise _StageFailure("IMAP_MESSAGE_FETCH_FAILED")
        raw, _unused_size = _fetch_literal_for_uid(data, uid, require_size=False)
    except _StageFailure:
        raise
    except Exception:
        raise _StageFailure("IMAP_MESSAGE_FETCH_FAILED") from None
    if len(raw) > _MAX_MESSAGE_BYTES:
        raise HTTPException(413, "Mail item exceeds safe size limit")
    message = BytesParser(policy=policy.default).parsebytes(raw)
    if not _summary(uid, validity, message, alias):
        raise HTTPException(404, "Mail item not found")
    return message


class _PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.suppressed = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.suppressed += 1
        elif not self.suppressed and tag in ("br", "div", "p", "li"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head") and self.suppressed:
            self.suppressed -= 1
        elif not self.suppressed and tag in ("div", "p", "li"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(data)


def _body(message) -> str:
    plain = None
    html = None
    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        if part.get_filename():
            continue
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except (LookupError, UnicodeError, ValueError):
            continue
        if ctype == "text/plain" and plain is None:
            plain = content
        elif ctype == "text/html" and html is None:
            html = content
    if plain is not None:
        return _clean(plain, 14000)
    if html is not None:
        parser = _PlainHTML()
        parser.feed(html[:80000])
        return _clean(" ".join(parser.parts), 14000)
    return ""


def _looks_like_declared_file(content_type: str, data: bytes) -> bool:
    """Reject obvious MIME spoofing; this is not a malware/antivirus scan."""
    if content_type == "application/pdf":
        return b"%PDF-" in data[:32]
    if content_type == "image/jpeg":
        return data.startswith(bytes.fromhex("ffd8ff"))
    if content_type == "image/png":
        return data.startswith(bytes.fromhex("89504e470d0a1a0a"))
    if content_type == "image/webp":
        return data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    return False


def _attachments(message):
    items = []
    for index, part in enumerate(message.walk()):
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        if ctype not in _ALLOWED_ATTACHMENTS:
            continue
        if part.get_content_disposition() != "attachment" and not part.get_filename():
            continue
        data = part.get_payload(decode=True)
        if not isinstance(data, bytes) or len(data) > _MAX_ATTACHMENT_BYTES:
            continue
        if not _looks_like_declared_file(ctype, data):
            continue
        filename = part.get_filename() or ("Συνημμένο" + _ALLOWED_ATTACHMENTS[ctype])
        name = _clean(filename.replace("\\", "/").split("/")[-1], 90)
        items.append((index, name or "Συνημμένο", ctype, data))
    return items


def _attachment_count(message) -> int:
    return sum(
        1 for part in message.walk()
        if not part.is_multipart() and (
            part.get_content_disposition() == "attachment"
            or part.get_filename()
        )
    )


def _no_store(response: Response):
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"


def _read(operation):
    try:
        username, password, alias = _settings()
        with _session(username, password) as (mail, validity):
            return operation(mail, validity, alias)
    except HTTPException:
        raise
    except _StageFailure as error:
        # One allowlisted stage code only; no exception, credentials or PHI.
        _LOG.warning("icloud_mail_diagnostic code=%s", error.code)
        raise HTTPException(503, {"code": error.code}) from None
    except Exception:
        _LOG.warning("icloud_mail_diagnostic code=IMAP_UNEXPECTED_FAILURE")
        raise HTTPException(503, {"code": "IMAP_UNEXPECTED_FAILURE"}) from None


def build_icloud_mail_router() -> APIRouter:
    router = APIRouter(
        prefix="/clinical/icloud-mail", tags=["clinical-icloud-mail"],
        dependencies=[Depends(_require_clinical_key)],
    )

    @router.get("/peek")
    def peek(response: Response):
        _no_store(response)
        return {
            "source": "icloud_mail", "scope": "recipient_alias_inbox_only",
            "filter": "unseen",
            **_read(lambda mail, validity, alias: _list_items(
                mail, validity, alias, 4, unseen_only=True)),
        }

    @router.get("")
    def list_mail(
        response: Response,
        limit: int = Query(default=20, ge=1, le=25),
        cursor: str | None = Query(default=None, max_length=40),
    ):
        _no_store(response)
        return {
            "source": "icloud_mail", "scope": "recipient_alias_inbox_only",
            "filter": "all",
            **_read(lambda mail, validity, alias: _list_items(mail, validity, alias, limit, cursor)),
        }

    @router.get("/items/{item_id}")
    def read_mail(item_id: str, response: Response):
        _no_store(response)

        def result(mail, validity, alias):
            message = _message(mail, validity, alias, item_id)
            header = _summary(item_id.split("-")[1], validity, message, alias)
            available = _attachments(message)
            return {
                "source": "icloud_mail", "item": {
                    **header, "body": _body(message),
                    "attachments_unavailable": max(0, _attachment_count(message) - len(available)),
                    "attachments": [
                        {"index": index, "name": name, "content_type": content_type, "size": len(data)}
                        for index, name, content_type, data in available
                    ],
                },
            }
        return _read(result)

    @router.get("/items/{item_id}/attachments/{index}")
    def download(item_id: str, index: int):
        if not 0 <= index <= 100:
            raise HTTPException(404, "Attachment not found")

        def result(mail, validity, alias):
            message = _message(mail, validity, alias, item_id)
            for number, _name, content_type, data in _attachments(message):
                if number == index:
                    ext = _ALLOWED_ATTACHMENTS[content_type]
                    return Response(
                        content=data, media_type="application/octet-stream",
                        headers={
                            "Content-Disposition": 'attachment; filename="clinical-attachment-' + str(index) + ext + '"',
                            "Cache-Control": "private, no-store",
                            "X-Content-Type-Options": "nosniff",
                            "Content-Security-Policy": "sandbox",
                        },
                    )
            raise HTTPException(404, "Attachment not found")
        return _read(result)

    return router
