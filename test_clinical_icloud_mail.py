"""Offline isolated tests of the actual read-only iCloud IMAP adapter."""
import os
from contextlib import contextmanager
from email.message import EmailMessage

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import clinical_icloud_mail as ic


ALIAS = "practice.alias@icloud.com"


def mail(to=ALIAS, subject="Μαγνητική τομογραφία", body="Αποτέλεσμα εξέτασης",
         sender="patient@example.invalid", newsletter=False, attached=False,
         unsupported=False):
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Date"] = "Thu, 29 Oct 2026 09:12:00 +0200"
    msg["Subject"] = subject
    if newsletter:
        msg["List-Unsubscribe"] = "<mailto:unsubscribe@example.invalid>"
    msg.set_content(body)
    if attached:
        msg.add_attachment(b"%PDF-1.7\nSynthetic test PDF",
                           maintype="application", subtype="pdf",
                           filename="scan.pdf")
    if unsupported:
        msg.add_attachment(b"Sample test archive", maintype="application",
                           subtype="zip", filename="imaging-study.zip")
    return msg.as_bytes()


class FakeMail:
    def __init__(self):
        self.rows = {
            "1": mail(to="private@icloud.com", subject="Μαγνητική προσωπικό"),
            "2": mail(subject="Weekly offers", newsletter=True),
            "3": mail(subject="Καλημέρα", body="Μπορούμε να μιλήσουμε;"),
            "4": mail(subject="Αποτελέσματα MRI", body="<svg onload=alert(1)>",
                      attached=True),
        }
        self.operations = []

    def uid(self, operation, *args):
        self.operations.append((operation, args))
        if operation == "SEARCH":
            assert args[:2] == (None, "UID")
            assert args[3:] == ("OR", "TO", '"' + ALIAS + '"',
                                "CC", '"' + ALIAS + '"')
            low, high = map(int, args[2].split(":"))
            assert 1 <= low <= high and high - low < ic._UID_WINDOW
            uids = [int(uid) for uid in self.rows if low <= int(uid) <= high]
            return "OK", [b" ".join(str(uid).encode() for uid in sorted(uids))]
        if operation == "FETCH":
            uid, spec = args
            row = self.rows.get(uid)
            if row is None:
                return "OK", [None]
            if "HEADER" in spec:
                header = row.split(b"\n\n", 1)[0] + b"\r\n\r\n"
                return "OK", [(b"1 (UID " + uid.encode() +
                              b" RFC822.SIZE " + str(len(row)).encode() + b")",
                              header)]
            assert spec == "(BODY.PEEK[])"
            return "OK", [(b"1 (BODY[])", row)]
        raise AssertionError("Mutation or unexpected IMAP operation: " + operation)

    def response(self, key):
        assert key == "UIDNEXT"
        next_uid = max((int(uid) for uid in self.rows), default=0) + 1
        return "UIDNEXT", [str(next_uid).encode("ascii")]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", "synthetic-clinical-key")
    monkeypatch.setenv("ICLOUD_CLINICAL_READ_ENABLED", "1")
    monkeypatch.setenv("ICLOUD_CLINICAL_IMAP_USERNAME", "primary_icloud_mail")
    monkeypatch.setenv("ICLOUD_CLINICAL_APP_PASSWORD", "synthetic-only-password")
    monkeypatch.setenv("ICLOUD_CLINICAL_ALIAS", ALIAS)
    fake = FakeMail()
    @contextmanager
    def source(username, password):
        assert username == "primary_icloud_mail"
        assert password == "synthetic-only-password"
        yield fake, "765"
    monkeypatch.setattr(ic, "_session", source)
    app = FastAPI()
    app.include_router(ic.build_icloud_mail_router())
    return TestClient(app), fake


def get(client, path):
    return client.get("/clinical/icloud-mail" + path,
                      headers={"X-Clinical-Key": "synthetic-clinical-key"})


def test_authorization_and_disabled_fail_closed(client, monkeypatch):
    http, _ = client
    assert http.get("/clinical/icloud-mail/peek").status_code == 401
    assert http.get("/clinical/icloud-mail/peek",
                    headers={"X-Clinical-Key": "wrong"}).status_code == 401
    monkeypatch.delenv("ICLOUD_CLINICAL_READ_ENABLED")
    assert get(http, "/peek").status_code == 503
    monkeypatch.setenv("ICLOUD_CLINICAL_READ_ENABLED", "1")
    monkeypatch.delenv("ICLOUD_CLINICAL_APP_PASSWORD")
    assert get(http, "/peek").status_code == 503


def test_alias_only_bulk_excluded_and_review_bucket(client):
    http, source = client
    response = get(http, "/peek")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    payload = response.json()
    assert payload["scope"] == "recipient_alias_inbox_only"
    assert [(x["id"], x["category"]) for x in payload["items"]] == [
        ("765-4", "clinical"), ("765-3", "review"),
    ]
    assert payload["next_cursor"] is None
    assert payload["skipped_count"] == 0
    assert payload["coverage_complete"] is True
    assert "private@icloud.com" not in response.text
    assert "unsubscribe" not in response.text
    assert all(op in ("SEARCH", "FETCH") for op, args in source.operations)
    assert all("BODY.PEEK" in args[-1] for op, args in source.operations if op == "FETCH")
    assert "synthetic-only-password" not in response.text


def test_detail_safe_body_and_pdf_attachment(client):
    http, _ = client
    response = get(http, "/items/765-4")
    assert response.status_code == 200
    item = response.json()["item"]
    assert item["subject"] == "Αποτελέσματα MRI"
    assert "<svg onload=alert(1)>" in item["body"]  # rendered with textContent, never HTML
    assert item["attachments"][0]["name"] == "scan.pdf"
    index = item["attachments"][0]["index"]
    download = get(http, "/items/765-4/attachments/" + str(index))
    assert download.status_code == 200
    assert download.content.startswith(b"%PDF")
    assert download.headers["Content-Type"] == "application/octet-stream"
    assert "attachment" in download.headers["Content-Disposition"]
    assert "scan.pdf" not in download.headers["Content-Disposition"]
    assert download.headers["Cache-Control"] == "private, no-store"


def test_cannot_guess_private_mail_or_stale_uidvalidity(client):
    http, source = client
    assert get(http, "/items/765-1").status_code == 404
    assert get(http, "/items/766-4").status_code == 404
    assert get(http, "/items/765-999").status_code == 422
    assert get(http, "/items/765-4/attachments/99").status_code == 404
    source.rows["4"] = mail(to="private@icloud.com", subject="Μαγνητική")
    assert get(http, "/items/765-4").status_code == 404


def test_oversized_message_denied_before_body_download(client):
    http, source = client
    source.rows["4"] = mail(subject="Αποτέλεσμα", body="x" * (ic._MAX_MESSAGE_BYTES + 1))
    response = get(http, "/items/765-4")
    assert response.status_code == 413
    assert not any(op == "FETCH" and args[-1] == "(BODY.PEEK[])" for op, args in source.operations)


def test_no_secret_or_provider_message_in_errors(client, monkeypatch):
    http, _ = client
    @contextmanager
    def failed(_username, _password):
        raise RuntimeError("synthetic-only-password ERROR patient data")
        yield
    monkeypatch.setattr(ic, "_session", failed)
    response = get(http, "/peek")
    assert response.status_code == 503
    assert "synthetic-only-password" not in response.text
    assert "patient data" not in response.text



def test_generated_medical_notification_is_not_silently_dropped(client):
    http, fake = client
    fake.rows["2"] = mail(subject="Laboratory results available", newsletter=True)
    response = get(http, "/peek")
    assert response.status_code == 200
    assert ("765-2", "review") in [
        (row["id"], row["category"]) for row in response.json()["items"]
    ]


def test_one_oversized_header_does_not_hide_other_patient_messages(client):
    http, fake = client
    fake.rows["4"] = mail(subject="MRI " + ("Z" * (ic._MAX_HEADER_BYTES + 100)))
    response = get(http, "/peek")
    assert response.status_code == 200
    payload = response.json()
    assert payload["skipped_count"] == 1
    assert payload["coverage_complete"] is False
    assert any(row["id"] == "765-3" for row in payload["items"])
    assert not any(row["id"] == "765-4" for row in payload["items"])
    assert not any(
        op == "FETCH" and args[-1] == "(BODY.PEEK[])"
        for op, args in fake.operations
    )


def test_real_imap_session_contract_is_readonly_and_tls(monkeypatch):
    actions = []

    class FakeIMAP:
        def __init__(self, host, port, ssl_context, timeout):
            assert host == "imap.mail.me.com"
            assert port == 993 and timeout == 8
            assert ssl_context.verify_mode
            actions.append("connect")

        def login(self, username, password):
            assert username == "example" and password == "fake-pass"
            actions.append("login")
            return "OK", [b"logged in"]

        def select(self, mailbox, readonly=False):
            assert mailbox == "INBOX" and readonly is True
            actions.append("examine")
            return "OK", [b"2"]

        def response(self, key):
            assert key == "UIDVALIDITY"
            return "UIDVALIDITY", [b"765"]

        def logout(self):
            actions.append("logout")
            return "BYE", [b"closed"]

    monkeypatch.setattr(ic.imaplib, "IMAP4_SSL", FakeIMAP)
    with ic._session("example", "fake-pass") as (_client, uidvalidity):
        assert uidvalidity == "765"
    assert actions == ["connect", "login", "examine", "logout"]



def test_unsupported_attachment_is_exposed_as_missing_not_an_empty_list(client):
    http, source = client
    source.rows["4"] = mail(subject="MRI report", attached=True, unsupported=True)
    response = get(http, "/items/765-4")
    assert response.status_code == 200
    item = response.json()["item"]
    assert len(item["attachments"]) == 1
    assert item["attachments_unavailable"] == 1
    assert "imaging-study.zip" not in response.text


def test_oversized_attachment_is_exposed_as_missing(client):
    http, source = client
    source.rows["4"] = mail(subject="MRI report", unsupported=True)
    # Treat a small unsupported archive as absent from the download allowlist.
    item = get(http, "/items/765-4").json()["item"]
    assert item["attachments"] == []
    assert item["attachments_unavailable"] == 1



def test_older_clinical_mail_survives_more_than_80_recent_irrelevant_candidates(client):
    http, fake = client
    fake.rows = {
        str(i): mail(subject="Weekly offers", newsletter=True)
        for i in range(1, 101)
    }
    fake.rows["5"] = mail(subject="Αποτελέσματα εξέτασης MRI")
    first = get(http, "/peek")
    assert first.status_code == 200
    assert first.json()["items"] == []
    assert first.json()["coverage_complete"] is False
    assert first.json()["next_cursor"] == "765-20"
    second = get(http, "/?cursor=" + first.json()["next_cursor"])
    assert second.status_code == 200
    assert [(row["id"], row["category"]) for row in second.json()["items"]] == [
        ("765-5", "clinical"),
    ]
    assert second.json()["next_cursor"] is None
    assert second.json()["coverage_complete"] is True
    windows = [
        args[2] for op, args in fake.operations if op == "SEARCH"
    ]
    assert windows == ["1:100", "1:20"]


def test_no_false_empty_when_sparse_uid_range_needs_continuation(client):
    http, fake = client
    fake.rows = {
        "1": mail(subject="Εργαστηριακά αποτελέσματα"),
        "10000": mail(to="private@icloud.com", subject="Private"),
    }
    result = get(http, "/peek")
    assert result.status_code == 200
    assert result.json()["items"] == []
    assert result.json()["coverage_complete"] is False
    assert result.json()["next_cursor"]
    assert sum(op == "SEARCH" for op, args in fake.operations) == ic._MAX_WINDOWS


def test_cursor_is_bounded_and_rejects_stale_mailbox(client):
    http, _ = client
    assert get(http, "/?cursor=invalid").status_code == 422
    assert get(http, "/?cursor=766-1").status_code == 409
    assert get(http, "/?cursor=765-99999999").status_code == 422


def test_empty_mailbox_has_confirmed_coverage(client):
    http, fake = client
    fake.rows.clear()
    result = get(http, "/peek")
    assert result.status_code == 200
    assert result.json()["items"] == []
    assert result.json()["coverage_complete"] is True
    assert result.json()["next_cursor"] is None


def test_spoofed_pdf_mime_does_not_become_downloadable(client):
    http, fake = client
    sample = EmailMessage()
    sample["From"] = "sample@example.invalid"
    sample["To"] = ALIAS
    sample["Subject"] = "MRI results"
    sample.set_content("Attached patient report")
    sample.add_attachment(
        b"<html>not a PDF</html>", maintype="application",
        subtype="pdf", filename="imaging.pdf"
    )
    fake.rows["4"] = sample.as_bytes()
    detail = get(http, "/items/765-4")
    assert detail.status_code == 200
    assert detail.json()["item"]["attachments"] == []
    assert detail.json()["item"]["attachments_unavailable"] == 1
    assert get(http, "/items/765-4/attachments/1").status_code == 404


# Diagnostic failures are synthetic, stage-only, and never read real provider mail.
@pytest.mark.parametrize("phase,expected_code,endpoint", [
    ("connect", "TLS_CONNECTION_FAILED", "/peek"),
    ("connect_timeout", "TLS_CONNECTION_FAILED", "/peek"),
    ("auth", "IMAP_AUTH_FAILED", "/peek"),
    ("auth_timeout", "IMAP_AUTH_FAILED", "/peek"),
    ("auth_rejected", "IMAP_AUTH_FAILED", "/peek"),
    ("select", "INBOX_SELECT_FAILED", "/peek"),
    ("select_disconnect", "INBOX_SELECT_FAILED", "/peek"),
    ("uidvalidity", "UIDVALIDITY_OR_UIDNEXT_FAILED", "/peek"),
    ("uidnext", "UIDVALIDITY_OR_UIDNEXT_FAILED", "/peek"),
    ("search", "IMAP_SEARCH_FAILED", "/peek"),
    ("headers", "IMAP_HEADER_FETCH_FAILED", "/peek"),
    ("body", "IMAP_MESSAGE_FETCH_FAILED", "/items/765-1"),
])
def test_fixed_imap_stage_codes_hide_provider_errors(monkeypatch, caplog, phase, expected_code, endpoint):
    monkeypatch.setenv("CLINICAL_DATA_KEY", "synthetic-clinical-key")
    monkeypatch.setenv("ICLOUD_CLINICAL_READ_ENABLED", "1")
    monkeypatch.setenv("ICLOUD_CLINICAL_IMAP_USERNAME", "test-primary-username")
    monkeypatch.setenv("ICLOUD_CLINICAL_APP_PASSWORD", "synthetic-secret-password")
    monkeypatch.setenv("ICLOUD_CLINICAL_ALIAS", ALIAS)
    actions = []

    class StageIMAP:
        def __init__(self, host, port, ssl_context, timeout):
            assert host == "imap.mail.me.com" and port == 993 and timeout == 8
            assert ssl_context.verify_mode
            actions.append("tls")
            if phase == "connect":
                raise RuntimeError("synthetic-secret-password patient private provider error")
            if phase == "connect_timeout":
                raise TimeoutError("synthetic-secret-password patient private provider error")

        def login(self, username, password):
            assert username == "test-primary-username"
            assert password == "synthetic-secret-password"
            actions.append("login")
            if phase == "auth":
                raise RuntimeError("synthetic-secret-password patient private provider error")
            if phase == "auth_timeout":
                raise TimeoutError("synthetic-secret-password patient private provider error")
            if phase == "auth_rejected":
                return "NO", [b"synthetic-secret-password"]
            return "OK", [b"synthetic logged in"]

        def select(self, name, readonly=False):
            assert name == "INBOX" and readonly is True
            actions.append("examine")
            if phase == "select":
                return "NO", [b"synthetic-secret-password"]
            if phase == "select_disconnect":
                raise ConnectionError("synthetic-secret-password patient private provider error")
            return "OK", [b"one"]

        def response(self, name):
            actions.append(name)
            if name == "UIDVALIDITY":
                return "UIDVALIDITY", [b"invalid"] if phase == "uidvalidity" else [b"765"]
            assert name == "UIDNEXT"
            return "UIDNEXT", [b"invalid"] if phase == "uidnext" else [b"2"]

        def uid(self, method, *args):
            actions.append(method)
            if method == "SEARCH":
                assert args[0] is None and args[1] == "UID"
                if phase == "search":
                    return "NO", [b"synthetic-secret-password"]
                return "OK", [b"1"] if phase == "headers" else [b""]
            assert method == "FETCH"
            uid, spec = args
            assert uid == "1"
            if "HEADER" in spec:
                if phase == "headers":
                    return "NO", [b"synthetic-secret-password"]
                sample = mail(subject="MRI results")
                header = sample.split(b"\n\n", 1)[0] + b"\r\n\r\n"
                return "OK", [(b"1 (UID 1 RFC822.SIZE " + str(len(sample)).encode() + b")", header)]
            assert spec == "(BODY.PEEK[])" and phase == "body"
            return "NO", [b"synthetic-secret-password"]

        def logout(self):
            actions.append("logout")
            return "BYE", [b"gone"]

    monkeypatch.setattr(ic.imaplib, "IMAP4_SSL", StageIMAP)
    app = FastAPI()
    app.include_router(ic.build_icloud_mail_router())
    http = TestClient(app)

    # The clinician's existing Clinical Key is the only new API authority.
    assert http.get("/clinical/icloud-mail" + endpoint).status_code == 401
    assert actions == []
    response = http.get("/clinical/icloud-mail" + endpoint,
                        headers={"X-Clinical-Key": "synthetic-clinical-key"})
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": expected_code}}
    assert f"icloud_mail_diagnostic code={expected_code}" in caplog.text
    for forbidden in ("synthetic-secret-password", "test-primary-username",
                      "patient private provider error", "sample@example.invalid"):
        assert forbidden not in caplog.text
        assert forbidden not in response.text
    assert "examine" not in actions if phase in ("connect", "connect_timeout", "auth", "auth_timeout", "auth_rejected") else "examine" in actions
    assert "STORE" not in actions and "EXPUNGE" not in actions


def test_config_diagnostic_never_leaks_secret(client, monkeypatch, caplog):
    http, _ = client
    monkeypatch.delenv("ICLOUD_CLINICAL_READ_ENABLED")
    off = get(http, "/peek")
    assert off.status_code == 503
    assert off.json() == {"detail": {"code": "CONFIG_DISABLED"}}
    monkeypatch.setenv("ICLOUD_CLINICAL_READ_ENABLED", "1")
    monkeypatch.delenv("ICLOUD_CLINICAL_APP_PASSWORD")
    missing = get(http, "/peek")
    assert missing.status_code == 503
    assert missing.json() == {"detail": {"code": "CONFIG_MISSING"}}
    assert "synthetic-clinical-key" not in caplog.text


def test_unexpected_imap_exception_has_static_fallback(client, monkeypatch, caplog):
    http, _ = client

    @contextmanager
    def provider_exception(_username, _password):
        raise RuntimeError("synthetic-secret-password private patient subject")
        yield

    monkeypatch.setattr(ic, "_session", provider_exception)
    response = get(http, "/peek")
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "IMAP_UNEXPECTED_FAILURE"}}
    assert "icloud_mail_diagnostic code=IMAP_UNEXPECTED_FAILURE" in caplog.text
    assert "synthetic-secret-password" not in caplog.text
    assert "private patient subject" not in response.text
