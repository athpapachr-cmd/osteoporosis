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
         sender="patient@example.invalid", newsletter=False, attached=False):
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
            assert args == (None, "OR", "TO", '"' + ALIAS + '"',
                            "CC", '"' + ALIAS + '"')
            return "OK", [b"1 2 3 4"]
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
    assert get(http, "/items/765-999").status_code == 503
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
