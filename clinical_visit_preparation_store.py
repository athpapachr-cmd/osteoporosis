"""Temporary, encrypted per-booking drafts. Schema is provisioned separately."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from fastapi import HTTPException
from sqlalchemy import Column, DateTime, LargeBinary, String, delete, func, select, update
from sqlalchemy.orm import DeclarativeBase, Session


class PreparationBase(DeclarativeBase):
    pass


class PreparationDraft(PreparationBase):
    __tablename__ = 'clinical_visit_preparation_drafts'
    booking_key = Column(String(64), primary_key=True)
    booking_date = Column(String(10), nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    generation = Column(String(64))
    nonce = Column(LargeBinary)
    ciphertext = Column(LargeBinary)
    pending_token = Column(String(64))
    pending_until = Column(DateTime)


MARKER = 'preparation-key-marker-v1'
ZONE = ZoneInfo('Asia/Nicosia')


def keys():
    try:
        raw = base64.urlsafe_b64decode(os.environ['VISIT_PREPARATION_KEY'] + '===')
        if len(raw) != 32:
            raise ValueError()
        key = HKDF(algorithm=hashes.SHA256(), length=64, salt=None,
                   info=b'cockpit-visit-preparation-v1').derive(raw)
        return key[:32], key[32:]
    except (KeyError, ValueError, TypeError):
        raise HTTPException(503, 'Preparation storage unavailable') from None


def marker_value(signing):
    return hmac.new(signing, b'key-marker-v1', hashlib.sha256).hexdigest()


def _marker(db, signing):
    marker = db.scalar(select(PreparationDraft).where(PreparationDraft.booking_key == MARKER).with_for_update())
    if not marker or not marker.generation or not hmac.compare_digest(marker.generation, marker_value(signing)):
        raise HTTPException(503, 'Preparation storage unavailable')


def _now(db):
    value = db.scalar(select(func.clock_timestamp() if db.bind.dialect.name == "postgresql" else func.current_timestamp()))
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _opaque(signing, uid):
    return hmac.new(signing, b'booking:cal.com\0' + uid.encode(), hashlib.sha256).hexdigest()


def _aad(row):
    return json.dumps(['visit-preparation-v1', row.booking_key, row.booking_date,
                       row.expires_at.isoformat(), row.generation], separators=(',', ':')).encode()


def _expiry(day):
    midnight = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=ZONE)
    return midnight.astimezone(timezone.utc).replace(tzinfo=None)


def reserve(engine, uid, day):
    _, signing = keys()
    token = secrets.token_hex(32)
    key = _opaque(signing, uid)
    expiry = _expiry(day)
    with Session(engine) as db, db.begin():
        _marker(db, signing)
        now = _now(db)
        if expiry <= now:
            raise HTTPException(409, 'Preparation expired')
        row = db.get(PreparationDraft, key, with_for_update=True)
        if not row:
            row = PreparationDraft(booking_key=key, booking_date=day.isoformat(), expires_at=expiry)
            db.add(row)
        elif row.booking_date != day.isoformat() or row.expires_at != expiry:
            row.booking_date, row.expires_at = day.isoformat(), expiry
            row.generation = row.nonce = row.ciphertext = None
        row.pending_token = token
        row.pending_until = min(expiry, now + timedelta(minutes=20))
        db.flush()
    return key, token


def cancel(engine, key, token):
    _, signing = keys()
    with Session(engine) as db, db.begin():
        _marker(db, signing)
        db.execute(update(PreparationDraft).where(PreparationDraft.booking_key == key,
                   PreparationDraft.pending_token == token).values(pending_token=None, pending_until=None))


def publish(engine, key, token, payload):
    encryption, signing = keys()
    with Session(engine) as db, db.begin():
        _marker(db, signing)
        now = _now(db)
        row = db.get(PreparationDraft, key, with_for_update=True)
        if not row or row.pending_token != token or row.expires_at <= now or not row.pending_until or row.pending_until <= now:
            raise HTTPException(409, 'Preparation changed or expired')
        row.generation = token
        row.nonce = os.urandom(12)
        row.ciphertext = AESGCM(encryption).encrypt(row.nonce, json.dumps(payload, ensure_ascii=False,
            sort_keys=True, separators=(',', ':')).encode(), _aad(row))
        row.pending_token = None
        row.pending_until = None
        db.flush()


def read(engine, uid, day):
    encryption, signing = keys()
    key = _opaque(signing, uid)
    try:
        with Session(engine) as db, db.begin():
            _marker(db, signing)
            row = db.get(PreparationDraft, key, with_for_update=True)
            if not row or row.booking_date != day.isoformat() or row.expires_at <= _now(db) or not row.ciphertext:
                return None
            result = json.loads(AESGCM(encryption).decrypt(row.nonce, row.ciphertext, _aad(row)))
            if row.expires_at <= _now(db):
                return None
            return result
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, 'Preparation storage unavailable') from None


def check_available(engine, uid, day, generation):
    _, signing = keys()
    with Session(engine) as db, db.begin():
        _marker(db, signing)
        row = db.get(PreparationDraft, _opaque(signing, uid), with_for_update=True)
        return bool(row and row.booking_date == day.isoformat() and row.expires_at > _now(db)
                    and row.generation == generation and row.ciphertext)


def remove(engine, uid):
    _, signing = keys()
    with Session(engine) as db, db.begin():
        _marker(db, signing)
        db.execute(delete(PreparationDraft).where(PreparationDraft.booking_key == _opaque(signing, uid)))


def cleanup(engine):
    _, signing = keys()
    with Session(engine) as db, db.begin():
        _marker(db, signing)
        now = _now(db)
        db.execute(delete(PreparationDraft).where(PreparationDraft.booking_key != MARKER,
                   PreparationDraft.expires_at <= now))
        db.execute(update(PreparationDraft).where(PreparationDraft.booking_key != MARKER,
                   PreparationDraft.pending_until <= now).values(pending_token=None, pending_until=None))
