"""Synthetic cryptographic storage and concurrency invariants."""
from datetime import date, datetime, timedelta
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import clinical_visit_preparation_store as store


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv('VISIT_PREPARATION_KEY','YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXowMTIzNDU=')
    engine=create_engine('sqlite:///'+str(tmp_path/'synthetic.db'))
    store.PreparationBase.metadata.create_all(engine)
    _,signing=store.keys()
    with Session(engine) as session:
        session.add(store.PreparationDraft(booking_key=store.MARKER,booking_date='marker',
            expires_at=datetime(9999,12,31),generation=store.marker_value(signing)))
        session.commit()
    return engine


def test_ciphertext_restart_tamper_and_independent_bookings(db):
    day=date.today()+timedelta(days=1)
    k,t=store.reserve(db,'cal.com:one',day)
    store.publish(db,k,t,{'message':'Συνθετικό δεδομένο','generation':t},datetime.now(store.ZONE).date())
    other,u=store.reserve(db,'cal.com:two',day)
    store.publish(db,other,u,{'message':'Άλλο','generation':u},datetime.now(store.ZONE).date())
    assert store.read(db,'cal.com:one',day)['message']=='Συνθετικό δεδομένο'
    with Session(db) as session:
        row=session.get(store.PreparationDraft,k);second=session.get(store.PreparationDraft,other)
        assert 'Συνθετικό'.encode() not in row.ciphertext
        assert row.nonce!=second.nonce
        row.ciphertext=second.ciphertext;row.nonce=second.nonce;session.commit()
    with pytest.raises(HTTPException) as error:store.read(db,'cal.com:one',day)
    assert error.value.status_code==503


def test_deletion_cannot_resurrect_old_attempt(db):
    day=date.today()+timedelta(days=1)
    key,old=store.reserve(db,'cal.com:one',day)
    store.remove(db,'cal.com:one')
    key,new=store.reserve(db,'cal.com:one',day)
    with pytest.raises(HTTPException):store.publish(db,key,old,{'generation':old},datetime.now(store.ZONE).date())
    store.publish(db,key,new,{'generation':new},datetime.now(store.ZONE).date())
    assert store.read(db,'cal.com:one',day)['generation']==new


def test_failing_attempt_preserves_last_draft(db):
    day=date.today()+timedelta(days=1)
    key,old=store.reserve(db,'cal.com:one',day);store.publish(db,key,old,{'generation':old},datetime.now(store.ZONE).date())
    _,new=store.reserve(db,'cal.com:one',day);store.cancel(db,key,new)
    assert store.read(db,'cal.com:one',day)['generation']==old


def test_missing_and_wrong_marker_fail_closed_and_cleanup_keeps_marker(db,monkeypatch):
    store.cleanup(db)
    with Session(db) as session:
        assert session.get(store.PreparationDraft,store.MARKER) is not None
        session.delete(session.get(store.PreparationDraft,store.MARKER));session.commit()
    with pytest.raises(HTTPException):store.reserve(db,'cal.com:one',date.today()+timedelta(days=1))


def test_expiry_uses_cyprus_dst_midnight():
    assert store._expiry(date(2026,3,28)).isoformat()=='2026-03-28T22:00:00'
    assert store._expiry(date(2026,3,29)).isoformat()=='2026-03-29T21:00:00'


def test_database_aware_clock_is_normalised_to_utc():
    from datetime import timezone
    class Dialect: name='postgresql'
    class Bind: dialect=Dialect()
    class DB:
        bind=Bind()
        def __init__(self,value):self.value=value
        def scalar(self,_):return self.value
    target=datetime(2026,11,18,12,0,tzinfo=timezone.utc)
    for offset in (-8,3,11):
        observed=target.astimezone(timezone(timedelta(hours=offset)))
        assert store._now(DB(observed))==datetime(2026,11,18,12,0)
