"""Optional isolated PostgreSQL contract test; never connect to production."""
import os
import threading
import time
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import clinical_visit_preparation_store as store


@pytest.mark.skipif(not os.getenv('COCKPIT_SYNTHETIC_POSTGRES_URL'),reason='isolated PostgreSQL not provided')
def test_share_lock_fences_concurrent_insert_and_clock():
    url=os.environ['COCKPIT_SYNTHETIC_POSTGRES_URL']
    assert 'host=/tmp' in url and 'port=55483' in url
    engine=create_engine(url)
    with engine.begin() as db:
        db.execute(text('CREATE TABLE clinical_encounters (id text primary key, created_at timestamptz default clock_timestamp())'))
        db.execute(text('CREATE TABLE clinical_pending_items (id text primary key)'))
        db.execute(text('CREATE ROLE preparation_test_runtime'))
        db.execute(text('GRANT SELECT, INSERT, UPDATE, DELETE ON clinical_encounters, clinical_pending_items TO preparation_test_runtime'))
    started=threading.Event();finished=threading.Event();failure=[]
    def insert():
        try:
            with engine.begin() as db:
                db.execute(text('SET LOCAL ROLE preparation_test_runtime'))
                started.set()
                db.execute(text("INSERT INTO clinical_pending_items VALUES ('synthetic')"))
            finished.set()
        except Exception as error:failure.append(error)
    try:
        with engine.begin() as db:
            db.execute(text('SET LOCAL ROLE preparation_test_runtime'))
            db.execute(text("SET LOCAL lock_timeout = '2s'"))
            db.execute(text('LOCK TABLE clinical_encounters, clinical_pending_items IN SHARE MODE'))
            worker=threading.Thread(target=insert);worker.start()
            assert started.wait(2)
            assert not finished.wait(.25)
        assert finished.wait(3)
        worker.join(3)
        assert not failure
        # A writer which starts first commits before the waiting reader gets
        # SHARE. Its newly signed row must fall within the *post-lock* cutoff.
        writer=engine.connect();transaction=writer.begin()
        writer.execute(text('SET LOCAL ROLE preparation_test_runtime'))
        writer.execute(text("INSERT INTO clinical_encounters(id) VALUES ('new-synthetic')"))
        waiting=threading.Event();checked=[]
        def waiting_reader():
            try:
                with engine.begin() as reader:
                    reader.execute(text('SET LOCAL ROLE preparation_test_runtime'))
                    reader.execute(text("SET LOCAL lock_timeout = '2s'"))
                    waiting.set()
                    reader.execute(text('LOCK TABLE clinical_encounters, clinical_pending_items IN SHARE MODE'))
                    cutoff=reader.scalar(text('SELECT clock_timestamp()'))
                    checked.append(reader.scalar(text('SELECT count(*) FROM clinical_encounters WHERE created_at <= :cutoff'),{'cutoff':cutoff}))
            except Exception as error:failure.append(error)
        reader_thread=threading.Thread(target=waiting_reader);reader_thread.start();assert waiting.wait(2)
        time.sleep(.1);transaction.commit();writer.close();reader_thread.join(3)
        assert checked==[1] and not failure
        with Session(engine) as session:
            session.execute(text("SET TIME ZONE 'Pacific/Honolulu'"))
            now=store._now(session).replace(tzinfo=timezone.utc)
            assert abs((datetime.now(timezone.utc)-now).total_seconds())<5
    finally:
        with engine.begin() as db:
            db.execute(text('DROP TABLE clinical_pending_items, clinical_encounters'))
            db.execute(text('DROP ROLE preparation_test_runtime'))
        engine.dispose()
