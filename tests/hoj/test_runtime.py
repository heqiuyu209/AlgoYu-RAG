"""Bounded inference fairness, expiry and owner cleanup without network I/O."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import ServiceError
from learning_os.hoj.runtime import InferenceGate, locked


def test_waiting_document_runs_after_at_most_four_queries():
    gate = InferenceGate()
    entered, release = Event(), Event()
    order = []
    deadline = now_ms() + 30000

    def hold():
        entered.set()
        assert release.wait(5)

    with ThreadPoolExecutor(max_workers=7) as pool:
        active = pool.submit(gate.run, hold, deadline, query=False)
        try:
            assert entered.wait(2)
            queries = [pool.submit(gate.run, lambda: order.append("q"), deadline, query=True) for _ in range(5)]
            document = pool.submit(gate.run, lambda: order.append("d"), deadline, query=False)
            with gate._condition:
                assert gate._condition.wait_for(
                    lambda: gate._waiting_queries == 5 and gate._waiting_documents == 1, timeout=2)
        finally:
            release.set()
        active.result(timeout=3)
        for query in queries:
            query.result(timeout=3)
        document.result(timeout=3)
    assert order == ["q", "q", "q", "q", "d", "q"]


@pytest.mark.parametrize("query", [False, True])
def test_expired_waiter_leaves_queue_without_executing(query):
    gate = InferenceGate()
    entered, release = Event(), Event()
    calls = []

    def hold():
        entered.set()
        assert release.wait(5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(gate.run, hold, now_ms() + 30000, query=False)
        try:
            assert entered.wait(2)
            waiting = pool.submit(gate.run, lambda: calls.append(True), now_ms() + 100, query=query)
            with pytest.raises(ServiceError) as error:
                waiting.result(timeout=2)
            assert error.value.code == "DEADLINE_EXCEEDED"
            with gate._condition:
                assert gate._waiting_queries == gate._waiting_documents == 0
            assert calls == []
        finally:
            release.set()
        active.result(timeout=3)
    assert gate.run(lambda: "available", now_ms() + 30000, query=True) == "available"


@pytest.mark.parametrize("query", [False, True])
def test_encoder_failure_releases_inference_owner(query):
    gate = InferenceGate()

    def fail():
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        gate.run(fail, now_ms() + 30000, query=query)
    assert gate.run(lambda: "available", now_ms() + 30000, query=not query) == "available"
    assert gate._waiting_queries == gate._waiting_documents == 0


def test_nested_inference_fails_without_deadlocking_owner():
    gate = InferenceGate()
    deadline = now_ms() + 30000
    with pytest.raises(ServiceError) as error:
        gate.run(lambda: gate.run(lambda: None, deadline, query=True), deadline, query=False)
    assert error.value.code == "INFERENCE_REENTRANCY"
    assert gate.run(lambda: "available", deadline, query=True) == "available"


def test_storage_wait_respects_deadline_and_does_not_unlock_another_owner():
    lock = Lock()
    lock.acquire()

    def waiting():
        with locked(lock, now_ms() + 100):
            pytest.fail("expired waiter acquired the storage lock")

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(ServiceError) as error:
                pool.submit(waiting).result(timeout=2)
            assert error.value.code == "DEADLINE_EXCEEDED"
            assert lock.locked()
    finally:
        lock.release()
    with locked(lock, now_ms() + 30000):
        assert lock.locked()
