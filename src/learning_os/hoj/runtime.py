"""Single-owner local storage and fair, bounded-batch inference admission."""
from contextlib import contextmanager
import threading
import weakref

from learning_os.hoj.auth import check_deadline, now_ms
from learning_os.hoj.contracts import ServiceError


@contextmanager
def locked(lock, deadline):
    while True:
        check_deadline(deadline)
        remaining = max(.001, min(.1, (deadline - now_ms()) / 1000))
        if lock.acquire(timeout=remaining):
            break
    try:
        check_deadline(deadline)
        yield
    finally:
        lock.release()


class StorageOwner:
    def __init__(self):
        self.lock = threading.RLock()
        self.ingest_lock = threading.Lock()


class InferenceGate:
    """One model call at a time; queries run between document batches.

    A waiting document gets a turn after at most four query calls, avoiding
    ingestion starvation. Expired waiters leave the queue before encoding.
    """
    def __init__(self):
        self._condition = threading.Condition()
        self._active = False
        self._owner_thread = None
        self._waiting_queries = 0
        self._waiting_documents = 0
        self._query_streak = 0

    def run(self, operation, deadline, *, query):
        with self._condition:
            if self._owner_thread == threading.get_ident():
                raise ServiceError(503, "INFERENCE_REENTRANCY")
            if query:
                self._waiting_queries += 1
            else:
                self._waiting_documents += 1
            self._condition.notify_all()
            try:
                while True:
                    check_deadline(deadline)
                    other_turn = (self._waiting_documents > 0 and self._query_streak >= 4) if query else (
                        self._waiting_queries > 0 and self._query_streak < 4)
                    if not self._active and not other_turn:
                        self._active = True
                        self._owner_thread = threading.get_ident()
                        break
                    self._condition.wait(max(.001, min(.1, (deadline - now_ms()) / 1000)))
            finally:
                if query:
                    self._waiting_queries -= 1
                else:
                    self._waiting_documents -= 1
                self._condition.notify_all()
        try:
            check_deadline(deadline)
            result = operation()
            check_deadline(deadline)
            return result
        finally:
            with self._condition:
                self._active = False
                self._owner_thread = None
                self._query_streak = self._query_streak + 1 if query else 0
                self._condition.notify_all()


_registry_lock = threading.Lock()
_storage = weakref.WeakKeyDictionary()
_inference = weakref.WeakKeyDictionary()


def storage_owner(client):
    with _registry_lock:
        owner = _storage.get(client)
        if owner is None:
            owner = StorageOwner()
            _storage[client] = owner
        return owner


def inference_owner(embedder):
    with _registry_lock:
        owner = _inference.get(embedder)
        if owner is None:
            owner = InferenceGate()
            _inference[embedder] = owner
        return owner
