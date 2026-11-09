"""Thread-safe idempotency bookkeeping for single-batch retries.

The fingerprint for a batch is derived from its absolute row range plus
a content hash, so retrying the exact same unit of work twice (whether
by this process or by a racing concurrent dispatch) can be recognized
and short-circuited instead of writing duplicate data.

Because only a 16-hex-character prefix of the sha256 digest is used as
the fingerprint, two different (row-range, content) pairs could in
theory collide on that prefix. The registry stores the full digest
alongside the prefix so a collision is never mistaken for "already
written": only an exact full-digest match counts as a hit.
"""

import hashlib
import threading


def compute_fingerprint(start_row, end_row, frame):
    """Return (prefix16, full_hash) for a batch's row range + content."""
    content = frame.to_csv(index=False)
    payload = f"{start_row}:{end_row}:{content}".encode("utf-8")
    full_hash = hashlib.sha256(payload).hexdigest()
    prefix = full_hash[:16]
    return prefix, full_hash


class IdempotencyRegistry:
    """Thread-safe store of "already written" batch fingerprints.

    Lookups and writes for a given prefix are serialized via a
    per-prefix lock, so two threads racing to process the exact same
    batch (same prefix *and* full hash) cannot both decide it's safe to
    write at the same time. Different content that happens to collide
    on the 16-char prefix is kept distinguishable because entries are
    keyed by the full hash underneath each prefix bucket.
    """

    def __init__(self):
        self._master_lock = threading.Lock()
        self._prefix_locks = {}
        # prefix -> {full_hash: "success"}
        self._entries = {}

    def get_lock(self, prefix):
        with self._master_lock:
            lock = self._prefix_locks.get(prefix)
            if lock is None:
                lock = threading.Lock()
                self._prefix_locks[prefix] = lock
            return lock

    def is_recorded(self, prefix, full_hash):
        """Return True if this exact (prefix, full_hash) already succeeded.

        Must be called while holding the lock returned by
        ``get_lock(prefix)`` to avoid a check-then-act race.
        """
        bucket = self._entries.get(prefix)
        if not bucket:
            return False
        return bucket.get(full_hash) == "success"

    def record(self, prefix, full_hash):
        """Record a successful write. Must be called under the prefix lock."""
        bucket = self._entries.setdefault(prefix, {})
        bucket[full_hash] = "success"
