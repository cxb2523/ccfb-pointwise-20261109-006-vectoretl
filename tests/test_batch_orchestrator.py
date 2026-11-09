import json
import threading
import time

import pandas as pd
import pytest

from vector_etl import Orchestrator
from vector_etl.batching import RowBatch


class StreamingLargeFileSource:
    """A mock "large file" source that streams rows lazily.

    Real file-shaped data (>128 rows, enough to force several
    read-batches and several write-splits), produced by a generator so
    the orchestrator never has to hold the whole file in memory.
    """

    def __init__(self, total_rows, payload_chars=200):
        self.total_rows = total_rows
        self.payload_chars = payload_chars

    def iter_rows(self):
        for i in range(self.total_rows):
            yield {"id": i, "text": f"line {i} " + ("a" * self.payload_chars)}


class FastMockEmbedding:
    """Deterministic, instantaneous stand-in for a real embedding call."""

    def embed(self, df, embed_column="__concat_final"):
        df = df.copy()
        df["embeddings"] = [[0.0, 0.0] for _ in range(len(df))]
        return df


class ScriptedFailureTarget:
    """Mock destination used to drive the retry/skip/concurrency scenario.

    - ``flaky_range`` fails on its first two write attempts, then
      succeeds on the third.
    - ``dead_range`` fails on every attempt, forcing the orchestrator
      to exhaust retries and skip it.
    - Every other batch succeeds on the first attempt.

    Call bookkeeping is guarded by a lock because the orchestrator
    dispatches batches across a thread pool.
    """

    def __init__(self, flaky_range, dead_range):
        self.flaky_range = flaky_range
        self.dead_range = dead_range
        self._lock = threading.Lock()
        self.call_counts = {}
        self.seen_ids_by_range = {}

    def write_data(self, df, columns, domain=None):
        start = int(df["__row_index"].iloc[0])
        end = int(df["__row_index"].iloc[-1])
        key = (start, end)

        with self._lock:
            count = self.call_counts.get(key, 0) + 1
            self.call_counts[key] = count
            self.seen_ids_by_range[key] = df["id"].tolist()

        if key == self.dead_range:
            raise RuntimeError(f"destination permanently unavailable for batch {key}")
        if key == self.flaky_range and count < 3:
            raise RuntimeError(f"transient destination error for batch {key}, attempt {count}")

        return True


class CountingTarget:
    def __init__(self):
        self.call_count = 0
        self._lock = threading.Lock()

    def write_data(self, df, columns, domain=None):
        with self._lock:
            self.call_count += 1
        return True


TOTAL_ROWS = 1300
FLAKY_RANGE = (500, 627)
DEAD_RANGE = (1000, 1127)

# With batch_size=500 and a 128-row write cap, the read-batches
# [0-499], [500-999], [1000-1299] split into write-batches:
#   [0-127] [128-255] [256-383] [384-499]
#   [500-627] [628-755] [756-883] [884-999]
#   [1000-1127] [1128-1255] [1256-1299]
EXPECTED_WRITE_RANGES = [
    (0, 127), (128, 255), (256, 383), (384, 499),
    (500, 627), (628, 755), (756, 883), (884, 999),
    (1000, 1127), (1128, 1255), (1256, 1299),
]


@pytest.fixture
def scripted_target():
    return ScriptedFailureTarget(flaky_range=FLAKY_RANGE, dead_range=DEAD_RANGE)


def run_scripted_orchestrator(tmp_path, target):
    report_path = tmp_path / "report.json"
    orchestrator = Orchestrator(
        source_config=StreamingLargeFileSource(TOTAL_ROWS),
        embedding_config=FastMockEmbedding(),
        target_config=target,
        embed_columns=["text"],
        batch_size=500,
        max_write_batch_size=128,
        max_retries=3,
        max_workers=4,
        report_path=str(report_path),
    )
    result = orchestrator.run()
    return orchestrator, result, report_path


def test_large_file_batches_split_retry_skip_and_partial_success(tmp_path, scripted_target):
    start = time.monotonic()
    orchestrator, result, report_path = run_scripted_orchestrator(tmp_path, scripted_target)
    elapsed = time.monotonic() - start

    dead_len = DEAD_RANGE[1] - DEAD_RANGE[0] + 1

    # -- overall partial-success accounting, kept explicit and separate --
    assert result.status == "partial_success"
    assert result.total_rows == TOTAL_ROWS
    assert result.failed_count == dead_len
    assert result.success_count == TOTAL_ROWS - dead_len

    # -- skipped batch is reported with its row range + a usable traceback --
    assert len(result.skipped_batches) == 1
    skipped = result.skipped_batches[0]
    assert (skipped.start_row, skipped.end_row) == DEAD_RANGE
    assert "destination permanently unavailable" in skipped.error
    assert "RuntimeError" in skipped.traceback

    # -- retry counts: flaky batch retried twice then succeeded, dead batch
    #    exhausted the max_retries=3 budget before being skipped --
    assert result.retries["500-627"] == 2
    assert result.retries["1000-1127"] == 3
    for other_start, other_end in EXPECTED_WRITE_RANGES:
        if (other_start, other_end) in (FLAKY_RANGE, DEAD_RANGE):
            continue
        assert result.retries[f"{other_start}-{other_end}"] == 0

    # -- every write-batch was dispatched exactly once at the splitting
    #    layer, and exactly the expected number of times against the
    #    destination (no batch silently dropped, none duplicated) --
    assert set(scripted_target.call_counts.keys()) == set(EXPECTED_WRITE_RANGES)
    assert scripted_target.call_counts[FLAKY_RANGE] == 3
    assert scripted_target.call_counts[DEAD_RANGE] == 4
    for key in EXPECTED_WRITE_RANGES:
        if key in (FLAKY_RANGE, DEAD_RANGE):
            continue
        assert scripted_target.call_counts[key] == 1

    # -- row -> vector correspondence survived the split: each batch the
    #    destination saw contains exactly the absolute row ids for its range,
    #    in order, never shuffled or mixed with another batch's rows --
    for (rstart, rend), ids in scripted_target.seen_ids_by_range.items():
        assert ids == list(range(rstart, rend + 1))

    # -- concurrency actually happened: the flaky (0.5s+1s backoff) and
    #    dead (0.5s+1s+2s backoff) batches overlap on different worker
    #    threads instead of summing serially (~5s minimum if serialized).
    assert elapsed < 4.5

    # -- JSON report on disk matches the returned stats item by item --
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)

    result_dict = result.to_dict()
    assert report == result_dict
    assert report["status"] == result.status
    assert report["total_rows"] == result.total_rows
    assert report["success_count"] == result.success_count
    assert report["failed_count"] == result.failed_count
    assert report["skipped_batches"] == [b.to_dict() for b in result.skipped_batches]
    assert report["retries"] == result.retries


def test_concurrent_duplicate_dispatch_is_blocked_by_idempotency_check(tmp_path):
    target = CountingTarget()
    orchestrator = Orchestrator(
        source_config=object(),
        embedding_config=FastMockEmbedding(),
        target_config=target,
        embed_columns=["text"],
        report_path=None,
    )

    df = pd.DataFrame({
        "id": range(10),
        "text": [f"t{i}" for i in range(10)],
        "__row_index": range(10),
    })
    batch = RowBatch(0, 9, df)

    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        orchestrator._process_write_batch(batch)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The idempotency check (serialized by the per-fingerprint lock)
    # must prevent the same batch from being written twice, even though
    # two threads raced to process it concurrently.
    assert target.call_count == 1
    assert orchestrator._success_count == len(df)
    assert orchestrator._failed_count == 0
