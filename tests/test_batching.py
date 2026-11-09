import tracemalloc

import pandas as pd
import pytest

from vector_etl.batching import (
    RowBatch,
    iter_row_batches,
    split_oversized_batch,
    ROW_INDEX_COLUMN,
)


def _rows(n, payload_chars=50):
    for i in range(n):
        yield {"id": i, "text": f"row-{i}-" + ("x" * payload_chars)}


def test_iter_row_batches_sizes_and_ranges():
    batches = list(iter_row_batches(_rows(1300), batch_size=500))

    assert [len(b) for b in batches] == [500, 500, 300]
    assert [(b.start_row, b.end_row) for b in batches] == [(0, 499), (500, 999), (1000, 1299)]

    # row index column is attached and matches absolute position
    all_rows = pd.concat([b.frame for b in batches], ignore_index=True)
    assert all_rows[ROW_INDEX_COLUMN].tolist() == list(range(1300))
    assert all_rows["id"].tolist() == list(range(1300))


def test_iter_row_batches_exact_multiple_has_no_trailing_empty_batch():
    batches = list(iter_row_batches(_rows(1000), batch_size=500))
    assert len(batches) == 2
    assert all(len(b) == 500 for b in batches)


def test_iter_row_batches_rejects_non_positive_batch_size():
    with pytest.raises(ValueError):
        list(iter_row_batches(_rows(10), batch_size=0))


def test_iter_row_batches_peak_memory_does_not_scale_with_total_rows():
    payload_chars = 2000

    def measure_peak(total_rows):
        tracemalloc.start()
        tracemalloc.clear_traces()
        before, _ = tracemalloc.get_traced_memory()
        for batch in iter_row_batches(_rows(total_rows, payload_chars), batch_size=200):
            # Touch the batch then drop it; a correct generator never
            # needs to keep more than one batch alive at a time.
            len(batch.frame)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak - before

    small_peak = measure_peak(2_000)
    large_peak = measure_peak(20_000)

    # If the implementation regressed to "read everything up front",
    # peak memory for 10x the rows would also be ~10x. Bounded batching
    # should keep it close to flat.
    assert large_peak < small_peak * 3


def test_split_oversized_batch_returns_original_when_within_limit():
    df = pd.DataFrame({"val": range(50)})
    batch = RowBatch(10, 59, df)
    subs = split_oversized_batch(batch, max_size=128)
    assert subs == [batch]


def test_split_oversized_batch_preserves_row_to_value_mapping():
    total = 500
    base_offset = 1000
    df = pd.DataFrame({"val": range(total)})
    batch = RowBatch(base_offset, base_offset + total - 1, df)

    subs = split_oversized_batch(batch, max_size=128)

    assert [len(s) for s in subs] == [128, 128, 128, 116]

    expected_starts = [1000, 1128, 1256, 1384]
    expected_ends = [1127, 1255, 1383, 1499]
    assert [s.start_row for s in subs] == expected_starts
    assert [s.end_row for s in subs] == expected_ends

    # contiguous, non-overlapping coverage of the original range
    for prev, nxt in zip(subs, subs[1:]):
        assert prev.end_row + 1 == nxt.start_row

    # merging the sub-batches back must reproduce the original order
    merged = pd.concat([s.frame for s in subs], ignore_index=True)
    assert merged["val"].tolist() == df["val"].tolist()

    # each row's absolute position still lines up with its original value
    for sub in subs:
        offset = sub.start_row - base_offset
        expected_values = list(range(offset, offset + len(sub)))
        assert sub.frame["val"].tolist() == expected_values


def test_split_oversized_batch_rejects_non_positive_max_size():
    df = pd.DataFrame({"val": range(10)})
    batch = RowBatch(0, 9, df)
    with pytest.raises(ValueError):
        split_oversized_batch(batch, max_size=0)
