"""Connector-layer batching utilities.

Provides a memory-bounded way to turn a lazy row source into fixed-size
``RowBatch`` objects, and a way to split an oversized batch into
sub-batches that respect a target's single-write row cap while keeping
an exact, re-mappable link back to the original absolute row numbers.
"""

import logging
from dataclasses import dataclass

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500
MAX_WRITE_BATCH_SIZE = 128

# Column added to every row so that splitting/merging batches can never
# desync the row<->vector correspondence: it always carries the row's
# absolute, 0-indexed position in the original source.
ROW_INDEX_COLUMN = "__row_index"


@dataclass
class RowBatch:
    """A contiguous slice of source rows, tagged with its absolute range.

    ``start_row``/``end_row`` are 0-indexed and inclusive, and refer to
    the row's position in the overall source, not the position within
    this batch.
    """

    start_row: int
    end_row: int
    frame: pd.DataFrame

    def __len__(self):
        return len(self.frame)

    def __post_init__(self):
        if len(self.frame) != (self.end_row - self.start_row + 1):
            raise ValueError(
                f"RowBatch range [{self.start_row}, {self.end_row}] does not "
                f"match frame length {len(self.frame)}"
            )


def iter_row_batches(row_iterable, batch_size=DEFAULT_BATCH_SIZE):
    """Consume an iterable/generator of rows and yield RowBatch objects.

    ``row_iterable`` can be any iterable of dict-likes or pandas Series.
    Only ``batch_size`` rows are ever held in memory at once (plus
    whatever buffering the caller's own generator does), so peak memory
    does not grow with the total number of rows in the source.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    buffer = []
    start_row = 0
    row_index = 0

    for row in row_iterable:
        record = dict(row)
        record[ROW_INDEX_COLUMN] = row_index
        buffer.append(record)
        row_index += 1

        if len(buffer) == batch_size:
            end_row = start_row + len(buffer) - 1
            yield RowBatch(start_row, end_row, pd.DataFrame(buffer))
            start_row = end_row + 1
            buffer = []

    if buffer:
        end_row = start_row + len(buffer) - 1
        yield RowBatch(start_row, end_row, pd.DataFrame(buffer))


def split_oversized_batch(batch, max_size=MAX_WRITE_BATCH_SIZE):
    """Split a RowBatch into sub-batches no larger than ``max_size``.

    Sub-batches tile the parent's absolute row range contiguously, with
    no gaps or overlaps, and preserve row order exactly (so the row ->
    vector correspondence survives the split). If the batch already
    fits, a single-element list containing the original batch is
    returned.
    """
    if max_size <= 0:
        raise ValueError("max_size must be a positive integer")

    total = len(batch)
    if total <= max_size:
        return [batch]

    sub_batches = []
    offset = 0
    while offset < total:
        chunk = batch.frame.iloc[offset:offset + max_size].reset_index(drop=True)
        sub_start = batch.start_row + offset
        sub_end = sub_start + len(chunk) - 1
        sub_batches.append(RowBatch(sub_start, sub_end, chunk))
        offset += max_size

    return sub_batches
