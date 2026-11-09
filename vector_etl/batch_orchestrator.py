"""Batched, retrying, partially-tolerant ETL orchestration.

This is the resilience-focused sibling of vector_etl.orchestrator.
ETLOrchestrator: instead of reading a source fully into memory and
raising on the first write failure, it:

  * reads the source through a generator bounded by ``batch_size``
    rows (connector-layer batching), so peak memory doesn't grow with
    the total row count,
  * aligns every write with the target's single-call row cap
    (``max_write_batch_size``), splitting oversized batches into
    sub-batches that keep an exact mapping back to their original
    absolute row range,
  * retries a failing sub-batch with backoff, using a row-range +
    content fingerprint to avoid double-writing on retry/duplicate
    dispatch,
  * runs sub-batches concurrently across a thread pool while keeping
    retry bookkeeping and idempotency checks thread-safe,
  * and finishes with an explicit partial-success result (success
    count, failed count, skipped row ranges) instead of letting one
    bad batch raise and erase everything already written.
"""

import logging
import time
import traceback

import pandas as pd

from .batching import (
    DEFAULT_BATCH_SIZE,
    MAX_WRITE_BATCH_SIZE,
    iter_row_batches,
    split_oversized_batch,
    RowBatch,
)
from .idempotency import IdempotencyRegistry, compute_fingerprint
from .results import ETLRunResult, SkippedBatch, derive_status
from .text_utils import generate_concat_column

import threading
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_BACKOFF_SCHEDULE = (0.5, 1, 2)
DEFAULT_REPORT_PATH = "vector_etl_report.json"


class Orchestrator:
    """Library entry point for the batched, resilient ETL pipeline.

    ``source_config``/``embedding_config``/``target_config`` may each be
    either a config dict (resolved via the existing source/embedding/
    target factories) or an already-constructed object implementing the
    corresponding interface, which makes this easy to exercise with
    test doubles.
    """

    def __init__(
        self,
        source_config,
        embedding_config,
        target_config,
        embed_columns=None,
        batch_size=DEFAULT_BATCH_SIZE,
        max_write_batch_size=MAX_WRITE_BATCH_SIZE,
        max_retries=3,
        backoff_schedule=DEFAULT_BACKOFF_SCHEDULE,
        max_workers=4,
        domain=None,
        report_path=DEFAULT_REPORT_PATH,
    ):
        self.source = self._resolve_source(source_config)
        self.embedding = self._resolve_embedding(embedding_config)
        self.target = self._resolve_target(target_config)
        self.source_config = source_config
        self.embed_columns = embed_columns or []
        self.batch_size = batch_size
        self.max_write_batch_size = max_write_batch_size
        self.max_retries = max_retries
        self.backoff_schedule = tuple(backoff_schedule)
        self.max_workers = max_workers
        self.report_path = report_path

        if domain is not None:
            self.domain = domain
        elif isinstance(source_config, dict):
            self.domain = source_config.get('table', source_config.get('source_data_type'))
        else:
            self.domain = None

        self._idempotency = IdempotencyRegistry()
        self._agg_lock = threading.Lock()
        self._success_count = 0
        self._failed_count = 0
        self._skipped = []
        self._retries = {}
        # Row ranges already reflected in success/failed counts. A given
        # absolute row range is only ever owned by one logical batch, so
        # this guards against double-counting if the same batch is ever
        # dispatched twice (e.g. a racing duplicate submission) instead
        # of just retried sequentially within one _process_write_batch call.
        self._counted_ranges = set()

    # -- resolution -------------------------------------------------

    @staticmethod
    def _resolve_source(source_config):
        if isinstance(source_config, dict):
            from vector_etl.source_mods import get_source_class
            return get_source_class(source_config)
        return source_config

    @staticmethod
    def _resolve_embedding(embedding_config):
        if isinstance(embedding_config, dict):
            from vector_etl.embedding_mods import get_embedding_model
            return get_embedding_model(embedding_config)
        return embedding_config

    @staticmethod
    def _resolve_target(target_config):
        if isinstance(target_config, dict):
            from vector_etl.target_mods import get_target_database
            return get_target_database(target_config)
        return target_config

    # -- reading ------------------------------------------------------

    def _iter_source_rows(self):
        """Adapt whatever the source exposes into a lazy row generator.

        Preferred path is an explicit ``iter_rows()`` generator (truly
        bounded memory). Falls back to ``fetch_data()``, which may
        itself already be a generator of DataFrame batches (as with
        DatabaseSource) or a single, fully materialized DataFrame.
        """
        if hasattr(self.source, "iter_rows"):
            for row in self.source.iter_rows():
                yield row
            return

        data = self.source.fetch_data()

        if isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                yield row
            return

        for item in data:
            if isinstance(item, pd.DataFrame):
                for _, row in item.iterrows():
                    yield row
            else:
                yield item

    def _embed_batch(self, row_batch):
        df = generate_concat_column(row_batch.frame, self.embed_columns)
        df = self.embedding.embed(df)
        return RowBatch(row_batch.start_row, row_batch.end_row, df)

    # -- writing with retry + idempotency ------------------------------

    @staticmethod
    def _range_key(write_batch):
        return f"{write_batch.start_row}-{write_batch.end_row}"

    def _record_success(self, write_batch, retries_used):
        key = self._range_key(write_batch)
        with self._agg_lock:
            self._retries[key] = retries_used
            if key in self._counted_ranges:
                logger.info(
                    "Batch [%s-%s] outcome already counted, not double-counting",
                    write_batch.start_row, write_batch.end_row,
                )
                return
            self._counted_ranges.add(key)
            self._success_count += len(write_batch)

    def _record_skip(self, write_batch, exc, tb_text, retries_used):
        key = self._range_key(write_batch)
        with self._agg_lock:
            self._retries[key] = retries_used
            if key in self._counted_ranges:
                logger.info(
                    "Batch [%s-%s] outcome already counted, not double-counting",
                    write_batch.start_row, write_batch.end_row,
                )
                return
            self._counted_ranges.add(key)
            self._failed_count += len(write_batch)
            self._skipped.append(
                SkippedBatch(
                    start_row=write_batch.start_row,
                    end_row=write_batch.end_row,
                    error=str(exc),
                    traceback=tb_text,
                )
            )

    def _process_write_batch(self, write_batch):
        """Write one sub-batch with retry + idempotency, never raising.

        The per-fingerprint lock serializes the whole
        check-idempotency / write / record sequence for this exact
        (row-range, content) unit, so a concurrent duplicate dispatch
        of the same batch is either blocked until the first attempt
        finishes (and then recognized as already-written) or correctly
        treated as distinct work if it only *collides* on the 16-char
        fingerprint prefix but carries different content.
        """
        prefix, full_hash = compute_fingerprint(
            write_batch.start_row, write_batch.end_row, write_batch.frame
        )
        lock = self._idempotency.get_lock(prefix)

        with lock:
            if self._idempotency.is_recorded(prefix, full_hash):
                logger.info(
                    "Batch [%s-%s] already written, skipping duplicate write",
                    write_batch.start_row, write_batch.end_row,
                )
                self._record_success(write_batch, retries_used=0)
                return

            last_exc = None
            last_tb = None
            max_attempts = self.max_retries + 1

            for attempt in range(1, max_attempts + 1):
                try:
                    self.target.write_data(
                        write_batch.frame, list(self.embed_columns), self.domain
                    )
                    self._idempotency.record(prefix, full_hash)
                    self._record_success(write_batch, retries_used=attempt - 1)
                    return
                except Exception as exc:  # noqa: BLE001 - must not abort the run
                    last_exc = exc
                    last_tb = traceback.format_exc()
                    logger.warning(
                        "Batch [%s-%s] attempt %d/%d failed: %s",
                        write_batch.start_row, write_batch.end_row,
                        attempt, max_attempts, exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(self.backoff_schedule[attempt - 1])

            logger.error(
                "Batch [%s-%s] exhausted retries, skipping",
                write_batch.start_row, write_batch.end_row,
            )
            self._record_skip(write_batch, last_exc, last_tb, retries_used=max_attempts - 1)

    # -- orchestration --------------------------------------------------

    def run(self):
        logger.info("Starting batched ETL run...")

        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for row_batch in iter_row_batches(self._iter_source_rows(), self.batch_size):
                embedded_batch = self._embed_batch(row_batch)
                for write_batch in split_oversized_batch(embedded_batch, self.max_write_batch_size):
                    futures.append(executor.submit(self._process_write_batch, write_batch))

            wait_futures(futures)

        for future in futures:
            # Re-raise unexpected (non-batch) errors, e.g. programming
            # bugs inside _process_write_batch itself, instead of
            # silently losing them. Normal batch failures are already
            # caught inside _process_write_batch and never land here.
            future.result()

        with self._agg_lock:
            success_count = self._success_count
            failed_count = self._failed_count
            skipped_batches = list(self._skipped)
            retries = dict(self._retries)

        status = derive_status(success_count, failed_count)
        result = ETLRunResult(
            status=status,
            total_rows=success_count + failed_count,
            success_count=success_count,
            failed_count=failed_count,
            skipped_batches=skipped_batches,
            retries=retries,
        )

        if self.report_path:
            result.write_report(self.report_path)
            logger.info("Wrote ETL run report to %s", self.report_path)

        logger.info(
            "Batched ETL run finished: status=%s success=%d failed=%d skipped_batches=%d",
            status, success_count, failed_count, len(skipped_batches),
        )
        return result
