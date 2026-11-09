import pandas as pd

from vector_etl.idempotency import IdempotencyRegistry, compute_fingerprint


def test_compute_fingerprint_is_stable_for_same_range_and_content():
    df = pd.DataFrame({"val": range(5)})
    prefix1, full1 = compute_fingerprint(10, 14, df)
    prefix2, full2 = compute_fingerprint(10, 14, df.copy())

    assert prefix1 == prefix2
    assert full1 == full2
    assert full1.startswith(prefix1)
    assert len(prefix1) == 16


def test_compute_fingerprint_differs_for_different_content_or_range():
    df_a = pd.DataFrame({"val": range(5)})
    df_b = pd.DataFrame({"val": range(5, 10)})

    _, full_same_range_diff_content = compute_fingerprint(10, 14, df_b)
    _, full_original = compute_fingerprint(10, 14, df_a)
    assert full_original != full_same_range_diff_content

    _, full_diff_range_same_content = compute_fingerprint(20, 24, df_a)
    assert full_original != full_diff_range_same_content


def test_registry_recognizes_exact_repeat_as_already_written():
    registry = IdempotencyRegistry()
    prefix, full_hash = "abc123abc123abc1", "abc123abc123abc1" + "0" * 48

    with registry.get_lock(prefix):
        assert registry.is_recorded(prefix, full_hash) is False
        registry.record(prefix, full_hash)

    with registry.get_lock(prefix):
        assert registry.is_recorded(prefix, full_hash) is True


def test_registry_does_not_misjudge_prefix_collision_as_already_written():
    """Two different batches whose fingerprints happen to share the same
    16-char prefix must not be confused with one another: only an exact
    full-hash match should count as 'already written'.
    """
    registry = IdempotencyRegistry()
    prefix = "deadbeefdeadbeef"
    full_hash_a = prefix + "a" * 48
    full_hash_b = prefix + "b" * 48

    lock_a = registry.get_lock(prefix)
    lock_b = registry.get_lock(prefix)
    assert lock_a is lock_b  # same prefix must serialize through one lock

    with lock_a:
        assert registry.is_recorded(prefix, full_hash_a) is False
        registry.record(prefix, full_hash_a)

    with lock_b:
        # Different content, colliding prefix: must not be treated as a
        # duplicate of batch A's write.
        assert registry.is_recorded(prefix, full_hash_b) is False
        registry.record(prefix, full_hash_b)

    with registry.get_lock(prefix):
        assert registry.is_recorded(prefix, full_hash_a) is True
        assert registry.is_recorded(prefix, full_hash_b) is True
