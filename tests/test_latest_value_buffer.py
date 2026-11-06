import threading
import time
from concurrent.futures import ThreadPoolExecutor

from visual_servo_system.runtime.latest_value_buffer import LatestValueBuffer


def test_publish_replaces_old_value_and_advances_sequence() -> None:
    buffer = LatestValueBuffer[str]()

    assert buffer.get_latest() is None
    assert buffer.publish("old") == 1
    assert buffer.publish("new") == 2
    assert buffer.get_latest() == (2, "new")


def test_wait_for_newer_unblocks_after_publish() -> None:
    buffer = LatestValueBuffer[str]()
    sequence = buffer.publish("first")

    def delayed_publish() -> None:
        time.sleep(0.03)
        buffer.publish("second")

    thread = threading.Thread(target=delayed_publish)
    thread.start()
    try:
        assert buffer.wait_for_newer(sequence, timeout_s=0.5) == (2, "second")
    finally:
        thread.join()


def test_wait_for_newer_returns_none_on_timeout() -> None:
    buffer = LatestValueBuffer[int]()

    started = time.monotonic()
    assert buffer.wait_for_newer(0, timeout_s=0.02) is None
    assert time.monotonic() - started >= 0.015


def test_buffer_rejects_invalid_wait_arguments() -> None:
    buffer = LatestValueBuffer[int]()
    try:
        buffer.wait_for_newer(-1, timeout_s=0.1)
    except ValueError as error:
        assert "sequence" in str(error)
    else:
        raise AssertionError("negative sequence must be rejected")


def test_multiple_producers_receive_unique_sequences_under_contention() -> None:
    buffer = LatestValueBuffer[tuple[int, int]]()
    producer_count = 8
    publishes_per_producer = 250
    start_gate = threading.Barrier(producer_count)
    published_by_sequence: dict[int, tuple[int, int]] = {}
    result_lock = threading.Lock()

    def produce(producer_id: int) -> None:
        start_gate.wait()
        for local_index in range(publishes_per_producer):
            value = (producer_id, local_index)
            sequence = buffer.publish(value)
            with result_lock:
                published_by_sequence[sequence] = value

    with ThreadPoolExecutor(max_workers=producer_count) as executor:
        futures = [executor.submit(produce, producer_id) for producer_id in range(producer_count)]
        for future in futures:
            future.result(timeout=5.0)

    expected_count = producer_count * publishes_per_producer
    assert sorted(published_by_sequence) == list(range(1, expected_count + 1))
    latest = buffer.get_latest()
    assert latest is not None
    assert latest == (expected_count, published_by_sequence[expected_count])


def test_clear_invalidates_old_snapshot_and_advances_generation() -> None:
    buffer = LatestValueBuffer[str]()
    published = buffer.publish("dry-run")

    cleared = buffer.clear()

    assert cleared > published
    assert buffer.get_latest() is None
    assert buffer.wait_for_newer(cleared, 0.0) is None

