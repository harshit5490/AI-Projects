from app.utils.retry import calculate_backoff


def test_backoff_increases_with_attempt():
    delay_0 = calculate_backoff(0)
    delay_1 = calculate_backoff(1)
    delay_2 = calculate_backoff(2)

    assert delay_0 >= 1.0
    assert delay_1 >= 2.0
    assert delay_2 >= 4.0


def test_backoff_does_not_exceed_maximum():
    delay = calculate_backoff(
        attempt=10,
        max_delay=10.0,
    )

    assert delay <= 10.0