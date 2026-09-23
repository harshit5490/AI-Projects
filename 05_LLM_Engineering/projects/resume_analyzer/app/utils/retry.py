import random


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
) -> float:
    """Calculate exponential backoff with jitter."""

    exponential_delay = base_delay * (2 ** attempt)

    jitter = random.uniform(0, 0.5)

    return min(
        exponential_delay + jitter,
        max_delay,
    )