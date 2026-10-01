import asyncio

import pytest

from app.security.rate_limit import InMemoryRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit_and_expires_window() -> None:
    now = [10.0]
    limiter = InMemoryRateLimiter(2, 5, clock=lambda: now[0])
    assert await limiter.consume("subject:a") == (True, 0)
    assert await limiter.consume("subject:a") == (True, 0)
    allowed, retry = await limiter.consume("subject:a")
    assert allowed is False and retry == 5
    now[0] += 5
    assert await limiter.consume("subject:a") == (True, 0)


@pytest.mark.asyncio
async def test_rate_limiters_isolate_different_subjects() -> None:
    limiter = InMemoryRateLimiter(1, 60, clock=lambda: 100.0)
    assert await limiter.consume("subject:alice") == (True, 0)
    assert await limiter.consume("subject:bob") == (True, 0)
    assert (await limiter.consume("subject:alice"))[0] is False


@pytest.mark.asyncio
async def test_concurrent_rate_limit_consumption_is_atomic() -> None:
    limiter = InMemoryRateLimiter(7, 60, clock=lambda: 100.0)
    results = await asyncio.gather(*(limiter.consume("client:one") for _ in range(40)))
    assert sum(allowed for allowed, _ in results) == 7
    assert all(retry >= 1 for allowed, retry in results if not allowed)


def test_rate_limiter_rejects_nonpositive_configuration() -> None:
    with pytest.raises(ValueError):
        InMemoryRateLimiter(0, 60)
    with pytest.raises(ValueError):
        InMemoryRateLimiter(10, 0)
