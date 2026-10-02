"""Thread-safe, in-memory chat rate limiter for a single application process."""

import asyncio
import math
import time
from collections import defaultdict, deque
from collections.abc import Callable


class InMemoryRateLimiter:
    """Sliding-window limiter; state is process-local and not horizontally shared."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 1 or window_seconds < 1:
            raise ValueError("Rate-limit values must be positive")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def consume(self, key: str) -> tuple[bool, int]:
        """Consume one slot; return allowed and suggested Retry-After seconds."""
        now = self._clock()
        cutoff = now - self.window_seconds
        async with self._lock:
            for bucket_key, bucket in list(self._events.items()):
                while bucket and bucket[0] <= cutoff:
                    bucket.popleft()
                if not bucket:
                    del self._events[bucket_key]
            events = self._events.setdefault(key, deque())
            if len(events) >= self.max_requests:
                retry_after = max(1, math.ceil(self.window_seconds - (now - events[0])))
                return False, retry_after
            events.append(now)
            return True, 0
