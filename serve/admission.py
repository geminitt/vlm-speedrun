"""Admission control for a server that can only work on one request at a time.

Without a bound, every extra request waits in line and latency grows without limit
under overload. With one, requests beyond the bound are rejected at once, so a
client can retry elsewhere instead of waiting for a reply that will be too late.
"""
import threading


class Admission:
    def __init__(self, max_in_flight):
        self.max_in_flight = max_in_flight
        self.in_flight = 0
        self._lock = threading.Lock()

    def try_enter(self):
        """Claim a slot (waiting or running). False means: reject the request."""
        with self._lock:
            if self.in_flight >= self.max_in_flight:
                return False
            self.in_flight += 1
            return True

    def leave(self):
        with self._lock:
            self.in_flight -= 1
