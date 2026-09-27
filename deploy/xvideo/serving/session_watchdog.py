"""Separate an idle client from bounded asynchronous processing stalls."""
class SessionWatchdog:
    def __init__(self, idle_timeout=10.0, stall_timeout=120.0):
        self.idle_timeout = float(idle_timeout)
        self.stall_timeout = max(self.idle_timeout, float(stall_timeout))
        self.busy_since = None
        self.progress_at = None
        self.progress = None

    def check(self, *, now, last_activity, pending, progress):
        if pending:
            if self.busy_since is None:
                self.busy_since = self.progress_at = now
                self.progress = progress
            elif progress != self.progress:
                self.progress_at = now
                self.progress = progress
            # More input alone cannot keep a stuck worker alive indefinitely.
            if now - self.progress_at >= self.stall_timeout:
                return 'stalled'
            return None
        if self.busy_since is not None:
            # Give the client a normal idle interval after queued work drains.
            self.progress_at = now
        self.busy_since = None
        if now - max(last_activity, self.progress_at or last_activity) >= self.idle_timeout:
            return 'idle'
        return None
