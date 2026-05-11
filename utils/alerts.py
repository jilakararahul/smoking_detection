"""
SMS alert integration via Twilio.

Credentials are loaded from .streamlit/secrets.toml:

    [twilio]
    account_sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    auth_token  = "your_auth_token"
    from_number = "+14155238886"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SMSAlerter:
    """Sends SMS via Twilio with a cooldown to prevent alert flooding."""

    to_number: str
    cooldown_seconds: int = 30
    _last_sent: float = field(default=0.0, init=False, repr=False)
    _client: object = field(default=None, init=False, repr=False)

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from twilio.rest import Client
            import streamlit as st
            cfg = st.secrets["twilio"]
            self._client = Client(cfg["account_sid"], cfg["auth_token"])
            return self._client
        except KeyError:
            raise RuntimeError(
                "Twilio credentials not found in .streamlit/secrets.toml. "
                "Add a [twilio] section with account_sid, auth_token, from_number."
            )
        except ImportError:
            raise RuntimeError(
                "twilio package not installed. Run: pip install twilio"
            )

    def send(self, body: str) -> bool:
        """Send an SMS. Returns False if within cooldown, True if sent."""
        now = time.time()
        if now - self._last_sent < self.cooldown_seconds:
            return False

        import streamlit as st
        from_number = st.secrets["twilio"]["from_number"]
        client = self._get_client()
        client.messages.create(to=self.to_number, from_=from_number, body=body)
        self._last_sent = now
        return True

    def ready(self) -> bool:
        return (time.time() - self._last_sent) >= self.cooldown_seconds


class SlidingWindowTracker:
    """
    Tracks cigarette detections over a rolling time window and fires an
    alert once the detection ratio exceeds a threshold.

    Instead of requiring continuous presence (which resets on a single
    missed frame), this looks at the fraction of frames in the last
    ``window_seconds`` where a cigarette was detected. This tolerates
    brief occlusions, head turns, or single missed detections.

    Returns True exactly once per smoking event. Resets automatically
    when the cigarette has been mostly absent (ratio < reset_ratio).
    """

    def __init__(
        self,
        window_seconds: float = 15.0,
        ratio_threshold: float = 0.50,
        reset_ratio: float = 0.15,
        min_frames: int = 10,
    ) -> None:
        self.window          = window_seconds
        self.ratio_threshold = ratio_threshold
        self.reset_ratio     = reset_ratio
        self.min_frames      = min_frames
        self._events: list[tuple[float, bool]] = []
        self._alerted: bool = False

    def update(self, cigarette_detected: bool) -> bool:
        """Call once per frame. Returns True the first time the threshold is crossed."""
        now = time.time()
        self._events.append((now, cigarette_detected))

        cutoff = now - self.window
        self._events = [(t, d) for t, d in self._events if t >= cutoff]

        if len(self._events) < self.min_frames:
            return False

        ratio = sum(1 for _, d in self._events if d) / len(self._events)

        if self._alerted:
            if ratio < self.reset_ratio:
                self._alerted = False
            return False

        if ratio >= self.ratio_threshold:
            self._alerted = True
            return True

        return False

    @property
    def detection_ratio(self) -> float:
        if not self._events:
            return 0.0
        return sum(1 for _, d in self._events if d) / len(self._events)

    def reset(self) -> None:
        self._events.clear()
        self._alerted = False
