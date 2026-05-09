"""
SMS Alert Integration — Twilio
================================
Sends an SMS notification when a cigarette is detected.

Credentials are loaded from Streamlit secrets (.streamlit/secrets.toml):

    [twilio]
    account_sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    auth_token  = "your_auth_token"
    from_number = "+14155238886"   # your Twilio number

Usage
-----
    from utils.alerts import SMSAlerter, SlidingWindowTracker
    alerter = SMSAlerter(to_number="+447700900000")
    alerter.send("Cigarette detected — 2 cigarette(s) found in uploaded image.")

    # For live camera — alert if cigarette detected in ≥50 % of frames over 15 s
    tracker = SlidingWindowTracker(window_seconds=15, ratio_threshold=0.50)
    if tracker.update(cigarette_present):   # returns True exactly once per event
        alerter.send("Sustained smoking detected.")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SMSAlerter:
    """Wraps Twilio SMS sending with a cooldown to avoid alert flooding."""

    to_number: str
    cooldown_seconds: int = 30         # minimum gap between consecutive alerts
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
        """
        Send an SMS. Returns True if sent, False if within cooldown window.
        Raises RuntimeError if credentials are missing.
        """
        now = time.time()
        if now - self._last_sent < self.cooldown_seconds:
            return False   # still within cooldown

        import streamlit as st
        from_number = st.secrets["twilio"]["from_number"]

        client = self._get_client()
        client.messages.create(to=self.to_number, from_=from_number, body=body)
        self._last_sent = now
        return True

    def ready(self) -> bool:
        """True if outside the cooldown window."""
        return (time.time() - self._last_sent) >= self.cooldown_seconds


class SlidingWindowTracker:
    """
    Sliding-window cigarette detection tracker.

    Rationale
    ---------
    Rather than requiring the cigarette to be *continuously* in frame
    (which resets on a single missed frame), this tracker looks at the
    last ``window_seconds`` of frames and fires an alert when the cigarette
    was detected in at least ``ratio_threshold`` of those frames.

    This tolerates normal behaviour like the cigarette briefly leaving the
    frame, someone turning their head, or a single missed detection.

    Behaviour
    ---------
    - Collects per-frame detection events (True/False) with timestamps.
    - Prunes events older than ``window_seconds`` on every update.
    - Waits until at least ``min_frames`` have been collected (avoids
      firing immediately on the very first few frames).
    - Returns True exactly ONCE per "smoking event".  The alert resets
      after the cigarette has been mostly absent (detection ratio drops
      below ``reset_ratio``) so a future sustained event can trigger again.

    Thread safety
    -------------
    Designed for use from a single background thread.
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

        self._events: list[tuple[float, bool]] = []   # (timestamp, detected)
        self._alerted: bool = False

    def update(self, cigarette_detected: bool) -> bool:
        """
        Call once per frame.  Returns True exactly once when the sliding
        window ratio first crosses ``ratio_threshold``.
        """
        now = time.time()
        self._events.append((now, cigarette_detected))

        # Prune events outside the window
        cutoff = now - self.window
        self._events = [(t, d) for t, d in self._events if t >= cutoff]

        if len(self._events) < self.min_frames:
            return False   # not enough data yet

        ratio = sum(1 for _, d in self._events if d) / len(self._events)

        if self._alerted:
            # Reset once the window is mostly clear — allows a future event to fire
            if ratio < self.reset_ratio:
                self._alerted = False
            return False

        if ratio >= self.ratio_threshold:
            self._alerted = True
            return True

        return False

    @property
    def detection_ratio(self) -> float:
        """Fraction of frames in the current window where cigarette was detected."""
        if not self._events:
            return 0.0
        return sum(1 for _, d in self._events if d) / len(self._events)

    @property
    def window_fill(self) -> float:
        """How many seconds of data are in the current window (0 → window_seconds)."""
        if not self._events:
            return 0.0
        return self._events[-1][0] - self._events[0][0]

    def reset(self) -> None:
        """Manually reset (e.g. after the stream stops)."""
        self._events.clear()
        self._alerted = False
