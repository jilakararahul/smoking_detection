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
    from utils.alerts import SMSAlerter
    alerter = SMSAlerter(to_number="+447700900000")
    alerter.send("Cigarette detected — 2 cigarette(s) found in uploaded image.")
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
