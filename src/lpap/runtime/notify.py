"""Pushover notifications. Silent no-op without credentials."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def enabled() -> bool:
    return bool(os.environ.get("PUSHOVER_TOKEN") and os.environ.get("PUSHOVER_USER"))


def send(title: str, message: str, *, priority: int = 0) -> bool:
    """Return ``True`` when the message was accepted. Never raises."""
    if not enabled():
        return False
    payload = urllib.parse.urlencode(
        {
            "token": os.environ["PUSHOVER_TOKEN"],
            "user": os.environ["PUSHOVER_USER"],
            "title": title,
            "message": message,
            "priority": str(priority),
        }
    ).encode()
    try:
        with urllib.request.urlopen(PUSHOVER_URL, data=payload, timeout=20) as response:
            return json.loads(response.read()).get("status") == 1
    except Exception:
        return False
