"""Safety layer — blocks dangerous keyboard/click patterns BEFORE executing."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.api.schemas import Action

# Strings that, if typed, MUST be blocked.
DANGEROUS_TYPED: list[str] = [
    "rm -rf /", "rm -rf ~", "rm -rf *",
    "shutdown", "reboot",
    "format c:", "format /dev",
    "dd if=/dev/zero", "dd if=/dev/urandom",
    "sudo ", "su -",
    ":(){:|:&};:",  # fork bomb
    "mkfs.",
    "chmod 777 /", "chown root /",
]

# Key combos that MUST be blocked
DANGEROUS_KEYS: set[str] = {
    "ctrl+alt+del", "ctrl+alt+f1", "ctrl+alt+f2",
    "windows+l", "alt+f4",  # log out / kill window
}


@dataclass
class SafetyVerdict:
    allowed: bool
    reason: str = ""


def check_action(action: Action) -> SafetyVerdict:
    if action.type == "type" and action.text:
        text = action.text.lower()
        for danger in DANGEROUS_TYPED:
            if danger.lower() in text:
                return SafetyVerdict(allowed=False, reason=f"dangerous typed pattern: '{danger}'")
    if action.type == "key" and action.key:
        if action.key.lower() in DANGEROUS_KEYS:
            return SafetyVerdict(allowed=False, reason=f"dangerous key combo: '{action.key}'")
    if action.type == "click" and action.coords is None:
        return SafetyVerdict(allowed=False, reason="click action without coordinates")
    return SafetyVerdict(allowed=True)
