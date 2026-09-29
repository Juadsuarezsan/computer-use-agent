"""Safety layer — blocks dangerous actions BEFORE they reach the VM.

Three mechanisms, evaluated in order (see ``docs/safety.md``):

1. **Typed-text blocklist** (:data:`DANGEROUS_TYPED`, :data:`DANGEROUS_REGEX`):
   destructive shell commands, privilege escalation, piping remote scripts
   into a shell, raw device writes, fork bombs.
2. **Key-combo blocklist** (:data:`DANGEROUS_KEYS`): combos that log out,
   switch TTY or kill the session.
3. **URL allowlist** (:data:`ALLOWED_URL_SCHEMES`, :func:`url_allowed`): typed
   URLs must point to the sandbox (``file://``, localhost) or an explicitly
   allowed domain.

Anything matching :data:`HITL_PATTERNS` is not blocked outright but flagged
``requires_human`` so an operator can approve it (payments, mass deletion,
sending e-mail to external domains).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from src.api.schemas import Action

DANGEROUS_TYPED: list[str] = [
    # destructive file-system commands
    "rm -rf /",
    "rm -rf ~",
    "rm -rf *",
    "rm -rf .",
    "rm -fr /",
    "del /f /s /q",
    "format c:",
    "format /dev",
    "mkfs.",
    "dd if=/dev/zero",
    "dd if=/dev/urandom",
    "dd if=/dev/random",
    "> /dev/sda",
    "> /dev/nvme",
    "> /dev/hd",
    "shred ",
    # privilege escalation / system control
    "sudo ",
    "su -",
    "su root",
    "doas ",
    "shutdown",
    "reboot",
    "poweroff",
    "halt -p",
    "init 0",
    "init 6",
    "systemctl poweroff",
    "systemctl reboot",
    "chmod 777 /",
    "chmod -r 777 /",
    "chown root /",
    "chown -r ",
    # fork bombs
    ":(){:|:&};:",
    ":(){ :|:& };:",
    # remote code execution
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
    "invoke-webrequest",
    "iex (",
    # credentials / exfiltration
    "cat /etc/shadow",
    "cat ~/.ssh/id_",
    "history -c",
]
"""Literal substrings (case-insensitive) that must never be typed."""

DANGEROUS_REGEX: list[tuple[str, str]] = [
    (r"\brm\s+-[a-z]*r[a-z]*f?[a-z]*\s+[/~*.]", "recursive rm on / ~ * or ."),
    (r"\brm\s+-[a-z]*r[a-z]*\s+--no-preserve-root", "rm --no-preserve-root"),
    (r"\b(curl|wget)\b[^|\n]*\|\s*(sudo\s+)?(sh|bash|zsh|python[0-9.]*)\b", "piping a download into a shell"),
    (r"\bdd\s+[^\n]*of=/dev/(sd|hd|nvme|mmcblk|disk)", "raw write to a block device"),
    (r">\s*/dev/(sd|hd|nvme|mmcblk|disk)[a-z0-9]*", "redirect into a block device"),
    (r"\bmkfs(\.[a-z0-9]+)?\b", "formatting a file system"),
    (r"\b(chmod|chown)\s+-[a-z]*R[a-z]*\s+[^\n]*\s/(\s|$)", "recursive chmod/chown on /"),
    (r"\bkill\s+-9\s+-1\b", "killing every process"),
    (r"\bdrop\s+(database|table)\b", "SQL DROP statement"),
    (r"\btruncate\s+table\b", "SQL TRUNCATE statement"),
    (r"\bdelete\s+from\s+\w+\s*;?\s*$", "SQL DELETE without WHERE"),
    (r"\bgit\s+push\b[^\n]*--force\b", "git force push"),
    (r"\bgit\s+push\b[^\n]*\s-f\b", "git force push"),
]
"""Regexes (case-insensitive) with a human-readable reason."""

DANGEROUS_KEYS: set[str] = {
    "ctrl+alt+del",
    "ctrl+alt+delete",
    "ctrl+alt+backspace",
    "ctrl+alt+f1",
    "ctrl+alt+f2",
    "ctrl+alt+f3",
    "ctrl+alt+f4",
    "ctrl+alt+f5",
    "ctrl+alt+f6",
    "ctrl+alt+f7",
    "super+l",
    "windows+l",
    "meta+l",
    "alt+f4",
    "alt+sysrq+b",
    "alt+sysrq+o",
    "ctrl+alt+t",  # opens a terminal: the agent must stay inside the target app
}
"""Key combos (lower-case, ``+``-joined) that are never sent."""

ALLOWED_URL_SCHEMES: set[str] = {"file", "http", "https", "about"}
ALLOWED_HOSTS: set[str] = {"localhost", "127.0.0.1", "0.0.0.0", "sandbox", "vm"}
"""Hosts that may be navigated to. Extend via :func:`allow_hosts`."""

HITL_PATTERNS: list[tuple[str, str]] = [
    (r"\b(pay|payment|purchase|checkout|place\s+order|buy\s+now|transfer\s+funds?)\b", "payment / purchase"),
    (r"\b(delete\s+all|remove\s+all|purge|wipe|erase\s+all)\b", "mass deletion"),
    (r"\b(send\s+(e-?mail|invoice|message)\s+to\s+[^\s]+@(?!x\.com)[^\s]+)", "e-mail to external domain"),
    (r"\b(card\s*number|cvv|iban|swift|routing\s*number)\b", "financial credential"),
]
"""Patterns that require a human in the loop rather than an outright block."""


@dataclass
class SafetyVerdict:
    """Result of :func:`check_action`.

    Attributes:
        allowed: ``False`` when the action must not be executed.
        reason: Human-readable reason for a block or a HITL flag.
        requires_human: ``True`` when the action is allowed only after approval.
        rule: Identifier of the rule that fired (for the audit log).
    """

    allowed: bool
    reason: str = ""
    requires_human: bool = False
    rule: str = ""
    matched: list[str] = field(default_factory=list)


def allow_hosts(hosts: set[str]) -> None:
    """Extend the navigation allowlist (used by the sandbox / tests)."""
    ALLOWED_HOSTS.update(h.lower() for h in hosts)


def url_allowed(text: str) -> tuple[bool, str]:
    """Check whether a typed string that looks like a URL points to an allowed host."""
    candidate = text.strip()
    if "://" not in candidate:
        return True, ""
    parsed = urlparse(candidate)
    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        return False, f"url scheme not allowed: {parsed.scheme!r}"
    if parsed.scheme in {"file", "about"}:
        return True, ""
    host = (parsed.hostname or "").lower()
    if host in ALLOWED_HOSTS:
        return True, ""
    return False, f"host not in allowlist: {host!r}"


def _check_text(text: str) -> SafetyVerdict | None:
    lowered = text.lower()
    for danger in DANGEROUS_TYPED:
        if danger in lowered:
            return SafetyVerdict(
                allowed=False,
                reason=f"dangerous typed pattern: '{danger.strip()}'",
                rule="typed_blocklist",
                matched=[danger],
            )
    for pattern, reason in DANGEROUS_REGEX:
        if re.search(pattern, lowered, flags=re.IGNORECASE | re.MULTILINE):
            return SafetyVerdict(
                allowed=False,
                reason=f"dangerous typed pattern: {reason}",
                rule="typed_regex",
                matched=[pattern],
            )
    ok, why = url_allowed(text)
    if not ok:
        return SafetyVerdict(allowed=False, reason=why, rule="url_allowlist", matched=[text[:80]])
    for pattern, reason in HITL_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return SafetyVerdict(
                allowed=True,
                requires_human=True,
                reason=f"requires human approval: {reason}",
                rule="hitl",
                matched=[pattern],
            )
    return None


def check_action(action: Action) -> SafetyVerdict:
    """Evaluate one action against the blocklists and the HITL policy.

    Args:
        action: The action the reasoner wants to execute.

    Returns:
        A :class:`SafetyVerdict`; ``allowed=False`` means "do not execute".
    """
    if action.type == "type" and action.text:
        verdict = _check_text(action.text)
        if verdict is not None:
            return verdict
    if action.type == "task_complete" and action.text:
        # A final answer that itself contains a shell command is suspicious,
        # but it is never executed: only HITL flags apply.
        verdict = _check_text(action.text)
        if verdict is not None and verdict.requires_human:
            return verdict
    if action.type == "key" and action.key:
        normalized = action.key.lower().replace(" ", "")
        if normalized in DANGEROUS_KEYS:
            return SafetyVerdict(
                allowed=False,
                reason=f"dangerous key combo: '{action.key}'",
                rule="key_blocklist",
                matched=[normalized],
            )
    if action.type in {"click", "double_click", "right_click", "mouse_move"} and action.coords is None:
        return SafetyVerdict(
            allowed=False, reason=f"{action.type} action without coordinates", rule="no_coords"
        )
    return SafetyVerdict(allowed=True)


def blocked_rules_catalog() -> dict[str, list[str]]:
    """Return every rule in a serialisable form (used by ``docs/safety.md`` and the API)."""
    return {
        "typed_blocklist": list(DANGEROUS_TYPED),
        "typed_regex": [reason for _, reason in DANGEROUS_REGEX],
        "key_blocklist": sorted(DANGEROUS_KEYS),
        "url_allowlist": sorted(ALLOWED_HOSTS),
        "hitl": [reason for _, reason in HITL_PATTERNS],
    }
