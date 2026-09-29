"""Application settings (environment variables / ``.env``).

Every knob that touches an external system lives here so the domain code never
reads ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
"""Pinned production model id (never an alias such as ``claude-sonnet-4-5``)."""

VmMode = Literal["fake", "playwright", "xdotool"]


class Settings(BaseSettings):
    """Runtime configuration.

    Attributes:
        anthropic_api_key: Key for the Anthropic API. ``None`` selects the offline
            :class:`~src.agent.reasoner.StubReasoner`.
        anthropic_model: Pinned model id used for reasoning, verification and the
            final validator.
        computer_use_tool: Anthropic computer-use tool version.
        llm_timeout_s: Per-request timeout for every LLM call.
        llm_max_retries: Attempts (with exponential backoff) per LLM call.
        vm_mode: Which VM driver to build (``fake`` / ``playwright`` / ``xdotool``).
        chromium_path: Chromium binary for the Playwright sandbox.
        display_width / display_height: Virtual screen size sent to the model.
        database_url: PostgreSQL DSN for the audit log; empty selects SQLite.
        audit_sqlite_path: SQLite fallback file (``:memory:`` allowed).
        cors_origins: Comma-separated list of allowed origins (never ``*`` in prod).
        rate_limit: slowapi rate-limit string for ``/api/*``.
        langchain_tracing_v2 / langchain_api_key / langchain_project: LangSmith
            wiring; LangGraph picks these up from the environment automatically.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default=DEFAULT_MODEL, alias="ANTHROPIC_MODEL")
    computer_use_tool: str = Field(default="computer_20250124", alias="COMPUTER_USE_TOOL_VERSION")
    llm_timeout_s: float = Field(default=60.0, alias="LLM_TIMEOUT_S", gt=0)
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES", ge=1, le=10)
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS", ge=64)

    vm_mode: VmMode = Field(default="fake", alias="VM_MODE")
    chromium_path: str = Field(default="/opt/pw-browsers/chromium", alias="CHROMIUM_PATH")
    display_width: int = Field(default=1024, alias="DISPLAY_WIDTH", ge=320, le=4096)
    display_height: int = Field(default=768, alias="DISPLAY_HEIGHT", ge=240, le=4096)
    screenshot_dir: str = Field(default="data/screenshots", alias="SCREENSHOT_DIR")
    webapps_dir: str = Field(default=str(REPO_ROOT / "sandbox" / "webapps"), alias="WEBAPPS_DIR")
    tasks_file: str = Field(default=str(REPO_ROOT / "data" / "eval" / "tasks.json"), alias="TASKS_FILE")
    vm_vnc_host: str = Field(default="localhost", alias="VM_VNC_HOST")
    vm_vnc_port: int = Field(default=5900, alias="VM_VNC_PORT")
    max_steps_per_task: int = Field(default=30, alias="MAX_STEPS_PER_TASK", ge=1, le=100)

    database_url: str = Field(default="", alias="DATABASE_URL")
    audit_sqlite_path: str = Field(default="data/audit.sqlite3", alias="AUDIT_SQLITE_PATH")

    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")
    rate_limit: str = Field(default="10/minute", alias="RATE_LIMIT")

    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="computer-use-agent", alias="LANGCHAIN_PROJECT")

    @property
    def cors_origin_list(self) -> list[str]:
        """Parsed CORS origins (empty entries removed)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        """Whether a real Anthropic client can be built."""
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
