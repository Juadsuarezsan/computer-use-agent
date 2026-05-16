from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")
    computer_use_tool: str = Field(default="computer_20250124", alias="COMPUTER_USE_TOOL_VERSION")
    vm_vnc_host: str = Field(default="localhost", alias="VM_VNC_HOST")
    vm_vnc_port: int = Field(default=5900, alias="VM_VNC_PORT")
    max_steps_per_task: int = Field(default=30, alias="MAX_STEPS_PER_TASK")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
