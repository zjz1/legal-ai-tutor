# -*- coding: utf-8 -*-
"""集中配置。

设计要点：
1. 导入时不读取环境变量，避免测试与多实例互相污染；
2. 所有解析失败都抛中文可读的 ConfigError，而不是裸 ValueError；
3. 支持项目根目录下的 .env（已存在的环境变量优先）。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

PROVIDERS = ("mock", "ollama", "openai")


def _project_root() -> Path:
    """源码运行时返回仓库根；安装为 wheel 时退回当前工作目录。"""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").exists() else Path.cwd()


PROJECT_ROOT = _project_root()


def load_dotenv(path: Path) -> None:
    """极简 .env 解析：仅支持 KEY=VALUE 与 # 注释，不覆盖已有环境变量。"""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None or value == "" else value.strip()


def _int(env: Mapping[str, str], name: str, default: int, minimum: int | None = None) -> int:
    raw = _get(env, name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(
            f"环境变量 {name} 必须是整数，当前值为 {raw!r}",
            hint="例如 OLLAMA_TIMEOUT=25；不要带单位或引号",
        ) from None
    if minimum is not None and value < minimum:
        raise ConfigError(f"环境变量 {name} 不能小于 {minimum}，当前值为 {value}")
    return value


def _float(env: Mapping[str, str], name: str, default: float, minimum: float | None = None) -> float:
    raw = _get(env, name, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"环境变量 {name} 必须是数字，当前值为 {raw!r}") from None
    if minimum is not None and value < minimum:
        raise ConfigError(f"环境变量 {name} 不能小于 {minimum}，当前值为 {value}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name, "true" if default else "false").lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"环境变量 {name} 必须是 true/false，当前值为 {raw!r}")


@dataclass(frozen=True)
class Settings:
    """一次加载、全局只读的配置快照。"""

    provider: str
    ollama_base_url: str
    local_model: str
    cloud_api_base: str
    cloud_api_key: str
    cloud_model: str
    request_timeout: float
    total_deadline: float
    max_concurrency: int
    retries: int
    circuit_failures: int
    circuit_cooldown: float
    api_token: str
    rate_limit_per_minute: int
    allow_client_key: bool
    law_dir: Path
    log_level: str
    temperature: float

    @property
    def has_cloud(self) -> bool:
        return bool(self.cloud_api_key)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """读取并校验配置。传入 env 便于测试注入。"""
    if env is None:
        load_dotenv(PROJECT_ROOT / ".env")
        env = os.environ

    provider = _get(env, "LLM_PROVIDER", "mock").lower()
    if provider not in PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER 只能是 {'/'.join(PROVIDERS)}，当前值为 {provider!r}",
            hint="离线开发用 mock；本地模型用 ollama；云端用 openai",
        )

    law_dir = Path(_get(env, "LAW_DIR", str(PROJECT_ROOT / "data" / "laws"))).expanduser()
    if not law_dir.is_absolute():
        law_dir = (PROJECT_ROOT / law_dir).resolve()

    return Settings(
        provider=provider,
        ollama_base_url=_get(env, "OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        local_model=_get(env, "LOCAL_MODEL", "legal-qwen"),
        cloud_api_base=_get(env, "CLOUD_API_BASE", "https://api.deepseek.com/v1").rstrip("/"),
        cloud_api_key=_get(env, "CLOUD_API_KEY", ""),
        cloud_model=_get(env, "CLOUD_MODEL", "deepseek-chat"),
        request_timeout=_float(env, "REQUEST_TIMEOUT", 25.0, minimum=1.0),
        total_deadline=_float(env, "TOTAL_DEADLINE", 60.0, minimum=5.0),
        max_concurrency=_int(env, "MAX_CONCURRENCY", 1, minimum=1),
        retries=_int(env, "RETRIES", 3, minimum=1),
        circuit_failures=_int(env, "CIRCUIT_FAILURES", 5, minimum=1),
        circuit_cooldown=_float(env, "CIRCUIT_COOLDOWN", 60.0, minimum=1.0),
        api_token=_get(env, "API_TOKEN", ""),
        rate_limit_per_minute=_int(env, "RATE_LIMIT_PER_MINUTE", 30, minimum=1),
        allow_client_key=_bool(env, "ALLOW_CLIENT_KEY", False),
        law_dir=law_dir,
        log_level=_get(env, "LOG_LEVEL", "INFO").upper(),
        temperature=_float(env, "TEMPERATURE", 0.3, minimum=0.0),
    )
