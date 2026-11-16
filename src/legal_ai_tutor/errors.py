# -*- coding: utf-8 -*-
"""统一异常体系：携带错误码与 HTTP 状态，便于 API 层统一映射。"""

from __future__ import annotations


class LegalAIError(Exception):
    """项目内所有可预期异常的基类。"""

    code: str = "INTERNAL"
    http_status: int = 500

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        return payload


class ConfigError(LegalAIError):
    """环境变量缺失或格式非法。"""

    code = "CONFIG_INVALID"
    http_status = 500


class ProviderError(LegalAIError):
    """模型提供方调用失败。"""

    code = "PROVIDER_ERROR"
    http_status = 502
    retryable = True

    def __init__(self, message: str, *, provider: str = "", status: int | None = None, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.provider = provider
        self.status = status


class ProviderUnavailable(ProviderError):
    """连接不上提供方（服务未启动、DNS 失败等）。"""

    code = "PROVIDER_UNAVAILABLE"


class ProviderTimeout(ProviderError):
    """调用超时。"""

    code = "PROVIDER_TIMEOUT"


class ProviderAuthError(ProviderError):
    """密钥缺失或被拒绝，重试无意义。"""

    code = "PROVIDER_AUTH"
    retryable = False


class BudgetExceeded(LegalAIError):
    """整条请求超出时间预算。"""

    code = "BUDGET_EXCEEDED"
    http_status = 504


class SchemaError(LegalAIError):
    """模型输出不符合结构化约束。"""

    code = "SCHEMA_INVALID"
    http_status = 422


class CitationError(LegalAIError):
    """法条引用无法通过校验。"""

    code = "CITATION_INVALID"
    http_status = 422
