FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY data ./data

RUN pip install .

EXPOSE 8000

# 通过工厂函数创建应用，避免 import 时读取环境配置
CMD ["uvicorn", "legal_ai_tutor.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
