# -*- coding: utf-8 -*-
"""命令行入口：serve / ask / expert / eval / index。

示例：
    python -m legal_ai_tutor.cli ask "口头借款合同是否成立？"
    python -m legal_ai_tutor.cli expert "张三借款20万到期不还..."
    python -m legal_ai_tutor.cli eval --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import PROJECT_ROOT, load_settings
from .evaluation import Evaluator, format_report, load_dataset
from .logging_setup import configure_logging
from .runtime import build_runtime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="legal-ai-tutor", description="法律 AI 助手命令行")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动 HTTP 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    ask = sub.add_parser("ask", help="本地模式提问")
    ask.add_argument("question")

    expert = sub.add_parser("expert", help="专家模式提问")
    expert.add_argument("question")

    evaluate = sub.add_parser("eval", help="批量评估")
    evaluate.add_argument("--dataset", default=str(PROJECT_ROOT / "eval" / "samples.jsonl"))
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--out", default=None, help="汇总结果写入的 JSON 路径")
    evaluate.add_argument("--cache", default=str(PROJECT_ROOT / "eval" / ".cache.json"))

    sub.add_parser("index", help="查看法条库与检索器状态")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)

    if args.command == "serve":
        import uvicorn

        from .api.app import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return 0

    runtime = build_runtime(settings)

    if args.command == "ask":
        result = asyncio.run(runtime.workflow.local_answer(args.question))
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "expert":
        result = asyncio.run(runtime.workflow.expert_answer(args.question))
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval":
        samples = load_dataset(Path(args.dataset))
        evaluator = Evaluator(runtime.workflow, cache_path=Path(args.cache) if args.cache else None)
        report = asyncio.run(evaluator.run(samples, limit=args.limit))
        print(format_report(report))
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n已写入 {out_path}")
        return 0

    if args.command == "index":
        print(f"法条库：{len(runtime.corpus)} 条")
        print(f"检索器：{'就绪' if runtime.retriever else '为空（请检查 LAW_DIR 下的 sample_articles.jsonl）'}")
        print(f"提供方：{settings.provider} / 实际模型：{runtime.router.local.model}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
