#!/usr/bin/env python3
"""Validate the CodeFest AI researcher notebook.

Modes:
- --offline: static contract checks, no keys required.
- --deps: import installed notebook dependencies.
- --live: run key-gated OpenRouter/Tavily checks when keys exist; checks Phoenix span setup when server is available.
- --strict-live: fail instead of skip when live keys are absent.
- --start-phoenix: start a temporary local Phoenix server for the Phoenix live check.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "codefest_ai_researcher_masterclass.ipynb"

REQUIRED_HEADINGS = [
    "Настройка",
    "OpenRouter",
    "Tavily",
    "Phoenix",
    "Первый вызов модели",
    "Инструменты",
    "Цикл агента",
    "Состояние и задачи",
    "Планировщик",
    "Поисковый подагент",
    "Итоговый отчёт",
    "Куда развивать систему",
]

REQUIRED_CONTENT = [
    "Агент = Модель + Обвязка",
    "обвязка",
    "ReAct",
    "контур оценки качества",
    "context rot",
    "context anxiety",
    "https://app.tavily.com/",
    "getpass",
    "Phoenix",
    "OpenRouter",
    "Tavily",
]


def load_notebook() -> dict[str, Any]:
    if not NOTEBOOK.exists():
        raise AssertionError(f"Notebook not found: {NOTEBOOK}")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, "Notebook must be nbformat v4"
    assert isinstance(nb.get("cells"), list) and nb["cells"], "Notebook has no cells"
    return nb


def cell_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def normalize_code_for_ast(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            lines.append("pass")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def validate_static(nb: dict[str, Any]) -> None:
    markdown = "\n".join(cell_text(c) for c in nb["cells"] if c.get("cell_type") == "markdown")
    markdown_lower = markdown.lower()
    code_cells = [cell_text(c) for c in nb["cells"] if c.get("cell_type") == "code"]
    full_text = markdown + "\n" + "\n".join(code_cells)
    for heading in REQUIRED_HEADINGS:
        assert heading.lower() in markdown_lower, f"Missing required section/content marker: {heading}"
    for marker in REQUIRED_CONTENT:
        assert marker in full_text, f"Missing required content marker: {marker}"

    assert code_cells, "No code cells found"
    for idx, source in enumerate(code_cells, start=1):
        ast.parse(normalize_code_for_ast(source), filename=f"notebook-cell-{idx}")

    assert full_text.count("GAP") >= 3, "Expected at least 3 gap cells"
    assert "LangChain" in full_text and "LangGraph" in full_text, "Framework non-goal/comparison missing"

    forbidden = [
        "CACHED_SEARCH_RESULTS",
        "demo_fixture_search",
        "demo-fixture",
        "Offline demo ответ",
        "FakeMessage",
        "Tavily fallback:",
        "заглуш",
        "fixture",
        "fallback path",
        "имитац",
    ]
    for marker in forbidden:
        assert marker not in full_text, f"Stub/fallback marker must not remain: {marker}"

    assert "name=\"delegate_search\"" in full_text, "delegate_search tool missing"
    assert "name=\"save_report\"" in full_text, "save_report tool missing"
    assert "def run_researcher" in full_text, "run_researcher missing"
    assert "call_model(messages, tools=research_registry.schemas())" in full_text, "Main researcher must use tool-enabled model loop"
    assert "delegate_search(batch" not in full_text, "Main researcher still looks like direct Python batch pipeline"
    assert "save_report" in full_text and "report.md" in full_text, "Markdown report saving contract missing"
    assert "summarize_latest_run" in full_text, "Final run statistics cell missing"
    assert "tool.search_web" in full_text and "token_usage" in full_text, "Run statistics must include search/tool/token metrics"
    assert "Верни только JSON" not in full_text, "JSON-prompt planning/query workaround should be removed"


def validate_dependencies() -> None:
    warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
    import nbformat  # noqa: F401
    from openai import OpenAI  # noqa: F401
    from tavily import TavilyClient  # noqa: F401
    from phoenix.otel import register  # noqa: F401
    from openinference.instrumentation.openai import OpenAIInstrumentor  # noqa: F401


def wait_for_url(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def start_phoenix_server() -> subprocess.Popen[str]:
    log_path = ROOT / ".omx" / "tmp" / "phoenix-validate.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    proc: subprocess.Popen[str] = subprocess.Popen(
        [sys.executable, "-m", "phoenix.server.main", "serve"],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not wait_for_url("http://127.0.0.1:6006", timeout_s=25):
        proc.terminate()
        raise RuntimeError(f"Phoenix server did not start; see {log_path}")
    return proc


def validate_phoenix_span(start_server: bool = False) -> str:
    warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
    proc: Optional[subprocess.Popen[str]] = None
    try:
        if start_server:
            proc = start_phoenix_server()
        elif not wait_for_url("http://127.0.0.1:6006", timeout_s=1):
            return "skipped-no-phoenix-server"
        from phoenix.otel import register

        provider = register(
            project_name="codefest-ai-researcher-validation",
            endpoint="http://localhost:6006/v1/traces",
            protocol="http/protobuf",
            auto_instrument=True,
            batch=False,
            verbose=False,
        )
        tracer = provider.get_tracer(__name__)
        with tracer.start_as_current_span("validation.phoenix_smoke") as span:
            span.set_attribute("validation.status", "ok")
        return "ok-started-server" if start_server else "ok-span-created"
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def validate_openrouter(strict: bool) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        if strict:
            raise AssertionError("OPENROUTER_API_KEY is required for strict live validation")
        return "skipped-no-OPENROUTER_API_KEY"
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    errors: list[str] = []
    for model in [os.getenv("OPENROUTER_MODEL", "openrouter/auto"), os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/auto")]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Ответь одним словом: ping"}],
                max_tokens=128,
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            if text.strip():
                return f"ok-model={model}"
            errors.append(f"{model}: empty content")
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    raise AssertionError("OpenRouter returned no usable content: " + "; ".join(errors))


def validate_tavily(strict: bool) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        if strict:
            raise AssertionError("TAVILY_API_KEY is required for strict live validation")
        return "skipped-no-TAVILY_API_KEY"
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    result = client.search(
        query="LangGraph official documentation stateful agents checkpointing",
        max_results=1,
        search_depth="advanced",
        include_answer="advanced",
        include_raw_content="markdown",
    )
    items = result.get("results", []) if isinstance(result, dict) else []
    assert items, "Tavily returned no results"
    assert items[0].get("url"), "Tavily result has no URL"
    return "ok"


def validate_live(strict: bool = False, start_phoenix: bool = False) -> dict[str, str]:
    validate_dependencies()
    return {
        "phoenix": validate_phoenix_span(start_server=start_phoenix),
        "openrouter": validate_openrouter(strict=strict),
        "tavily": validate_tavily(strict=strict),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="static contract checks")
    parser.add_argument("--deps", action="store_true", help="dependency import checks")
    parser.add_argument("--live", action="store_true", help="live checks; skips missing keys unless --strict-live")
    parser.add_argument("--strict-live", action="store_true", help="fail live checks when keys are missing")
    parser.add_argument("--start-phoenix", action="store_true", help="start temporary local Phoenix server for the Phoenix live check")
    parser.add_argument("--all", action="store_true", help="run offline + deps + live")
    args = parser.parse_args()

    if not any([args.offline, args.deps, args.live, args.all]):
        args.offline = True

    if args.all:
        args.offline = args.deps = args.live = True

    if args.offline:
        nb = load_notebook()
        validate_static(nb)
        print("offline: ok")

    if args.deps:
        validate_dependencies()
        print("deps: ok")

    if args.live:
        result = validate_live(strict=args.strict_live, start_phoenix=args.start_phoenix)
        print("live:", json.dumps(result, ensure_ascii=False, sort_keys=True))

    print("Notebook validation passed:", NOTEBOOK.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
