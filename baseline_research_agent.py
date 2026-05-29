#!/usr/bin/env python3
"""Baseline deep-research agent for the CodeFest competition part.

This file is a standalone Python copy of the final agent assembled in
`notebooks/codefest_ai_researcher_masterclass.ipynb`.

Baseline architecture:
- OpenRouter-compatible OpenAI client for model calls.
- Tavily `search_web` tool.
- Tool registry + dispatcher.
- Main researcher with `generate_plan`, `delegate_search`, `modify_todo`, `save_report`.
- Search subagent that can only call `search_web` and must search at least twice.
- Optional Phoenix/OpenTelemetry spans when Phoenix packages/server are available.

Required live credentials:
- OPENROUTER_API_KEY
- TAVILY_API_KEY

Example:
    python baseline_research_agent.py \
      --query "Исследуй, какой подход выбрать для ..." \
      --output report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Literal, Optional

try:
    from openai import OpenAI
except Exception:  # Keep import/check mode usable before dependencies are installed.
    OpenAI = None  # type: ignore[assignment]

try:
    from opentelemetry.trace import Status, StatusCode
except Exception:
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]


TODAY = date.today().isoformat()
PRIMARY_MODEL = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.1")
FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/auto")
PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "codefest-ai-researcher-baseline")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REPORT_PATH = "report.md"


# ---------------------------------------------------------------------------
# Environment and optional tracing
# ---------------------------------------------------------------------------


def parse_dotenv_value(raw: str) -> str:
    """Minimal .env parser without python-dotenv."""
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    return value.split(" #", 1)[0].strip()


def load_dotenv_file(path: Optional[Path] = None, *, override: bool = False) -> Optional[Path]:
    """Load .env from cwd/parent without overwriting existing env vars."""
    if path is None:
        cwd = Path.cwd()
        candidates = [cwd / ".env", cwd.parent / ".env"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None or not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if override or key not in os.environ:
            os.environ[key] = parse_dotenv_value(raw_value)
    return path


if os.getenv("DISABLE_DOTENV") != "1":
    load_dotenv_file()


tracer = None
tracer_provider = None
PHOENIX_UI_URL: Optional[str] = None
PHOENIX_ENDPOINT: Optional[str] = None
PHOENIX_FORCE_FLUSH = os.getenv("PHOENIX_FORCE_FLUSH", "1") != "0"


def _span_preview(value: Any, limit: int = 8000) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _span_mime_type(value: Any) -> str:
    return "text/plain" if isinstance(value, str) else "application/json"


def _span_is_recording(span: Any) -> bool:
    return span is not None and (not hasattr(span, "is_recording") or span.is_recording())


def setup_phoenix_tracing(*, enable: bool = True) -> None:
    """Enable Phoenix tracing if packages and endpoint are available.

    Unlike the notebook, this baseline file does not auto-launch Phoenix by default:
    for competition runs it is safer to connect to an already configured collector.
    Start Phoenix separately (`phoenix serve`) or set PHOENIX_COLLECTOR_ENDPOINT.
    """
    global tracer, tracer_provider, PHOENIX_UI_URL, PHOENIX_ENDPOINT
    if not enable or os.getenv("DISABLE_PHOENIX") == "1":
        return
    try:
        from phoenix.otel import register

        phoenix_port = int(os.getenv("PHOENIX_PORT", "6006"))
        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or f"http://127.0.0.1:{phoenix_port}/v1/traces"
        endpoint = endpoint.strip().rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint += "/v1/traces"
        endpoint = endpoint.replace("http://localhost:", "http://127.0.0.1:")
        PHOENIX_ENDPOINT = endpoint
        PHOENIX_UI_URL = f"http://127.0.0.1:{phoenix_port}/"
        tracer_provider = register(
            project_name=PROJECT_NAME,
            endpoint=PHOENIX_ENDPOINT,
            protocol="http/protobuf",
            auto_instrument=True,
            batch=False,
        )
        tracer = tracer_provider.get_tracer(__name__)
    except Exception as exc:
        print(f"Phoenix tracing disabled: {type(exc).__name__}: {str(exc)[:200]}")
        tracer = None
        tracer_provider = None


def flush_traces(timeout_millis: int = 3000) -> None:
    if not PHOENIX_FORCE_FLUSH or tracer_provider is None:
        return
    try:
        tracer_provider.force_flush(timeout_millis=timeout_millis)
    except TypeError:
        tracer_provider.force_flush()
    except Exception as exc:
        print(f"Phoenix force_flush warning: {type(exc).__name__}: {str(exc)[:160]}")


class maybe_span:
    """Tiny no-op context manager so the agent works without Phoenix."""

    def __init__(self, name: str, *, kind: str = "CHAIN", input_value: Any = None, **attrs: Any):
        self.name = name
        self.kind = kind
        self.input_value = input_value
        self.attrs = attrs
        self._span_cm = None
        self._span = None

    def __enter__(self):
        if tracer is not None:
            self._span_cm = tracer.start_as_current_span(self.name)
            self._span = self._span_cm.__enter__()
            self._span.set_attribute("openinference.span.kind", self.kind)
            for key, value in self.attrs.items():
                self._span.set_attribute(key, str(value))
            if self.input_value is not None:
                self.set_input(self.input_value)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._span is not None and Status is not None and StatusCode is not None:
            if exc_type is None:
                self._span.set_status(Status(StatusCode.OK))
            else:
                self._span.record_exception(exc)
                self._span.set_status(Status(StatusCode.ERROR, str(exc)))
        if self._span_cm is not None:
            self._span_cm.__exit__(exc_type, exc, tb)
            flush_traces()

    def set_attribute(self, key: str, value: Any) -> None:
        if _span_is_recording(self._span):
            self._span.set_attribute(key, str(value))

    def set_input(self, value: Any) -> None:
        if _span_is_recording(self._span):
            self._span.set_attribute("input.value", _span_preview(value))
            self._span.set_attribute("input.mime_type", _span_mime_type(value))

    def set_output(self, value: Any) -> None:
        if _span_is_recording(self._span):
            self._span.set_attribute("output.value", _span_preview(value))
            self._span.set_attribute("output.mime_type", _span_mime_type(value))


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------


def live_model_available() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY") and OpenAI is not None)


def live_search_available() -> bool:
    return bool(os.getenv("TAVILY_API_KEY"))


def make_openrouter_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("Package `openai` is not installed. Install requirements.txt first.")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY обязателен: вызовы модели выполняются через live API.")
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://codefest.ru/",
            "X-Title": "CodeFest AI Researcher Baseline",
        },
    )


client = make_openrouter_client() if live_model_available() else None


def call_model(messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None, tool_choice: str = "auto") -> Any:
    """Single live model-call entrypoint through OpenRouter."""
    global client
    if client is None:
        client = make_openrouter_client()

    errors: list[str] = []
    # Keep the notebook behavior: try the primary model, then fallback.
    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            tool_names = [tool.get("function", {}).get("name") for tool in (tools or [])]
            with maybe_span(
                "model.call",
                kind="LLM",
                input_value={"model": model, "messages": messages, "tools": tool_names},
                model=model,
                tool_count=len(tools or []),
            ) as span:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice if tools else None,
                    temperature=0.2,
                )
                message = response.choices[0].message
                span.set_output(
                    {
                        "content": message.content,
                        "tool_calls": [
                            {"name": call.function.name, "arguments": call.function.arguments}
                            for call in (getattr(message, "tool_calls", None) or [])
                        ],
                    }
                )
                return message
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Все model calls завершились ошибкой:\n" + "\n".join(errors))


# ---------------------------------------------------------------------------
# Tool registry, dispatcher, and state
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Инструмент уже зарегистрирован: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Неизвестный инструмент: {name}")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)


Mode = Literal["plan", "execute", "final"]


@dataclass
class RunState:
    mode: Mode = "plan"
    todos: list[str] = field(default_factory=list)
    completed_todos: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    report_path: Optional[str] = None
    iteration_count: int = 0

    def add_todos(self, todos: list[str]) -> list[str]:
        added: list[str] = []
        existing = {todo.lower() for todo in self.todos + self.completed_todos}
        for todo in todos:
            clean = " ".join(todo.split())
            if clean and clean.lower() not in existing:
                self.todos.append(clean)
                existing.add(clean.lower())
                added.append(clean)
        return added

    def complete_todo(self, todo: str) -> bool:
        target = todo.strip().lower()
        for existing in list(self.todos):
            if existing.lower() == target:
                self.todos.remove(existing)
                self.completed_todos.append(existing)
                return True
        return False

    def remove_todos(self, todos: list[str]) -> tuple[list[str], list[str]]:
        removed: list[str] = []
        missing: list[str] = []
        for todo in todos:
            if self.complete_todo(todo):
                removed.append(todo)
            else:
                missing.append(todo)
        return removed, missing

    def is_incomplete(self) -> Optional[str]:
        if self.mode == "plan":
            return "Сначала вызови generate_plan и переведи задачу в execute mode."
        if self.todos:
            return "Остались незавершённые todos: " + "; ".join(self.todos)
        if self.report_path is None:
            return "Перед завершением вызови save_report и сохрани итоговый markdown-отчёт."
        return None

    def add_finding(
        self,
        question: str,
        notes: str,
        sources: list[str],
        evidence: Optional[list[dict[str, str]]] = None,
    ) -> None:
        self.findings.append(
            {
                "question": question,
                "notes": notes,
                "sources": sources,
                "evidence": evidence or [],
            }
        )


def _tool_call_name(call: Any) -> str:
    return call.function.name


def _tool_call_args(call: Any) -> dict[str, Any]:
    raw = call.function.arguments or "{}"
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def _tool_call_id(call: Any) -> str:
    return getattr(call, "id", "tool-call-id")


def dispatch_tool(call: Any, registry: ToolRegistry) -> dict[str, Any]:
    name = _tool_call_name(call)
    args = _tool_call_args(call)
    tool = registry.get(name)
    with maybe_span(
        "tool.dispatch",
        kind="TOOL",
        input_value={"tool_name": name, "args": args},
        tool_name=name,
        tool_args=args,
    ) as span:
        result = tool.handler(**args)
        span.set_output(result)
        return result


def assistant_message_to_dict(message: Any) -> dict[str, Any]:
    tool_calls = getattr(message, "tool_calls", None) or []
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": _tool_call_id(call),
                "type": "function",
                "function": {
                    "name": _tool_call_name(call),
                    "arguments": json.dumps(_tool_call_args(call), ensure_ascii=False),
                },
            }
            for call in tool_calls
        ]
    return payload


# ---------------------------------------------------------------------------
# Search tool and subagent
# ---------------------------------------------------------------------------


def _trim_text(value: Any, limit: int = 1800) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def normalize_tavily_results(raw: Any, source: str = "tavily") -> list[dict[str, str]]:
    if isinstance(raw, dict):
        items = raw.get("results", [])
        answer = raw.get("answer", "")
    else:
        items = raw or []
        answer = ""

    normalized: list[dict[str, str]] = []
    for item in items:
        normalized.append(
            {
                "title": str(item.get("title", "Без названия")),
                "url": str(item.get("url", "")),
                "content": _trim_text(item.get("content") or item.get("snippet") or "", 900),
                "raw_content": _trim_text(item.get("raw_content") or item.get("rawContent") or "", 1800),
                "answer": _trim_text(answer, 900),
                "score": str(item.get("score", "")),
                "source": source,
            }
        )
    return normalized


def require_live_search() -> None:
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY обязателен: поиск выполняется через Tavily API.")


def search_web(query: str, max_results: int = 8) -> list[dict[str, str]]:
    require_live_search()
    try:
        from tavily import TavilyClient
    except Exception as exc:
        raise RuntimeError("Package `tavily-python` is not installed. Install requirements.txt first.") from exc

    with maybe_span(
        "tool.search_web",
        kind="TOOL",
        input_value={"query": query, "max_results": max_results},
        query=query,
        max_results=max_results,
        live=True,
    ) as span:
        tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        raw = tavily.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer="advanced",
            include_raw_content="markdown",
            include_usage=True,
            auto_parameters=True,
            timeout=45,
        )
        results = normalize_tavily_results(raw, source="tavily")
        if not results:
            raise RuntimeError(f"Tavily вернул пустой результат для запроса: {query}")
        span.set_output(
            [
                {"title": item["title"], "url": item["url"], "score": item.get("score", "")}
                for item in results
            ]
        )
        return results


search_tool = ToolSpec(
    name="search_web",
    description="Искать в интернете через Tavily и возвращать расширенный контекст: answer, raw markdown, content, score и URL.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Поисковый запрос. Язык выбирается по контексту задачи и ожидаемым источникам.",
            },
            "max_results": {"type": "integer", "description": "Сколько результатов вернуть", "default": 8},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=search_web,
)


def collect_urls_from_text(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]>]+", text)
    return list(dict.fromkeys(urls))


def evidence_from_results(results: list[dict[str, str]], limit: int = 12) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in results:
        url = item.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        evidence.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("content") or item.get("answer") or item.get("raw_content", "")[:500],
            }
        )
        if len(evidence) >= limit:
            break
    return evidence


SEARCH_SUBAGENT_SYSTEM_PROMPT = f"""
Ты поисковый подагент deep research системы. Сегодня {TODAY}.
У тебя есть только один инструмент: search_web, который делает Tavily search.

Твоя роль узкая: собрать проверяемые заметки по делегированному вопросу для главного агента.
Ты не главный агент: не пиши финальный отчёт, не выбирай победителя во всей задаче, не меняй критерии сравнения и не расширяй область исследования без необходимости.

Правила:
1. Не отвечай из памяти. Для каждого задания вызови search_web минимум два раза с разными запросами.
2. Язык запроса выбирай по контексту: английский для глобальных технических/официальных источников, русский для русскоязычных тем, локальный язык для локальных источников.
3. Отфильтровывай нерелевантные результаты. Если запрос ушёл в слишком общий web/frameworks-контекст вместо AI-agent frameworks, сделай уточняющий запрос.
4. Итоговые заметки всегда пиши на русском.
5. Пиши сжато, но содержательно: дай факты, ограничения, противоречия и ссылки, а не самостоятельный обзорный отчёт.
6. Каждое важное фактическое утверждение снабжай markdown-ссылкой.
7. Не используй заголовок первого уровня `#`. Верни структуру: "Краткий ответ", "Факты с источниками", "Ограничения/спорные места", "Источники".
8. В разделе "Источники" перечисли все использованные URL.
""".strip()


def run_search_subagent(question: str, max_iterations: int = 6) -> dict[str, Any]:
    if not live_model_available():
        raise RuntimeError("OPENROUTER_API_KEY обязателен: search subagent должен вызывать live model.")
    require_live_search()

    sub_registry = ToolRegistry()
    sub_registry.register(search_tool)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SEARCH_SUBAGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Исследовательский вопрос: {question}"},
    ]
    search_count = 0
    all_results: list[dict[str, str]] = []

    with maybe_span("subagent.search", kind="AGENT", input_value={"question": question}, question=question) as span:
        for _iteration in range(1, max_iterations + 1):
            message = call_model(messages, tools=sub_registry.schemas())
            messages.append(assistant_message_to_dict(message))
            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:
                if search_count >= 2:
                    notes = message.content or ""
                    sources = collect_urls_from_text(notes)
                    if not sources:
                        sources = [item["url"] for item in all_results if item.get("url")]
                    payload = {
                        "question": question,
                        "notes": notes,
                        "sources": list(dict.fromkeys(sources)),
                        "evidence": evidence_from_results(all_results),
                        "search_count": search_count,
                    }
                    span.set_output(
                        {
                            "question": question,
                            "search_count": search_count,
                            "source_count": len(payload["sources"]),
                            "notes_preview": notes[:1200],
                        }
                    )
                    return payload
                messages.append(
                    {
                        "role": "user",
                        "content": "Ты ещё не сделал минимум два Tavily search calls. Вызови search_web с уточняющим запросом.",
                    }
                )
                continue

            for call in tool_calls:
                result = dispatch_tool(call, sub_registry)
                if _tool_call_name(call) == "search_web":
                    search_count += 1
                    all_results.extend(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _tool_call_id(call),
                        "name": _tool_call_name(call),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

    raise RuntimeError(f"Search subagent reached iteration limit before final notes: {question}")


def delegate_search(queries: list[str]) -> list[dict[str, Any]]:
    answers = []
    selected_queries = queries[:3]
    with maybe_span(
        "agent.delegate_search",
        kind="AGENT",
        input_value={"queries": selected_queries},
        query_count=len(selected_queries),
    ) as span:
        for query in selected_queries:
            answers.append(run_search_subagent(query))
        span.set_output(
            [
                {
                    "question": item["question"],
                    "search_count": item.get("search_count"),
                    "source_count": len(item.get("sources", [])),
                }
                for item in answers
            ]
        )
    return answers


# ---------------------------------------------------------------------------
# Main researcher tools and loop
# ---------------------------------------------------------------------------


DEMO_QUERY = (
    "Проведи исследование и выбери лучший Python-фреймворк для создания учебного deep research agent. "
    "Сначала сам определи критерии выбора и найди релевантные open-source кандидаты. "
    "Затем составь shortlist из 4–6 фреймворков, сравни их по архитектуре, зрелости, "
    "наблюдаемости, поддержке итеративного поиска, human-in-the-loop, типизации и простоте объяснения участникам. "
    "В финале дай рекомендацию для мастер-класса: какой фреймворк выбрать, почему, и какие альтернативы стоит упомянуть."
)


MAIN_AGENT_SYSTEM_PROMPT = f"""
Ты учебный исследовательский агент. Сегодня {TODAY}.

Ты решаешь задачу через доступные инструменты: строишь план, делегируешь поиск подагентам, обновляешь задачи и сохраняешь итоговый markdown-отчёт.
Главный агент отвечает за план, критерии сравнения, синтез, финальные выводы и сохранение отчёта.
Поисковые подагенты — только поставщики проверяемых заметок и источников.

Режимы:
1. В начале обязательно вызови generate_plan с 3–6 проверяемыми todos.
2. После generate_plan исследуй задачи через delegate_search. Это единственный способ веб-поиска для главного агента.
3. delegate_search принимает 1–3 distinct вопроса для поисковых подагентов. Формулируй их как узкие исследовательские вопросы: один аспект, один фреймворк или один тип доказательств на подагента.
4. Не делегируй подагенту роль главного автора: не проси его написать весь отчёт, выбрать общего победителя или сравнить всё сразу, если это должен синтезировать ты.
5. После получения результатов используй modify_todo, чтобы закрывать выполненные пункты или добавлять важные rabbit holes.
6. Финальный отчёт пиши на русском. Поисковые вопросы и запросы подагентов могут быть на языке лучших источников.

Правила работы с источниками:
7. Различай уровень источника:
   - официальная документация, changelog, GitHub организации/проекта и публикации команды продукта — основной источник фактов о возможностях, API, архитектуре и статусе проекта;
   - issue/PR/discussion в официальном репозитории — сильный сигнал о реальных ограничениях, но формулируй аккуратно;
   - независимые блоги, обзоры, benchmark-посты, Medium/dev.to/Substack — мнения, интерпретации и практический опыт, а не окончательная истина;
   - SEO-сравнения и vendor-блоги — полезны для ориентира, но не должны быть единственной опорой для сильного вывода.
8. Не выдавай мнение блогера за факт. Пиши "по оценке автора обзора", "в сравнительном обзоре утверждается", "официальная документация подтверждает".
9. Для спорных утверждений указывай уровень уверенности или контекст: "вероятно", "по доступным источникам", "для новых проектов выглядит рискованно", а не абсолютные формулировки.
10. Если источники расходятся, покажи расхождение и объясни, какой источник сильнее и почему.

Правила финального отчёта:
11. Используй обычные markdown-ссылки только в формате `[Источник](https://...)`. Не используй двойные скобки вида `[[Источник]](https://...)`.
12. В отчёте должны быть inline markdown-ссылки или явная привязка фактов к URL, а в конце — раздел "Источники".
13. Рекомендацию формулируй как архитектурный выбор под заданные критерии, а не как абсолютного победителя рынка. Если уместен гибрид, явно объясни роли компонентов.
14. Отделяй факты из официальных источников от выводов на основе обзоров и собственного синтеза.
15. Перед финальным ответом обязательно вызови save_report и сохрани markdown в файл report.md.
16. В чат не выводи весь отчёт; после save_report дай короткое резюме и путь к файлу.
""".strip()


def make_generate_plan_tool(run_state: RunState) -> ToolSpec:
    def generate_plan(todos: list[str]) -> dict[str, Any]:
        added = run_state.add_todos(todos[:6])
        run_state.mode = "execute"
        return {
            "result": "Plan accepted. Switch to execute mode and start using research tools.",
            "mode": run_state.mode,
            "todos": list(run_state.todos),
            "added": added,
        }

    return ToolSpec(
        name="generate_plan",
        description="Создать проверяемый план исследования и перевести агента из plan mode в execute mode.",
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 6,
                    "description": "3–6 distinct исследовательских задач. Последняя задача должна проверять ссылки/цитирование.",
                },
            },
            "required": ["todos"],
            "additionalProperties": False,
        },
        handler=generate_plan,
    )


def make_modify_todo_tool(run_state: RunState) -> ToolSpec:
    def modify_todo(action: str, todos: list[str]) -> dict[str, Any]:
        if action == "add":
            added = run_state.add_todos(todos)
            return {"action": action, "added": added, "todos": list(run_state.todos)}
        if action == "remove":
            removed, missing = run_state.remove_todos(todos)
            return {"action": action, "removed": removed, "missing": missing, "todos": list(run_state.todos)}
        raise ValueError("action must be 'add' or 'remove'")

    return ToolSpec(
        name="modify_todo",
        description="Добавить новые todos или отметить существующие todos выполненными.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "remove"]},
                "todos": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
            "required": ["action", "todos"],
            "additionalProperties": False,
        },
        handler=modify_todo,
    )


def make_delegate_search_tool(run_state: RunState) -> ToolSpec:
    def delegate_search_tool_handler(queries: list[str]) -> dict[str, Any]:
        results = delegate_search(queries[:3])
        for item in results:
            run_state.add_finding(item["question"], item["notes"], item["sources"], item["evidence"])
        return {"results": results, "finding_count": len(run_state.findings)}

    return ToolSpec(
        name="delegate_search",
        description="Делегировать 1–3 distinct исследовательских вопроса search subagents. Каждый subagent сам вызывает Tavily search_web.",
        parameters={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                    "description": "Вопросы для подагентов. Формулируй как исследовательские вопросы, не как однословные keywords.",
                },
            },
            "required": ["queries"],
            "additionalProperties": False,
        },
        handler=delegate_search_tool_handler,
    )


def make_save_report_tool(run_state: RunState, default_filename: str = DEFAULT_REPORT_PATH) -> ToolSpec:
    def save_report(filename: str, markdown: str) -> dict[str, Any]:
        path = Path(filename or default_filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("filename must be a relative path inside the working directory")
        if path.suffix.lower() != ".md":
            path = path.with_suffix(".md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        run_state.report_path = str(path)
        run_state.mode = "final"
        return {"result": "report_saved", "path": str(path), "chars": len(markdown)}

    return ToolSpec(
        name="save_report",
        description="Сохранить итоговый исследовательский отчёт в markdown-файл.",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": f"Относительный путь к .md файлу, например {default_filename}"},
                "markdown": {"type": "string", "description": "Полный markdown отчёт на русском с источниками"},
            },
            "required": ["filename", "markdown"],
            "additionalProperties": False,
        },
        handler=save_report,
    )


def make_research_registry(run_state: RunState, report_filename: str = DEFAULT_REPORT_PATH) -> ToolRegistry:
    active_registry = ToolRegistry()
    active_registry.register(make_generate_plan_tool(run_state))
    active_registry.register(make_modify_todo_tool(run_state))
    active_registry.register(make_delegate_search_tool(run_state))
    active_registry.register(make_save_report_tool(run_state, default_filename=report_filename))
    return active_registry


def require_live_agent_runtime() -> None:
    if not live_model_available():
        raise RuntimeError("OPENROUTER_API_KEY обязателен: главный агент должен вызывать live model.")
    require_live_search()


def run_researcher(query: str, max_iterations: int = 24, report_filename: str = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    require_live_agent_runtime()
    run_state = RunState(mode="plan")
    research_registry = make_research_registry(run_state, report_filename=report_filename)
    system_prompt = (
        MAIN_AGENT_SYSTEM_PROMPT
        + f"\n\nДля этого запуска сохрани итоговый отчёт через save_report(filename=\"{report_filename}\", markdown=...)."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    with maybe_span(
        "agent.full_researcher",
        kind="AGENT",
        input_value={"query": query},
        query=query,
        max_iterations=max_iterations,
    ) as run_span:
        for iteration in range(1, max_iterations + 1):
            run_state.iteration_count = iteration
            print(f"[baseline] iteration={iteration} mode={run_state.mode} pending_todos={len(run_state.todos)}")
            with maybe_span(
                "agent.iteration",
                kind="AGENT",
                input_value={"iteration": iteration, "mode": run_state.mode, "pending_todos": list(run_state.todos)},
                iteration=iteration,
                mode=run_state.mode,
                pending_todos=len(run_state.todos),
            ) as iteration_span:
                message = call_model(messages, tools=research_registry.schemas())
                messages.append(assistant_message_to_dict(message))
                tool_calls = getattr(message, "tool_calls", None) or []

                if not tool_calls:
                    reason = run_state.is_incomplete()
                    if reason:
                        messages.append({"role": "user", "content": reason})
                        print(f"[baseline] continuing: {reason}")
                        continue
                    payload = {
                        "final_message": message.content or "Отчёт сохранён.",
                        "report_path": run_state.report_path,
                        "state": run_state,
                    }
                    iteration_span.set_output(
                        {"final_message": payload["final_message"], "report_path": run_state.report_path}
                    )
                    run_span.set_output({"final_message": payload["final_message"], "report_path": run_state.report_path})
                    return payload

                tool_names = [_tool_call_name(call) for call in tool_calls]
                print(f"[baseline] tool_calls={tool_names}")
                iteration_span.set_output({"tool_calls": tool_names})
                for call in tool_calls:
                    result = dispatch_tool(call, research_registry)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _tool_call_id(call),
                            "name": _tool_call_name(call),
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

        raise RuntimeError("Главный агент достиг лимита итераций. Остаток: " + (run_state.is_incomplete() or "нет деталей"))


# ---------------------------------------------------------------------------
# CLI and smoke checks
# ---------------------------------------------------------------------------


def check_baseline_contract() -> dict[str, Any]:
    """Offline check: no live API calls; validates schemas and local wiring."""
    state = RunState(mode="plan")
    registry = make_research_registry(state, report_filename="baseline_report.md")
    schemas = registry.schemas()
    return {
        "today": TODAY,
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "live_model": live_model_available(),
        "live_search": live_search_available(),
        "tools": registry.names(),
        "schema_count": len(schemas),
        "required_tools_present": all(
            name in registry.names()
            for name in ["generate_plan", "modify_todo", "delegate_search", "save_report"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CodeFest baseline deep-research agent.")
    parser.add_argument("--query", "-q", default=DEMO_QUERY, help="Research task for the agent.")
    parser.add_argument("--query-file", help="Read research task from a UTF-8 text file.")
    parser.add_argument("--output", "-o", default=DEFAULT_REPORT_PATH, help="Relative markdown output path.")
    parser.add_argument("--max-iterations", type=int, default=24, help="Main-agent iteration limit.")
    parser.add_argument("--check", action="store_true", help="Run offline wiring/schema check and exit.")
    parser.add_argument("--no-phoenix", action="store_true", help="Disable optional Phoenix tracing setup.")
    args = parser.parse_args()

    if args.check:
        print(json.dumps(check_baseline_contract(), ensure_ascii=False, indent=2))
        return 0

    if args.query_file:
        query = Path(args.query_file).read_text(encoding="utf-8").strip()
    else:
        query = args.query.strip()
    if not query:
        raise SystemExit("Empty query")

    setup_phoenix_tracing(enable=not args.no_phoenix)
    result = run_researcher(query, max_iterations=args.max_iterations, report_filename=args.output)
    print("\n=== Baseline result ===")
    print(result["final_message"])
    print("Отчёт сохранён:", result["report_path"])
    if PHOENIX_UI_URL:
        print("Phoenix UI:", PHOENIX_UI_URL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
