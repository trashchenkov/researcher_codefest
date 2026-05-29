# CodeFest AI Researcher

Учебный репозиторий для мастер-класса: шаг за шагом собираем framework-free исследовательского агента на Python.

Идея мастер-класса:

> **Агент = Модель + обвязка (harness).**

Внутри обвязки мы показываем: вызовы модели через OpenRouter, инструменты, Tavily web search, планирование, состояние/todo, поискового подагента, сохранение markdown-отчёта и наблюдаемость через Phoenix.

## Что входит в репозиторий

- `notebooks/codefest_ai_researcher_masterclass.ipynb` — основной блокнот мастер-класса на русском.
- `baseline_research_agent.py` — standalone-версия базового исследовательского агента из блокнота.
- `.env.example` — шаблон переменных окружения без секретов.
- `requirements.txt` — зависимости для запуска блокнота и baseline.
- `participant_scoring_criteria.md` — критерии оценки улучшений участниками относительно baseline.

Презентация будет добавлена отдельно позже.

## Требования

- Python **3.12**.
- Ключ OpenRouter для live-вызовов модели.
- Ключ Tavily для live web search.

На macOS с Homebrew:

```bash
brew install python@3.12
```

## Установка

```bash
cd codefest-ai-researcher
/opt/homebrew/bin/python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Настройка ключей

```bash
cp .env.example .env
```

Заполните в `.env`:

```bash
OPENROUTER_API_KEY=...
TAVILY_API_KEY=...
```

По умолчанию используется:

```bash
OPENROUTER_MODEL=z-ai/glm-5.1
OPENROUTER_FALLBACK_MODEL=openrouter/auto
```

Для строго бесплатного эксперимента через OpenRouter можно поставить `:free` модель и для primary, и для fallback, например:

```bash
OPENROUTER_MODEL=openai/gpt-oss-120b:free
OPENROUTER_FALLBACK_MODEL=openai/gpt-oss-120b:free
```

Важно: бесплатные модели могут работать заметно хуже основной модели — хуже следовать инструкциям, нестабильно вызывать tools, терять ссылки или давать менее проверяемый отчёт. Также у free-моделей есть лимиты; одного полного запуска агента может хватить почти на дневной лимит полностью бесплатного аккаунта.

## Запуск блокнота

```bash
jupyter notebook notebooks/codefest_ai_researcher_masterclass.ipynb
```

Блокнот умеет сам поднять локальный Phoenix, если зависимости установлены и порт свободен. По умолчанию UI Phoenix открывается на:

```text
http://127.0.0.1:6006/
```

## Запуск standalone baseline

```bash
.venv/bin/python baseline_research_agent.py \
  --output report.md \
  --max-iterations 24
```

Отключить Phoenix для CLI-прогона:

```bash
.venv/bin/python baseline_research_agent.py --no-phoenix --output report.md
```

Если Phoenix уже запущен через `phoenix serve`, baseline отправит туда трейсы, если не передавать `--no-phoenix` и не задавать `DISABLE_PHOENIX=1`.

## Phoenix и версии зависимостей

Для воспроизводимости мастер-класса Phoenix-стек закреплён на проверенных версиях Phoenix 13:

```txt
arize-phoenix==13.12.0
arize-phoenix-evals==2.11.0
arize-phoenix-otel==0.15.0
fastapi==0.135.1
starlette==0.52.1
uvicorn==0.41.0
```

Не обновляйте Phoenix до latest без отдельной проверки блокнота и UI трейсинга.

## Проверка

Статическая/offline-проверка блокнота:

```bash
.venv/bin/python tests/validate_notebook.py --offline
```

Проверка импортов зависимостей:

```bash
.venv/bin/python tests/validate_notebook.py --deps
```

Live-проверки с ключами и Phoenix:

```bash
.venv/bin/python tests/validate_notebook.py --live --start-phoenix
```

## Безопасность

- Не коммитьте `.env`.
- Не публикуйте API-ключи в notebook outputs, отчётах или логах.
- Перед публикацией репозитория проверьте, что в блокноте нет выполненных outputs с локальными путями или секретами.
