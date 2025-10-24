# Repository Guidelines

## Project Structure & Module Organization
Core workflows live in `arbee/`, with `arbee/agents/` housing LangGraph agent logic, `arbee/workflow/` orchestrating multi-agent runs, and `arbee/api_clients/` plus `arbee/database/` handling external integrations. Shared helpers sit in `arbee/utils/`. CLI entry points and operational tooling are in `scripts/`, configuration defaults live in `config/settings.py`, and integration, math, and security checks are grouped under `tests/`.

## Build, Test, and Development Commands
- `pip install -e .` installs the package locally; add `".[dev]"` for linting and typing extras.
- `python scripts/run_polyseer.py --market https://...` runs the end-to-end market analysis; use `--help` for options.
- `pytest` executes the full test matrix; target individual suites with `pytest tests/test_single_market_workflow.py -vv`.
- `ruff check arbee tests` enforces lint rules, while `black arbee tests` formats sources with a 100-character line length.
- `mypy arbee` type-checks core modules using Python 3.11 settings.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints on new public functions, and prefer dataclasses or Pydantic models for structured data. Modules and packages are snake_case; classes are PascalCase; async functions use verb-based snake_case (`fetch_market_quotes`). Run `black` and `ruff` before push; keep imports organized via Ruff’s fixer.

## Testing Guidelines
Tests reside in `tests/` and follow `test_<feature>.py` with function names mirroring the behavior under test. Use pytest fixtures when touching database or external clients, and mark slow network calls with `@pytest.mark.integration`. Run `pytest --cov=arbee --cov-report=term-missing` before submitting and ensure new agents include scenario coverage in `test_agent_workflows.py`.

## Commit & Pull Request Guidelines
Commit messages follow the existing Sentence case style (`Refactor agents and enhance validation mechanisms`) and should describe the observable change. Each PR should summarize intent, outline risk areas, reference linked issues, and confirm `pytest`, linters, and type checks passed. Include screenshots or logs when altering external integrations or CLI outputs.

## Security & Configuration Tips
Keep secrets in `.env` (copy `.env.example`) and never commit credentialized files. Review `config/settings.py` before enabling new providers, and rotate API keys in Supabase if test runs involve real-money platforms. Use mocked clients in tests unless the environment explicitly sets sandbox credentials.
