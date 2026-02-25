# CLAUDE.md — Waybill Project Guide

## Project Overview

**Waybill** is a Python 3.12+ CLI tool for finding profitable cargo hauling routes in the MMORPG EVE Online. It queries EVE's public ESI (External Swagger Interface) REST API — no game authentication required — and ranks trade opportunities by net ISK profit and ISK per cubic meter.

**Naming quirk:** The installable CLI is `waybill`, but the Python package inside `src/` is `eve_haul`. The entry point wires them: `waybill = eve_haul.cli:main`.

---

## Tech Stack

- **Python 3.12+** (only runtime dep: `typer >= 0.12.0`)
- **Typer** for CLI (backed by Click + Rich)
- **SQLite** (stdlib `sqlite3`) for all local caching — no external DB
- **urllib.request** for HTTP — no requests/httpx
- **uv** for virtual env and package management
- **just** (justfile) as a task runner

No test suite, no linter config, no CI/CD pipeline exists yet.

---

## Project Structure

```
src/eve_haul/
├── __init__.py     # Version: __version__ = "0.1.0"
├── __main__.py     # Enables: python -m eve_haul
├── cli.py          # All Typer commands and presentation logic (~1600 lines)
├── esi.py          # ESI API client + SQLite cache layer (~1100 lines)
├── market.py       # Pure computation: ranking, BFS traversal, fee math (~390 lines)
└── models.py       # Frozen dataclasses: MarketOrder, ItemInfo, StationRef, etc.

docs/index.html     # Bootstrap 5 single-page web manual (EVE dark theme)
README.md           # Full CLI reference and usage documentation
TODO.md             # Planned: TUI mode
pyproject.toml      # Build config (setuptools, src layout)
uv.lock             # Locked dependency manifest
justfile            # install / uninstall / upgrade shortcuts
```

**Runtime data paths:**
- Cache DB: `~/.cache/waybill/waybill.sqlite` (overridable via `--cache-db`)
- Config: `~/.config/waybill/config.toml` (optional; missing file = use defaults)

---

## Commands

```bash
# Install locally for development
uv venv && uv pip install -e .

# Run commands
uv run waybill route --from-system "Hek" --to-system "Aldik" --cargo-m3 60000 --min-profit 100000
uv run waybill best-sell --item "Tritanium" --from-system "Aldik" --max-jumps 5 --sec-band hi
uv run waybill sync-items --verbose

# Task runner shortcuts
just install    # uv tool install --no-cache .  (installs globally)
just uninstall  # uv tool uninstall waybill
just upgrade    # uninstall + reinstall
```

---

## Architecture

### Layer separation (strict)

| Module | Role |
|---|---|
| `models.py` | Pure data — frozen dataclasses, no logic |
| `esi.py` | I/O — all network calls and SQLite reads/writes via `EsiClient` class |
| `market.py` | Pure computation — stateless functions, dependency-injected callables |
| `cli.py` | Presentation — Typer commands, table rendering, progress output |

### Key design decisions

- **Dependency injection for testability:** `compute_opportunities()` and graph traversal functions take callables (e.g., `type_info: Callable[[int], ItemInfo]`, `get_neighbors`) instead of coupling to `EsiClient`.
- **SQLite as universal cache:** Single file, upserts via `ON CONFLICT DO UPDATE`. Tables: `http_cache`, `item_types`, `market_orders`, `region_names`, `region_fetch_times`, `system_info`, `system_neighbors`, `station_names`.
- **Manual HTTP caching:** ETag-based conditional requests (HTTP 304) and `Cache-Control max-age` / `Expires` parsing — all hand-rolled with `urllib.request`.
- **Parallelism:** `ThreadPoolExecutor` for paginated market order fetching and bulk item type lookups. GIL protects `list.extend` so no explicit locking needed there.
- **BFS for graph traversal:** `systems_within_jumps`, `shortest_path`, `systems_within_route_jumps` all use `collections.deque`. Adjacency graph is lazily fetched from ESI and persisted to SQLite.
- **No global state** (except `EsiClient.last_region_fetch` dict and `cache_path`).

---

## Code Conventions

- `from __future__ import annotations` in every module
- Union types use `str | None` style (not `Optional[str]`)
- All dataclasses are `frozen=True`
- Comprehensive type hints on all function signatures and most locals
- `stdout` for output data, `stderr` for progress/status/errors (pipeline-composable)
- Parameterized SQL queries (`?` placeholders) throughout — no string interpolation
- Hand-rolled `Ansi` class for color (respects `NO_COLOR`, `TERM=dumb`, `isatty()`)
- Hand-rolled `render_table()` for column-aligned output (first col left, rest right)
- Hand-rolled `FetchProgress` / `CountProgress` using `\r` to `stderr` with thread locks
- `SimpleNamespace` bundles CLI args before passing to `run_*()` functions
- ISK formatted with `f"{value:,.2f}"` (comma-separated, 2 decimal places)

---

## Config File Format (`~/.config/waybill/config.toml`)

```toml
[item_type_refresh]
oldest_limit = 500
max_age_seconds = 604800  # 7 days

[market_cache]
ttl_seconds = 1800  # 30 minutes
```

Read with stdlib `tomllib`. Missing file silently uses defaults.

---

## EVE Online Domain Context

- **ISK** — in-game currency
- **m³** — cargo volume (cubic meters)
- **ESI** — EVE's public REST API (`https://esi.evetech.net`)
- **Regions / Systems / Stations** — the market hierarchy: regions contain systems, systems contain stations
- **Security bands:** `hi` (high-sec), `lo` (low-sec), `null` — relevant to route safety
- **Broker fees + sales tax** — applied to compute net profit from buy/sell spread
- **Instant orders** — "buy now" market orders; **resting orders** — limit orders that wait for a buyer

---

## What Does Not Exist Yet

- Tests (no pytest, no test files)
- Linter / formatter config (no ruff, mypy, flake8)
- CI/CD pipeline
- TUI mode (planned in TODO.md)
