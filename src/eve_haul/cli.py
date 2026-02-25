from __future__ import annotations

import math
import os
import sqlite3
import sys
import threading
import time
import tomllib
from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace
from typing import Any, Iterable

import typer

from .esi import EsiClient
from .market import (
    best_station_prices,
    best_prices_for_item,
    build_order_books,
    compute_opportunities,
    shortest_path,
    systems_within_jumps,
    systems_within_route_jumps,
)
from .mcp_server import create_server


def format_isk(value: float) -> str:
    return f"{value:,.2f}"


def format_volume(value: float) -> str:
    return f"{value:,.2f}"


def format_duration_minutes(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining = minutes % 60
    return f"{hours}h {remaining}m"


def format_duration_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining = int(round(seconds - (minutes * 60)))
    return f"{minutes}m {remaining}s"


def format_duration_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes = total_seconds // 60
    remaining = total_seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining}s"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining}s"


def describe_route_mode(buy_order_type: str, sell_order_type: str) -> str:
    if buy_order_type == "sell" and sell_order_type == "buy":
        return "instant (sell->buy)"
    if buy_order_type == "buy" and sell_order_type == "sell":
        return "resting (buy->sell)"
    return f"{buy_order_type}->{sell_order_type}"


class Ansi:
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"


def should_color(stream: Any = sys.stderr) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def style(text: str, use_color: bool, *codes: str) -> str:
    if not use_color or not codes:
        return text
    return f"{''.join(codes)}{text}{Ansi.RESET}"


class FetchProgress:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._start_time = time.monotonic()
        self._lock = threading.Lock()
        self._total_pages = 0
        self._completed_pages = 0
        self._region_totals: dict[int, int] = {}
        self._region_labels: dict[int, str] = {}
        self._current_region_label: str | None = None
        self._active = False
        self._last_len = 0

    def update(
        self,
        region_id: int,
        page: int,
        pages: int,
        region_label: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            if region_label:
                self._region_labels[region_id] = region_label
                self._current_region_label = region_label
            elif region_id in self._region_labels:
                self._current_region_label = self._region_labels[region_id]
            if region_id in self._region_totals:
                known_pages = self._region_totals[region_id]
                if known_pages != pages:
                    self._total_pages += pages - known_pages
                    self._region_totals[region_id] = pages
            else:
                self._region_totals[region_id] = pages
                self._total_pages += pages
            self._completed_pages += 1
            self._render_locked()

    def ensure_newline(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._active:
                sys.stderr.write("\n")
                sys.stderr.flush()
                self._active = False
                self._last_len = 0

    def _render_locked(self) -> None:
        elapsed = format_duration_elapsed(time.monotonic() - self._start_time)
        total_label = str(self._total_pages) if self._total_pages > 0 else "?"
        region_prefix = ""
        if self._current_region_label:
            region_prefix = f"{self._current_region_label} | "
        if self._total_pages > 0:
            percent = int(round((self._completed_pages / self._total_pages) * 100))
            progress_label = (
                f"{region_prefix}Fetch progress: {self._completed_pages}/{total_label} pages "
                f"({percent}%) | elapsed {elapsed}"
            )
        else:
            progress_label = (
                f"{region_prefix}Fetch progress: {self._completed_pages}/{total_label} pages "
                f"| elapsed {elapsed}"
            )
        padding = ""
        if self._last_len > len(progress_label):
            padding = " " * (self._last_len - len(progress_label))
        sys.stderr.write(f"\r{progress_label}{padding}")
        sys.stderr.flush()
        self._last_len = len(progress_label)
        self._active = True


class CountProgress:
    def __init__(self, label: str, enabled: bool = True) -> None:
        self.label = label
        self.enabled = enabled
        self._start_time = time.monotonic()
        self._lock = threading.Lock()
        self._active = False
        self._last_len = 0
        self._completed = 0
        self._total = 0

    def update(self, completed: int, total: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._completed = completed
            self._total = total
            self._render_locked()

    def ensure_newline(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._active:
                sys.stderr.write("\n")
                sys.stderr.flush()
                self._active = False
                self._last_len = 0

    def _render_locked(self) -> None:
        elapsed = format_duration_elapsed(time.monotonic() - self._start_time)
        total_label = str(self._total) if self._total > 0 else "?"
        if self._total > 0:
            percent = int(round((self._completed / self._total) * 100))
            progress_label = (
                f"{self.label}: {self._completed}/{total_label} "
                f"({percent}%) | elapsed {elapsed}"
            )
        else:
            progress_label = (
                f"{self.label}: {self._completed}/{total_label} "
                f"| elapsed {elapsed}"
            )
        padding = ""
        if self._last_len > len(progress_label):
            padding = " " * (self._last_len - len(progress_label))
        sys.stderr.write(f"\r{progress_label}{padding}")
        sys.stderr.flush()
        self._last_len = len(progress_label)
        self._active = True


def render_table(headers: list[str], rows: Iterable[list[str]]) -> str:
    rows = list(rows)
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def format_row(cells: list[str]) -> str:
        parts = []
        for idx, cell in enumerate(cells):
            align = ">" if idx > 0 else "<"
            parts.append(f"{cell:{align}{widths[idx]}}")
        return "  ".join(parts)

    output = [format_row(headers)]
    output.append("  ".join("-" * width for width in widths))
    for row in rows:
        output.append(format_row(row))
    return "\n".join(output)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_config(verbose: bool = False) -> dict[str, Any]:
    config_path = os.path.expanduser("~/.config/waybill/config.toml")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "rb") as handle:
            return tomllib.load(handle)
    except Exception as exc:
        if verbose:
            print(
                f"Warning: failed to read config at {config_path}: {exc}",
                file=sys.stderr,
            )
        return {}


def get_item_type_refresh_settings(config: dict[str, Any]) -> tuple[int, int]:
    section = config.get("item_type_refresh")
    if not isinstance(section, dict):
        section = {}
    limit = _coerce_int(section.get("oldest_limit"), 10)
    max_age_seconds = _coerce_int(section.get("max_age_seconds"), 0)
    return max(0, limit), max(0, max_age_seconds)


def get_market_cache_ttl(
    config: dict[str, Any],
    cli_value: int | None,
    default: int = 14400,
) -> int:
    if cli_value is not None:
        return max(0, cli_value)
    section = config.get("market_cache")
    if not isinstance(section, dict):
        section = {}
    ttl = _coerce_int(section.get("ttl_seconds"), default)
    return max(0, ttl)


def refresh_oldest_item_types(
    client: EsiClient,
    verbose: bool,
    limit: int = 10,
    max_age_seconds: int = 0,
    workers: int = 2,
) -> None:
    fetched, _skipped, errors = client.refresh_oldest_type_info(
        limit=limit,
        max_age_seconds=max_age_seconds,
        workers=workers,
    )
    if verbose:
        print(
            f"Refreshed {fetched} oldest item types. Errors: {errors}.",
            file=sys.stderr,
        )


def print_region_cache_preflight(
    client: EsiClient,
    region_ids: Iterable[int],
    region_names: dict[int, str],
    cache_ttl: int,
    force_refresh: bool,
) -> None:
    now = int(time.time())
    total_est_seconds = 0.0
    unknown_estimates = 0
    estimated_regions = 0
    use_color = should_color(sys.stderr)
    print(style("Cache preflight:", use_color, Ansi.BOLD), file=sys.stderr)

    for region_id in region_ids:
        region_label = f"{region_names.get(region_id, region_id)} ({region_id})"
        last_fetch = client.get_region_last_fetch(region_id)
        pages_limit = (
            int(last_fetch["pages"])
            if last_fetch and last_fetch.get("pages") is not None
            else None
        )
        if cache_ttl <= 0:
            avg_seconds, samples = client.get_region_fetch_average(
                region_id, cached=False
            )
            if avg_seconds is None:
                avg_seconds, samples = client.get_region_fetch_average(region_id)
            if avg_seconds is None:
                estimate = "est fetch unknown (no history)"
                unknown_estimates += 1
            else:
                estimate = f"est fetch {format_duration_seconds(avg_seconds)} (avg of {samples})"
                total_est_seconds += avg_seconds
                estimated_regions += 1
            status_label = style("cache disabled", use_color, Ansi.MAGENTA)
            print(
                f"Region {region_label} ({status_label}): will fetch; {estimate}.",
                file=sys.stderr,
            )
            continue

        meta = client.get_region_cache_meta(region_id, pages_limit=pages_limit)
        rows_count = int(meta.get("rows_count") or 0)
        min_expires_at = meta.get("min_expires_at")
        max_fetched_at = meta.get("max_fetched_at")

        if max_fetched_at:
            fetched_dt = datetime.fromtimestamp(int(max_fetched_at), tz=timezone.utc)
            fetched_label = fetched_dt.strftime("%Y-%m-%d %H:%M UTC")
        else:
            fetched_label = "unknown"

        overdue = False
        if force_refresh:
            status = "forced refresh"
            refresh_label = "forced refresh requested"
            needs_fetch = True
        elif rows_count == 0 or min_expires_at is None:
            status = "needs refresh"
            refresh_label = "no cache found"
            needs_fetch = True
        elif pages_limit is not None and rows_count < pages_limit:
            status = "needs refresh"
            refresh_label = "cache incomplete"
            needs_fetch = True
        else:
            seconds_left = int(min_expires_at) - now
            if seconds_left > 0:
                minutes_left = (seconds_left + 59) // 60
                refresh_label = f"next refresh in {format_duration_minutes(minutes_left)}"
                status = "cache active"
                needs_fetch = False
            else:
                minutes_over = (abs(seconds_left) + 59) // 60
                refresh_label = f"refresh overdue by {format_duration_minutes(minutes_over)}"
                status = "needs refresh"
                needs_fetch = True
                overdue = True

        if pages_limit is not None:
            refresh_label = f"{refresh_label}; pages cached {rows_count}/{pages_limit}"

        estimate = ""
        if needs_fetch:
            avg_seconds, samples = client.get_region_fetch_average(
                region_id, cached=False
            )
            if avg_seconds is None:
                avg_seconds, samples = client.get_region_fetch_average(region_id)
            if avg_seconds is None:
                estimate = " est fetch unknown (no history)"
                unknown_estimates += 1
            else:
                estimate = f" est fetch {format_duration_seconds(avg_seconds)} (avg of {samples})"
                total_est_seconds += avg_seconds
                estimated_regions += 1

        if force_refresh:
            status_color = Ansi.MAGENTA
        elif needs_fetch:
            status_color = Ansi.RED if overdue else Ansi.YELLOW
        else:
            status_color = Ansi.GREEN
        status_label = style(status, use_color, status_color)
        print(
            f"Region {region_label} ({status_label}): last fetch {fetched_label}; "
            f"{refresh_label}.{estimate}",
            file=sys.stderr,
        )

    if estimated_regions > 0 or unknown_estimates > 0:
        if estimated_regions > 0:
            total_label = format_duration_seconds(total_est_seconds)
            summary = f"Total est fetch: {total_label}"
        else:
            summary = "Total est fetch: unknown"
        if unknown_estimates > 0:
            summary = f"{summary} (+{unknown_estimates} unknown)"
        print(summary, file=sys.stderr)


def format_region_fetch_summary(
    region_label: str,
    meta: dict[str, Any],
    use_color: bool,
) -> str:
    status = "cache hit" if meta.get("cached") else "refreshed"
    status_color = Ansi.GREEN if meta.get("cached") else Ansi.CYAN
    status_label = style(status, use_color, status_color)

    pages = meta.get("pages")
    cached_pages = meta.get("cached_pages")
    fetched_pages = meta.get("fetched_pages")

    page_label = ""
    if cached_pages is not None and fetched_pages is not None:
        if cached_pages == 0:
            page_label = f"{fetched_pages} pages fetched"
        elif fetched_pages == 0:
            page_label = f"{cached_pages} pages cached"
        else:
            page_label = f"{cached_pages} cached, {fetched_pages} fetched"
        if pages is not None and pages != cached_pages + fetched_pages:
            page_label = f"{pages} pages ({page_label})"
    elif pages is not None:
        page_label = f"{pages} pages"

    if page_label:
        return f"Region {region_label} ({status_label}): {page_label}."
    return f"Region {region_label} ({status_label})."


def print_total_elapsed(start_time: float, enabled: bool) -> None:
    if not enabled:
        return
    elapsed = format_duration_elapsed(time.monotonic() - start_time)
    print(f"Total elapsed: {elapsed}", file=sys.stderr)


def print_section_break(stream: Any, enabled: bool) -> None:
    if not enabled:
        return
    print("", file=stream)

app = typer.Typer(
    help="Find station-to-station hauling opportunities in EVE Online.",
    add_completion=False,
    no_args_is_help=True,
)


class SecBand(str, Enum):
    hi = "hi"
    low = "low"
    null = "null"
    all = "all"


@app.command("route", help="Find profitable hauling opportunities between stations.")
def route(
    from_station: str | None = typer.Option(None, "--from-station", help="Source station name"),
    from_system: str | None = typer.Option(
        None, "--from-system", help="Source system name (all NPC stations included)"
    ),
    to_station: str | None = typer.Option(None, "--to-station", help="Destination station name"),
    to_system: str | None = typer.Option(
        None, "--to-system", help="Destination system name (all NPC stations included)"
    ),
    cargo_m3: float = typer.Option(..., "--cargo-m3", help="Cargo limit in m³"),
    min_cargo_fill_pct: float = typer.Option(
        0.0,
        "--min-cargo-fill-pct",
        help="Minimum cargo utilization per row as a percentage (0-100)",
    ),
    min_profit: float = typer.Option(
        0.0, "--min-profit", help="Minimum total net profit (ISK)"
    ),
    broker_fee: float = typer.Option(
        0.03, "--broker-fee", help="Broker fee rate (e.g. 0.03 for 3%)"
    ),
    sales_tax: float = typer.Option(
        0.08, "--sales-tax", help="Sales tax rate (e.g. 0.08 for 8%)"
    ),
    limit: int = typer.Option(20, "--limit", help="Number of rows to display"),
    instant_only: bool = typer.Option(
        False,
        "--instant-only",
        help="Only consider immediate orders (sell at source, buy at destination)",
       ,
    ),
    backhaul_mode: bool = typer.Option(
        False,
        "--backhaul-mode",
        "--backual-mode",
        help="Prioritize hold fill first (sort by total volume, then net profit)",
       ,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Print progress information while fetching data"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Hide preflight, live fetch, and summary output",
       ,
    ),
    no_preflight: bool = typer.Option(
        False, "--no-preflight", help="Hide cache preflight section"
    ),
    no_live: bool = typer.Option(
        False, "--no-live", help="Hide live fetch status output"
    ),
    no_summary: bool = typer.Option(
        False, "--no-summary", help="Hide total elapsed summary line"
    ),
    no_output: bool = typer.Option(
        False, "--no-output", help="Hide final results output"
    ),
    allow_structures: bool = typer.Option(
        False,
        "--allow-structures",
        help="Allow station name resolution to match player structures",
       ,
    ),
    cache_db: str = typer.Option(
        "~/.cache/waybill/waybill.sqlite",
        "--cache-db",
        help="SQLite cache database path",
    ),
    market_cache_ttl: int | None = typer.Option(
        None,
        "--market-cache-ttl",
        help="Market order cache TTL in seconds (default 14400)",
    ),
    refresh_market: bool = typer.Option(
        False,
        "--refresh-market",
        help="Force refresh market orders (ignore cache)",
       ,
    ),
    page_workers: int = typer.Option(
        1, "--page-workers", help="Parallel workers for fetching region pages"
    ),
    off_route_jumps: int = typer.Option(
        0,
        "--off-route-jumps",
        help="Find better destinations within N jumps of the shortest route",
    ),
) -> None:
    if (from_station is None) == (from_system is None):
        typer.secho(
            "Error: provide exactly one of --from-station or --from-system.",
            err=True,
        )
        raise typer.Exit(code=2)
    if (to_station is None) == (to_system is None):
        typer.secho(
            "Error: provide exactly one of --to-station or --to-system.",
            err=True,
        )
        raise typer.Exit(code=2)

    args = SimpleNamespace(
        from_station=from_station,
        from_system=from_system,
        to_station=to_station,
        to_system=to_system,
        cargo_m3=cargo_m3,
        min_cargo_fill_pct=min_cargo_fill_pct,
        min_profit=min_profit,
        broker_fee=broker_fee,
        sales_tax=sales_tax,
        limit=limit,
        instant_only=instant_only,
        backhaul_mode=backhaul_mode,
        verbose=verbose,
        quiet=quiet,
        no_preflight=no_preflight,
        no_live=no_live,
        no_summary=no_summary,
        no_output=no_output,
        allow_structures=allow_structures,
        cache_db=cache_db,
        market_cache_ttl=market_cache_ttl,
        refresh_market=refresh_market,
        page_workers=page_workers,
        off_route_jumps=off_route_jumps,
    )
    code = run_find(args)
    if code:
        raise typer.Exit(code=code)


@app.command("best-sell", help="Find the best station to sell an item within a jump radius.")
def best_sell(
    item: str = typer.Option(..., "--item", help="Item name to sell"),
    from_system: str = typer.Option(..., "--from-system", help="Origin system name"),
    max_jumps: int = typer.Option(5, "--max-jumps", help="Maximum jumps from origin system"),
    min_security: float = typer.Option(
        0.0, "--min-security", help="Minimum security status for destination systems"
    ),
    max_security: float | None = typer.Option(
        None, "--max-security", help="Maximum security status for destination systems"
    ),
    sec_band: SecBand | None = typer.Option(
        None,
        "--sec-band",
        help="Convenience security band filter (hi=0.5+, low=0.1-0.4, null<=0.0, all=any)",
        case_sensitive=False,
    ),
    avoid_low: bool = typer.Option(
        False, "--avoid-low", help="Alias for --min-security 0.5 (high-sec only)"
    ),
    instant_only: bool = typer.Option(
        False, "--instant-only", help="Only consider immediate sell to buy orders"
    ),
    limit: int = typer.Option(10, "--limit", help="Number of rows to display"),
    broker_fee: float = typer.Option(
        0.03, "--broker-fee", help="Broker fee rate (e.g. 0.03 for 3%)"
    ),
    sales_tax: float = typer.Option(
        0.08, "--sales-tax", help="Sales tax rate (e.g. 0.08 for 8%)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Print progress information while fetching data"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Hide preflight, live fetch, and summary output",
       ,
    ),
    no_preflight: bool = typer.Option(
        False, "--no-preflight", help="Hide cache preflight section"
    ),
    no_live: bool = typer.Option(
        False, "--no-live", help="Hide live fetch status output"
    ),
    no_summary: bool = typer.Option(
        False, "--no-summary", help="Hide total elapsed summary line"
    ),
    no_output: bool = typer.Option(
        False, "--no-output", help="Hide final results output"
    ),
    cache_db: str = typer.Option(
        "~/.cache/waybill/waybill.sqlite",
        "--cache-db",
        help="SQLite cache database path",
    ),
    market_cache_ttl: int | None = typer.Option(
        None,
        "--market-cache-ttl",
        help="Market order cache TTL in seconds (default 14400)",
    ),
    refresh_market: bool = typer.Option(
        False,
        "--refresh-market",
        help="Force refresh market orders (ignore cache)",
       ,
    ),
    page_workers: int = typer.Option(
        1, "--page-workers", help="Parallel workers for fetching region pages"
    ),
) -> None:
    args = SimpleNamespace(
        item=item,
        from_system=from_system,
        max_jumps=max_jumps,
        min_security=min_security,
        max_security=max_security,
        sec_band=sec_band.value if sec_band else None,
        avoid_low=avoid_low,
        instant_only=instant_only,
        limit=limit,
        broker_fee=broker_fee,
        sales_tax=sales_tax,
        verbose=verbose,
        quiet=quiet,
        no_preflight=no_preflight,
        no_live=no_live,
        no_summary=no_summary,
        no_output=no_output,
        cache_db=cache_db,
        market_cache_ttl=market_cache_ttl,
        refresh_market=refresh_market,
        page_workers=page_workers,
    )
    code = run_best_sell(args)
    if code:
        raise typer.Exit(code=code)


@app.command("sync-items", help="Sync item type metadata for cached market orders.")
def sync_items(
    region: list[int] | None = typer.Option(
        None,
        "--region",
        help="Limit to a specific region_id (can be used multiple times)",
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Refetch type info even if already cached"
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Limit number of type_ids to fetch (0 = no limit, oldest refreshed first)",
    ),
    type_workers: int = typer.Option(
        2, "--type-workers", help="Parallel workers for fetching type info"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Print progress information while fetching data"
    ),
    no_preflight: bool = typer.Option(
        False, "--no-preflight", help="Hide cache preflight section"
    ),
    no_live: bool = typer.Option(
        False, "--no-live", help="Hide live fetch status output"
    ),
    no_summary: bool = typer.Option(
        False, "--no-summary", help="Hide total elapsed summary line"
    ),
    no_output: bool = typer.Option(
        False, "--no-output", help="Hide final results output"
    ),
    cache_db: str = typer.Option(
        "~/.cache/waybill/waybill.sqlite",
        "--cache-db",
        help="SQLite cache database path",
    ),
) -> None:
    args = SimpleNamespace(
        region=region,
        refresh=refresh,
        limit=limit,
        type_workers=type_workers,
        verbose=verbose,
        no_preflight=no_preflight,
        no_live=no_live,
        no_summary=no_summary,
        no_output=no_output,
        cache_db=cache_db,
    )
    code = run_cache_types(args)
    if code:
        raise typer.Exit(code=code)


@app.command("mcp")
def mcp_serve(
    cache_db: str = typer.Option(
        "~/.cache/waybill/waybill.sqlite",
        "--cache-db",
        help="SQLite cache database path",
    ),
) -> None:
    """Start an MCP server exposing EVE Online market data tools via stdio.

    Configure in Claude Desktop as:
    {"command": "waybill", "args": ["mcp"]}
    """
    server = create_server(cache_db=cache_db)
    server.run(transport="stdio")


def run_find(args: SimpleNamespace) -> int:
    start_time = time.monotonic()
    cache_path = os.path.expanduser(args.cache_db)
    client = EsiClient(cache_path=cache_path)
    config = load_config(args.verbose)
    refresh_limit, refresh_max_age = get_item_type_refresh_settings(config)
    args.market_cache_ttl = get_market_cache_ttl(config, args.market_cache_ttl)
    show_preflight = (not args.quiet) and (not args.no_preflight)
    show_live = (not args.quiet) and (not args.no_live)
    show_summary = (not args.quiet) and (not args.no_summary)
    show_output = not args.no_output

    def log(message: str) -> None:
        if args.verbose:
            print(message, file=sys.stderr)

    try:
        if args.from_system:
            log(f"Resolving source system '{args.from_system}'...")
            source_stations = client.resolve_system_stations(args.from_system)
        else:
            log(f"Resolving source station '{args.from_station}'...")
            source_stations = [
                client.resolve_station(
                    args.from_station, allow_structures=args.allow_structures
                )
            ]

        if args.to_system:
            log(f"Resolving destination system '{args.to_system}'...")
            destination_stations = client.resolve_system_stations(args.to_system)
        else:
            log(f"Resolving destination station '{args.to_station}'...")
            destination_stations = [
                client.resolve_station(
                    args.to_station, allow_structures=args.allow_structures
                )
            ]
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    source_system_id = source_stations[0].system_id
    destination_system_id = destination_stations[0].system_id

    if args.off_route_jumps < 0:
        print("Error: --off-route-jumps must be >= 0.", file=sys.stderr)
        return 2
    if args.min_cargo_fill_pct < 0.0 or args.min_cargo_fill_pct > 100.0:
        print("Error: --min-cargo-fill-pct must be between 0 and 100.", file=sys.stderr)
        return 2

    log(f"Source stations: {len(source_stations)}")
    log(f"Destination stations: {len(destination_stations)}")
    region_ids = {station.region_id for station in source_stations + destination_stations}
    orders: list = []
    region_names: dict[int, str] = {
        region_id: client.get_region_name(region_id) for region_id in region_ids
    }
    if show_preflight:
        print_region_cache_preflight(
            client,
            region_ids,
            region_names,
            args.market_cache_ttl,
            args.refresh_market,
        )
        print_section_break(sys.stderr, show_live or show_summary)
    use_color = should_color(sys.stderr)
    fetch_progress = FetchProgress(enabled=show_live)
    fetched_region_ids: set[int] = set()

    def fetch_region(region_id: int) -> None:
        if region_id in fetched_region_ids:
            return
        region_label = f"{region_names.get(region_id, region_id)} ({region_id})"
        if args.verbose and show_live:
            if args.market_cache_ttl > 0:
                refresh_note = " (forced refresh)" if args.refresh_market else ""
                print(
                    f"Fetching market orders for region {region_label} "
                    f"(cache TTL {args.market_cache_ttl}s){refresh_note}...",
                    file=sys.stderr,
                )
            else:
                refresh_note = " (forced refresh)" if args.refresh_market else ""
                print(
                    f"Fetching market orders for region {region_label} (no cache){refresh_note}...",
                    file=sys.stderr,
                )
        progress = None
        if show_live:
            def progress(
                page: int,
                pages: int,
                region_id: int = region_id,
                region_label: str = region_label,
            ) -> None:
                fetch_progress.update(region_id, page, pages, region_label)

        region_orders = client.fetch_region_orders(
            region_id,
            cache_ttl=args.market_cache_ttl,
            progress=progress,
            page_workers=args.page_workers,
            force_refresh=args.refresh_market,
        )
        if args.verbose and show_live:
            fetch_progress.ensure_newline()
            print(
                f"Fetched {len(region_orders):,} orders from region {region_label}.",
                file=sys.stderr,
            )
        meta = client.last_region_fetch.get(region_id)
        if meta and show_live:
            fetch_progress.ensure_newline()
            print(
                format_region_fetch_summary(region_label, meta, use_color),
                file=sys.stderr,
            )
        orders.extend(region_orders)
        fetched_region_ids.add(region_id)

    for region_id in region_ids:
        fetch_region(region_id)
    fetch_progress.ensure_newline()
    print_section_break(sys.stderr, show_live and show_summary)

    log("Building order books...")
    source_station_ids = [station.station_id for station in source_stations]
    destination_station_ids = [station.station_id for station in destination_stations]
    (
        source_sells,
        source_buys,
        destination_sells,
        destination_buys,
        order_stats,
    ) = build_order_books(
        orders,
        source_station_ids,
        destination_station_ids,
    )

    if args.verbose:
        def log_station_counts(label: str, stations: list) -> None:
            total_orders = 0
            for station in stations:
                counts = order_stats[label].get(
                    station.station_id, {"total": 0, "buy": 0, "sell": 0}
                )
                total_orders += counts["total"]
                log(
                    f"{label.title()} station '{station.name}': "
                    f"{counts['total']:,} orders "
                    f"(buy {counts['buy']:,}, sell {counts['sell']:,})"
                )
            log(f"{label.title()} stations total orders: {total_orders:,}")

        log_station_counts("source", source_stations)
        if args.from_system:
            source_total = sum(
                order_stats["source"].get(st.station_id, {"total": 0})["total"]
                for st in source_stations
            )
            log(f"Source system '{args.from_system}': {source_total:,} orders total")

        log_station_counts("destination", destination_stations)
        if args.to_system:
            dest_total = sum(
                order_stats["destination"].get(st.station_id, {"total": 0})["total"]
                for st in destination_stations
            )
            log(f"Destination system '{args.to_system}': {dest_total:,} orders total")

    log("Computing opportunities...")
    opportunities = compute_opportunities(
        source_sells,
        source_buys,
        destination_sells,
        destination_buys,
        client.get_type_info,
        cargo_m3=args.cargo_m3,
        broker_fee=args.broker_fee,
        sales_tax=args.sales_tax,
        min_profit=args.min_profit,
        min_cargo_fill_ratio=args.min_cargo_fill_pct / 100.0,
        backhaul_mode=args.backhaul_mode,
        instant_only=args.instant_only,
    )

    if not opportunities:
        if show_output:
            print("No profitable opportunities found.")
        refresh_oldest_item_types(
            client,
            args.verbose,
            limit=refresh_limit,
            max_age_seconds=refresh_max_age,
        )
        return 0

    show_mode = not args.instant_only
    top_opps = opportunities[: args.limit]
    rows = []
    has_resting_mode = False
    for opp in top_opps:
        mode = describe_route_mode(opp.buy_order_type, opp.sell_order_type)
        if opp.buy_order_type == "buy" and opp.sell_order_type == "sell":
            has_resting_mode = True
        row = [
            opp.item_name,
            format_isk(opp.buy_price),
            format_isk(opp.sell_price),
            format_isk(opp.net_profit_total),
            format_isk(opp.isk_per_m3),
            f"{(opp.total_volume / args.cargo_m3 * 100.0):.2f}%",
            str(opp.quantity),
            format_volume(opp.total_volume),
        ]
        if show_mode:
            row.insert(1, mode)
        rows.append(row)

    headers = [
        "Item",
        "Source Price",
        "Destination Price",
        "Net Profit",
        "ISK/m³",
        "Fill %",
        "Qty",
        "Total Volume",
    ]
    if show_mode:
        headers.insert(1, "Mode")

    print_total_elapsed(start_time, show_summary)
    if show_output:
        print_section_break(sys.stdout, show_preflight or show_live or show_summary)
        table = render_table(headers, rows)
        print(table)
        if args.min_cargo_fill_pct > 0:
            print(f"Filter: minimum cargo fill {args.min_cargo_fill_pct:.2f}%.")
        if args.backhaul_mode:
            print("Sort: backhaul mode (Fill volume -> Net Profit -> ISK/m³).")
        if show_mode and has_resting_mode:
            print(
                "Legend: instant = immediate fills (source sell -> destination buy). "
                "resting = you place orders and wait (source buy -> destination sell). "
                "Use --instant-only for immediate routes only."
            )

    if show_output and args.off_route_jumps > 0 and top_opps:
        off_route_rows: list[list[str]] = []
        improved = 0
        best_delta = 0.0
        off_route_note: str | None = None

        route = shortest_path(
            source_system_id,
            destination_system_id,
            client.get_system_neighbors,
        )
        if not route:
            off_route_note = (
                f"No route found between systems {source_system_id} and {destination_system_id}."
            )

        off_route_systems = {}
        off_route_system_info = {}
        off_route_station_ids: list[int] = []
        off_route_station_to_system: dict[int, int] = {}
        off_route_station_detour: dict[int, int] = {}
        off_route_station_names: dict[int, str] = {}

        if off_route_note is None:
            corridor = systems_within_route_jumps(
                route,
                args.off_route_jumps,
                client.get_system_neighbors,
            )
            off_route_systems = {
                system_id: dist for system_id, dist in corridor.items() if dist > 0
            }
            if not off_route_systems:
                off_route_note = (
                    f"No off-route systems within {args.off_route_jumps} jumps of the shortest path."
                )

        if off_route_note is None:
            off_route_region_ids: set[int] = set()
            for system_id in off_route_systems:
                info = client.get_system_info(system_id)
                off_route_system_info[system_id] = info
                off_route_region_ids.add(info.region_id)

            for region_id in off_route_region_ids:
                fetch_region(region_id)

            for system_id, detour in off_route_systems.items():
                station_ids = client.get_system_station_ids(system_id)
                for station_id in station_ids:
                    off_route_station_ids.append(station_id)
                    off_route_station_to_system[station_id] = system_id
                    off_route_station_detour[station_id] = detour

            if not off_route_station_ids:
                off_route_note = (
                    f"No NPC stations within {args.off_route_jumps} jumps of the shortest path."
                )

        type_ids = {opp.type_id for opp in top_opps}
        if off_route_note is None:
            off_route_station_names = client.get_station_names(off_route_station_ids)

            source_sells_by_type = best_station_prices(
                orders,
                source_station_ids,
                is_buy_order=False,
                prefer_min=True,
                type_filter=type_ids,
            )
            source_buys_by_type = best_station_prices(
                orders,
                source_station_ids,
                is_buy_order=True,
                prefer_min=False,
                type_filter=type_ids,
            )
            dest_buys_by_type = best_station_prices(
                orders,
                off_route_station_ids,
                is_buy_order=True,
                prefer_min=False,
                type_filter=type_ids,
            )
            dest_sells_by_type = best_station_prices(
                orders,
                off_route_station_ids,
                is_buy_order=False,
                prefer_min=True,
                type_filter=type_ids,
            )

            type_info_cache = {}

            def get_type(type_id: int):
                info = type_info_cache.get(type_id)
                if info is None:
                    info = client.get_type_info(type_id)
                    type_info_cache[type_id] = info
                return info

        for opp in top_opps:
            mode = describe_route_mode(opp.buy_order_type, opp.sell_order_type)
            base_net = opp.net_profit_total
            off_net = None
            delta = None
            station_label = "-"
            system_label = "-"
            region_label = "-"
            detour_label = "-"

            if off_route_note is None:
                if opp.buy_order_type == "sell" and opp.sell_order_type == "buy":
                    source_best = source_sells_by_type.get(opp.type_id)
                    dest_best = dest_buys_by_type.get(opp.type_id)
                else:
                    source_best = source_buys_by_type.get(opp.type_id)
                    dest_best = dest_sells_by_type.get(opp.type_id)

                if source_best and dest_best:
                    item = get_type(opp.type_id)
                    if item.volume > 0:
                        max_units = int(math.floor(args.cargo_m3 / item.volume))
                    else:
                        max_units = 0
                    quantity = min(max_units, source_best.volume, dest_best.volume)
                    if quantity > 0:
                        net_sell_price = dest_best.price * (1.0 - args.broker_fee - args.sales_tax)
                        net_profit_per_unit = net_sell_price - source_best.price
                        if net_profit_per_unit > 0:
                            candidate_net = net_profit_per_unit * quantity
                            if candidate_net > base_net:
                                off_net = candidate_net
                                delta = candidate_net - base_net
                                station_id = dest_best.station_id
                                station_label = off_route_station_names.get(
                                    station_id, str(station_id)
                                )
                                system_id = off_route_station_to_system.get(station_id)
                                if system_id is not None:
                                    info = off_route_system_info.get(system_id)
                                    if info:
                                        system_label = info.name
                                        region_label = region_names.get(
                                            info.region_id, str(info.region_id)
                                        )
                                    detour_label = str(
                                        off_route_station_detour.get(station_id, "-")
                                    )
                                improved += 1
                                best_delta = max(best_delta, delta)

            row = [
                opp.item_name,
                format_isk(base_net),
                format_isk(off_net) if off_net is not None else "-",
                format_isk(delta) if delta is not None else "-",
                detour_label,
                station_label,
                system_label,
                region_label,
            ]
            if show_mode:
                row.insert(1, mode)
            off_route_rows.append(row)

        off_route_headers = [
            "Item",
            "Base Net",
            "Off-route Net",
            "Delta",
            "Detour",
            "Station",
            "System",
            "Region",
        ]
        if show_mode:
            off_route_headers.insert(1, "Mode")

        print("")
        print(
            f"Off-route upgrades (within {args.off_route_jumps} jumps of the shortest path):"
        )
        print(render_table(off_route_headers, off_route_rows))
        if off_route_note is not None:
            print(f"Summary: {off_route_note}")
        elif improved:
            print(
                f"Summary: {improved}/{len(top_opps)} items have a better off-route destination. "
                f"Best uplift: {format_isk(best_delta)}."
            )
        else:
            print(
                f"Summary: no better off-route destinations within {args.off_route_jumps} jumps."
            )
    refresh_oldest_item_types(
        client,
        args.verbose,
        limit=refresh_limit,
        max_age_seconds=refresh_max_age,
    )
    return 0


def run_best_sell(args: SimpleNamespace) -> int:
    start_time = time.monotonic()
    cache_path = os.path.expanduser(args.cache_db)
    client = EsiClient(cache_path=cache_path)
    config = load_config(args.verbose)
    refresh_limit, refresh_max_age = get_item_type_refresh_settings(config)
    args.market_cache_ttl = get_market_cache_ttl(config, args.market_cache_ttl)
    show_preflight = (not args.quiet) and (not args.no_preflight)
    show_live = (not args.quiet) and (not args.no_live)
    show_summary = (not args.quiet) and (not args.no_summary)
    show_output = not args.no_output

    def log(message: str) -> None:
        if args.verbose:
            print(message, file=sys.stderr)

    try:
        item_type_id = client.resolve_type_id(args.item)
        item_info = client.get_type_info(item_type_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        origin_system_id = client.resolve_system_id(args.from_system)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    min_security = args.min_security
    max_security = args.max_security

    if args.sec_band:
        if args.avoid_low or args.max_security is not None or args.min_security != 0.0:
            print(
                "Error: --sec-band cannot be combined with --min-security, --max-security, or --avoid-low.",
                file=sys.stderr,
            )
            return 2
        if args.sec_band == "hi":
            min_security = 0.5
            max_security = None
        elif args.sec_band == "low":
            min_security = 0.1
            max_security = 0.4
        elif args.sec_band == "null":
            min_security = -1.0
            max_security = 0.0
        elif args.sec_band == "all":
            min_security = -1.0
            max_security = None

    if args.avoid_low:
        min_security = max(min_security, 0.5)

    if min_security < -1.0 or min_security > 1.0:
        print("Error: --min-security must be between -1.0 and 1.0.", file=sys.stderr)
        return 2
    if max_security is not None and (max_security < -1.0 or max_security > 1.0):
        print("Error: --max-security must be between -1.0 and 1.0.", file=sys.stderr)
        return 2
    if max_security is not None and min_security > max_security:
        print("Error: --min-security must be <= --max-security.", file=sys.stderr)
        return 2

    if max_security is None:
        security_label = f"(min security {min_security:.1f})"
    else:
        security_label = f"(security {min_security:.1f}-{max_security:.1f})"
    log(f"Searching within {args.max_jumps} jumps of '{args.from_system}' {security_label}...")

    systems = systems_within_jumps(
        origin_system_id,
        args.max_jumps,
        min_security,
        max_security,
        client.get_system_neighbors,
        client.get_system_info,
    )

    if not systems:
        print("No systems found within the given constraints.")
        return 0

    system_map = {info.system_id: (info, jumps) for info, jumps in systems}
    station_ids: list[int] = []
    station_to_system: dict[int, int] = {}
    for info, _jumps in systems:
        station_list = client.get_system_station_ids(info.system_id)
        for station_id in station_list:
            station_ids.append(station_id)
            station_to_system[station_id] = info.system_id

    if not station_ids:
        print("No NPC stations found within the jump range.")
        return 0

    station_names = client.get_station_names(station_ids)

    region_ids = {info.region_id for info, _ in systems}
    region_names = {region_id: client.get_region_name(region_id) for region_id in region_ids}

    orders: list = []
    if show_preflight:
        print_region_cache_preflight(
            client,
            region_ids,
            region_names,
            args.market_cache_ttl,
            args.refresh_market,
        )
        print_section_break(sys.stderr, show_live or show_summary)
    use_color = should_color(sys.stderr)
    fetch_progress = FetchProgress(enabled=show_live)
    for region_id in region_ids:
        region_label = f"{region_names.get(region_id, region_id)} ({region_id})"
        if args.verbose and show_live:
            if args.market_cache_ttl > 0:
                refresh_note = " (forced refresh)" if args.refresh_market else ""
                print(
                    f"Fetching market orders for region {region_label} "
                    f"(cache TTL {args.market_cache_ttl}s){refresh_note}...",
                    file=sys.stderr,
                )
            else:
                refresh_note = " (forced refresh)" if args.refresh_market else ""
                print(
                    f"Fetching market orders for region {region_label} (no cache){refresh_note}...",
                    file=sys.stderr,
                )

        progress = None
        if show_live:
            def progress(
                page: int,
                pages: int,
                region_id: int = region_id,
                region_label: str = region_label,
            ) -> None:
                fetch_progress.update(region_id, page, pages, region_label)

        region_orders = client.fetch_region_orders(
            region_id,
            cache_ttl=args.market_cache_ttl,
            progress=progress,
            page_workers=args.page_workers,
            force_refresh=args.refresh_market,
        )
        if args.verbose and show_live:
            fetch_progress.ensure_newline()
            print(
                f"Fetched {len(region_orders):,} orders from region {region_label}.",
                file=sys.stderr,
            )
        meta = client.last_region_fetch.get(region_id)
        if meta and show_live:
            fetch_progress.ensure_newline()
            print(
                format_region_fetch_summary(region_label, meta, use_color),
                file=sys.stderr,
            )
        orders.extend(region_orders)

    fetch_progress.ensure_newline()
    print_section_break(sys.stderr, show_live and show_summary)
    log("Evaluating stations...")
    best_by_station = best_prices_for_item(orders, station_ids, item_type_id)

    results: list[list[str]] = []
    ranked: list[tuple[float, int, int, list[str]]] = []
    for station_id, prices in best_by_station.items():
        system_id = station_to_system.get(station_id)
        if system_id is None:
            continue
        info, jumps = system_map[system_id]
        station_name = station_names.get(station_id, str(station_id))
        region_name = region_names.get(info.region_id, str(info.region_id))

        def add_row(mode: str, raw_price: float | None, volume: int) -> None:
            if raw_price is None:
                return
            net_price = raw_price * (1.0 - args.broker_fee - args.sales_tax)
            isk_per_m3 = net_price / item_info.volume if item_info.volume > 0 else 0.0
            row = [
                station_name,
                info.name,
                region_name,
                str(jumps),
                f"{info.security:.1f}",
                mode,
                format_isk(net_price),
                format_isk(isk_per_m3),
                str(volume),
            ]
            ranked.append((net_price, -jumps, volume, row))

        if args.instant_only:
            add_row("buy", prices["buy_price"], int(prices["buy_volume"]))
        else:
            add_row("buy", prices["buy_price"], int(prices["buy_volume"]))
            add_row("sell", prices["sell_price"], int(prices["sell_volume"]))

    if not ranked:
        if show_output:
            print("No sell opportunities found for that item.")
        refresh_oldest_item_types(
            client,
            args.verbose,
            limit=refresh_limit,
            max_age_seconds=refresh_max_age,
        )
        return 0

    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    for _score, _jumps, _volume, row in ranked[: args.limit]:
        results.append(row)

    headers = [
        "Station",
        "System",
        "Region",
        "Jumps",
        "Sec",
        "Mode",
        "Net Price",
        "ISK/m³",
        "Volume",
    ]
    if args.instant_only:
        headers.remove("Mode")
        for row in results:
            row.pop(5)

    print_total_elapsed(start_time, show_summary)
    if show_output:
        print_section_break(sys.stdout, show_preflight or show_live or show_summary)
        print(f"Item: {item_info.name} (type {item_type_id})")
        print(render_table(headers, results))
    refresh_oldest_item_types(
        client,
        args.verbose,
        limit=refresh_limit,
        max_age_seconds=refresh_max_age,
    )
    return 0


def run_cache_types(args: SimpleNamespace) -> int:
    start_time = time.monotonic()
    cache_path = os.path.expanduser(args.cache_db)
    client = EsiClient(cache_path=cache_path)
    show_preflight = not args.no_preflight
    show_live = not args.no_live
    show_summary = not args.no_summary
    show_output = not args.no_output
    use_color = should_color(sys.stderr)

    with sqlite3.connect(cache_path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        if args.region:
            placeholders = ",".join("?" for _ in args.region)
            region_filter = f"WHERE mo.region_id IN ({placeholders})"
            region_params = list(args.region)
        else:
            region_filter = ""
            region_params = []

        if args.limit and args.limit > 0:
            rows = conn.execute(
                f"""
                SELECT mo.type_id, it.updated_at
                FROM market_orders mo
                LEFT JOIN item_types it ON it.type_id = mo.type_id
                {region_filter}
                GROUP BY mo.type_id
                ORDER BY (it.updated_at IS NOT NULL) ASC, it.updated_at ASC
                LIMIT ?
                """,
                region_params + [args.limit],
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT DISTINCT mo.type_id
                FROM market_orders mo
                {region_filter}
                """,
                region_params,
            ).fetchall()

    type_ids = [int(row["type_id"]) for row in rows]

    if not type_ids:
        if show_output:
            print("No type_ids found in market_orders.")
        return 0

    effective_refresh = args.refresh or (args.limit and args.limit > 0)

    if show_preflight:
        print(style("Type cache preflight:", use_color, Ansi.BOLD), file=sys.stderr)
        if args.region:
            region_label = ", ".join(str(region_id) for region_id in args.region)
            print(f"Regions: {region_label}", file=sys.stderr)
        print(f"Type ids: {len(type_ids):,} (from market_orders)", file=sys.stderr)
        if args.limit and args.limit > 0:
            print(f"Limit: {args.limit} (oldest first)", file=sys.stderr)
        if effective_refresh:
            if args.refresh:
                refresh_note = style("enabled", use_color, Ansi.MAGENTA)
            else:
                refresh_note = style("forced (limit)", use_color, Ansi.MAGENTA)
        else:
            refresh_note = style("disabled", use_color, Ansi.GREEN)
        print(f"Refresh: {refresh_note}", file=sys.stderr)
        print(f"Workers: {args.type_workers}", file=sys.stderr)
        print_section_break(sys.stderr, show_live or show_summary)

    progress = None
    type_progress = None
    if show_live:
        type_progress = CountProgress("Type info", enabled=True)

        def progress(done: int, total: int) -> None:
            type_progress.update(done, total)

    fetched, skipped, errors = client.cache_type_info_bulk(
        type_ids,
        refresh=effective_refresh,
        workers=args.type_workers,
        progress=progress,
    )
    if type_progress:
        type_progress.ensure_newline()
    print_section_break(sys.stderr, show_live and show_summary)

    print_total_elapsed(start_time, show_summary)
    if show_output:
        print_section_break(sys.stdout, show_preflight or show_live or show_summary)
        print(
            f"Cached {fetched:,} item types. Skipped {skipped:,} already cached. "
            f"Errors: {errors}."
        )
    return 0


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        print("Interrupted. Exiting.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
