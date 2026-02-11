# waybill

A Python 3.12 CLI tool to find profitable station-to-station hauling opportunities in EVE Online using public ESI endpoints (no auth).

## What It Does

- Pulls **regional** market orders from ESI.
- Filters orders for your source/destination stations or systems.
- Computes **net profit** (after broker fee + sales tax).
- Ranks by **ISK/m³**, then liquidity.
- Caches ESI responses and normalized orders in SQLite for speed.

## Requirements

- Python 3.12+
- `uv`

## Setup

```bash
uv venv
uv pip install -e .
```

## Quick Start

```bash
uv run waybill route \
  --from-station "Hek VIII - Moon 12 - Boundless Creation Factory" \
  --to-system "Aldik" \
  --cargo-m3 60000 \
  --min-profit 500000 \
  --instant-only \
  --verbose
```

## Commands

### `route`
Find profitable hauling opportunities between stations/systems.


```bash
uv run waybill route \
  --from-system "Hek" \
  --to-system "Aldik" \
  --cargo-m3 60000 \
  --min-profit 100000 \
  --instant-only
```

Add off-route upgrade hints (detours within N jumps of the shortest path):

```bash
uv run waybill route \
  --from-system "Hek" \
  --to-system "Aldik" \
  --cargo-m3 60000 \
  --min-profit 100000 \
  --off-route-jumps 2 \
  --instant-only
```

### `best-sell`
Find the best stations (within N jumps) to sell a specific item.

```bash
uv run waybill best-sell \
  --item "Tritanium" \
  --from-system "Aldik" \
  --max-jumps 5 \
  --sec-band hi \
  --instant-only
```

Security filters:

- Security status ranges from about `-1.0` (deep null) to `1.0` (high-sec).
- Use `--min-security` / `--max-security` for precise ranges.
- Or use `--sec-band hi|low|null|all` as a shorthand.
- `--avoid-low` is an alias for `--min-security 0.5`.

### `sync-items`
Populate `item_types` for type IDs already present in cached market orders. When `--limit` is set, the oldest cached
types are refreshed first.

```bash
uv run waybill sync-items --verbose
```

Force refresh cached type info:

```bash
uv run waybill sync-items --refresh
```

## How It Works

### Market Data Scope

- ESI exposes **regional** market order endpoints.
- The tool fetches **entire region(s)**, then filters to your stations/systems.
- If your jump radius crosses into another region, those regions are fetched too.

### Order Logic

- `--instant-only`:
  - Buy at source **sell orders**.
  - Sell at destination **buy orders**.
- Without `--instant-only`:
  - Also considers **buy orders at source** and **sell orders at destination** (non-instant).

### Ranking

- Primary: ISK per m³ (net)
- Secondary: liquidity (order volume at best price)

## CLI Options (Highlights)

### `route`

- `--from-station` or `--from-system`
- `--to-station` or `--to-system`
- `--cargo-m3`
- `--min-profit`
- `--instant-only`
- `--broker-fee` (default 0.03)
- `--sales-tax` (default 0.08)
- `--limit` (default 20)
- `--off-route-jumps` (detour search within N jumps of the shortest path)
- `--verbose`
- `--quiet` (hide cache preflight/status lines)
- `--market-cache-ttl` (default 14400 seconds)
- `--refresh-market` (force refresh, ignore cache)
- `--page-workers` (default 1)

### `best-sell`

- `--item`
- `--from-system`
- `--max-jumps`
- `--min-security`
- `--max-security`
- `--sec-band` (hi|low|null|all)
- `--avoid-low` (alias for `--min-security 0.5`)
- `--instant-only`
- `--limit`
- `--broker-fee` (default 0.03)
- `--sales-tax` (default 0.08)
- `--verbose`
- `--quiet` (hide cache preflight/status lines)
- `--market-cache-ttl` (default 14400 seconds)
- `--refresh-market` (force refresh, ignore cache)
- `--page-workers` (default 1)

### `sync-items`

- `--region` (repeatable)
- `--refresh`
- `--limit`
- `--type-workers` (default 2)
- `--verbose`

## Caching

- Default cache DB: `~/.cache/waybill/waybill.sqlite`
- HTTP responses cached in `http_cache`.
- Normalized market orders in `market_orders`.
- Item metadata in `item_types`.
- Region names in `region_names`.
- System info + stargate graph in `system_info` and `system_neighbors`.

Cache freshness:

- `--market-cache-ttl` sets a **minimum** freshness window (default 4 hours).
- `--refresh-market` forces a fresh pull and updates the cache immediately.
- Fetch progress and counts are shown with `--verbose`, but cache status is always printed.
- The tool prints the region name, last fetch time, and next refresh (hours/minutes) after each region fetch.

### Config file

Waybill reads optional settings from `~/.config/waybill/config.toml`. If the file is missing, defaults are used and nothing is created.

Item type refresh (runs after `route` and `best-sell` output):

```toml
[item_type_refresh]
oldest_limit = 10        # refresh this many oldest type records (0 disables)
max_age_seconds = 0      # only refresh types older than this age (0 = no age filter)
```

Market cache:

```toml
[market_cache]
ttl_seconds = 14400      # minimum cache freshness window (0 disables cache)
```

## SQLite Tables (Quick Reference)

- `market_orders`: normalized orders from ESI
- `item_types`: name + volume for item types
- `region_names`: region ID to name
- `system_info`: system name, security, region
- `system_neighbors`: stargate graph
- `station_names`: station ID to name
- `http_cache`: raw ESI responses

## Output Columns

### `route`

- `Item`
- `Buy`
- `Sell`
- `Net Profit`
- `ISK/m³`
- `Qty`
- `Total Volume`
- `Mode` (only when non-instant)

With `--off-route-jumps`, an additional table is printed:

- `Item`
- `Base Net`
- `Off-route Net`
- `Delta`
- `Detour` (jumps off the shortest path)
- `Station`
- `System`
- `Region`
- `Mode` (only when non-instant)

### `best-sell`

- `Station`
- `System`
- `Region`
- `Jumps`
- `Sec`
- `Net Price`
- `ISK/m³`
- `Volume`
- `Mode` (only when non-instant)

## Performance Tips

- Use `--market-cache-ttl 14400` (or higher) to avoid re-fetching.
- Use `--page-workers 2` or `4` to speed up large regions.
- Keep `--verbose` on for page-level progress.
- `sync-items` can be throttled with `--limit` and `--type-workers`.

## Limitations

- No authentication.
- Player structures are not supported with public ESI.
- Market data is regional and may be large to fetch.
- Item type cache is lazy unless you run `sync-items`.

## Examples

Find deals between two stations:

```bash
uv run waybill route \
  --from-station "Hek VIII - Moon 12 - Boundless Creation Factory" \
  --to-station "Aldik VIII - Moon 2 - Sebiestor Tribe Bureau" \
  --cargo-m3 60000 \
  --min-profit 500000 \
  --instant-only
```

Best sell within 3 jumps in highsec:

```bash
uv run waybill best-sell \
  --item "Tritanium" \
  --from-system "Aldik" \
  --max-jumps 3 \
  --sec-band hi \
  --instant-only
```

Avoid high-sec (low/null only) within 5 jumps:

```bash
uv run waybill best-sell \
  --item "Plagioclase II-Grade" \
  --from-system "Aldik" \
  --max-jumps 5 \
  --max-security 0.4
```
