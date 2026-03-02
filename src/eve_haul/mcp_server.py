from __future__ import annotations

import dataclasses
import os
import urllib.error
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .esi import EsiClient
from .market import (
    best_prices_for_item,
    build_order_books,
    compute_opportunities,
    shortest_path,
    systems_within_jumps,
)

SEC_BAND_MAP: dict[str, tuple[float, float | None]] = {
    "hi": (0.5, None),
    "low": (0.1, 0.4),
    "null": (-1.0, 0.0),
    "all": (-1.0, None),
}


def _resolve_security_band(
    sec_band: str | None,
    min_security: float,
    max_security: float | None,
) -> tuple[float, float | None]:
    if sec_band is not None:
        band = SEC_BAND_MAP.get(sec_band)
        if band is None:
            raise ToolError(f"Invalid sec_band '{sec_band}'. Choose from: hi, low, null, all.")
        return band
    return min_security, max_security


def create_server(cache_db: str) -> FastMCP:
    client = EsiClient(os.path.expanduser(cache_db))
    mcp = FastMCP(
        "Waybill — EVE Online Market Intelligence",
        instructions=(
            "Query live EVE Online market data via the public ESI API. "
            "Find profitable hauling routes, best sell stations, item prices, "
            "and system navigation data. All market data is cached locally."
        ),
    )

    @mcp.tool
    def find_hauling_opportunities(
        cargo_m3: float = 60000,
        min_profit: float = 0,
        broker_fee: float = 0.03,
        sales_tax: float = 0.08,
        instant_only: bool = False,
        backhaul_mode: bool = False,
        min_cargo_fill_pct: float = 0,
        limit: int = 20,
        market_cache_ttl: int = 14400,
        from_station: str | None = None,
        from_system: str | None = None,
        to_station: str | None = None,
        to_system: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find profitable cargo hauling opportunities between two locations.

        Provide exactly one of from_station or from_system, and exactly one of
        to_station or to_system. Returns opportunities ranked by ISK/m³.
        """
        if from_station is None and from_system is None:
            raise ToolError("Provide either from_station or from_system.")
        if from_station is not None and from_system is not None:
            raise ToolError("Provide only one of from_station or from_system, not both.")
        if to_station is None and to_system is None:
            raise ToolError("Provide either to_station or to_system.")
        if to_station is not None and to_system is not None:
            raise ToolError("Provide only one of to_station or to_system, not both.")

        try:
            if from_station is not None:
                source_refs = [client.resolve_station(from_station)]
            else:
                source_refs = client.resolve_system_stations(from_system)  # type: ignore[arg-type]
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        try:
            if to_station is not None:
                dest_refs = [client.resolve_station(to_station)]
            else:
                dest_refs = client.resolve_system_stations(to_system)  # type: ignore[arg-type]
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        source_station_ids = [ref.station_id for ref in source_refs]
        dest_station_ids = [ref.station_id for ref in dest_refs]
        region_ids = {ref.region_id for ref in source_refs} | {ref.region_id for ref in dest_refs}

        all_orders = []
        for region_id in region_ids:
            all_orders.extend(client.fetch_region_orders(region_id, cache_ttl=market_cache_ttl))

        source_sells, source_buys, dest_sells, dest_buys, _ = build_order_books(
            all_orders, source_station_ids, dest_station_ids
        )

        opps = compute_opportunities(
            source_sells,
            source_buys,
            dest_sells,
            dest_buys,
            client.get_type_info,
            cargo_m3,
            broker_fee,
            sales_tax,
            min_profit,
            min_cargo_fill_pct / 100.0,
            backhaul_mode,
            instant_only,
        )

        return [dataclasses.asdict(opp) for opp in opps[:limit]]

    @mcp.tool
    def find_best_sell_stations(
        item: str,
        from_system: str,
        max_jumps: int = 5,
        instant_only: bool = False,
        broker_fee: float = 0.03,
        sales_tax: float = 0.08,
        limit: int = 10,
        market_cache_ttl: int = 14400,
        sec_band: str | None = None,
        min_security: float = 0.0,
        max_security: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find the best stations to sell an item within a jump radius.

        sec_band accepts: 'hi', 'low', 'null', 'all'. When provided, overrides
        min_security and max_security.
        """
        try:
            type_id = client.resolve_type_id(item)
            item_info = client.get_type_info(type_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        try:
            origin_system_id = client.resolve_system_id(from_system)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        try:
            min_sec, max_sec = _resolve_security_band(sec_band, min_security, max_security)
        except ToolError:
            raise

        systems = systems_within_jumps(
            origin_system_id,
            max_jumps,
            min_sec,
            max_sec,
            client.get_system_neighbors,
            client.get_system_info,
        )

        if not systems:
            return []

        system_map = {info.system_id: (info, jumps) for info, jumps in systems}
        station_ids: list[int] = []
        station_to_system: dict[int, int] = {}
        for info, _jumps in systems:
            for station_id in client.get_system_station_ids(info.system_id):
                station_ids.append(station_id)
                station_to_system[station_id] = info.system_id

        if not station_ids:
            return []

        region_ids = {info.region_id for info, _ in systems}
        region_names = {rid: client.get_region_name(rid) for rid in region_ids}
        station_names = client.get_station_names(station_ids)

        all_orders = []
        for region_id in region_ids:
            all_orders.extend(client.fetch_region_orders(region_id, cache_ttl=market_cache_ttl))

        best_by_station = best_prices_for_item(all_orders, station_ids, type_id)

        rows: list[tuple[float, dict[str, Any]]] = []
        for station_id, prices in best_by_station.items():
            system_id = station_to_system.get(station_id)
            if system_id is None:
                continue
            info, jumps = system_map[system_id]
            station_name = station_names.get(station_id, str(station_id))
            region_name = region_names.get(info.region_id, str(info.region_id))

            def _add_row(
                mode: str,
                raw_price: float | None,
                volume: int,
                station_id: int = station_id,
                station_name: str = station_name,
                info: Any = info,
                region_name: str = region_name,
                jumps: int = jumps,
            ) -> None:
                if raw_price is None:
                    return
                net_price = raw_price * (1.0 - broker_fee - sales_tax)
                rows.append((
                    net_price,
                    {
                        "station_id": station_id,
                        "station_name": station_name,
                        "system_id": info.system_id,
                        "system_name": info.name,
                        "region_id": info.region_id,
                        "region_name": region_name,
                        "jumps": jumps,
                        "security": info.security,
                        "mode": mode,
                        "raw_price": raw_price,
                        "net_price": net_price,
                        "volume": volume,
                    },
                ))

            _add_row("buy", prices.get("buy_price"), int(prices.get("buy_volume") or 0))
            if not instant_only:
                _add_row("sell", prices.get("sell_price"), int(prices.get("sell_volume") or 0))

        rows.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in rows[:limit]]

    @mcp.tool
    def get_item_info(item_name: str) -> dict[str, Any]:
        """Look up EVE item metadata: type_id, name, and volume in m³."""
        try:
            type_id = client.resolve_type_id(item_name)
            info = client.get_type_info(type_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return {"type_id": info.type_id, "name": info.name, "volume": info.volume}

    @mcp.tool
    def resolve_station(station_name: str) -> dict[str, Any]:
        """Resolve a station name to its station_id, system_id, and region_id."""
        try:
            ref = client.resolve_station(station_name)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return dataclasses.asdict(ref)

    @mcp.tool
    def get_system_info(system_name: str) -> dict[str, Any]:
        """Look up system metadata: system_id, name, security status, and region_id."""
        try:
            system_id = client.resolve_system_id(system_name)
            info = client.get_system_info(system_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return dataclasses.asdict(info)

    @mcp.tool
    def find_systems_within_jumps(
        from_system: str,
        max_jumps: int = 5,
        sec_band: str | None = None,
        min_security: float = -1.0,
        max_security: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find all systems reachable within N jumps, sorted by distance ascending.

        sec_band accepts: 'hi', 'low', 'null', 'all'. When provided, overrides
        min_security and max_security.
        """
        try:
            origin_id = client.resolve_system_id(from_system)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        try:
            min_sec, max_sec = _resolve_security_band(sec_band, min_security, max_security)
        except ToolError:
            raise

        results = systems_within_jumps(
            origin_id,
            max_jumps,
            min_sec,
            max_sec,
            client.get_system_neighbors,
            client.get_system_info,
        )

        output = []
        for info, jumps in results:
            row = dataclasses.asdict(info)
            row["jumps"] = jumps
            output.append(row)
        output.sort(key=lambda x: x["jumps"])
        return output

    @mcp.tool
    def find_shortest_path(from_system: str, to_system: str) -> dict[str, Any]:
        """Find the shortest jump path between two systems.

        Returns path as a list of system_ids, and the jump count. If no route
        exists, returns an empty path and jumps=0.
        """
        try:
            from_id = client.resolve_system_id(from_system)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        try:
            to_id = client.resolve_system_id(to_system)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        path = shortest_path(from_id, to_id, client.get_system_neighbors)
        return {
            "from_system_id": from_id,
            "to_system_id": to_id,
            "jumps": max(0, len(path) - 1),
            "path": path,
        }

    @mcp.tool
    def get_item_orders(
        item_name: str,
        region_name: str,
        market_cache_ttl: int = 14400,
    ) -> dict[str, Any]:
        """Get current market orders for an item in a region.

        Returns best buy and sell prices per station that has orders for the item.
        """
        try:
            type_id = client.resolve_type_id(item_name)
        except (ValueError, urllib.error.HTTPError) as exc:
            raise ToolError(str(exc)) from exc

        try:
            region_id = client.resolve_region_id(region_name)
        except (ValueError, urllib.error.HTTPError) as exc:
            raise ToolError(str(exc)) from exc

        try:
            orders = client.fetch_item_orders(region_id, type_id, cache_ttl=market_cache_ttl)
        except urllib.error.HTTPError as exc:
            raise ToolError(f"Failed to fetch market orders: {exc}") from exc
        # Player structures have IDs > 1e9 and are not resolvable without auth.
        # Only include NPC station IDs (< 1_000_000_000).
        station_ids = {
            o.location_id for o in orders
            if o.location_id < 1_000_000_000
        }

        prices_by_station = best_prices_for_item(orders, station_ids, type_id)
        station_names = client.get_station_names(station_ids)
        resolved_region_name = client.get_region_name(region_id)

        stations: dict[str, dict[str, Any]] = {}
        for station_id, prices in prices_by_station.items():
            if prices.get("buy_price") is None and prices.get("sell_price") is None:
                continue
            stations[str(station_id)] = {
                "station_name": station_names.get(station_id, str(station_id)),
                "buy_price": prices.get("buy_price"),
                "buy_volume": prices.get("buy_volume"),
                "sell_price": prices.get("sell_price"),
                "sell_volume": prices.get("sell_volume"),
            }

        return {
            "item_name": item_name,
            "type_id": type_id,
            "region_id": region_id,
            "region_name": resolved_region_name,
            "stations": stations,
        }

    return mcp
